"""Train one backbone of the multimodal track on packed rows (torchrun, one process per GPU).

The language model, optimizer groups, LR/momentum/weight-decay schedule and checkpoint format are
NextTransformer's; this trainer adds the frozen encoder, the visual inputs and the track's data and budget.

  torchrun --standalone --nproc_per_node 8 -m ntv.train --method dense_full --train-rows <dir> \
      --val-rows <dir> --tars <dir> --encoder <dir> --resources <dir> --output <dir> --updates 5000
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from nexttransformer.models.config import ModelConfig
from nexttransformer.native.optimizer import set_schedule
from nexttransformer.runtime.checkpoint import load_checkpoint, save_checkpoint
from nexttransformer.training.optimizer import build_optimizer
from ntv.data import IMAGE_TOKENS, PackedRows, RankRows, collate
from ntv.encoder import FrozenSiglip
from ntv.model import VISUAL_DIM, build_mm_model, track_config, visual_inputs
from ntv.spatial import ORDERS, POSITIONS, RAW_DIM, RawPatches


def visual_encoder(kind, path, device):
    """(encoder, feature width) for --visual."""
    return (FrozenSiglip(str(path), device), VISUAL_DIM) if kind == "siglip" else (RawPatches(), RAW_DIM)


@dataclass(frozen=True)
class Schedule:
    """NextTransformer's schedule constants over the track's number of updates."""
    steps: int
    warmup_steps: int = 40
    warmdown_ratio: float = .65
    final_lr_fraction: float = .05


@dataclass(frozen=True)
class Settings:
    method: str
    updates: int
    rows_per_update: int
    micro_batch: int
    sequence_len: int
    seed: int
    train_rows: str
    eval_rows: int
    visual: str = "siglip"
    order: str = "raster"
    position: str = "rope1d"
    conv2d: str = "none"


class Cursor:
    """Checkpointed data position: the next update (rows are a pure function of the update)."""

    def __init__(self, update=0):
        self.update = update

    def state_dict(self):
        return {"update": self.update}

    def load_state_dict(self, state):
        self.update = state["update"]


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True)
    p.add_argument("--model", type=Path, help="calibrated model configuration (JSON fields of ModelConfig)")
    p.add_argument("--shape", help="smoke-only overrides, e.g. width=768,heads=6,layers=12,mlp_width=3072")
    p.add_argument("--train-rows", type=Path, required=True)
    p.add_argument("--val-rows", type=Path, required=True)
    p.add_argument("--tars", type=Path, required=True)
    p.add_argument("--val-tars", type=Path)
    p.add_argument("--encoder", type=Path, required=True)
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--updates", type=int, required=True, help="updates in the schedule (the budget)")
    p.add_argument("--stop-after", type=int, help="stop after this many updates (smoke); default: --updates")
    p.add_argument("--rows-per-update", type=int, default=128)
    p.add_argument("--micro-batch", type=int, default=2)
    p.add_argument("--sequence-len", type=int, default=8192)
    p.add_argument("--sparse-backend", default="reference")
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-rows", type=int, default=128)
    p.add_argument("--checkpoint-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--flops-per-token", type=float, help="audited training FLOPs per token, for throughput logs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--visual", choices=("siglip", "raw32"), default="siglip",
                   help="frozen SigLIP2 features or raw 32 px patches")
    p.add_argument("--order", choices=ORDERS, default="raster", help="patch order of each image in the sequence")
    p.add_argument("--position", choices=POSITIONS, default="rope1d", help="position mode of the attention layers")
    p.add_argument("--conv2d", choices=("none", "k", "qkv"), default="none",
                   help="2D short convolution at image positions in linear-attention layers")
    return p.parse_args(argv)


def model_config(args):
    if args.model is not None:
        fields = json.loads(args.model.read_text())
        fields.pop("layer_mixers", None)
        config = ModelConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in fields.items()})
    else:
        config = track_config(args.method, sequence_len=args.sequence_len)
    if args.shape:
        config = replace(config, **{k: int(v) for k, v in (item.split("=") for item in args.shape.split(","))})
    if config.method != args.method or config.sequence_len != args.sequence_len:
        raise ValueError("model configuration does not match --method / --sequence-len")
    return config


def loader(dataset, sampler, micro_batch, workers):
    return DataLoader(dataset, batch_size=micro_batch, sampler=sampler, collate_fn=collate, num_workers=workers,
                      pin_memory=True, prefetch_factor=4 if workers else None, persistent_workers=workers > 0)


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, encoder, batches, token_bytes, device):
    """Caption and text BPB; caption BPB again with every image's features moved to the next image."""
    was_training = model.training
    model.eval()
    sums = torch.zeros(5, dtype=torch.float64, device=device)
    for batch in batches:
        batch = to_device(batch, device)
        x, y, mask = batch["inputs"], batch["targets"], batch["image_mask"]
        features = encoder(batch["images"])
        valid = y >= 0
        weight = (token_bytes[y.clamp_min(0)] * valid).double()
        caption, text = batch["caption"] & valid, batch["text"] & valid
        with visual_inputs(model, mask, features):
            nats = model(x, y, loss_reduction="none").view_as(y).double()
        n = batch["images"].shape[0]
        moved = features.view(n, IMAGE_TOKENS, -1).roll(1, 0).reshape_as(features)
        with visual_inputs(model, mask, moved):
            shuffled = model(x, y, loss_reduction="none").view_as(y).double()
        sums += torch.stack(((nats * caption).sum(), (weight * caption).sum(), (nats * text).sum(),
                             (weight * text).sum(), (shuffled * caption).sum()))
    if dist.is_initialized():
        dist.all_reduce(sums)
    model.train(was_training)
    bits = lambda nats, nbytes: (nats / (math.log(2) * nbytes)).item()
    return dict(caption_bpb=bits(sums[0], sums[1]), text_bpb=bits(sums[2], sums[3]),
                caption_bpb_shuffled=bits(sums[4], sums[1]), caption_bytes=int(sums[1]), text_bytes=int(sums[3]))


def main(argv=None):
    args = parse(argv)
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        dist.init_process_group("nccl")
    device = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    config = model_config(args)
    train_set = PackedRows(args.train_rows, args.tars)
    val_set = PackedRows(args.val_rows, args.val_tars or args.tars)
    per_rank = args.rows_per_update // world
    if args.rows_per_update % world or per_rank % args.micro_batch:
        raise ValueError("rows per update must split into whole micro-batches on every rank")
    accumulation = per_rank // args.micro_batch
    stop = args.stop_after or args.updates
    if len(train_set) < stop * args.rows_per_update:
        raise ValueError(f"{len(train_set)} packed rows; {stop} updates need {stop * args.rows_per_update}")
    settings = Settings(config.method, args.updates, args.rows_per_update, args.micro_batch, config.sequence_len,
                        args.seed, str(args.train_rows), args.eval_rows, args.visual, args.order, args.position,
                        args.conv2d)
    spatial = dict(order=args.order, position=args.position, conv2d=args.conv2d)
    encoder, visual_dim = visual_encoder(args.visual, args.encoder, device)
    model = build_mm_model(config, device, seed=args.seed, visual_dim=visual_dim, spatial=spatial)
    if config.has_sparse:
        model.set_sparse_backend(args.sparse_backend)
    optimizer = build_optimizer(model)  # NextTransformer's groups; the track keeps its LRs and batch tokens
    cursor = Cursor()
    args.output.mkdir(parents=True, exist_ok=True)
    latest = args.output / "latest.json"
    if latest.exists():
        checkpoint = args.output / json.loads(latest.read_text())["checkpoint"]
        load_checkpoint(checkpoint, model, optimizer, cursor, rank, world)
    start = cursor.update
    model.set_training_stage(start)
    wrapped = DDP(model, device_ids=[local], broadcast_buffers=False) if world > 1 else model
    token_bytes = torch.load(args.resources / "tokenizer" / "token_bytes.pt", map_location=device)
    schedule = Schedule(args.updates)

    batches = iter(loader(train_set, RankRows(args.rows_per_update, rank, world, start, stop), args.micro_batch,
                          args.workers))
    val_rows = [r for r in range(min(args.eval_rows, len(val_set))) if r % world == rank]
    val_loader = loader(val_set, val_rows, args.micro_batch, 2)
    log = (args.output / "metrics.jsonl").open("a") if rank == 0 else None

    def record(item):
        if log is not None:
            log.write(json.dumps(item) + "\n")
            log.flush()
            print(json.dumps(item), flush=True)

    record(dict(event="start", update=start, world=world, accumulation=accumulation, settings=asdict(settings),
                model=config.serialized(), parameters=sum(p.numel() for p in model.parameters()),
                git=os.environ.get("NTV_SOURCE", "")))
    if start == 0:
        record(dict(event="eval", update=0, **evaluate(model, encoder, val_loader, token_bytes, device)))
    window = []
    for update in range(start, stop):
        model.set_step(update)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
        began = time.perf_counter()
        data_seconds, bad = 0., 0
        optimizer.zero_grad(set_to_none=True)
        lr = set_schedule(optimizer, update, schedule)
        totals = torch.zeros(2, dtype=torch.float64, device=device)
        for micro in range(accumulation):
            t = time.perf_counter()
            batch = to_device(next(batches), device)
            data_seconds += time.perf_counter() - t
            bad += batch["bad_images"]
            features = encoder(batch["images"])
            model.set_row_offset(int(batch["rows"][0]))
            sync = wrapped.no_sync() if world > 1 and micro < accumulation - 1 else nullcontext()
            with sync, visual_inputs(model, batch["image_mask"], features):
                loss = wrapped(batch["inputs"], batch["targets"])
                (loss / accumulation).backward()
            lm = model.last_lm_loss if model.last_lm_loss is not None else loss
            aux = model.last_aux_loss if model.last_aux_loss is not None else torch.zeros((), device=device)
            totals += torch.stack((lm.detach().double(), aux.detach().double())) / accumulation
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
        optimizer.step()
        if world > 1:
            dist.all_reduce(totals)
            totals /= world
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - began
        window.append(seconds)
        done = update + 1
        if done % args.log_every == 0 or done == stop or update == start:
            tokens = args.rows_per_update * config.sequence_len
            mean = sum(window) / len(window)
            item = dict(event="train", update=done, loss=totals[0].item(), aux=totals[1].item(), lr_multiplier=lr,
                        grad_norm=grad_norm.item(), seconds=mean, data_seconds=data_seconds,
                        tokens_per_second=tokens / mean, bad_images=bad,
                        peak_gib=torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.)
            if args.flops_per_token:
                item["model_tflops_per_gpu"] = args.flops_per_token * tokens / mean / world / 1e12
            record(item)
            window = []
        if done % args.eval_every == 0 or done == stop:
            record(dict(event="eval", update=done, **evaluate(model, encoder, val_loader, token_bytes, device)))
        if done % args.checkpoint_every == 0 or done == stop:
            cursor.update = done
            path = save_checkpoint(args.output, model, optimizer, cursor, config, settings, dict(step=done), rank, world)
            record(dict(event="checkpoint", update=done, path=path.name))
    record(dict(event="end", update=stop))
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
