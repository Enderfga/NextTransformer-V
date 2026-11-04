"""The frozen vision encoder: SigLIP2 so400m/16 at 384 px, patch features merged 2 x 2.

Training uses 384 px (12 x 12 merged tokens). Larger square images (a side divisible by 32) give a finer token grid
for zero-shot resolution transfer; the encoder's position embedding is then interpolated, as SigLIP supports.
"""
import torch

GRID = 24          # 384 / 16 patches per side
PATCH = 16
MERGE = 2
WIDTH = 1152


class FrozenSiglip(torch.nn.Module):
    def __init__(self, path, device, dtype=torch.bfloat16):
        super().__init__()
        from transformers import SiglipVisionModel
        self.model = SiglipVisionModel.from_pretrained(path, dtype=dtype).to(device).eval().requires_grad_(False)
        self.dtype = dtype

    @torch.no_grad()
    def forward(self, images):
        """uint8 (N, 3, S, S) -> (N * (S/32)^2, 4608), images in order, tokens row-major over the merged grid.
        S = 384 is the training resolution (144 tokens per image)."""
        if images.shape[0] == 0:
            return torch.zeros((0, MERGE * MERGE * WIDTH), device=images.device, dtype=self.dtype)
        side = images.shape[-1] // PATCH
        if images.shape[-1] % (PATCH * MERGE) or images.shape[-2] != images.shape[-1]:
            raise ValueError(f"images must be square with a side divisible by {PATCH * MERGE}")
        pixels = images.to(self.dtype).div(127.5).sub(1.)  # SigLIP normalization: mean 0.5, std 0.5
        hidden = self.model(pixel_values=pixels, interpolate_pos_encoding=side != GRID).last_hidden_state
        n = hidden.shape[0]
        if hidden.shape[1:] != (side * side, WIDTH):
            raise ValueError(f"unexpected encoder output {tuple(hidden.shape)}")
        g = side // MERGE
        merged = hidden.view(n, g, MERGE, g, MERGE, WIDTH).permute(0, 1, 3, 2, 4, 5)
        return merged.reshape(n * g * g, MERGE * MERGE * WIDTH)
