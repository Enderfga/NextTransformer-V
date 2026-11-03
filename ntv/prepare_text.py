"""Text resources from a partial download of the pinned ClimbMix revision.

NextTransformer's tokenizer is trained by nanochat's `tok_train` on the first 2B characters of the training
shards in sorted order, so the first shards of the pinned revision reproduce it exactly. The result must match
NextTransformer's pinned SHA-256; anything else is refused.

  python -m ntv.prepare_text --data-dir <climbmix dir> --resources <resources dir>
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from nexttransformer.protocol.spec import REPO, TOKENIZER_SHA256
from nexttransformer.runtime.data import tokenizer_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    args = parser.parse_args()
    args.resources.mkdir(parents=True, exist_ok=True)
    link = args.resources / "base_data_climbmix"
    if not link.exists():
        link.symlink_to(args.data_dir.resolve(), target_is_directory=True)
    if not (args.resources / "tokenizer" / "tokenizer.pkl").exists():
        env = dict(os.environ, NANOCHAT_BASE_DIR=str(args.resources))
        subprocess.run([sys.executable, "-m", "nanochat.tok_train"], cwd=REPO, env=env, check=True)
    digest = tokenizer_fingerprint(args.resources)
    if digest != TOKENIZER_SHA256:
        raise SystemExit(f"tokenizer differs from NextTransformer's pin: {digest}")
    print(f"tokenizer matches NextTransformer's pin {digest}")


if __name__ == "__main__":
    main()
