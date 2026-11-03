#!/usr/bin/env bash
# Build the track's Python environment on a node. Idempotent.
# Usage: NTV_ROOT=<node-local root> bash scripts/bootstrap.sh   (run from the code directory)
set -euo pipefail
: "${NTV_ROOT:?set NTV_ROOT to the node-local working root}"
export UV_CACHE_DIR="$NTV_ROOT/uv-cache" UV_PYTHON_INSTALL_DIR="$NTV_ROOT/python" UV_LINK_MODE=copy
mkdir -p "$NTV_ROOT"/{bin,logs}
if [ ! -x "$NTV_ROOT/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$NTV_ROOT/bin" UV_NO_MODIFY_PATH=1 sh
fi
UV="$NTV_ROOT/bin/uv"
"$UV" sync --extra gpu
"$UV" pip install --no-deps --require-hashes -r third_party/NextTransformer/requirements/linear.txt
.venv/bin/python - <<'PY'
import torch, transformers, fla, nexttransformer
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())
print("transformers", transformers.__version__, "fla", fla.__version__)
PY
touch "$NTV_ROOT/.env_ready"
