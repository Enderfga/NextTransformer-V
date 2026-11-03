#!/usr/bin/env bash
# Download pinned inputs into $NTV_ROOT/data. Idempotent: `hf download` skips complete files.
# Usage: NTV_ROOT=<root> bash scripts/download.sh <item> [args]
#   climbmix <n_train_shards>   first n train shards + dev (06541) + validation (06542)
#   cc12m <first> <last>        webdataset shards cc12m-train-<first>..<last> (inclusive)
#   cc3m <first> <last>         webdataset shards cc3m-train-<first>..<last> (inclusive)
#   siglip2                     frozen vision encoder
#   coco5k                      COCO 2014 Karpathy 5k test images and captions (evaluation)
set -euo pipefail
: "${NTV_ROOT:?set NTV_ROOT}"
HF="$(pwd)/.venv/bin/hf"
D="$NTV_ROOT/data"; mkdir -p "$D"
CLIMBMIX_REV=915333b4f8b8684f39aeaafea600fea6f43fb703
CC12M_REV=796118f2eabdb9984f23f7f15d1e74d388612fc6
CC3M_REV=46f3d69f840e59d77d52e8decfe5baec97e94c7f
SIGLIP2_REV=dd658faac399427308559e2c3ac1e99cbe43845d
COCO5K_REV=551c4f7667f06fa82b4ef0a07617bfc4cf324ac3
case "$1" in
  climbmix)
    files=$(printf "shard_%05d.parquet " $(seq 0 $(( $2 - 1 ))))
    "$HF" download karpathy/climbmix-400b-shuffle --repo-type dataset --revision "$CLIMBMIX_REV" \
      $files shard_06541.parquet shard_06542.parquet --local-dir "$D/climbmix" ;;
  cc12m)
    files=$(for i in $(seq "$2" "$3"); do printf "cc12m-train-%04d.tar " "$i"; done)
    "$HF" download pixparse/cc12m-wds --repo-type dataset --revision "$CC12M_REV" \
      $files _info.json --local-dir "$D/cc12m" ;;
  cc3m)
    files=$(for i in $(seq "$2" "$3"); do printf "cc3m-train-%04d.tar " "$i"; done)
    "$HF" download pixparse/cc3m-wds --repo-type dataset --revision "$CC3M_REV" \
      $files _info.json --local-dir "$D/cc3m" ;;
  siglip2)
    "$HF" download google/siglip2-so400m-patch16-384 --revision "$SIGLIP2_REV" --local-dir "$D/siglip2-so400m-patch16-384" ;;
  coco5k)
    "$HF" download nlphuji/mscoco_2014_5k_test_image_text_retrieval --repo-type dataset --revision "$COCO5K_REV" \
      --local-dir "$D/coco5k"
    [ -d "$D/coco5k/images" ] || .venv/bin/python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
      "$D/coco5k/images_mscoco_2014_5k_test.zip" "$D/coco5k/images" ;;
  *) echo "unknown item: $1" >&2; exit 2 ;;
esac
echo "done: $*"
