#!/usr/bin/env bash
# Clone SGLang at the exact commit these results were measured on, apply the
# LLaDA-Hybrid patch, and install it.
#
#   ./sglang/apply.sh [target-dir]     (default: ./sglang-src)
#
# The patch is against upstream sgl-project/sglang @ BASE_COMMIT below. It adds
# the hybrid attention backend, its fused Triton kernel, the model class, and the
# decoder and memory-pool changes the block-hybrid path needs. Nothing else
# in SGLang is modified, so `git diff` against the base commit is the complete
# statement of what this work changes in the serving engine.
set -euo pipefail

BASE_COMMIT=2b47bd3a348bf5ca52a3a4910b2e22c851798576   # 2026-03-23, between v0.5.9 and v0.5.10
REPO=https://github.com/sgl-project/sglang.git

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$HERE/llada-hybrid.patch"
TARGET="${1:-$(dirname "$HERE")/sglang-src}"

if [ -e "$TARGET" ]; then
  echo "error: $TARGET already exists; remove it or pass a different path" >&2
  exit 1
fi

echo "==> cloning $REPO"
git clone "$REPO" "$TARGET"

echo "==> checking out $BASE_COMMIT"
git -C "$TARGET" checkout --detach "$BASE_COMMIT"

echo "==> applying $(basename "$PATCH")"
git -C "$TARGET" apply --verbose "$PATCH"

cat <<EOF

Patched SGLang is at: $TARGET

Install it (a CUDA build takes a while):

    pip install -e "$TARGET/python"

Then check the model class is registered:

    python -c "from sglang.srt.models.llada_fast import LLaDAFastForCausalLM; print('ok')"
EOF
