#!/usr/bin/env bash
# Runs on the 灿果 box (DGX OS, arm64, root). Idempotent.
set -uo pipefail
cd "$(dirname "$0")"

echo "[bootstrap] arch=$(uname -m) python=$(python3 -V 2>&1)"

if ! python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "[bootstrap] installing torch (cu130, aarch64)"
  pip3 install --no-cache-dir --break-system-packages \
    --index-url https://download.pytorch.org/whl/cu130 torch
fi

pip3 install --no-cache-dir --break-system-packages -r requirements.txt

python3 - <<'PY'
import torch
p = torch.cuda.get_device_properties(0)
print(f"[bootstrap] torch={torch.__version__} cuda={torch.version.cuda} "
      f"cc={p.major}.{p.minor} vram={p.total_memory/2**30:.1f}GiB sm={p.multi_processor_count}")
print(f"[bootstrap] arch list: {torch.cuda.get_arch_list()}")
for m in ("diffusers", "transformers", "av", "cv2"):
    try:
        mod = __import__(m)
        print(f"[bootstrap] {m}={getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"[bootstrap] {m} MISSING: {e}")
PY

command -v ffmpeg >/dev/null || echo "[bootstrap] WARNING: ffmpeg not on PATH"
command -v nvidia-smi >/dev/null || echo "[bootstrap] WARNING: nvidia-smi not on PATH"

echo "[bootstrap] done"
