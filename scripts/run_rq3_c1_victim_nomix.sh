#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data/tiny-imagenet-200}"
RUN_ROOT="${RUN_ROOT:-results_c1_victim_nomix}"
FIXED_PROBE_DIR="${FIXED_PROBE_DIR:-fixed_probe}"
EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"

echo "===== RQ3 C1 victim-only ConvNeXt, no Mixup/CutMix ====="
echo "RUN_ROOT=${RUN_ROOT}"
echo "EPOCHS=${EPOCHS}"
echo "BATCH_SIZE=${BATCH_SIZE}"

python scripts/verify_rq3_implementation.py

if [[ ! -f "$FIXED_PROBE_DIR/fixed_probe_indices.npy" ]]; then
  python scripts/make_fixed_probe.py \
    --data-root "$DATA_ROOT" \
    --out-dir "$FIXED_PROBE_DIR" \
    --num-images 20 \
    --seed 9090
fi

python scripts/train_split_convnext.py \
  --data-root "$DATA_ROOT" \
  --output-dir "$RUN_ROOT" \
  --fixed-probe-dir "$FIXED_PROBE_DIR" \
  --cut c1 \
  --defense none \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --lr 0.004 \
  --warmup-epochs 20 \
  --update-freq 1 \
  --mix-prob 0.0 \
  --model-ema-decay 0.9999 \
  --amp-dtype bf16 \
  --grad-clip 1.0 \
  --amp

python - <<'PY'
import json
from pathlib import Path
p = Path("results_c1_victim_nomix/none/c1/victim/victim_summary.json")
d = json.loads(p.read_text())
print("===== C1 victim-only no Mixup/CutMix =====")
for k in ["best_val_acc", "final_val_acc", "final_raw_val_acc", "final_ema_val_acc", "final_train_acc", "total_runtime_sec"]:
    v = d.get(k)
    print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")
PY

echo "[RQ3 C1 victim-only no Mixup/CutMix finished]"
