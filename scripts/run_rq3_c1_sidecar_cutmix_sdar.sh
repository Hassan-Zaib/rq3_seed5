#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data/tiny-imagenet-200}"
RUN_ROOT="${RUN_ROOT:-results_c1_sidecar_cutmix_sdar}"
FIXED_PROBE_DIR="${FIXED_PROBE_DIR:-fixed_probe}"
EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-256}"
ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"

echo "===== RQ3 C1 ConvNeXt victim with Mixup/CutMix + sidecar SDAR ====="
echo "RUN_ROOT=${RUN_ROOT}"
echo "EPOCHS=${EPOCHS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "ATTACK_BATCH_SIZE=${ATTACK_BATCH_SIZE}"

python scripts/verify_rq3_implementation.py

if [[ ! -f "$FIXED_PROBE_DIR/fixed_probe_indices.npy" ]]; then
  python scripts/make_fixed_probe.py \
    --data-root "$DATA_ROOT" \
    --out-dir "$FIXED_PROBE_DIR" \
    --num-images 20 \
    --seed 9090
fi

python scripts/run_sdar_sidecar_convnext.py \
  --data-root "$DATA_ROOT" \
  --run-root "$RUN_ROOT" \
  --fixed-probe-dir "$FIXED_PROBE_DIR" \
  --cut c1 \
  --defense none \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --attack-batch-size "$ATTACK_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --lr 0.004 \
  --warmup-epochs 20 \
  --mix-prob 1.0 \
  --model-ema-decay 0.9999 \
  --amp-dtype bf16 \
  --grad-clip 1.0 \
  --amp

echo "[RQ3 C1 sidecar CutMix SDAR finished]"
