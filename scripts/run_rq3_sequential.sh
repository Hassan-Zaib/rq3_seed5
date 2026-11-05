#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data/tiny-imagenet-200}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS="${EPOCHS:-300}"
SDAR_STEPS="${SDAR_STEPS:-20000}"
SDAR_BATCH_SIZE="${SDAR_BATCH_SIZE:-128}"
LR="${LR:-0.004}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-20}"
UPDATE_FREQ="${UPDATE_FREQ:-1}"
MIX_PROB="${MIX_PROB:-1.0}"
EMA_DECAY="${EMA_DECAY:-0.9999}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
FIXED_PROBE_DIR="${FIXED_PROBE_DIR:-fixed_probe}"
DEFENSES="${DEFENSES:-none q2 q4 q8}"

echo "===== RQ3 smoke: fixed probe and activation shapes ====="
python scripts/verify_rq3_implementation.py
if [[ ! -f "$FIXED_PROBE_DIR/fixed_probe_indices.npy" ]]; then
  python scripts/make_fixed_probe.py --data-root "$DATA_ROOT" --out-dir "$FIXED_PROBE_DIR" --num-images 20 --seed 9090
else
  echo "Reusing immutable fixed probe from $FIXED_PROBE_DIR"
fi
sha256sum "$FIXED_PROBE_DIR/fixed_probe_indices.npy" "$FIXED_PROBE_DIR/fixed_probe_images.npy"

for cut in c1 c2 c3 c4; do
  python scripts/check_shapes.py --cut "$cut"
done

echo "===== RQ3 smoke: 1 epoch victim + 20-step attack on c1 none ====="
python scripts/train_split_convnext.py \
  --data-root "$DATA_ROOT" \
  --output-dir results_smoke \
  --fixed-probe-dir "$FIXED_PROBE_DIR" \
  --cut c1 \
  --defense none \
  --epochs 1 \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --lr "$LR" \
  --warmup-epochs 0 \
  --update-freq "$UPDATE_FREQ" \
  --mix-prob "$MIX_PROB" \
  --model-ema-decay "$EMA_DECAY" \
  --amp-dtype "$AMP_DTYPE" \
  --grad-clip "$GRAD_CLIP" \
  --amp
python scripts/run_sdar_on_convnext_smashed.py \
  --data-root "$DATA_ROOT" \
  --run-root results_smoke \
  --fixed-probe-dir "$FIXED_PROBE_DIR" \
  --cut c1 \
  --defense none \
  --steps 20 \
  --batch-size "$SDAR_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --no-compute-lpips

for defense in $DEFENSES; do
  echo "===== RQ3 full group: defense=${defense} ====="
  for cut in c1 c2 c3 c4; do
    victim_summary="results/${defense}/${cut}/victim/victim_summary.json"
    attack_summary="results/${defense}/${cut}/sdar/full_sdar_summary.json"
    legacy_attack_summary="results/${defense}/${cut}/sdar/sdar_summary.json"

    if [[ -f "$victim_summary" ]]; then
      echo "===== Skip completed victim ${defense} ${cut} ====="
    else
      echo "===== Train ConvNeXt-Tiny TinyImageNet ${defense} ${cut} ====="
      python scripts/train_split_convnext.py \
        --data-root "$DATA_ROOT" \
        --output-dir results \
        --fixed-probe-dir "$FIXED_PROBE_DIR" \
        --cut "$cut" \
        --defense "$defense" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --lr "$LR" \
        --warmup-epochs "$WARMUP_EPOCHS" \
        --update-freq "$UPDATE_FREQ" \
        --mix-prob "$MIX_PROB" \
        --model-ema-decay "$EMA_DECAY" \
        --amp-dtype "$AMP_DTYPE" \
        --grad-clip "$GRAD_CLIP" \
        --amp
    fi

    if [[ ! -f "$attack_summary" && -f "$legacy_attack_summary" ]]; then
      echo "===== Remove legacy decoder-only attack ${defense} ${cut} ====="
      rm -rf "results/${defense}/${cut}/sdar"
    fi

    if [[ -f "$attack_summary" ]]; then
      echo "===== Skip completed attack ${defense} ${cut} ====="
    else
      echo "===== Full SDAR ConvNeXt-Tiny TinyImageNet ${defense} ${cut} ====="
      python scripts/run_sdar_on_convnext_smashed.py \
        --data-root "$DATA_ROOT" \
        --run-root results \
        --fixed-probe-dir "$FIXED_PROBE_DIR" \
        --cut "$cut" \
        --defense "$defense" \
        --steps "$SDAR_STEPS" \
        --batch-size "$SDAR_BATCH_SIZE" \
        --num-workers "$NUM_WORKERS"
    fi
  done
  python scripts/make_probe_sheet.py --run-root results --defense "$defense" --out-dir results/probe_sheets
  python scripts/print_rq3_results.py --run-root results --defense "$defense"
done

python scripts/print_rq3_results.py --run-root results --defense all
echo "[RQ3 ConvNeXt Tiny quant sequential finished]"
