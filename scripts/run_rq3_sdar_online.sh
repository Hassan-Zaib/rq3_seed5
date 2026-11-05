#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data/tiny-imagenet-200}"
RUN_ROOT="${RUN_ROOT:-results_sdar_online}"
FIXED_PROBE_DIR="${FIXED_PROBE_DIR:-fixed_probe}"
DEFENSES="${DEFENSES:-none q8 q4 q2}"
CUTS="${CUTS:-c1 c2 c3 c4}"
SDAR_STEPS="${SDAR_STEPS:-40000}"
SDAR_BATCH_SIZE="${SDAR_BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-0}"
ATTACK_POOL="${ATTACK_POOL:-train}"
FG_OPTIMIZER="${FG_OPTIMIZER:-adamw}"
FG_LR="${FG_LR:-0.002}"
FG_WEIGHT_DECAY="${FG_WEIGHT_DECAY:-0.05}"
FG_SCHEDULE="${FG_SCHEDULE:-warmup_cosine}"
FG_WARMUP_STEPS="${FG_WARMUP_STEPS:-3000}"
FG_MIN_LR="${FG_MIN_LR:-0.000001}"
FG_LABEL_SMOOTHING="${FG_LABEL_SMOOTHING:-0.1}"
FG_GRAD_CLIP="${FG_GRAD_CLIP:-1.0}"
FG_MIX_PROB="${FG_MIX_PROB:-1.0}"
FG_MIXUP_ALPHA="${FG_MIXUP_ALPHA:-0.8}"
FG_CUTMIX_ALPHA="${FG_CUTMIX_ALPHA:-1.0}"
FG_CUTMIX_SWITCH_PROB="${FG_CUTMIX_SWITCH_PROB:-0.5}"
MODEL_EMA_DECAY="${MODEL_EMA_DECAY:-0.9999}"
RUN_SMOKE="${RUN_SMOKE:-1}"
SEED="${SEED:-5678}"

echo "===== RQ3 clean protocol: ConvNeXt-Tiny + online SDAR only ====="
echo "RUN_ROOT=${RUN_ROOT}"
echo "SEED=${SEED}"
echo "DEFENSES=${DEFENSES}"
echo "CUTS=${CUTS}"
echo "SDAR_STEPS=${SDAR_STEPS}"
echo "SDAR_BATCH_SIZE=${SDAR_BATCH_SIZE}"
echo "ATTACK_POOL=${ATTACK_POOL}"
echo "FG_OPTIMIZER=${FG_OPTIMIZER}"
echo "FG_LR=${FG_LR}"
echo "FG_SCHEDULE=${FG_SCHEDULE}"
echo "FG_MIX_PROB=${FG_MIX_PROB}"
echo "MODEL_EMA_DECAY=${MODEL_EMA_DECAY}"
echo "AMP=bf16"

python scripts/verify_rq3_implementation.py

if [[ ! -f "$FIXED_PROBE_DIR/fixed_probe_indices.npy" ]]; then
  python scripts/make_fixed_probe.py \
    --data-root "$DATA_ROOT" \
    --out-dir "$FIXED_PROBE_DIR" \
    --num-images 20 \
    --seed 9090
else
  echo "Reusing immutable fixed probe from $FIXED_PROBE_DIR"
fi
sha256sum "$FIXED_PROBE_DIR/fixed_probe_indices.npy" "$FIXED_PROBE_DIR/fixed_probe_images.npy"

for cut in $CUTS; do
  python scripts/check_shapes.py --cut "$cut"
done

if [[ "$RUN_SMOKE" == "1" ]]; then
  echo "===== RQ3 online SDAR smoke: none c1, 20 steps ====="
  python scripts/run_sdar_on_convnext_smashed.py \
    --data-root "$DATA_ROOT" \
    --run-root "${RUN_ROOT}_smoke" \
    --fixed-probe-dir "$FIXED_PROBE_DIR" \
    --cut c1 \
    --defense none \
    --steps 20 \
    --batch-size "$SDAR_BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --attack-pool "$ATTACK_POOL" \
    --victim-train-transform \
    --fg-optimizer "$FG_OPTIMIZER" \
    --fg-lr "$FG_LR" \
    --fg-weight-decay "$FG_WEIGHT_DECAY" \
    --fg-schedule "$FG_SCHEDULE" \
    --fg-warmup-steps "$FG_WARMUP_STEPS" \
    --fg-min-lr "$FG_MIN_LR" \
    --fg-label-smoothing "$FG_LABEL_SMOOTHING" \
    --fg-mix-prob "$FG_MIX_PROB" \
    --fg-mixup-alpha "$FG_MIXUP_ALPHA" \
    --fg-cutmix-alpha "$FG_CUTMIX_ALPHA" \
    --fg-cutmix-switch-prob "$FG_CUTMIX_SWITCH_PROB" \
    --fg-grad-clip "$FG_GRAD_CLIP" \
    --amp \
    --amp-dtype bf16 \
    --model-ema \
    --model-ema-decay "$MODEL_EMA_DECAY" \
    --no-compute-lpips
fi

for defense in $DEFENSES; do
  echo "===== RQ3 online SDAR group: defense=${defense} ====="
  for cut in $CUTS; do
    attack_summary="${RUN_ROOT}/${defense}/${cut}/sdar/full_sdar_summary.json"
    if [[ -f "$attack_summary" ]]; then
      echo "===== Skip completed online SDAR ${defense} ${cut} ====="
    else
      echo "===== Online SDAR ConvNeXt-Tiny TinyImageNet ${defense} ${cut} ====="
      python scripts/run_sdar_on_convnext_smashed.py \
        --data-root "$DATA_ROOT" \
        --run-root "$RUN_ROOT" \
        --fixed-probe-dir "$FIXED_PROBE_DIR" \
        --cut "$cut" \
        --seed "$SEED" \
        --split-seed 42 \
        --defense "$defense" \
        --steps "$SDAR_STEPS" \
        --batch-size "$SDAR_BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --attack-pool "$ATTACK_POOL" \
        --victim-train-transform \
        --fg-optimizer "$FG_OPTIMIZER" \
        --fg-lr "$FG_LR" \
        --fg-weight-decay "$FG_WEIGHT_DECAY" \
        --fg-schedule "$FG_SCHEDULE" \
        --fg-warmup-steps "$FG_WARMUP_STEPS" \
        --fg-min-lr "$FG_MIN_LR" \
        --fg-label-smoothing "$FG_LABEL_SMOOTHING" \
        --fg-mix-prob "$FG_MIX_PROB" \
        --fg-mixup-alpha "$FG_MIXUP_ALPHA" \
        --fg-cutmix-alpha "$FG_CUTMIX_ALPHA" \
        --fg-cutmix-switch-prob "$FG_CUTMIX_SWITCH_PROB" \
        --fg-grad-clip "$FG_GRAD_CLIP" \
        --amp \
        --amp-dtype bf16 \
        --model-ema \
        --model-ema-decay "$MODEL_EMA_DECAY"
    fi
  done
  python scripts/make_probe_sheet.py \
    --run-root "$RUN_ROOT" \
    --defense "$defense" \
    --out-dir "${RUN_ROOT}/probe_sheets"
  python scripts/print_rq3_sdar_online_results.py --run-root "$RUN_ROOT" --defense "$defense"
done

python scripts/print_rq3_sdar_online_results.py --run-root "$RUN_ROOT" --defense all
echo "[RQ3 clean ConvNeXt online SDAR finished]"
