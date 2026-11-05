#!/usr/bin/env bash
# Tests checkpoint/resume of scripts/run_sdar_on_convnext_smashed.py on the GPU with the real Tiny ImageNet:
#   1. the original script (before checkpointing was added) straight to 40 steps
#   2. the new script straight to 40 steps, saving a checkpoint every 10 steps
#   3. the new script, checkpoint every 10 steps, SIGTERM after step 20, then the same command again to 40
# then compares 1 vs 2 (checkpointing changes nothing) and 2 vs 3 (resuming changes nothing): final weights of
# every model (incl. EMA), the per-step loss history, the reconstructions and the summary. Prints PASS or FAIL.
#
# All runs use the job command of scripts/run_one.sh (the DGX seed-5 settings) with --steps 40; only
# --batch-size may be lowered for this test. Every run goes through a small wrapper that turns on PyTorch's
# deterministic mode (cuDNN deterministic, deterministic algorithms where they exist), so the GPU runs can be
# compared; the scripts themselves are unchanged. Output: test_resume/ in the repository folder; the test's
# resume checkpoints are deleted at the end. Needs a git clone (the original script is read from git).
#
# Usage: bash scripts/test_resume.sh [--data-root DIR] [--batch-size N] [--defense D] [--cut C]
#        (defaults: data/tiny-imagenet-200, 256, noise0.05, c1)
set -euo pipefail
cd "$(dirname "$0")/.."

# The original script is the one in this commit (main before checkpointing was added).
ORIGINAL_COMMIT=a24a236
STEPS=40
DATA_ROOT=data/tiny-imagenet-200
BATCH_SIZE=256
DEFENSE=noise0.05
CUT=c1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --defense) DEFENSE="$2"; shift 2 ;;
    --cut) CUT="$2"; shift 2 ;;
    *) echo "usage: bash scripts/test_resume.sh [--data-root DIR] [--batch-size N] [--defense D] [--cut C]" >&2; exit 2 ;;
  esac
done

if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "PyTorch does not see a GPU; this test runs on the GPU" >&2
  exit 1
fi
[[ -d "$DATA_ROOT/train" && -f "$DATA_ROOT/val/val_annotations.txt" ]] || {
  echo "not a Tiny ImageNet folder: $DATA_ROOT (README section 2)" >&2
  exit 1
}

OUT=test_resume
rm -rf "$OUT"
# The resume checkpoints are deleted at the end, also when the test fails or is interrupted.
trap 'rm -rf "$OUT"/runs/*/*/*/sdar/checkpoints' EXIT
mkdir -p "$OUT/original/scripts" "$OUT/logs"
git show "${ORIGINAL_COMMIT}:scripts/run_sdar_on_convnext_smashed.py" > "$OUT/original/scripts/run_sdar_on_convnext_smashed.py"
ln -s ../../src "$OUT/original/src"
cat > "$OUT/deterministic_run.py" <<'EOF'
import runpy
import sys

import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
EOF
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# The job command from run_one.sh, with the test's run root, steps and batch size.
read -ra JOB <<< "$(DRY_RUN=1 bash scripts/run_one.sh "$DEFENSE" "$CUT")"
[[ "${JOB[0]:-}" == python ]] || { echo "run_one.sh did not print a command: ${JOB[*]}" >&2; exit 1; }
job_args() {  # $1 = run root
  local out=() i
  for ((i = 2; i < ${#JOB[@]}; i++)); do
    case "${JOB[i]}" in
      --run-root) out+=(--run-root "$1"); i=$((i + 1)) ;;
      --data-root) out+=(--data-root "$DATA_ROOT"); i=$((i + 1)) ;;
      --steps) out+=(--steps "$STEPS"); i=$((i + 1)) ;;
      --batch-size) out+=(--batch-size "$BATCH_SIZE"); i=$((i + 1)) ;;
      *) out+=("${JOB[i]}") ;;
    esac
  done
  printf '%s\n' "${out[@]}"
}
ARGS_ORIGINAL=(); ARGS_STRAIGHT=(); ARGS_RESUMED=()
while IFS= read -r a; do ARGS_ORIGINAL+=("$a"); done < <(job_args "$OUT/runs/original")
while IFS= read -r a; do ARGS_STRAIGHT+=("$a"); done < <(job_args "$OUT/runs/straight")
while IFS= read -r a; do ARGS_RESUMED+=("$a"); done < <(job_args "$OUT/runs/resumed")
run() { python "$OUT/deterministic_run.py" "$@"; }

echo "===== 1. original script, ${STEPS} steps (${DEFENSE} ${CUT}, batch ${BATCH_SIZE}) ====="
run "$OUT/original/scripts/run_sdar_on_convnext_smashed.py" "${ARGS_ORIGINAL[@]}" > "$OUT/logs/1_original.log" 2>&1

echo "===== 2. new script, ${STEPS} steps, checkpoint every 10 ====="
run scripts/run_sdar_on_convnext_smashed.py "${ARGS_STRAIGHT[@]}" --ckpt-every 10 > "$OUT/logs/2_straight.log" 2>&1

echo "===== 3. new script, checkpoint every 10, SIGTERM after step 20, then rerun to ${STEPS} ====="
CKPT_DIR="$OUT/runs/resumed/${DEFENSE}/${CUT}/sdar/checkpoints"
run scripts/run_sdar_on_convnext_smashed.py "${ARGS_RESUMED[@]}" --ckpt-every 10 > "$OUT/logs/3a_until_sigterm.log" 2>&1 &
pid=$!
until [[ -f "$CKPT_DIR/ckpt_step0000020.pt" ]] || ! kill -0 "$pid" 2>/dev/null; do sleep 0.1; done
kill -TERM "$pid" 2>/dev/null || true
status=0
wait "$pid" || status=$?
if [[ "$status" != 143 ]]; then
  echo "FAIL: the first part of run 3 exited with $status instead of 143 (saved on SIGTERM); see $OUT/logs/3a_until_sigterm.log"
  exit 1
fi
grep -h "Signal 15: saved" "$OUT/logs/3a_until_sigterm.log"
run scripts/run_sdar_on_convnext_smashed.py "${ARGS_RESUMED[@]}" --ckpt-every 10 > "$OUT/logs/3b_resumed.log" 2>&1
grep -h "Resumed from" "$OUT/logs/3b_resumed.log" || { echo "FAIL: run 3 did not resume"; exit 1; }

echo "===== compare ====="
python - "$OUT/runs" "$DEFENSE" "$CUT" <<'EOF'
import json
import sys
from pathlib import Path

import numpy as np
import torch

runs, defense, cut = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
ATOL, RTOL = 1e-4, 1e-3  # used only when a pair is not bit-identical


def load(name):
    d = runs / name / defense / cut / "sdar"
    last = torch.load(d / "full_sdar_last.pt", map_location="cpu", weights_only=False)
    tensors = {}
    for key, state in last.items():
        if key != "summary" and state is not None:
            tensors.update({f"{key}.{k}": v for k, v in state.items()})
    history = np.load(d / "full_sdar_history.npy", allow_pickle=True).item()
    arrays = {n: np.load(d / n) for n in ("reconstructions.npy", "fixed_probe_reconstructions.npy")}
    summary = json.loads((d / "full_sdar_summary.json").read_text())
    return tensors, history, arrays, summary


def compare(a_name, b_name):
    a, b = load(a_name), load(b_name)
    exact, close, worst = True, True, {}
    pairs = [(f"weight {k}", a[0][k].double().numpy(), b[0][k].double().numpy()) for k in a[0]]
    pairs += [(f"history {k}", np.asarray(a[1][k]), np.asarray(b[1][k])) for k in a[1]]
    pairs += [(n, a[2][n], b[2][n]) for n in a[2]]
    if set(a[0]) != set(b[0]) or set(a[1]) != set(b[1]):
        print(f"{a_name} vs {b_name}: different weight or history keys")
        return False
    for label, x, y in pairs:
        if x.shape != y.shape:
            print(f"{a_name} vs {b_name}: {label} shape {x.shape} vs {y.shape}")
            return False
        if not np.array_equal(x, y):
            exact = False
            close = close and np.allclose(x, y, atol=ATOL, rtol=RTOL)
            group = label.split(" ")[0] if not label.endswith(".npy") else label
            worst[group] = max(worst.get(group, 0.0), float(np.max(np.abs(x - y))))
    summary_diff = [k for k in a[3] if k != "total_runtime_sec" and a[3][k] != b[3].get(k)]
    if exact:
        print(f"{a_name} vs {b_name}: bit-identical ({len(a[0])} weight tensors, {len(a[1])} history series, "
              f"reconstructions); summary fields differing: {summary_diff or 'none'}")
    else:
        verdict = "within tolerance" if close else "NOT within tolerance"
        print(f"{a_name} vs {b_name}: not bit-identical, {verdict} (atol {ATOL}, rtol {RTOL}); max |diff| {worst}; "
              f"summary fields differing: {summary_diff}")
    return close


ok_1_2 = compare("original", "straight")
ok_2_3 = compare("straight", "resumed")
print("PASS" if ok_1_2 and ok_2_3 else "FAIL")
sys.exit(0 if ok_1_2 and ok_2_3 else 1)
EOF
