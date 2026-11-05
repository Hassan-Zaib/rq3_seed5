#!/usr/bin/env bash
# Checks the repository without training anything:
#   1. syntax of every Python and shell file
#   2. jobs_seed5.txt = the 5 defenses x c1-c4
#   3. the fixed probe files are unchanged
#   4. scripts/run_one.sh builds the same python command as the DGX seed-5 run (dgx/run_seeds45_one_defense.sh),
#      for all 20 jobs. The DGX launcher is executed as is, with python replaced by a stub that only records
#      its arguments. The only allowed differences are the two paths --run-root and --fixed-probe-dir.
#   5. run_one.sh refuses other seeds and skips a job whose full_sdar_summary.json exists
# Run from anywhere: bash check_seed5.sh
set -euo pipefail
cd "$(dirname "$0")"
REPO="$(pwd)"
PY="$(command -v python || command -v python3)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }

echo "== 1. syntax"
n=0
while IFS= read -r -d '' f; do
  PYTHONPYCACHEPREFIX="$TMP/pycache" "$PY" -m py_compile "$f" || fail "python syntax: $f"
  n=$((n + 1))
done < <(find . -name '*.py' -not -path './.git/*' -print0)
echo "ok: $n python files"
n=0
while IFS= read -r -d '' f; do
  bash -n "$f" || fail "shell syntax: $f"
  n=$((n + 1))
done < <(find . -name '*.sh' -not -path './.git/*' -print0)
echo "ok: $n shell scripts"

echo "== 2. jobs_seed5.txt"
expected_jobs="$(for d in none q4 q2 noise0.01 noise0.05; do for c in c1 c2 c3 c4; do echo "$d $c"; done; done)"
[[ "$(cat jobs_seed5.txt)" == "$expected_jobs" ]] || fail "jobs_seed5.txt is not the 20 expected lines"
echo "ok: 20 jobs"

echo "== 3. fixed probe"
"$PY" - <<'EOF'
import hashlib, sys
from pathlib import Path
expected = {
    "fixed_probe_images.npy": "e0b16d8c278f85b85dd5b7aa18c1ce2cd44f6e822edd21b86f2fe10f4c8b7217",
    "fixed_probe_images_norm.npy": "7e92b7c12f5e0adf04c46ae425bab6b4962cdc21411aeef6013517d1211566a3",
    "fixed_probe_indices.npy": "3816111b0b920b4a3a04058186c19240b86dd9090e34c56a88888e33d81308e2",
    "fixed_probe_labels.npy": "b817dfed4b67dced25ba37baf123ea9b29d967753a64d2e4eba6ef161efa564d",
    "fixed_probe_meta.json": "92a5331bdf2835e7dc81dd1b8e9f0cfa3656275642e33cffa4eaff78a9ccb611",
}
for name, sha in expected.items():
    got = hashlib.sha256((Path("fixed_probe") / name).read_bytes()).hexdigest()
    if got != sha:
        sys.exit(f"FAIL: fixed_probe/{name} sha256 {got} != {sha}")
print(f"ok: {len(expected)} fixed probe files")
EOF

echo "== 4. run_one.sh command vs DGX seed-5 command"
# The DGX launcher cds into ~/saia_sok/RQ3/rq3_final_gaussian_noise: give it a fake HOME holding a copy of scripts/.
DGX_DIR="$TMP/home/saia_sok/RQ3/rq3_final_gaussian_noise"
mkdir -p "$DGX_DIR" "$TMP/bin"
cp -R scripts "$DGX_DIR/"
cat > "$TMP/bin/python" <<'EOF'
#!/usr/bin/env bash
printf '%q ' python "$@" >> "$STUB_CALLS"
echo >> "$STUB_CALLS"
EOF
printf '#!/usr/bin/env bash\nexit 0\n' > "$TMP/bin/sha256sum"
chmod +x "$TMP/bin/python" "$TMP/bin/sha256sum"
export STUB_CALLS="$TMP/dgx_calls.txt"
: > "$STUB_CALLS"
for d in none q4 q2 noise0.01 noise0.05; do
  HOME="$TMP/home" PATH="$TMP/bin:$PATH" bash dgx/run_seeds45_one_defense.sh "$d"
done
normalize() { sed -E 's/--run-root [^ ]+ /--run-root RUN_ROOT /; s/--fixed-probe-dir [^ ]+ /--fixed-probe-dir FIXED_PROBE_DIR /'; }
grep -c -- 'run_sdar_on_convnext_smashed.py .*--seed 4 ' "$STUB_CALLS" | grep -qx 20 || fail "DGX launcher did not produce 20 seed-4 commands"
# A copy of the repository without results, so that no job is skipped.
mkdir -p "$TMP/repo"
cp -R scripts "$TMP/repo/"
n=0
while read -r d c; do
  dgx="$(grep -- 'run_sdar_on_convnext_smashed.py ' "$STUB_CALLS" | grep -- "--seed 5 " | grep -- "--cut $c " | grep -- "--defense $d " || true)"
  [[ "$(printf '%s\n' "$dgx" | grep -c .)" == 1 ]] || fail "expected one DGX seed-5 command for $d $c"
  ours="$(cd "$TMP/repo" && DRY_RUN=1 bash scripts/run_one.sh "$d" "$c")"
  [[ "$(printf '%s\n' "$dgx" | normalize)" == "$(printf '%s\n' "$ours" | normalize)" ]] || {
    echo "DGX:  $dgx" >&2
    echo "ours: $ours" >&2
    fail "command differs for $d $c"
  }
  [[ "$dgx" == *"--run-root rq3_seeds/seed5_$d "* ]] || fail "unexpected DGX run root for $d $c"
  [[ "$ours" == *"--run-root results_seed5 --fixed-probe-dir fixed_probe "* ]] || fail "unexpected paths for $d $c"
  n=$((n + 1))
done < jobs_seed5.txt
echo "ok: $n commands identical to the DGX seed-5 commands (except --run-root and --fixed-probe-dir)"
echo "example: $(cd "$TMP/repo" && DRY_RUN=1 bash scripts/run_one.sh none c1)"

echo "== 5. seed and skip"
if (cd "$TMP/repo" && SEED=4 DRY_RUN=1 bash scripts/run_one.sh none c1) 2>/dev/null; then fail "SEED=4 was accepted"; fi
if (cd "$TMP/repo" && DRY_RUN=1 bash scripts/run_one.sh none c1 4) 2>/dev/null; then fail "a third argument was accepted"; fi
if (cd "$TMP/repo" && DRY_RUN=1 bash scripts/run_one.sh q8 c1) 2>/dev/null; then fail "defense q8 was accepted"; fi
mkdir -p "$TMP/repo/results_seed5/q2/c3/sdar"
echo '{}' > "$TMP/repo/results_seed5/q2/c3/sdar/full_sdar_summary.json"
out="$(cd "$TMP/repo" && bash scripts/run_one.sh q2 c3)" || fail "skip did not exit 0"
[[ "$out" == Skip* ]] || fail "existing summary was not skipped: $out"
echo "ok: other seeds refused, finished job skipped"

[[ ! -e "$REPO/data" || -d "$REPO/data" && ! -L "$REPO/data" ]] || fail "data/ must be your own folder, not the DGX link"
echo "ALL CHECKS PASSED"
