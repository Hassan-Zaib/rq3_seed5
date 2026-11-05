#!/usr/bin/env bash
D="$1"
cd ~/saia_sok/RQ3/rq3_final_gaussian_noise
for S in 4 5; do
  SEED=$S RUN_SMOKE=0 RUN_ROOT=rq3_seeds/seed${S}_${D} \
  FIXED_PROBE_DIR=$HOME/saia_sok/RQ3/fixed_probe \
  DEFENSES="$D" CUTS="c1 c2 c3 c4" \
  bash scripts/run_rq3_sdar_online.sh > rq3_seed${S}_${D}.log 2>&1
done
