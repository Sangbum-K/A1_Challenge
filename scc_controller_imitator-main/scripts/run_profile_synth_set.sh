#!/usr/bin/env bash
# 한 dataset 에 대해 5 profile (p1_aggr~p5_cons) 순차 synth_batch + v_scc→pedal_label.
# 사용: bash scripts/run_profile_synth_set.sh <ds> <frame_list> <raw_base> <prebake_npz> [N_SHARDS=8]
set -euo pipefail
DS="${1:?ds}"; FL="${2:?frame_list}"; RAW="${3:?raw_base}"; PB="${4:?prebake_npz}"; N="${5:-8}"
D=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data
PY=/home/ailab-12/miniforge3/envs/ml_con/bin/python
RUN=/home/ailab-12/git/scc_controller_imitator/scripts/run_synth_batch_shards.sh
CONV=/home/ailab-12/git/scc_controller_imitator/scripts/gen_profile_pedal_labels.py

# profile 표 (확정)
declare -A THW=( [p1]=0.4 [p2]=0.8 [p3]=1.2 [p4]=1.6 [p5]=2.0 )
declare -A SMG=( [p1]=2.0 [p2]=3.0 [p3]=4.0 [p4]=5.5 [p5]=7.0 )

for prof in p1 p2 p3 p4 p5; do
  OUT="$D/scc_synth_${prof}_${DS}"
  echo "==================== [$DS/$prof] th=${THW[$prof]} sm=${SMG[$prof]} → $OUT ===================="
  TH="${THW[$prof]}" SM="${SMG[$prof]}" WARMUP=0 \
    bash "$RUN" "$FL" "$RAW" "$OUT" "$N"
  echo "-------------------- [$DS/$prof] v_scc → pedal_label --------------------"
  $PY "$CONV" --ds "$DS" --prebake "$PB" --scc-dir "$OUT" --profile "$prof"
done
echo "[DONE] $DS all 5 profiles"
