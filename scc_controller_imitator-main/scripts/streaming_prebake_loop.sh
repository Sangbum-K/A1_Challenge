#!/bin/bash
# streaming prebake: 매 INTERVAL 초 마다 manifest re-build (filter-existing) + prebake 재실행.
# train_iy --reload-on-change 가 prebake npz mtime 체크 후 Dataset rebuild.
#
# 사용:
#   bash scripts/streaming_prebake_loop.sh spd            # 600s interval default
#   INTERVAL=300 bash scripts/streaming_prebake_loop.sh spd
#
# 지원 track 키:
#   spd            speedway 0519_agent1_pertX_pitX
#   sangam_pertO   sangam 0519_pertO_pitX
#   sangam_pertX   sangam 0519_pertX_pitX
set -e

TRACK_KEY="${1:?usage: $0 <spd|sangam_pertO|sangam_pertX>}"
INTERVAL="${INTERVAL:-600}"
WORKERS="${WORKERS:-4}"     # 합성과 CPU 경쟁 피하기

case "$TRACK_KEY" in
  spd)
    TRACK=speedway_track
    POLICY=0519_agent1_pertX_pitX
    FRAME_LIST=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/scc_frame_list_spd.txt
    SCC_DIR=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/scc_synth_v2_spd
    RAND_BANK=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/scc_traj_bank_spd
    LOG_BANK=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/scc_logged_bank_spd
    MANIFEST=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/scc_synth_pair_manifest_spd.json
    PREBAKE=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/prebake/f1_prebake_iy_spd_streaming.npz
    AUX_DIR=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/aux_targets_spd_fx640
    ;;
  sangam_pertO)
    TRACK=sangam_track
    POLICY=0519_pertO_pitX
    FRAME_LIST=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/scc_frame_list_sangam_pertO.txt
    SCC_DIR=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/scc_synth_v2_sangam_pertO
    RAND_BANK=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/scc_traj_bank_sangam_pertO
    LOG_BANK=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/scc_logged_bank_sangam_pertO
    MANIFEST=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/scc_synth_pair_manifest_sangam_pertO_streaming.json
    PREBAKE=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/prebake/f1_prebake_iy_sangam_pertO_streaming.npz
    AUX_DIR=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/aux_targets_sangam_fx640
    ;;
  sangam_pertX)
    TRACK=sangam_track
    POLICY=0519_pertX_pitX
    FRAME_LIST=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/scc_frame_list_sangam_pertX.txt
    SCC_DIR=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/scc_synth_v2_sangam_pertX
    RAND_BANK=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/scc_traj_bank_sangam_pertX
    LOG_BANK=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/scc_logged_bank_sangam_pertX
    MANIFEST=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/scc_synth_pair_manifest_sangam_pertX_streaming.json
    PREBAKE=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/prebake/f1_prebake_iy_sangam_pertX_streaming.npz
    AUX_DIR=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/aux_targets_sangam_fx640
    ;;
  *)
    echo "unknown track key: $TRACK_KEY" >&2
    exit 1
    ;;
esac

RAW_BASE=/home/ailab-12/AILabSSD/02_Data/57_A1_E2E/00_raw
ROOT=/home/ailab-12/git/scc_controller_imitator
PY=~/miniforge3/envs/ml_con/bin/python

echo "[stream-loop] track=$TRACK_KEY interval=${INTERVAL}s workers=$WORKERS"
echo "[stream-loop] manifest=$MANIFEST"
echo "[stream-loop] prebake =$PREBAKE"

cd "$ROOT"
iter=0
while true; do
  iter=$((iter+1))
  TS=$(date +%H:%M:%S)
  N_SCC=$(find "$SCC_DIR" -name "*.scc.txt" 2>/dev/null | wc -l)
  echo "[stream-loop $iter] $TS  scc.txt count: $N_SCC"

  # 1) manifest re-build
  $PY scripts/build_pair_manifest.py \
      --frame-list "$FRAME_LIST" \
      --track "$TRACK" --policy "$POLICY" \
      --raw-base "$RAW_BASE" \
      --scc-dir "$SCC_DIR" \
      --rand-bank "$RAND_BANK" \
      --log-bank "$LOG_BANK" \
      --k 100 \
      --filter-existing \
      --out "$MANIFEST" 2>&1 | tail -3

  # 2) prebake (obj_target 채우기: --aux-policy-dir 적용)
  $PY -u scripts/prebake_iy_dataset.py \
      --manifest "$MANIFEST" \
      --out "$PREBAKE" \
      --workers "$WORKERS" \
      --prev-dtick 5 \
      --aux-policy-dir "${POLICY}:${AUX_DIR}" 2>&1 | tail -3

  echo "[stream-loop $iter] sleep ${INTERVAL}s"
  sleep "$INTERVAL"
done
