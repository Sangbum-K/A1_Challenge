#!/usr/bin/env bash
# ACC closed-loop 런처 (acc-test branch) — prof635 + 5f a_2 predicted vL_rel 주입 (vrel 기반 제동).
#
# = run_closed_loop_prof635.sh + MODEL_ACC=1.
#   f_2 = prof635_side_predvrel_ttc (brake_only), a_2 vrel = a2_mf5f_acc (5f offsets 10,20,30,40,50).
#   i2_mem/gate 경로는 v7b 2-frame 유지(학습 정합). vrel scalar 만 5f 로 산출 후 f_2 에 주입.
#
# 사용:
#   bash scripts/run_closed_loop_acc.sh sangam mild
#   bash scripts/run_closed_loop_acc.sh speedway high --viz-a2
#   bash scripts/run_closed_loop_acc.sh sangam mild --rosmaster=http://localhost:11311
#   bash scripts/run_closed_loop_acc.sh sangam mild --f1=656   # f_1 main #656 override (--f1/--sync-ini 는 prof635 런처로 패스스루)
#
# env:
#   ACC_F2_CKPT       f_2 ckpt 교체 (기본 prof635_side_predvrel_ttc.pt; signed 옵션은 prof635_side_predvrel_signed.pt + DELTA_MODE=signed)
#   ACC_A2_MF5F_CKPT  5f a_2 ckpt 교체 (기본 a2_mf5f_acc.pt)
#   ACC_VREL_EMA      주입 closing 시간평활 (기본 0.6; ↑=부드럽게)
#   (gate 는 모델 내부 learned NN — rule/평활 env 옵션 없음)
#   DELTA_MODE        brake_only(기본)|signed (f_2 ckpt 학습값과 일치 필수)
#   F2_DELTA_SCALE    brake delta 증폭 (ACC 런처 기본 1.3, 1.0=불변; 강제동 1.2~1.3, >1.5 cruise 과제동)
set -e
SCC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_ACC=1
export F2_DELTA_SCALE="${F2_DELTA_SCALE:-1.3}"   # ACC 기본 강제동 1.3 (env 로 override 가능, 예: F2_DELTA_SCALE=1.0)
exec bash "$SCC_ROOT/scripts/run_closed_loop_prof635.sh" "$@"
