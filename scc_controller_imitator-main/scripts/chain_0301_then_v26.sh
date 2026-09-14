#!/bin/bash
# 순차 chain: B4 phase1(71305) 종료 → v0301 main finetune → v26 min-ACC.
# 각 launcher 는 학습을 background 로 띄우고 pid 파일을 남기므로, 그 pid 를 기다린다.
cd /home/ailab-12/git/scc_controller_imitator
LOG=/tmp/chain_0301_v26.log
exec > >(tee -a "$LOG") 2>&1
echo "[chain] $(date) — B4 phase1(71305) 종료 대기..."
tail --pid=71305 -f /dev/null 2>/dev/null || true
sleep 8
echo "[chain] $(date) — v0301 main finetune launch"
bash scripts/launch_v25c2_main_v0301only.sh
sleep 15
P1=$(cat /tmp/v25c2_main_v0301.pid 2>/dev/null)
echo "[chain] v0301 finetune pid=$P1 대기..."
[ -n "$P1" ] && tail --pid="$P1" -f /dev/null 2>/dev/null || true
sleep 8
if [ ! -f outputs_multi/iy_v25c2_main_v0301only/best_loss.pt ]; then
  echo "[chain] ERR: v0301 best_loss.pt 없음 — v26 중단"; exit 1
fi
echo "[chain] $(date) — v26 min-ACC launch (resume v0301 main)"
bash scripts/launch_v26_Bminacc.sh
echo "[chain] $(date) — done (v26 학습 진행 중)"
