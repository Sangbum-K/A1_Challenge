#!/usr/bin/env bash
# 현재 진행 중인 SCC profile 학습 chain(train_scc_profiles_g2.py, hook 없이 가동된 것)의
# 각 profile 완료를 chain log 로 감지 → publish_test_weights.sh 자동 호출.
#   재시작 없이 "profile 끝날 때마다 자동 push" 를 달성 (학습 ep 손실 0).
#   ※ 미래 chain 은 train_scc_profiles_g2.py 내부 hook 으로 자동 publish → 이 watcher 불필요.
#
# 사용: nohup bash scripts/publish_watcher.sh > /tmp/publish_watcher.log 2>&1 &
#   기본 감시: p5 p2 p4 p1 (p3 는 이미 publish 됨). 인자로 변경 가능.
set -u
SCC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCC_ROOT"
LOG="${CHAIN_LOG:-/tmp/scc_chain_g2.log}"
PROFILES=("${@:-p5 p2 p4 p1}")
# 단일 문자열 인자("p5 p2 ..")로 와도 분해
PROFILES=($(echo "${PROFILES[@]}"))

echo "[watcher] start $(date '+%H:%M:%S')  log=$LOG  감시=${PROFILES[*]}"
for p in "${PROFILES[@]}"; do
    echo "[watcher] $p 완료 대기..."
    # chain log 에 'profile p{N} 완료 rc=0' 출현까지 대기 (chain 죽으면 무한대기 방지: wrapper 생존 체크)
    while ! grep -q "profile $p 완료 rc=0" "$LOG" 2>/dev/null; do
        if ! pgrep -f "train_scc_profiles_g2.py" >/dev/null 2>&1; then
            # wrapper 죽음 — 이미 완료됐는지 마지막 확인 후 종료
            grep -q "profile $p 완료 rc=0" "$LOG" 2>/dev/null && break
            echo "[watcher] chain wrapper 종료됨, $p 미완료 → watcher 중단" >&2
            exit 1
        fi
        sleep 30
    done
    echo "[watcher] $p 완료 감지 → publish $(date '+%H:%M:%S')"
    bash scripts/publish_test_weights.sh "$p" || echo "[watcher] $p publish 실패(rc=$?)" >&2
done
echo "[watcher] 전 profile publish 완료 $(date '+%H:%M:%S')"
