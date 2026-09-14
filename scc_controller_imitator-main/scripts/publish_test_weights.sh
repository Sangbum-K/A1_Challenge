#!/usr/bin/env bash
# SCC profile 학습 결과(side ckpt)를 테스트용으로 data/ckpts 에 publish + push.
#
# 배경 (2026-05-27):
#   - SCC profile 학습(train_scc_profiles_g2.py)은 outputs_multi/prof635_side_scc_p{N}_g2/best.pt
#     (main654 frozen + f_2 side + gate_head(obj+vrel 2sigmoid) 전체)을 저장.
#   - closed_loop prof635 경로 gate = 내부 learned gate_head → SCC gate 와 동일 구조.
#     따라서 분리경로(PROF635_CKPT=side + MAIN_CKPT=654/656)로 즉시 테스트 가능 (A1E2EModel 불필요).
#   - multicar prefix=scc_p{N}_f654|f656 (run_multicar.sh:emit_model_env) 로 선택.
#   - e2e 단일 .pt(A1E2EModel-scc, 규정 §1.1 본선 제출)는 closed_loop wire 가 별도로 필요 →
#     본 스크립트는 테스트용 분리경로 ckpt 만 publish. (본선 e2e 통합은 별도 트랙)
#
# 사용:
#   bash scripts/publish_test_weights.sh p3              # 단일 profile
#   bash scripts/publish_test_weights.sh p3 p5 p2 p4 p1  # 일괄
#   PUBLISH_NO_PUSH=1 bash scripts/publish_test_weights.sh p3   # 복사+commit 만, push 안 함
#   (chain hook: train_scc_profiles_g2.py 가 profile 완료마다 이 스크립트를 1 profile 로 호출)
set -e
SCC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCC_ROOT"
[[ $# -ge 1 ]] || { echo "usage: $0 <profile...>  (예: p3 또는 p3 p5 p2 p4 p1)" >&2; exit 1; }

published=()
for p in "$@"; do
    src="outputs_multi/prof635_side_scc_${p}_g2/best.pt"
    if [[ ! -f "$src" ]]; then
        echo "[skip] $src 없음 (학습 미완료?)" >&2
        continue
    fi
    dst="data/ckpts/scc_${p}_side.pt"
    cp -f "$src" "$dst"
    git add -f "$dst"
    # 학습 best val 기록 (있으면)
    best=$(grep -oE "saved best \([0-9.]+\)" "outputs_multi/prof635_side_scc_${p}_g2/train.log" 2>/dev/null | tail -1)
    echo "[publish] $dst  ($best)"
    published+=("$p")
done

if [[ ${#published[@]} -eq 0 ]]; then
    echo "[publish] 게시할 profile 없음 — 종료" >&2; exit 0
fi

# f656 prefix 가 쓰는 main_prof656.pth 미tracked 면 함께 추가 (1회).
if ! git ls-files --error-unmatch data/ckpts/main_prof656.pth >/dev/null 2>&1; then
    [[ -f data/ckpts/main_prof656.pth ]] && { git add -f data/ckpts/main_prof656.pth; echo "[publish] + main_prof656.pth (f656 prefix 용, 최초 tracked)"; }
fi

git commit -q -m "ckpt(test): SCC profile side weight publish (${published[*]}) — multicar scc_p{N}_f654/f656

분리경로 테스트용 (closed_loop prof635 + 내부 gate_head). e2e 단일(§1.1)은 별도.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>" || { echo "[publish] commit 변경 없음"; exit 0; }
echo "[publish] committed: ${published[*]}"

if [[ "${PUBLISH_NO_PUSH:-0}" == "1" ]]; then
    echo "[publish] PUBLISH_NO_PUSH=1 → push 생략"
else
    git push origin acc-test 2>&1 | tail -3
    echo "[publish] pushed: scc_{${published[*]}}_side.pt → origin/acc-test"
fi
