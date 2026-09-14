#!/bin/bash
# Phase 2 — a_2 (mf20 + DriverB 추가) 재학습 결과를 downstream 에 정합.
#   ※ a_2 개선이 measure 로 확인된 뒤에만 실행할 것 (Phase 1 = train_a2_n20_db.sh).
#
# 체인 (a_2 가 바뀌면 i2_mem → f_2/gate/scc → closed_loop 전부 재정합 필요):
#   Step 1 (이 스크립트): 새 a_2 best → data/ckpts/a2_mf20_v2s_n20_db.pt + 기존 i2_mem 백업.
#   Step 2 (이 스크립트): i2_mem_mf20 5개 재생성 (새 a_2, GPU). 기존 파일 덮어씀(백업 후).
#   Step 3 (수동): f_2/gate/scc 재학습 — 새 i2_mem 자동 사용(DATASETS 경로 동일).
#   Step 4 (수동): closed_loop MODEL_A2_V6_CKPT → a2_mf20_v2s_n20_db.pt + verify(cos>0.99).
#   Step 5 (수동): publish_test_weights.sh + push.
#
# 사용:  bash scripts/run_phase2_a2db_rebuild.sh           # dry-run (계획만)
#        bash scripts/run_phase2_a2db_rebuild.sh --apply   # Step 1+2 실제 실행
#        A2_NEW=<path> 로 a_2 best override (기본 mf20f_v2s_5to100_db/best_loss.pt)
set -e
ROOT=/home/ailab-12/git/scc_controller_imitator
PY=/home/ailab-12/miniforge3/envs/ml_con/bin/python
D=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data
cd "$ROOT"
APPLY=""; [[ "$1" == "--apply" ]] && APPLY=1
A2_NEW="${A2_NEW:-$ROOT/outputs_multi/a2_b0_abl/mf20f_v2s_5to100_db/best_loss.pt}"
A2_DST="$ROOT/data/ckpts/a2_mf20_v2s_n20_db.pt"
OFFS="5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100"
# i2_mem 정합 대상 5개 (closed_loop/f_2 입력. pertO/pertX 는 lead 0 이지만 cache 정합 위해 포함).
DSETS=(sangam_lead_0520 sangam_lead_0521 sangam_pertO_v2 sangam_pertX_v2 DriverB_new)

echo "[phase2] a_2 new = $A2_NEW"
[[ -f "$A2_NEW" ]] || { echo "[err] a_2 best 없음 (Phase1 미완?): $A2_NEW" >&2; exit 1; }
echo "[phase2] APPLY=${APPLY:-dry-run}"

# ---- Step 1: ckpt 복사 + 기존 i2_mem 백업 ----
echo "--- Step 1: a_2 ckpt 복사 + i2_mem 백업 ---"
echo "  cp $A2_NEW -> $A2_DST"
BAK="$D/cache/_bak_i2_pre_db_$(date +%Y%m%d_%H%M)"
for ds in "${DSETS[@]}"; do echo "  backup i2_mem_mf20_${ds}.npz -> $BAK/"; done
if [[ -n "$APPLY" ]]; then
  cp -f "$A2_NEW" "$A2_DST"
  mkdir -p "$BAK"
  for ds in "${DSETS[@]}"; do
    f="$D/cache/i2_mem_mf20_${ds}.npz"; [[ -f "$f" ]] && cp -f "$f" "$BAK/" || echo "  (없음: $f)"
  done
fi

# ---- Step 2: i2_mem_mf20 5개 재생성 (새 a_2) ----
echo "--- Step 2: i2_mem_mf20 재생성 (a_2=$A2_DST) ---"
for ds in "${DSETS[@]}"; do
  i1="$D/cache/i1_tok_cache_${ds}_prev20.npz"
  out="$D/cache/i2_mem_mf20_${ds}.npz"
  echo "  build $ds: $i1 -> $out"
  if [[ -n "$APPLY" ]]; then
    [[ -f "$i1" ]] || { echo "  [err] i1 cache 없음: $i1" >&2; exit 1; }
    "$PY" -u scripts/build_i2_cache_mf20.py \
      --i1-cache "$i1" --out "$out" --a2-ckpt "$A2_DST" \
      --d-token 256 --d-out 128 --prev-offsets "$OFFS" 2>&1 | tail -6
  fi
done

echo ""
echo "=== Step 1+2 ${APPLY:+완료}${APPLY:-(dry-run, --apply 로 실행)} ==="
cat <<'NEXT'
--- 이후 수동 (a_2 개선 확인 후) ---
Step 3  재학습 (새 i2_mem 자동 사용):
          bash   scripts/train_scc_profiles_g2.sh     # side+gate profile p1-5
          python scripts/train_scc_nogate.py          # no_gate (p3,p5 등)
Step 4  closed_loop a_2 교체 + 검증:
          # scripts/closed_loop_node.py 의 MODEL_A2_V6_CKPT 를 a2_mf20_v2s_n20_db.pt 로
          CUDA_VISIBLE_DEVICES="" python scripts/verify_i2_live_vs_cache.py --mf20 ...   # cos>0.99
Step 5  게시:
          bash scripts/publish_test_weights.sh ; git push origin acc-test
NEXT
