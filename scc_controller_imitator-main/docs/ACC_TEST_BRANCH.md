# ACC test branch (`acc-test`)

vrel(상대속도) 기반 제동 ACC 버전 closed-loop wiring. 예선 전 라이브 테스트용 별도 branch.
main / qualifier branch 는 미수정. 기존 distbrake 경로는 `MODEL_ACC` 미설정 시 그대로(회귀 없음).

## 추론 경로 (ACC ON)

```
카메라 → seg backbone(fx640_v3) tokenizer → curr i1 token (40,256)
  ├─[기존 유지] v7b 2-frame a_2 → i2_mem(40,128) + a2_logits(40,7)  → f_2 KV(cross-attn) + gate
  └─[ACC 신규] 5-frame A2ModelMultiFrame(offsets 10,20,30,40,50 tick) → in-lane(16-25) lead 의
                predicted vL_rel → closing(= -vL_rel) scalar → f_2 에 gt_vrel 로 주입(side_vrel_embed)
f_2 = prof635_side_predvrel_ttc (brake_only) → pedal delta → gate(EMA) → final pedal
```

학습 정합: predvrel f_2 ckpt 는 `i2_mem/a2_logits = v7b 2-frame cache` + `gt_vrel = 5f closing cache`
(`build_pred_vrel_cache.py`) 로 학습됨. 따라서 라이브도 i2_mem/gate 는 v7b 유지, vrel scalar 만 5f 로
산출해 주입한다. (i2_mem 자체를 5f 로 바꾸지 않음 — 학습 분포와 어긋나면 SCC/제동 붕괴, 2026-05-25 lesson.)

타이밍: i1 cache 1 tick = 10ms (offset 10..50 = 100..500ms). 라이브는 inference-tick 이 아니라
wall-clock 시간으로 t_now - off*10ms 에 가장 가까운 토큰을 버퍼에서 fetch (오프라인 검증 step3 에서
cache 의 tick 기반 선택과 동일 frame 선택 확인).

## ckpt (data/ckpts/, git -f 추적)

| 파일 | 원본 | 설명 |
|---|---|---|
| `a2_mf5f_acc.pt` | `outputs_multi/a2/mf_sweep_5f_10_20_30_40_50/best_loss.pt` | 5f A2ModelMultiFrame, offsets[10,20,30,40,50], d_out128, 7ch. vL_rel corr 0.84 |
| `prof635_side_predvrel_ttc.pt` | `outputs_multi/prof635_side_predvrel_ttc/best.pt` | f_2 side, predvrel 주입 + ttc ACC 라벨, brake_only. val_pedal 0.5118 |
| `prof635_side_predvrel_signed.pt` | `outputs_multi/prof635_side_predvrel_signed/best.pt` | (옵션) signed delta. ACC modulation 약했음 — 기본 비권장 |

기존 공유 ckpt: `main_prof654.pth`(frozen main), `a2_v7b_d128_alpha075_best_loss.pt`(i2_mem/gate),
`seg_pretrain_fx640_v3_backbone.pt`(tokenizer) 는 그대로 사용.

## 라이브 테스트 명령

```bash
# ACC ON (기본 sangam mild)
MODEL_ACC=1 bash scripts/run_closed_loop_prof635.sh sangam mild
# 또는 동등한 전용 런처
bash scripts/run_closed_loop_acc.sh sangam mild
bash scripts/run_closed_loop_acc.sh speedway high --viz-a2     # a_2 overlay 보며

# (회귀 확인용) ACC OFF = 기존 distbrake
bash scripts/run_closed_loop_prof635.sh sangam mild
```

env:
- `ACC_VREL_EMA=0.6` 주입 closing 시간평활(↑=부드럽게, flicker 억제)
- `DELTA_MODE=brake_only`(기본). signed ckpt 쓰면 `ACC_F2_CKPT=.../prof635_side_predvrel_signed.pt DELTA_MODE=signed`
- `GATE_SMOOTH=ema|distill|beat` gate 평활(현행 그대로)

## 관전 포인트

로그: `[prof635+ACC] ... obj_amax=.. gate=.. vrel=+X.Xm/s -> pedal=.. (acc=.. brk=..)`

- 앞차 **접근(closing>0, 거리 감소)** 시: `vrel` 양수 증가 → `delta` 음(제동) 강화 → `brk` escalate.
- 앞차 **이격/정속(closing≤0)** 시: `vrel` 0 부근/음수 → 제동 약함/무제동, 가속 유지.
- 앞차 없음(`obj_amax < 0.3`, gate≈0): vrel≈0, base pedal 그대로(깨끗한 가속/출발).
- steer 는 frozen main(#654) 그대로 — ACC 가 종방향만 건드림, 횡방향 거동 변화 없어야 정상.

## 오프라인 검증 (PASS)

`scripts/test_acc_closedloop.py` (GPU/CPU 양쪽 PASS, exit 0):
- step1: 라이브 5f forward vs `pred_vrel` cache closing — max|err| 0.000(GPU)/0.007(CPU) m/s < 0.5 tol. 5f 추론 정합 확인.
- step2: f_2 closing sweep(-8→+10 m/s) — finite·범위 OK, brake 0.15→0.62 monotonic escalate, delta vrel 반응 확인, steer 안정.
- step3(별도 스니펫): 라이브 wall-clock 토큰 fetch == cache tick 기반 offset 선택(동일 frame).

```bash
PYTHONPATH=.:./external python scripts/test_acc_closedloop.py            # GPU
CUDA_VISIBLE_DEVICES="" PYTHONPATH=.:./external python scripts/test_acc_closedloop.py   # CPU
```

## 알려진 한계

- **강접근 escalation 약함**: predicted vL_rel corr 0.84 라 강하게 다가오는 차에서 제동 증가폭이
  포화된다(step2: closing +2~+10 구간 delta -1.14→-1.17 로 거의 평탄). 약접근/정속 구분은 되지만
  급제동이 필요한 강접근에서 부족할 수 있음 — 밤사이 eval 과 일치.
- **i2_mem/gate 는 여전히 v7b 2-frame** perception. 앞차 인지(obj)/거리(gate) 품질은 v7b 수준
  (DriverB obj rec 63%). 5f 는 vrel scalar 만 보강.
- live frame rate 가 20Hz 미만으로 떨어지면 offset 토큰이 시간상 멀어져(가장 가까운 것 fallback)
  vrel 정합 저하 가능. 20Hz 권장.
- signed 버전은 ACC modulation 이 약해 brake_only 를 기본으로 둠.
