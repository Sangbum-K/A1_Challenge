#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_gate_stability.py — SCC lead-brake gate 안정성 vs 응답성 오프라인 비교 (read-only)

목적
----
closed-loop 의 lead-brake gate 깜빡임("깔짝깔짝")을 줄이는 여러 stateless/stateful
aggregation·shaping·temporal 변형을, 캐시된 a_2 logit(v7b) 시계열에 적용해 정량 비교한다.
어떤 설정이 flip(게이트 토글)·jerk 를 가장 줄이면서 onset 제동 지연과 no-lead FP 를
현행 배포(amax/S5/EMA0.7) 대비 나빠지지 않게 하는지 추천한다.

배포 코드와 격리 (unimport)
---------------------------
closed_loop_node.py 의 게이트 로직을 여기서 독립 재구현한다. 배포 코드 import/수정 없음.
배포 로직(closed_loop_node.py L622-628) 등가:
    obj_amax = a2_logits[0, 16:26, 0].max()
    g_raw    = sigmoid((obj_amax - 0.3) * 5.0)
    gate     = EMA*prev + (1-EMA)*g_raw     # EMA=0.7
    (게이트 활성 판정 = gate > 0.5)

데이터
------
{D}/cache/i2_mem_cache_{ds}_v7b_prev20.npz : a2_logits(N,40,7), keys(N,)
{D}/cache/inlane_gt_{ds}.npz               : keys(N,), inlane_dist(N,) [유한&<60 = in-lane lead 존재]
keys 포맷 = "{run}__t{ts}", ts step=5(=0.05s @ 20Hz), run 내 gap 없음 → 시계열 복원 가능.

사용
----
  python scripts/exp_gate_stability.py
  SCC_DATA_DIR=... python scripts/exp_gate_stability.py
"""
import os
import sys
import math
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------- 설정
D = os.environ.get("SCC_DATA_DIR",
                   "/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data")
CACHE = os.path.join(D, "cache")

INLANE_BINS = slice(16, 26)   # in-lane bin 16~25
OBJ_CH = 0                    # ch0 = obj logit
GT_DIST_MAX = 60.0            # 유한 & <60 m = in-lane lead 존재
FPS = 20.0                    # 20 Hz → step=5 = 0.05 s
GATE_ON = 0.5                 # 게이트 on 판정 임계

# onset 정의: GT lead 가 없음->있음(dist<60 진입)으로 전이한 프레임부터
# gate>0.5 까지 걸린 프레임 수. (smoothing 의 대가 = 응답성)

DATASETS = {
    "sangam_lead_0520": "lead",   # 거의 전구간 lead
    "sangam_lead_0521": "lead",   # lead + 일부 no-lead, onset 多
    "DriverB_new":      "mixed",  # lead/no-lead 풍부, onset 558 — onset latency 주력
    "sangam_pertX_v2":  "nolead", # in-lane lead 0 — FP 측정
}

# ---------------------------------------------------------------- 변형 정의
# 각 변형 = (label, agg_fn, shape_fn, ema)
#   agg_fn  : (in_lane_obj_logits[10]) -> scalar  (in-lane reduce)
#   shape_fn: scalar -> g_raw in [0,1]
#   ema     : 0=stateless, >0=IIR  gate = ema*prev + (1-ema)*g_raw


def agg_amax(x):
    return float(np.max(x))


def agg_mean(x):
    return float(np.mean(x))


def agg_topk3(x):
    s = np.sort(x)[::-1]
    return float(np.mean(s[:3]))


def agg_logsumexp(x, T=1.0):
    # T*logsumexp(x/T) — smooth-max; T=1
    m = np.max(x)
    return float(T * (m / T + math.log(np.sum(np.exp((x - m) / T)))))


def make_sigmoid(scale, thresh=0.3):
    def f(v):
        return 1.0 / (1.0 + math.exp(-(v - thresh) * scale))
    return f


def make_softramp(width=0.8, thresh=0.0):
    # clip((v - thresh)/width, 0, 1) — 선형 ramp
    def f(v):
        return float(min(1.0, max(0.0, (v - thresh) / width)))
    return f


# ---- aggregation 별 출력 스케일이 달라 sigmoid THRESH 0.3 이 mean/logsumexp 에는
#      부적절(mean=항상 off, logsumexp=항상 on). 공정 비교 위해 각 agg 분포 중점으로
#      재보정한 THRESH 를 별도 사용 (DriverB_new GT 기준 no-lead/lead median 중점):
#        amax: nl=-0.32 lead=+0.59 -> mid 0.13 (배포 0.3 은 lead 쪽으로 보수적)
#        mean: nl=-1.08 lead=-0.65 -> mid -0.85
#        logsumexp: nl=+1.37 lead=+1.97 -> mid +1.67
TH_AMAX = 0.3       # 배포값 유지 (분리 양호)
TH_MEAN = -0.85
TH_LSE = 1.67

# 대표 조합 (전수 아님). agg / shape / ema
VARIANTS = [
    # label,                       agg,            shape,                       ema
    ("amax / S5 / EMA0  (baseline-noEMA)", agg_amax,      make_sigmoid(5.0, TH_AMAX), 0.0),
    ("amax / S5 / EMA0.7 (DEPLOYED)",      agg_amax,      make_sigmoid(5.0, TH_AMAX), 0.7),
    ("amax / S5 / EMA0.3",                 agg_amax,      make_sigmoid(5.0, TH_AMAX), 0.3),
    ("amax / S5 / EMA0.5",                 agg_amax,      make_sigmoid(5.0, TH_AMAX), 0.5),
    ("amax / S3 / EMA0",                   agg_amax,      make_sigmoid(3.0, TH_AMAX), 0.0),
    ("amax / S2 / EMA0",                   agg_amax,      make_sigmoid(2.0, TH_AMAX), 0.0),
    ("mean / S5 / EMA0 (TH-cal)",          agg_mean,      make_sigmoid(5.0, TH_MEAN), 0.0),
    ("mean / S3 / EMA0.5 (TH-cal)",        agg_mean,      make_sigmoid(3.0, TH_MEAN), 0.5),
    ("topk3 / S5 / EMA0",                  agg_topk3,     make_sigmoid(5.0, TH_AMAX), 0.0),
    ("topk3 / S5 / EMA0.5",                agg_topk3,     make_sigmoid(5.0, TH_AMAX), 0.5),
    ("topk3 / S3 / EMA0",                  agg_topk3,     make_sigmoid(3.0, TH_AMAX), 0.0),
    ("logsumexp / S3 / EMA0 (TH-cal)",     agg_logsumexp, make_sigmoid(3.0, TH_LSE), 0.0),
    ("logsumexp / S3 / EMA0.5 (TH-cal)",   agg_logsumexp, make_sigmoid(3.0, TH_LSE), 0.5),
    ("softramp(w0.8) / EMA0",              agg_amax,      make_softramp(0.8, 0.0), 0.0),
]


# ---------------------------------------------------------------- 시계열 복원
def load_dataset(ds):
    """run 단위로 정렬된 시퀀스 리스트 반환.
    각 시퀀스 = dict(run, obj_inlane(T,10), has_lead(T,) bool, dist(T,))"""
    za = np.load(os.path.join(CACHE, f"i2_mem_cache_{ds}_v7b_prev20.npz"),
                 allow_pickle=True)
    a2 = za["a2_logits"].astype(np.float32)         # (N,40,7)
    ka = [str(x) for x in za["keys"]]
    obj_inlane = a2[:, INLANE_BINS, OBJ_CH]          # (N,10)

    zg = np.load(os.path.join(CACHE, f"inlane_gt_{ds}.npz"), allow_pickle=True)
    kg = [str(x) for x in zg["keys"]]
    dist_g = np.asarray(zg["inlane_dist"], dtype=np.float32)
    gt_dist = {k: d for k, d in zip(kg, dist_g)}

    runs = defaultdict(list)
    for i, key in enumerate(ka):
        run, ts = key.split("__t")
        runs[run].append((int(ts), i, key))

    seqs = []
    for run, items in runs.items():
        items.sort()
        idx = [i for _, i, _ in items]
        keys = [k for _, _, k in items]
        obj = obj_inlane[idx]                        # (T,10)
        dist = np.array([gt_dist.get(k, np.inf) for k in keys], dtype=np.float32)
        has = np.isfinite(dist) & (dist < GT_DIST_MAX)
        seqs.append(dict(run=run, obj=obj, has_lead=has, dist=dist))
    return seqs


# ---------------------------------------------------------------- gate 시뮬
def run_gate(seqs, agg, shape, ema):
    """변형 1개를 모든 시퀀스에 적용 → 전 프레임 gate 배열 + per-seq 분해 반환."""
    gates = []          # per-seq gate arrays
    for s in seqs:
        obj = s["obj"]                               # (T,10)
        T = obj.shape[0]
        g = np.empty(T, dtype=np.float64)
        prev = 0.0
        for t in range(T):
            v = agg(obj[t])
            graw = shape(v)
            gv = ema * prev + (1.0 - ema) * graw if ema > 0.0 else graw
            prev = gv
            g[t] = gv
        gates.append(g)
    return gates


# ---------------------------------------------------------------- 지표
def metrics(seqs, gates):
    """flip%, jerk, onset latency(frames), no-lead FP%, lead recall%."""
    flips = 0
    flip_denom = 0
    jerk_sum = 0.0
    jerk_n = 0
    onset_lats = []        # frames
    fp_on = 0
    fp_tot = 0
    rec_on = 0
    rec_tot = 0

    for s, g in zip(seqs, gates):
        on = g > GATE_ON
        T = len(g)
        # flip: 연속 프레임에서 on 상태 교차
        if T >= 2:
            flips += int(np.sum(on[1:] != on[:-1]))
            flip_denom += (T - 1)
            jerk_sum += float(np.sum(np.abs(np.diff(g))))
            jerk_n += (T - 1)
        has = s["has_lead"]
        # onset latency: no-lead->lead 전이점부터 gate on 까지
        for t in range(1, T):
            if has[t] and not has[t - 1]:
                # lead 가 유지되는 구간 안에서 첫 on 까지
                lat = None
                for u in range(t, T):
                    if not has[u]:
                        break               # lead 가 사라지면 이 onset 종료(미검출)
                    if on[u]:
                        lat = u - t
                        break
                if lat is not None:
                    onset_lats.append(lat)
                # 미검출 onset 은 latency 표본에서 제외(별도 recall 로 반영)
        # FP: no-lead 프레임에서 on
        nl = ~has
        fp_on += int(np.sum(on & nl))
        fp_tot += int(np.sum(nl))
        # recall: lead 프레임에서 on
        rec_on += int(np.sum(on & has))
        rec_tot += int(np.sum(has))

    flip_pct = 100.0 * flips / flip_denom if flip_denom else float("nan")
    jerk = jerk_sum / jerk_n if jerk_n else float("nan")
    onset_frames = float(np.median(onset_lats)) if onset_lats else float("nan")
    onset_mean = float(np.mean(onset_lats)) if onset_lats else float("nan")
    fp_pct = 100.0 * fp_on / fp_tot if fp_tot else float("nan")
    rec_pct = 100.0 * rec_on / rec_tot if rec_tot else float("nan")
    return dict(flip=flip_pct, jerk=jerk,
                onset_med=onset_frames, onset_mean=onset_mean,
                n_onset=len(onset_lats),
                fp=fp_pct, recall=rec_pct)


# ---------------------------------------------------------------- main
def main():
    print(f"[exp_gate_stability] D={D}")
    print(f"[exp_gate_stability] datasets={list(DATASETS)}  FPS={FPS}")

    # 데이터셋 그룹: stability/jerk/recall 은 lead+mixed, onset 은 mixed(+lead),
    # FP 는 no-lead 전용으로 분리 측정.
    data = {}
    for ds in DATASETS:
        data[ds] = load_dataset(ds)
        n = sum(len(s["obj"]) for s in data[ds])
        nrun = len(data[ds])
        nlead = sum(int(s["has_lead"].sum()) for s in data[ds])
        print(f"  loaded {ds:18s}: {nrun:4d} runs  {n:7d} frames  "
              f"lead={nlead:7d} ({100.0*nlead/max(n,1):4.1f}%)")

    lead_seqs = data["sangam_lead_0520"] + data["sangam_lead_0521"] + data["DriverB_new"]
    nolead_seqs = data["sangam_pertX_v2"]
    onset_seqs = data["DriverB_new"] + data["sangam_lead_0521"]   # onset 풍부

    # ------- 표 헤더
    hdr = (f"{'variant':38s} | {'flip%':>6s} | {'jerk':>7s} | "
           f"{'onset_f':>7s} | {'onset_s':>7s} | {'FP%':>6s} | {'recall%':>7s} | {'n_onset':>7s}")
    print("\n" + "=" * len(hdr))
    print("조합 × 지표  (flip/jerk/recall = lead+mixed,  FP = no-lead(pertX),  onset = DriverB+0521)")
    print("onset_f = median onset latency (frames),  onset_s = sec @ %.0fHz" % FPS)
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for label, agg, shape, ema in VARIANTS:
        g_lead = run_gate(lead_seqs, agg, shape, ema)
        m_lead = metrics(lead_seqs, g_lead)
        g_nl = run_gate(nolead_seqs, agg, shape, ema)
        m_nl = metrics(nolead_seqs, g_nl)
        g_on = run_gate(onset_seqs, agg, shape, ema)
        m_on = metrics(onset_seqs, g_on)

        flip = m_lead["flip"]
        jerk = m_lead["jerk"]
        recall = m_lead["recall"]
        fp = m_nl["fp"]
        onset_f = m_on["onset_med"]
        onset_s = onset_f / FPS if onset_f == onset_f else float("nan")
        n_onset = m_on["n_onset"]
        rows.append((label, flip, jerk, onset_f, onset_s, fp, recall, n_onset))
        print(f"{label:38s} | {flip:6.2f} | {jerk:7.4f} | "
              f"{onset_f:7.1f} | {onset_s:7.3f} | {fp:6.2f} | {recall:7.2f} | {n_onset:7d}")
    print("=" * len(hdr))

    # ------- 추천: deployed 대비 onset/FP 안 나빠지면서 flip/jerk 최소인 stateless
    dep = next(r for r in rows if "DEPLOYED" in r[0])
    _, dflip, djerk, donf, _, dfp, drec, _ = dep
    print(f"\n[현행 배포 baseline] flip={dflip:.2f}%  jerk={djerk:.4f}  "
          f"onset={donf:.1f}f  FP={dfp:.2f}%  recall={drec:.2f}%")
    print("[stateless(EMA0) 후보 중 deployed 대비 onset·FP 비악화 & flip 최소]")
    EPS_ON = 1.0      # onset 1 frame 까지 허용
    EPS_FP = 1.0      # FP 1%p 까지 허용
    EPS_REC = 5.0     # recall 5%p 까지 허용 (게이트가 lead 를 켜긴 해야 함)
    cands = []
    for r in rows:
        label, flip, jerk, onf, ons, fp, rec, no = r
        # stateless 판정: EMA0 (소수점 EMA 아님)
        is_stateless = ("EMA0" in label) and ("EMA0." not in label)
        if not is_stateless:
            continue
        ok = (onf <= donf + EPS_ON) and (fp <= dfp + EPS_FP) and (rec >= drec - EPS_REC)
        cands.append((flip, jerk, ok, r))
    cands.sort(key=lambda x: (not x[2], x[0], x[1]))
    for flip, jerk, ok, r in cands:
        label = r[0]
        mark = "OK " if ok else "   "
        print(f"  [{mark}] {label:38s} flip={flip:6.2f}%  jerk={jerk:.4f}  "
              f"onset={r[3]:.1f}f  FP={r[5]:.2f}%  recall={r[6]:.1f}%")


if __name__ == "__main__":
    main()
