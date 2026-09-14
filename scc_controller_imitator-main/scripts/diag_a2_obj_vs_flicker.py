#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diag_a2_obj_vs_flicker.py — "obj 인지는 좋은데 왜 gate 가 깜빡이나" 모순 규명 (read-only)

배경 모순
---------
prof635 closed-loop gate = sigmoid((obj_amax - 0.3)*5),
  obj_amax = a2_logits[0, 16:26, 0].max()   (in-lane bin16~25, ch0=obj logit)
- 과거 분석: no-lead obj_amax ≈ -0.46, lead ≈ +1.41 로 분리, per-frame FP 거의 없음 → "obj 인지 good".
- 그런데 closed-loop "잘됬다 안됬다" + gate flip@0.3 ≈ 14.9% → 검출이 좋고 잘 분리됐다면
  0.3 근처에서 떨릴 이유가 없음 → 모순.

이 스크립트
-----------
배포 a2_v7b cached a2_logits 로:
 1. per-frame obj 품질(prec/recall/FP@thr0.3) vs temporal flip@0.3 / Δobj 를 동시 산출 → 모순 정량화.
 2. flip stratify: (a) GT 거리 band, (b) in-lane 경계(lead 가 bin16/25 가장자리 vs 중앙),
    (c) obj_amax 분포 — near-threshold [0.1,0.5] 띠 질량.
 3. obj_amax 분포 bimodal+분리 vs 0.3 근처 무거운 질량 검정.
 4. clear lead(근접<15m, in-lane 중앙) 에서 flip≈0 인가?
 5. in-lane window(bin16~25) 인공물: lead 의 argmax bin 위치가 16/25 경계인지, flip 이 경계와 상관되나.

데이터
------
{D}/cache/i2_mem_cache_{ds}_v7b_prev20.npz : a2_logits(N,40,7) f16, keys(N,)
{D}/cache/inlane_gt_{ds}.npz               : keys(N,), inlane_dist(N,), inlane_closing(N,)
keys = "{run}__t{ts}", ts step=5(=0.05s @20Hz).
a2_logits[:, b, 0]=obj logit, [:, b, 6]=dist(×40). bin16~25 = in-lane center 10 bins.
"""
import os
from collections import defaultdict
import numpy as np

D = os.environ.get("SCC_DATA_DIR",
                   "/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data")
CACHE = os.path.join(D, "cache")

BIN_LO, BIN_HI = 16, 25        # inclusive, in-lane center 10 bins
OBJ_CH, DIST_CH = 0, 6
DIST_SCALE = 40.0
THR = 0.3                      # 배포 gate threshold (obj_amax 기준)
LOGIT_OBJ_THR = 0.0           # per-bin obj 존재 판정 (sigmoid>0.5)
GT_DIST_MAX = 60.0            # in-lane lead 존재 cutoff
NEAR_BAND = (0.1, 0.5)        # near-threshold 띠
CLEAR_NEAR_M = 15.0          # clear lead = 근접 < 15m

# lead 데이터 / no-lead 데이터
LEAD_DS = ["sangam_lead_0520", "sangam_lead_0521", "DriverB_new"]
NOLEAD_DS = ["sangam_pertX_v2"]
ALL_DS = LEAD_DS + NOLEAD_DS


def load(ds):
    za = np.load(os.path.join(CACHE, f"i2_mem_cache_{ds}_v7b_prev20.npz"), allow_pickle=True)
    a2 = za["a2_logits"].astype(np.float32)        # (N,40,7)
    ka = [str(x) for x in za["keys"]]
    obj_in = a2[:, BIN_LO:BIN_HI + 1, OBJ_CH]       # (N,10) in-lane obj logit
    dist_in = a2[:, BIN_LO:BIN_HI + 1, DIST_CH] * DIST_SCALE  # (N,10) per-bin dist
    zg = np.load(os.path.join(CACHE, f"inlane_gt_{ds}.npz"), allow_pickle=True)
    gt = {str(k): float(d) for k, d in zip(zg["keys"], zg["inlane_dist"])}
    runs = defaultdict(list)
    for i, k in enumerate(ka):
        run, ts = k.rsplit("__t", 1)
        runs[run].append((int(ts), i))
    seqs = []
    for run, items in runs.items():
        items.sort()
        idx = [i for _, i in items]
        keys = [ka[i] for i in idx]
        gdist = np.array([gt.get(k, np.inf) for k in keys], dtype=np.float32)
        seqs.append(dict(run=run, obj=obj_in[idx], dist=dist_in[idx],
                         gdist=gdist, has=np.isfinite(gdist) & (gdist < GT_DIST_MAX)))
    return seqs


def main():
    print(f"[diag_a2_obj_vs_flicker] D={D}\n")
    data = {ds: load(ds) for ds in ALL_DS}
    for ds in ALL_DS:
        n = sum(len(s["obj"]) for s in data[ds])
        nl = sum(int(s["has"].sum()) for s in data[ds])
        print(f"  {ds:18s}: {len(data[ds]):4d} runs  {n:7d} fr  GT-lead={nl:7d} ({100*nl/max(n,1):4.1f}%)")

    lead = sum((data[d] for d in LEAD_DS), [])
    nolead = sum((data[d] for d in NOLEAD_DS), [])

    # ============ 1. per-frame obj 품질 vs temporal flip (모순 정량화) =========
    # per-frame: obj_amax = max in-lane obj logit; gate on = obj_amax > THR
    # GT positive = has (GT in-lane lead). FP = on & ~has. recall = on & has.
    def collect(seqs):
        amax = np.concatenate([s["obj"].max(1) for s in seqs])
        has = np.concatenate([s["has"] for s in seqs])
        return amax, has
    amax_l, has_l = collect(lead)
    amax_n, has_n = collect(nolead)
    amax_all = np.concatenate([amax_l, amax_n])
    has_all = np.concatenate([has_l, has_n])
    on_all = amax_all > THR
    tp = int(np.sum(on_all & has_all)); fp = int(np.sum(on_all & ~has_all))
    fn = int(np.sum(~on_all & has_all)); tn = int(np.sum(~on_all & ~has_all))
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    print("\n=== [1] per-frame obj 품질 @thr0.3 (lead+nolead 통합) ===")
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  precision={prec:.3f}  recall={rec:.3f}  FP-rate(nolead)={fp/max(fp+tn,1):.3f}")
    print(f"  obj_amax  GT-lead: mean={amax_all[has_all].mean():+.3f} med={np.median(amax_all[has_all]):+.3f} "
          f"std={amax_all[has_all].std():.3f}")
    print(f"  obj_amax  no-lead: mean={amax_all[~has_all].mean():+.3f} med={np.median(amax_all[~has_all]):+.3f} "
          f"std={amax_all[~has_all].std():.3f}")

    # temporal flip@0.3 + Δobj  (lead+mixed seqs; run 단위 연속)
    flips = denom = 0
    dobj_sum = 0.0; dobj_n = 0
    dobj_list = []
    for s in lead:
        a = s["obj"].max(1); on = a > THR
        if len(a) >= 2:
            flips += int(np.sum(on[1:] != on[:-1])); denom += len(a) - 1
            da = np.abs(np.diff(a)); dobj_sum += da.sum(); dobj_n += len(da)
            dobj_list.append(da)
    flip_pct = 100 * flips / max(denom, 1)
    dobj_mean = dobj_sum / max(dobj_n, 1)
    print(f"\n  temporal flip@0.3 = {flip_pct:.2f}%   (frame-to-frame gate toggle, lead seqs)")
    print(f"  mean |Δobj_amax|  = {dobj_mean:.3f}")
    print("  => 모순: per-frame 분리 양호한데 flip 이 높으면 '검출 good vs 시간 불안정' 모순 확인.")

    # ============ 2+3. obj_amax 분포 — near-threshold 띠 질량, bimodality ======
    print("\n=== [3] obj_amax 분포: near-threshold 띠 질량 (bimodal+분리 vs 0.3 근처 mass) ===")
    for name, a, h in [("GT-lead", amax_all[has_all], None), ("no-lead", amax_all[~has_all], None)]:
        near = np.mean((a >= NEAR_BAND[0]) & (a <= NEAR_BAND[1]))
        below = np.mean(a < NEAR_BAND[0]); above = np.mean(a > NEAR_BAND[1])
        print(f"  {name:8s}: <{NEAR_BAND[0]}={100*below:5.1f}%  "
              f"[{NEAR_BAND[0]},{NEAR_BAND[1]}](near-THR)={100*near:5.1f}%  "
              f">{NEAR_BAND[1]}={100*above:5.1f}%")
    # histogram (전체 + lead)
    edges = np.array([-3, -2, -1, -0.5, 0, 0.1, 0.3, 0.5, 1, 2, 3, 10])
    print("  obj_amax histogram (GT-lead frames):")
    hh, _ = np.histogram(amax_all[has_all], bins=edges)
    for j in range(len(hh)):
        print(f"    [{edges[j]:+5.1f},{edges[j+1]:+5.1f}): {100*hh[j]/max(hh.sum(),1):5.1f}%")

    # ============ 2a. flip stratify by GT distance band =======================
    print("\n=== [2a] flip stratify by GT distance band (전이 프레임에서 거리) ===")
    bands = [("near <15m", 0, 15), ("mid 15-40m", 15, 40), ("far 40-60m", 40, 60)]
    band_flip = {b[0]: [0, 0] for b in bands}   # [flips, denom]
    for s in lead:
        a = s["obj"].max(1); on = a > THR; gd = s["gdist"]
        for t in range(1, len(a)):
            # 이 transition 의 거리 = 두 프레임 중 in-lane 인 프레임의 GT 거리(있으면)
            d_ref = gd[t] if np.isfinite(gd[t]) else gd[t - 1]
            if not np.isfinite(d_ref):
                continue
            for bn, lo, hi in bands:
                if lo <= d_ref < hi:
                    band_flip[bn][1] += 1
                    if on[t] != on[t - 1]:
                        band_flip[bn][0] += 1
                    break
    for bn, lo, hi in bands:
        f, dn = band_flip[bn]
        print(f"  {bn:12s}: flip={100*f/max(dn,1):5.2f}%  (n_trans={dn})")

    # ============ 4. clear lead (근접<15m & in-lane 중앙) flip≈0? ==============
    # in-lane 중앙 = lead 의 argmax bin 이 중앙 6bin (글로벌 18~23) 안
    print("\n=== [4] clear-lead (GT<15m & lead argmax bin 중앙) 에서 flip / on-rate ===")
    cl_flip = cl_denom = 0; cl_on = cl_tot = 0
    for s in lead:
        a = s["obj"].max(1); on = a > THR; gd = s["gdist"]
        argb = s["obj"].argmax(1)        # 0..9 (= global bin16..25)
        # 중앙: local idx 2..7 (global 18..23)
        clear = (gd < CLEAR_NEAR_M) & np.isfinite(gd) & (argb >= 2) & (argb <= 7)
        cl_on += int(np.sum(on & clear)); cl_tot += int(np.sum(clear))
        for t in range(1, len(a)):
            if clear[t] and clear[t - 1]:
                cl_denom += 1
                if on[t] != on[t - 1]:
                    cl_flip += 1
    print(f"  clear-lead frames = {cl_tot}  (on-rate={100*cl_on/max(cl_tot,1):.2f}%)")
    print(f"  clear-lead flip   = {100*cl_flip/max(cl_denom,1):.2f}%  (n_trans={cl_denom})")
    print("  => flip≈0 면 검출 정상, flicker 는 원거리/경계뿐 (사용자 기억 '검출 good' 정합).")

    # ============ 5. in-lane window 인공물: lead argmax bin 경계 상관 ==========
    print("\n=== [5] in-lane window artifact: lead argmax bin 위치 분포 + 경계 flip ===")
    # GT lead 가 있는 프레임에서 in-lane argmax bin (local 0..9) 분포
    edge_local = {0, 1, 8, 9}   # bin16,17,24,25 = window 가장자리
    bin_hist = np.zeros(10)
    edge_flip = edge_denom = 0; ctr_flip = ctr_denom = 0
    for s in lead:
        a = s["obj"].max(1); on = a > THR
        argb = s["obj"].argmax(1); has = s["has"]
        for t in range(len(a)):
            if has[t]:
                bin_hist[argb[t]] += 1
        for t in range(1, len(a)):
            if not (has[t] or has[t - 1]):
                continue
            at_edge = (argb[t] in edge_local) or (argb[t - 1] in edge_local)
            if at_edge:
                edge_denom += 1
                if on[t] != on[t - 1]:
                    edge_flip += 1
            else:
                ctr_denom += 1
                if on[t] != on[t - 1]:
                    ctr_flip += 1
    tot = bin_hist.sum()
    print("  GT-lead argmax bin 분포 (local 0..9 = global 16..25; 0/1/8/9 = 경계):")
    print("   " + "  ".join(f"b{16+j}:{100*bin_hist[j]/max(tot,1):4.1f}%" for j in range(10)))
    print(f"  flip @ window 경계(argmax∈16,17,24,25) = {100*edge_flip/max(edge_denom,1):5.2f}% (n={edge_denom})")
    print(f"  flip @ window 중앙(argmax∈18..23)       = {100*ctr_flip/max(ctr_denom,1):5.2f}% (n={ctr_denom})")
    print("  => 경계 flip >> 중앙 flip 이면 in-lane 10-bin window 들락거림(geometry artifact) 이 원인.")


if __name__ == "__main__":
    main()
