#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
learned_temporal_gate.py — EMA gate 평활을 "규정-clean learned temporal filter" 로 대체 검증.

배경 (exp_gate_stability.py 에서 확정)
--------------------------------------
배포 gate (closed_loop_node.py L622-628):
    obj_amax = a2_logits[0, 16:26, 0].max()          # in-lane bin16~25, ch0=obj logit, max
    g_raw    = sigmoid((obj_amax - 0.3) * 5.0)
    gate     = 0.7*gate_prev + 0.3*g_raw             # EMA — cross-frame python IIR state (§1.2/1.3 risk)
- 안정성 지배 변수 = temporal smoothing(EMA). amax/S5 에서 EMA0->0.7 시 flip 11.45->3.29%.
- spatial aggregation(mean/topk/logsumexp)은 lead 신호가 1-bin 국소라 실패 → 평활은 시간축이 정답.
- 결론: temporal smoothing 필요. 단 EMA python IIR state 는 규정 risk → learned in-graph 모듈로 대체.

핵심 설계 (규정 clean)
----------------------
입력 = 최근 K 프레임 in-lane obj 신호의 "명시적 window" (a2_logits 를 K개 버퍼링).
이는 매 프레임 이미 계산하는 a2_logits 를 단순 stack 한 것 — recurrent hidden state 가
아니라 multi-frame batch input 과 동일한 의미. forward = 단일 differentiable pass,
cross-call 누적 state 없음 → §1.3 만족 (EMA 의 python IIR state 와 대비).

모듈 후보 (가벼운 순)
---------------------
  1. LearnableIIR  : learnable alpha 1개로 window 위 EMA 를 graph 내 unroll (NN 이 EMA 재현 sanity).
  2. TemporalConv1d: K-tap depthwise causal conv over obj 시퀀스 → smoothed gate.
  3. TemporalMLP   : window flatten → tiny MLP → smoothed gate (적응형).

학습 트랙
---------
  A. distill  : target = EMA0.7 gate. learned 가 규정-clean EMA 등가물을 재현하나?
  B. beat-EMA : loss = flip(jerk) + onset 응답성 + FP/recall. 고정 IIR 이 못하는 적응형 학습.

데이터/평가 = exp_gate_stability.py 와 동일. run-level train/val split (leakage 방지).
배포 closed_loop_node.py 는 수정하지 않음 (별도 파일).

사용
----
  OMP_NUM_THREADS=2 python scripts/learned_temporal_gate.py --module mlp
  SCC_DATA_DIR=... python scripts/learned_temporal_gate.py --module conv

closed_loop_node.py 배선 스케치 (배포 파일은 수정하지 않음 — 참고용)
-------------------------------------------------------------------
  # __init__: 명시적 입력 window 버퍼 (이미 있는 deque 인프라 재사용, cross-call IIR state 아님)
  self._obj_win = deque(maxlen=K)                 # 최근 K 프레임의 in-lane obj (10,) 또는 amax
  ckpt = torch.load(GATE_CKPT, map_location="cpu")
  self._gate_net = make_module(ckpt["module"], ckpt["K"]); self._gate_net.load_state_dict(ckpt["state"]); self._gate_net.eval()

  # run_cycle (현행 L622-628 의 EMA 3줄을 아래로 교체):
  obj_inlane = a2_logits[0, 16:26, 0].float().cpu().numpy()   # (10,)
  self._obj_win.append(obj_inlane)
  win = np.stack(list(self._obj_win))                          # (<=K,10)
  if win.shape[0] < K:                                         # 좌측 zero-pad (학습과 동일 causal)
      win = np.concatenate([np.zeros((K-win.shape[0],10),np.float32), win], 0)
  with torch.no_grad():
      gate = float(self._gate_net(torch.from_numpy(win[None]).float())[0])
  # forward = 단일 differentiable pass. window 는 매 프레임 새로 stack → cross-call 누적 state 없음 → §1.3 clean.
"""
import os
import sys
import math
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.temporal_gate import LearnableIIR, TemporalConv1d, TemporalMLP, make_module  # noqa: E402

# ---- GPU 금지 (다른 학습 중). CPU 강제 + thread 제한 ----
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
DEVICE = "cpu"

# ---------------------------------------------------------------- 설정
D = os.environ.get("SCC_DATA_DIR",
                   "/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data")
CACHE = os.path.join(D, "cache")

INLANE_BINS = slice(16, 26)   # in-lane bin 16~25
OBJ_CH = 0
GT_DIST_MAX = 60.0
FPS = 20.0
GATE_ON = 0.5

# 배포 gate 파라미터
DEP_THRESH = 0.3
DEP_SCALE = 5.0
DEP_EMA = 0.7

K = 16                        # window 길이 (프레임). 0.8s @ 20Hz — EMA0.7 유효 메모리와 동등 수준
SEED = 0

DATASETS = {
    "sangam_lead_0520": "lead",
    "sangam_lead_0521": "lead",
    "DriverB_new":      "mixed",
    "sangam_pertX_v2":  "nolead",
}


# ---------------------------------------------------------------- 시계열 복원 (exp_gate_stability 재사용)
def load_dataset(ds):
    """run 단위 정렬 시퀀스 리스트. 각 시퀀스 = dict(run, obj(T,10), has_lead(T,) bool)
    메모리: i2_mem(1.3GB) 는 건드리지 않고 a2_logits 만 읽는다."""
    za = np.load(os.path.join(CACHE, os.environ.get("GATE_I2_PAT", "i2_mem_mf20_{ds}").format(ds=ds) + ".npz"),
                 allow_pickle=True)   # db=i2_mem_mf20(기본), 구버전=GATE_I2_PAT="i2_mem_cache_{ds}_v7b_prev20"
    a2 = za["a2_logits"]                              # (N,40,7) float16, lazy
    ka = [str(x) for x in za["keys"]]
    obj_inlane = np.asarray(a2[:, INLANE_BINS, OBJ_CH], dtype=np.float32)  # (N,10)
    del a2, za

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
        obj = obj_inlane[idx]
        dist = np.array([gt_dist.get(k, np.inf) for k in keys], dtype=np.float32)
        has = np.isfinite(dist) & (dist < GT_DIST_MAX)
        seqs.append(dict(run=run, ds=ds, obj=obj, has_lead=has, dist=dist))
    return seqs


# ---------------------------------------------------------------- windowed feature 빌드
def build_windows(seqs, K):
    """각 시퀀스 → 프레임별 (K,10) window (causal: 과거 K-1 + 현재, 좌측 zero-pad).
    반환 per-seq: X(T,K,10) float32, amax_seq(T,K) (window 의 per-frame amax), has(T,)
    이건 cross-call state 가 아니라 명시적 입력 window — closed_loop 에선 deque(maxlen=K) 로 버퍼."""
    out = []
    for s in seqs:
        obj = s["obj"]                               # (T,10)
        T = obj.shape[0]
        Xpad = np.concatenate([np.zeros((K - 1, 10), np.float32), obj], axis=0)  # (T+K-1,10)
        # sliding window view
        W = np.stack([Xpad[t:t + K] for t in range(T)], axis=0)                  # (T,K,10)
        out.append(dict(X=W, has=s["has_lead"], run=s["run"], ds=s["ds"]))
    return out


# ---------------------------------------------------------------- EMA reference gate (target/baseline)
def ema_gate_seq(obj_seq, ema, thresh=DEP_THRESH, scale=DEP_SCALE):
    """배포 EMA gate 를 시퀀스에 적용 (stateful). obj_seq (T,10) → gate (T,)."""
    T = obj_seq.shape[0]
    amax = obj_seq.max(axis=1)                       # (T,)
    g = np.empty(T, np.float64)
    prev = 0.0
    for t in range(T):
        graw = 1.0 / (1.0 + math.exp(-(amax[t] - thresh) * scale))
        gv = ema * prev + (1.0 - ema) * graw if ema > 0.0 else graw
        prev = gv
        g[t] = gv
    return g.astype(np.float32)


# 모듈 클래스(LearnableIIR/TemporalConv1d/TemporalMLP/make_module)는 src.utils.temporal_gate 에서 import
# (closed_loop_node 와 동일 소스 → ckpt state_dict 호환). 위 import 참조.


# ---------------------------------------------------------------- 지표 (exp_gate_stability 동일)
def metrics(seqs_has, gates):
    flips = flip_denom = 0
    jerk_sum = 0.0
    jerk_n = 0
    onset_lats = []
    fp_on = fp_tot = rec_on = rec_tot = 0
    for has, g in zip(seqs_has, gates):
        on = g > GATE_ON
        T = len(g)
        if T >= 2:
            flips += int(np.sum(on[1:] != on[:-1]))
            flip_denom += (T - 1)
            jerk_sum += float(np.sum(np.abs(np.diff(g))))
            jerk_n += (T - 1)
        for t in range(1, T):
            if has[t] and not has[t - 1]:
                lat = None
                for u in range(t, T):
                    if not has[u]:
                        break
                    if on[u]:
                        lat = u - t
                        break
                if lat is not None:
                    onset_lats.append(lat)
        nl = ~has
        fp_on += int(np.sum(on & nl))
        fp_tot += int(np.sum(nl))
        rec_on += int(np.sum(on & has))
        rec_tot += int(np.sum(has))
    return dict(
        flip=100.0 * flips / flip_denom if flip_denom else float("nan"),
        jerk=jerk_sum / jerk_n if jerk_n else float("nan"),
        onset_med=float(np.median(onset_lats)) if onset_lats else float("nan"),
        n_onset=len(onset_lats),
        fp=100.0 * fp_on / fp_tot if fp_tot else float("nan"),
        recall=100.0 * rec_on / rec_tot if rec_tot else float("nan"),
    )


# ---------------------------------------------------------------- 모듈 → per-seq gate (eval)
@torch.no_grad()
def module_gate_seq(model, win_seqs, batch=4096):
    """windowed 입력으로 per-seq gate 배열 (eval). cross-seq 누적 없음."""
    model.eval()
    gates = []
    for ws in win_seqs:
        X = torch.from_numpy(ws["X"])                # (T,K,10)
        outs = []
        for i in range(0, X.shape[0], batch):
            outs.append(model(X[i:i + batch]).cpu().numpy())
        gates.append(np.concatenate(outs).astype(np.float64))
    return gates


# ---------------------------------------------------------------- train
def flatten_windows(win_seqs):
    """학습용 flat tensor: X(M,K,10), has(M,), onset_w(M,) onset 가중."""
    Xs, hs, ws = [], [], []
    for s in win_seqs:
        X = s["X"]; has = s["has"]
        Xs.append(X); hs.append(has)
        # onset 근방(전이 후 0.5s 내) 가중치 부여 — 응답성 학습 강화
        w = np.ones(len(has), np.float32)
        for t in range(1, len(has)):
            if has[t] and not has[t - 1]:
                w[t:t + 10] = 3.0
        ws.append(w)
    return (np.concatenate(Xs), np.concatenate(hs), np.concatenate(ws))


def train_distill(model, train_win, train_obj_seqs, epochs=8, lr=1e-2):
    """target = EMA0.7 gate. window 입력 → EMA0.7 출력 회귀."""
    # target gate (EMA0.7) per-seq, run 순서 보존하여 flat
    tgt_list = [ema_gate_seq(s["obj"], DEP_EMA) for s in train_obj_seqs]
    X = np.concatenate([s["X"] for s in train_win])
    tgt = np.concatenate(tgt_list)
    Xt = torch.from_numpy(X); yt = torch.from_numpy(tgt)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Xt.shape[0]; bs = 8192
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            pred = model(Xt[idx])
            loss = F.mse_loss(pred, yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if ep == epochs - 1 or ep % 2 == 0:
            print(f"    [distill ep{ep}] mse={tot/n:.5f}")
    return model


def train_beat(model, train_win, epochs=12, lr=5e-3,
               w_flip=0.0, w_jerk=2.0, w_cls=1.0, w_onset=2.0):
    """beat-EMA: BCE(gate vs GT lead) + temporal smoothness(jerk) + onset 응답성.
    jerk 는 시퀀스 내 인접 프레임 gate 차이로 계산 → mini-batch 를 시퀀스 단위로 구성."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    seqs = train_win
    pos_rate = np.mean([s["has"].mean() for s in seqs])
    pos_w = float((1 - pos_rate) / max(pos_rate, 1e-3))   # class imbalance
    for ep in range(epochs):
        model.train()
        order = np.random.permutation(len(seqs))
        tot = totc = totj = 0.0
        nb = 0
        for si in order:
            s = seqs[si]
            X = torch.from_numpy(s["X"])             # (T,K,10)
            has = torch.from_numpy(s["has"].astype(np.float32))
            wsamp = torch.from_numpy(
                np.where(s["has"][1:] != s["has"][:-1], 3.0, 1.0).astype(np.float32)
            ) if len(s["has"]) > 1 else None
            g = model(X)                             # (T,)
            # classification: gate 가 GT lead 를 따라야 (FP/recall 유지)
            w_pos = torch.where(has > 0.5, torch.tensor(pos_w), torch.tensor(1.0))
            cls = F.binary_cross_entropy(g.clamp(1e-6, 1 - 1e-6), has, weight=w_pos)
            # temporal smoothness: 인접 gate 차 (flip/jerk 억제). onset 전이 프레임은 가중↓ (빠른 변화 허용)
            if g.shape[0] > 1:
                dif = (g[1:] - g[:-1]).abs()
                if wsamp is not None:
                    # 전이(onset/offset) 에서는 smoothness penalty 완화 → 빠른 응답 허용
                    jw = torch.where(wsamp > 1.5, torch.tensor(0.2), torch.tensor(1.0))
                    jerk = (dif * jw).mean()
                else:
                    jerk = dif.mean()
            else:
                jerk = torch.tensor(0.0)
            loss = w_cls * cls + w_jerk * jerk
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); totc += cls.item(); totj += jerk.item(); nb += 1
        if ep == epochs - 1 or ep % 3 == 0:
            print(f"    [beat ep{ep}] loss={tot/nb:.4f} cls={totc/nb:.4f} jerk={totj/nb:.4f}")
    return model


# ---------------------------------------------------------------- run-level split
def split_runs(seqs, frac=0.7, seed=SEED):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(seqs)); rng.shuffle(idx)
    ntr = max(1, int(round(len(seqs) * frac)))
    tr = [seqs[i] for i in idx[:ntr]]
    va = [seqs[i] for i in idx[ntr:]] or [seqs[idx[-1]]]
    return tr, va


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=K)
    ap.add_argument("--module", default="mlp", choices=["iir", "conv", "mlp"])
    ap.add_argument("--epochs_distill", type=int, default=8)
    ap.add_argument("--epochs_beat", type=int, default=12)
    ap.add_argument("--save", default="", help="beat(track B) 모델 저장 경로")
    ap.add_argument("--save-distill", default="", help="distill(track A) 모델 저장 경로")
    args = ap.parse_args()
    Kw = args.K
    torch.manual_seed(SEED); np.random.seed(SEED)

    print(f"[learned_temporal_gate] D={D}  K={Kw}  module={args.module}  device={DEVICE}")
    data = {}
    for ds in DATASETS:
        data[ds] = load_dataset(ds)
        n = sum(len(s["obj"]) for s in data[ds])
        nlead = sum(int(s["has_lead"].sum()) for s in data[ds])
        print(f"  loaded {ds:18s}: {len(data[ds]):4d} runs  {n:7d} frames  lead={100.0*nlead/max(n,1):4.1f}%")

    # ----- run-level split (leakage 방지). lead+mixed = 학습/평가 본체, pertX(no-lead) = FP 평가 전용 split -----
    lead_all = data["sangam_lead_0520"] + data["sangam_lead_0521"] + data["DriverB_new"]
    nolead_all = data["sangam_pertX_v2"]
    onset_all = data["DriverB_new"] + data["sangam_lead_0521"]

    lead_tr, lead_va = split_runs(lead_all, 0.7)
    nl_tr, nl_va = split_runs(nolead_all, 0.6)        # no-lead 도 train 에 일부 포함 (FP 억제 학습)
    # onset 평가셋은 val 쪽 lead 의 mixed run 으로 (leakage 없게 val 만)
    va_runs = set(id(s) for s in lead_va)
    onset_va = [s for s in onset_all if id(s) in va_runs]
    print(f"  split: lead train {len(lead_tr)} / val {len(lead_va)} runs ; "
          f"nolead train {len(nl_tr)} / val {len(nl_va)} runs ; onset_va {len(onset_va)}")

    train_obj = lead_tr + nl_tr
    train_win = build_windows(train_obj, Kw)

    val_lead_win = build_windows(lead_va, Kw)
    val_nl_win = build_windows(nl_va, Kw)
    val_onset_win = build_windows(onset_va, Kw) if onset_va else build_windows(lead_va, Kw)
    val_lead_has = [s["has_lead"] for s in lead_va]
    val_nl_has = [s["has_lead"] for s in nl_va]
    val_onset_has = [s["has_lead"] for s in (onset_va if onset_va else lead_va)]

    # ================= baseline: EMA gates (val) =================
    def ema_eval(ema):
        g_lead = [ema_gate_seq(s["obj"], ema) for s in lead_va]
        g_nl = [ema_gate_seq(s["obj"], ema) for s in nl_va]
        g_on = [ema_gate_seq(s["obj"], ema) for s in (onset_va if onset_va else lead_va)]
        ml = metrics(val_lead_has, g_lead)
        mn = metrics(val_nl_has, g_nl)
        mo = metrics(val_onset_has, g_on)
        return dict(flip=ml["flip"], jerk=ml["jerk"], recall=ml["recall"],
                    fp=mn["fp"], onset=mo["onset_med"], n_onset=mo["n_onset"])

    rows = {}
    rows["EMA0.7 (DEPLOYED)"] = ema_eval(0.7)
    rows["EMA0.5"] = ema_eval(0.5)
    rows["EMA0 (no smooth)"] = ema_eval(0.0)

    # ================= track A: distill (target EMA0.7) =================
    print("\n[track A] distill (target = EMA0.7)")
    mA = make_module(args.module, Kw).to(DEVICE)
    mA = train_distill(mA, train_win, train_obj, epochs=args.epochs_distill)
    g_lead = module_gate_seq(mA, val_lead_win)
    g_nl = module_gate_seq(mA, val_nl_win)
    g_on = module_gate_seq(mA, val_onset_win)
    mlA = metrics(val_lead_has, g_lead); mnA = metrics(val_nl_has, g_nl); moA = metrics(val_onset_has, g_on)
    rows[f"learned-distill ({args.module})"] = dict(
        flip=mlA["flip"], jerk=mlA["jerk"], recall=mlA["recall"],
        fp=mnA["fp"], onset=moA["onset_med"], n_onset=moA["n_onset"])
    # distill 충실도: val 에서 learned vs EMA0.7 gate L1
    emaref = [ema_gate_seq(s["obj"], 0.7) for s in lead_va]
    l1 = np.mean([np.mean(np.abs(a - b)) for a, b in zip(g_lead, emaref)])
    print(f"    distill fidelity vs EMA0.7 (val lead) mean|Δ| = {l1:.4f}")
    if args.module == "iir":
        a = torch.sigmoid(mA.alpha_logit).item()
        print(f"    learned IIR alpha = {a:.4f} (target EMA=0.7), thresh={mA.thresh.item():.3f}, scale={mA.scale.item():.3f}")

    # ================= track B: beat-EMA =================
    print("\n[track B] beat-EMA (cls + smoothness + onset)")
    mB = make_module(args.module, Kw).to(DEVICE)
    mB = train_beat(mB, train_win, epochs=args.epochs_beat)
    g_lead = module_gate_seq(mB, val_lead_win)
    g_nl = module_gate_seq(mB, val_nl_win)
    g_on = module_gate_seq(mB, val_onset_win)
    mlB = metrics(val_lead_has, g_lead); mnB = metrics(val_nl_has, g_nl); moB = metrics(val_onset_has, g_on)
    rows[f"learned-beat ({args.module})"] = dict(
        flip=mlB["flip"], jerk=mlB["jerk"], recall=mlB["recall"],
        fp=mnB["fp"], onset=moB["onset_med"], n_onset=moB["n_onset"])

    # ================= 표 =================
    order = ["EMA0 (no smooth)", "EMA0.5", "EMA0.7 (DEPLOYED)",
             f"learned-distill ({args.module})", f"learned-beat ({args.module})"]
    hdr = (f"{'variant':28s} | {'flip%':>6s} | {'jerk':>7s} | {'onset_f':>7s} | "
           f"{'onset_s':>7s} | {'FP%':>6s} | {'recall%':>7s} | {'n_onset':>7s}")
    print("\n" + "=" * len(hdr))
    print("VAL (run-level, leakage-free) — flip/jerk/recall=lead_va, FP=pertX_va, onset=mixed_va")
    print("=" * len(hdr)); print(hdr); print("-" * len(hdr))
    for name in order:
        r = rows[name]
        ons_s = r["onset"] / FPS if r["onset"] == r["onset"] else float("nan")
        print(f"{name:28s} | {r['flip']:6.2f} | {r['jerk']:7.4f} | {r['onset']:7.1f} | "
              f"{ons_s:7.3f} | {r['fp']:6.2f} | {r['recall']:7.2f} | {r['n_onset']:7d}")
    print("=" * len(hdr))

    if args.save:
        torch.save({"module": args.module, "K": Kw, "state": mB.state_dict()}, args.save)
        print(f"[saved] beat model -> {args.save}")
    if args.save_distill:
        torch.save({"module": args.module, "K": Kw, "state": mA.state_dict()}, args.save_distill)
        print(f"[saved] distill model -> {args.save_distill}")


if __name__ == "__main__":
    main()
