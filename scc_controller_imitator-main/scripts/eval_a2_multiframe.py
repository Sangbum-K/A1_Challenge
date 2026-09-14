"""a_2 multi-frame / yaw_shift ablation 평가 (cache-only).

비교 대상 (한 표):
  - v7b      : 2프레임 7ch (yaw_shift ON, but cache-eval d_yaw=0 → 사실상 no-op).  A2Model.
  - 임의의 A2ModelMultiFrame ckpt 들 (--mf NAME PATH NFRAMES YAW).

held-out 동일 셋에서:
  obj precision/recall/F1 (thr 0.5)
  dist MAE (GT obj token)
  vL_abs MAE, vL_rel corr (Pearson), vY_rel corr   ★핵심★
  거리별 vL_rel corr (0-30 근거리 강조)
  프레임 안정성: 연속 tick(Δ5) |Δobj_amax(center)| 평균, flip@0.3

eval 시 yaw_shift on 모델은 prebake d_yaw 주입(학습과 동일). off/v7b 는 d_yaw=0.

사용:
  python scripts/eval_a2_multiframe.py \
    --i1-cache <held-out cache.npz> --aux-dir <held-out aux> [--run-filter X] \
    --prebake <prev20 prebake for held-out (yaw on 모델용)> \
    --v7b data/ckpts/a2_v7b_d128_alpha075_best_loss.pt \
    --mf 2f_noyaw outputs_multi/a2_2f_noyaw/best_loss.pt 2 off \
    --mf 2f_yaw   outputs_multi/a2_2f_yaw/best_loss.pt   2 on \
    --mf 3f       outputs_multi/a2_multiframe_3f/best_loss.pt 3 off \
    --mf 4f       outputs_multi/a2_multiframe_4f/best_loss.pt 4 off
"""
import argparse, glob, re, sys, os, json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model.image_models.a2_model import A2Model, A2ModelMultiFrame
from src.model.image_models.iy_model import yaw_shift_tokens_batch

# aux ch: [obj, vL_abs, vY_abs, vL_rel, vY_rel, prox, dist]
AUX_OBJ, AUX_VLABS, AUX_VYABS, AUX_VLREL, AUX_VYREL, AUX_PROX, AUX_DIST = range(7)
CENTER = slice(16, 26)
DIST_BUCKETS = [(0, 10), (10, 20), (20, 30), (30, 60)]
AUX_PAT = re.compile(r".+?__([^_]+(?:_[^_]+)*?)__t(\d+)\.npz$")
PREV_OFFSETS = {2: [20], 3: [20, 40], 4: [20, 40, 60]}
SCALE_V, SCALE_DIST = 20.0, 40.0
DYAW_CLAMP = 0.30


def _wrap_clamp_dyaw(dy):
    two_pi = 2.0 * 3.141592653589793
    dy = dy - two_pi * torch.round(dy / two_pi)
    return dy.clamp(-DYAW_CLAMP, DYAW_CLAMP)


def build_dyaw_lut(prebake_paths):
    lut = {}
    for p in prebake_paths or []:
        if not p or not Path(p).exists():
            continue
        z = np.load(p, allow_pickle=True)
        ck = np.asarray(z["curr_keys"]); dy = np.asarray(z["d_yaws"]).astype(np.float32)
        for i in range(len(ck)):
            k = str(ck[i])
            if k not in lut:
                lut[k] = float(dy[i])
    return lut


def load_cache(i1p, aux_dir, run_filter, all_offsets):
    """all_offsets: sorted list of every tick offset referenced by any model (union).
    각 sample 은 offset→pidx dict 를 보관 → 모델별 임의 offset 조회 가능."""
    z = np.load(i1p, allow_pickle=True)
    tokens = z["tokens"]; keys = z["keys"].tolist()
    k2i = {k: i for i, k in enumerate(keys)}
    samples = []   # (curr_idx, target(40,7), run, tick, {off: pidx})
    offs = sorted(set(all_offsets))
    for f in sorted(glob.glob(str(Path(aux_dir) / "*.npz"))):
        fname = Path(f).name
        m = AUX_PAT.match(fname)
        if not m:
            continue
        run, tick = m.group(1), int(m.group(2))
        # run_filter(policy) 는 파일명 전체에 적용 — group(1) 은 run-token 만
        if run_filter and run_filter not in fname:
            continue
        key = f"{run}__t{tick:07d}"
        if key not in k2i:
            continue
        ci = k2i[key]
        # 가장자리(없는 offset) = 직전(더 가까운) offset 의 idx, 최종 self (학습 dataset 과 동일 규칙)
        pmap = {}; last = ci
        for off in offs:
            pi = k2i.get(f"{run}__t{tick - off:07d}", last)
            pmap[off] = pi; last = pi
        t = np.load(f)["target"].astype(np.float32)
        samples.append((ci, t, run, tick, pmap))
    return tokens, k2i, samples


def val_split_mask(n_total, n_caches_concat, val_ratio, seed):
    """학습과 동일한 random_split(seed) 로 val sample 의 global idx set 재현.
    n_total = ConcatDataset 전체 길이 (= 학습 pool sample 수, 순서 동일).
    returns set of global sample indices (0..n_total-1) that fall in val.
    """
    import torch as _t
    n_val = int(n_total * val_ratio)
    n_tr = n_total - n_val
    g = _t.Generator().manual_seed(seed)
    perm = _t.randperm(n_total, generator=g).tolist()
    return set(perm[n_tr:])   # random_split 은 [train, val] 순서로 잘라 val = 뒤쪽 perm


@torch.no_grad()
def run_v7b(model, tokens, samples, k2i, dev, bs=512):
    proj = model.encoder.proj
    objs, preds = [], []
    for i in range(0, len(samples), bs):
        chunk = samples[i:i+bs]
        idx = np.array([c[0] for c in chunk])
        pidx = np.array([c[4][20] for c in chunk])   # offset 20
        tc = torch.from_numpy(tokens[idx].astype(np.float32)).to(dev)
        tp = torch.from_numpy(tokens[pidx].astype(np.float32)).to(dev)
        ego = torch.zeros(tc.size(0), 2, device=dev)
        dy = torch.zeros(tc.size(0), device=dev)
        fx = torch.full((tc.size(0),), 640.0, device=dev)
        tp_rot = yaw_shift_tokens_batch(tp, dy, fx=fx)
        tcp = proj(tc.transpose(1, 2)).transpose(1, 2)
        tpp = proj(tp_rot.transpose(1, 2)).transpose(1, 2)
        fi = model.fusion(tcp, tpp, ego)
        logits = model.head(model.fc_proj(fi))   # (B,40,7)
        objs.append(torch.sigmoid(logits[..., 0]).cpu().numpy())
        preds.append(logits[..., 1:7].cpu().numpy())
    return np.concatenate(objs), np.concatenate(preds)


@torch.no_grad()
def run_mf(model, tokens, samples, dev, offs, yaw, dyaw_lut, bs=512):
    """offs: 이 모델의 prev tick offset list (예: [10,20,30]). frame0=curr 제외."""
    proj = model.encoder.proj
    objs, preds = [], []
    for i in range(0, len(samples), bs):
        chunk = samples[i:i+bs]
        idx = np.array([c[0] for c in chunk])
        fx = torch.full((len(chunk),), 640.0, device=dev)
        ego = torch.zeros(len(chunk), 2, device=dev)
        if yaw == "on":
            dy20 = torch.tensor([dyaw_lut.get(f"{c[2]}__t{c[3]:07d}", 0.0) for c in chunk],
                                dtype=torch.float32, device=dev)
            dy20 = _wrap_clamp_dyaw(dy20)
        proj_list = []
        # frame 0 = curr
        tc = torch.from_numpy(tokens[idx].astype(np.float32)).to(dev)
        proj_list.append(proj(tc.transpose(1, 2)).transpose(1, 2))
        for fi, off in enumerate(offs):
            pidx = np.array([c[4][off] for c in chunk])
            tok = torch.from_numpy(tokens[pidx].astype(np.float32)).to(dev)
            if yaw == "on":
                tok = yaw_shift_tokens_batch(tok, dy20 * (off / 20.0), fx=fx)
            proj_list.append(proj(tok.transpose(1, 2)).transpose(1, 2))
        fi_mem = model.fusion(proj_list, ego)
        logits = model.head(model.fc_proj(fi_mem))   # (B,40,7)
        objs.append(torch.sigmoid(logits[..., 0]).cpu().numpy())
        preds.append(logits[..., 1:7].cpu().numpy())
    return np.concatenate(objs), np.concatenate(preds)


def _corr(a, b):
    if len(a) < 5 or np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def metrics(obj_prob, pred_reg, gt, thr=0.5):
    """pred_reg (N,40,6) normalized (학습 scale). gt (N,40,7) raw."""
    gt_obj = gt[..., AUX_OBJ] > 0.5
    pred_obj = obj_prob > thr
    tp = int((pred_obj & gt_obj).sum()); fp = int((pred_obj & ~gt_obj).sum())
    fn = int((~pred_obj & gt_obj).sum())
    prec = tp/(tp+fp) if (tp+fp) else 0.0
    rec = tp/(tp+fn) if (tp+fn) else 0.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    out = {"prec": prec, "rec": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
           "n_pos": int(gt_obj.sum())}
    # denorm preds: reg idx0=vL_abs..3=vY_rel scale_v, 4=prox 1, 5=dist scale_dist
    m = gt_obj
    if m.sum() < 5:
        out.update({"dist_mae": float("nan"), "vLabs_mae": float("nan"),
                    "vLrel_corr": float("nan"), "vLrel_mae": float("nan"),
                    "vYrel_corr": float("nan"), "vLrel_corr_near": float("nan"),
                    "vLrel_pred_std": float("nan"), "vLrel_gt_std": float("nan")})
        return out
    dist_pred = pred_reg[..., 5] * SCALE_DIST
    vLabs_pred = pred_reg[..., 0] * SCALE_V
    vLrel_pred = pred_reg[..., 2] * SCALE_V
    vYrel_pred = pred_reg[..., 3] * SCALE_V
    out["dist_mae"] = float(np.abs(dist_pred[m] - gt[..., AUX_DIST][m]).mean())
    out["vLabs_mae"] = float(np.abs(vLabs_pred[m] - gt[..., AUX_VLABS][m]).mean())
    out["vLrel_mae"] = float(np.abs(vLrel_pred[m] - gt[..., AUX_VLREL][m]).mean())
    out["vLrel_corr"] = _corr(vLrel_pred[m], gt[..., AUX_VLREL][m])
    out["vYrel_corr"] = _corr(vYrel_pred[m], gt[..., AUX_VYREL][m])
    out["vLrel_pred_std"] = float(vLrel_pred[m].std())
    out["vLrel_gt_std"] = float(gt[..., AUX_VLREL][m].std())
    # near-range vL_rel corr (GT dist < 30)
    near = m & (gt[..., AUX_DIST] < 30)
    if near.sum() >= 5:
        out["vLrel_corr_near"] = _corr(vLrel_pred[near], gt[..., AUX_VLREL][near])
    else:
        out["vLrel_corr_near"] = float("nan")
    return out


def stability(obj_prob, samples, gate_thr=0.3, file_step=5):
    rt2row = {(s[2], s[3]): r for r, s in enumerate(samples)}
    amax = obj_prob[:, CENTER].max(axis=1)
    deltas, flips, npair = [], 0, 0
    for r, s in enumerate(samples):
        run, tick = s[2], s[3]
        pr = rt2row.get((run, tick - file_step))
        if pr is None:
            continue
        npair += 1
        deltas.append(abs(amax[r] - amax[pr]))
        if (amax[r] > gate_thr) != (amax[pr] > gate_thr):
            flips += 1
    if npair == 0:
        return {"mean_dobj": float("nan"), "flip_rate": float("nan"), "n_pairs": 0}
    return {"mean_dobj": float(np.mean(deltas)), "flip_rate": flips / npair, "n_pairs": npair}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i1-cache", action="append", required=True)
    ap.add_argument("--aux-dir", action="append", required=True)
    ap.add_argument("--run-filter", action="append", default=[])
    ap.add_argument("--prebake", action="append", default=[])
    ap.add_argument("--v7b", required=True)
    ap.add_argument("--mf", nargs="+", action="append", default=[],
                    metavar="NAME PATH NFRAMES YAW [OFFSETS]",
                    help="multi-frame ckpt: NAME PATH NFRAMES YAW [OFFSETS]. "
                         "OFFSETS=comma-sep prev tick offsets (예: 10,20,30). "
                         "생략 시 PREV_OFFSETS[NFRAMES] fallback (legacy 20/40/60).")
    ap.add_argument("--val-split", action="store_true",
                    help="학습 pool 을 그대로 받아 seed=0 random_split 의 val 5%만 평가 "
                         "(모든 모델 동일 held-out, lead-rich). cache/aux 순서 = launcher 와 동일해야")
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="/tmp/a2_multiframe_eval.json")
    args = ap.parse_args()
    assert len(args.i1_cache) == len(args.aux_dir)
    dev = args.device
    rfs = list(args.run_filter) + [None] * (len(args.i1_cache) - len(args.run_filter))
    dyaw_lut = build_dyaw_lut(args.prebake)
    print(f"[dyaw lut] {len(dyaw_lut)} entries (for yaw=on models)")

    v7b = A2Model(d_token=256, d_out=128).to(dev).eval()
    sd = torch.load(args.v7b, map_location=dev, weights_only=False)
    v7b.load_state_dict(sd["model"] if "model" in sd else sd)

    mf_models = []   # (name, model, offs, yaw)
    union_offsets = {20}   # v7b 가 offset 20 사용
    for spec in args.mf:
        if len(spec) == 4:
            name, path, nf, yaw = spec
            nf = int(nf); offs = list(PREV_OFFSETS[nf])
        elif len(spec) == 5:
            name, path, nf, yaw, offstr = spec
            offs = [int(x) for x in offstr.split(",") if x.strip()]
            nf = 1 + len(offs)
        else:
            raise ValueError(f"--mf needs 4 or 5 args, got {len(spec)}: {spec}")
        s = torch.load(path, map_location=dev, weights_only=False)
        sd_mf = s["model"] if "model" in s else s
        # ckpt 의 encoder.proj.weight (d_token, C_b, 1) 에서 d_token / C_b 추론 (b0=192, v2_s=256)
        pw = sd_mf.get("encoder.proj.weight")
        d_tok_ck = int(pw.shape[0]) if pw is not None else 256
        c_b_ck = int(pw.shape[1]) if pw is not None else 256
        mm = A2ModelMultiFrame(d_token=d_tok_ck, d_out=128, n_frames=nf).to(dev).eval()
        if c_b_ck != mm.encoder.proj.in_channels:
            import torch.nn as _nn
            mm.encoder.proj = _nn.Conv1d(c_b_ck, d_tok_ck, kernel_size=1).to(dev)
        mm.load_state_dict(sd_mf)
        mf_models.append((name, mm, offs, yaw))
        print(f"[mf-arch] {name}: d_token={d_tok_ck} C_b={c_b_ck}")
        union_offsets.update(offs)
        print(f"[mf] {name}: nf={nf} offs={offs} yaw={yaw} <- {path}")
    union_offsets = sorted(union_offsets)

    # === build concat sample list (학습과 동일 순서) ===
    cache_samples = []   # per-cache (tokens, k2i, samples, rf, stem)
    concat_len = 0
    for i1p, ad, rf in zip(args.i1_cache, args.aux_dir, rfs):
        tokens, k2i, samples = load_cache(i1p, ad, rf, union_offsets)
        cache_samples.append((tokens, k2i, samples, Path(i1p).stem))
        concat_len += len(samples)
    val_set = None
    if args.val_split:
        val_set = val_split_mask(concat_len, len(cache_samples), args.val_ratio, args.seed)
        print(f"[val-split] concat_len={concat_len}  val={len(val_set)} (ratio {args.val_ratio})")

    # v7b 는 256ch encoder. cache C_b 가 다르면(b0=192) shape mismatch → v7b skip.
    cache_cb = int(cache_samples[0][0].shape[-1]) if cache_samples else 256
    v7b_cb = int(v7b.encoder.proj.in_channels)
    eval_v7b = (cache_cb == v7b_cb)
    if not eval_v7b:
        print(f"[v7b] SKIP — cache C_b={cache_cb} != v7b proj in={v7b_cb} (b0 run)")

    # aggregate across caches (val_split 시 val sample 만)
    all_obj = {}; all_reg = {}; all_gt = []; all_samples = []
    if eval_v7b:
        all_obj["v7b"] = []; all_reg["v7b"] = []
    for nm, *_ in mf_models:
        all_obj[nm] = []; all_reg[nm] = []
    gidx = 0
    for (tokens, k2i, samples, stem) in cache_samples:
        if val_set is not None:
            sel = [j for j in range(len(samples)) if (gidx + j) in val_set]
            gidx += len(samples)
            samples = [samples[j] for j in sel]
        else:
            gidx += len(samples)
        if not samples:
            continue
        gt = np.stack([s[1] for s in samples], axis=0)
        print(f"\n##### {stem}  N={len(samples)}  pos={ (gt[...,0]>0.5).mean()*100:.2f}% #####",
              flush=True)
        if eval_v7b:
            o_v, r_v = run_v7b(v7b, tokens, samples, k2i, dev)
            all_obj["v7b"].append(o_v); all_reg["v7b"].append(r_v)
        for (nm, mm, offs, yaw) in mf_models:
            o, r = run_mf(mm, tokens, samples, dev, offs, yaw, dyaw_lut)
            all_obj[nm].append(o); all_reg[nm].append(r)
        all_gt.append(gt)
        for s in samples:
            all_samples.append((s[0], s[1], f"{stem}::{s[2]}", s[3], s[4]))

    gt = np.concatenate(all_gt, axis=0)
    result = {}
    names = (["v7b"] if eval_v7b else []) + [nm for nm, *_ in mf_models]
    print("\n" + "=" * 110)
    hdr = (f"{'model':10s} {'objP':>6s} {'objR':>6s} {'F1':>6s} | "
           f"{'vLrel_c':>8s} {'vLrel_cN':>9s} {'vLabs_MAE':>9s} {'vLrel_MAE':>9s} {'vYrel_c':>8s} | "
           f"{'distMAE':>8s} | {'flip@.3':>8s} {'Δobj':>7s}")
    print(hdr); print("-" * 110)
    for nm in names:
        o = np.concatenate(all_obj[nm]); r = np.concatenate(all_reg[nm])
        m = metrics(o, r, gt)
        st = stability(o, all_samples)
        result[nm] = {**m, "stab": st}
        print(f"{nm:10s} {m['prec']*100:5.1f}% {m['rec']*100:5.1f}% {m['f1']*100:5.1f}% | "
              f"{m['vLrel_corr']:8.3f} {m['vLrel_corr_near']:9.3f} {m['vLabs_mae']:9.3f} "
              f"{m['vLrel_mae']:9.3f} {m['vYrel_corr']:8.3f} | "
              f"{m['dist_mae']:7.3f}m | {st['flip_rate']*100:7.2f}% {st['mean_dobj']:7.4f}")
    print("=" * 110)
    # ── obj-bin 수치 분석: present(in-lane lead 있음) vs absent(없음) 의 obj_amax(center 16:26 max prob) 분포 ──
    gt_c = gt[:, 16:26, 0].max(axis=1)          # (N,) GT center obj
    pres = gt_c > 0.5; absn = ~pres
    print(f"[obj-bin] in-lane(16:26) obj_amax(prob) — present(n={int(pres.sum())}) vs absent(n={int(absn.sum())})")
    print(f"{'model':10s} {'PRES mean':>9s} {'p10':>6s} {'p50':>6s} | {'ABS mean':>9s} {'p90':>6s} {'p99':>6s} | {'gap':>6s} {'rec@.5':>7s} {'FP@.5':>7s} {'FP@.3':>7s}")
    _objbin = {}
    for nm in names:
        o = np.concatenate(all_obj[nm]); amax = o[:, 16:26].max(axis=1)   # (N,) pred center prob
        pm, am = amax[pres], amax[absn]; _objbin[nm] = (pm, am)
        rec = (pm > 0.5).mean()*100; fp5 = (am > 0.5).mean()*100; fp3 = (am > 0.3).mean()*100
        print(f"{nm:10s} {pm.mean():9.3f} {np.percentile(pm,10):6.3f} {np.percentile(pm,50):6.3f} | "
              f"{am.mean():9.3f} {np.percentile(am,90):6.3f} {np.percentile(am,99):6.3f} | "
              f"{pm.mean()-am.mean():6.3f} {rec:6.1f}% {fp5:6.1f}% {fp3:6.1f}%")
    print("=" * 110)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        nn = len(names); fig, axes = plt.subplots(1, nn, figsize=(5.5 * nn, 4.2), squeeze=False)
        for i, nm in enumerate(names):
            pm, am = _objbin[nm]; ax = axes[0][i]
            ax.hist(am, bins=40, range=(0, 1), alpha=0.55, color="tab:red",
                    density=True, label=f"absent (no lead) n={len(am)}")
            ax.hist(pm, bins=40, range=(0, 1), alpha=0.55, color="tab:green",
                    density=True, label=f"present (lead) n={len(pm)}")
            ax.axvline(0.5, color="k", ls="--", lw=1.2, label="thr 0.5")
            ax.axvline(0.574, color="dimgray", ls=":", lw=1.2, label="gate (logit 0.3)")
            ax.set_title(f"{nm}  rec={(pm>0.5).mean()*100:.0f}%  FP@.5={(am>0.5).mean()*100:.0f}%  "
                         f"gap={pm.mean()-am.mean():.2f}", fontsize=10)
            ax.set_xlabel("in-lane (bin16-25) obj_amax  [sigmoid prob]"); ax.set_ylabel("density")
            ax.legend(fontsize=8)
        fig.suptitle("a_2 object detection: present vs absent obj_amax distribution", fontsize=12)
        fig.tight_layout(); png = os.path.join(os.path.dirname(args.out), "objbin_dist.png")
        fig.savefig(png, dpi=95); print(f"[plot] {png}")
    except Exception as e:
        print("[plot] fail:", e)
    # ── 거리별 FN 분석: present lead 을 GT 거리로 binning → 모델별 FN(미검출 amax<0.5)% ──
    _gc = gt[:, 16:26, 0]; _posb = _gc.argmax(axis=1)
    _ld = gt[np.arange(gt.shape[0]), 16 + _posb, AUX_DIST]     # (N,) lead dist(m); present 만 유효
    _db = [0, 10, 20, 30, 40, 60, 9999]; _dl = ["0-10", "10-20", "20-30", "30-40", "40-60", "60+"]
    _pn = [int((pres & (_ld >= _db[j]) & (_ld < _db[j + 1])).sum()) for j in range(len(_dl))]
    print("\n[FN by distance] present lead 미검출률(amax<0.5) — GT 거리(m) 별")
    print("model".ljust(10) + "".join(f"{l:>9s}" for l in _dl))
    print("present_n".ljust(10) + "".join(f"{n:>9d}" for n in _pn))
    _fnby = {}
    for nm in names:
        amax = np.concatenate(all_obj[nm])[:, 16:26].max(axis=1)
        row = nm[:9].ljust(10); fr = []
        for j in range(len(_dl)):
            mm = pres & (_ld >= _db[j]) & (_ld < _db[j + 1])
            f = ((amax[mm] < 0.5).mean() * 100) if mm.sum() else float("nan")
            fr.append(f); row += (f"{f:8.1f}%" if mm.sum() else "    -    ")
        _fnby[nm] = fr; print(row)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        xc = [(_db[j] + min(_db[j + 1], 70)) / 2 for j in range(len(_dl))]
        plt.figure(figsize=(7, 4.5))
        for nm in names:
            plt.plot(xc, _fnby[nm], marker="o", label=nm)
        plt.xlabel("GT lead distance (m)"); plt.ylabel("FN rate % (missed lead, obj_amax<0.5)")
        plt.title("a_2 false-negative (miss) rate by lead distance"); plt.legend(); plt.grid(alpha=0.3)
        p2 = os.path.join(os.path.dirname(args.out), "fn_by_dist.png"); plt.tight_layout(); plt.savefig(p2, dpi=95)
        print(f"[plot] {p2}")
    except Exception as e:
        print("[plot-fn] fail:", e)
    print("=" * 110)
    # ── FP 산점도: absent(no lead) 인데 obj_amax>0.5(FP) 인 프레임을 모델-PRED (거리, closing) 평면에 ──
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        show = names
        fig, axes = plt.subplots(1, len(show), figsize=(5 * len(show), 4.2), squeeze=False)
        for i, nm in enumerate(show):
            o = np.concatenate(all_obj[nm]); r = np.concatenate(all_reg[nm])     # (N,40),(N,40,6)
            amax = o[:, 16:26].max(axis=1); fpb = 16 + o[:, 16:26].argmax(axis=1)
            fp = absn & (amax > 0.5); idx = np.where(fp)[0]; bn = fpb[fp]
            pdist = r[idx, bn, 5] * SCALE_DIST            # pred dist(m)
            pclos = -r[idx, bn, 2] * SCALE_V              # pred closing(m/s, +=접근)
            ax = axes[0][i]
            ax.scatter(pdist, pclos, s=9, alpha=0.4, c="tab:red")
            ax.axhline(0, color="k", lw=0.6); ax.grid(alpha=0.3)
            ax.set_xlim(0, 80); ax.set_ylim(-10, 20)
            ax.set_title(f"{nm}  FP n={len(idx)} (FP@.5={(amax[absn] > 0.5).mean() * 100:.0f}%)", fontsize=10)
            ax.set_xlabel("pred distance (m)"); ax.set_ylabel("pred closing (m/s, +=approach)")
        fig.suptitle("False-positive (absent→pred lead): pred distance x closing plane")
        fig.tight_layout(); p3 = os.path.join(os.path.dirname(args.out), "fp_dist_closing.png")
        fig.savefig(p3, dpi=95); print(f"[plot] {p3}")
    except Exception as e:
        print("[plot-fp] fail:", e)
    print("=" * 110)
    # ── FN 산점도: present(lead 있음) 프레임을 GT (거리, closing) 평면에 → TP(green)/FN(red) ──
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        _lvrel = gt[np.arange(gt.shape[0]), 16 + _posb, AUX_VLREL]   # (N,) GT vL_rel(m/s); present 만 유효
        gtclos = -_lvrel                                            # closing(m/s, +=접근)
        show = names
        fig, axes = plt.subplots(1, len(show), figsize=(5 * len(show), 4.2), squeeze=False)
        for i, nm in enumerate(show):
            amax = np.concatenate(all_obj[nm])[:, 16:26].max(axis=1)
            tp = pres & (amax > 0.5); fn = pres & (amax < 0.5); ax = axes[0][i]
            ax.scatter(_ld[tp], gtclos[tp], s=7, alpha=0.25, c="tab:green", label=f"TP n={int(tp.sum())}")
            ax.scatter(_ld[fn], gtclos[fn], s=12, alpha=0.55, c="tab:red", label=f"FN n={int(fn.sum())}")
            ax.axhline(0, color="k", lw=0.6); ax.grid(alpha=0.3)
            ax.set_xlim(0, 80); ax.set_ylim(-10, 20)
            ax.set_title(f"{nm}  FN={(amax[pres] < 0.5).mean() * 100:.0f}% (rec={(amax[pres] > 0.5).mean() * 100:.0f}%)", fontsize=10)
            ax.set_xlabel("GT distance (m)"); ax.set_ylabel("GT closing (m/s, +=approach)"); ax.legend(fontsize=8)
        fig.suptitle("False-negative (missed lead): GT distance x closing plane  (red=FN)")
        fig.tight_layout(); p4 = os.path.join(os.path.dirname(args.out), "fn_dist_closing.png")
        fig.savefig(p4, dpi=95); print(f"[plot] {p4}")
    except Exception as e:
        print("[plot-fn2] fail:", e)
    print("=" * 110)
    # ── high-closing far-miss 정량: far(GT dist>=40m) present 를 closing 버킷별 FN% + 위험 miss 비율/TTC ──
    farb = pres & (_ld >= 40.0); cl = gtclos                      # closing(m/s, +=approach)
    cbk = [(-1e9, 0.0, "recede<=0"), (0.0, 2.0, "mild 0-2"), (2.0, 5.0, "moder 2-5"), (5.0, 1e9, "fast >5")]
    print(f"\n[far-band FN x closing] far=GT dist>=40m (far_present_n={int(farb.sum())})  HIGH=closing>=2m/s & dist>=40 & miss")
    print("far_n".ljust(10) + "".join(f"{int((farb & (cl >= c[0]) & (cl < c[1])).sum()):>11d}" for c in cbk))
    print("model".ljust(10) + "".join(f"{c[2]:>11s}" for c in cbk) + f"{'HIGHmiss%':>11s}{'TTCmed':>9s}")
    for nm in names:
        amax = np.concatenate(all_obj[nm])[:, 16:26].max(axis=1); row = nm[:9].ljust(10)
        for c in cbk:
            mm = farb & (cl >= c[0]) & (cl < c[1])
            row += (f"{(amax[mm] < 0.5).mean() * 100:10.1f}%" if mm.sum() else "     -     ")
        hm = farb & (cl >= 2.0) & (amax < 0.5)                    # high-closing far-miss (위험)
        hpct = hm.sum() / max(int(pres.sum()), 1) * 100
        ttc = (_ld[hm] / np.clip(cl[hm], 0.1, None)) if hm.sum() else np.array([np.nan])
        row += f"{hpct:10.2f}%{np.median(ttc):8.1f}s"
        print(row)
    print("(HIGHmiss% = 전체 present 대비, 거리>=40 & 접근>=2m/s 인데 놓친 비율 / TTCmed = 그 miss 들의 중앙 TTC)")
    print("=" * 110)
    # ── gate threshold sweep — 전 모델 operating characteristic (rec/farRec/nearRec/prec/F1/FP/HIGHmiss) ──
    #    far(>=40m)는 거의 0521 단일 dataset (GT dist confound) — far metric 해석 시 유의.
    farp = pres & (_ld >= 40.0); nearp = pres & (_ld < 40.0)
    thrs = [0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]
    print("\n[gate sweep] 전 모델 threshold 별 operating characteristic (far=0521 편중 주의)")
    _sweep = {}
    for nm in names:
        amax = np.concatenate(all_obj[nm])[:, 16:26].max(axis=1)
        print(f"\n-- {nm} --  {'prob':>5s}{'rec':>7s}{'farR':>7s}{'nearR':>7s}{'prec':>7s}{'F1':>6s}{'FP%':>7s}{'HIGH%':>7s}")
        rows = []
        for t in thrs:
            tp = int((amax[pres] >= t).sum()); fp = int((amax[absn] >= t).sum())
            rec = tp / max(int(pres.sum()), 1) * 100
            frec = (amax[farp] >= t).mean() * 100 if farp.sum() else float("nan")
            nrec = (amax[nearp] >= t).mean() * 100 if nearp.sum() else float("nan")
            prec = tp / (tp + fp) * 100 if (tp + fp) else float("nan")
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            fpr = fp / max(int(absn.sum()), 1) * 100
            hm = (farp & (cl >= 2.0) & (amax < t)).sum() / max(int(pres.sum()), 1) * 100
            rows.append((t, rec, frec, nrec, prec, f1, fpr, hm))
            print(f"{'':6s}{t:5.2f}{rec:6.1f}%{frec:6.1f}%{nrec:6.1f}%{prec:6.1f}%{f1:5.1f}{fpr:6.2f}%{hm:6.2f}%")
        _sweep[nm] = rows
        best = max(rows, key=lambda r: r[5])
        print(f"   best F1={best[5]:.1f} @ prob {best[0]:.2f} (rec {best[1]:.0f}% prec {best[4]:.0f}% FP {best[6]:.1f}% farR {best[2]:.0f}%)")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
        for nm in names:
            arr = np.array([(r[6], r[1], r[2]) for r in _sweep[nm]])   # FP, rec, farRec
            a1.plot(arr[:, 0], arr[:, 1], "o-", label=nm, alpha=0.8)
            a2.plot(arr[:, 0], arr[:, 2], "s-", label=nm, alpha=0.8)
        a1.set_xlabel("FP rate %"); a1.set_ylabel("overall recall %"); a1.set_title("recall vs FP"); a1.grid(alpha=0.3); a1.legend(fontsize=8)
        a2.set_xlabel("FP rate %"); a2.set_ylabel("far recall % (>=40m, ~0521)"); a2.set_title("far recall vs FP"); a2.grid(alpha=0.3); a2.legend(fontsize=8)
        ps = os.path.join(os.path.dirname(args.out), "gate_sweep_allmodels.png")
        fig.tight_layout(); fig.savefig(ps, dpi=95); print(f"[plot] {ps}")
    except Exception as e:
        print("[plot-sweep] fail:", e)
    print("=" * 110)
    # ── 속도(vL_rel) 정확도 거리별: detected(amax>0.5) present 만, 거리 bin 별 (pred-gt) bias/std (m/s) ──
    _db2 = [0, 10, 20, 30, 40, 60, 9999]; _dl2 = ["0-10", "10-20", "20-30", "30-40", "40-60", "60+"]
    print("\n[vL_rel error by distance] detected(amax>0.5) present only — bias(mean)±std (m/s), (n)")
    print("model".ljust(10) + "".join(f"{l:>16s}" for l in _dl2))
    _verr = {}
    for nm in names:
        o = np.concatenate(all_obj[nm]); r = np.concatenate(all_reg[nm])
        amax = o[:, 16:26].max(axis=1); bn = 16 + o[:, 16:26].argmax(axis=1)
        det = pres & (amax > 0.5); idx = np.where(det)[0]; b = bn[det]
        pv = r[idx, b, 2] * SCALE_V; gv = gt[idx, b, AUX_VLREL]; gd = gt[idx, b, AUX_DIST]
        err = pv - gv; row = nm[:9].ljust(10); st = []
        for j in range(len(_dl2)):
            mm = (gd >= _db2[j]) & (gd < _db2[j + 1])
            if mm.sum() >= 3:
                bias = float(err[mm].mean()); sd = float(err[mm].std()); st.append((bias, sd, int(mm.sum())))
                row += f"{bias:+.1f}±{sd:.1f}({int(mm.sum())})".rjust(16)
            else:
                st.append((float("nan"), float("nan"), int(mm.sum()))); row += f"{'-':>16s}"
        _verr[nm] = st; print(row)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        xc = [(_db2[j] + min(_db2[j + 1], 70)) / 2 for j in range(len(_dl2))]
        plt.figure(figsize=(7.5, 4.8))
        for nm in names:
            ys = [s[0] for s in _verr[nm]]; es = [s[1] for s in _verr[nm]]
            plt.errorbar(xc, ys, yerr=es, marker="o", capsize=3, label=nm, alpha=0.8)
        plt.axhline(0, color="k", lw=0.7); plt.grid(alpha=0.3)
        plt.xlabel("GT lead distance (m)"); plt.ylabel("vL_rel error (pred-gt, m/s)  bias±std")
        plt.title("a_2 closing-speed accuracy by distance (detected present)"); plt.legend(fontsize=8)
        pv2 = os.path.join(os.path.dirname(args.out), "vrel_err_by_dist.png")
        plt.tight_layout(); plt.savefig(pv2, dpi=95); print(f"[plot] {pv2}")
    except Exception as e:
        print("[plot-vrel] fail:", e)
    print("=" * 110)
    _ref = names[0]
    print(f"GT vL_rel std={metrics(np.concatenate(all_obj[_ref]), np.concatenate(all_reg[_ref]), gt)['vLrel_gt_std']:.3f}, "
          f"n_pos={int((gt[...,0]>0.5).sum())}, n_pairs={result[_ref]['stab']['n_pairs']}")
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
