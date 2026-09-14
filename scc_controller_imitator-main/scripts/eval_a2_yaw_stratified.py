"""a_2 2-frame yaw_shift ablation — dx-stratified read-only 재평가 (CPU 강제).

목적: yaw_shift 가 "실제로 동작하는" 고-yaw 프레임(|dx|>=1)에서만 ON vs OFF 를 비교해
      yaw_shift 제거 가부를 판정. dx==0 stratum 은 sanity (ON≈OFF 여야 정상).

dx = round(fx * wrap_clamp(d_yaw) / bin_w),  fx=640, bin_w=32, DYAW_CLAMP=0.30.
d_yaw 는 prebake(_a2inlane_prev20) 의 curr_keys→d_yaws lut 에서 sample key 로 조회.
  lut miss / empty-key → d_yaw=0 → dx==0 (학습과 동일 동작).

두 ckpt 모두 동일 프레임 집합에서 forward:
  - yaw ckpt(a2_2f_yaw):   학습대로 yaw_shift ON (prev frame 을 dx bin roll).
  - noyaw ckpt(a2_2f_noyaw): yaw_shift OFF (d_yaw=0, prev 회전 없음).
stratum 은 sample 의 (lut) d_yaw 로 결정 — 두 모델 공통 (동일 frame 집합 보장).

사용:
  CUDA_VISIBLE_DEVICES="" python scripts/eval_a2_yaw_stratified.py \
    --i1-cache <cache.npz> --aux-dir <aux> [--run-filter X] \
    --prebake <prev20 prebake ...> \
    --yaw-ckpt outputs_multi/a2_2f_yaw/best_loss.pt \
    --noyaw-ckpt outputs_multi/a2_2f_noyaw/best_loss.pt
"""
import argparse, glob, re, sys, os, json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model.image_models.a2_model import A2ModelMultiFrame
from src.model.image_models.iy_model import yaw_shift_tokens_batch

AUX_OBJ, AUX_VLABS, AUX_VYABS, AUX_VLREL, AUX_VYREL, AUX_PROX, AUX_DIST = range(7)
AUX_PAT = re.compile(r".+?__([^_]+(?:_[^_]+)*?)__t(\d+)\.npz$")
SCALE_V, SCALE_DIST = 20.0, 40.0
FX, BIN_W, DYAW_CLAMP = 640.0, 32.0, 0.30


def _wrap_clamp_np(dy):
    two_pi = 2.0 * np.pi
    dy = dy - two_pi * np.round(dy / two_pi)
    return np.clip(dy, -DYAW_CLAMP, DYAW_CLAMP)


def build_dyaw_lut(prebake_paths):
    lut = {}
    for p in prebake_paths or []:
        if not p or not Path(p).exists():
            continue
        z = np.load(p, allow_pickle=True)
        ck = np.asarray(z["curr_keys"]).astype(str)
        dy = np.asarray(z["d_yaws"]).astype(np.float32)
        for i in range(len(ck)):
            k = ck[i]
            if k and k not in lut:
                lut[k] = float(dy[i])
    return lut


def load_cache(i1p, aux_dir, run_filter):
    z = np.load(i1p, allow_pickle=True)
    tokens = z["tokens"]; keys = z["keys"].tolist()
    k2i = {k: i for i, k in enumerate(keys)}
    samples = []   # (curr_idx, target(40,7), key, prev_idx_off20)
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
        pi = k2i.get(f"{run}__t{tick - 20:07d}", ci)
        t = np.load(f)["target"].astype(np.float32)
        samples.append((ci, t, key, pi))
    return tokens, samples


@torch.no_grad()
def run_model(model, tokens, samples, dyaw_np, yaw_on, dev, bs=512):
    """2-frame forward. yaw_on=True 면 prev 를 wrap_clamp(d_yaw) 로 roll."""
    proj = model.encoder.proj
    objs, preds = [], []
    for i in range(0, len(samples), bs):
        chunk = samples[i:i + bs]
        idx = np.array([c[0] for c in chunk])
        pidx = np.array([c[3] for c in chunk])
        ego = torch.zeros(len(chunk), 2, device=dev)
        fx = torch.full((len(chunk),), FX, device=dev)
        tc = torch.from_numpy(tokens[idx].astype(np.float32)).to(dev)
        tp = torch.from_numpy(tokens[pidx].astype(np.float32)).to(dev)
        if yaw_on:
            dy = torch.from_numpy(_wrap_clamp_np(dyaw_np[i:i + bs])).float().to(dev)
            tp = yaw_shift_tokens_batch(tp, dy, fx=fx)
        else:
            tp = yaw_shift_tokens_batch(tp, torch.zeros(len(chunk), device=dev), fx=fx)
        proj_list = [proj(tc.transpose(1, 2)).transpose(1, 2),
                     proj(tp.transpose(1, 2)).transpose(1, 2)]
        fi = model.fusion(proj_list, ego)
        logits = model.head(model.fc_proj(fi))   # (B,40,7)
        objs.append(torch.sigmoid(logits[..., 0]).cpu().numpy())
        preds.append(logits[..., 1:7].cpu().numpy())
    return np.concatenate(objs), np.concatenate(preds)


def _corr(a, b):
    if len(a) < 5 or np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def metrics(obj_prob, pred_reg, gt, frame_mask, thr=0.3):
    """frame_mask (N,) bool: stratum 에 속하는 frame. obj/reg 는 token-level."""
    fm = frame_mask
    if fm.sum() == 0:
        return None
    op = obj_prob[fm]; pr = pred_reg[fm]; g = gt[fm]
    gt_obj = g[..., AUX_OBJ] > 0.5
    pred_obj = op > thr
    tp = int((pred_obj & gt_obj).sum()); fp = int((pred_obj & ~gt_obj).sum())
    fn = int((~pred_obj & gt_obj).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    m = gt_obj
    out = {"n_frames": int(fm.sum()), "n_pos_tok": int(m.sum()),
           "prec": prec, "rec": rec, "f1": f1}
    if m.sum() < 5:
        out.update({"dist_mae": float("nan"), "vLrel_corr": float("nan"),
                    "vYrel_corr": float("nan"), "vLrel_pred_std": float("nan")})
        return out
    dist_pred = pr[..., 5] * SCALE_DIST
    vLrel_pred = pr[..., 2] * SCALE_V
    vYrel_pred = pr[..., 3] * SCALE_V
    out["dist_mae"] = float(np.abs(dist_pred[m] - g[..., AUX_DIST][m]).mean())
    out["vLrel_corr"] = _corr(vLrel_pred[m], g[..., AUX_VLREL][m])
    out["vYrel_corr"] = _corr(vYrel_pred[m], g[..., AUX_VYREL][m])
    out["vLrel_pred_std"] = float(vLrel_pred[m].std())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i1-cache", action="append", required=True)
    ap.add_argument("--aux-dir", action="append", required=True)
    ap.add_argument("--run-filter", action="append", default=[])
    ap.add_argument("--prebake", action="append", default=[])
    ap.add_argument("--yaw-ckpt", required=True)
    ap.add_argument("--noyaw-ckpt", required=True)
    ap.add_argument("--obj-thr", type=float, default=0.3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="/tmp/a2_yaw_stratified.json")
    args = ap.parse_args()
    assert len(args.i1_cache) == len(args.aux_dir)
    dev = args.device
    rfs = list(args.run_filter) + [None] * (len(args.i1_cache) - len(args.run_filter))
    lut = build_dyaw_lut(args.prebake)
    print(f"[dyaw lut] {len(lut)} non-empty keys", flush=True)

    yawm = A2ModelMultiFrame(d_token=256, d_out=128, n_frames=2).to(dev).eval()
    s = torch.load(args.yaw_ckpt, map_location=dev, weights_only=False)
    yawm.load_state_dict(s["model"] if "model" in s else s)
    noyawm = A2ModelMultiFrame(d_token=256, d_out=128, n_frames=2).to(dev).eval()
    s = torch.load(args.noyaw_ckpt, map_location=dev, weights_only=False)
    noyawm.load_state_dict(s["model"] if "model" in s else s)
    print(f"[ckpt] yaw <- {args.yaw_ckpt}\n       noyaw <- {args.noyaw_ckpt}", flush=True)

    # aggregate frames across caches
    all_obj_y, all_reg_y, all_obj_n, all_reg_n, all_gt, all_dyaw = [], [], [], [], [], []
    per_cache_stat = []
    for i1p, ad, rf in zip(args.i1_cache, args.aux_dir, rfs):
        tokens, samples = load_cache(i1p, ad, rf)
        if not samples:
            print(f"[skip] {Path(i1p).stem} rf={rf}: 0 samples", flush=True)
            continue
        dyaw_np = np.array([lut.get(s_[2], 0.0) for s_ in samples], dtype=np.float64)
        hit = float(np.mean([s_[2] in lut for s_ in samples])) * 100
        dx = np.round(FX * _wrap_clamp_np(dyaw_np) / BIN_W).astype(int)
        gt = np.stack([s_[1] for s_ in samples], axis=0)
        print(f"##### {Path(i1p).stem} rf={rf}  N={len(samples)}  lut_hit={hit:.1f}%  "
              f"dx==0 {np.mean(dx==0)*100:.1f}% |dx|==1 {np.mean(np.abs(dx)==1)*100:.1f}% "
              f"|dx|>=2 {np.mean(np.abs(dx)>=2)*100:.1f}%  pos_tok={(gt[...,0]>0.5).mean()*100:.2f}%",
              flush=True)
        oy, ry = run_model(yawm, tokens, samples, dyaw_np, True, dev)
        on, rn = run_model(noyawm, tokens, samples, dyaw_np, False, dev)
        all_obj_y.append(oy); all_reg_y.append(ry)
        all_obj_n.append(on); all_reg_n.append(rn)
        all_gt.append(gt); all_dyaw.append(dyaw_np)
        per_cache_stat.append((Path(i1p).stem, rf, len(samples), hit))

    obj_y = np.concatenate(all_obj_y); reg_y = np.concatenate(all_reg_y)
    obj_n = np.concatenate(all_obj_n); reg_n = np.concatenate(all_reg_n)
    gt = np.concatenate(all_gt); dyaw = np.concatenate(all_dyaw)
    dx = np.round(FX * _wrap_clamp_np(dyaw) / BIN_W).astype(int)

    strata = {"dx==0": dx == 0, "|dx|==1": np.abs(dx) == 1, "|dx|>=2": np.abs(dx) >= 2,
              "|dx|>=1": np.abs(dx) >= 1, "ALL": np.ones_like(dx, dtype=bool)}
    result = {"per_cache": per_cache_stat, "strata": {}}
    print("\n" + "=" * 118)
    hdr = (f"{'stratum':9s} {'model':6s} {'Nfrm':>7s} {'pos_tok':>8s} | "
           f"{'objP':>6s} {'objR':>6s} {'F1':>6s} | {'distMAE':>8s} | "
           f"{'vLrel_c':>8s} {'vYrel_c':>8s} {'vLrel_std':>9s}")
    print(hdr); print("-" * 118)
    for sname, smask in strata.items():
        my = metrics(obj_y, reg_y, gt, smask, args.obj_thr)
        mn = metrics(obj_n, reg_n, gt, smask, args.obj_thr)
        result["strata"][sname] = {"ON": my, "OFF": mn}
        if my is None:
            print(f"{sname:9s}  (empty)"); continue
        for tag, mm in [("ON", my), ("OFF", mn)]:
            print(f"{sname:9s} {tag:6s} {mm['n_frames']:>7,} {mm['n_pos_tok']:>8,} | "
                  f"{mm['prec']*100:5.1f}% {mm['rec']*100:5.1f}% {mm['f1']*100:5.1f}% | "
                  f"{mm['dist_mae']:7.3f}m | {mm['vLrel_corr']:8.3f} {mm['vYrel_corr']:8.3f} "
                  f"{mm['vLrel_pred_std']:9.3f}")
        print("-" * 118)
    print("=" * 118)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
