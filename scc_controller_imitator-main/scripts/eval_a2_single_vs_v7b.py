"""a_2 single-frame obj·dist  vs  v7b (2프레임 7ch) held-out 비교 (cache-only).

i_1 cache (raw backbone token, N×40×256) + aux_targets 로 두 모델을 동일 데이터에 평가:
  - obj precision/recall/F1 (per-token, thr 0.5)
  - 거리별 recall (0-10 / 10-20 / 20-30 / 30-60 m) — GT dist 기준 positive token bucket
  - dist MAE (obj=True token)
  - 프레임 간 obj 안정성: 연속 tick (Δtick=file_step) 쌍의
      |Δobj_amax(center cols 16:26)|  평균,  gate flip율 @thr0.3 (obj_amax sigmoid 가 0.3 경계 넘나든 비율)

v7b path:  cache tok → encoder.proj → fusion(prev=yaw_shift(self), ego=0) → fc_proj → head[7]
single path: cache tok → encoder.proj → sa → fc_proj → head[2]  (prev 불필요)

사용:
  python scripts/eval_a2_single_vs_v7b.py \
    --i1-cache .../i1_tok_cache_spd_nolead_0520.npz \
    --aux-dir  .../aux_targets_spd_nolead_0520 \
    --v7b data/ckpts/a2_v7b_d128_alpha075_best_loss.pt \
    --single outputs_multi/a2_single_objdist_v1/best.pt \
    --scale-dist 40.0
"""
import argparse, glob, re, sys, os, json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model.image_models.a2_model import A2Model, A2ModelSingleObjDist
from src.model.image_models.iy_model import yaw_shift_tokens_batch

AUX_OBJ, AUX_DIST = 0, 6
CENTER = slice(16, 26)
DIST_BUCKETS = [(0, 10), (10, 20), (20, 30), (30, 60)]


def load_cache(i1p, aux_dir):
    z = np.load(i1p, allow_pickle=True)
    tokens = z["tokens"]; keys = z["keys"].tolist()
    k2i = {k: i for i, k in enumerate(keys)}
    pat = re.compile(r".+?__([^_]+(?:_[^_]+)*?)__t(\d+)\.npz$")
    samples = []  # (cache_idx, target(40,7), run, tick)
    for f in sorted(glob.glob(str(Path(aux_dir) / "*.npz"))):
        m = pat.match(Path(f).name)
        if not m: continue
        run, tick = m.group(1), int(m.group(2))
        key = f"{run}__t{tick:07d}"
        if key not in k2i: continue
        t = np.load(f)["target"].astype(np.float32)
        samples.append((k2i[key], t, run, tick))
    return tokens, k2i, samples


@torch.no_grad()
def run_v7b(model, tokens, samples, k2i, dev, bs=512):
    """returns obj_prob (N,40), dist_pred (N,40) denorm."""
    proj = model.encoder.proj
    objs, dists = [], []
    sd = model.head.fc  # noqa (head)
    for i in range(0, len(samples), bs):
        chunk = samples[i:i+bs]
        idx = np.array([c[0] for c in chunk])
        # prev = self (ego=0, no yaw shift → prev_tok rot = prev = self idx-20 if exists else self)
        pidx = []
        for (ci, t, run, tick) in chunk:
            pk = f"{run}__t{tick-20:07d}"
            pidx.append(k2i.get(pk, ci))
        pidx = np.array(pidx)
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
        dists.append((logits[..., 6] * 40.0).cpu().numpy())  # v7b dist = ch6, scale 40
    return np.concatenate(objs), np.concatenate(dists)


@torch.no_grad()
def run_single(model, tokens, samples, dev, scale_dist, bs=512):
    proj = model.encoder.proj
    objs, dists = [], []
    for i in range(0, len(samples), bs):
        idx = np.array([c[0] for c in samples[i:i+bs]])
        tc = torch.from_numpy(tokens[idx].astype(np.float32)).to(dev)
        tcp = proj(tc.transpose(1, 2)).transpose(1, 2)
        logits = model.forward_tokens(tcp)        # (B,40,2)
        objs.append(torch.sigmoid(logits[..., 0]).cpu().numpy())
        dists.append((logits[..., 1] * scale_dist).cpu().numpy())
    return np.concatenate(objs), np.concatenate(dists)


def metrics(obj_prob, dist_pred, gt, thr=0.5):
    gt_obj = gt[..., AUX_OBJ] > 0.5
    gt_dist = gt[..., AUX_DIST]
    pred_obj = obj_prob > thr
    tp = int((pred_obj & gt_obj).sum()); fp = int((pred_obj & ~gt_obj).sum())
    fn = int((~pred_obj & gt_obj).sum()); tn = int((~pred_obj & ~gt_obj).sum())
    prec = tp/(tp+fp) if (tp+fp) else 0.0
    rec = tp/(tp+fn) if (tp+fn) else 0.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    out = {"prec": prec, "rec": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    # 거리별 recall (GT positive token 을 GT dist bucket 으로)
    out["rec_by_dist"] = {}
    for lo, hi in DIST_BUCKETS:
        m = gt_obj & (gt_dist >= lo) & (gt_dist < hi)
        nb = int(m.sum())
        r = float((pred_obj & m).sum()) / nb if nb else float("nan")
        out["rec_by_dist"][f"{lo}-{hi}"] = {"recall": r, "n": nb}
    # dist MAE on GT positive
    m = gt_obj
    if m.sum() > 0:
        out["dist_mae"] = float(np.abs(dist_pred[m] - gt_dist[m]).mean())
        out["dist_rmse"] = float(np.sqrt(((dist_pred[m] - gt_dist[m]) ** 2).mean()))
    return out


def stability(obj_prob, samples, gate_thr=0.3):
    """연속 tick (Δtick=5) 쌍의 |Δobj_amax(center)| 평균 + gate flip율.
    obj_amax = max over center cols. gate = obj_amax_prob > gate_thr.
    """
    # build (run,tick)->row
    rt2row = {(s[2], s[3]): r for r, s in enumerate(samples)}
    amax = obj_prob[:, CENTER].max(axis=1)   # (N,) prob
    deltas, flips, npair = [], 0, 0
    for r, s in enumerate(samples):
        run, tick = s[2], s[3]
        pr = rt2row.get((run, tick - 5))     # file_step=5
        if pr is None: continue
        npair += 1
        deltas.append(abs(amax[r] - amax[pr]))
        g_now = amax[r] > gate_thr; g_prev = amax[pr] > gate_thr
        if g_now != g_prev: flips += 1
    if npair == 0:
        return {"mean_dobj": float("nan"), "flip_rate": float("nan"), "n_pairs": 0}
    return {"mean_dobj": float(np.mean(deltas)), "flip_rate": flips / npair, "n_pairs": npair}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i1-cache", nargs="+", required=True)
    ap.add_argument("--aux-dir", nargs="+", required=True)
    ap.add_argument("--v7b", required=True)
    ap.add_argument("--single", required=True)
    ap.add_argument("--scale-dist", type=float, default=40.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="/tmp/a2_single_vs_v7b.json")
    args = ap.parse_args()
    assert len(args.i1_cache) == len(args.aux_dir)
    dev = args.device

    # load both models
    v7b = A2Model(d_token=256, d_out=128).to(dev).eval()
    sd = torch.load(args.v7b, map_location=dev, weights_only=False)
    v7b.load_state_dict(sd["model"] if "model" in sd else sd)
    single = A2ModelSingleObjDist(d_token=256, d_out=128, n_ch=2, n_sa=2).to(dev).eval()
    sd2 = torch.load(args.single, map_location=dev, weights_only=False)
    single.load_state_dict(sd2["model"] if "model" in sd2 else sd2)

    result = {}
    for i1p, ad in zip(args.i1_cache, args.aux_dir):
        name = Path(i1p).stem
        tokens, k2i, samples = load_cache(i1p, ad)
        gt = np.stack([s[1] for s in samples], axis=0)  # (N,40,7)
        print(f"\n##### {name}  N={len(samples)}  gt_pos_rate={ (gt[...,0]>0.5).mean()*100:.2f}% #####", flush=True)
        o_v, d_v = run_v7b(v7b, tokens, samples, k2i, dev)
        o_s, d_s = run_single(single, tokens, samples, dev, args.scale_dist)
        m_v = metrics(o_v, d_v, gt); m_s = metrics(o_s, d_s, gt)
        st_v = stability(o_v, samples); st_s = stability(o_s, samples)
        result[name] = {"N": len(samples), "v7b": {**m_v, "stab": st_v},
                        "single": {**m_s, "stab": st_s}}
        def pr(tag, m, st):
            rb = "  ".join(f"{k}:{v['recall']*100:4.1f}%(n{v['n']})" for k, v in m["rec_by_dist"].items())
            print(f"  [{tag:7s}] prec={m['prec']*100:5.1f}% rec={m['rec']*100:5.1f}% F1={m['f1']*100:5.1f}%"
                  f"  distMAE={m.get('dist_mae', float('nan')):.3f}m")
            print(f"           rec@dist: {rb}")
            print(f"           stab: mean|Δobj|={st['mean_dobj']:.4f}  flip@0.3={st['flip_rate']*100:.2f}% (pairs {st['n_pairs']})")
        pr("v7b", m_v, st_v)
        pr("single", m_s, st_s)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
