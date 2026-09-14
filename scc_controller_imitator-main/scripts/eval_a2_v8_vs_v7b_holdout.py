"""a_2 v8_balanced275k  vs  a2_v7b (배포)  held-out 비교 (cache-only, dataset stratify).

held-out 정의:
  train(train_a2_v5_from_cache.py)과 동일한 ConcatDataset + random_split(seed)으로
  val subset 의 global index 를 재현 → dataset 별로 분해.
  => 학습에 안 들어간 동일 샘플로 두 ckpt 평가 (leakage 없음).

metric (per-token, obj=True 위치 회귀):
  obj prec/recall/F1 (thr 0.5)
  dist MAE/RMSE
  vL_rel corr (Pearson), vY_rel corr  ← 핵심: 상대속도 인지
  거리별 recall (0-10/10-20/20-30/30-60 m)
  dataset stratify: 특히 spd(speedway)·pertO(close-lead) vs DriverB.

두 ckpt 모두 A2Model(d_out=128, 7ch) — run_v7b 와 동일 forward.

사용:
  python scripts/eval_a2_v8_vs_v7b_holdout.py \
    --v7b data/ckpts/a2_v7b_d128_alpha075_best_loss.pt \
    --v8  outputs_multi/a2/a2_v8_balanced275k/best_loss.pt \
    --val-ratio 0.05 --seed 0
"""
import argparse, glob, re, sys, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import ConcatDataset, random_split
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model.image_models.a2_model import A2Model
from src.model.image_models.iy_model import yaw_shift_tokens_batch
from scripts.train_a2_v5_from_cache import A2CacheDataset

# GT 7ch layout = [obj, vL_abs, vY_abs, vL_rel, vY_rel, prox, dist]
# model reg(1:7) 6-vec = [vL_abs, vY_abs, vL_rel, vY_rel, prox, dist] → vL_rel=2, vY_rel=3, dist=5
AUX_OBJ = 0
GT_VL_REL, GT_VY_REL, GT_DIST = 3, 4, 6
DIST_BUCKETS = [(0, 10), (10, 20), (20, 30), (30, 60)]
SCALE_V, SCALE_DIST = 20.0, 40.0


D = "/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data"
SPECS = [
    ("sangam_lead_0520", f"{D}/cache/i1_tok_cache_sangam_lead_0520_prev20.npz", f"{D}/aux_targets_sangam_lead_0520", None),
    ("sangam_lead_0521", f"{D}/cache/i1_tok_cache_sangam_lead_0521_prev20.npz", f"{D}/aux_targets_sangam_lead_0521", None),
    ("DriverB_new",      f"{D}/cache/i1_tok_cache_DriverB_new_prev20.npz",      f"{D}/aux_targets_DriverB_new",      None),
    ("spd",              f"{D}/cache/i1_tok_cache_spd_v2_prev20.npz",           f"{D}/aux_targets_spd_nolead_0520",  None),
    ("pertO",            f"{D}/cache/i1_tok_cache_sangam_pertO_v2_prev20.npz",  f"{D}/aux_targets_sangam_fx640",     "pertO"),
    ("pertX",            f"{D}/cache/i1_tok_cache_sangam_pertX_v2_prev20.npz",  f"{D}/aux_targets_sangam_fx640",     "pertX"),
]


@torch.no_grad()
def run_model(model, ds, idx_list, dev, bs=512):
    """ds(A2CacheDataset) 의 idx_list 샘플에 대해 obj_prob, reg_denorm(6), gt(40,7) 반환."""
    objs, regs, gts = [], [], []
    proj = model.encoder.proj
    for s in range(0, len(idx_list), bs):
        chunk = idx_list[s:s+bs]
        tc = torch.stack([torch.from_numpy(ds.tokens[ds.samples[i][0]].astype(np.float32)) for i in chunk]).to(dev)
        tp = torch.stack([torch.from_numpy(ds.tokens[ds.prev_idx[i]].astype(np.float32)) for i in chunk]).to(dev)
        gt = np.stack([np.load(ds.samples[i][1])["target"].astype(np.float32) for i in chunk])  # (B,40,7)
        ego = torch.zeros(tc.size(0), 2, device=dev)
        dy = torch.zeros(tc.size(0), device=dev)
        fx = torch.full((tc.size(0),), 640.0, device=dev)
        tp_rot = yaw_shift_tokens_batch(tp, dy, fx=fx)
        tcp = proj(tc.transpose(1, 2)).transpose(1, 2)
        tpp = proj(tp_rot.transpose(1, 2)).transpose(1, 2)
        fi = model.fusion(tcp, tpp, ego)
        logits = model.head(model.fc_proj(fi))   # (B,40,7)
        objs.append(torch.sigmoid(logits[..., 0]).cpu().numpy())
        # denorm reg: scale = [v,v,v,v,1,dist]
        reg = logits[..., 1:7].cpu().numpy()
        reg = reg * np.array([SCALE_V, SCALE_V, SCALE_V, SCALE_V, 1.0, SCALE_DIST], dtype=np.float32)
        regs.append(reg)
        gts.append(gt)
    return np.concatenate(objs), np.concatenate(regs), np.concatenate(gts)


def corr(a, b):
    if a.size < 3 or np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def metrics(obj_prob, reg, gt, thr=0.5):
    gt_obj = gt[..., AUX_OBJ] > 0.5
    pred_obj = obj_prob > thr
    tp = int((pred_obj & gt_obj).sum()); fp = int((pred_obj & ~gt_obj).sum())
    fn = int((~pred_obj & gt_obj).sum()); tn = int((~pred_obj & ~gt_obj).sum())
    prec = tp/(tp+fp) if (tp+fp) else 0.0
    rec = tp/(tp+fn) if (tp+fn) else 0.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    out = {"prec": prec, "rec": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
           "gt_pos_rate": float(gt_obj.mean())}
    m = gt_obj
    if m.sum() > 0:
        # reg idx: vL_rel=2, vY_rel=3, dist=5 (in 6-vec)
        d_pred = reg[..., 5][m]; d_gt = gt[..., GT_DIST][m]
        out["dist_mae"] = float(np.abs(d_pred - d_gt).mean())
        out["dist_rmse"] = float(np.sqrt(((d_pred - d_gt) ** 2).mean()))
        vlp = reg[..., 2][m]; vlg = gt[..., GT_VL_REL][m]
        vyp = reg[..., 3][m]; vyg = gt[..., GT_VY_REL][m]
        out["vL_rel_corr"] = corr(vlp, vlg)
        out["vY_rel_corr"] = corr(vyp, vyg)
        out["vL_rel_mae"] = float(np.abs(vlp - vlg).mean())
        out["vL_rel_pred_std"] = float(np.std(vlp))
        out["vL_rel_gt_std"] = float(np.std(vlg))
        # 거리별 recall
        out["rec_by_dist"] = {}
        gt_dist = gt[..., GT_DIST]
        for lo, hi in DIST_BUCKETS:
            mm = gt_obj & (gt_dist >= lo) & (gt_dist < hi)
            nb = int(mm.sum())
            out["rec_by_dist"][f"{lo}-{hi}"] = {
                "recall": float((pred_obj & mm).sum())/nb if nb else float("nan"), "n": nb}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v7b", default="data/ckpts/a2_v7b_d128_alpha075_best_loss.pt")
    ap.add_argument("--v8", required=True)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="/tmp/a2_v8_vs_v7b_holdout.json")
    args = ap.parse_args()
    dev = args.device

    # build datasets (train 과 동일 순서/필터)
    ds_list = []
    for name, c, a, rf in SPECS:
        ds_list.append(A2CacheDataset(c, a, run_filter=rf, fx=640.0))
    full = ConcatDataset(ds_list)
    N = len(full)
    n_val = int(N * args.val_ratio)
    n_tr = N - n_val
    # train 과 동일 split (seed)
    tr, va = random_split(full, [n_tr, n_val], generator=torch.Generator().manual_seed(args.seed))
    val_global = set(va.indices)
    print(f"[holdout] total N={N:,}  val(held-out)={len(val_global):,}", flush=True)

    # global idx -> (ds_i, local_idx)
    offsets = np.cumsum([0] + [len(d) for d in ds_list])
    per_ds_val = {i: [] for i in range(len(ds_list))}
    for g in val_global:
        ds_i = int(np.searchsorted(offsets, g, side="right") - 1)
        per_ds_val[ds_i].append(g - offsets[ds_i])

    # load both models
    def load(p):
        m = A2Model(d_token=256, d_out=128).to(dev).eval()
        sd = torch.load(p, map_location=dev, weights_only=False)
        m.load_state_dict(sd["model"] if isinstance(sd, dict) and "model" in sd else sd)
        return m
    v7b = load(args.v7b)
    v8 = load(args.v8)

    result = {}
    for ds_i, (name, c, a, rf) in enumerate(SPECS):
        local_idx = sorted(per_ds_val[ds_i])
        if not local_idx:
            continue
        ds = ds_list[ds_i]
        o7, r7, gt = run_model(v7b, ds, local_idx, dev)
        o8, r8, _  = run_model(v8, ds, local_idx, dev)
        m7 = metrics(o7, r7, gt)
        m8 = metrics(o8, r8, gt)
        result[name] = {"N": len(local_idx), "v7b": m7, "v8": m8}
        print(f"\n##### {name}  held-out N={len(local_idx):,}  gt_pos={m7.get('gt_pos_rate',0)*100:.2f}% #####", flush=True)
        for tag, m in [("v7b", m7), ("v8 ", m8)]:
            print(f"  [{tag}] prec={m['prec']*100:5.1f}% rec={m['rec']*100:5.1f}% F1={m['f1']*100:5.1f}%"
                  f"  distMAE={m.get('dist_mae',float('nan')):.2f}m"
                  f"  vL_rel:corr={m.get('vL_rel_corr',float('nan')):.3f} mae={m.get('vL_rel_mae',float('nan')):.2f}"
                  f"(predstd={m.get('vL_rel_pred_std',float('nan')):.2f}/gtstd={m.get('vL_rel_gt_std',float('nan')):.2f})"
                  f"  vY_rel:corr={m.get('vY_rel_corr',float('nan')):.3f}", flush=True)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
