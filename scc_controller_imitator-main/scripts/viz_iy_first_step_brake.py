"""close-lead approach sample 에서 모델의 first-step ax (제어 첫 명령) brake 정확도 검증.

대상: val set 의 close-lead (dist<30m) 중 GT first-step ax < 0 (brake 결정)
체크:
  1. sign-match: GT < 0 인데 pred 도 < 0 (브레이크 결정 여부)
  2. magnitude: GT vs pred scatter (clip 후 correlation)
  3. strong brake detection (GT < -3 → pred < -1)
  4. 가짜 (GT > 0 인데 pred < 0) false brake

출력: outputs_multi/viz_resynth/first_step_brake/
"""
from __future__ import annotations
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import timm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model.image_models.iy_model import IyModel

DATA = Path("/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data")
RAW = Path("/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/raw_cache/speedway_track/0519_agent1_pertX_pitX")
ROI_Y0, ROI_Y1 = 235, 485
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_key(k): r, t = k.rsplit("__t", 1); return r, int(t)


def load_roi(run, tick):
    p = RAW / run / "CAMERA_1" / f"{tick}_Intensity.jpeg"
    if not p.exists(): return None
    im = np.array(Image.open(p).convert("RGB"))
    return im[ROI_Y0:ROI_Y1].copy()


def normalize(roi):
    x = roi.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prebake", default=str(DATA / "prebake/f1_prebake_iy_spd_streaming.npz"))
    ap.add_argument("--i1-cache", default=str(DATA / "cache/i1_tok_cache_spd_v2.npz"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--close-meta", default=str(DATA / "cache/close_lead_meta_v4.npz"))
    ap.add_argument("--seg-ckpt", default=None)  # None → registry 사용
    ap.add_argument("--out", default="outputs_multi/viz_resynth/first_step_brake")
    ap.add_argument("--n-eval", type=int, default=500, help="평가할 sample 수")
    ap.add_argument("--worst-show", type=int, default=12, help="worst case 시각화 sample 수")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # seg backbone (registry)
    if args.seg_ckpt is None:
        from src.utils.ckpt_registry import get_best
        b = get_best("seg_backbone")
        args.seg_ckpt = b["path"] if b else "data/ckpts/seg_pretrain_v2s_full_best_backbone.pt"
    print(f"[brake-eval] seg_ckpt: {args.seg_ckpt}")

    cm = np.load(args.i1_cache, allow_pickle=True)
    cmeta = json.loads(str(cm["meta"]))
    C_b = int(cmeta["C_b"])
    print(f"[brake-eval] cache backbone: {cmeta.get('seg_ckpt', '?')}")
    if cmeta.get("seg_ckpt") != args.seg_ckpt:
        print(f"⚠️ cache seg_ckpt 와 live seg_ckpt 불일치 — token distribution mismatch 위험")

    backbone = timm.create_model(cmeta["backbone"], pretrained=False, features_only=True, out_indices=[-1])
    sd = torch.load(args.seg_ckpt, map_location="cpu", weights_only=False)
    own = backbone.state_dict()
    fixed = {k.replace("backbone.", ""): v for k, v in sd.items()
             if k.replace("backbone.", "") in own and own[k.replace("backbone.", "")].shape == v.shape}
    backbone.load_state_dict(fixed, strict=False)
    backbone = backbone.to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)
    print(f"  seg backbone preload: {len(fixed)} keys")

    _sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(_sd, dict) and "model" in _sd: _sd = _sd["model"]
    has_obj = any("obj_head" in k for k in _sd.keys())
    model = IyModel(C_b=C_b, d_model=128, nhead=8, num_layers=6, fx=640.0,
                     use_obj_gate=has_obj).to(device).eval()
    miss, unex = model.load_state_dict(_sd, strict=False)
    print(f"  IyModel ckpt: missing={len(miss)} unexpected={len(unex)}")

    # prebake + close_meta
    z = np.load(args.prebake, allow_pickle=True)
    curr_keys = z["curr_keys"].tolist(); prev_keys = z["prev_keys"].tolist()
    d_yaws = z["d_yaws"]; ego_5d = z["ego_5d"]
    traj_all = z["traj"]; target_all = z["target"]; mask_all = z["mask"]
    cz = np.load(args.close_meta, allow_pickle=True)
    cl_map = dict(zip(cz["keys"].tolist(), cz["dists"]))

    # close-lead 만 sampling
    rng = np.random.RandomState(args.seed)
    cand = [i for i in range(len(curr_keys))
            if mask_all[i].any() and cl_map.get(curr_keys[i], np.inf) < 30]
    print(f"[brake-eval] close-lead candidates: {len(cand)}")
    pick = rng.choice(cand, size=min(args.n_eval, len(cand)), replace=False).tolist()
    print(f"[brake-eval] evaluating {len(pick)} samples")

    # inference + 첫 step (t=0~5) ax 모음
    rec = []
    for i, idx in enumerate(pick):
        ck = curr_keys[idx]; pk = prev_keys[idx]
        run_c, tk_c = parse_key(ck); run_p, tk_p = parse_key(pk)
        roi_c = load_roi(run_c, tk_c); roi_p = load_roi(run_p, tk_p)
        if roi_c is None or roi_p is None: continue
        ic = normalize(roi_c).unsqueeze(0).to(device)
        ip = normalize(roi_p).unsqueeze(0).to(device)
        with torch.no_grad():
            fc = F.adaptive_avg_pool2d(backbone(ic)[-1], (1, 40)).squeeze(2).transpose(1, 2)
            fp = F.adaptive_avg_pool2d(backbone(ip)[-1], (1, 40)).squeeze(2).transpose(1, 2)
            e5 = torch.from_numpy(ego_5d[idx]).unsqueeze(0).to(device)
            t40 = torch.from_numpy(traj_all[idx]).unsqueeze(0).to(device)
            d_y = torch.tensor([float(d_yaws[idx])], device=device)
            fx_t = torch.tensor([640.0], device=device)
            out = model({"tok_curr": fc, "tok_prev_raw": fp, "d_yaw": d_y,
                          "ego_5d": e5, "traj_40": t40, "fx": fx_t})
            pred = out[0] if isinstance(out, tuple) else out
        pred_ax_0 = float(pred[0, 0, 1].cpu())          # first step
        pred_ax_avg5 = float(pred[0, :5, 1].mean().cpu())
        gt_ax_0 = float(target_all[idx, 0, 1])
        gt_ax_avg5 = float(target_all[idx, :5, 1].mean())
        rec.append(dict(
            ck=ck, dist=cl_map[ck], vx=float(ego_5d[idx, 0]),
            gt_ax_0=gt_ax_0, gt_ax_avg5=gt_ax_avg5,
            pred_ax_0=pred_ax_0, pred_ax_avg5=pred_ax_avg5,
            roi=roi_c,
        ))
        if (i+1) % 50 == 0: print(f"  [{i+1}/{len(pick)}]")

    gt_a = np.array([r["gt_ax_0"] for r in rec])
    pr_a = np.array([r["pred_ax_0"] for r in rec])
    gt_5 = np.array([r["gt_ax_avg5"] for r in rec])
    pr_5 = np.array([r["pred_ax_avg5"] for r in rec])

    # 통계
    print()
    print(f"=== first-step ax (t=0, 즉시 제어 명령) ===")
    print(f"  GT mean={gt_a.mean():+.2f}  std={gt_a.std():.2f}  range[{gt_a.min():+.1f}, {gt_a.max():+.1f}]")
    print(f"  pred mean={pr_a.mean():+.2f} std={pr_a.std():.2f} range[{pr_a.min():+.1f}, {pr_a.max():+.1f}]")
    print(f"  MAE={np.abs(gt_a-pr_a).mean():.2f}")
    # sign match
    gt_brake = gt_a < 0; pr_brake = pr_a < 0
    tp = (gt_brake & pr_brake).sum(); fn = (gt_brake & ~pr_brake).sum()
    fp = (~gt_brake & pr_brake).sum(); tn = (~gt_brake & ~pr_brake).sum()
    n = len(gt_a)
    print(f"  brake sign-match: {(tp+tn)/n*100:.1f}%  TP={tp} FN={fn} FP={fp} TN={tn}")
    if tp+fn > 0: print(f"  brake recall (GT brake → pred brake): {tp/(tp+fn)*100:.1f}%")
    if tp+fp > 0: print(f"  brake precision (pred brake → GT brake): {tp/(tp+fp)*100:.1f}%")
    # strong brake
    strong = gt_a < -3.0
    if strong.sum() > 0:
        strong_correct = (gt_a[strong] < -1).sum() if False else (pr_a[strong] < -1).sum()
        print(f"  strong brake (GT < -3): n={strong.sum()}  pred < -1 detection: {strong_correct/strong.sum()*100:.1f}%")

    # 0~5 step avg
    print(f"\n=== first 5-step ax avg (t=0~0.5s) ===")
    print(f"  GT mean={gt_5.mean():+.2f}  pred mean={pr_5.mean():+.2f}  MAE={np.abs(gt_5-pr_5).mean():.2f}")

    # plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    # (a) scatter GT vs pred first-step
    ax = axes[0, 0]
    sc = ax.scatter(gt_a, pr_a, c=np.array([r["dist"] for r in rec]), cmap="viridis", s=12, alpha=0.6)
    lim = max(abs(gt_a).max(), abs(pr_a).max(), 5)
    ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.5)
    ax.axhline(0, color="red", alpha=0.3); ax.axvline(0, color="red", alpha=0.3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.grid(alpha=0.3)
    ax.set_xlabel("GT first-step ax (m/s²)"); ax.set_ylabel("pred first-step ax (m/s²)")
    ax.set_title(f"GT vs pred (t=0, n={len(rec)})\nsign-match {(tp+tn)/n*100:.1f}%")
    plt.colorbar(sc, ax=ax, label="dist (m)")
    # (b) histogram (GT brake 영역만)
    ax = axes[0, 1]
    ax.hist(gt_a, bins=40, alpha=0.5, label="GT", color="black")
    ax.hist(pr_a, bins=40, alpha=0.5, label="pred", color="red")
    ax.axvline(0, color="blue", linestyle="--", alpha=0.5)
    ax.set_xlabel("first-step ax (m/s²)"); ax.set_ylabel("count")
    ax.set_title("ax 분포 비교 — pred 가 brake 영역 잘 cover 하는지")
    ax.legend(); ax.grid(alpha=0.3)
    # (c) GT-pred error vs dist
    ax = axes[1, 0]
    err = pr_a - gt_a
    sc = ax.scatter([r["dist"] for r in rec], err, c=gt_a, cmap="coolwarm", s=12, alpha=0.6, vmin=-10, vmax=10)
    ax.axhline(0, color="black", alpha=0.5)
    ax.set_xlabel("close-lead dist (m)"); ax.set_ylabel("pred - GT  first-step ax (m/s²)")
    ax.set_title("dist 별 error — 가까울수록 정확한지")
    ax.grid(alpha=0.3); plt.colorbar(sc, ax=ax, label="GT ax")
    # (d) first 5-step avg scatter
    ax = axes[1, 1]
    ax.scatter(gt_5, pr_5, s=12, alpha=0.5, color="#16A085")
    lim = max(abs(gt_5).max(), abs(pr_5).max(), 5)
    ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.5)
    ax.axhline(0, color="red", alpha=0.3); ax.axvline(0, color="red", alpha=0.3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.grid(alpha=0.3)
    ax.set_xlabel("GT 0~0.5s ax avg"); ax.set_ylabel("pred 0~0.5s ax avg")
    ax.set_title(f"first 5-step avg (smoother)  MAE={np.abs(gt_5-pr_5).mean():.2f}")
    fig.suptitle(f"close-lead first-step brake validation  ckpt={Path(args.ckpt).parent.name}",
                  fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig(out_dir / "summary.png", dpi=110); plt.close()

    # worst: GT brake (< -1) 인데 pred ≥ -0.5 (brake 약함)
    worst = sorted(rec, key=lambda r: r["pred_ax_0"] - r["gt_ax_0"])[-args.worst_show:]
    fig, axes = plt.subplots(len(worst), 1, figsize=(12, len(worst)*2.5))
    if len(worst) == 1: axes = [axes]
    for ax, r in zip(axes, worst):
        ax.imshow(r["roi"])
        ax.set_title(f"GT ax_0={r['gt_ax_0']:+.2f}  pred ax_0={r['pred_ax_0']:+.2f}  "
                      f"err={r['pred_ax_0']-r['gt_ax_0']:+.2f}  dist={r['dist']:.0f}m vx={r['vx']:.0f}",
                      fontsize=10, color="red", fontweight="bold")
        ax.axis("off")
    fig.suptitle("WORST: GT brake 인데 pred 약함 (brake 안 됨)", fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.savefig(out_dir / "worst_underbrake.png", dpi=100); plt.close()

    # html
    html = out_dir / "index.html"
    with open(html, "w") as f:
        f.write("<!doctype html><html><head><meta charset=utf-8>")
        f.write("<title>close-lead first-step brake validation</title>")
        f.write("<style>body{font-family:sans-serif;background:#1c1c1c;color:#ddd;margin:12px}img{width:100%}.card{background:#2a2a2a;padding:8px;border-radius:6px;margin-bottom:14px}h2{color:#fff}</style>")
        f.write("</head><body>")
        f.write(f"<h2>close-lead (dist&lt;30m) first-step ax 검증 — n={len(rec)}</h2>")
        f.write(f"<p>ckpt: {Path(args.ckpt).parent.name}<br>MAE={np.abs(gt_a-pr_a).mean():.3f} m/s² (t=0)<br>"
                f"brake sign-match {(tp+tn)/n*100:.1f}% (TP {tp} / FN {fn} / FP {fp} / TN {tn})</p>")
        f.write("<div class=card><img src=summary.png></div>")
        f.write("<h2>WORST under-brake (pred 가 GT brake 따라가지 못함)</h2>")
        f.write("<div class=card><img src=worst_underbrake.png></div>")
        f.write("</body></html>")
    print(f"  → {html}")


if __name__ == "__main__":
    main()
