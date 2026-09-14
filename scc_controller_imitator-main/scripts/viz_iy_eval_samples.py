"""IyModel inference 결과 검토 — 다양한 sample 의 GT vs pred 종합.

각 sample:
  - ROI image + obj_prob bar overlay (gate 학습 시)
  - GT vs pred steer / ax curves (40-step)
  - 통계: steer/ax MAE, max error
  - close-lead dist + ego vx 표기

사용:
  python scripts/viz_iy_eval_samples.py \
      --prebake .../f1_prebake_iy_spd_streaming.npz \
      --i1-cache .../i1_tok_cache_spd_v2.npz \
      --ckpt outputs_multi/iy_spd_streaming_objgate_close/best_loss.pt \
      --close-meta .../close_lead_meta_v3.npz \
      --raw-base .../raw_cache/speedway_track/0519_agent1_pertX_pitX \
      --n 24 --mode mixed \
      --out outputs_multi/viz_resynth/eval_samples
"""
from __future__ import annotations
import os, sys, argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import timm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.model.image_models.iy_model import IyModel

ROI_Y0, ROI_Y1 = 235, 485
ROI_H, IMG_W = ROI_Y1 - ROI_Y0, 1280
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_key(k):
    r, t = k.rsplit("__t", 1); return r, int(t)


def load_roi(raw_base, run, tick):
    p = raw_base / run / "CAMERA_1" / f"{tick}_Intensity.jpeg"
    if not p.exists(): return None
    im = np.array(Image.open(p).convert("RGB"))
    return im[ROI_Y0:ROI_Y1].copy()


def normalize(roi):
    x = roi.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prebake", required=True)
    ap.add_argument("--i1-cache", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--close-meta", default=None)
    ap.add_argument("--raw-base", required=True, type=Path)
    ap.add_argument("--seg-ckpt", default="data/ckpts/seg_pretrain_v2s_full_best_backbone.pt")
    ap.add_argument("--fx", type=float, default=640.0)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--num-layers", type=int, default=6)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--mode", choices=["close", "far", "mixed", "random", "worst"],
                    default="mixed", help="sample 선별 전략")
    ap.add_argument("--close-thr", type=float, default=30.0)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # backbone
    cm = np.load(args.i1_cache, allow_pickle=True)
    cmeta = json.loads(str(cm["meta"]))
    backbone_name = cmeta["backbone"]; C_b = int(cmeta["C_b"])
    backbone = timm.create_model(backbone_name, pretrained=False, features_only=True, out_indices=[-1])
    if Path(args.seg_ckpt).exists():
        sd = torch.load(args.seg_ckpt, map_location="cpu", weights_only=False)
        own = backbone.state_dict(); n = 0
        for k, v in sd.items():
            k2 = k.replace("backbone.", "")
            if k2 in own and own[k2].shape == v.shape: own[k2] = v; n += 1
        backbone.load_state_dict(own); print(f"  seg backbone: {n} keys")
    backbone = backbone.to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)

    # IyModel + obj_gate auto-detect
    _sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(_sd, dict) and "model" in _sd: _sd = _sd["model"]
    has_obj = any("obj_head" in k for k in _sd.keys())
    print(f"  ckpt has obj_head: {has_obj}")
    model = IyModel(C_b=C_b, d_model=args.d_model, nhead=args.nhead,
                     num_layers=args.num_layers, fx=args.fx,
                     use_obj_gate=has_obj).to(device).eval()
    miss, unex = model.load_state_dict(_sd, strict=False)
    print(f"  ckpt: missing={len(miss)} unexpected={len(unex)}")

    # prebake
    z = np.load(args.prebake, allow_pickle=True)
    curr_keys = z["curr_keys"].tolist(); prev_keys = z["prev_keys"].tolist()
    d_yaws = z["d_yaws"]; ego_5d = z["ego_5d"]
    traj_all = z["traj"]; target_all = z["target"]; mask_all = z["mask"]

    cl_map = {}
    if args.close_meta:
        cz = np.load(args.close_meta, allow_pickle=True)
        cl_map = dict(zip(cz["keys"].tolist(), cz["dists"]))

    # sample 선별
    rng = np.random.RandomState(args.seed)
    N = len(curr_keys)
    valid = [i for i in range(N) if mask_all[i].any()]
    if args.mode == "close":
        cand = [i for i in valid if cl_map.get(curr_keys[i], np.inf) < args.close_thr]
        pick = rng.choice(cand, size=min(args.n, len(cand)), replace=False).tolist()
    elif args.mode == "far":
        cand = [i for i in valid if cl_map.get(curr_keys[i], np.inf) >= args.close_thr]
        pick = rng.choice(cand, size=min(args.n, len(cand)), replace=False).tolist()
    elif args.mode == "mixed":
        half = args.n // 2
        cand_c = [i for i in valid if cl_map.get(curr_keys[i], np.inf) < args.close_thr]
        cand_f = [i for i in valid if cl_map.get(curr_keys[i], np.inf) >= args.close_thr]
        pick = rng.choice(cand_c, size=min(half, len(cand_c)), replace=False).tolist() + \
               rng.choice(cand_f, size=min(args.n - half, len(cand_f)), replace=False).tolist()
    elif args.mode == "random":
        pick = rng.choice(valid, size=min(args.n, len(valid)), replace=False).tolist()
    else:  # worst — needs full sweep, skipped for speed
        pick = rng.choice(valid, size=min(args.n, len(valid)), replace=False).tolist()
    pick.sort()
    print(f"  picked {len(pick)} samples ({args.mode})")

    obj_capture = {}
    if has_obj:
        def hook_obj(_m, _i, o): obj_capture["logit"] = o.detach().squeeze(-1)
        model.obj_head.register_forward_hook(hook_obj)

    rec = []
    for ti, idx in enumerate(pick):
        ck = curr_keys[idx]; pk = prev_keys[idx]
        run_c, tk_c = parse_key(ck); run_p, tk_p = parse_key(pk)
        roi_c = load_roi(args.raw_base, run_c, tk_c)
        roi_p = load_roi(args.raw_base, run_p, tk_p)
        if roi_c is None or roi_p is None:
            print(f"  [skip] image miss {ck}"); continue
        ic = normalize(roi_c).unsqueeze(0).to(device)
        ip = normalize(roi_p).unsqueeze(0).to(device)
        with torch.no_grad():
            fc = backbone(ic)[-1]
            tok_c = F.adaptive_avg_pool2d(fc, (1, 40)).squeeze(2).transpose(1, 2)
            fp = backbone(ip)[-1]
            tok_p = F.adaptive_avg_pool2d(fp, (1, 40)).squeeze(2).transpose(1, 2)
            e5 = torch.from_numpy(ego_5d[idx]).unsqueeze(0).to(device)
            t40 = torch.from_numpy(traj_all[idx]).unsqueeze(0).to(device)
            d_y = torch.tensor([float(d_yaws[idx])], device=device)
            fx_t = torch.tensor([args.fx], device=device)
            out = model({"tok_curr": tok_c, "tok_prev_raw": tok_p, "d_yaw": d_y,
                          "ego_5d": e5, "traj_40": t40, "fx": fx_t})
            pred = out[0] if isinstance(out, tuple) else out
        pred_np = pred[0].cpu().numpy()       # (40, 2)
        gt = target_all[idx]                  # (40, 2)
        m  = mask_all[idx]                    # (40,)
        st_mae = np.abs(pred_np[m, 0] - gt[m, 0]).mean()
        ax_mae = np.abs(pred_np[m, 1] - gt[m, 1]).mean()
        st_max = np.abs(pred_np[m, 0] - gt[m, 0]).max()
        ax_max = np.abs(pred_np[m, 1] - gt[m, 1]).max()
        obj_prob = torch.sigmoid(obj_capture["logit"][0]).cpu().numpy() if has_obj else None
        rec.append(dict(
            roi=roi_c, ck=ck, pred=pred_np, gt=gt, mask=m,
            dist=cl_map.get(ck, float("inf")), vx=float(ego_5d[idx, 0]),
            st_mae=st_mae, ax_mae=ax_mae, st_max=st_max, ax_max=ax_max,
            obj_prob=obj_prob,
        ))
        print(f"  [{ti+1}/{len(pick)}] {ck}  dist={rec[-1]['dist']:.1f}  vx={rec[-1]['vx']:.1f}  "
              f"st_mae={st_mae:.3f}° ax_mae={ax_mae:.3f} st_max={st_max:.3f} ax_max={ax_max:.3f}")

    # per-sample plot
    bin_w = IMG_W / 40
    for ri, r in enumerate(rec):
        fig, axes = plt.subplots(3, 1, figsize=(15, 7),
                                  gridspec_kw=dict(height_ratios=[1, 0.7, 0.7]))
        axes[0].imshow(r["roi"])
        if r["obj_prob"] is not None:
            for b in range(40):
                v = float(r["obj_prob"][b])
                axes[0].axvspan(b*bin_w, (b+1)*bin_w, ymin=0, ymax=v*0.35,
                                color="lime", alpha=0.45)
            axes[0].text(20, 25, f"obj_prob bar (max={r['obj_prob'].max():.2f})",
                         color="lime", fontsize=10, fontweight="bold",
                         bbox=dict(facecolor="black", alpha=0.5, pad=2))
        tag = "CLOSE" if r["dist"] < 30 else ("FAR" if np.isfinite(r["dist"]) else "NO-LEAD")
        axes[0].set_title(f"[{tag}] {r['ck']}  dist={r['dist']:.1f}m  vx={r['vx']:.1f}m/s  "
                          f"|  steer mae={r['st_mae']:.3f}° (max {r['st_max']:.2f})  "
                          f"ax mae={r['ax_mae']:.3f} (max {r['ax_max']:.2f})", fontsize=10)
        axes[0].axis("off")
        tt = np.arange(40) * 0.1
        axes[1].plot(tt, r["gt"][:, 0], "k--", lw=1.4, label="GT steer (deg)")
        axes[1].plot(tt, r["pred"][:, 0], "r-",  lw=1.6, label="pred steer")
        axes[1].set_xlim(0, 4); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=9, loc="upper right")
        axes[1].set_xlabel("t (s)"); axes[1].set_ylabel("steer (deg)")
        axes[2].plot(tt, r["gt"][:, 1], "k--", lw=1.4, label="GT ax (m/s²)")
        axes[2].plot(tt, r["pred"][:, 1], "b-",  lw=1.6, label="pred ax")
        axes[2].set_xlim(0, 4); axes[2].grid(alpha=0.3); axes[2].legend(fontsize=9, loc="upper right")
        axes[2].set_xlabel("t (s)"); axes[2].set_ylabel("ax (m/s²)")
        fname = f"{ri:02d}_{tag}_{r['ck'].replace('__','_')}.png"
        plt.tight_layout(); plt.savefig(args.out / fname, dpi=110); plt.close()

    # summary: error distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].hist([r["st_mae"] for r in rec], bins=20, color="crimson", alpha=0.7)
    axes[0].set_xlabel("steer MAE (deg)"); axes[0].set_ylabel("count")
    axes[0].set_title(f"steer MAE — mean={np.mean([r['st_mae'] for r in rec]):.3f}°")
    axes[0].grid(alpha=0.3)
    axes[1].hist([r["ax_mae"] for r in rec], bins=20, color="steelblue", alpha=0.7)
    axes[1].set_xlabel("ax MAE (m/s²)"); axes[1].set_ylabel("count")
    axes[1].set_title(f"ax MAE — mean={np.mean([r['ax_mae'] for r in rec]):.3f}m/s²")
    axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(args.out / "summary_mae.png", dpi=120); plt.close()

    # html
    html = args.out / "index.html"
    with open(html, "w") as f:
        f.write("<!doctype html><html><head><meta charset=utf-8>")
        f.write(f"<title>IyModel inference samples — {args.mode}</title>")
        f.write("<style>body{font-family:sans-serif;background:#1c1c1c;color:#ddd;margin:12px}"
                ".card{background:#2a2a2a;padding:8px;border-radius:6px;margin-bottom:14px}"
                ".card img{width:100%;display:block}h2{color:#fff}"
                ".close{border-left:4px solid #E74C3C}.far{border-left:4px solid #5D6D7E}"
                ".no_lead{border-left:4px solid #566573}</style>")
        f.write("</head><body>")
        f.write(f"<h2>IyModel inference samples — mode={args.mode}  ckpt={Path(args.ckpt).parent.name}</h2>")
        f.write("<p>각 sample: ROI image (위) + obj_prob bar (gate 학습 시) | GT 검정점선 vs pred 빨강 (steer) / 파랑 (ax). "
                f"전체 평균 MAE: steer={np.mean([r['st_mae'] for r in rec]):.3f}° "
                f"ax={np.mean([r['ax_mae'] for r in rec]):.3f}m/s²</p>")
        f.write("<div class=card><img src='summary_mae.png'></div>")
        # sort by tag (CLOSE, FAR, NO-LEAD)
        for r_i, r in enumerate(rec):
            tag = "CLOSE" if r["dist"] < 30 else ("FAR" if np.isfinite(r["dist"]) else "NO-LEAD")
            css = "close" if tag == "CLOSE" else ("far" if tag == "FAR" else "no_lead")
            fname = f"{r_i:02d}_{tag}_{r['ck'].replace('__','_')}.png"
            f.write(f"<div class='card {css}'><div style='font-size:12px;color:#aaa'>"
                    f"[{tag}] {r['ck']} | dist={r['dist']:.1f}m | vx={r['vx']:.1f}m/s "
                    f"| st_mae={r['st_mae']:.3f}° ax_mae={r['ax_mae']:.3f}m/s²</div>")
            f.write(f"<img src='{fname}' loading=lazy></div>")
        f.write("</body></html>")
    print(f"  index → {html}")


if __name__ == "__main__":
    main()
