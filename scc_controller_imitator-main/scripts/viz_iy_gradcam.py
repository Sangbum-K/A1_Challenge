"""IyModel ax-target GradCAM 시각화 — 모델이 전방차량을 보는지 검증.

각 sample 에 대해:
  1. raw image curr/prev → backbone (seg preloaded) → forward activation hook
  2. IyModel forward → pred (B, 40, 2)
  3. target = max-brake(pred[:, t_win, 1]) → backward
  4. backbone last conv 의 (∂y/∂A) × A → GradCAM heatmap → image overlay

close_lead_meta_v3 에서 가까운 lead 있는 sample 만 sampling.

사용:
  python scripts/viz_iy_gradcam.py \
      --prebake .../prebake/f1_prebake_iy_spd_streaming.npz \
      --i1-cache .../cache/i1_tok_cache_spd_v2.npz \
      --ckpt outputs_multi/iy_spd_streaming_w20/best_loss.pt \
      --close-meta .../cache/close_lead_meta_v3.npz \
      --raw-base .../raw_cache/speedway_track/0519_agent1_pertX_pitX \
      --n 16 --close-thr 30 \
      --out outputs_multi/viz_resynth/gradcam_spd
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


def parse_key(k: str):
    # 260519_seg0__t0000005 → ("260519_seg0", 5)
    run, tail = k.rsplit("__t", 1)
    return run, int(tail)


def load_roi(raw_base: Path, run: str, tick: int) -> np.ndarray:
    p = raw_base / run / "CAMERA_1" / f"{tick}_Intensity.jpeg"
    if not p.exists(): return None
    im = np.array(Image.open(p).convert("RGB"))   # (720, 1280, 3)
    return im[ROI_Y0:ROI_Y1].copy()              # (250, 1280, 3)


def normalize(roi: np.ndarray) -> torch.Tensor:
    x = roi.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1))  # (3, H, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prebake", required=True)
    ap.add_argument("--i1-cache", required=True, help="cache 의 meta 에서 C_b/backbone 추출")
    ap.add_argument("--ckpt", required=True, help="IyModel ckpt (.pt)")
    ap.add_argument("--close-meta", required=True)
    ap.add_argument("--raw-base", required=True, type=Path,
                    help="raw_cache/{track}/{policy} root (run dir 의 부모)")
    ap.add_argument("--seg-ckpt", default="data/ckpts/seg_pretrain_v2s_full_best_backbone.pt")
    ap.add_argument("--fx", type=float, default=640.0)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--num-layers", type=int, default=6)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--close-thr", type=float, default=30.0)
    ap.add_argument("--t-window", type=int, default=10,
                    help="ax 대상 시간 step (1~40). target = mean over [0, t_window)")
    ap.add_argument("--use-obj-gate", action="store_true",
                    help="IyModel use_obj_gate (학습 ckpt 가 use_obj_gate 였으면 같이 켜기)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # cache meta → backbone / C_b
    cm = np.load(args.i1_cache, allow_pickle=True)
    cmeta = json.loads(str(cm["meta"]))
    backbone_name = cmeta["backbone"]; C_b = int(cmeta["C_b"])
    print(f"[viz-gradcam] backbone={backbone_name} C_b={C_b}")

    # backbone (seg preloaded)
    backbone = timm.create_model(backbone_name, pretrained=False, features_only=True, out_indices=[-1])
    if Path(args.seg_ckpt).exists():
        sd = torch.load(args.seg_ckpt, map_location="cpu", weights_only=False)
        own = backbone.state_dict(); n = 0
        for k, v in sd.items():
            k2 = k.replace("backbone.", "")
            if k2 in own and own[k2].shape == v.shape: own[k2] = v; n += 1
        backbone.load_state_dict(own)
        print(f"  seg backbone preload: {n} keys")
    backbone = backbone.to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)

    # IyModel
    model = IyModel(C_b=C_b, d_model=args.d_model, nhead=args.nhead,
                     num_layers=args.num_layers, fx=args.fx,
                     use_obj_gate=args.use_obj_gate).to(device).eval()
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
    miss, unex = model.load_state_dict(sd, strict=False)
    print(f"  ckpt: missing={len(miss)} unexpected={len(unex)}")
    for p in model.parameters(): p.requires_grad_(False)

    # prebake — curr_keys, prev_keys, d_yaws, ego_5d, traj
    z = np.load(args.prebake, allow_pickle=True)
    sample_keys = z["sample_keys"].tolist()
    curr_keys   = z["curr_keys"].tolist()
    prev_keys   = z["prev_keys"].tolist()
    d_yaws      = z["d_yaws"]
    ego_5d      = z["ego_5d"]
    traj_all    = z["traj"]
    target_all  = z["target"]
    mask_all    = z["mask"]

    # close_lead_meta 매핑 (key → dist)
    cz = np.load(args.close_meta, allow_pickle=True)
    cl_keys = cz["keys"].tolist()
    cl_dists = cz["dists"]
    cl_map = dict(zip(cl_keys, cl_dists))

    # select close-lead samples (valid curr_key + dist < thr)
    rng = np.random.RandomState(args.seed)
    cand = [i for i, ck in enumerate(curr_keys)
            if ck in cl_map and cl_map[ck] < args.close_thr and mask_all[i].any()]
    print(f"  candidates (curr in close_meta & dist<{args.close_thr}): {len(cand)}")
    if not cand:
        print("  NO close-lead candidates — fall back to random valid samples")
        cand = [i for i, ck in enumerate(curr_keys) if mask_all[i].any()][:args.n*4]
    pick = rng.choice(cand, size=min(args.n, len(cand)), replace=False).tolist()
    pick.sort()

    # 직접 fc_feat.retain_grad() 사용 — features_only timm 의 list output 으로 인한 hook 미동작 회피

    rec = []
    for n_done, idx in enumerate(pick):
        ck = curr_keys[idx]; pk = prev_keys[idx]
        run_c, tk_c = parse_key(ck); run_p, tk_p = parse_key(pk)
        roi_c = load_roi(args.raw_base, run_c, tk_c)
        roi_p = load_roi(args.raw_base, run_p, tk_p)
        if roi_c is None or roi_p is None:
            print(f"  [skip] image miss: {ck}")
            continue
        ic = normalize(roi_c).unsqueeze(0).to(device)
        ip = normalize(roi_p).unsqueeze(0).to(device)
        ic.requires_grad_(True)

        # backbone forward (with grad). features_only → list; 마지막 feat 만 사용 + retain_grad
        fc_raw = backbone(ic)
        fc_feat = fc_raw[-1] if isinstance(fc_raw, (list, tuple)) else fc_raw  # (1, C_b, H', W')
        fc_feat.retain_grad()
        fp_raw = backbone(ip)
        fp_feat = fp_raw[-1] if isinstance(fp_raw, (list, tuple)) else fp_raw
        # token (curr 만 GradCAM 대상; prev 는 forward only)
        tok_c = F.adaptive_avg_pool2d(fc_feat, (1, 40)).squeeze(2).transpose(1, 2)
        tok_p = F.adaptive_avg_pool2d(fp_feat, (1, 40)).squeeze(2).transpose(1, 2)

        # IyModel
        e5 = torch.from_numpy(ego_5d[idx]).unsqueeze(0).to(device)
        t40 = torch.from_numpy(traj_all[idx]).unsqueeze(0).to(device)
        d_y = torch.tensor([float(d_yaws[idx])], device=device)
        fx_t = torch.tensor([args.fx], device=device)
        out = model({"tok_curr": tok_c, "tok_prev_raw": tok_p, "d_yaw": d_y,
                      "ego_5d": e5, "traj_40": t40, "fx": fx_t})
        pred = out[0] if isinstance(out, tuple) else out          # (1, 40, 2)
        ax_pred = pred[0, :args.t_window, 1]                       # (t_win,)
        # target = -ax mean (max brake — 가까운 lead 보고 brake 한 정도)
        # 부호 무관, 가속도 절댓값 합도 가능 — 일단 직접 mean
        target = ax_pred.mean()

        model.zero_grad(set_to_none=True)
        target.backward()

        A = fc_feat[0].detach()              # (C_b, H', W')
        G = fc_feat.grad[0].detach()         # (C_b, H', W')
        alpha = G.mean(dim=(1, 2))           # (C_b,)
        cam = torch.relu((alpha[:, None, None] * A).sum(dim=0))   # (H', W')
        cam_np = cam.cpu().numpy()
        if cam_np.max() > 0: cam_np = cam_np / cam_np.max()
        # upsample to ROI size (250, 1280)
        cam_up = np.array(Image.fromarray((cam_np * 255).astype(np.uint8)).resize((IMG_W, ROI_H), Image.BILINEAR)) / 255.0

        # plot: ROI curr + overlay, prev, ax curve
        fig, axes = plt.subplots(3, 1, figsize=(16, 8), gridspec_kw=dict(height_ratios=[1, 1, 0.8]))
        axes[0].imshow(roi_c); axes[0].imshow(cam_up, cmap="jet", alpha=0.45)
        axes[0].set_title(f"{ck}  dist={cl_map.get(ck, np.nan):.1f}m  vx={ego_5d[idx,0]:.1f}  "
                          f"ax_pred[0:{args.t_window}].mean={target.item():+.2f}m/s²",
                          fontsize=11)
        axes[0].axis("off")
        axes[1].imshow(roi_p); axes[1].set_title(f"prev {pk}  d_yaw={d_yaws[idx]:+.3f} rad", fontsize=10)
        axes[1].axis("off")
        # ax curve: target GT vs pred
        ax_p = pred[0, :, 1].detach().cpu().numpy()
        st_p = pred[0, :, 0].detach().cpu().numpy()
        ax_g = target_all[idx, :, 1]
        st_g = target_all[idx, :, 0]
        msk  = mask_all[idx]
        tt = np.arange(40) * 0.1
        axes[2].plot(tt, ax_g, "k--", lw=1.2, label="GT ax")
        axes[2].plot(tt, ax_p, "r-",  lw=1.6, label="pred ax")
        axes[2].axvspan(0, args.t_window * 0.1, alpha=0.12, color="orange", label=f"GradCAM target (mean over t<{args.t_window*0.1:.1f}s)")
        axes[2].set_xlabel("t (s)"); axes[2].set_ylabel("ax (m/s²)")
        axes[2].set_xlim(0, 4); axes[2].grid(alpha=0.3); axes[2].legend(fontsize=9, loc="lower right")
        axes[2].set_title(f"GT vs pred  steer mae={float(np.abs(st_p-st_g)[msk].mean()):.3f}°  ax mae={float(np.abs(ax_p-ax_g)[msk].mean()):.3f}m/s²", fontsize=10)
        fname = f"{n_done:02d}_{ck.replace('__','_')}_dist{cl_map.get(ck,0):.0f}.png"
        plt.tight_layout()
        plt.savefig(args.out / fname, dpi=110)
        plt.close()
        rec.append((ck, float(cl_map.get(ck, np.nan)), float(target.item()), fname))
        print(f"  [{n_done+1}/{len(pick)}] {fname}")

    # index html
    html = args.out / "index.html"
    with open(html, "w") as f:
        f.write("<!doctype html><html><head><meta charset=utf-8>")
        f.write("<title>IyModel ax GradCAM (close-lead samples)</title>")
        f.write("<style>body{font-family:sans-serif;background:#1c1c1c;color:#ddd;margin:12px}.card{background:#2a2a2a;padding:8px;border-radius:6px;margin-bottom:14px}.card img{width:100%;display:block}h2{color:#fff}</style>")
        f.write("</head><body>")
        f.write(f"<h2>IyModel ax-target GradCAM — close-lead samples (dist<{args.close_thr}m, t_win<={args.t_window*0.1:.1f}s)</h2>")
        f.write("<p>backbone last conv (tf_efficientnetv2_s features[-1], 1×40 → adaptive_avg_pool 적용 전) 의 activation × gradient. ax mean (0~t_win) 에 대한 attribution. ")
        f.write("앞차가 가까운데 모델이 그 위치를 잘 보고 ax 결정하는지 검증.</p>")
        for ck, d, t, fname in rec:
            f.write(f"<div class=card><div style='font-size:12px;color:#aaa'>{ck} | dist={d:.1f}m | ax_target_mean={t:+.2f}m/s²</div>")
            f.write(f"<img src='{fname}' loading=lazy></div>")
        f.write("</body></html>")
    print(f"[viz-gradcam] index → {html}")


if __name__ == "__main__":
    main()
