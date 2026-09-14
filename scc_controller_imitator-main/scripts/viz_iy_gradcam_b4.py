"""B4 (Professor main + Side SCC cross-attn) brake GradCAM — 이미지까지.

전체 파이프라인에 grad 흐르게:
  image_curr/prev → seg backbone → (1,C_b,H',W') [hook]
    → adaptive_pool(1,40) → token(1,40,C_b)
    → a_2 v7b: encoder.proj → fusion → fc_proj → i2_mem(1,40,128)
    → IyModel B4 (side cross-attn, Q=traj KV=i2_mem) → pred ax
  target = -ax[step0] (brake 방향) → backward → seg backbone conv feat 의 GradCAM

lead-TTC 샘플만 선별 (inlane_gt: dist<20 & closing>1).
brake attribution 이 in-path lead 위치에 localize 되는지 검증.

사용:
  python scripts/viz_iy_gradcam_b4.py --n 16 --out outputs_multi/viz/gradcam_b4
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
from src.model.image_models.iy_model import IyModel, yaw_shift_tokens_batch
from src.model.image_models.a2_model import A2Model

ROI_Y0, ROI_Y1 = 235, 485
ROI_H, IMG_W = ROI_Y1 - ROI_Y0, 1280
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

D = os.environ.get("SCC_DATA_DIR") or ("/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data" if os.path.isdir("/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data") else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "local"))
CK = "/home/ailab-12/git/scc_controller_imitator/data/ckpts"


def parse_key(k):
    run, tail = k.rsplit("__t", 1)
    return run, int(tail)


def load_roi(raw_base, track, policy, run, tick):
    p = Path(raw_base) / track / policy / run / "CAMERA_1" / f"{tick}_Intensity.jpeg"
    if not p.exists():
        return None
    im = np.array(Image.open(p).convert("RGB"))
    if im.shape[0] >= ROI_Y1:
        im = im[ROI_Y0:ROI_Y1]
    return im.copy()


def normalize(roi):
    x = roi.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prebake", default=f"{D}/prebake/f1_prebake_iy_sangam_lead_0520_v1_brakeaug_prev20.npz")
    ap.add_argument("--inlane-gt", default=f"{D}/cache/inlane_gt_sangam_lead_0520.npz")
    ap.add_argument("--manifest", default=f"{D}/manifest_i1cache_sangam_lead_0520.json")
    ap.add_argument("--iy-ckpt", default=f"{CK}/iy_v25k_B4_gttc_accgate_best_loss.pt")
    ap.add_argument("--a2-ckpt", default=f"{CK}/a2_v7b_d128_alpha075_best_loss.pt")
    ap.add_argument("--seg-ckpt", default=f"{CK}/seg_pretrain_combined_v1_best_miou_backbone.pt")
    ap.add_argument("--backbone", default="tf_efficientnetv2_s")
    ap.add_argument("--fx", type=float, default=640.0)
    ap.add_argument("--traj-zero-after", type=int, default=20)
    ap.add_argument("--scale-ax", type=float, default=10.0)
    ap.add_argument("--dist-thr", type=float, default=20.0)
    ap.add_argument("--closing-thr", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("outputs_multi/viz/gradcam_b4"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    mf = json.load(open(args.manifest))
    track, policy, raw_base = mf["track"], mf["policy"], mf["raw_base"]
    print(f"[gradcam-b4] raw={raw_base} track={track} policy={policy}", flush=True)

    # seg backbone (combined_v1) — grad 는 feature map 에만 (params frozen)
    backbone = timm.create_model(args.backbone, pretrained=False, features_only=True, out_indices=[-1])
    sd = torch.load(args.seg_ckpt, map_location="cpu", weights_only=False)
    own = backbone.state_dict(); n = 0
    for k, v in sd.items():
        k2 = k.replace("backbone.", "")
        if k2 in own and own[k2].shape == v.shape:
            own[k2] = v; n += 1
    backbone.load_state_dict(own)
    backbone = backbone.to(dev).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    print(f"  seg backbone preload {n} keys", flush=True)

    # a_2 v7b
    a2 = A2Model(img_backbone=args.backbone, d_token=256, n_heads=4, n_ch=7, d_out=128).to(dev).eval()
    sda = torch.load(args.a2_ckpt, map_location=dev, weights_only=False)
    if isinstance(sda, dict) and "model" in sda: sda = sda["model"]
    own = a2.state_dict()
    sda = {k: v for k, v in sda.items() if k in own and own[k].shape == v.shape}
    ma, ua = a2.load_state_dict(sda, strict=False)
    for p in a2.parameters():
        p.requires_grad_(False)
    print(f"  a_2 v7b: missing={len(ma)} unexpected={len(ua)}", flush=True)

    # IyModel B4
    model = IyModel(C_b=256, d_model=128, nhead=8,   # train default nhead=8 (param shape 동일하나 head 분할 일치 필요)
                    use_obj_gate=False, use_side_branch=False, use_a2_feat=False,
                    side_num_layers=4, side_dim_feedforward=512,
                    main_no_image=True, side_cross_attn=True, fx=args.fx).to(dev).eval()
    sdm = torch.load(args.iy_ckpt, map_location=dev, weights_only=False)
    if isinstance(sdm, dict) and "model" in sdm: sdm = sdm["model"]
    mm, um = model.load_state_dict(sdm, strict=False)
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"  IyModel B4: missing={len(mm)} unexpected={len(um)}", flush=True)

    # prebake + inlane_gt
    pb = np.load(args.prebake, allow_pickle=True)
    curr_keys = pb["curr_keys"].tolist()
    prev_keys = pb["prev_keys"].tolist()
    d_yaws = pb["d_yaws"]; ego_5d = pb["ego_5d"]; traj = pb["traj"]; target = pb["target"]
    zg = np.load(args.inlane_gt, allow_pickle=True)
    g_k2i = {k: i for i, k in enumerate(zg["keys"].tolist())}
    g_dist = zg["inlane_dist"]; g_clos = zg["inlane_closing"]

    # lead-TTC candidates
    cand = []
    for i, ck in enumerate(curr_keys):
        gi = g_k2i.get(ck)
        if gi is None: continue
        d, c = g_dist[gi], g_clos[gi]
        if np.isfinite(d) and d < args.dist_thr and c > args.closing_thr:
            cand.append((i, float(d), float(c)))
    print(f"  lead-TTC candidates: {len(cand)}", flush=True)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(cand)
    pick = cand[:args.n]

    rec = []
    for n_done, (idx, gd, gc) in enumerate(pick):
        ck, pk = curr_keys[idx], prev_keys[idx]
        run_c, tk_c = parse_key(ck); run_p, tk_p = parse_key(pk)
        roi_c = load_roi(raw_base, track, policy, run_c, tk_c)
        roi_p = load_roi(raw_base, track, policy, run_p, tk_p)
        if roi_c is None or roi_p is None:
            print(f"  [skip] img miss {ck}", flush=True); continue
        ic = normalize(roi_c).unsqueeze(0).to(dev)
        ip = normalize(roi_p).unsqueeze(0).to(dev)
        ic.requires_grad_(True)   # backbone params 는 frozen 이지만 input grad 로 activation graph 유지

        fc_feat = backbone(ic)[-1]; fc_feat.retain_grad()      # (1,C_b,H',W')
        fp_feat = backbone(ip)[-1]
        tok_c = F.adaptive_avg_pool2d(fc_feat, (1, 40)).squeeze(2).transpose(1, 2)  # (1,40,C_b)
        tok_p = F.adaptive_avg_pool2d(fp_feat, (1, 40)).squeeze(2).transpose(1, 2)

        e5 = torch.from_numpy(ego_5d[idx]).unsqueeze(0).float().to(dev)
        d_y = torch.tensor([float(d_yaws[idx])], device=dev)
        fx_t = torch.tensor([args.fx], device=dev)
        tok_p_rot = yaw_shift_tokens_batch(tok_p, d_y, fx=fx_t)
        # a_2 v7b: proj → fusion → fc_proj
        tc_proj = a2.encoder.proj(tok_c.transpose(1, 2)).transpose(1, 2)
        tp_proj = a2.encoder.proj(tok_p_rot.transpose(1, 2)).transpose(1, 2)
        ego_mot = e5[:, :2]
        fi_mem_256 = a2.fusion(tc_proj, tp_proj, ego_mot)
        i2_mem = a2.fc_proj(fi_mem_256)                        # (1,40,128)
        a2_logits = a2.head(i2_mem)[0].detach().cpu().numpy()  # (40,7) for overlay

        t40 = torch.from_numpy(traj[idx]).unsqueeze(0).float().to(dev)
        if args.traj_zero_after is not None:
            t40 = t40.clone(); t40[:, args.traj_zero_after:, :] = 0.0
        t40.requires_grad_(True)                               # steer→traj attribution 용
        out = model({"i2_mem": i2_mem, "traj_40": t40, "ego_5d": e5, "fx": fx_t})
        pred = out[0] if isinstance(out, tuple) else out       # (1,40,2)
        ax0 = pred[0, 0, 1]                                    # 이미 physical m/s²
        steer0 = pred[0, 0, 0]                                 # deg

        # (1) brake → image grad-cam
        model.zero_grad(set_to_none=True)
        if fc_feat.grad is not None: fc_feat.grad = None
        if t40.grad is not None: t40.grad = None
        (-ax0).backward(retain_graph=True)                     # brake 방향
        A = fc_feat[0].detach()                                # (C_b,H',W')
        G = fc_feat.grad[0].detach()
        alpha = G.mean(dim=(1, 2))
        cam = torch.relu((alpha[:, None, None] * A).sum(0))     # (H',W')
        cam = cam.cpu().numpy()
        if cam.max() > 0: cam = cam / cam.max()
        cam_up = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((IMG_W, ROI_H), Image.BILINEAR)) / 255.0

        # (2) steer → traj 입력 attribution: 어느 waypoint 이 조향을 결정하는지
        if fc_feat.grad is not None: fc_feat.grad = None
        if t40.grad is not None: t40.grad = None
        steer0.backward()                                      # graph 재사용 (retain 위)
        traj_attr = t40.grad[0].abs().sum(dim=-1).detach().cpu().numpy()  # (40,) per-waypoint |grad|
        if traj_attr.max() > 0: traj_attr = traj_attr / traj_attr.max()

        # a_2 per-bin: obj prob + dist (in-lane = bin 16-25)
        obj_p = 1.0 / (1.0 + np.exp(-a2_logits[:, 0]))
        dist_b = a2_logits[:, 6]; dist_b = np.where(dist_b < 2, dist_b * 40.0, dist_b)
        ax_pred_full = pred[0, :, 1].detach().cpu().numpy()   # physical m/s²
        ax_tgt_full = target[idx, :, 1]

        steer_v = float(steer0.item())
        traj_xy = traj[idx]  # (40,3) ego frame [x_fwd, y_left, ...]
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 1, height_ratios=[1.1, 1.0, 1.0, 0.7])
        # row0: image + brake grad-cam
        ax0p = fig.add_subplot(gs[0]); ax0p.imshow(roi_c); ax0p.imshow(cam_up, cmap="jet", alpha=0.45)
        ax0p.set_title(f"{ck}  GT in-lane dist={gd:.1f}m closing={gc:+.1f}m/s  "
                       f"ax[0]={ax0.item():+.2f}  steer[0]={steer_v:+.2f}°  (brake→image attribution)", fontsize=11)
        ax0p.axis("off")
        # row1: a_2 per-bin obj strip
        ax1p = fig.add_subplot(gs[1]); ax1p.imshow(roi_c)
        for b in range(40):
            x0 = b / 40.0 * IMG_W; x1 = (b + 1) / 40.0 * IMG_W
            if 16 <= b <= 25:
                ax1p.axvspan(x0, x1, ymin=0.0, ymax=0.04, color="lime", alpha=0.5)
            if obj_p[b] > 0.5:
                ax1p.axvspan(x0, x1, ymin=0.04, ymax=0.04 + 0.25 * obj_p[b], color="red", alpha=0.4)
        ax1p.set_title("a_2(v7b) per-bin obj prob (red) | in-lane bins 16-25 (green)", fontsize=10)
        ax1p.axis("off")
        # row2: BEV traj colored by steer→traj attribution (어느 waypoint 이 조향 결정)
        ax2p = fig.add_subplot(gs[2])
        from matplotlib.patches import Rectangle
        ax2p.add_patch(Rectangle((-2.5, -1), 5, 2, ec="k", fc="gray", alpha=0.5))
        na = min(args.traj_zero_after or 40, 40)  # active waypoints
        sc = ax2p.scatter(traj_xy[:na, 0], traj_xy[:na, 1], c=traj_attr[:na], cmap="hot",
                          s=30 + 220 * traj_attr[:na], vmin=0, vmax=1, edgecolors="k", linewidths=0.3, zorder=3)
        ax2p.plot(traj_xy[:na, 0], traj_xy[:na, 1], "-", color="#4488ff", lw=0.8, alpha=0.5, zorder=2)
        cb = fig.colorbar(sc, ax=ax2p, fraction=0.025); cb.set_label("|∂steer/∂waypoint| (norm)")
        ax2p.set_xlabel("x_fwd (m)"); ax2p.set_ylabel("y_left (m)"); ax2p.grid(alpha=0.3)
        ax2p.set_aspect("equal"); ax2p.set_xlim(-3, 40); ax2p.set_ylim(-8, 8)
        amax = int(np.argmax(traj_attr[:na]))
        ax2p.set_title(f"steer→traj attribution (steer={steer_v:+.2f}°) — 최대 기여 waypoint #{amax} (t={amax*0.1:.1f}s)", fontsize=10)
        # row3: ax curve
        ax3p = fig.add_subplot(gs[3]); tt = np.arange(40) * 0.1
        ax3p.plot(tt, ax_tgt_full, "k--", lw=1.2, label="GT ax (raw)")
        ax3p.plot(tt, ax_pred_full, "r-", lw=1.6, label="pred ax")
        ax3p.axvline(0.0, color="orange", lw=2, label="step0")
        ax3p.set_xlabel("t (s)"); ax3p.set_ylabel("ax (m/s²)")
        ax3p.set_xlim(0, 4); ax3p.grid(alpha=0.3); ax3p.legend(fontsize=9, loc="lower right")
        in_obj = obj_p[16:26].max()
        ax3p.set_title(f"in-lane(16-25) max obj_p={in_obj:.2f}  min in-lane dist={dist_b[16:26].min():.1f}m", fontsize=10)
        fn = f"{n_done:02d}_{ck.replace('__','_')}_d{gd:.0f}_c{gc:.0f}.png"
        plt.tight_layout(); plt.savefig(args.out / fn, dpi=100); plt.close()
        rec.append((ck, gd, gc, float(ax0.item()), fn))
        print(f"  [{n_done+1}/{len(pick)}] {fn}  ax0={ax0.item():+.2f}", flush=True)

    html = args.out / "index.html"
    with open(html, "w") as f:
        f.write("<!doctype html><meta charset=utf-8><style>body{font-family:sans-serif;background:#1c1c1c;color:#ddd}.c{background:#2a2a2a;padding:8px;margin:12px 0;border-radius:6px}.c img{width:100%}</style>")
        f.write(f"<h2>B4 brake GradCAM — lead-TTC (dist&lt;{args.dist_thr} & closing&gt;{args.closing_thr})</h2>")
        f.write("<p>이미지→seg backbone→a_2 v7b fusion→i2_mem→side cross-attn→ax 전체 grad. "
                "jet = step0 brake(-ax) attribution. brake 가 in-path lead 위치에 localize 되는지 확인.</p>")
        for ck, gd, gc, ax0, fn in rec:
            f.write(f"<div class=c><div style='font-size:12px;color:#aaa'>{ck} | dist={gd:.1f}m closing={gc:+.1f} | ax0={ax0:+.2f}</div><img src='{fn}' loading=lazy></div>")
    print(f"[gradcam-b4] index → {html}  ({len(rec)} samples)", flush=True)


if __name__ == "__main__":
    main()
