"""Open-loop SCC inference + 시각화.

best ckpt (또는 latest) 로드 → val 또는 train sample 일부에 대해 1-step pred 후 시각화:
  - CAMERA_1 image + meta objects
  - BEV: reference traj + GT-integrated traj + Pred-integrated traj + lead box
  - 제어 시퀀스: GT vs Pred (steer/40, accel)
  - MAE/MSE 표시

Usage:
  ~/miniforge3/envs/ml_con/bin/python scripts/eval_open_loop.py \
    --ckpt outputs_multi/scc_v2_pt654/best.pt --n 8
"""
from __future__ import annotations
import os, sys, glob, pickle, argparse, json, random
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)   # scc_controller_imitator path 먼저
from src.datamodule.scc_dataset import SCCSynthDataset, _pad_traj, ROI_Y_TOP, ROI_Y_BOT, IMAGENET_MEAN, IMAGENET_STD, STEER_DIV
from src.model.image_models.image_scc_models_v2 import SCCControllerImitatorV2
# jiwon_e2e 는 별도 alias 로 import
sys.path.append("/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/jiwon_e2e/e2e_racing")
import importlib
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_e2e_ipm", "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/jiwon_e2e/e2e_racing/src/data/ipm.py")
_ipm = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_ipm)
_invert_extrinsic = _ipm._invert_extrinsic
_world_to_image = _ipm._world_to_image

CALIB_DIR = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/jiwon_e2e/e2e_racing/calibration"
K_CAM = np.load(f"{CALIB_DIR}/CAMERA_1__intrinsic.npy")
E_CAM = np.load(f"{CALIB_DIR}/CAMERA_1__extrinsic.npy")
R_V2C, T_V2C = _invert_extrinsic(E_CAM)
META_JSON = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/multiagnet_compact_meta.json"
X_MAX, Y_MIN, CELL = 80.0, -25.0, 0.5
L_WB, TS = 2.7, 0.1

def bbox_to_ego(b):
    cx, cy, _, _ = b
    return X_MAX - cy*CELL, cx*CELL + Y_MIN


def forward_integrate(vx0, gt_steer_deg, gt_accel):
    T = len(gt_steer_deg)
    out = np.zeros((T + 1, 4))
    v = float(vx0); x, y, yaw = 0., 0., 0.
    out[0] = [x, y, yaw, v]
    for t in range(T):
        delta = np.deg2rad(float(gt_steer_deg[t])); a = float(gt_accel[t])
        v_next = max(0.0, v + a * TS); v_mid = 0.5 * (v + v_next)
        x += v_mid * np.cos(yaw) * TS; y += v_mid * np.sin(yaw) * TS
        yaw += v_mid / L_WB * np.tan(delta) * TS; v = v_next
        out[t + 1] = [x, y, yaw, v]
    return out


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default="data/scc_synth_k100_v1")
    ap.add_argument("--prefix", default="fwd50")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out-dir", default="outputs_multi/eval_open_loop")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=11)
    return ap.parse_args()


def visualize_one(s, pred, meta_objs, out_path):
    """s: sample dict (from dataset). pred: (40, 2) [steer/40, accel]"""
    gt_steer_pad = s["target"][:, 0].cpu().numpy() * STEER_DIV
    gt_accel_pad = s["target"][:, 1].cpu().numpy()
    pred_steer = pred[:, 0] * STEER_DIV
    pred_accel = pred[:, 1]
    # logged 30-step 만 평가 (random 은 40)
    if s["traj_tag"] == "logged":
        T_use = 30
    else:
        T_use = 40
    mae_st = float(np.mean(np.abs(gt_steer_pad[:T_use] - pred_steer[:T_use])))
    mae_ac = float(np.mean(np.abs(gt_accel_pad[:T_use] - pred_accel[:T_use])))

    fig = plt.figure(figsize=(28, 16))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.3])

    # (0,:2) image with markers
    ax_img = fig.add_subplot(gs[0, :2])
    img_path = s.get("image_curr_path_str") or s.get("frame_id")
    if isinstance(img_path, str) and os.path.exists(img_path):
        img = np.array(Image.open(img_path).convert("RGB"))
        ax_img.imshow(img)
        H, W = img.shape[:2]
        ax_img.add_patch(Rectangle((0, 235), W, 250, ec='lime', fc='none', lw=2))
        for ex, ey, _, _ in meta_objs:
            if not (1.5 < ex < 200 and abs(ey) < 50): continue
            cam_pt = _world_to_image(np.array([[ex, ey, -0.13]]), K_CAM, R_V2C, T_V2C)[0]
            u, v, z = cam_pt
            if z > 0 and 0 <= u < W and 0 <= v < H:
                ax_img.scatter([u], [v], s=900, facecolors='none', edgecolors='yellow', lw=4)
                ax_img.text(u+15, v, f"({ex:.0f},{ey:.1f})", color='yellow', fontsize=12,
                             bbox=dict(facecolor='black', alpha=0.6))
    ax_img.set_title(f"{s['frame_id']} [{s['traj_tag']}]   MAE: steer={mae_st:.2f}° accel={mae_ac:.2f}m/s²",
                     fontsize=14)
    ax_img.axis('off')

    # (0,2) controls GT vs Pred
    ax_c = fig.add_subplot(gs[0, 2])
    t = np.arange(T_use) * TS
    ax_c.plot(t, gt_accel_pad[:T_use], 'r-', lw=2, label='gt accel')
    ax_c.plot(t, pred_accel[:T_use], 'r--', lw=2, label='pred accel')
    ax2 = ax_c.twinx()
    ax2.plot(t, gt_steer_pad[:T_use], 'b-', lw=2, label='gt steer')
    ax2.plot(t, pred_steer[:T_use], 'b--', lw=2, label='pred steer')
    ax_c.set_ylabel('accel m/s²', color='r')
    ax2.set_ylabel('steer deg', color='b')
    ax_c.set_xlabel('t (s)')
    ax_c.legend(loc='upper left'); ax2.legend(loc='upper right')
    ax_c.grid(True, alpha=0.3)
    ax_c.set_title("GT vs Pred control")

    # (1,:) BEV
    ax_bev = fig.add_subplot(gs[1, :])
    # reference traj
    ref = s["traj_30"].cpu().numpy()
    ax_bev.plot(ref[:T_use, 0], ref[:T_use, 1], 'k--', lw=2, label="reference traj")

    # GT-integrated
    vx0 = float(s["ego_5d"][0].cpu())
    sol_gt = forward_integrate(vx0, gt_steer_pad[:T_use], gt_accel_pad[:T_use])
    ax_bev.plot(sol_gt[:, 0], sol_gt[:, 1], 'g-', lw=2.5, label="GT control → traj")

    # Pred-integrated
    sol_pr = forward_integrate(vx0, pred_steer[:T_use], pred_accel[:T_use])
    ax_bev.plot(sol_pr[:, 0], sol_pr[:, 1], 'r-', lw=2.5, label="Pred control → traj")

    # ego
    ax_bev.add_patch(Rectangle((-2.5, -1), 5, 2, ec='k', fc='lightgray', alpha=0.7, label='ego'))
    # all meta objects + lead
    for ex, ey, L, W in meta_objs:
        if not (1.5 < ex < 100 and abs(ey) < 20): continue
        ax_bev.add_patch(Rectangle((ex-L/2, ey-W/2), L, W, ec='gray', fc='lightgray', alpha=0.4))
        ax_bev.text(ex, ey + W/2 + 0.3, f"({ex:.0f},{ey:.1f})", color='gray', fontsize=10, ha='center')
    if bool(s["lead_valid"].cpu()):
        lx, ly, vx_l, vy_l, L, W = [float(x) for x in s["lead_info"].cpu()]
        ax_bev.add_patch(Rectangle((lx-L/2, ly-W/2), L, W, ec='red', fc='salmon', alpha=0.7))
        ax_bev.annotate("", xy=(lx+vx_l, ly+vy_l), xytext=(lx, ly),
                          arrowprops=dict(arrowstyle="->", color='red', lw=2))
        ax_bev.text(lx, ly + W/2 + 0.5, f"LEAD\nd={lx:.1f}m\n|v|={np.hypot(vx_l, vy_l):.1f}",
                     color='red', fontsize=12, ha='center', fontweight='bold')

    ax_bev.set_aspect('equal')
    xmax = max(60, ref[:, 0].max() + 10, sol_gt[:, 0].max() + 10, sol_pr[:, 0].max() + 10)
    ymax = max(10, abs(ref[:, 1]).max() + 3, abs(sol_gt[:, 1]).max() + 3, abs(sol_pr[:, 1]).max() + 3)
    ax_bev.set_xlim(-5, xmax); ax_bev.set_ylim(-ymax, ymax)
    ax_bev.set_xlabel("x [m] (ego forward)"); ax_bev.set_ylabel("y [m] (ego left)")
    ax_bev.grid(True, alpha=0.3)
    ax_bev.legend(loc='upper right', fontsize=11)
    ax_bev.set_title(f"ego vx={vx0:.1f}m/s  steps={T_use}")

    fig.tight_layout()
    fig.savefig(out_path, dpi=85)
    plt.close(fig)
    return mae_st, mae_ac


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ckpt
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    ckpt_args = ckpt.get("args", {})
    print(f"loaded ckpt: {args.ckpt} (epoch={ckpt.get('epoch', '?')}, best_val={ckpt.get('best_val', '?')})")
    img_backbone = ckpt_args.get("img_backbone", "efficientnet_b0")
    print(f"  using img_backbone={img_backbone} (from ckpt args)")

    model = SCCControllerImitatorV2(img_backbone=img_backbone).to(args.device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # dataset
    ds = SCCSynthDataset(args.data_dir, (args.prefix,), split=args.split, val_ratio=0.1)
    # frame_id → meta
    META = json.load(open(META_JSON))
    meta_by_fid = {m["frame_id"]: m for m in META}

    # 다양 sample: traj_tag 별 (logged + 다른 random)
    rng = random.Random(args.seed)
    by_fid_tag = {}
    for k in range(len(ds)):
        s = ds[k]
        key = (s["frame_id"], s["traj_tag"])
        by_fid_tag.setdefault(s["frame_id"], []).append(k)
    fids = list(by_fid_tag.keys())
    rng.shuffle(fids)
    pick_idx = []
    for fid in fids:
        if len(pick_idx) >= args.n: break
        # logged + 1 random
        ks = by_fid_tag[fid]
        for k in ks[:2]:
            pick_idx.append(k)
            if len(pick_idx) >= args.n: break
    print(f"picks: {len(pick_idx)} samples from {len(set(ds[k]['frame_id'] for k in pick_idx))} frames")

    summary = []
    for k in pick_idx:
        s = ds[k]
        # raw image path 복원
        s["image_curr_path_str"] = meta_by_fid.get(s["frame_id"], {}).get("frame_id", "")
        # original path from pkl
        fp_str = None
        for fp, i in ds.idx[k:k+1]:
            samp = ds._get_sample(fp, i)
            fp_str = samp["image_curr_path"]
        s["image_curr_path_str"] = fp_str

        # forward
        batch = {}
        for key in ["image_curr", "image_prev_rot", "ego_5d", "traj_30"]:
            batch[key] = s[key].unsqueeze(0).to(args.device)
        with torch.no_grad():
            pred = model(batch)[0].cpu().numpy()  # (40, 2)

        meta_objs = []
        for o in meta_by_fid.get(s["frame_id"], {}).get("objects", []):
            ex, ey = bbox_to_ego(o["bbox"])
            meta_objs.append((ex, ey, 4.5, 2.0))
        out = os.path.join(args.out_dir, f"eval_{s['frame_id']}_{s['traj_tag']}.png")
        mae_st, mae_ac = visualize_one(s, pred, meta_objs, out)
        summary.append({"frame_id": s["frame_id"], "traj_tag": s["traj_tag"],
                          "mae_steer_deg": mae_st, "mae_accel": mae_ac, "out": out})
        print(f"  {s['frame_id'][:55]} {s['traj_tag']}  MAE st={mae_st:.2f}° ac={mae_ac:.2f}  → {os.path.basename(out)}")

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if summary:
        print(f"\noverall MAE: steer={np.mean([x['mae_steer_deg'] for x in summary]):.2f}° "
              f"accel={np.mean([x['mae_accel'] for x in summary]):.2f}")


if __name__ == "__main__":
    main()
