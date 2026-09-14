"""fwd50 / lead / nolead 합성 sample 시각화.

요소:
- CAMERA_1 전방 이미지 (image_curr_path) + meta objects projection markers
- BEV (ego frame): ego box, lead box, **모든 frame objects**, ref traj, gt-based integrated traj
- gt_accel(t), gt_steer(t)

lead_info schema (from synthesize_scc_kn_random.py): [lx, ly, vx, vy, L, W]
"""
from __future__ import annotations
import os, sys, glob, pickle, argparse, json
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, '/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/jiwon_e2e/e2e_racing')
from src.data.ipm import _invert_extrinsic, _world_to_image

OUT_DIR = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/scc_synth_k100_v1"
META_JSON = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/multiagnet_compact_meta.json"
CALIB_DIR = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/jiwon_e2e/e2e_racing/calibration"
# CAMERA_1 (front)
K_CAM = np.load(f"{CALIB_DIR}/CAMERA_1__intrinsic.npy")
E_CAM = np.load(f"{CALIB_DIR}/CAMERA_1__extrinsic.npy")
R_V2C, T_V2C = _invert_extrinsic(E_CAM)
# BEV bbox → ego conversion
X_MAX, Y_MIN, CELL = 80.0, -25.0, 0.5
def bbox_to_ego(b):
    cx, cy, _, _ = b
    return X_MAX - cy*CELL, cx*CELL + Y_MIN  # (ego_x, ego_y)

# kinematic bicycle wheelbase (acados MPC config — base.md)
L_WB = 2.7
# MPC dt
TS = 0.1


def forward_integrate(ego_5d, gt_steer_deg, gt_accel):
    """Forward integrate kinematic bicycle for visualization.
    ego_5d: (5,) [vx, vy, ax, ay, yaw_rate]   (yaw is ego frame, so start at 0)
    gt_steer_deg: (T,) target tire angle (degrees) — front-wheel
    gt_accel: (T,) target acceleration (m/s²)
    Returns (T+1, 4): x, y, yaw_rad, v at each step.
    """
    T = len(gt_steer_deg)
    out = np.zeros((T + 1, 4))
    v = float(ego_5d[0])    # vx
    x, y, yaw = 0.0, 0.0, 0.0
    out[0] = [x, y, yaw, v]
    for t in range(T):
        delta = np.deg2rad(float(gt_steer_deg[t]))
        a = float(gt_accel[t])
        # advance
        v_next = max(0.0, v + a * TS)
        v_mid = 0.5 * (v + v_next)
        x += v_mid * np.cos(yaw) * TS
        y += v_mid * np.sin(yaw) * TS
        yaw += v_mid / L_WB * np.tan(delta) * TS
        v = v_next
        out[t + 1] = [x, y, yaw, v]
    return out


def visualize_sample(s, meta_objects, out_path, title_extra=""):
    """meta_objects: frame 의 모든 차 [(ego_x, ego_y, L, W), ...]"""
    fig = plt.figure(figsize=(34, 22))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.3], width_ratios=[1, 1, 1])

    # (0,0): CAMERA_1 image with object markers
    ax_img = fig.add_subplot(gs[0, :2])
    if os.path.exists(s["image_curr_path"]):
        img = np.array(Image.open(s["image_curr_path"]).convert("RGB"))
        ax_img.imshow(img)
        H, W_img = img.shape[:2]
        ax_img.add_patch(Rectangle((0, 235), W_img, 250, ec='lime', fc='none', lw=2, label='ROI[235,485]'))
        # project all meta objects
        for ex, ey, L, W in meta_objects:
            if not (1.5 < ex < 200 and abs(ey) < 50): continue
            cam_pt = _world_to_image(np.array([[ex, ey, -0.13]]), K_CAM, R_V2C, T_V2C)[0]
            u, v, z = cam_pt
            if z > 0 and 0 <= u < W_img and 0 <= v < H:
                ax_img.scatter([u], [v], s=900, facecolors='none', edgecolors='yellow', lw=4)
                ax_img.text(u+12, v, f"({ex:.0f},{ey:.1f})", color='yellow', fontsize=14,
                            bbox=dict(facecolor='black', alpha=0.6))
            else:
                ax_img.text(20, 100 + 18*meta_objects.index((ex,ey,L,W)),
                            f"out-of-FOV obj ({ex:.0f},{ey:.1f})", color='red', fontsize=14,
                            bbox=dict(facecolor='black', alpha=0.7))
    ax_img.set_title(f"CAMERA_1: {os.path.basename(s['image_curr_path'])}  (yellow=meta-obj projected)")
    ax_img.legend(loc="upper right", fontsize=14)
    ax_img.axis('off')

    # (0,2): controls
    ax_ctrl = fig.add_subplot(gs[0, 2])
    t = np.arange(30) * TS
    ax_ctrl.plot(t, s["gt_accel_30"], 'C3-', lw=1.5, label='gt accel (m/s²)')
    ax_ctrl_2 = ax_ctrl.twinx()
    ax_ctrl_2.plot(t, s["gt_steer_30"], 'C0-', lw=1.5, label='gt steer (deg)')
    ax_ctrl.set_xlabel('t (s)')
    ax_ctrl.set_ylabel('accel m/s²', color='C3')
    ax_ctrl_2.set_ylabel('steer deg', color='C0')
    ax_ctrl.axhline(0, color='gray', lw=0.5)
    ax_ctrl.set_title('GT control (MPC solver output)')
    ax_ctrl.grid(True, alpha=0.3)

    # (1,:): BEV top-down
    ax_bev = fig.add_subplot(gs[1, :])
    # reference traj (input to MPC)
    ref = s["traj_30"]  # (30, 4) - 보통 x, y, yaw, v
    # 누적 좌표인지 절대 좌표인지 확인 — 보통 absolute ego-local
    ax_bev.plot(ref[:, 0], ref[:, 1], 'k--', lw=1.5, label="reference traj (input)")
    ax_bev.scatter(ref[:, 0], ref[:, 1], c='k', s=10, alpha=0.4)

    # gt-based forward-integrated solved traj
    sol = forward_integrate(s["ego_5d"], s["gt_steer_30"], s["gt_accel_30"])
    ax_bev.plot(sol[:, 0], sol[:, 1], 'C2-', lw=2.2, label="GT control → integrated traj")
    ax_bev.scatter(sol[::5, 0], sol[::5, 1], c='C2', s=20)

    # ego box
    ax_bev.add_patch(Rectangle((-2.5, -1.0), 5, 2, ec='k', fc='lightgray', alpha=0.7, label='ego'))

    # 모든 meta objects (gray box)
    for ex, ey, L, W in meta_objects:
        ax_bev.add_patch(Rectangle((ex-L/2, ey-W/2), L, W, ec='gray', fc='lightgray', alpha=0.4))
        ax_bev.text(ex, ey + W/2 + 0.3, f"({ex:.0f},{ey:.1f})", color='gray', fontsize=11, ha='center')

    # lead box (sample 의 lead — pick_front_lead 결과)
    if s["lead_valid"]:
        lx, ly, vx_l, vy_l, L, W = s["lead_info"]
        ax_bev.add_patch(Rectangle((float(lx)-float(L)/2, float(ly)-float(W)/2),
                                   float(L), float(W), ec='red', fc='salmon', alpha=0.7))
        ax_bev.annotate("", xy=(float(lx)+float(vx_l), float(ly)+float(vy_l)),
                        xytext=(float(lx), float(ly)),
                        arrowprops=dict(arrowstyle="->", color='red', lw=2))
        speed = float(np.hypot(vx_l, vy_l))
        t_arr = np.arange(31) * TS
        lead_x_t = float(lx) + float(vx_l) * t_arr
        lead_y_t = float(ly) + float(vy_l) * t_arr
        ax_bev.plot(lead_x_t, lead_y_t, 'r:', lw=1.2, alpha=0.7, label=f"lead const-vel ({speed:.1f}m/s)")
        ax_bev.text(float(lx), float(ly) + float(W)/2 + 0.5,
                    f"LEAD\nd={float(lx):.1f}m,y={float(ly):.1f}\n|v|={speed:.1f}",
                    color='red', fontsize=14, ha='center', fontweight='bold')

    ax_bev.set_aspect("equal")
    xmax = max(60, ref[:, 0].max() + 10, sol[:, 0].max() + 10)
    ymax = max(10, abs(ref[:, 1]).max() + 3, abs(sol[:, 1]).max() + 3)
    ax_bev.set_xlim(-5, xmax)
    ax_bev.set_ylim(-ymax, ymax)
    ax_bev.set_xlabel("x [m] (ego forward)")
    ax_bev.set_ylabel("y [m] (ego left)")
    ax_bev.grid(True, alpha=0.3)
    ax_bev.legend(loc="upper right", fontsize=14)

    vx0 = float(s["ego_5d"][0])
    title = (f"{s['frame_id']}  [{s['traj_tag']}]   ego vx={vx0:.1f} m/s,  "
             f"lead_valid={s['lead_valid']}  {title_extra}")
    ax_bev.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    print(f"saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="fwd50", help="pkl prefix (fwd50, lead, lead2, nolead, ...)")
    ap.add_argument("--frame-id", default=None)
    ap.add_argument("--traj-tag", default=None, help="logged | random_*")
    ap.add_argument("--n", type=int, default=4, help="num samples (collage)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pkls = sorted([fp for fp in glob.glob(os.path.join(OUT_DIR, f"{args.prefix}_*.pkl"))
                    if os.path.getsize(fp) > 10000])
    if not pkls:
        print(f"no non-empty pkl with prefix {args.prefix}_")
        return
    print(f"candidate pkls: {len(pkls)}")

    # frame_id → objects (BEV bbox 변환)
    meta = json.load(open(META_JSON))
    meta_by_fid = {m["frame_id"]: m for m in meta}

    picks = []
    for fp in pkls:
        d = pickle.load(open(fp, "rb"))
        for s in d["samples"]:
            if args.frame_id and s["frame_id"] != args.frame_id: continue
            if args.traj_tag and s["traj_tag"] != args.traj_tag: continue
            picks.append(s)
            if len(picks) >= args.n: break
        if len(picks) >= args.n: break
    if not picks:
        print("no matching sample"); return
    print(f"selected {len(picks)} sample(s)")

    for k, s in enumerate(picks):
        m = meta_by_fid.get(s["frame_id"], {})
        meta_objs = []
        for o in m.get("objects", []):
            ex, ey = bbox_to_ego(o["bbox"])
            meta_objs.append((ex, ey, 4.5, 2.0))  # approx L,W
        suffix = f"_{k}" if args.n > 1 else ""
        out_path = args.out or os.path.join(OUT_DIR, f"viz_{args.prefix}_{s['frame_id']}{suffix}_{s['traj_tag']}.png")
        visualize_sample(s, meta_objs, out_path)


if __name__ == "__main__":
    main()
