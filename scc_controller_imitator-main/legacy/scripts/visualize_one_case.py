"""한 케이스 시각화: reference trajectory + MPC solved trajectory + lead object 시계열 (위치+속도).

플롯:
  - reference traj (ego-local 좌표, dashed)
  - solved_trajectory (MPC pred_X 의 x,y, solid)
  - lead object 시계열 (시간 따라 x,y 위치, 화살표 = 속도방향)
  - 이미지 (현재 + 이전 prev_rotated)

→ 경로와 객체가 overlap 하는지 시각적으로 판단 가능.
"""
from __future__ import annotations
import os, sys, time, glob, pickle, json
sys.path.insert(0, "/home/ailab-12/git/A1_Rule/devel/lib/python3/dist-packages")
SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "acados_scc", "scripts")
sys.path.insert(0, SCRIPTS_DIR)
BUILD_CACHE = os.path.join(os.path.dirname(SCRIPTS_DIR), "build_cache")
os.makedirs(BUILD_CACHE, exist_ok=True)
os.chdir(BUILD_CACHE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from PIL import Image

from MPC.NMPC_class_delayed_kinematic_cat import (
    Nonlinear_Delayed_Kinematic_Cat_Model_Predictive_Controller as MPC_Class,
)
from utils.trajectory_process import (
    npize_vehicle_state, npize_trajectory_local, npize_objects, resample_objects,
    generate_road_map, frenet_transform, frenet_mu_transform, global_to_local,
)
# Helpers from K=1 synth
from synthesize_scc_k1 import (
    bev_bbox_to_ego_local, make_vs_msg, wp_rel_to_traj_msg,
    lead_to_objs_msg, run_cycle, pick_front_lead, get_image_paths,
    COMPACT_NPZ, COMPACT_META, INDEX_JSON, RAW_IMG_BASE,
)

OUT_DIR = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/scc_synth_v1"
N_ITER_RTI = 20  # was 8 — 충분 수렴 위해 ↑


def run_one_and_get_pred(mpc, vs_msg, traj_msg, objs_msg, n_iter=N_ITER_RTI):
    """RTI n_iter 반복 후 마지막 결과 + MPC predicted trajectory 반환."""
    mpc.is_MPC_init = True
    mpc.pred_X = None; mpc.pred_U = None; mpc.X_init = None
    mpc.fallback_count = 0; mpc.last_mpc_status = None
    target_pred = None
    cio = None
    for _ in range(n_iter):
        out = run_cycle(mpc, vs_msg, traj_msg, objs_msg)
        target_cart, target_pred_now, target_input, stats, cio_now = out
        target_pred = np.asarray(target_pred_now).copy() if target_pred_now is not None else target_pred
        if cio_now is not None:
            cio = {
                'steer': np.asarray(cio_now['steer'], dtype=np.float64).copy(),
                'acceleration': np.asarray(cio_now['acceleration'], dtype=np.float64).copy(),
            }
    return target_pred, cio


def main():
    print("loading compact npz + meta ...")
    z = np.load(COMPACT_NPZ)
    meta_list = json.loads(open(COMPACT_META).read())
    print(f"  {len(meta_list)} frames")
    idx_data = json.loads(open(INDEX_JSON).read())
    run_dt = {e["run_id"]: float(e.get("dt_per_tick_ms", 10.0)) for e in idx_data}

    # SCC-relevant 한 frame 선택: lead 가 가까이 (x<40) 같은 차로 (|y|<3)
    cand = None
    for m in meta_list:
        i = m["i"]
        ego = z[f"s{i}__ego_pose"]
        if float(ego[3]) < 5.0:
            continue
        for o in m["objects"]:
            x_m, y_m, _, _ = bev_bbox_to_ego_local(o["bbox"])
            if 5.0 < x_m < 40.0 and abs(y_m) < 3.0:
                cand = m
                break
        if cand:
            break
    if cand is None:
        print("no candidate frame found")
        return
    print(f"chosen frame: {cand['frame_id']}")

    i = cand["i"]
    ego = z[f"s{i}__ego_pose"]
    wp = z[f"s{i}__wp_rel_gt"]
    lead = pick_front_lead(cand["objects"], max_x=120.0, max_y_abs=5.0)
    if lead is None:
        # 가장 가까운 전방 (lane 범위 무시) 사용
        best = None; best_x = 999
        for o in cand["objects"]:
            x_m, y_m, L, W = bev_bbox_to_ego_local(o["bbox"])
            if x_m > 2.0 and x_m < best_x:
                best_x = x_m
                v = o.get("velocity", [0.0, 0.0])
                best = (x_m, y_m, L, W, np.asarray(v, dtype=np.float64))
        lead = best
    print(f"  ego: vx={ego[3]:.2f}, vy={ego[4]:.3f}")
    if lead:
        print(f"  lead: x={lead[0]:.1f}, y={lead[1]:.1f}, L={lead[2]:.1f}, W={lead[3]:.1f}, vel=({lead[4][0]:.2f}, {lead[4][1]:.2f})")

    mpc = MPC_Class()
    vx = float(ego[3])
    ax_est = float((wp[0, 3] - vx) / 0.1)
    vs_msg = make_vs_msg(ego, ax=ax_est, ay=0.0, tire_angle=0.0)
    traj_msg = wp_rel_to_traj_msg(wp)
    from autohyu_msgs.msg import PredictObjects
    objs_no = PredictObjects()
    objs_with = lead_to_objs_msg(lead) if lead else PredictObjects()

    print("\nRunning MPC: no-lead ...")
    pred_no, cio_no = run_one_and_get_pred(mpc, vs_msg, traj_msg, objs_no)
    print("Running MPC: with-lead ...")
    pred_lead, cio_lead = run_one_and_get_pred(mpc, vs_msg, traj_msg, objs_with)

    # reference trajectory (ego local, cumulative)
    abs_xy = np.cumsum(wp[:, :2], axis=0)
    ref_x = np.concatenate([[0], abs_xy[:, 0]])
    ref_y = np.concatenate([[0], abs_xy[:, 1]])

    # lead time series — 등속 가정
    Ts = mpc.Ts_MPC
    N = mpc.N_MPC + 1
    if lead:
        t = np.arange(N) * Ts
        lead_x = lead[0] + lead[4][0] * t
        lead_y = lead[1] + lead[4][1] * t
    else:
        lead_x = None; lead_y = None

    # solved trajectory (target_pred): [x, y, yaw, v, delta, ddelta]
    sol_no_x = pred_no[:, 0] if pred_no is not None else None
    sol_no_y = pred_no[:, 1] if pred_no is not None else None
    sol_le_x = pred_lead[:, 0] if pred_lead is not None else None
    sol_le_y = pred_lead[:, 1] if pred_lead is not None else None

    # image
    curr_path, prev_path = get_image_paths(cand, run_dt.get(cand["run_id"], 10.0))
    img_curr = np.array(Image.open(curr_path).convert("RGB")) if os.path.exists(curr_path) else None
    img_prev = np.array(Image.open(prev_path).convert("RGB")) if os.path.exists(prev_path) else None

    # Plot
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.4])
    # Top: images
    ax_curr = fig.add_subplot(gs[0, 0])
    ax_prev = fig.add_subplot(gs[0, 1])
    ax_ctrl = fig.add_subplot(gs[0, 2])
    if img_curr is not None:
        ax_curr.imshow(img_curr); ax_curr.set_title("image_curr"); ax_curr.axis("off")
    if img_prev is not None:
        ax_prev.imshow(img_prev); ax_prev.set_title("image_prev"); ax_prev.axis("off")
    if cio_no is not None:
        ax_ctrl.plot(np.arange(len(cio_no['acceleration'])) * Ts, cio_no['acceleration'],
                     label='accel (no lead)', color='C0')
    if cio_lead is not None:
        ax_ctrl.plot(np.arange(len(cio_lead['acceleration'])) * Ts, cio_lead['acceleration'],
                     label='accel (with lead)', color='C3')
    ax_ctrl.set_title("acceleration over horizon")
    ax_ctrl.set_xlabel("t (s)"); ax_ctrl.set_ylabel("accel (m/s²)")
    ax_ctrl.grid(True); ax_ctrl.legend()

    # Bottom: top-down view (ego local)
    ax = fig.add_subplot(gs[1, :])
    ax.plot(ref_x, ref_y, 'k--', lw=1.5, label="reference trajectory")
    if sol_no_x is not None:
        ax.plot(sol_no_x, sol_no_y, 'C0-', lw=2, label=f"MPC solved (no lead)")
    if sol_le_x is not None:
        ax.plot(sol_le_x, sol_le_y, 'C3-', lw=2, label=f"MPC solved (with lead)")
    # ego at origin
    ax.add_patch(Rectangle((-2.5, -1), 5, 2, ec='k', fc='lightgray', alpha=0.6, label="ego"))
    # lead positions over time
    if lead_x is not None:
        # 시계열 점선
        ax.plot(lead_x, lead_y, 'r:', lw=1, alpha=0.6)
        # 5개 시점에 박스 + velocity 화살표
        sample_idx = [0, N // 4, N // 2, 3 * N // 4, N - 1]
        for k in sample_idx:
            # 박스
            cx, cy = lead_x[k], lead_y[k]
            L, W = lead[2], lead[3]
            ax.add_patch(Rectangle((cx - L/2, cy - W/2), L, W,
                                    ec='r', fc='salmon', alpha=0.4))
            ax.text(cx, cy + W/2 + 0.3, f"t={k*Ts:.1f}s", color='red',
                    fontsize=8, ha='center')
        # 첫 시점의 velocity 화살표
        vx_l, vy_l = lead[4][0], lead[4][1]
        speed = float(np.hypot(vx_l, vy_l))
        ax.annotate("", xy=(lead[0] + vx_l, lead[1] + vy_l),
                    xytext=(lead[0], lead[1]),
                    arrowprops=dict(arrowstyle="->", color='red', lw=2))
        ax.text(lead[0], lead[1] - 1.5, f"|v|={speed:.1f} m/s",
                color='red', fontsize=10, ha='center')

    ax.set_aspect("equal")
    ax.set_xlim(-5, max(60, ref_x.max() + 5))
    ax.set_ylim(-10, 10)
    ax.set_xlabel("x [m] (ego forward)"); ax.set_ylabel("y [m] (ego left)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_title(f"Top-down: ref / MPC solved / lead time series   |   ego vx={vx:.1f} m/s, "
                 f"lead_dist={lead[0]:.1f}m, lead_vel=({lead[4][0]:.1f}, {lead[4][1]:.1f}) m/s" if lead else f"ego vx={vx:.1f}")

    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "one_case_topdown.png")
    fig.savefig(out_png, dpi=120)
    print(f"\nsaved: {out_png}")
    print(f"frame_id: {cand['frame_id']}")
    if cio_no is not None and cio_lead is not None:
        a0 = cio_no['acceleration'][:8]
        a1 = cio_lead['acceleration'][:8]
        print(f"accel diff (lead - no_lead)[:8]: {np.round(a1 - a0, 3)}")


if __name__ == "__main__":
    main()
