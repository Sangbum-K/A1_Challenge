"""다수 케이스 시각화: lead 가 경로 안/옆/밖 다양한 시나리오 비교.

각 case 마다 top-down view (reference + solved trajectory + lead 시계열).
경로상 객체 overlap 여부에 따른 SCC 동작 차이 확인.

Categories:
  - in-path  : lead |y|<1.5  (같은 차로)
  - side     : 1.5 < lead |y| < 4 (옆 차로)
  - far_side : lead |y| > 4 (멀리 옆)
  - far_ahead: lead x > 60 (멀리 전방)
"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, "/home/ailab-12/git/A1_Rule/devel/lib/python3/dist-packages")
SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "acados_scc", "scripts")
sys.path.insert(0, SCRIPTS_DIR)
BUILD_CACHE = os.path.join(os.path.dirname(SCRIPTS_DIR), "build_cache")
os.chdir(BUILD_CACHE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from MPC.NMPC_class_delayed_kinematic_cat import (
    Nonlinear_Delayed_Kinematic_Cat_Model_Predictive_Controller as MPC_Class,
)
from synthesize_scc_k1 import (
    bev_bbox_to_ego_local, make_vs_msg, wp_rel_to_traj_msg,
    lead_to_objs_msg, run_cycle,
    COMPACT_NPZ, COMPACT_META,
)
from autohyu_msgs.msg import PredictObjects

OUT_DIR = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/scc_synth_v1"
N_ITER = 20  # was 8 — 충분 수렴 위해 ↑


def categorize(lead_x, lead_y):
    if abs(lead_y) < 1.5:
        if lead_x > 60: return "far_ahead"
        return "in_path"
    if abs(lead_y) < 4.0:
        return "side"
    return "far_side"


def pick_cases_by_category(meta_list, z, target_per_cat=2):
    """원하는 카테고리별로 cases 선택."""
    buckets = {"in_path": [], "side": [], "far_side": [], "far_ahead": []}
    for m in meta_list:
        i = m["i"]
        ego = z[f"s{i}__ego_pose"]
        if float(ego[3]) < 5.0:
            continue
        # 가장 가까운 전방 객체 선택
        best = None; best_x = 999
        for o in m["objects"]:
            xm, ym, L, W = bev_bbox_to_ego_local(o["bbox"])
            if xm > 2.0 and xm < best_x:
                best_x = xm
                v = o.get("velocity", [0.0, 0.0])
                best = (xm, ym, L, W, np.asarray(v, dtype=np.float64))
        if best is None:
            continue
        cat = categorize(best[0], best[1])
        if len(buckets[cat]) < target_per_cat:
            buckets[cat].append((m, best))
        if all(len(v) >= target_per_cat for v in buckets.values()):
            break
    return buckets


def run_and_pred(mpc, vs_msg, traj_msg, objs_msg, n_iter=N_ITER):
    mpc.is_MPC_init = True
    mpc.pred_X = None; mpc.pred_U = None; mpc.X_init = None
    mpc.fallback_count = 0; mpc.last_mpc_status = None
    target_pred = None; cio = None
    for _ in range(n_iter):
        out = run_cycle(mpc, vs_msg, traj_msg, objs_msg)
        _, tp, _, _, cio_now = out
        if tp is not None:
            target_pred = np.asarray(tp).copy()
        if cio_now is not None:
            cio = {
                'acceleration': np.asarray(cio_now['acceleration'], dtype=np.float64).copy(),
                'steer': np.asarray(cio_now['steer'], dtype=np.float64).copy(),
            }
    return target_pred, cio


def plot_case(ax, m, lead, z, mpc, Ts):
    i = m["i"]
    ego = z[f"s{i}__ego_pose"]
    wp = z[f"s{i}__wp_rel_gt"]
    vx = float(ego[3])

    # reference (cumulative wp)
    abs_xy = np.cumsum(wp[:, :2], axis=0)
    ref_x = np.concatenate([[0.0], abs_xy[:, 0]])
    ref_y = np.concatenate([[0.0], abs_xy[:, 1]])

    ax_est = float((wp[0, 3] - vx) / 0.1) if vx > 0 else 0.0
    vs_msg = make_vs_msg(ego, ax=ax_est, ay=0.0, tire_angle=0.0)
    traj_msg = wp_rel_to_traj_msg(wp)
    objs_no = PredictObjects()
    objs_with = lead_to_objs_msg(lead)

    pred_no, cio_no = run_and_pred(mpc, vs_msg, traj_msg, objs_no)
    pred_lead, cio_lead = run_and_pred(mpc, vs_msg, traj_msg, objs_with)

    # lead 시계열
    N = mpc.N_MPC + 1
    t = np.arange(N) * Ts
    lead_x = lead[0] + lead[4][0] * t
    lead_y = lead[1] + lead[4][1] * t

    ax.plot(ref_x, ref_y, 'k--', lw=1.2, label="reference")
    if pred_no is not None:
        ax.plot(pred_no[:, 0], pred_no[:, 1], 'C0-', lw=1.8, label="solved no-lead")
    if pred_lead is not None:
        ax.plot(pred_lead[:, 0], pred_lead[:, 1], 'C3-', lw=1.8, label="solved +lead")
    # ego
    ax.add_patch(Rectangle((-2.5, -1), 5, 2, ec='k', fc='lightgray', alpha=0.5))
    # lead time series
    ax.plot(lead_x, lead_y, 'r:', lw=0.8, alpha=0.5)
    for k in [0, N // 4, N // 2, 3 * N // 4, N - 1]:
        cx, cy = lead_x[k], lead_y[k]
        L, W = lead[2], lead[3]
        ax.add_patch(Rectangle((cx - L/2, cy - W/2), L, W,
                                ec='r', fc='salmon', alpha=0.35))
    # velocity arrow
    vx_l, vy_l = lead[4][0], lead[4][1]
    ax.annotate("", xy=(lead[0] + vx_l * 0.5, lead[1] + vy_l * 0.5),
                xytext=(lead[0], lead[1]),
                arrowprops=dict(arrowstyle="->", color='r', lw=1.5))

    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, max(60, ref_x.max() + 5))
    ax.set_ylim(-12, 12)
    # diff summary
    if cio_no is not None and cio_lead is not None:
        diff = cio_lead['acceleration'][:5] - cio_no['acceleration'][:5]
        diff_str = f"Δaccel[:5] mean={diff.mean():.2f} m/s²"
    else:
        diff_str = "N/A"
    cat = categorize(lead[0], lead[1])
    title = (f"[{cat}] lead=({lead[0]:.1f},{lead[1]:.1f}) vel=({lead[4][0]:.1f},{lead[4][1]:.1f}) | "
             f"ego vx={vx:.1f} | {diff_str}")
    ax.set_title(title, fontsize=9)


def main():
    z = np.load(COMPACT_NPZ)
    meta_list = json.loads(open(COMPACT_META).read())
    print(f"loaded {len(meta_list)} frames")

    buckets = pick_cases_by_category(meta_list, z, target_per_cat=2)
    cases = []
    for cat in ["in_path", "side", "far_side", "far_ahead"]:
        for m, lead in buckets[cat]:
            cases.append((cat, m, lead))
    print(f"selected {len(cases)} cases:")
    for cat, m, lead in cases:
        print(f"  [{cat}] {m['frame_id']}  lead=({lead[0]:.1f},{lead[1]:.1f})")

    mpc = MPC_Class()
    Ts = mpc.Ts_MPC

    n = len(cases)
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(16, 4 * nrows))
    axes = axes.flatten()
    for i, (cat, m, lead) in enumerate(cases):
        plot_case(axes[i], m, lead, z, mpc, Ts)
        if i == 0:
            axes[i].legend(loc="upper right", fontsize=8)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Lead car position vs MPC solved trajectory — SCC necessity by overlap",
                 fontsize=12)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "multi_cases_topdown.png")
    fig.savefig(out_png, dpi=110)
    print(f"\nsaved: {out_png}")


if __name__ == "__main__":
    main()
