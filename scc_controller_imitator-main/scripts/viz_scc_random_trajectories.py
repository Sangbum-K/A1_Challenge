"""전방 차량 있는 frame 에서 30 random trajectory 에 SCC replan 적용 → 시각화.

SCC replan 은 cpp SccSpeedReplanner 의 Python mirror (viz 전용).
파라미터: t_headway=1.0s, s_min=3.0m, lane_half_width=2.0m, max_decel=15 m/s², max_accel=10.

사용:
  ~/miniforge3/envs/ml_con/bin/python scripts/viz_scc_random_trajectories.py [--frame-id <fid>] [--n 30]
"""
from __future__ import annotations
import os, sys, json, pickle, argparse
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]

POOL_NPZ = ROOT / "data/v0237_traj_pool.npz"
META_JSON = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/multiagnet_compact_meta.json"
COMPACT_NPZ = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/multiagnet_compact.npz"

T_HEADWAY = 1.0
S_MIN = 3.0
LANE_HALF = 2.0
MAX_DECEL = 15.0
MAX_ACCEL = 10.0
DT_OBJ = 0.1

# BEV bbox → ego (synth_k1 convention)
BEV_X_MAX = 80.0; BEV_Y_MIN = -25.0; BEV_CELL = 0.5
def bbox_to_ego(b):
    cx, cy, w, h = b
    return (BEV_X_MAX - cy*BEV_CELL, BEV_Y_MIN + cx*BEV_CELL,
            h*BEV_CELL, w*BEV_CELL)


def project_to_frenet(px, py, ref_x, ref_y, cum_s):
    """Nearest-neighbor frenet projection — synth_k1 / cpp 와 동일."""
    if len(ref_x) < 2:
        return 0.0, 0.0
    d2 = (ref_x - px)**2 + (ref_y - py)**2
    k = int(np.argmin(d2))
    s_val = cum_s[k]
    if k + 1 < len(ref_x):
        tx, ty = ref_x[k+1] - ref_x[k], ref_y[k+1] - ref_y[k]
    else:
        tx, ty = ref_x[k] - ref_x[k-1], ref_y[k] - ref_y[k-1]
    tn = np.hypot(tx, ty)
    if tn < 1e-6:
        return s_val, 0.0
    tx, ty = tx/tn, ty/tn
    rx, ry = px - ref_x[k], py - ref_y[k]
    n_val = tx*ry - ty*rx
    return s_val, n_val


def scc_replan(traj_xy_yaw_speed, objs_state_xy):
    """traj: (N, 4) [x, y, yaw, speed].  objs_state_xy: list of (K, 2) per object.
       반환: v_new (N,), debug dict.
    """
    N = len(traj_xy_yaw_speed)
    ref_x = traj_xy_yaw_speed[:, 0]
    ref_y = traj_xy_yaw_speed[:, 1]
    v_orig = traj_xy_yaw_speed[:, 3].copy()
    # cum_s
    cum_s = np.zeros(N)
    for i in range(1, N):
        cum_s[i] = cum_s[i-1] + np.hypot(ref_x[i]-ref_x[i-1], ref_y[i]-ref_y[i-1])
    s_max = cum_s[-1]
    # arrival time per wp
    t_arr = np.zeros(N)
    for i in range(1, N):
        v_avg = max(0.5, 0.5*(v_orig[i-1] + v_orig[i]))
        t_arr[i] = t_arr[i-1] + (cum_s[i] - cum_s[i-1]) / v_avg

    # objects: lane filter + (t, s) trajectory
    leads = []   # each = (t_arr (K,), s_arr (K,) with nan if out-of-lane)
    for obj_states in objs_state_xy:
        K = len(obj_states)
        ts = np.array([k*DT_OBJ for k in range(K)])
        ss = np.full(K, np.nan)
        any_in = False
        for k in range(K):
            sv, nv = project_to_frenet(obj_states[k, 0], obj_states[k, 1], ref_x, ref_y, cum_s)
            if abs(nv) <= LANE_HALF and 0 <= sv <= s_max:
                ss[k] = sv
                any_in = True
        if any_in: leads.append((ts, ss))

    # per wp: closest in-lane lead's s_at_t_i, gap → v_safe
    v_new = v_orig.copy()
    gap_arr = np.full(N, np.nan)
    lead_s_arr = np.full(N, np.nan)
    for i in range(N):
        t_i = t_arr[i]
        closest_s = np.nan
        for (ts, ss) in leads:
            if t_i <= ts[0]: sat = ss[0]
            elif t_i >= ts[-1]: sat = ss[-1]
            else:
                k1 = int(np.searchsorted(ts, t_i, side='right'))
                k0 = k1 - 1
                if np.isnan(ss[k0]) or np.isnan(ss[k1]):
                    continue
                a = (t_i - ts[k0]) / (ts[k1] - ts[k0])
                sat = ss[k0] + a * (ss[k1] - ss[k0])
            if np.isnan(sat): continue
            if sat > cum_s[i]:
                if np.isnan(closest_s) or sat < closest_s:
                    closest_s = sat
        if not np.isnan(closest_s):
            gap = closest_s - cum_s[i]
            v_safe = (gap - S_MIN) / T_HEADWAY
            v_new[i] = min(v_orig[i], max(0.0, v_safe))
            gap_arr[i] = gap
            lead_s_arr[i] = closest_s

    # backward pass: decel limit
    for i in range(N-2, -1, -1):
        dt_i = t_arr[i+1] - t_arr[i]
        if dt_i > 0:
            lim = v_new[i+1] + MAX_DECEL * dt_i
            if v_new[i] > lim: v_new[i] = lim
    # forward pass: accel limit
    for i in range(1, N):
        dt_i = t_arr[i] - t_arr[i-1]
        if dt_i > 0:
            lim = v_new[i-1] + MAX_ACCEL * dt_i
            if v_new[i] > lim: v_new[i] = lim

    return v_new, {"cum_s": cum_s, "t_arr": t_arr, "gap": gap_arr, "lead_s": lead_s_arr,
                    "v_orig": v_orig, "v_new": v_new}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-id", type=str, default=None,
                    help="특정 frame_id. default: 가장 가까운 lead 자동 선택")
    ap.add_argument("--n", type=int, default=30, help="random trajectory 수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="outputs_multi/scc_random_30.png")
    args = ap.parse_args()

    # 1) frame 선택
    meta = json.load(open(META_JSON))
    if args.frame_id:
        m = next((x for x in meta if x["frame_id"] == args.frame_id), None)
        if m is None: raise SystemExit(f"frame_id 없음: {args.frame_id}")
    else:
        # closest lead 자동
        best = None
        for x in meta:
            for o in x["objects"]:
                ex, ey, _, _ = bbox_to_ego(o["bbox"])
                if 1.5 < ex < 15 and abs(ey) < 8:
                    if best is None or ex < best[1]:
                        best = (x, ex)
                    break
        m = best[0]
        print(f"auto-pick: {m['frame_id']} (closest lead at {best[1]:.1f}m)")

    # 2) 객체 (ego-local, 등속 propagation 40 steps)
    objs_state = []
    print("\nobjects in frame:")
    for o in m["objects"]:
        ex, ey, L, W = bbox_to_ego(o["bbox"])
        # velocity 는 jiwon_e2e 의 buggy version → 그냥 정지로 가정 (시각화 목적)
        # 실 pipeline 은 cpp 가 raw 에서 fix 적용
        v = o.get("velocity", [0.0, 0.0])
        # synth_k1 의 lead_to_objs_msg 처럼 등속 propagate
        states = np.array([(ex + v[0]*k*DT_OBJ, ey + v[1]*k*DT_OBJ) for k in range(40)])
        objs_state.append(states)
        print(f"  obj at ({ex:.1f}, {ey:+.1f}) L={L:.1f} W={W:.1f}  vel=({v[0]:+.2f},{v[1]:+.2f})")

    # 3) random pool trajectory N 개 + logged 1 개 (compact npz 의 wp_rel_gt 누적)
    pool = np.load(POOL_NPZ)
    trajs_pool = pool["trajs"]    # (P, 40, 4) [x, y, yaw, speed], ego-local
    v0s = pool["v0s"]
    z = np.load(COMPACT_NPZ)
    wp_rel = z[f"s{m['i']}__wp_rel_gt"]
    ego = z[f"s{m['i']}__ego_pose"]
    ego_vx = float(ego[3])
    print(f"ego vx={ego_vx:.2f} m/s")
    # logged: cumulative
    logged = np.zeros((30, 4))
    logged[:, 0] = np.cumsum(wp_rel[:, 0])
    logged[:, 1] = np.cumsum(wp_rel[:, 1])
    logged[:, 2] = np.cumsum(wp_rel[:, 2])
    logged[:, 3] = wp_rel[:, 3]

    rng = np.random.RandomState(args.seed)
    # vx_tolerance 5 within ego_vx
    valid = np.where(np.abs(v0s - ego_vx) < 5.0)[0]
    if len(valid) < args.n:
        valid = np.arange(len(trajs_pool))
    pick = rng.choice(valid, size=min(args.n, len(valid)), replace=False)
    pool_set = [trajs_pool[k] for k in pick]
    all_trajs = [("logged", logged)] + [(f"rand_{k}", trajs_pool[k]) for k in pick]
    print(f"trajectories: 1 logged + {len(pool_set)} random")

    # 4) SCC replan each
    results = []
    for tag, traj in all_trajs:
        v_new, dbg = scc_replan(traj, objs_state)
        results.append((tag, traj, v_new, dbg))

    # 5) viz
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1.0, 1.0], width_ratios=[2, 1.2],
                            hspace=0.30, wspace=0.20)

    # (0, 0) BEV: all trajectories + objects
    ax = fig.add_subplot(gs[0, 0])
    ax.add_patch(Rectangle((-2.5, -1), 5, 2, ec="k", fc="lightgray", alpha=0.7))
    # objects
    for o in m["objects"]:
        ex, ey, L, W = bbox_to_ego(o["bbox"])
        if not (-2 < ex < 100 and abs(ey) < 25): continue
        in_lane_initial = abs(ey) <= LANE_HALF and 0 <= ex   # 초기 시점 가까운 추정
        ax.add_patch(Rectangle((ex-L/2, ey-W/2), L, W,
                                ec='red' if in_lane_initial else 'gray',
                                fc='salmon' if in_lane_initial else 'lightgray',
                                alpha=0.7 if in_lane_initial else 0.4))
        ax.text(ex, ey+W/2+0.3, f"({ex:.0f},{ey:+.1f})", fontsize=8, ha="center",
                  color='red' if in_lane_initial else 'dimgray')
    # all trajectories
    for tag, traj, v_new, dbg in results:
        is_logged = (tag == "logged")
        ax.plot(traj[:, 0], traj[:, 1], color="black" if is_logged else "C0",
                 lw=2 if is_logged else 0.7, alpha=1.0 if is_logged else 0.35,
                 label="logged" if is_logged else None)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 100); ax.set_ylim(-15, 15)
    ax.set_xlabel("x [m] forward"); ax.set_ylabel("y [m] left")
    ax.set_title(f"{m['frame_id']}  —  logged + {len(pool_set)} random trajectories")
    ax.legend()

    # (0, 1) summary text
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis("off")
    info_lines = [
        f"frame:  {m['frame_id']}",
        f"i={m['i']}  ego vx={ego_vx:.2f} m/s",
        f"",
        f"SCC config:",
        f"  T_headway = {T_HEADWAY} s",
        f"  s_min     = {S_MIN} m",
        f"  lane_half = ±{LANE_HALF} m",
        f"  max_decel = {MAX_DECEL} m/s²",
        f"  max_accel = {MAX_ACCEL} m/s²",
        f"",
        f"objects (BEV):  {len(objs_state)}",
        f"trajectories: 1 logged + {len(pool_set)} random",
    ]
    ax_info.text(0.02, 0.95, "\n".join(info_lines), va="top", fontsize=11,
                  family="monospace", transform=ax_info.transAxes)

    # (1, :) all v_orig vs v_scc on s axis
    ax_v = fig.add_subplot(gs[1, :])
    for tag, traj, v_new, dbg in results:
        s = dbg["cum_s"]
        is_logged = (tag == "logged")
        ax_v.plot(s, dbg["v_orig"], color="gray", lw=0.5, alpha=0.3)
        ax_v.plot(s, v_new, color="black" if is_logged else "C3",
                   lw=2 if is_logged else 0.7, alpha=1.0 if is_logged else 0.35)
    ax_v.set_xlabel("s (m, cumulative arc)")
    ax_v.set_ylabel("speed (m/s)")
    ax_v.set_title("v_orig (gray, faint) vs v_scc (black=logged, red=random) — SCC replan 결과")
    ax_v.grid(True, alpha=0.3)

    # (2, :) speed reduction histogram
    ax_h = fig.add_subplot(gs[2, :])
    reductions = []
    n_changed = 0
    for tag, traj, v_new, dbg in results:
        diff = dbg["v_orig"] - v_new
        max_red = float(diff.max())
        reductions.append(max_red)
        if max_red > 0.1: n_changed += 1
    bins = np.arange(0, max(20, max(reductions)+1), 1.0)
    ax_h.hist(reductions, bins=bins, edgecolor="black")
    ax_h.set_xlabel("max speed reduction (m/s)  — SCC 가 가장 강하게 brake 한 wp 의 (v_orig - v_scc)")
    ax_h.set_ylabel("trajectory count")
    ax_h.set_title(f"trajectory 별 최대 감속폭 분포   ({n_changed}/{len(results)} trajectory 가 SCC 영향 받음)")
    ax_h.grid(True, alpha=0.3)

    plt.suptitle(f"30 random trajectory SCC replan — {m['frame_id']}", fontsize=13, y=0.995)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
