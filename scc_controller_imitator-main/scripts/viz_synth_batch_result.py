"""synth_batch.cpp 의 .scc.txt 출력 시각화.

3 panel:
  (1) BEV: original trajectory, scc speed-colored trajectory, lead 위치
  (2) speed profile vs s (v_orig vs v_scc, gap-to-lead overlay)
  (3) controller output: lateral steer / longitudinal ax,trq,pedals

사용:
  python scripts/viz_synth_batch_result.py /tmp/scc_synth_v2/260513_48__t0015395.scc.txt
"""
from __future__ import annotations
import sys, os, math, argparse
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def parse_scc_txt(path):
    """파일 → dict with header / traj_table / cmd / long_traj / lat_traj."""
    out = {
        "header": {}, "in_lane_obj_ids": [],
        "traj": [], "cmd": {},
        "long_traj": [], "lat_traj": [],
    }
    cur = None     # 'traj' / 'long_traj' / 'lat_traj'
    for line in open(path):
        line = line.rstrip()
        if not line: continue
        if line.startswith("#"):
            # comment, but may switch section
            if "ctrl" in line.lower() and "vx" in line.lower():
                # long_traj or lat_traj header — distinguished by previous N
                pass
            continue
        if line.startswith("in_lane_obj_ids:"):
            ids = line.split(":", 1)[1].strip().split()
            out["in_lane_obj_ids"] = [int(x) for x in ids if x]
            continue
        if "=" in line and not line[0].isdigit():
            k, v = line.split("=", 1)
            k = k.strip(); v = v.strip()
            if k == "N":
                out["header"][k] = int(v); cur = "traj"
            elif k == "long_traj_N":
                out["header"][k] = int(v); cur = "long_traj"
            elif k == "lat_traj_N":
                out["header"][k] = int(v); cur = "lat_traj"
            elif k.startswith("run="):
                # first comment line: "# run=... tick=... dt_per_tick_ms=..."
                pass
            else:
                try:
                    out["cmd"][k] = float(v)
                except ValueError:
                    out["cmd"][k] = v
            continue
        # numeric row
        parts = line.split()
        if not parts: continue
        try:
            nums = [float(x) for x in parts]
        except ValueError:
            continue
        if cur == "traj":
            out["traj"].append(nums)
        elif cur == "long_traj":
            out["long_traj"].append(nums)
        elif cur == "lat_traj":
            out["lat_traj"].append(nums)
    # post: parse first line "# run=... tick=... dt_per_tick_ms=..."
    with open(path) as f:
        first = f.readline().lstrip("# ").strip()
        for kv in first.split():
            if "=" in kv:
                k, v = kv.split("=", 1)
                try:
                    out["header"][k] = float(v) if "." in v else int(v) if v.isdigit() else v
                except: out["header"][k] = v
    return out


def viz_one(path, out_path):
    d = parse_scc_txt(path)
    traj = np.array(d["traj"]) if d["traj"] else np.zeros((0, 7))
    # cols: i cum_s t_i v_orig v_scc lead_s gap
    if traj.size == 0:
        print(f"  empty: {path}")
        return False

    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.5, 1.0, 1.0, 0.6],
                            hspace=0.4, wspace=0.25)

    # (0, 0) BEV — trajectory + lead 위치 marker
    # ego-local (cum_s, 0) 라인은 진짜 path 가 아니라 longitudinal 위치만. real x,y 는 별도 인풋이 없음.
    # → cum_s 만으로 1-D 시각화: x=cum_s, y=0
    ax_traj1d = fig.add_subplot(gs[0, :])
    cum_s = traj[:, 1]
    v_orig = traj[:, 3]
    v_scc  = traj[:, 4]
    # speed 색으로 점 표시
    scat = ax_traj1d.scatter(cum_s, np.zeros_like(cum_s),
                                c=v_scc, cmap="viridis", s=60, vmin=0, vmax=max(35, v_orig.max()))
    plt.colorbar(scat, ax=ax_traj1d, label="v_scc (m/s)", fraction=0.03, pad=0.02)
    # original speed 비교 (반투명)
    ax_traj1d.scatter(cum_s, np.zeros_like(cum_s) + 0.15,
                        c=v_orig, cmap="viridis", s=30, alpha=0.4,
                        vmin=0, vmax=max(35, v_orig.max()), marker='s')
    ax_traj1d.text(cum_s[0], +0.5, "wp speed (top=v_orig, bottom=v_scc)", fontsize=9)
    # lead 위치 (lead_s) 표시
    lead_s_col = traj[:, 5]
    gap_col    = traj[:, 6]
    valid = ~np.isnan(lead_s_col)
    if valid.any():
        ls = lead_s_col[valid]
        ax_traj1d.scatter(ls, np.zeros_like(ls) - 0.3,
                            c="red", marker="v", s=80, label="lead_s_at_t(wp)")
        for i, lsi in zip(np.where(valid)[0][:3], ls[:3]):
            ax_traj1d.annotate(f"wp{i}→{lsi:.1f}", (lsi, -0.3),
                                 xytext=(lsi, -0.6), fontsize=8,
                                 ha="center", arrowprops=dict(arrowstyle="->", color="red", lw=0.5))
    ax_traj1d.set_xlim(-2, max(cum_s) + 5)
    ax_traj1d.set_ylim(-1.0, 1.0)
    ax_traj1d.set_xlabel("s (m, cumulative arc length from ego)")
    ax_traj1d.set_yticks([])
    ax_traj1d.set_title(
        f"{d['header'].get('run','?')} tick={d['header'].get('tick','?')}   "
        f"in_lane_obj_ids={d['in_lane_obj_ids']}   N={d['header'].get('N','?')}", fontsize=11)
    ax_traj1d.legend(loc="upper right", fontsize=9)
    ax_traj1d.grid(True, alpha=0.3)

    # (1, 0) speed profile + (1, 1) gap profile
    ax_v = fig.add_subplot(gs[1, 0])
    ax_v.plot(cum_s, v_orig, "C0--", label="v_orig", lw=1.5)
    ax_v.plot(cum_s, v_scc, "C3-", label="v_scc", lw=2)
    ax_v.set_xlabel("s (m)"); ax_v.set_ylabel("speed (m/s)")
    ax_v.legend(loc="lower right"); ax_v.grid(True, alpha=0.3)
    ax_v.set_title("speed profile (v_orig vs SCC-replanned)")

    ax_g = fig.add_subplot(gs[1, 1])
    if valid.any():
        ax_g.plot(cum_s[valid], gap_col[valid], "C2-o", label="gap to lead")
        ax_g.axhline(3.0, color="gray", ls=":", label="s_min=3m")
    else:
        ax_g.text(0.5, 0.5, "no in-lane lead", ha="center", va="center",
                   transform=ax_g.transAxes)
    ax_g.set_xlabel("s (m)"); ax_g.set_ylabel("gap (m)")
    ax_g.legend(); ax_g.grid(True, alpha=0.3)
    ax_g.set_title("longitudinal gap to closest in-lane lead")

    # (2, 0) longitudinal control trajectory (if any) — speed, accel
    long_traj = np.array(d["long_traj"]) if d["long_traj"] else np.zeros((0, 8))
    ax_lng = fig.add_subplot(gs[2, 0])
    if long_traj.size > 0:
        # cols: i time vx ax steer trq accel_pedal brake_pedal
        t = long_traj[:, 1]
        ax_lng.plot(t, long_traj[:, 2], "C0-", label="cmd vx (m/s)")
        ax_lng_2 = ax_lng.twinx()
        ax_lng_2.plot(t, long_traj[:, 3], "C3-", label="cmd ax (m/s²)")
        ax_lng.set_xlabel("t (s)"); ax_lng.set_ylabel("vx", color="C0")
        ax_lng_2.set_ylabel("ax", color="C3")
        ax_lng.set_title("LongitudinalControl 출력 (control_trajectory)")
        ax_lng.legend(loc="upper left", fontsize=9)
        ax_lng_2.legend(loc="upper right", fontsize=9)
        ax_lng.grid(True, alpha=0.3)
    else:
        ax_lng.text(0.5, 0.5, "no long_traj output (N=0)", ha="center", va="center",
                     transform=ax_lng.transAxes)
        ax_lng.set_title("LongitudinalControl 출력 — empty")

    # (2, 1) lateral control trajectory (if any) — steer
    lat_traj = np.array(d["lat_traj"]) if d["lat_traj"] else np.zeros((0, 5))
    ax_lat = fig.add_subplot(gs[2, 1])
    if lat_traj.size > 0 and lat_traj.shape[0] > 2:
        t = lat_traj[:, 1]
        ax_lat.plot(t, lat_traj[:, 4], "C0-", label="cmd steer (rad)")
        ax_lat.set_xlabel("t (s)"); ax_lat.set_ylabel("steer (rad)")
        ax_lat.legend(); ax_lat.grid(True, alpha=0.3)
        ax_lat.set_title("LateralControl 출력 (control_trajectory)")
    else:
        ax_lat.text(0.5, 0.5, f"lat_traj N={lat_traj.shape[0]}\n(controller FALLBACK 가능)",
                     ha="center", va="center", transform=ax_lat.transAxes,
                     color="red", fontsize=11)
        ax_lat.set_title("LateralControl 출력 — fallback?")

    # (3, :) instant command summary
    ax_cmd = fig.add_subplot(gs[3, :])
    ax_cmd.axis("off")
    cmd = d["cmd"]
    txt = "  ".join([
        f"lat_steer={cmd.get('lat_steer_tire_deg', float('nan')):+.2f}°",
        f"long_ax={cmd.get('long_ax', float('nan')):+.3f} m/s²",
        f"long_trq={cmd.get('long_trq', float('nan')):+.1f} Nm",
        f"accel_pedal={cmd.get('long_accel_pedal', float('nan')):.3f}",
        f"brake_pedal={cmd.get('long_brake_pedal', float('nan')):.3f}",
        f"long_traj_N={int(cmd.get('long_traj_N', 0))}",
    ])
    ax_cmd.text(0.02, 0.5, "instant command (t=0):  " + txt,
                  fontsize=11, family="monospace",
                  transform=ax_cmd.transAxes, va="center")

    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help=".scc.txt 파일들")
    ap.add_argument("--out", default="outputs_multi/synth_batch_viz")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for fp in args.files:
        name = Path(fp).stem.replace(".scc", "")
        out_path = Path(args.out) / f"{name}.png"
        ok = viz_one(fp, out_path)
        if ok:
            print(f"  → {out_path}")


if __name__ == "__main__":
    main()
