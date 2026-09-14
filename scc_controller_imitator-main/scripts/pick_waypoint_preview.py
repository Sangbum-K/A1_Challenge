"""지도에서 임의 점 클릭 → 추출된 40 waypoint 표시 (raceline_dt0p1s 검증).
ego_yaw 는 최근접점 raceline heading 으로 가정. 추출된 ego-frame 40점을 world 로 역변환해 overlay.

  python scripts/pick_waypoint_preview.py --track sangam --map high           # 인터랙티브(클릭)
  python scripts/pick_waypoint_preview.py --track sangam --map high --selftest # 클릭없이 샘플점 static 검증
좌클릭=그 점을 ego 로 가정해 40 waypoint 추출/표시. u 지움, q 종료.
"""
import os, sys, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
from src.utils import raceline_dt0p1s as R

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = f"{REPO}/data"


def heading_at(xy, i):
    j = min(i + 1, len(xy) - 1); k = max(j - 1, 0)
    return float(np.arctan2(xy[j, 1] - xy[k, 1], xy[j, 0] - xy[k, 0]))


def ego_to_world(traj, ex, ey, e_yaw):
    c, s = np.cos(e_yaw), np.sin(e_yaw)
    wx = ex + c * traj[:, 0] - s * traj[:, 1]
    wy = ey + s * traj[:, 0] + c * traj[:, 1]
    return wx, wy


def draw_base(ax, m):
    xy = m["xy"]; N = m["N"]
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=np.arange(N), s=5, cmap="Greys", alpha=0.5, zorder=1)
    if m["polygon"] is not None:
        p = m["polygon"]; ax.plot(np.append(p[:, 0], p[0, 0]), np.append(p[:, 1], p[0, 1]), "-", color="#DC291E", lw=1.3, zorder=3)
    ax.scatter(*xy[0], c="lime", s=200, marker="*", edgecolor="k", zorder=6)
    ax.scatter(*xy[N - 1], c="black", s=130, marker="X", zorder=6)
    ax.scatter(*xy[m["P"]], c="red", s=80, marker="s", zorder=6); ax.annotate(f"P={m['P']}", xy[m["P"]], color="red", fontsize=10)
    ax.scatter(*xy[m["M"]], c="magenta", s=80, marker="D", zorder=6); ax.annotate(f"M={m['M']}", xy[m["M"]], color="magenta", fontsize=10)
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_xlabel("x"); ax.set_ylabel("y")


def extract_at(m, ex, ey):
    i0, in_pit, last = R.nearest_idx(m, ex, ey)
    e_yaw = heading_at(m["xy"], i0)
    traj, meta = R.sample_traj40(m, ex, ey, e_yaw)
    wx, wy = ego_to_world(traj, ex, ey, e_yaw)
    return traj, meta, e_yaw, wx, wy


def plot_extraction(ax, m, ex, ey, traj, meta, wx, wy, persist):
    arts = []
    arts.append(ax.scatter([ex], [ey], c="blue", s=130, marker="*", zorder=8))
    arts.append(ax.scatter(*m["xy"][meta["i0"]], c="orange", s=90, zorder=8))
    nreal = 40 - meta["n_virtual"]
    sc = ax.scatter(wx[:nreal], wy[:nreal], c=np.arange(nreal), cmap="viridis", s=28, zorder=7, edgecolor="k", lw=0.3)
    arts.append(sc)
    if meta["n_virtual"] > 0:
        arts.append(ax.scatter(wx[nreal:], wy[nreal:], c="red", marker="x", s=45, zorder=7))
    arts.append(ax.annotate(f"i0={meta['i0']} {'PIT' if meta['in_pit'] else 'TRACK'} virt={meta['n_virtual']}",
                            (ex, ey), fontsize=10, color="blue", weight="bold"))
    print(f"[extract] ego=({ex:.1f},{ey:.1f}) i0={meta['i0']} in_pit={meta['in_pit']} "
          f"n_virtual={meta['n_virtual']} v[0]={traj[0,2]:.1f} v[-1]={traj[-1,2]:.1f}", flush=True)
    if not persist:
        for a in arts: a.remove()
    return arts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=["sangam", "speedway"])
    ap.add_argument("--map", default="high", choices=["mild", "high"])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    m = R.load_map(args.track, args.map, DATA)
    print(f"[map] {args.track} {args.map} N={m['N']} P={m['P']} M={m['M']}", flush=True)

    if args.selftest:
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # 샘플 ego: 트랙 시작/중간/M근처/끝근처 + pit 내부 1점
        xy = m["xy"]
        cand = [m["P"] + 50, (m["P"] + m["M"]) // 2, m["M"] - 5, m["M"] + 1, max(1, m["P"] // 2)]
        fig, axes = plt.subplots(1, len(cand), figsize=(5 * len(cand), 6))
        for ax, ci in zip(axes, cand):
            ex, ey = xy[ci, 0] + 1.0, xy[ci, 1] - 1.0   # 약간 오프셋
            draw_base(ax, m)
            traj, meta, eyaw, wx, wy = extract_at(m, ex, ey)
            plot_extraction(ax, m, ex, ey, traj, meta, wx, wy, persist=True)
            pad = 60; ax.set_xlim(min(wx.min(), ex) - pad, max(wx.max(), ex) + pad)
            ax.set_ylim(min(wy.min(), ey) - pad, max(wy.max(), ey) + pad)
            ax.set_title(f"ego near idx {ci}", fontsize=10)
        plt.tight_layout()
        out = f"{REPO}/outputs_multi/viz/waypoint_preview_{args.track}_{args.map}.png"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, dpi=110, bbox_inches="tight"); plt.close()
        print(f"[selftest done] {out}")
        return

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(13, 11)); draw_base(ax, m)
    title = ax.set_title(f"[{args.track} {args.map}] 좌클릭=ego 가정 40 waypoint 추출 (u지움 q종료)")
    state = {"arts": []}

    def on_click(event):
        if event.inaxes != ax or event.xdata is None: return
        tb = getattr(fig.canvas, "toolbar", None)
        if tb is not None and getattr(tb, "mode", ""): return
        if event.button != 1: return
        for a in state["arts"]:
            try: a.remove()
            except Exception: pass
        traj, meta, eyaw, wx, wy = extract_at(m, event.xdata, event.ydata)
        state["arts"] = plot_extraction(ax, m, event.xdata, event.ydata, traj, meta, wx, wy, persist=True)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "u":
            for a in state["arts"]:
                try: a.remove()
                except Exception: pass
            state["arts"] = []; fig.canvas.draw_idle()
        elif event.key == "q": plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


if __name__ == "__main__":
    main()
