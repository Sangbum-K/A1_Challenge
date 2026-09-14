"""pit 경계 부근 off-line ego 점들의 최근접 i0 / in_pit / 추출 traj 변화 검증.
P(피트끝) 부근에 수직 sweep 으로 ego 를 깔고, ego→i0 할당 + 추출 40점을 본다.
경계 넘을 때(폴리곤 in/out) 검색범위가 [0,P]↔[P,M] 로 바뀌는 효과 확인."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.utils import raceline_dt0p1s as R

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); DATA = f"{REPO}/data"


def heading_at(xy, i):
    j = min(i + 1, len(xy) - 1); k = max(j - 1, 0)
    return float(np.arctan2(xy[j, 1] - xy[k, 1], xy[j, 0] - xy[k, 0]))


def main(track, mp):
    m = R.load_map(track, mp, DATA); xy = m["xy"]; P = m["P"]
    fig, axes = plt.subplots(1, 2, figsize=(18, 8.5))
    # 전체 + 경계 zoom
    for ax, zoom in zip(axes, [False, True]):
        ax.scatter(xy[:, 0], xy[:, 1], c=np.arange(m["N"]), s=4, cmap="Greys", alpha=0.4)
        pl = m["polygon"]; ax.plot(np.append(pl[:, 0], pl[0, 0]), np.append(pl[:, 1], pl[0, 1]), "-", color="#DC291E", lw=1.5)
        ax.scatter(*xy[P], c="red", s=120, marker="s", zorder=6); ax.annotate(f"P={P}", xy[P], color="red", fontsize=11)
        ax.scatter(*xy[m["M"]], c="magenta", s=90, marker="D", zorder=6)
        ax.scatter(*xy[0], c="lime", s=160, marker="*", edgecolor="k", zorder=6)
        # P 부근 longitudinal 위치마다 수직 sweep
        for ci in [P - 40, P - 15, P, P + 15, P + 40]:
            ci = max(1, min(ci, m["N"] - 2))
            th = heading_at(xy, ci); nx, ny = -np.sin(th), np.cos(th)   # normal
            for off in np.linspace(-12, 12, 9):
                ex, ey = xy[ci, 0] + nx * off, xy[ci, 1] + ny * off
                i0, in_pit, _ = R.nearest_idx(m, ex, ey)
                col = "#DC291E" if in_pit else "#005195"
                ax.scatter([ex], [ey], c=col, s=18, zorder=7)
                ax.plot([ex, xy[i0, 0]], [ey, xy[i0, 1]], "-", color=col, lw=0.5, alpha=0.6, zorder=5)
        ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_xlabel("x"); ax.set_ylabel("y")
        if zoom:
            pad = 70; ax.set_xlim(xy[P, 0] - pad, xy[P, 0] + pad); ax.set_ylim(xy[P, 1] - pad, xy[P, 1] + pad)
            ax.set_title(f"{track} {mp} pit-boundary zoom (red=in_pit→[0,P], blue=track→[P,M]; line=ego→i0)", fontsize=9)
        else:
            ax.set_title(f"{track} {mp}: off-line ego near P nearest-assignment", fontsize=10)
    plt.tight_layout()
    out = f"{REPO}/outputs_multi/viz/pit_boundary_{track}_{mp}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close()
    print(f"[done] {out}")


if __name__ == "__main__":
    for t in ["sangam", "speedway"]:
        main(t, "high")
