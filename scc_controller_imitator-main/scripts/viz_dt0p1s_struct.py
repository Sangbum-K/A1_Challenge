"""dt0p1s raceline 을 index 색으로 그려 구조(main lap vs pit-out tail) + polygon 확인. cutoff 판정용."""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/home/ailab-12/git/scc_controller_imitator"
JOBS = [
    ("sangam",   f"{REPO}/data/control/raceline_sangam_dt0p1s.csv",   f"{REPO}/data/control/pit_polygon_sangam.json"),
    ("speedway", f"{REPO}/data/control/raceline_speedway_dt0p1s.csv", f"{REPO}/data/control/pit_polygon_speedway.json"),
]


def load_dt(f):
    return np.array([[float(x) for x in l.split(";") if x.strip()]
                     for l in open(f) if l.strip() and not l.startswith("#")])


fig, axes = plt.subplots(1, 2, figsize=(17, 8))
for ax, (trk, rf, pf) in zip(axes, JOBS):
    a = load_dt(rf); xy = a[:, 1:3]; N = len(xy)
    poly = np.array(json.load(open(pf))["points"])
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=np.arange(N), s=6, cmap="viridis")
    ax.plot(np.append(poly[:, 0], poly[0, 0]), np.append(poly[:, 1], poly[0, 1]),
            "-", color="#DC291E", lw=2.0, label="pit polygon")
    ax.scatter([xy[0, 0]], [xy[0, 1]], c="lime", s=120, marker="*", zorder=6, label="idx 0 (start)")
    ax.scatter([xy[-1, 0]], [xy[-1, 1]], c="red", s=120, marker="X", zorder=6, label=f"idx {N-1} (end)")
    # pit polygon 안에 들어가는 raceline index 구간
    from matplotlib.path import Path as MP
    inside = MP(np.vstack([poly, poly[:1]])).contains_points(xy)
    idxs = np.where(inside)[0]
    rng = f"{idxs.min()}~{idxs.max()}" if len(idxs) else "none"
    ax.set_title(f"{trk} dt0p1s (N={N}) | pit-polygon inside idx {rng} ({len(idxs)})", fontsize=11)
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(sc, ax=ax, label="raceline index")
    print(f"[{trk}] N={N} | inside-polygon idx {rng} count={len(idxs)}", flush=True)
plt.tight_layout()
out = f"{REPO}/outputs_multi/viz/dt0p1s_struct.png"
plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
print(f"[done] {out}")
