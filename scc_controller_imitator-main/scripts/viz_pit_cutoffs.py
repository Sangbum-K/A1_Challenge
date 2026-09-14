"""피트끝(P)/피트머지(M) 검수용: dt0p1s 라인의 pit 구간을 index 라벨 달아 확대.
track 별 1 figure: 행=mild/high, 열=full(폴리곤+후보) / pit-zoom(index 라벨).
후보 P=첫 폴리곤블록 끝, M=마지막 폴리곤블록 시작 (사용자가 확인/수정)."""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MP

REPO = "/home/ailab-12/git/scc_controller_imitator"
TRACKS = ["sangam", "speedway"]


def load_dt(f):
    return np.array([[float(x) for x in l.split(";") if x.strip()]
                     for l in open(f) if l.strip() and not l.startswith("#")])


def blocks_inside(xy, poly):
    ins = MP(np.vstack([poly, poly[:1]])).contains_points(xy)
    bl, i, N = [], 0, len(xy)
    while i < N:
        if ins[i]:
            j = i
            while j < N and ins[j]: j += 1
            bl.append((i, j - 1)); i = j
        else: i += 1
    return bl


for trk in TRACKS:
    poly = np.array(json.load(open(f"{REPO}/data/control/pit_polygon_{trk}.json"))["points"])
    fig, axes = plt.subplots(2, 2, figsize=(17, 13))
    for r, mp in enumerate(["mild", "high"]):
        a = load_dt(f"{REPO}/data/control/raceline_{trk}_{mp}.csv"); xy = a[:, 1:3]; N = len(xy)
        bl = blocks_inside(xy, poly)
        P = bl[0][1] if bl else 0          # 첫 블록 끝 = 피트끝 후보
        M = bl[-1][0] if len(bl) > 1 else N - 1   # 마지막 블록 시작 = 피트머지 후보
        axF, axZ = axes[r, 0], axes[r, 1]
        # full
        sc = axF.scatter(xy[:, 0], xy[:, 1], c=np.arange(N), s=5, cmap="viridis")
        axF.plot(np.append(poly[:, 0], poly[0, 0]), np.append(poly[:, 1], poly[0, 1]), "-", color="#DC291E", lw=1.5)
        axF.scatter(*xy[0], c="lime", s=240, marker="*", edgecolor="k", zorder=7)
        axF.scatter(*xy[N - 1], c="black", s=160, marker="X", zorder=7)
        axF.annotate(f"START 0", xy[0], fontsize=11, color="green", weight="bold")
        axF.annotate(f"END {N-1}", xy[N - 1], fontsize=11, color="black", weight="bold")
        axF.scatter(*xy[P], c="red", s=90, marker="s", zorder=6)
        axF.scatter(*xy[M], c="magenta", s=90, marker="D", zorder=6)
        axF.annotate(f"P={P}", xy[P], fontsize=11, color="red", weight="bold")
        axF.annotate(f"M={M}", xy[M], fontsize=11, color="magenta", weight="bold")
        axF.set_title(f"{trk} {mp} (N={N}) full | blocks={bl}", fontsize=10)
        axF.set_aspect("equal"); axF.grid(alpha=0.3); plt.colorbar(sc, ax=axF, label="idx")
        # zoom: pit 영역(첫 블록 + 마지막 블록 주변) index 라벨
        idx_pit = list(range(max(0, bl[0][0]), min(N, P + 30)))
        idx_mrg = list(range(max(0, M - 20), min(N, (bl[-1][1] if bl else N) + 10)))
        sel = sorted(set(idx_pit + idx_mrg))
        sx, sy = xy[sel, 0], xy[sel, 1]
        axZ.plot(xy[:, 0], xy[:, 1], "-", color="#cccccc", lw=0.6)
        axZ.plot(np.append(poly[:, 0], poly[0, 0]), np.append(poly[:, 1], poly[0, 1]), "-", color="#DC291E", lw=1.5)
        axZ.scatter(sx, sy, c="#005195", s=12, zorder=4)
        for k in sel:
            if k % 20 == 0:    # 듬성히 (20 단위)
                axZ.annotate(str(k), xy[k], fontsize=8, color="#002060")
        axZ.scatter(*xy[0], c="lime", s=260, marker="*", edgecolor="k", zorder=7, label="START 0")
        if 0 <= N - 1 < N:
            axZ.scatter(*xy[N - 1], c="black", s=170, marker="X", zorder=7, label=f"END {N-1}")
        axZ.scatter(*xy[P], c="red", s=130, marker="s", zorder=6, label=f"P (pit end) ~{P}")
        axZ.scatter(*xy[M], c="magenta", s=130, marker="D", zorder=6, label=f"M (pit merge) ~{M}")
        pad = 25
        bx = np.r_[sx, xy[P, 0], xy[M, 0], xy[0, 0], xy[N - 1, 0]]
        by = np.r_[sy, xy[P, 1], xy[M, 1], xy[0, 1], xy[N - 1, 1]]
        axZ.set_xlim(bx.min() - pad, bx.max() + pad); axZ.set_ylim(by.min() - pad, by.max() + pad)
        axZ.set_aspect("equal"); axZ.grid(alpha=0.3); axZ.set_title(f"{trk} {mp} pit-zoom (idx labels /20)", fontsize=10)
        axZ.legend(fontsize=9, loc="best")
        print(f"[{trk} {mp}] N={N} blocks={bl} -> P~{P} M~{M}", flush=True)
    plt.tight_layout()
    out = f"{REPO}/outputs_multi/viz/pit_cutoffs_{trk}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close()
    print(f"[done] {out}")
