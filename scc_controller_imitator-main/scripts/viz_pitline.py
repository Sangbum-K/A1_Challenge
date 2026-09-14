"""pit-line(fitout) JSON 을 raceline(main+pit) 위에 overlay 시각화. 와꾸 확인용.
라벨 영문(폰트 깨짐 방지)."""
import os, sys, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

UP = "/home/ailab-12/.claude/uploads/04d26f57-da4a-4388-9164-8d6bba8fcb50"
REPO = "/home/ailab-12/git/scc_controller_imitator"
JOBS = [  # (track, pitline_json, raceline_csv)
    ("sangam",   f"{UP}/5d9c92a6-fitout_points_output_sangam.json", f"{REPO}/data/control/raceline_sangam.csv"),
    ("speedway", f"{UP}/6de40f24-fitout_points_output_yongin.json", f"{REPO}/data/control/raceline_speedway.csv"),
]


def load_pitline(p):
    d = json.load(open(p))[0]
    pts = np.array(d["points"], dtype=np.float64)
    return pts  # (N,3) x,y,z


def load_raceline(p):
    rows, main_n = [], None
    for line in open(p):
        line = line.strip()
        if line.startswith("#"):
            if "main_segment_rows" in line:
                try: main_n = int(line.split("=")[1].split("(")[0].strip())
                except Exception: pass
            continue
        if not line: continue
        v = [float(x) for x in line.split(";") if x.strip()]
        if len(v) >= 3: rows.append(v)
    arr = np.array(rows, dtype=np.float64)
    return arr, main_n


fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
for ax, (trk, pj, rc) in zip(axes, JOBS):
    pit = load_pitline(pj)
    arr, main_n = load_raceline(rc)
    rx, ry = arr[:, 1], arr[:, 2]
    n = main_n if main_n else len(arr)
    ax.plot(rx[:n], ry[:n], "-", color="#005195", lw=1.0, label=f"raceline main ({n})")
    if n < len(arr):
        ax.plot(rx[n:], ry[n:], "-", color="#01A0E9", lw=1.2, label=f"raceline pit ({len(arr)-n})")
    ax.plot(pit[:, 0], pit[:, 1], ".-", color="#DC291E", ms=3, lw=0.8, label=f"pit-line fitout ({len(pit)})")
    ax.scatter([pit[0, 0]], [pit[0, 1]], c="green", s=60, zorder=5, label="pit-line start")
    ax.scatter([pit[-1, 0]], [pit[-1, 1]], c="black", s=60, zorder=5, label="pit-line end")
    ax.set_aspect("equal"); ax.set_title(f"{trk}", fontsize=12); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    print(f"[{trk}] pit-line {pit.shape} x[{pit[:,0].min():.0f},{pit[:,0].max():.0f}] y[{pit[:,1].min():.0f},{pit[:,1].max():.0f}] z[{pit[:,2].min():.1f},{pit[:,2].max():.1f}]")
    print(f"[{trk}] raceline x[{rx.min():.0f},{rx.max():.0f}] y[{ry.min():.0f},{ry.max():.0f}] main_n={main_n} total={len(arr)}", flush=True)

fig.suptitle("pit-line (fitout) overlaid on raceline (main+pit)", fontsize=13)
plt.tight_layout()
os.makedirs(f"{REPO}/outputs_multi/viz", exist_ok=True)
out = f"{REPO}/outputs_multi/viz/pitline_overlay.png"
plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close()
print(f"[done] {out}")
