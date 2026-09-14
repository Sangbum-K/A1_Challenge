"""트랙 지도 전체 레이어 시각화: outer boundary(exterior+interior ring) + raceline(main) + pit-line.
좌표는 모두 동일 world frame. 라벨 영문(폰트 깨짐 방지)."""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

UP = "/home/ailab-12/.claude/uploads/04d26f57-da4a-4388-9164-8d6bba8fcb50"
REPO = "/home/ailab-12/git/scc_controller_imitator"
SANGAM_OUTLINES = [  # (label, file)
    ("outline old (8715)", f"{UP}/231864c1-road_mesh_out_line.json"),
    ("outline 260402 (8614)", f"{UP}/993d9435-road_mesh_out_line_260402.json"),
]
JOBS = [
    dict(track="sangam", outlines=SANGAM_OUTLINES,
         pit=f"{UP}/5d9c92a6-fitout_points_output_sangam.json",
         race=f"{REPO}/data/control/raceline_sangam_w_pit.csv"),
    dict(track="speedway", outlines=[("outline", f"{UP}/6c2b2571-road_mesh_out_line.json")],
         pit=f"{UP}/6de40f24-fitout_points_output_yongin.json",
         race=f"{REPO}/data/control/raceline_speedway_dt0p1s.csv"),
]


def load_outline(p):
    d = json.load(open(p))[0]
    ext = np.array(d["points"], dtype=np.float64)
    ints = [np.array(it["points"], dtype=np.float64) for it in d["interiors"]]
    return ext, ints


def load_pit(p):
    return np.array(json.load(open(p))[0]["points"], dtype=np.float64)


def load_race(p):
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
    return np.array(rows, dtype=np.float64), main_n


fig, axes = plt.subplots(1, 2, figsize=(17, 8))
for ax, job in zip(axes, JOBS):
    trk = job["track"]
    shades = ["#999999", "#333333"]
    for i, (lab, of) in enumerate(job["outlines"]):
        ext, ints = load_outline(of)
        c = shades[i % len(shades)]
        ax.plot(ext[:, 0], ext[:, 1], "-", color=c, lw=0.8, label=f"{lab} ext")
        for j, inr in enumerate(ints):
            ax.plot(inr[:, 0], inr[:, 1], "--", color=c, lw=0.8, label=f"{lab} int" if j == 0 else None)
        print(f"[{trk}] {lab}: ext {len(ext)} x[{ext[:,0].min():.0f},{ext[:,0].max():.0f}] | int rings {len(ints)} ({','.join(str(len(a)) for a in ints)})", flush=True)
    arr, main_n = load_race(job["race"]); n = main_n or len(arr)
    ax.plot(arr[:n, 1], arr[:n, 2], "-", color="#005195", lw=1.2, label=f"raceline main ({n})")
    if n < len(arr):
        ax.plot(arr[n:, 1], arr[n:, 2], "-", color="#8E44AD", lw=1.8, label=f"raceline pit ({len(arr)-n})")
    pit = load_pit(job["pit"])
    ax.plot(pit[:, 0], pit[:, 1], ".-", color="#DC291E", ms=2, lw=1.0, label=f"pit-line ({len(pit)})")
    ax.scatter([pit[0, 0]], [pit[0, 1]], c="green", s=50, zorder=6)
    ax.scatter([pit[-1, 0]], [pit[-1, 1]], c="orange", s=50, zorder=6)
    ax.set_aspect("equal"); ax.set_title(trk, fontsize=12); ax.legend(fontsize=7, loc="best"); ax.grid(alpha=0.3)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")

fig.suptitle("track map: outer boundary (ext solid / int dashed) + raceline + pit-line", fontsize=13)
plt.tight_layout()
os.makedirs(f"{REPO}/outputs_multi/viz", exist_ok=True)
out = f"{REPO}/outputs_multi/viz/track_map.png"
plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
print(f"[done] {out}")
