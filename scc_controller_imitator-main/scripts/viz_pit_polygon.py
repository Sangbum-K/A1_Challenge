"""찍은 pit 폴리곤(6점) 검증 시각화: full + 줌인 2패널 + point-in-polygon 통계.
pit-line/pit주행선 이 폴리곤 안, main 이 밖이면 OK (raceline_search pit/track 분리용)."""
import os, sys, json, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath

UP = "/home/ailab-12/.claude/uploads/04d26f57-da4a-4388-9164-8d6bba8fcb50"
REPO = "/home/ailab-12/git/scc_controller_imitator"
CFG = {
    "sangam": dict(outline=f"{UP}/993d9435-road_mesh_out_line_260402.json",
                   pit=f"{UP}/5d9c92a6-fitout_points_output_sangam.json",
                   race=f"{REPO}/data/control/raceline_sangam_w_pit.csv"),
    "speedway": dict(outline=f"{UP}/6c2b2571-road_mesh_out_line.json",
                     pit=f"{UP}/6de40f24-fitout_points_output_yongin.json",
                     race=f"{REPO}/data/control/raceline_speedway_w_pit.csv"),
}


def load_outline(p):
    d = json.load(open(p))[0]
    return np.array(d["points"])[:, :2], [np.array(it["points"])[:, :2] for it in d.get("interiors", [])]


def load_pit(p): return np.array(json.load(open(p))[0]["points"])[:, :2]


def load_race(p):
    rows, main_n = [], None
    for ln in open(p):
        ln = ln.strip()
        if ln.startswith("#"):
            if "main_segment_rows" in ln:
                try: main_n = int(ln.split("=")[1].split("(")[0].strip())
                except Exception: pass
            continue
        if ln:
            v = [float(x) for x in ln.split(";") if x.strip()]
            if len(v) >= 3: rows.append(v)
    return np.array(rows), main_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=list(CFG))
    ap.add_argument("--poly", required=True, help="pit_polygon JSON")
    args = ap.parse_args()
    C = CFG[args.track]
    poly = np.array(json.load(open(args.poly))["points"])
    ext, ints = load_outline(C["outline"]); pit = load_pit(C["pit"])
    arr, n = load_race(C["race"]); n = n or len(arr)
    main_xy, pit_xy = arr[:n, 1:3], arr[n:, 1:3]

    path = MplPath(np.vstack([poly, poly[:1]]))
    def inside(xy): return path.contains_points(xy)
    print(f"[{args.track}] point-in-polygon:")
    print(f"  fitout pit-line : {inside(pit).sum()}/{len(pit)} inside")
    print(f"  raceline pit    : {inside(pit_xy).sum()}/{len(pit_xy)} inside")
    print(f"  raceline main   : {inside(main_xy).sum()}/{len(main_xy)} inside  (적을수록 좋음)")

    fig, axes = plt.subplots(1, 2, figsize=(17, 8))
    pad = 40
    bb = [poly[:, 0].min()-pad, poly[:, 0].max()+pad, poly[:, 1].min()-pad, poly[:, 1].max()+pad]
    for ax, zoom in zip(axes, [False, True]):
        ax.plot(ext[:, 0], ext[:, 1], "-", color="#bbbbbb", lw=0.7)
        for inr in ints: ax.plot(inr[:, 0], inr[:, 1], "--", color="#bbbbbb", lw=0.7)
        ax.plot(main_xy[:, 0], main_xy[:, 1], "-", color="#005195", lw=1.0, label="raceline main")
        ax.plot(pit_xy[:, 0], pit_xy[:, 1], "-", color="#8E44AD", lw=1.8, label="raceline pit")
        ax.plot(pit[:, 0], pit[:, 1], ".-", color="#DC291E", ms=3, lw=1.0, label="pit-line fitout")
        ax.fill(poly[:, 0], poly[:, 1], color="#01A0E9", alpha=0.25)
        ax.plot(np.append(poly[:, 0], poly[0, 0]), np.append(poly[:, 1], poly[0, 1]),
                "o-", color="#01A0E9", lw=2.0, ms=8, label="pit polygon (6)")
        for i, (x, y) in enumerate(poly): ax.annotate(f"v{i}", (x, y), fontsize=9, color="#002060")
        ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        if zoom: ax.set_xlim(bb[0], bb[1]); ax.set_ylim(bb[2], bb[3]); ax.set_title("zoom (polygon)")
        else: ax.legend(fontsize=8); ax.set_title(f"{args.track} full")
    plt.tight_layout()
    out = f"{REPO}/outputs_multi/viz/pit_polygon_{args.track}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
