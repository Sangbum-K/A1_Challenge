"""트랙 지도(외곽+raceline+pit-line) 위에서 마우스 클릭으로 pit 폴리곤을 찍고 좌표를 저장하는 임시 툴.

사용:
    python scripts/pick_pit_polygon.py --track sangam      # 또는 speedway
    python scripts/pick_pit_polygon.py --track sangam --out data/control/pit_polygon_sangam.json

조작:
    좌클릭        : 폴리곤 vertex 추가
    우클릭 / u    : 마지막 vertex 취소 (undo)
    c             : 전체 초기화
    s / Enter     : 저장 (out 경로 + 화면 좌표 출력)
    q / 창 닫기   : 종료 (자동으로도 매 클릭마다 out 에 autosave 됨)
    * 좌상단 돋보기/이동(pan/zoom) 모드에선 클릭이 vertex 로 안 들어감 (지도 확대용).

저장 포맷: JSON {track, points:[[x,y],...]}.  raceline append 용 스니펫도 같이 출력.
"""
import os, sys, json, argparse
import numpy as np
import matplotlib.pyplot as plt   # 인터랙티브 backend (Agg 강제 안 함)

UP = "/home/ailab-12/.claude/uploads/04d26f57-da4a-4388-9164-8d6bba8fcb50"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYERS = {
    "sangam": dict(
        outline=f"{UP}/993d9435-road_mesh_out_line_260402.json",   # 최신
        pit=f"{UP}/5d9c92a6-fitout_points_output_sangam.json",
        race=f"{REPO}/data/control/raceline_sangam_w_pit.csv"),     # main+pit (pit 주행선 포함)
    "speedway": dict(
        outline=f"{UP}/6c2b2571-road_mesh_out_line.json",
        pit=f"{UP}/6de40f24-fitout_points_output_yongin.json",
        race=f"{REPO}/data/control/raceline_speedway_dt0p1s.csv"),   # snailpit dt0p1s 주행라인
}


def load_outline(p):
    d = json.load(open(p))[0]
    ext = np.array(d["points"], dtype=np.float64)
    ints = [np.array(it["points"], dtype=np.float64) for it in d.get("interiors", [])]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=list(LAYERS.keys()))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    L = LAYERS[args.track]
    out = args.out or f"{REPO}/data/control/pit_polygon_{args.track}.json"

    ext, ints = load_outline(L["outline"])
    pit = load_pit(L["pit"])
    arr, main_n = load_race(L["race"]); n = main_n or len(arr)

    fig, ax = plt.subplots(figsize=(13, 11))
    ax.plot(ext[:, 0], ext[:, 1], "-", color="#999999", lw=0.8, label="outline ext")
    for inr in ints:
        ax.plot(inr[:, 0], inr[:, 1], "--", color="#999999", lw=0.8, label="outline int")
    ax.plot(arr[:n, 1], arr[:n, 2], "-", color="#005195", lw=1.0, label=f"raceline main ({n})")
    if n < len(arr):    # pit 주행선 segment (raceline pit)
        ax.plot(arr[n:, 1], arr[n:, 2], "-", color="#8E44AD", lw=1.6, label=f"raceline pit ({len(arr)-n})")
    ax.plot(pit[:, 0], pit[:, 1], ".-", color="#DC291E", ms=3, lw=1.0, label=f"pit-line fitout ({len(pit)})")
    ax.scatter([pit[0, 0]], [pit[0, 1]], c="green", s=60, zorder=6, label="pit start")
    ax.scatter([pit[-1, 0]], [pit[-1, 1]], c="orange", s=60, zorder=6, label="pit end")
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.legend(fontsize=8, loc="best")

    verts = []                         # 클릭된 vertex (world xy)
    dots, = ax.plot([], [], "o", color="#FF7F0E", ms=8, zorder=10)
    poly_line, = ax.plot([], [], "-", color="#FF7F0E", lw=1.8, zorder=9)
    title = ax.set_title("")

    def autosave():
        json.dump({"track": args.track, "points": [[float(x), float(y)] for x, y in verts]},
                  open(out, "w"), indent=2)

    def redraw():
        if verts:
            vx = [v[0] for v in verts]; vy = [v[1] for v in verts]
            dots.set_data(vx, vy)
            if len(verts) >= 2:
                poly_line.set_data(vx + [vx[0]], vy + [vy[0]])   # 닫힌 폴리곤 미리보기
            else:
                poly_line.set_data([], [])
        else:
            dots.set_data([], []); poly_line.set_data([], [])
        title.set_text(f"[{args.track}] vertices: {len(verts)}   (좌클릭 추가 / u 취소 / c 초기화 / s 저장 / q 종료)")
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return
        tb = getattr(fig.canvas, "toolbar", None)
        if tb is not None and getattr(tb, "mode", ""):
            return                      # pan/zoom 모드 클릭은 무시
        if event.button == 1:           # 좌클릭 추가
            verts.append((event.xdata, event.ydata))
        elif event.button == 3 and verts:   # 우클릭 undo
            verts.pop()
        autosave(); redraw()

    def on_key(event):
        if event.key == "u" and verts:
            verts.pop(); autosave(); redraw()
        elif event.key == "c":
            verts.clear(); autosave(); redraw()
        elif event.key in ("s", "enter"):
            autosave()
            print(f"[saved] {out}  ({len(verts)} pts)")
            for i, (x, y) in enumerate(verts):
                print(f"  v{i}: {x:.3f}, {y:.3f}")
            print("\n# raceline append 용 (t;x;y;yaw, t=폴리곤 표식 9999):")
            for x, y in verts:
                print(f"9999.0;{x:.4f};{y:.4f};0.0")
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    print(f"[pick_pit_polygon] track={args.track}  out={out}")
    print("좌클릭으로 폴리곤 점을 찍으세요. s 로 저장, q 로 종료. (매 클릭 autosave)")
    plt.show()
    autosave()
    print(f"[done] {out} ({len(verts)} pts)")


if __name__ == "__main__":
    main()
