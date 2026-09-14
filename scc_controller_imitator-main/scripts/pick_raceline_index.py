"""dt0p1s raceline 위에서 마우스 클릭 → 최근접 점의 INDEX 를 표시/출력. P(피트끝)/M(피트머지) 인덱스 검수용.
START(idx0)·END(idx N-1) 강조. pit polygon overlay.

사용:
    python scripts/pick_raceline_index.py --track sangam --map high     # mild|high
조작:
    좌클릭   : 그 위치의 최근접 raceline index 표시 + 콘솔 출력 (+ 점 누적)
    u        : 마지막 표시 취소
    c        : 전체 지우기
    s        : 누적 index 목록 저장(out)/출력
    q        : 종료
    * 좌상단 돋보기/이동 모드에선 클릭 무시 (확대용) — 확대해서 정밀 클릭 권장.
"""
import os, sys, json, argparse
import numpy as np
import matplotlib.pyplot as plt   # 인터랙티브 backend

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_dt(f):
    return np.array([[float(x) for x in l.split(";") if x.strip()]
                     for l in open(f) if l.strip() and not l.startswith("#")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=["sangam", "speedway"])
    ap.add_argument("--map", default="high", choices=["mild", "high"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rf = f"{REPO}/data/control/raceline_{args.track}_{args.map}.csv"
    pf = f"{REPO}/data/control/pit_polygon_{args.track}.json"
    out = args.out or f"{REPO}/data/control/picked_idx_{args.track}_{args.map}.json"
    a = load_dt(rf); xy = a[:, 1:3]; N = len(xy)
    t = a[:, 0]
    poly = np.array(json.load(open(pf))["points"]) if os.path.exists(pf) else None

    fig, ax = plt.subplots(figsize=(13, 11))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=np.arange(N), s=7, cmap="viridis", zorder=2)
    plt.colorbar(sc, ax=ax, label="raceline index")
    if poly is not None:
        ax.plot(np.append(poly[:, 0], poly[0, 0]), np.append(poly[:, 1], poly[0, 1]),
                "-", color="#DC291E", lw=1.5, zorder=3, label="pit polygon")
    ax.scatter(*xy[0], c="lime", s=300, marker="*", edgecolor="k", zorder=6, label="START idx 0")
    ax.scatter(*xy[N - 1], c="black", s=180, marker="X", zorder=6, label=f"END idx {N-1}")
    ax.annotate("START 0", xy[0], fontsize=11, color="green", weight="bold")
    ax.annotate(f"END {N-1}", xy[N - 1], fontsize=11, color="black", weight="bold")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="best")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    title = ax.set_title("")

    picks = []; arts = []

    def autosave():
        json.dump({"track": args.track, "map": args.map, "N": N, "picked_idx": picks},
                  open(out, "w"), indent=2)

    def settitle():
        title.set_text(f"[{args.track} {args.map}] N={N}  좌클릭=최근접 idx  picks={picks}  (u취소 c지움 s저장 q종료)")
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return
        tb = getattr(fig.canvas, "toolbar", None)
        if tb is not None and getattr(tb, "mode", ""):
            return
        if event.button != 1:
            return
        d2 = (xy[:, 0] - event.xdata) ** 2 + (xy[:, 1] - event.ydata) ** 2
        i = int(np.argmin(d2))
        picks.append(i)
        p1 = ax.scatter(*xy[i], c="red", s=90, marker="o", zorder=7)
        p2 = ax.annotate(f"idx {i}", xy[i], fontsize=11, color="red", weight="bold", zorder=8)
        arts.append((p1, p2))
        print(f"[click] nearest idx = {i}   xy=({xy[i,0]:.2f}, {xy[i,1]:.2f})   t={t[i]:.1f}s", flush=True)
        autosave(); settitle()

    def on_key(event):
        if event.key == "u" and picks:
            picks.pop(); a1, a2 = arts.pop(); a1.remove(); a2.remove(); autosave(); settitle()
        elif event.key == "c":
            picks.clear()
            for a1, a2 in arts: a1.remove(); a2.remove()
            arts.clear(); autosave(); settitle()
        elif event.key == "s":
            autosave(); print(f"[saved] {out}  picks={picks}", flush=True)
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    settitle()
    print(f"[pick_raceline_index] {rf}  N={N}  out={out}")
    print("좌클릭하면 최근접 raceline index 가 콘솔/그림에 표시됩니다. q 종료.")
    plt.show()
    autosave()
    print(f"[done] picks={picks} -> {out}")


if __name__ == "__main__":
    main()
