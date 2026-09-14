"""최종 raceline csv (pitstop_raceline_snail_merger, output=time_resampled) 에서
pit end / pit merge point 를 추출.

snail_loop 구조 (confirmed, 2026-05-27):
  - 라인은 단일 연속 trajectory: pit_start(idx 0) → pit lane(저속 8.33 m/s 정속 구간)
    → pit_end(P, 가속 시작·pit polygon 이탈) → main racing line → ... → pit_merge(M).
  - pit_merge M 은 헤더 `# index_merge (time array, snapped) = ...` 로 명시됨 (confirmed).
  - pit_end P 는 repo 규약(src/utils/raceline_dt0p1s.py 의 P)과 동일하게
    pit polygon 의 첫 연속 in-polygon block 끝(폴리곤 이탈 index)으로 정의.
    교차검증으로 pit 정속(8.33 m/s)이 끝나고 가속이 시작되는 speed-based index 도 함께 보고.

컬럼 (confirmed by reading csv):
  output=time_resampled  →  `# t_s; x_m; y_m; psi_rad`  (semicolon-delim, '#' 주석)
  (기존 raceline_dt0p1s 의 t;x;y;psi 와 동일. s/curv/speed/pit 컬럼 없음 → pit flag 컬럼 부재.)

사용:
  python scripts/extract_pit_points.py --csv <file> --track sangam   # polygon 자동 매칭
  python scripts/extract_pit_points.py --csv <file>                  # track 헤더에서 추정
  python scripts/extract_pit_points.py --batch <dir>                 # 디렉토리 내 final csv 일괄
"""
from __future__ import annotations
import os
import sys
import re
import argparse
import json
import glob
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.utils.raceline_search import point_in_polygon  # noqa: E402

PIT_SPEED = 8.3333  # m/s, pit lane 정속 (30 km/h). confirmed: 4 csv 모두 시작 정속값.
DT = 0.1


def load_csv(path: str):
    """final csv → (rows (N,4)[t,x,y,psi], header dict)."""
    hdr = {}
    rows = []
    for l in open(path):
        ls = l.strip()
        if not ls:
            continue
        if ls.startswith("#"):
            m = re.search(r"index_merge\s*\(time array, snapped\)\s*=\s*(\d+)", ls)
            if m:
                hdr["index_merge"] = int(m.group(1))
            m = re.search(r"source_arc_csv\s*=\s*(.+)", ls)
            if m:
                hdr["source"] = m.group(1).strip()
            continue
        rows.append([float(x) for x in ls.split(";") if x.strip()])
    return np.array(rows, dtype=np.float64), hdr


def arc_length(xy: np.ndarray) -> np.ndarray:
    """누적 arc-length s (m). s[0]=0."""
    d = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    return np.concatenate([[0.0], np.cumsum(d)])


def speed(xy: np.ndarray, dt: float = DT) -> np.ndarray:
    d = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])) / dt
    return np.concatenate([d, d[-1:]])


def first_inpoly_block_end(xy: np.ndarray, poly: np.ndarray) -> int:
    """pit polygon 첫 연속 in-polygon block 의 마지막 index (= polygon-exit, repo 규약 P)."""
    inside = np.array([point_in_polygon(x, y, poly) for x, y in xy])
    if not inside.any():
        return -1
    start = int(np.argmax(inside))           # 첫 True
    j = start
    while j + 1 < len(inside) and inside[j + 1]:
        j += 1
    return j


def speed_based_pit_end(xy: np.ndarray, tol: float = 0.5) -> int:
    """pit 정속(8.33 m/s) 이 처음 벗어나는 index (가속 시작) — 교차검증용."""
    v = speed(xy)
    dev = np.abs(v - PIT_SPEED) > tol
    return int(np.argmax(dev)) if dev.any() else -1


def guess_track(path: str, hdr: dict) -> str | None:
    s = (os.path.basename(path) + " " + hdr.get("source", "")).lower()
    for t in ("sangam", "speedway"):
        if t in s:
            return t
    return None


def load_polygon(track: str):
    pf = f"{REPO}/data/control/pit_polygon_{track}.json"
    if os.path.exists(pf):
        return np.array(json.load(open(pf))["points"])
    return None


def extract(path: str, track: str | None = None):
    rows, hdr = load_csv(path)
    xy = rows[:, 1:3]
    N = len(rows)
    s = arc_length(xy)
    v = speed(xy)
    track = track or guess_track(path, hdr)
    poly = load_polygon(track) if track else None

    # pit_merge M : 헤더 index_merge (confirmed). 없으면 폴리곤 tail block 으로 fallback.
    M = hdr.get("index_merge", None)

    # pit_end P : repo 규약 = polygon-exit. 폴리곤 없으면 speed-based fallback.
    P_speed = speed_based_pit_end(xy)
    if poly is not None:
        P = first_inpoly_block_end(xy, poly)
        p_method = "polygon-exit"
    else:
        P = P_speed
        p_method = "speed-based (no polygon)"

    def pt(i):
        if i is None or i < 0 or i >= N:
            return None
        return dict(index=int(i), s=round(float(s[i]), 3),
                    x=round(float(xy[i, 0]), 3), y=round(float(xy[i, 1]), 3),
                    v=round(float(v[i]), 2))

    return dict(
        path=path, track=track, N=N,
        pit_end=pt(P), pit_end_method=p_method,
        pit_end_speed_based=pt(P_speed),
        pit_merge=pt(M),
        source=hdr.get("source", ""),
    )


def fmt(r: dict) -> str:
    out = []
    out.append(f"=== {os.path.basename(r['path'])} ===")
    out.append(f"  track={r['track']}  N={r['N']}  source={r['source']}")
    pe, pm, ps = r["pit_end"], r["pit_merge"], r["pit_end_speed_based"]
    out.append(f"  pit_end    ({r['pit_end_method']:24s}): "
               + (f"idx={pe['index']:5d}  s={pe['s']:9.2f}  x={pe['x']:11.3f}  y={pe['y']:11.3f}  v={pe['v']:.1f}" if pe else "N/A"))
    out.append(f"  pit_end    (speed-based cross-check    ): "
               + (f"idx={ps['index']:5d}  s={ps['s']:9.2f}  x={ps['x']:11.3f}  y={ps['y']:11.3f}  v={ps['v']:.1f}" if ps else "N/A"))
    out.append(f"  pit_merge  (header index_merge          ): "
               + (f"idx={pm['index']:5d}  s={pm['s']:9.2f}  x={pm['x']:11.3f}  y={pm['y']:11.3f}  v={pm['v']:.1f}" if pm else "N/A"))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="단일 final raceline csv")
    ap.add_argument("--batch", help="디렉토리 — snail merger time_resampled csv 일괄")
    ap.add_argument("--track", default=None, choices=["sangam", "speedway"])
    ap.add_argument("--json-out", default=None, help="결과 json 저장 경로")
    args = ap.parse_args()

    files = []
    if args.csv:
        files = [args.csv]
    elif args.batch:
        for f in sorted(glob.glob(f"{args.batch}/*.csv")):
            # snail_merger time_resampled 만 (header 로 식별)
            head = open(f).readline()
            if "pitstop_raceline_snail_merger" in head:
                # output=time_resampled 만 (merger-mode arc csv 는 컬럼 다름 → 제외)
                hl = "".join(open(f).readlines()[:4])
                if "output=time_resampled" in head:
                    files.append(f)
    else:
        ap.error("--csv 또는 --batch 필요")

    results = []
    for f in files:
        r = extract(f, args.track)
        results.append(r)
        print(fmt(r))
    if args.json_out:
        json.dump(results, open(args.json_out, "w"), indent=2)
        print(f"\n[saved] {args.json_out}")


if __name__ == "__main__":
    main()
