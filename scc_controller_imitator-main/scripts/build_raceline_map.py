"""통합 raceline 맵 빌더: dt0p1s 라인 + pit polygon(6점) → 배포용 단일 파일.

출력: data/control/{track}_{map}_{P:05d}_{M:05d}.csv
  # 헤더 (P/M 명시)
  t;x;y;psi   (N raceline 행, dt0p1s 원본 그대로)
  -1;px;py;0  (마지막 6행 = pit lane 폴리곤 vertex, t=-1 sentinel)
cutoff P/M 는 파일명, 폴리곤은 파일 내장 → 자체완결(배포 일치).
"""
import os, json, glob, numpy as np
from src.utils.raceline_dt0p1s import CUTOFFS  # (track,map)->(P,M)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CTRL = f"{REPO}/data/control"


def build(track, mp):
    P, M = CUTOFFS[(track, mp)]
    src = f"{CTRL}/raceline_{track}_{mp}.csv"
    rows = [l for l in open(src) if l.strip() and not l.startswith("#")]
    arr = np.array([[float(x) for x in l.split(";") if x.strip()] for l in rows])
    poly = np.array(json.load(open(f"{CTRL}/pit_polygon_{track}.json"))["points"])
    assert len(poly) == 6, f"polygon must be 6 pts, got {len(poly)}"
    out = f"{CTRL}/{track}_{mp}_{P:05d}_{M:05d}.csv"
    with open(out, "w") as f:
        f.write(f"# raceline map (dt0p1s + pit polygon)  track={track} map={mp}\n")
        f.write(f"# pit_end P={P}  pit_merge M={M}  raceline_rows={len(arr)}  polygon_rows=6(last)\n")
        f.write("# t_s; x_m; y_m; psi_rad   (last 6 rows: t=-1, x;y = pit polygon vertex)\n")
        for r in arr:
            f.write(f"{r[0]:.4f};{r[1]:.6f};{r[2]:.6f};{r[3]:.6f}\n")
        for px, py in poly:
            f.write(f"-1.0000;{px:.6f};{py:.6f};0.0000\n")
    print(f"[built] {os.path.basename(out)}  raceline={len(arr)} +polygon6  P={P} M={M}")
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, REPO)
    for trk in ["sangam", "speedway"]:
        for mp in ["high", "mild"]:
            build(trk, mp)
