"""dt0p1s raceline waypoint 추출 (closed-loop deployment용).

규약 (사용자 spec 2026-05-25):
- 라인 연결: pit_start(0) → pit_end(P)/주행시작 → ... → pit_merge(M) → 주행끝(N-1).
- speed: 0.1s 등시간 샘플 → ||Δxy|| / 0.1 (마지막은 직전 복사).
- 최근접 idx: ego 가 pit polygon **안** → [0, P] 에서만, **밖** → [P, M] 에서만. (M~end 는 탐색 제외)
- waypoint: 최근접 i0 부터 앞으로 40점(0.1s, =traj40) 추출 → 차량좌표 [x_fwd, y_left, speed].
- 끝점 룰: 사용가능 끝(track=M, pit=P) 초과분은 마지막점에서 heading·speed 로 직선 외삽(가상점).
- 이후 0.05s/T6 보간 등은 build_traj6 (기 개발) 따름.
"""
from __future__ import annotations
import os
import json
import numpy as np
from .raceline_search import point_in_polygon

# (track, map) -> (P pit_end, M pit_merge). 사용자 검수 확정값 (2026-05-27 최종 snail_loop 맵 기준).
# 통합 csv 있으면 load_map 이 파일명에서 P/M 파싱 — 이 dict 는 구포맷 fallback 용.
CUTOFFS = {
    ("sangam", "high"):   (255, 1245),
    ("sangam", "mild"):   (257, 1269),
    ("speedway", "high"): (263, 976),
    ("speedway", "mild"): (261, 985),
}
DT = 0.1


def derive_speed(xy: np.ndarray, dt: float = DT) -> np.ndarray:
    if len(xy) < 2:
        return np.zeros(len(xy), dtype=np.float64)
    d = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])) / dt
    return np.concatenate([d, d[-1:]])


def load_raceline_csv(path: str) -> np.ndarray:
    """dt0p1s csv ('#' 주석, semicolon t;x;y;psi) → (N,4)."""
    rows = [[float(x) for x in l.split(";") if x.strip()]
            for l in open(path) if l.strip() and not l.startswith("#")]
    return np.array(rows, dtype=np.float64)


def load_map(track: str, mp: str, data_dir: str):
    """→ dict(xy, speed, polygon, P, M, N).
    통합 맵 `{track}_{map}_{P:05d}_{M:05d}.csv`: 마지막 6행=pit 폴리곤, P/M 은 파일명.
    파일 없으면 구 포맷(raceline_{track}_{map}.csv + pit_polygon_{track}.json + CUTOFFS) fallback."""
    import glob
    cand = sorted(glob.glob(f"{data_dir}/control/{track}_{mp}_[0-9]*_[0-9]*.csv"))
    if cand:
        path = cand[-1]
        stem = os.path.splitext(os.path.basename(path))[0]
        P, M = (int(x) for x in stem.split("_")[-2:])     # {track}_{map}_PPPPP_MMMMM
        arr = load_raceline_csv(path)
        race, poly = arr[:-6], arr[-6:, 1:3]              # 마지막 6행 = 폴리곤
        xy = race[:, 1:3]
        return dict(xy=xy, speed=derive_speed(xy), polygon=poly, P=int(P), M=int(M), N=len(xy), path=path)
    # fallback (구 포맷)
    arr = load_raceline_csv(f"{data_dir}/control/raceline_{track}_{mp}.csv")
    xy = arr[:, 1:3]
    pf = f"{data_dir}/control/pit_polygon_{track}.json"
    poly = np.array(json.load(open(pf))["points"]) if os.path.exists(pf) else None
    if (track, mp) not in CUTOFFS:
        raise ValueError(f"cutoff 미정: {(track, mp)}")
    P, M = CUTOFFS[(track, mp)]
    return dict(xy=xy, speed=derive_speed(xy), polygon=poly, P=int(P), M=int(M), N=len(xy), path=None)


def ego_in_pit(ex: float, ey: float, polygon) -> bool:
    return polygon is not None and point_in_polygon(ex, ey, polygon)


def nearest_idx(m: dict, ex: float, ey: float) -> tuple[int, bool, int]:
    """최근접 idx (pit/track 범위 룰). → (i0, in_pit, last_usable_idx)."""
    in_pit = ego_in_pit(ex, ey, m["polygon"])
    lo, hi = (0, m["P"]) if in_pit else (m["P"], m["M"])
    hi = min(hi, m["N"] - 1)
    seg = m["xy"][lo:hi + 1]
    d2 = (seg[:, 0] - ex) ** 2 + (seg[:, 1] - ey) ** 2
    return lo + int(np.argmin(d2)), in_pit, hi


def extract_waypoints(m: dict, i0: int, ex: float, ey: float, e_yaw: float,
                      n: int = 40, dt: float = DT) -> np.ndarray:
    """i0 부터 앞으로 n 점 → 차량좌표 (n,3)[x_fwd, y_left, speed].
    라인은 pit→track 연속 → forward 는 끊지 않고 라인 끝(N-1)까지 따라감.
    N-1 초과분만 마지막점 heading·speed 로 직선 외삽 (가상점)."""
    xy, speed = m["xy"], m["speed"]
    li = m["N"] - 1                              # forward bound = 라인 끝 (pit_end/merge 에서 안 끊음)
    if li >= 1:
        hdx, hdy = xy[li, 0] - xy[li - 1, 0], xy[li, 1] - xy[li - 1, 1]
        hn = np.hypot(hdx, hdy) + 1e-9
        hdx, hdy = hdx / hn, hdy / hn
    else:
        hdx, hdy = np.cos(e_yaw), np.sin(e_yaw)
    c, s = np.cos(-e_yaw), np.sin(-e_yaw)
    out = np.zeros((n, 3), dtype=np.float32)
    for k in range(n):
        j = i0 + k
        if j <= li:
            px, py, v = xy[j, 0], xy[j, 1], speed[j]
        else:                                   # 가상 직선점
            steps = j - li
            v = float(speed[li])
            px = xy[li, 0] + hdx * v * dt * steps
            py = xy[li, 1] + hdy * v * dt * steps
        dx, dy = px - ex, py - ey
        out[k, 0] = c * dx - s * dy             # x_fwd
        out[k, 1] = s * dx + c * dy             # y_left
        out[k, 2] = v
    return out


def sample_traj40(m: dict, ex: float, ey: float, e_yaw: float, n: int = 40):
    """편의: 최근접(pit/track 범위 룰) + 40점 forward 추출(연속 라인). → (traj40 (n,3) ego-frame, meta)."""
    i0, in_pit, _ = nearest_idx(m, ex, ey)       # 범위 룰은 최근접에만
    traj = extract_waypoints(m, i0, ex, ey, e_yaw, n=n)
    n_virtual = max(0, (i0 + n - 1) - (m["N"] - 1))   # 라인 끝 초과분만 가상
    return traj, dict(i0=i0, in_pit=in_pit, n_virtual=n_virtual)
