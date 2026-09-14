#!/usr/bin/env python
"""본선용 최종 snail_loop 맵 csv 4개 → closed-loop 통합 dt0p1s 포맷으로 변환·교체.

배경 (2026-05-27):
  - 사용자 제공 최종 4맵(`*_snail_loop_UID_01_*.csv`, columns `t;x;y;psi`)이 본선 raceline.
  - closed-loop 는 `data/control/{track}_{map}_{P:05d}_{M:05d}.csv`(본문 + 마지막 6행 pit polygon,
    P/M=파일명) 를 raceline_dt0p1s.load_map 으로 읽는다.
  - 새 맵은 polygon 6행이 없으나 좌표계가 기존과 동일(확인: pit 바닥 시작점이 기존 polygon 내부)
    → 기존 통합 csv 의 polygon 6행을 그대로 재사용.
  - P/M(pit_end/pit_merge)는 사용자 검수 index(0-based row). 기존 4개 통합 csv 는 교체(제거).

검증/변환만 수행. data/control 외 미접촉(학습 cache/prebake 무관)."""
from __future__ import annotations
import os, glob, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CTRL = f"{ROOT}/data/control"
UP = "/home/ailab-12/.claude/uploads/04d26f57-da4a-4388-9164-8d6bba8fcb50"

# (track, map) -> 새 맵 파일 basename (uploads). 사용자 제공 본선 최종 맵.
#   2026-05-27 2차: sangam mild/high 재생성 (ay22=mild, ay24=high. 파일명 끝 10자리=PPPPP MMMMM).
#   speedway 는 1차(어제) 생성본 유지 — 여기서 미처리.
NEW = {
    ("sangam", "mild"):   "060fc951-sangam_mild_ay22_vmax_kph225_laptime101s_snail_loop_UID_01_000000025201248.csv",
    ("sangam", "high"):   "5fb65c4a-sangam_mild3_ay24_vmax_kph225_laptime98s_snail_loop_UID_01_000000025501228.csv",
}
# (track, map) -> (P pit_end, M pit_merge). 파일명 끝 10자리(00000_P_M 중 P,M) 에서 파싱한 값.
PM = {
    ("sangam", "mild"):   (252, 1248),   # ...025201248 → P=252 M=1248
    ("sangam", "high"):   (255, 1228),   # ...025501228 → P=255 M=1228
}


def data_lines(path: str) -> list[str]:
    """주석(#)·빈 줄 제외한 data 행 (rstrip; semicolon 유지)."""
    out = []
    with open(path) as f:
        for ln in f:
            s = ln.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out


def existing_polygon(track: str, mp: str) -> list[str]:
    """기존 통합 csv 의 마지막 6행(pit polygon, `-1.0000;x;y;0.0000`) 반환."""
    cand = sorted(glob.glob(f"{CTRL}/{track}_{mp}_[0-9]*_[0-9]*.csv"))
    if not cand:
        raise FileNotFoundError(f"기존 통합 csv 없음: {track}_{mp} (polygon source)")
    poly = data_lines(cand[-1])[-6:]
    if len(poly) != 6 or not all(p.startswith("-1") for p in poly):
        raise ValueError(f"{cand[-1]} 마지막 6행이 polygon(t=-1) 아님: {poly}")
    return poly


def main():
    apply = "--apply" in sys.argv
    for (track, mp), nf in NEW.items():
        npath = os.path.join(UP, nf)
        if not os.path.exists(npath):
            print(f"[err] 새 맵 없음: {npath}", file=sys.stderr); sys.exit(1)
        body = data_lines(npath)                       # raceline 본문 (t;x;y;psi)
        N = len(body)
        P, M = PM[(track, mp)]
        if not (0 <= P < M < N):
            print(f"[err] {track}_{mp}: P/M 범위 오류 P={P} M={M} N={N}", file=sys.stderr); sys.exit(1)
        poly = existing_polygon(track, mp)             # 기존 polygon 6행 재사용

        out_name = f"{track}_{mp}_{P:05d}_{M:05d}.csv"
        out_path = os.path.join(CTRL, out_name)
        header = [
            f"# raceline map (dt0p1s + pit polygon)  track={track} map={mp}",
            f"# pit_end P={P}  pit_merge M={M}  raceline_rows={N}  polygon_rows=6(last)",
            "# t_s; x_m; y_m; psi_rad   (last 6 rows: t=-1, x;y = pit polygon vertex)",
            "# source: " + nf,
        ]
        content = "\n".join(header + body + poly) + "\n"

        # 교체 대상(기존 통합 csv) — 새 파일명과 다른 것
        old = [p for p in sorted(glob.glob(f"{CTRL}/{track}_{mp}_[0-9]*_[0-9]*.csv"))
               if os.path.basename(p) != out_name]
        print(f"[{track}_{mp}] N={N} P={P} M={M} → {out_name}  (제거: {[os.path.basename(o) for o in old]})")
        if apply:
            with open(out_path, "w") as f:
                f.write(content)
            for o in old:
                os.remove(o)
            print(f"  written {out_path}")
    if not apply:
        print("\n(dry-run. 실제 교체는 --apply)")


if __name__ == "__main__":
    main()
