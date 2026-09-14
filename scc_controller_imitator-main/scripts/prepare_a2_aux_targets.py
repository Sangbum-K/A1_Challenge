"""a_2 (AuxHead) 학습용 per-lateral-bin GT 생성.

각 frame 당 image 의 lateral 40 bin (uniform over W=1280) 별로:
  ch 0  obj           : 1 if 가까운 객체 in bin else 0
  ch 1  v_long_abs    : 객체 절대 전방 속도 (m/s, ego basis). own→ego basis rotate 만
  ch 2  v_lat_abs     : 객체 절대 좌측 속도 (m/s, ego basis)
  ch 3  v_long_rel    : 상대 = v_abs − v_ego.  음수 = ego 가 따라잡는 중
  ch 4  v_lat_rel     : 상대 좌측 속도
  ch 5  prox          : exp(-d / D_SCALE), d=평면 거리. 가까울수록 큼
  ch 6  dist          : 평면 거리 d (m, raw). 객체 없으면 0. (앞차 거리 회귀 head 학습용)

  → 모델이 절대 / 상대 둘 다 추론하도록 GT 제공 (loss 가중은 학습 config 에서).

projection (pinhole):
  ego frame: x=forward, y=left   (raw EGO/OBJECT 모두 world frame, ego basis 로 변환)
  camera   : x=right (= -y_ego), z=forward (= x_ego)
  u_pix = cx - fx * y_ego / x_ego          (object 가 ego 의 좌측이면 u<cx, 우측이면 u>cx)
  bin    = floor(u_pix * 40 / W)            (0..39, FOV 밖이면 skip)

확장성:
  - 현재 image set : fx=320, cx=640, W=1280  (CAMERA_1)
  - 미래 image set : fx=640 (다른 카메라) — 같은 script 에 --fx 만 바꿔서 별도 dir 생성
    출력 dir 명 / npz metadata 에 fx 가 명시되므로 dataset loader 가 매칭 검증 가능.

사용:
  python scripts/prepare_a2_aux_targets.py --fx 320 --out-tag fx320
  python scripts/prepare_a2_aux_targets.py --fx 640 --out-tag fx640

출력 file:
  {OUT_BASE}_{tag}/{frame_id}.npz
    target : (40, 7) float32
             [obj, v_long_abs, v_lat_abs, v_long_rel, v_lat_rel, prox, dist]
    meta   : dict {fx, cx, W, n_bins, d_scale, fov_deg, max_dist_m}
"""
from __future__ import annotations
import os, sys, json, argparse
from pathlib import Path
import numpy as np

DEG2RAD = np.pi / 180.0

DEFAULT_META = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/multiagnet_compact_meta.json"
DEFAULT_RAW  = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/jiwon_e2e/e2e_racing/data/raw"
DEFAULT_OUT  = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/aux_targets"

IMG_W = 1280
N_BINS = 40
D_SCALE_M = 20.0       # prox = exp(-d / D_SCALE) → 20m 에서 0.37, 40m 에서 0.13
MAX_DIST_M = 80.0      # 80m 너머는 제외 (sensor noise + small FOV)


def parse_ego(ego_file: Path):
    """raw EGO_INFO/{tick}.txt → dict 또는 None."""
    if not ego_file.exists(): return None
    out = {}
    for line in ego_file.read_text().splitlines():
        if ":" not in line: continue
        k, _, v = line.partition(":")
        k = k.strip(); v = v.strip()
        if k in ("position", "orientation", "velocity"):
            nums = [float(t) for t in v.split()]
            if len(nums) >= 3: out[k] = nums
    # required keys
    for k in ("position", "orientation", "velocity"):
        if k not in out: return None
    return out


def parse_objects(obj_file: Path):
    """raw OBJECT_INFO/object_info_{tick}.txt → list of dicts.
    각 line columns:
      0:name 1-3:pos(x,y,z) world 4-6:orient(roll,pitch,yaw_deg) 7-9:size(L,W,H)
      10-12:velocity(vx,vy,vz) own-frame 13:track_id
    """
    if not obj_file.exists(): return []
    out = []
    for line in obj_file.read_text().splitlines():
        toks = line.split()
        if len(toks) < 14: continue
        try:
            out.append({
                "name": toks[0],
                "pos":  [float(toks[1]), float(toks[2]), float(toks[3])],
                "yaw_rad": float(toks[6]) * DEG2RAD,
                "size": [float(toks[7]), float(toks[8]), float(toks[9])],
                "vel_own": [float(toks[10]), float(toks[11]), float(toks[12])],
                "track_id": int(toks[13]),
            })
        except (ValueError, IndexError):
            continue
    return out


def world_to_ego_xy(xw: float, yw: float, ex: float, ey: float, e_yaw: float):
    dx = xw - ex; dy = yw - ey
    c, s = np.cos(-e_yaw), np.sin(-e_yaw)
    return c * dx - s * dy, s * dx + c * dy   # x_ego_forward, y_ego_left


def obj_vel_in_ego(vx_own: float, vy_own: float, obj_yaw: float, ego_yaw: float):
    """own frame → ego frame velocity (forward, left)."""
    dyaw = obj_yaw - ego_yaw
    c, s = np.cos(dyaw), np.sin(dyaw)
    return c * vx_own - s * vy_own, s * vx_own + c * vy_own


N_CH = 7    # [obj, v_long_abs, v_lat_abs, v_long_rel, v_lat_rel, prox, dist]


def _process_one_aux(args_tuple):
    """ProcessPool worker — 1 frame 처리."""
    m, raw_base, fx, cx, img_w, n_bins, d_scale, max_dist, out_dir_str = args_tuple
    try:
        track = m["track"]; policy = m["policy"]
        run = m["run_id"]; tick = int(m["tick"])
        fid = m["frame_id"]
        run_dir = Path(raw_base) / track / policy / run
        ego = parse_ego(run_dir / "EGO_INFO" / f"{tick}.txt")
        if ego is None: return ("skip_no_ego", fid, None)
        objs = parse_objects(run_dir / "OBJECT_INFO" / f"object_info_{tick}.txt")
        target = build_target_for_frame(
            ego, objs, fx=fx, cx=cx, img_w=img_w, n_bins=n_bins,
            d_scale=d_scale, max_dist=max_dist)
        np.savez_compressed(Path(out_dir_str) / f"{fid}.npz", target=target)
        return ("ok", fid, target[:, 0])
    except Exception as e:
        return ("err", str(e), None)


def build_target_for_frame(ego: dict, objs: list, fx: float, cx: float,
                           img_w: int = IMG_W, n_bins: int = N_BINS,
                           d_scale: float = D_SCALE_M, max_dist: float = MAX_DIST_M):
    """ego/obj dicts → target (n_bins, 7) float32.

    bin 당 가장 가까운 (평면 거리) 객체로 채움. 비어 있으면 모두 0.
    """
    target = np.zeros((n_bins, N_CH), dtype=np.float32)
    if not objs: return target

    ex, ey = ego["position"][0], ego["position"][1]
    e_yaw = ego["orientation"][2] * DEG2RAD
    ego_vx, ego_vy = ego["velocity"][0], ego["velocity"][1]   # ego-own frame (m/s)

    bin_w = img_w / n_bins
    per_bin = [None] * n_bins   # (dist, v_long_abs, v_lat_abs, v_long_rel, v_lat_rel, prox)

    e_c, e_s = np.cos(-e_yaw), np.sin(-e_yaw)

    for o in objs:
        # 객체 중심 → ego frame (center 거리 / max_dist 필터용)
        xe_c, ye_c = world_to_ego_xy(o["pos"][0], o["pos"][1], ex, ey, e_yaw)
        if xe_c <= 0.1: continue
        d_center = float(np.hypot(xe_c, ye_c))
        if d_center > max_dist: continue

        # 4 corner (객체 local: ±L/2, ±W/2) → world → ego
        L, W = o["size"][0], o["size"][1]
        cl, sl = np.cos(o["yaw_rad"]), np.sin(o["yaw_rad"])
        corners_local = np.array([[+L/2, +W/2], [+L/2, -W/2],
                                    [-L/2, +W/2], [-L/2, -W/2]])
        # rotate to world (col-vec: world = R @ local + pos)
        cw_x = corners_local[:, 0] * cl - corners_local[:, 1] * sl + o["pos"][0]
        cw_y = corners_local[:, 0] * sl + corners_local[:, 1] * cl + o["pos"][1]
        # world → ego
        dx = cw_x - ex; dy = cw_y - ey
        xe_corners = e_c * dx - e_s * dy
        ye_corners = e_s * dx + e_c * dy
        # 카메라 앞쪽 corner 만 projection 가능 (xe > 0.1)
        fwd = xe_corners > 0.1
        if not fwd.any(): continue
        u_corners = cx - fx * ye_corners[fwd] / xe_corners[fwd]
        # 클리핑 + bin 범위
        u_min = float(u_corners.min()); u_max = float(u_corners.max())
        u_min = max(0.0, u_min); u_max = min(float(img_w), u_max)
        if u_max <= u_min: continue
        bi_min = max(0, int(u_min / bin_w))
        bi_max = min(n_bins, int(np.ceil(u_max / bin_w)))
        if bi_max <= bi_min: continue

        # 절대속도 (ego basis): own → ego basis rotate 만 (자차 운동 미반영)
        v_long_abs, v_lat_abs = obj_vel_in_ego(
            o["vel_own"][0], o["vel_own"][1], o["yaw_rad"], e_yaw)
        v_long_rel = v_long_abs - ego_vx
        v_lat_rel  = v_lat_abs  - ego_vy
        prox = float(np.exp(-d_center / d_scale))

        # 객체가 차지하는 모든 bin 에 동일 값으로 채움. 더 가까운 객체 우선.
        for bi in range(bi_min, bi_max):
            if per_bin[bi] is None or d_center < per_bin[bi][0]:
                per_bin[bi] = (d_center, v_long_abs, v_lat_abs, v_long_rel, v_lat_rel, prox)

    for bi, slot in enumerate(per_bin):
        if slot is None: continue
        d, vL_abs, vY_abs, vL_rel, vY_rel, prox = slot
        target[bi, 0] = 1.0
        target[bi, 1] = float(vL_abs)
        target[bi, 2] = float(vY_abs)
        target[bi, 3] = float(vL_rel)
        target[bi, 4] = float(vY_rel)
        target[bi, 5] = float(prox)
        target[bi, 6] = float(d)
    return target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fx", type=float, default=320.0)
    ap.add_argument("--cx", type=float, default=None,
                    help="기본: img_w/2")
    ap.add_argument("--img-w", type=int, default=IMG_W)
    ap.add_argument("--n-bins", type=int, default=N_BINS)
    ap.add_argument("--d-scale", type=float, default=D_SCALE_M)
    ap.add_argument("--max-dist", type=float, default=MAX_DIST_M)
    ap.add_argument("--meta", default=DEFAULT_META)
    ap.add_argument("--raw-base", default=DEFAULT_RAW)
    ap.add_argument("--out-base", default=DEFAULT_OUT)
    ap.add_argument("--out-tag", default=None,
                    help="기본: fx{fx} (예: fx320)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cx = args.cx if args.cx is not None else args.img_w / 2.0
    tag = args.out_tag if args.out_tag else f"fx{int(args.fx)}"
    out_dir = Path(f"{args.out_base}_{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_cfg = {
        "fx": float(args.fx), "cx": float(cx),
        "img_w": int(args.img_w), "n_bins": int(args.n_bins),
        "d_scale": float(args.d_scale), "max_dist": float(args.max_dist),
        "fov_deg": float(2 * np.degrees(np.arctan(args.img_w / 2 / args.fx))),
        "bin_w_px": float(args.img_w / args.n_bins),
        "src_camera": "CAMERA_1",
    }
    print(f"[a_2 target] config:")
    for k, v in meta_cfg.items(): print(f"  {k} = {v}")
    print(f"out_dir: {out_dir}")

    # 같은 config 의 cfg.json 도 저장 (dataset loader 가 검증)
    with open(out_dir / "config.json", "w") as f:
        json.dump(meta_cfg, f, indent=2)

    meta = json.load(open(args.meta))
    if args.limit: meta = meta[: args.limit]
    print(f"[data] {len(meta)} frames")

    # multiprocessing: 큰 데이터 (DriverB_new 135K) 빠른 처리 위해 ProcessPool 적용
    from concurrent.futures import ProcessPoolExecutor
    import time as _t
    n_workers = max(4, os.cpu_count() // 2)
    print(f"[workers] {n_workers}")
    tasks = [(m, args.raw_base, args.fx, cx, args.img_w, args.n_bins,
              args.d_scale, args.max_dist, str(out_dir)) for m in meta]
    n_ok = 0; n_skip_no_ego = 0; n_no_obj = 0
    bin_hits = np.zeros(args.n_bins, dtype=np.int64)
    t0 = _t.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for i, (status, fid, bins) in enumerate(ex.map(_process_one_aux, tasks, chunksize=64)):
            if status == "skip_no_ego": n_skip_no_ego += 1
            elif status == "err": pass   # ignore individual errors
            else:
                n_ok += 1
                if bins is not None:
                    if bins.sum() == 0: n_no_obj += 1
                    bin_hits += (bins > 0).astype(np.int64)
            if (i + 1) % 2000 == 0:
                rate = (i + 1) / max(0.001, _t.time() - t0)
                print(f"  [{i+1}/{len(tasks)}] ok={n_ok} skip={n_skip_no_ego} no_obj={n_no_obj}  rate={rate:.0f}/s  ETA={(len(tasks)-i-1)/max(rate,1):.0f}s", flush=True)

    print(f"\n[done] {n_ok} frames -> {out_dir}")
    print(f"  skip_no_ego={n_skip_no_ego}  no_obj_in_fov={n_no_obj}")
    print(f"  per-bin hit count (40 bins, total {bin_hits.sum()}):")
    for i in range(0, args.n_bins, 4):
        print(f"    bin{i:2d}: {bin_hits[i]:4d}  bin{i+1:2d}: {bin_hits[i+1]:4d}  "
              f"bin{i+2:2d}: {bin_hits[i+2]:4d}  bin{i+3:2d}: {bin_hits[i+3]:4d}")


if __name__ == "__main__":
    main()
