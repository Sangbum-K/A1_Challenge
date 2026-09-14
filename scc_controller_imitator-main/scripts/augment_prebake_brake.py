"""prebake post-process — close + approach sample 의 GT ax 강제 brake floor 적용.

원인: MPC 합성 GT 가 가까운 lead (dist<10m) + 강한 approach (vrel<-5) 영역에서
cruise/accel 결정 → 모델이 그 패턴 학습 → closed_loop 에서 brake 안 함.

규칙 (per-sample):
  lookup aux_targets npz → 가장 가까운 lead 의 (dist, vrel)
  if dist < DIST_THR and vrel < VREL_THR:
    target[:, 1] = np.minimum(target[:, 1], BRAKE_FLOOR)   # ax floor 강제

사용:
  python scripts/augment_prebake_brake.py \
      --prebake .../f1_prebake_iy_spd_streaming.npz \
      --aux-dir-spec speedway_track__0519_agent1_pertX_pitX:.../aux_targets_spd_fx640 \
      --out .../f1_prebake_iy_spd_streaming_brakeaug.npz
"""
from __future__ import annotations
import argparse, json, os, re
from pathlib import Path
import numpy as np


def parse_aux_dir_spec(spec_list):
    """list of 'frame_id_prefix:aux_dir' → dict[prefix → Path]"""
    m = {}
    for s in spec_list:
        if ":" not in s: continue
        p, d = s.split(":", 1)
        m[p] = Path(d)
    return m


def get_lead_dist_vrel(aux_path: Path):
    """aux .npz → (lead_dist, lead_vrel) or (None, None)"""
    if not aux_path.exists(): return None, None
    try:
        t = np.load(aux_path)["target"]
        objs = t[:, 0] > 0.5
        if not objs.any(): return None, None
        valid_dists = t[objs, 6]
        valid_vrels = t[objs, 3]
        mi = np.argmin(valid_dists)
        return float(valid_dists[mi]), float(valid_vrels[mi])
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prebake", required=True)
    ap.add_argument("--aux-dir-spec", nargs="*", default=[],
                    help="여러 'frame_id_prefix:aux_dir' (multi-policy 처리). prefix 가 frame_id 의 시작과 match")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dist-thr", type=float, default=15.0)
    ap.add_argument("--vrel-thr", type=float, default=-3.0,
                    help="vrel < vrel_thr (음수, ego 따라잡는 중)")
    ap.add_argument("--brake-floor", type=float, default=-5.0,
                    help="target ax = min(orig, brake_floor)")
    ap.add_argument("--also-mild", action="store_true",
                    help="dist<25 + vrel<-2 인 sample 도 -3 floor (약한 brake 학습)")
    args = ap.parse_args()

    aux_map = parse_aux_dir_spec(args.aux_dir_spec)
    print(f"[brake-aug] aux_map: {list(aux_map.keys())}")

    z = np.load(args.prebake, allow_pickle=True)
    sample_keys = z["sample_keys"].tolist()
    sample_tags = z["sample_tags"].tolist()
    curr_keys = z["curr_keys"].tolist()
    prev_keys = z["prev_keys"].tolist()
    d_yaws = z["d_yaws"]
    sample_fx = z["sample_fx"] if "sample_fx" in z.files else np.full(len(sample_keys), 640.0, dtype=np.float32)
    ego_5d = z["ego_5d"]; traj = z["traj"]
    target = z["target"].copy()         # (N, 40, 2)
    mask = z["mask"]
    obj_target = z["obj_target"]

    print(f"[brake-aug] N samples: {len(sample_keys)}")

    # frame_id 추정: sample_key 에서 run + tick 추출. tag 가 logged 인 경우.
    # sample_key format: "{run}__t{tick:07d}__logged" or "{run}__t{tick:07d}__rand_NNNNN"
    # aux 는 frame_key 단위 — same run+tick 가진 모든 logged/rand sample 이 같은 aux 공유
    # frame_id format: "track__policy__run__t{tick:07d}" → aux file 이름
    # prebake 에서 track/policy 정보 없음 → aux_map 의 prefix 로 추정 (모든 prefix 시도해서 match 하는 거 사용)

    n_modified = 0; n_skipped = 0; n_aux_miss = 0
    last_log = 0
    # cache: same curr_key (logged + rands) → lookup 1번만
    cache_lookup = {}
    for i, ck in enumerate(curr_keys):
        if ck in cache_lookup:
            lead_d, lead_v = cache_lookup[ck]
        else:
            lead_d, lead_v = None, None
            run, tk_str = ck.rsplit("__t", 1)
            tick = int(tk_str)
            for prefix, aux_dir in aux_map.items():
                fid = f"{prefix}__{run}__t{tick:07d}"
                lead_d, lead_v = get_lead_dist_vrel(aux_dir / f"{fid}.npz")
                if lead_d is not None: break
            cache_lookup[ck] = (lead_d, lead_v)
        if lead_d is None:
            n_aux_miss += 1; continue
        # 조건 check
        applied = False
        if lead_d < args.dist_thr and lead_v < args.vrel_thr:
            orig = target[i, :, 1].copy()
            target[i, :, 1] = np.minimum(orig, args.brake_floor)
            if not np.allclose(orig, target[i, :, 1]):
                n_modified += 1; applied = True
        if args.also_mild and not applied and lead_d < 25 and lead_v < -2:
            orig = target[i, :, 1].copy()
            target[i, :, 1] = np.minimum(orig, -3.0)
            if not np.allclose(orig, target[i, :, 1]):
                n_modified += 1
        n_skipped += 0 if applied else 1
        if i - last_log >= 50000:
            print(f"  [{i}/{len(curr_keys)}] modified={n_modified} aux_miss={n_aux_miss}")
            last_log = i

    print(f"[brake-aug] DONE")
    print(f"  total: {len(curr_keys)}")
    print(f"  modified: {n_modified}")
    print(f"  aux miss: {n_aux_miss}")
    print(f"  conditions: dist<{args.dist_thr} + vrel<{args.vrel_thr} → ax floor={args.brake_floor}")
    if args.also_mild:
        print(f"             + (also-mild) dist<25 + vrel<-2 → ax floor=-3")

    # save (same format)
    out_tmp = args.out + ".partial.npz"
    print(f"[brake-aug] saving → {args.out}")
    np.savez_compressed(out_tmp,
                        sample_keys=np.array(sample_keys),
                        sample_tags=np.array(sample_tags),
                        curr_keys=np.array(curr_keys),
                        prev_keys=np.array(prev_keys),
                        d_yaws=d_yaws,
                        sample_fx=sample_fx,
                        ego_5d=ego_5d,
                        traj=traj,
                        target=target,
                        mask=mask,
                        obj_target=obj_target,
                        prev_dtick=int(z["prev_dtick"]) if "prev_dtick" in z.files else 5)
    os.replace(out_tmp, args.out)
    print("  done.")


if __name__ == "__main__":
    main()
