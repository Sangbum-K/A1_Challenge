"""prebake post-process — distance × vrel 결합 smooth brake floor.

규칙 (per-sample, closest lead 의 dist + vrel):
  dist factor (가까울수록 강함):
    dist >= 10m  : 0  (no constraint)
    5 < dist < 10: (10 - dist) / 5
    dist <= 5m   : 1
  vrel-based max brake intensity (다가올수록 강함):
    vrel >= 0    : max -5 (cruise, 부드러운 brake)
    -5 < vrel < 0: linear -5 → -15
    vrel <= -5   : max -15 (강한 approach, full brake)

  floor = max_brake_at_vrel * dist_factor
  target_ax = min(orig, floor)   # 기존이 더 강하면 유지

예시:
  dist=10 vrel=any  → 0          (변경 없음)
  dist=5  vrel=0    → -5         (cruise close)
  dist=5  vrel=-5   → -15        (strong approach close)
  dist=8  vrel=-5   → -6
  dist=8  vrel=0    → -2         (cruise mid, mild)
  dist=8  vrel=-3   → -4.4

사용:
  python scripts/augment_prebake_smoothbrake.py \
      --prebake .../f1_prebake_iy_spd_streaming_brakeaug.npz \
      --aux-dir-spec 0519_agent1_pertX_pitX:.../aux_targets_spd_fx640 \
      --out .../f1_prebake_iy_spd_streaming_smoothbrake.npz
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np


def parse_aux_dir_spec(spec_list):
    m = {}
    for s in spec_list:
        if ":" not in s: continue
        p, d = s.split(":", 1)
        m[p] = Path(d)
    return m


def get_lead_dist_vrel(aux_path: Path):
    """aux .npz → closest lead (dist, vrel). (None, None) if no obj"""
    if not aux_path.exists(): return None, None
    try:
        t = np.load(aux_path)["target"]
        objs = t[:, 0] > 0.5
        if not objs.any(): return None, None
        dists = t[objs, 6]
        vrels = t[objs, 3]
        mi = int(np.argmin(dists))
        return float(dists[mi]), float(vrels[mi])
    except Exception:
        return None, None


def smooth_brake_floor(dist: float, vrel: float) -> float:
    """
    dist factor (가까울수록 강함):
        dist >= 10  : 0  (no constraint, returns +inf)
        5 < dist < 10: (10 - dist) / 5  ∈ (0, 1)
        dist <= 5   : 1
    vrel-based max brake (다가올수록 강함):
        vrel >= 0   : -5   (cruise, mild floor)
        -5 < vrel < 0: linear -5 → -15
        vrel <= -5  : -15  (strong approach)
    floor = max_brake * dist_factor      (0 to -15)
    """
    if dist >= 10.0:
        return float('inf')
    # dist factor
    if dist <= 5.0:
        df = 1.0
    else:
        df = (10.0 - dist) / 5.0      # dist=10→0, dist=5→1
    # vrel-based max brake
    if vrel >= 0.0:
        max_b = -5.0
    elif vrel <= -5.0:
        max_b = -15.0
    else:
        # vrel ∈ (-5, 0): linear -5 → -15
        max_b = -5.0 + (vrel / -5.0) * (-10.0)   # vrel=-3 → -11
    return max_b * df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prebake", required=True)
    ap.add_argument("--aux-dir-spec", nargs="*", default=[],
                    help="여러 'frame_id_prefix:aux_dir' (multi-policy)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    aux_map = parse_aux_dir_spec(args.aux_dir_spec)
    print(f"[smooth-brake] aux_map: {list(aux_map.keys())}", flush=True)

    z = np.load(args.prebake, allow_pickle=True)
    sample_keys = z["sample_keys"].tolist()
    curr_keys = z["curr_keys"].tolist()
    target = z["target"].copy()
    N = len(curr_keys)
    print(f"[smooth-brake] N samples: {N:,}", flush=True)

    n_modified = 0; n_aux_miss = 0
    n_far = 0   # dist >= 10 (no change)
    n_mid = 0   # 5 < dist < 10
    n_close = 0 # dist <= 5
    n_cruise = 0   # vrel >= 0
    n_approach = 0 # vrel <= -5
    cache = {}
    for i, ck in enumerate(curr_keys):
        if not ck or "__t" not in ck:   # invalid / empty key
            n_aux_miss += 1
            continue
        if ck in cache:
            dist, vrel = cache[ck]
        else:
            dist, vrel = None, None
            run, tk_str = ck.rsplit("__t", 1)
            try:
                tick = int(tk_str)
            except ValueError:
                n_aux_miss += 1
                cache[ck] = (None, None)
                continue
            for prefix, aux_dir in aux_map.items():
                fid = f"{prefix}__{run}__t{tick:07d}"
                dist, vrel = get_lead_dist_vrel(aux_dir / f"{fid}.npz")
                if dist is not None: break
            cache[ck] = (dist, vrel)
        if dist is None:
            n_aux_miss += 1; continue
        if dist >= 10.0:
            n_far += 1; continue
        floor = smooth_brake_floor(dist, vrel)
        if dist <= 5.0:
            n_close += 1
        else:
            n_mid += 1
        if vrel >= 0: n_cruise += 1
        elif vrel <= -5: n_approach += 1
        orig = target[i, :, 1].copy()
        target[i, :, 1] = np.minimum(orig, floor)
        if not np.allclose(orig, target[i, :, 1]):
            n_modified += 1
        if i % 100000 == 0 and i > 0:
            print(f"  [{i}/{N}] mod={n_modified} mid={n_mid} close={n_close} cruise={n_cruise} approach={n_approach} aux_miss={n_aux_miss}",
                  flush=True)

    print(f"\n[smooth-brake] DONE", flush=True)
    print(f"  N total      : {N:,}")
    print(f"  modified     : {n_modified:,}")
    print(f"  dist <= 5m   : {n_close:,}")
    print(f"  5 < dist <10 : {n_mid:,}")
    print(f"  dist >= 10m  : {n_far:,}  (no change)")
    print(f"  vrel >= 0 (cruise)    : {n_cruise:,}  (max -5)")
    print(f"  vrel <= -5 (approach) : {n_approach:,}  (max -15)")
    print(f"  aux miss     : {n_aux_miss:,}")

    # save
    out_tmp = args.out + ".partial.npz"
    print(f"[smooth-brake] saving → {args.out}", flush=True)
    save_kwargs = {}
    for k in z.files:
        if k == "target":
            save_kwargs[k] = target
        else:
            save_kwargs[k] = z[k]
    np.savez_compressed(out_tmp, **save_kwargs)
    os.replace(out_tmp, args.out)
    print("  done.")


if __name__ == "__main__":
    main()
