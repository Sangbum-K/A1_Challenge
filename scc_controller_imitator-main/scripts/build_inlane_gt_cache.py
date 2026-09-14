"""GT inlane cache 빌더 — aux_targets 에서 frame 별 in-lane(bin16-25) lead 의
GT dist + closing(-vL_rel) 계산. ttc-augment 가 추론값(과대평가) 대신 GT 사용하도록.

각 frame: in-lane bin(16-25) 중 obj=True 인 최근접 lead 의
  inlane_dist    = GT dist (m)
  inlane_closing = -GT vL_rel (m/s, 양수=접근)   ← frame-diff 불필요, GT 직접

사용:
  python scripts/build_inlane_gt_cache.py --aux-dir <dir> --out <npz> [--run-filter <substr>]
"""
import argparse, os, sys
from pathlib import Path
import numpy as np

BIN_LO, BIN_HI = 16, 25
SCALE_D = 40.0   # aux dist 가 normalized 면 ×40

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aux-dir", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-filter", default="", help="파일명에 이 substr 포함된 것만 (빈값=전체)")
    args = ap.parse_args()

    keys, idist, iclos = [], [], []
    n_obj = 0
    for ad in args.aux_dir:
        ad = Path(ad)
        files = [f for f in os.listdir(ad) if f.endswith(".npz") and (args.run_filter in f)]
        print(f"[{ad.name}] {len(files)} files (filter='{args.run_filter}')", flush=True)
        for f in files:
            stem = f[:-4]; toks = stem.split("__")
            key = "__".join(toks[-2:]) if len(toks) >= 2 else stem
            a = np.load(ad / f, allow_pickle=True)["target"]   # (40,7)
            cen = a[BIN_LO:BIN_HI+1]
            om = cen[:, 0] > 0.5
            keys.append(key)
            if om.sum() == 0:
                idist.append(np.inf); iclos.append(0.0); continue
            d = cen[om, 6].astype(np.float64)
            if d.max() < 2.0: d = d * SCALE_D
            j = int(np.argmin(d))
            vL = float(cen[om, 3][j])
            if abs(vL) < 2.0 and cen[om, 3].max() < 2.0:  # vL 도 normalized 인 경우 방어 (대개 raw)
                pass
            idist.append(float(d[j]))
            iclos.append(-vL)            # 양수 = 접근
            n_obj += 1
    keys = np.array(keys)
    idist = np.array(idist, dtype=np.float32)
    iclos = np.array(iclos, dtype=np.float32)
    np.savez_compressed(args.out, keys=keys, inlane_dist=idist, inlane_closing=iclos)
    print(f"[saved] {args.out}  N={len(keys)}  in-lane lead 있음={n_obj} ({n_obj/max(len(keys),1)*100:.0f}%)", flush=True)
    m = np.isfinite(idist)
    if m.any():
        print(f"  inlane_dist (lead): mean={idist[m].mean():.1f}m  closing: mean={iclos[m].mean():+.2f} (접근>0)", flush=True)

if __name__ == "__main__":
    main()
