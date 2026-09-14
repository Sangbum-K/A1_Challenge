"""aux_targets_*/ 의 dist channel 에서 frame 별 min_dist_ahead 추출.

prebake / dataset 가 close-lead oversample / eval split 에 사용.

각 aux .npz: target shape (40, 7) → [obj, vL_abs, vY_abs, vL_rel, vY_rel, prox, dist]
min_dist_ahead = dist[obj>0.5].min() (없으면 +inf)

frame_key 형식: aux 파일명 stem == prebake 의 curr_key (둘 다 "track__policy__run__tNNNNNNN").

출력 npz:
  keys   (N,) str    "track__policy__run__tNNNNNNN"
  dists  (N,) float  m  (np.inf for no obj)
"""
from __future__ import annotations
import os, sys, argparse, glob, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np


DEFAULT_INS = [
    "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/aux_targets_fx320",
    "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/aux_targets_fx640",
    "/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/aux_targets_spd_fx640",
]
DEFAULT_OUT = "/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/cache/close_lead_meta_v2.npz"


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dirs", nargs="*", default=DEFAULT_INS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    return ap.parse_args()


def _norm_key(stem: str) -> str:
    """aux filename stem (track__policy__run__t{6d}) → curr_key (run__t{7d}).

    Examples:
      'sangam_track__MultiAgnet__260513_10__t015150' → '260513_10__t0015150'
      'sangam_track__MultiAgnet__260513__t004500'    → '260513__t0004500'
    """
    parts = stem.split("__")
    if len(parts) < 4 or not parts[-1].startswith("t"):
        return stem
    tick_str = parts[-1][1:]
    try:
        tick = int(tick_str)
    except ValueError:
        return stem
    run = parts[-2]
    return f"{run}__t{tick:07d}"


def process_file(path: str):
    try:
        z = np.load(path)
        t = z["target"]
        obj = t[:, 0]; dist = t[:, 6]
        valid = dist[obj > 0.5]
        d = float(valid.min()) if len(valid) else float("inf")
    except Exception:
        d = float("inf")
    return (_norm_key(Path(path).stem), d)


def main():
    args = get_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    files = []
    for d in args.in_dirs:
        if not os.path.isdir(d):
            print(f"[skip] {d} (not exist)")
            continue
        fs = sorted(glob.glob(os.path.join(d, "*.npz")))
        print(f"[in] {d}: {len(fs)} files")
        files.extend(fs)
    if args.limit: files = files[:args.limit]
    print(f"[total] {len(files)} aux files")

    keys = []; dists = []
    t0 = time.time(); n_done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for k, d in ex.map(process_file, files, chunksize=64):
            keys.append(k); dists.append(d)
            n_done += 1
            if n_done % 5000 == 0:
                dt = time.time() - t0
                rate = n_done / max(dt, 1e-3)
                eta  = (len(files) - n_done) / max(rate, 1e-3)
                print(f"  [{n_done}/{len(files)}] rate={rate:.0f}/s  ETA={eta:.0f}s", flush=True)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    dists_a = np.array(dists, dtype=np.float32)
    n_close30 = int((dists_a < 30).sum())
    n_close50 = int((dists_a < 50).sum())
    print(f"  distribution: <30m={n_close30}/{len(files)} ({n_close30/len(files)*100:.1f}%)  "
          f"<50m={n_close50} ({n_close50/len(files)*100:.1f}%)  inf={int(np.isinf(dists_a).sum())}")

    print(f"[saving] {args.out}")
    np.savez_compressed(args.out, keys=np.array(keys), dists=dists_a)
    print("  done.")


if __name__ == "__main__":
    main()
