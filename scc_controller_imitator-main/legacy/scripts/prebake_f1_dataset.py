"""f_1 학습 데이터 전체 pre-bake (in-memory dataset 용).

manifest 모든 sample 에 대해:
  - traj_ego (40, 3)   [x, y, speed] in ego frame
  - target  (40, 2)    [steer deg, ax m/s²]  long_traj 처음 40 row
  - mask    (40,)      valid target steps

frame 별 ego_5d (4570 unique) 도 별도 저장.
출력 단일 npz:
  - sample_keys      : list[str]  (key__tag 형식)
  - frame_keys       : list[str]  (run__tNNNNNNN)
  - sample_frame_idx : (N_samples,)  → frame_keys 의 인덱스
  - ego_5d           : (N_frames, 5)
  - traj            : (N_samples, 40, 3)
  - target          : (N_samples, 40, 2)
  - mask            : (N_samples, 40) bool
"""
from __future__ import annotations
import os, sys, json, argparse, time, struct
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.datamodule.scc_synth_pair_dataset import (
    DEG2RAD, TRAJ_N, TARG_N, DEFAULT_MANIFEST,
    _parse_ego, _parse_scc_txt, _load_rand_traj, _load_logged_traj, _world_to_ego_traj,
)


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--out", default="/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/f1_prebake_v1.npz")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    return ap.parse_args()


def process_sample(s, raw_base, rand_bank, log_bank, frame_key_to_pose):
    """returns dict or None."""
    key = s["key"]
    pose = frame_key_to_pose.get(key)
    if pose is None: return None
    ex, ey, e_yaw = pose
    # trajectory
    if s["tag"] == "logged":
        bp = Path(log_bank) / f"{key}.bin"
        if not bp.exists(): return None
        traj_w = _load_logged_traj(bp)
    else:
        bp = Path(rand_bank) / f"{key}.trajs.bin"
        if not bp.exists(): return None
        traj_w = _load_rand_traj(bp, s["rand_k"])
        if traj_w is None: return None
    traj_ego = _world_to_ego_traj(traj_w[:TRAJ_N], ex, ey, e_yaw)
    if traj_ego.shape[0] < TRAJ_N:
        pad = np.tile(traj_ego[-1:], (TRAJ_N - traj_ego.shape[0], 1))
        traj_ego = np.concatenate([traj_ego, pad], axis=0)
    # target: lat_traj 의 steer (col 4, deg) + long_traj 의 ax (col 3, m/s²)
    try:
        scc = _parse_scc_txt(Path(s["scc"]))
    except (FileNotFoundError, OSError):
        return None
    long_t = scc["long_traj"]; lat_t = scc["lat_traj"]
    target = np.zeros((TARG_N, 2), dtype=np.float32)
    mask = np.zeros((TARG_N,), dtype=np.bool_)
    m_real = min(long_t.shape[0], lat_t.shape[0], TARG_N)
    if m_real > 0:
        target[:m_real, 0] = lat_t[:m_real, 4]   # lateral steer (rad)
        target[:m_real, 1] = long_t[:m_real, 3]  # longitudinal ax (m/s²)
        mask[:m_real] = True
    return {
        "traj": traj_ego.astype(np.float32),
        "target": target,
        "mask": mask,
    }


def process_batch(args_tuple):
    """worker entry: process a batch of samples."""
    batch_samples, raw_base, rand_bank, log_bank, frame_key_to_pose = args_tuple
    out = []
    for i, s in batch_samples:
        r = process_sample(s, raw_base, rand_bank, log_bank, frame_key_to_pose)
        out.append((i, r))
    return out


def main():
    args = get_args()
    print(f"[prebake] manifest {args.manifest}", flush=True)
    m = json.load(open(args.manifest))
    samples = m["samples"]
    if args.limit: samples = samples[:args.limit]
    print(f"  samples: {len(samples)}", flush=True)

    # frame_keys (unique) + ego_5d preload (sequential, 4570 reads only)
    frame_keys = sorted({s["key"] for s in samples})
    print(f"  unique frame keys: {len(frame_keys)}", flush=True)
    raw_p = Path(m["raw_base"])
    # build (run, tick, track, policy) lookup per frame_key
    frame_meta = {}
    for s in samples:
        if s["key"] not in frame_meta: frame_meta[s["key"]] = s
    print(f"[prebake] reading ego for {len(frame_keys)} frames ...", flush=True)
    ego_5d_arr = np.zeros((len(frame_keys), 5), dtype=np.float32)
    pose_lookup = {}      # key → (ex, ey, e_yaw)
    t0 = time.time(); n_ok = 0
    for i, k in enumerate(frame_keys):
        s = frame_meta[k]
        ego_p = raw_p / s["track"] / s["policy"] / s["run_id"] / "EGO_INFO" / f"{s['tick']}.txt"
        ego = _parse_ego(ego_p)
        if ego is None: continue
        ex, ey = ego["position"][0], ego["position"][1]
        e_yaw = ego["orientation"][2] * DEG2RAD
        ego_5d_arr[i] = [ego["velocity"][0], ego["velocity"][1],
                          ego["acceleration"][0], ego["acceleration"][1],
                          ego["angularVelocity"][2] * DEG2RAD]
        pose_lookup[k] = (ex, ey, e_yaw)
        n_ok += 1
        if (i + 1) % 1000 == 0:
            dt = time.time() - t0
            print(f"  [{i+1}/{len(frame_keys)}] ok={n_ok}  rate={(i+1)/dt:.1f}/s", flush=True)
    print(f"  ego ok={n_ok}/{len(frame_keys)}  in {time.time()-t0:.1f}s", flush=True)

    # frame_key → idx
    frame_key_to_idx = {k: i for i, k in enumerate(frame_keys)}

    # per-sample processing (parallel)
    print(f"[prebake] processing {len(samples)} samples (workers={args.workers}) ...", flush=True)
    N = len(samples)
    traj_arr = np.zeros((N, 40, 3), dtype=np.float32)
    target_arr = np.zeros((N, 40, 2), dtype=np.float32)
    mask_arr = np.zeros((N, 40), dtype=np.bool_)
    sample_frame_idx = np.zeros(N, dtype=np.int32)
    sample_tags = []
    sample_keys = []

    # 사용 가능한 frame 만 처리
    indexed = [(i, s) for i, s in enumerate(samples) if s["key"] in pose_lookup]
    print(f"  samples with valid ego: {len(indexed)}/{N}", flush=True)

    BATCH = 500
    batches = [indexed[i:i+BATCH] for i in range(0, len(indexed), BATCH)]
    t0 = time.time(); n_done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = []
        for b in batches:
            futures.append(ex.submit(
                process_batch,
                (b, m["raw_base"], m["rand_bank"], m["log_bank"], pose_lookup)))
        for fi, fut in enumerate(as_completed(futures)):
            results = fut.result()
            for sample_i, r in results:
                if r is None: continue
                traj_arr[sample_i] = r["traj"]
                target_arr[sample_i] = r["target"]
                mask_arr[sample_i] = r["mask"]
                sample_frame_idx[sample_i] = frame_key_to_idx[samples[sample_i]["key"]]
            n_done += len(results)
            if (fi + 1) % 10 == 0:
                dt = time.time() - t0
                rate = n_done / max(dt, 1e-3)
                eta = (len(indexed) - n_done) / max(rate, 1e-3)
                print(f"  [{n_done}/{len(indexed)}] rate={rate:.0f}/s  ETA={eta:.0f}s", flush=True)
    print(f"  processed in {time.time()-t0:.1f}s", flush=True)

    # also build sample tags / keys
    for s in samples:
        sample_keys.append(s["key"] + "__" + (s["tag"] if s["tag"]=="logged" else f"rand_{s['rand_k']:05d}"))
        sample_tags.append(s["tag"])

    out = args.out
    print(f"[prebake] saving → {out}  (~{(traj_arr.nbytes + target_arr.nbytes + mask_arr.nbytes + ego_5d_arr.nbytes)/1e6:.0f} MB raw)", flush=True)
    np.savez_compressed(out,
                        frame_keys=np.array(frame_keys),
                        ego_5d=ego_5d_arr,
                        sample_keys=np.array(sample_keys),
                        sample_tags=np.array(sample_tags),
                        sample_frame_idx=sample_frame_idx,
                        traj=traj_arr,
                        target=target_arr,
                        mask=mask_arr)
    print(f"  done.", flush=True)


if __name__ == "__main__":
    main()
