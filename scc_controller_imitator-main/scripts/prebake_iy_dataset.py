"""i_y refactor prebake — scc_synth_iy_dataset 호환.

scc_synth_pair_dataset 의 prebake_f1_dataset 와 동일하지만 추가로:
  - curr_keys   (N_samples,) str  "run__tNNNNNNN"
  - prev_keys   (N_samples,) str  (manifest 의 prev_dtick 기반)
  - d_yaws      (N_samples,) float32  curr_yaw - prev_yaw (rad)

→ i_1 token cache lookup 가능. fi_mem 사전 fusion 안 함 (model 측에서 i_y + i_2).

사용:
  python -u scripts/prebake_iy_dataset.py --prev-dtick 5
  python -u scripts/prebake_iy_dataset.py --prev-dtick 20 --out .../f1_prebake_iy_prev200.npz
"""
from __future__ import annotations
import os, sys, json, argparse, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.datamodule.scc_synth_pair_dataset import (
    DEG2RAD, TRAJ_N, TARG_N, DEFAULT_MANIFEST,
    _parse_ego, _parse_scc_txt, _load_rand_traj, _load_logged_traj, _world_to_ego_traj,
)

DEFAULT_OUT = "/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/prebake/f1_prebake_iy_v1.npz"

# fx 매핑 — Multi-Agent 계열 (기존 운영) 만 320, 그 외 모든 신규 정책은 640
# rule: substring match on policy. MultiAgnet/MultiAgent/MultiAgent_SCC 같은 표기 변형 모두 320 으로
FX_320_KEYWORDS = ("MultiAgnet", "MultiAgent")     # 오타 + 정상 모두 cover
DEFAULT_FX = 640.0


def _resolve_fx(policy: str) -> float:
    """policy (driver folder name) → fx.
    MultiAgnet/MultiAgent (legacy multi-agent) → 320, 그 외 신규 (agent1, DriverB_new, ...) → 640.
    """
    for kw in FX_320_KEYWORDS:
        if kw in policy:
            return 320.0
    return DEFAULT_FX


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--prev-dtick", type=int, default=20,
                    help="prev image offset tick (5=50ms, 20=200ms, ...). a_2 v4 학습 환경 (PREV_DTICK=20) 과 일치")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--aux-policy-dir", action="append", default=[],
                    help="policy:aux_dir 매핑 (예: 0519_pertO_pitX:.../aux_targets_sangam_fx640). "
                         "지정 시 obj_targets (N,40) 도 prebake 에 포함 — i_g gate 학습용")
    return ap.parse_args()


def process_sample(s, raw_base, rand_bank, log_bank, prev_dtick, pose_lookup, prev_yaw_lookup, aux_dir=None):
    """returns dict 또는 None. aux_dir 주어지면 obj_target (40,) 도 포함"""
    key = s["key"]
    if key not in pose_lookup: return None
    ex, ey, e_yaw, ego_5d = pose_lookup[key]
    run = s["run_id"]; tick = int(s["tick"])
    prev_tick = tick - prev_dtick

    # curr / prev image keys
    curr_key = f"{run}__t{tick:07d}"
    prev_key = f"{run}__t{prev_tick:07d}"
    # prev yaw (pre-cached, no NFS in sample loop)
    prev_yaw_key = (s["track"], s["policy"], run, prev_tick)
    py = prev_yaw_lookup.get(prev_yaw_key)
    d_yaw = (e_yaw - py) if py is not None else 0.0

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

    # target (lat_traj steer + long_traj ax)
    try:
        scc = _parse_scc_txt(Path(s["scc"]))
    except (FileNotFoundError, OSError):
        return None
    long_t = scc["long_traj"]; lat_t = scc["lat_traj"]
    target = np.zeros((TARG_N, 2), dtype=np.float32)
    mask = np.zeros((TARG_N,), dtype=np.bool_)
    m_real = min(long_t.shape[0], lat_t.shape[0], TARG_N)
    if m_real > 0:
        target[:m_real, 0] = lat_t[:m_real, 4]   # steer (deg)
        target[:m_real, 1] = long_t[:m_real, 3]  # ax
        mask[:m_real] = True
    # obj_target (40,) — aux 가 있으면 channel 0 (obj 0/1)
    obj_target = np.zeros(40, dtype=np.float32)
    if aux_dir is not None:
        ap2 = Path(aux_dir) / f"{s['frame_id']}.npz"
        if ap2.exists():
            try:
                t = np.load(ap2)["target"]
                obj_target = (t[:, 0] > 0.5).astype(np.float32)
            except Exception: pass
    return {
        "traj":       traj_ego.astype(np.float32),
        "target":     target,
        "mask":       mask,
        "curr_key":   curr_key,
        "prev_key":   prev_key,
        "d_yaw":      float(d_yaw),
        "ego_5d":     ego_5d.astype(np.float32),
        "obj_target": obj_target,
    }


def process_batch(args_tuple):
    batch_samples, raw_base, rand_bank, log_bank, prev_dtick, pose_lookup, prev_yaw_lookup, aux_policy_map = args_tuple
    return [(i, process_sample(s, raw_base, rand_bank, log_bank, prev_dtick,
                                pose_lookup, prev_yaw_lookup,
                                aux_dir=aux_policy_map.get(s.get("policy"))))
            for i, s in batch_samples]


def _parse_yaw_only(args_tuple):
    """worker: prev ego file 에서 yaw 만 추출. (key, yaw_rad) or (key, None)"""
    raw_base, track, policy, run, tick = args_tuple
    ego_p = Path(raw_base) / track / policy / run / "EGO_INFO" / f"{tick}.txt"
    ego = _parse_ego(ego_p)
    if ego is None: return ((track, policy, run, tick), None)
    return ((track, policy, run, tick), ego["orientation"][2] * DEG2RAD)


def _parse_curr_ego(args_tuple):
    """worker: curr ego — return (key, (ex, ey, e_yaw, ego_5d)) or (key, None)"""
    raw_base, key, track, policy, run, tick = args_tuple
    ego_p = Path(raw_base) / track / policy / run / "EGO_INFO" / f"{tick}.txt"
    ego = _parse_ego(ego_p)
    if ego is None: return (key, None)
    ex, ey = ego["position"][0], ego["position"][1]
    e_yaw = ego["orientation"][2] * DEG2RAD
    ego_5d = np.array([
        ego["velocity"][0], ego["velocity"][1],
        ego["acceleration"][0], ego["acceleration"][1],
        ego["angularVelocity"][2] * DEG2RAD,
    ], dtype=np.float32)
    return (key, (ex, ey, e_yaw, ego_5d))


def main():
    args = get_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"[prebake-iy] manifest {args.manifest}  prev_dtick={args.prev_dtick}", flush=True)
    m = json.load(open(args.manifest))
    samples = m["samples"]
    if args.limit: samples = samples[:args.limit]
    print(f"  samples: {len(samples)}", flush=True)

    # ego_5d + pose per frame_key
    frame_keys = sorted({s["key"] for s in samples})
    frame_meta = {}
    for s in samples:
        if s["key"] not in frame_meta: frame_meta[s["key"]] = s
    print(f"  unique frame keys: {len(frame_keys)}  parsing curr ego ...", flush=True)
    raw_p = Path(m["raw_base"])
    pose_lookup = {}

    # parallel ego parse (curr) — was sequential @ 4/s
    curr_args = [(m["raw_base"], k, frame_meta[k]["track"], frame_meta[k]["policy"],
                  frame_meta[k]["run_id"], frame_meta[k]["tick"]) for k in frame_keys]
    t0 = time.time(); n_ok = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (k, v) in enumerate(ex.map(_parse_curr_ego, curr_args, chunksize=64)):
            if v is None: continue
            pose_lookup[k] = v
            n_ok += 1
            if (i + 1) % 2000 == 0:
                dt = time.time() - t0
                print(f"  [{i+1}/{len(frame_keys)}] ego_ok={n_ok}  rate={(i+1)/dt:.0f}/s", flush=True)
    print(f"  curr ego ok={n_ok}/{len(frame_keys)} in {time.time()-t0:.1f}s", flush=True)

    # parallel prev ego parse (yaw only) — was per-sample NFS, now pre-cached
    prev_tick_keys = set()
    for s in samples:
        prev_tick_keys.add((s["track"], s["policy"], s["run_id"], int(s["tick"]) - args.prev_dtick))
    prev_args = [(m["raw_base"], t, p, r, tk) for (t, p, r, tk) in prev_tick_keys]
    prev_yaw_lookup = {}
    t0 = time.time(); n_pok = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (k, y) in enumerate(ex.map(_parse_yaw_only, prev_args, chunksize=64)):
            if y is not None:
                prev_yaw_lookup[k] = y
                n_pok += 1
            if (i + 1) % 5000 == 0:
                dt = time.time() - t0
                print(f"  [prev {i+1}/{len(prev_args)}] ok={n_pok}  rate={(i+1)/dt:.0f}/s", flush=True)
    print(f"  prev yaw ok={n_pok}/{len(prev_args)} in {time.time()-t0:.1f}s", flush=True)

    # aux policy → dir 매핑
    aux_policy_map = {}
    for s in args.aux_policy_dir:
        if ":" in s:
            pol, d = s.split(":", 1)
            aux_policy_map[pol] = d
    if aux_policy_map:
        print(f"[aux] obj_target bake from: {aux_policy_map}", flush=True)

    # parallel sample processing
    N = len(samples)
    traj_arr   = np.zeros((N, 40, 3), dtype=np.float32)
    target_arr = np.zeros((N, 40, 2), dtype=np.float32)
    mask_arr   = np.zeros((N, 40), dtype=np.bool_)
    d_yaw_arr  = np.zeros(N, dtype=np.float32)
    ego_5d_arr = np.zeros((N, 5), dtype=np.float32)
    obj_target_arr = np.zeros((N, 40), dtype=np.float32)
    curr_keys = [""] * N; prev_keys = [""] * N

    indexed = [(i, s) for i, s in enumerate(samples) if s["key"] in pose_lookup]
    print(f"  valid samples (curr ego ok): {len(indexed)}/{N}", flush=True)

    BATCH = 500
    batches = [indexed[i:i+BATCH] for i in range(0, len(indexed), BATCH)]
    t0 = time.time(); n_done = 0; n_ok = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_batch,
                              (b, m["raw_base"], m["rand_bank"], m["log_bank"], args.prev_dtick,
                               pose_lookup, prev_yaw_lookup, aux_policy_map))
                     for b in batches]
        for fi, fut in enumerate(as_completed(futures)):
            for sample_i, r in fut.result():
                if r is None: continue
                traj_arr[sample_i]       = r["traj"]
                target_arr[sample_i]     = r["target"]
                mask_arr[sample_i]       = r["mask"]
                d_yaw_arr[sample_i]      = r["d_yaw"]
                ego_5d_arr[sample_i]     = r["ego_5d"]
                obj_target_arr[sample_i] = r["obj_target"]
                curr_keys[sample_i]      = r["curr_key"]
                prev_keys[sample_i]      = r["prev_key"]
                n_ok += 1
            n_done += len(fut.result())
            if (fi + 1) % 10 == 0:
                dt = time.time() - t0
                rate = n_done / max(dt, 1e-3)
                eta  = (len(indexed) - n_done) / max(rate, 1e-3)
                print(f"  [{n_done}/{len(indexed)}] ok={n_ok}  rate={rate:.0f}/s  ETA={eta:.0f}s", flush=True)
    print(f"  done in {time.time()-t0:.1f}s, ok={n_ok}/{N}", flush=True)

    # sample tags / keys / fx
    sample_keys = []
    sample_tags = []
    sample_fx   = np.zeros(N, dtype=np.float32)
    for i, s in enumerate(samples):
        sample_keys.append(s["key"] + "__" + (s["tag"] if s["tag"]=="logged" else f"rand_{s['rand_k']:05d}"))
        sample_tags.append(s["tag"])
        sample_fx[i] = _resolve_fx(s["policy"])
    fx_unique, fx_counts = np.unique(sample_fx, return_counts=True)
    print(f"  fx distribution: {dict(zip(fx_unique.tolist(), fx_counts.tolist()))}", flush=True)

    # atomic write: 임시 path 에 저장 후 rename — streaming train reload race 방지
    # np.savez_compressed 는 .npz 확장자 자동 부여하므로, 임시 path 도 .npz 끝나야 함
    out_tmp = args.out + ".partial.npz"
    print(f"[prebake-iy] saving → {args.out} (atomic via {out_tmp})", flush=True)
    np.savez_compressed(out_tmp,
                        sample_keys=np.array(sample_keys),
                        sample_tags=np.array(sample_tags),
                        curr_keys=np.array(curr_keys),
                        prev_keys=np.array(prev_keys),
                        d_yaws=d_yaw_arr,
                        sample_fx=sample_fx,
                        ego_5d=ego_5d_arr,
                        traj=traj_arr,
                        target=target_arr,
                        mask=mask_arr,
                        obj_target=obj_target_arr,
                        prev_dtick=args.prev_dtick)
    os.replace(out_tmp, args.out)
    print("  done.", flush=True)


if __name__ == "__main__":
    main()
