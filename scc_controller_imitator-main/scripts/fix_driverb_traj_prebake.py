"""DriverB_new prebake traj corruption fix.

Root cause: the original prebake `f1_prebake_iy_DriverB_new_logged_fixed_prev20.npz`
stored a corrupted `traj` array (speed↔per-step-displacement corr ≈ 0.23, speed never < 25 m/s).
Key alignment (curr_key ↔ manifest key) is CORRECT; the raw logged bank + raw EGO_INFO are FINE.
Recomputing `_world_to_ego_traj(raw_bin, raw_ego_pose)` reproduces the healthy traj
(disp↔speed corr ≈ 0.98). So we rebuild ONLY the `traj` array from the raw source and
copy every other array verbatim. Index alignment preserved; invalid (all-zero) samples
are left untouched.

Output: f1_prebake_iy_DriverB_new_logged_fixed_prev20_trajfix.npz  (never overwrites original).
"""
from __future__ import annotations
import os, sys, json, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.datamodule.scc_synth_pair_dataset import (
    _load_logged_traj, _world_to_ego_traj, _parse_ego, DEG2RAD, TRAJ_N,
)

DATA = "/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data"
ORIG = f"{DATA}/prebake/f1_prebake_iy_DriverB_new_logged_fixed_prev20.npz"
OUT  = f"{DATA}/prebake/f1_prebake_iy_DriverB_new_logged_fixed_prev20_trajfix.npz"
MANIFEST = f"{DATA}/manifest_i1cache_DriverB_new.json"


def _rebuild_one(args):
    """returns (idx, traj40x3 or None)."""
    idx, key, run, tick, raw_base, track, policy, log_bank = args
    binp = Path(log_bank) / f"{key}.bin"
    if not binp.exists():
        return (idx, None)
    egop = Path(raw_base) / track / policy / run / "EGO_INFO" / f"{tick}.txt"
    ego = _parse_ego(egop)
    if ego is None:
        return (idx, None)
    traj_w = _load_logged_traj(binp)  # (N,4) world [x,y,yaw,speed]
    ex, ey = ego["position"][0], ego["position"][1]
    e_yaw = ego["orientation"][2] * DEG2RAD
    traj_ego = _world_to_ego_traj(traj_w[:TRAJ_N], ex, ey, e_yaw)  # (N,3) ego [x_fwd,y_left,speed]
    if traj_ego.shape[0] < TRAJ_N:
        pad = np.tile(traj_ego[-1:], (TRAJ_N - traj_ego.shape[0], 1))
        traj_ego = np.concatenate([traj_ego, pad], axis=0)
    return (idx, traj_ego.astype(np.float32))


def main():
    workers = 12
    print(f"[trajfix] loading original prebake {ORIG}", flush=True)
    d = np.load(ORIG, allow_pickle=True)
    keys = list(d.keys())
    arrs = {k: d[k] for k in keys}
    N = arrs["traj"].shape[0]
    print(f"  N={N}  arrays={keys}", flush=True)

    m = json.load(open(MANIFEST))
    samples = m["samples"]
    raw_base = m["raw_base"]; track = m["track"]; policy = m["policy"]; log_bank = m["log_bank"]
    assert len(samples) == N, f"manifest N {len(samples)} != prebake N {N}"

    tags = arrs["sample_tags"]
    curr_keys = arrs["curr_keys"]

    # Only rebuild samples that the original build accepted (non-invalid: have a curr_key).
    # Invalid (all-zero) samples are preserved verbatim to keep training validity semantics.
    todo = []
    for i, s in enumerate(samples):
        if tags[i] == "invalid" or not curr_keys[i]:
            continue
        # sanity: curr_key alignment
        assert curr_keys[i] == s["key"], f"key misalign idx {i}: {curr_keys[i]} != {s['key']}"
        todo.append((i, s["key"], s["run_id"], int(s["tick"]),
                     raw_base, track, policy, log_bank))
    print(f"  rebuilding traj for {len(todo)} valid samples "
          f"({N - len(todo)} invalid preserved as-is)", flush=True)

    new_traj = arrs["traj"].copy()
    t0 = time.time(); n_ok = 0; n_fail = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for j, (idx, tj) in enumerate(ex.map(_rebuild_one, todo, chunksize=128)):
            if tj is None:
                n_fail += 1
                continue
            new_traj[idx] = tj
            n_ok += 1
            if (j + 1) % 20000 == 0:
                dt = time.time() - t0
                print(f"  [{j+1}/{len(todo)}] ok={n_ok} fail={n_fail} rate={(j+1)/dt:.0f}/s", flush=True)
    print(f"  rebuild done: ok={n_ok} fail={n_fail} in {time.time()-t0:.1f}s", flush=True)
    if n_fail:
        print(f"  WARNING: {n_fail} samples failed to rebuild (raw missing) — kept original traj", flush=True)

    arrs["traj"] = new_traj
    out_tmp = OUT + ".partial.npz"
    print(f"[trajfix] saving → {OUT}", flush=True)
    np.savez_compressed(out_tmp, **arrs)
    os.replace(out_tmp, OUT)
    print("  done.", flush=True)


if __name__ == "__main__":
    main()
