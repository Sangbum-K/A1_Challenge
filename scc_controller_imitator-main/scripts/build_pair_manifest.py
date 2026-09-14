"""신규 track/policy 의 scc_synth_v2_*/ + scc_traj_bank_*/ + scc_logged_bank_*/ →
scc_synth_pair_manifest_*.json 생성.

기존 manifest 형식 (raw_base, rand_bank, log_bank, samples=[{key, track, policy, run_id, tick, scc, tag, rand_k}]).

사용:
  python scripts/build_pair_manifest.py \
      --frame-list .../scc_frame_list_spd.txt \
      --track speedway_track --policy 0519_agent1_pertX_pitX \
      --raw-base /home/ailab-12/AILabSSD/02_Data/57_A1_E2E/00_raw \
      --scc-dir .../scc_synth_v2_spd \
      --rand-bank .../scc_traj_bank_spd \
      --log-bank .../scc_logged_bank_spd \
      --k 100 \
      --out .../scc_synth_pair_manifest_spd.json
"""
from __future__ import annotations
import argparse, json, time, os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-list", type=Path, required=True)
    ap.add_argument("--track",      required=True)
    ap.add_argument("--policy",     required=True)
    ap.add_argument("--raw-base",   type=Path, required=True)
    ap.add_argument("--scc-dir",    type=Path, required=True)
    ap.add_argument("--rand-bank",  type=Path, required=True)
    ap.add_argument("--log-bank",   type=Path, required=True)
    ap.add_argument("--k",          type=int, required=True,
                    help="frame 당 trajectory 수 (logged 1 + random K-1)")
    ap.add_argument("--out",        type=Path, required=True)
    ap.add_argument("--filter-existing", action="store_true",
                    help="scc_dir 에 실제로 있는 sample 만 manifest 에 포함 (권장)")
    args = ap.parse_args()

    K_rand = args.k - 1

    # frame_list 의 (run, tick) 만 사용 — 각각 1 logged + K_rand random samples
    frames = []
    with open(args.frame_list) as f:
        for line in f:
            toks = line.strip().split()
            if len(toks) < 2 or toks[0].startswith("#"): continue
            frames.append((toks[0], int(toks[1])))
    print(f"[manifest] {len(frames)} frames; K={args.k}")

    # scc 파일 존재 set (filter-existing 일 때만)
    scc_existing = None
    if args.filter_existing:
        print(f"[manifest] scanning {args.scc_dir} for existing scc.txt …")
        t0 = time.time()
        scc_existing = {p.name for p in args.scc_dir.glob("*.scc.txt")}
        print(f"  {len(scc_existing)} scc.txt files in {time.time()-t0:.1f}s")

    samples = []; n_filt_logged = n_filt_rand = 0
    for run, tick in frames:
        key = f"{run}__t{tick:07d}"
        frame_id = f"{args.track}__{args.policy}__{run}__t{tick:07d}"
        # logged
        scc_logged = args.scc_dir / f"{key}__logged.scc.txt"
        if scc_existing is None or scc_logged.name in scc_existing:
            samples.append({
                "scc":      str(scc_logged), "key": key, "tag": "logged", "rand_k": -1,
                "frame_id": frame_id, "run_id": run, "tick": tick,
                "track":    args.track, "policy": args.policy,
            })
        else: n_filt_logged += 1
        # random
        for k in range(K_rand):
            scc_rand = args.scc_dir / f"{key}__rand_{k:05d}.scc.txt"
            if scc_existing is None or scc_rand.name in scc_existing:
                samples.append({
                    "scc":      str(scc_rand), "key": key, "tag": "rand", "rand_k": k,
                    "frame_id": frame_id, "run_id": run, "tick": tick,
                    "track":    args.track, "policy": args.policy,
                })
            else: n_filt_rand += 1
    print(f"[manifest] {len(samples)} samples (filtered: logged={n_filt_logged} rand={n_filt_rand})")

    manifest = {
        "raw_base":  str(args.raw_base),
        "rand_bank": str(args.rand_bank),
        "log_bank":  str(args.log_bank),
        "track":     args.track,
        "policy":    args.policy,
        "k":         args.k,
        "samples":   samples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(manifest, f)
    sz = os.path.getsize(args.out)
    print(f"[manifest] wrote {sz/1e6:.1f} MB → {args.out}")


if __name__ == "__main__":
    main()
