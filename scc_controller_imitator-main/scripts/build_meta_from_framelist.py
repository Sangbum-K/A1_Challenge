"""scc_frame_list_*.txt + raw 경로 → multiagnet_compact_meta.json 호환 meta json.

prepare_a2_aux_targets.py / prebake_iy_dataset.py 가 입력으로 받는 형식:
  [{frame_id, track, policy, run_id, tick}, ...]

사용:
  python scripts/build_meta_from_framelist.py \
      --frame-list .../scc_frame_list_spd.txt \
      --track speedway_track --policy 0519_agent1_pertX_pitX \
      --out .../speedway_agent1_meta.json
"""
from __future__ import annotations
import json, argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-list", type=Path, required=True)
    ap.add_argument("--track",      required=True)
    ap.add_argument("--policy",     required=True)
    ap.add_argument("--out",        type=Path, required=True)
    args = ap.parse_args()

    entries = []
    with open(args.frame_list) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            toks = line.split()
            if len(toks) < 2: continue
            run = toks[0]; tick = int(toks[1])
            entries.append({
                "frame_id": f"{args.track}__{args.policy}__{run}__t{tick:07d}",
                "track":    args.track,
                "policy":   args.policy,
                "run_id":   run,
                "tick":     tick,
            })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(entries, open(args.out, "w"), indent=2)
    print(f"[meta] wrote {len(entries)} entries → {args.out}")


if __name__ == "__main__":
    main()
