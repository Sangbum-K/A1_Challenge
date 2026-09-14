"""manifest 의 scc 경로 prefix 일괄 치환 (NFS → local 등).

사용:
  python scripts/relocate_manifest_paths.py \
      --in  scc_synth_pair_manifest.json \
      --from /home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/scc_synth_v2 \
      --to   /home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/scc_synth_v2_sangam \
      --out scc_synth_pair_manifest_local.json
"""
from __future__ import annotations
import json, argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",   dest="inp", type=Path, required=True)
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--to",   dest="dst", required=True)
    ap.add_argument("--also-rand-bank", default=None, help="rand_bank prefix 도 같이 치환 (없으면 안 건드림)")
    ap.add_argument("--also-log-bank",  default=None, help="log_bank prefix 도 같이 치환")
    ap.add_argument("--out",  type=Path, required=True)
    args = ap.parse_args()

    print(f"[load] {args.inp}")
    m = json.load(open(args.inp))
    print(f"  samples: {len(m['samples'])}")
    n_replaced = 0
    for s in m["samples"]:
        if "scc" in s and s["scc"].startswith(args.src):
            s["scc"] = args.dst + s["scc"][len(args.src):]
            n_replaced += 1
    print(f"  scc replaced: {n_replaced}")

    if args.also_rand_bank and m.get("rand_bank"):
        m["rand_bank"] = args.also_rand_bank
        print(f"  rand_bank -> {args.also_rand_bank}")
    if args.also_log_bank and m.get("log_bank"):
        m["log_bank"] = args.also_log_bank
        print(f"  log_bank -> {args.also_log_bank}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(m, f)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
