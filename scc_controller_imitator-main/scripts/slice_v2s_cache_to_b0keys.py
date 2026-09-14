"""v2_s i1 cache 를 b0 subset cache 와 동일 key 집합으로 잘라 저장 (공정 비교용).

b0 cache(stride subset) 가 가진 key 만 v2_s cache 에서 추출 → 두 eval 이 정확히 동일 frame.
GPU 불필요 (디스크 슬라이싱).

사용:
  python scripts/slice_v2s_cache_to_b0keys.py \
    --v2s-cache <v2s 135k cache.npz> --b0-cache <b0 40k subset cache.npz> \
    --out <v2s 40k subset cache.npz>
"""
import argparse, json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2s-cache", required=True)
    ap.add_argument("--b0-cache", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    zb = np.load(args.b0_cache, allow_pickle=True)
    b0_keys = zb["keys"].tolist()
    b0_set = set(b0_keys)
    print(f"[slice] b0 subset keys={len(b0_keys):,}")

    zv = np.load(args.v2s_cache, allow_pickle=True)
    v2s_keys = zv["keys"].tolist()
    v2s_tok = zv["tokens"]
    k2i = {k: i for i, k in enumerate(v2s_keys)}
    miss = [k for k in b0_keys if k not in k2i]
    if miss:
        raise RuntimeError(f"[FAIL] {len(miss)} b0 keys missing in v2s cache (e.g. {miss[:3]})")

    # b0 cache 의 key 순서 그대로 v2s token 추출 → 동일 순서·집합
    sel = np.array([k2i[k] for k in b0_keys])
    out_tok = v2s_tok[sel]
    print(f"[slice] v2s subset tokens {out_tok.shape} (C_b={out_tok.shape[-1]})")
    np.savez_compressed(
        args.out,
        keys=np.array(b0_keys),
        tokens=out_tok,
        meta=json.dumps({"sliced_from": args.v2s_cache, "match_b0": args.b0_cache,
                         "n": len(b0_keys)}))
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
