"""prebake post-process — a_2 cache 의 in-lane lead 정보로 TTC-based ax brake floor 적용.

기존 augment_prebake_brake.py 와 비슷하지만:
  - aux GT 대신 a_2 cache (이제 fix 됨, AUC 0.98) 사용 — 모든 dataset 통일
  - in-lane gating (bin 16-25, obj logit > 0)
  - TTC kinematic: safe_decel = closing² / (2 × max(dist-margin, 0.1))
  - target[:, :TTC_STEPS, 1] = min(orig, -safe_decel)  (앞 N step 만)

사용:
  python scripts/augment_prebake_a2_inlane.py \
      --prebake .../f1_prebake_iy_sangam_lead_0520_v1_brakeaug_prev20.npz \
      --a2-cache .../cache/a2_out_cache_sangam_lead_0520_prev20_fixed.npz \
      --out .../f1_prebake_iy_sangam_lead_0520_v1_a2inlane_prev20.npz
"""
from __future__ import annotations
import argparse, os, time
from pathlib import Path
import numpy as np

# 학습 코드 (scc_synth_iy_dataset.py) 와 동일 hyperparams
BIN_LO, BIN_HI = 16, 25
OBJ_THR = 0.0
DIST_SCALE = 40.0
DTICK_CLOSING = 10
DT_S = 0.1
# TTC augment (학습 코드의 ttc-augment flags 와 동일)
TTC_STEPS = 10              # 앞 10 step 만 floor
MAX_BRAKE = 20.0
DIST_MAX = 50.0
TTC_MAX = 4.0
MARGIN = 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prebake", required=True)
    ap.add_argument("--a2-cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ttc-steps", type=int, default=TTC_STEPS)
    ap.add_argument("--max-brake", type=float, default=MAX_BRAKE)
    ap.add_argument("--dist-max", type=float, default=DIST_MAX)
    ap.add_argument("--ttc-max", type=float, default=TTC_MAX)
    ap.add_argument("--margin", type=float, default=MARGIN)
    args = ap.parse_args()

    print(f"[a2-inlane-aug] prebake: {args.prebake}")
    z = np.load(args.prebake, allow_pickle=True)
    sample_keys = z["sample_keys"].tolist()
    sample_tags = z["sample_tags"].tolist()
    curr_keys = z["curr_keys"].tolist()
    prev_keys = z["prev_keys"].tolist()
    d_yaws = z["d_yaws"]
    sample_fx = z["sample_fx"] if "sample_fx" in z.files else np.full(len(sample_keys), 640.0, dtype=np.float32)
    ego_5d = z["ego_5d"]; traj = z["traj"]
    target = z["target"].copy()         # (N, 40, 2)
    mask = z["mask"]
    obj_target = z["obj_target"]
    N = len(sample_keys)
    print(f"[a2-inlane-aug] N samples: {N:,}")

    print(f"[a2-inlane-aug] a_2 cache: {args.a2_cache}")
    ca = np.load(args.a2_cache, allow_pickle=True)
    a2_out = ca["a2_out"]   # (N_frame, 40, 7)
    a2_keys = ca["keys"].tolist()
    k2idx = {k: i for i, k in enumerate(a2_keys)}
    print(f"[a2-inlane-aug] cache frames: {len(a2_keys):,}")

    # per-curr_key cache (in-lane min dist) — 같은 curr_key 의 모든 sample 이 같은 lead info 공유
    cache_lookup = {}
    n_mod = 0; n_inlane = 0; n_no_inlane = 0; n_cache_miss = 0
    t0 = time.time()
    for i in range(N):
        ck = curr_keys[i]
        if ck in cache_lookup:
            d_now, closing = cache_lookup[ck]
        else:
            if ck not in k2idx:
                cache_lookup[ck] = (None, None); n_cache_miss += 1; continue
            a2c = a2_out[k2idx[ck]].astype(np.float32)
            obj_in = a2c[BIN_LO:BIN_HI+1, 0]
            mask_obj = obj_in > OBJ_THR
            if not mask_obj.any():
                cache_lookup[ck] = (None, None); n_no_inlane += 1; continue
            d_arr = a2c[BIN_LO:BIN_HI+1, 6] * DIST_SCALE
            d_now = float(d_arr[mask_obj].min())
            # closing from prev
            closing = 0.0
            try:
                run, t_str = ck.rsplit("__t", 1)
                pk_run = f"{run}__t{int(t_str) - DTICK_CLOSING:07d}"
                if pk_run in k2idx:
                    a2p = a2_out[k2idx[pk_run]].astype(np.float32)
                    op = a2p[BIN_LO:BIN_HI+1, 0] > OBJ_THR
                    if op.any():
                        d_prev = float((a2p[BIN_LO:BIN_HI+1, 6] * DIST_SCALE)[op].min())
                        closing = (d_prev - d_now) / DT_S
            except Exception: pass
            cache_lookup[ck] = (d_now, closing)
            n_inlane += 1

        if d_now is None: continue
        # TTC augment 조건
        if closing > 0.1:
            ttc = d_now / max(closing, 0.1)
            if d_now < args.dist_max or ttc < args.ttc_max:
                eff_d = max(d_now - args.margin, 0.1)
                safe_decel = min((closing ** 2) / (2.0 * eff_d), args.max_brake)
                orig = target[i, :args.ttc_steps, 1].copy()
                target[i, :args.ttc_steps, 1] = np.minimum(orig, -safe_decel)
                if not np.allclose(orig, target[i, :args.ttc_steps, 1]):
                    n_mod += 1
        if (i+1) % 200000 == 0:
            print(f"  [{i+1:,}/{N:,}] modified={n_mod:,}  in-lane={n_inlane:,}  cache-miss={n_cache_miss:,}  elapsed={time.time()-t0:.1f}s")

    print(f"[a2-inlane-aug] DONE elapsed={time.time()-t0:.1f}s")
    print(f"  total samples: {N:,}")
    print(f"  unique curr_keys: {len(cache_lookup):,}")
    print(f"  in-lane (unique frames): {n_inlane:,}")
    print(f"  no in-lane: {n_no_inlane:,}")
    print(f"  cache miss: {n_cache_miss:,}")
    print(f"  modified samples: {n_mod:,} ({n_mod/N*100:.1f}%)")
    print(f"  conditions: in-lane obj>{OBJ_THR}, closing>0.1, dist<{args.dist_max} OR ttc<{args.ttc_max}")
    print(f"             safe_decel = closing²/(2·(dist-{args.margin})), max {args.max_brake} m/s²")

    out_tmp = args.out + ".partial.npz"
    print(f"[a2-inlane-aug] saving → {args.out}")
    np.savez_compressed(out_tmp,
                        sample_keys=np.array(sample_keys),
                        sample_tags=np.array(sample_tags),
                        curr_keys=np.array(curr_keys),
                        prev_keys=np.array(prev_keys),
                        d_yaws=d_yaws,
                        sample_fx=sample_fx,
                        ego_5d=ego_5d,
                        traj=traj,
                        target=target,
                        mask=mask,
                        obj_target=obj_target,
                        prev_dtick=int(z["prev_dtick"]) if "prev_dtick" in z.files else 20)
    os.replace(out_tmp, args.out)
    print("  done.")


if __name__ == "__main__":
    main()
