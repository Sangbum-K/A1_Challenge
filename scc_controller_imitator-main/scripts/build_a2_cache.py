"""a_2 v4 출력 (40, 7) per-frame cache 빌더.

i_1 token cache 의 frame 별 tokens 을 입력으로 a_2 v4 의 fusion+head 만 실행 → (40, 7) cache 저장.
가정: ego_motion=[0,0] (per-frame 단순화. vL_rel 약간 부정확하나 dist/presence 는 정확).
prev_dtick: i_1 cache meta 의 [5] 사용 — i_1 cache 에 prev frame 도 같이 들어있음.

출력 cache 형식:
  keys:   (N_unique,) <U25       frame ID
  a2_out: (N_unique, 40, 7) f16  [obj, vL_abs, vY_abs, vL_rel, vY_rel, prox, dist]
  meta:   json string             {a2_ckpt, prev_dtick, ...}

사용:
  build_a2_cache.py --i1-cache <path> --out <path> [--a2-ckpt <path>] [--prev-key-map <path>]
  (단일 cache 처리. 6개 dataset = 6번 호출)
"""
import os, sys, json, time, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.model.image_models.a2_model import A2Model
from src.model.image_models.iy_model import yaw_shift_tokens_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i1-cache",  required=True, help="i_1 token cache npz path")
    ap.add_argument("--prebake",   required=True, help="prebake npz path (frame→prev mapping 추출용)")
    ap.add_argument("--out",       required=True, help="a_2 cache 저장 경로")
    ap.add_argument("--a2-ckpt",   default="/home/ailab-12/git/scc_controller_imitator/data/ckpts/a2_fx640_v4_best_loss.pt")
    ap.add_argument("--batch",     type=int, default=256)
    ap.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # 1. load i_1 cache (per-frame tokens)
    print(f"[load] i_1 cache: {args.i1_cache}")
    z1 = np.load(args.i1_cache, allow_pickle=True)
    tok_keys = z1["keys"].tolist()
    tok_all = z1["tokens"]   # (N_frame, 40, C_b)
    meta_i1 = json.loads(str(z1["meta"]))
    C_b = tok_all.shape[-1]
    print(f"  frames={len(tok_keys)}  shape={tok_all.shape}  C_b={C_b}  meta={meta_i1}")
    k2idx = {k: i for i, k in enumerate(tok_keys)}

    # 2. load prebake (frame→prev mapping)
    print(f"[load] prebake: {args.prebake}")
    z2 = np.load(args.prebake, allow_pickle=True)
    curr_keys = z2["curr_keys"]
    prev_keys = z2["prev_keys"]
    d_yaws = z2["d_yaws"]
    sample_fx = z2["sample_fx"] if "sample_fx" in z2.files else np.full(len(curr_keys), 320.0, dtype=np.float32)
    # per-frame (curr_key) 별 prev_key, d_yaw 추출 — 첫 occurrence 사용 (logged 기준)
    seen = set(); frame_meta = {}
    for i in range(len(curr_keys)):
        ck = curr_keys[i]
        if ck in seen: continue
        seen.add(ck)
        frame_meta[ck] = (prev_keys[i], float(d_yaws[i]), float(sample_fx[i]))
    print(f"  unique frames in prebake: {len(frame_meta)}")

    # 3. a_2 v4 model — fusion + head 만 사용 (encoder 우회)
    print(f"[load] a_2 v4: {args.a2_ckpt}")
    a2 = A2Model(d_token=C_b, n_ch=7).to(args.device).eval()
    sd = torch.load(args.a2_ckpt, map_location=args.device, weights_only=False)
    if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
    miss, unexp = a2.load_state_dict(sd, strict=False)
    print(f"  loaded: missing={len(miss)} unexpected={len(unexp)}  (encoder 부분 일부 missing 정상)")

    # 4. frames 중 cache + prebake 모두 hit 하는 것만 처리
    frames_to_process = [k for k in tok_keys if k in frame_meta]
    print(f"[match] frames in both cache & prebake: {len(frames_to_process)} / {len(tok_keys)}")

    a2_out = np.zeros((len(tok_keys), 40, 7), dtype=np.float16)
    # batch loop
    t0 = time.time()
    with torch.no_grad():
        for bi in range(0, len(frames_to_process), args.batch):
            batch_keys = frames_to_process[bi:bi+args.batch]
            # gather curr tokens
            ci = [k2idx[k] for k in batch_keys]
            tc = torch.from_numpy(tok_all[ci].astype(np.float32)).to(args.device)  # (B, 40, C_b)
            # gather prev tokens + d_yaw + fx
            prev_idx = []; dyaws = []; fxs = []
            for k in batch_keys:
                pk, dy, fx = frame_meta[k]
                pi = k2idx.get(pk, k2idx[k])  # prev 없으면 self (rare edge)
                prev_idx.append(pi)
                dyaws.append(dy); fxs.append(fx)
            tp = torch.from_numpy(tok_all[prev_idx].astype(np.float32)).to(args.device)
            dy_t = torch.tensor(dyaws, dtype=torch.float32, device=args.device)
            fx_t = torch.tensor(fxs, dtype=torch.float32, device=args.device)
            # yaw shift prev
            tp_rot = yaw_shift_tokens_batch(tp, dy_t, fx=fx_t)
            # ego_motion = 0 (simplification — frame 별 ego 변동 무시)
            ego_mot = torch.zeros(tc.shape[0], 2, device=args.device)
            # i_1 cache 는 backbone+pool 후, proj 전 token 임 (raw C_b)
            # → a_2 encoder.proj (Conv1d C_b → d_token) 통과 시켜야 학습 시 input 분포와 일치
            #   (이 step 누락 시 fusion 입력 분포 mismatch → output near-random ROC AUC 0.51, dist MAE 575m)
            tc_proj = a2.encoder.proj(tc.transpose(1, 2)).transpose(1, 2)      # (B, 40, d_token)
            tp_proj = a2.encoder.proj(tp_rot.transpose(1, 2)).transpose(1, 2)
            # a_2 forward (encoder backbone skip — proj+fusion+head)
            fi_mem = a2.fusion(tc_proj, tp_proj, ego_mot)  # (B, 40, d_token)
            out = a2.head(fi_mem)                            # (B, 40, 7)
            out_np = out.cpu().numpy().astype(np.float16)
            # store at original tok_keys index
            for j, k in enumerate(batch_keys):
                a2_out[k2idx[k]] = out_np[j]
            if (bi // args.batch + 1) % 20 == 0:
                pct = 100 * (bi + args.batch) / len(frames_to_process)
                el = time.time() - t0
                print(f"  {bi+args.batch}/{len(frames_to_process)}  ({pct:.1f}%, {el:.1f}s)")

    # 5. save
    meta_a2 = {
        "a2_ckpt": args.a2_ckpt,
        "i1_cache_source": args.i1_cache,
        "channels": ["obj", "vL_abs", "vY_abs", "vL_rel", "vY_rel", "prox", "dist"],
        "ego_motion_used": "zero_simplification",
        "n_frames": len(tok_keys),
        "n_processed": len(frames_to_process),
    }
    print(f"[save] {args.out}")
    np.savez_compressed(args.out, keys=np.array(tok_keys), a2_out=a2_out, meta=json.dumps(meta_a2))
    print(f"  done in {time.time()-t0:.1f}s")
    print(f"  channel-wise mean (over processed frames):")
    for c, ch in enumerate(meta_a2["channels"]):
        vals = a2_out[:len(frames_to_process), :, c].astype(np.float32)
        print(f"    {ch:10s}: mean={vals.mean():.3f}  std={vals.std():.3f}  min={vals.min():.2f}  max={vals.max():.2f}")


if __name__ == "__main__":
    main()
