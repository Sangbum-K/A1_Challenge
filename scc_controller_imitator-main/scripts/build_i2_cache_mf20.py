"""i_2 cache 빌더 (mf20 = A2ModelMultiFrame v2s 20-frame).

기존 build_i2_cache.py 는 2-frame A2Model(fusion(tc,tp,ego)) 로 i2_mem 생성.
이 스크립트는 20-frame A2ModelMultiFrame(v2s_n20, n_frames=21) 의
forward_tokens(tok_list, ego, return_mem=True) 로 i2_mem 생성.

입력  = i_1 prev20 cache (per-frame 단일 token (N,40,256), key=<run>__t<7d>).
        prev frame 은 cache 에서 key offset(tick - off)로 직접 조회 (train_a2_multiframe.py
        와 동일 규약 — 없으면 직전 offset idx fallback, 최종 self).
offset = 5,10,...,100 (20 prev) + frame0=curr → n_frames=21.
yaw_shift=off (mf20 학습이 off), ego_motion=0 (학습과 동일).

출력  = i2_mem (N,40,128) [= fc_proj 출력] + a2_logits (N,40,7) [= 20f obj logit, head 출력].

사용:
  python scripts/build_i2_cache_mf20.py \
      --i1-cache cache/i1_tok_cache_<ds>_prev20.npz \
      --a2-ckpt outputs_multi/a2_b0_abl/mf20f_v2s_5to100/best_loss.pt \
      --out cache/i2_mem_mf20_<ds>.npz
"""
import os, sys, json, time, argparse
from pathlib import Path
import numpy as np
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.model.image_models.a2_model import A2ModelMultiFrame

PREV_OFFSETS_DEFAULT = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i1-cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--a2-ckpt", required=True)
    ap.add_argument("--d-token", type=int, default=256)
    ap.add_argument("--d-out", type=int, default=128)
    ap.add_argument("--prev-offsets", type=str,
                    default=",".join(str(o) for o in PREV_OFFSETS_DEFAULT),
                    help="comma-sep prev tick offsets. n_frames = 1+len(offsets).")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    prev_offsets = [int(x) for x in args.prev_offsets.split(",") if x.strip()]
    n_frames = 1 + len(prev_offsets)
    print(f"[cfg] n_frames={n_frames}  prev_offsets={prev_offsets}")

    print(f"[load] i_1 prev20 cache: {args.i1_cache}")
    z1 = np.load(args.i1_cache, allow_pickle=True)
    tok_keys = z1["keys"].tolist()
    tok_all = z1["tokens"]                       # (N,40,C_b)
    C_b = tok_all.shape[-1]
    print(f"  frames={len(tok_keys)}  shape={tok_all.shape}  C_b={C_b}")
    k2idx = {k: i for i, k in enumerate(tok_keys)}

    # prev idx 미리 계산 (train_a2_multiframe.py 규약: 없으면 직전 offset idx, 최종 self)
    def _prev_idxs(ck):
        try:
            base, tk = ck.rsplit("__t", 1); tick = int(tk)
        except Exception:
            return [k2idx[ck]] * len(prev_offsets)
        pidxs = []; last = k2idx[ck]
        for off in prev_offsets:
            pk = f"{base}__t{tick - off:07d}"
            pi = k2idx.get(pk, last)
            pidxs.append(pi); last = pi
        return pidxs

    print(f"[load] a_2 mf20 (A2ModelMultiFrame n_frames={n_frames}): {args.a2_ckpt}")
    a2 = A2ModelMultiFrame(d_token=args.d_token, d_out=args.d_out, n_frames=n_frames).to(args.device).eval()
    # cache C_b 가 d_token 과 다르면 encoder.proj 재생성 (train_a2_multiframe.py 와 동일)
    if C_b != a2.encoder.proj.in_channels:
        import torch.nn as nn
        a2.encoder.proj = nn.Conv1d(C_b, args.d_token, kernel_size=1).to(args.device)
        print(f"  [proj] rebuilt encoder.proj: in={C_b} out={args.d_token}")
    sd = torch.load(args.a2_ckpt, map_location=args.device, weights_only=False)
    if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
    own = a2.state_dict()
    skip = [k for k in list(sd.keys()) if k in own and sd[k].shape != own[k].shape]
    for k in skip: del sd[k]
    miss, unexp = a2.load_state_dict(sd, strict=False)
    print(f"  loaded missing={len(miss)} unexpected={len(unexp)} shape-skip={len(skip)}")
    if len(miss) > 0:
        print(f"  [warn] missing keys (확인): {miss[:10]}")

    N = len(tok_keys)
    i2_mem = np.zeros((N, 40, args.d_out), dtype=np.float16)
    a2_logits = np.zeros((N, 40, 7), dtype=np.float16)
    t0 = time.time()
    with torch.no_grad():
        for bi in range(0, N, args.batch):
            batch_keys = tok_keys[bi:bi+args.batch]
            ci = [k2idx[k] for k in batch_keys]
            # tok_list: n_frames 개 (B,40,C_b). frame0=curr, 1..= prev offset.
            tok_list = [torch.from_numpy(tok_all[ci].astype(np.float32)).to(args.device)]
            for fi in range(len(prev_offsets)):
                pidx = [_prev_idxs(k)[fi] for k in batch_keys]
                tok_list.append(torch.from_numpy(tok_all[pidx].astype(np.float32)).to(args.device))
            ego_mot = torch.zeros(len(batch_keys), 2, device=args.device)
            logits, mem = a2.forward_tokens(tok_list, ego_mot, return_mem=True)  # (B,40,7),(B,40,128)
            mem_np = mem.cpu().numpy().astype(np.float16)
            log_np = logits.cpu().numpy().astype(np.float16)
            for j, k in enumerate(batch_keys):
                i2_mem[k2idx[k]] = mem_np[j]
                a2_logits[k2idx[k]] = log_np[j]
            if (bi // args.batch + 1) % 20 == 0:
                pct = 100 * (bi + args.batch) / N
                print(f"  {bi+args.batch}/{N}  ({pct:.1f}%, {time.time()-t0:.1f}s)")

    meta = {
        "a2_ckpt": args.a2_ckpt,
        "i1_cache_source": args.i1_cache,
        "model": "A2ModelMultiFrame",
        "n_frames": n_frames,
        "prev_offsets": prev_offsets,
        "d_out": args.d_out,
        "n_frames_cache": N,
        "yaw_shift": "off",
        "ego_motion": 0,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"[save] {args.out}")
    np.savez_compressed(args.out,
                        keys=np.array(tok_keys),
                        i2_mem=i2_mem,           # (N,40,128) — image_mem (20f)
                        a2_logits=a2_logits,     # (N,40,7) — 20f obj logit (gate 입력)
                        meta=json.dumps(meta))
    print(f"  done in {time.time()-t0:.1f}s")
    vals = i2_mem.astype(np.float32)
    print(f"  i2_mem: mean={vals.mean():.3f} std={vals.std():.3f} range [{vals.min():.2f},{vals.max():.2f}]")
    obj = a2_logits[:, 16:26, 0].astype(np.float32)   # in-lane obj logit (bin16-25)
    print(f"  in-lane(16-25) obj logit: mean={obj.mean():.3f} std={obj.std():.3f} "
          f"range [{obj.min():.2f},{obj.max():.2f}]  frac(>0)={float((obj>0).mean()):.3f}")


if __name__ == "__main__":
    main()
