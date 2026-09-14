"""i_2 cache 빌더 — a_2 v5 의 fc_proj 출력 (image_mem) + AuxHead 출력 (a_2 logits) 동시 저장.

i_1 cache 의 frame token 을 a_2 v5 encoder.proj → fusion → fc_proj 까지 inference
→ image_mem (B, 40, d_out=128) per-frame 저장. iy_model v6 의 main+side 입력.
+ 동시에 AuxHead 까지 통과해 a_2 logits 저장 → aux_targets GT 와 비교 시 사용 (검증).

사용:
  python scripts/build_i2_cache.py \
      --i1-cache cache/i1_*.npz \
      --prebake prebake/f1_*.npz \
      --out cache/i2_mem_cache_*.npz \
      --a2-ckpt data/ckpts/a2_v5_d128_best_loss.pt
"""
import os, sys, json, time, argparse
from pathlib import Path
import numpy as np
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.model.image_models.a2_model import A2Model
from src.model.image_models.iy_model import yaw_shift_tokens_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i1-cache", required=True)
    ap.add_argument("--prebake", default=None,
                    help="optional. per-frame 실제 d_yaw/prev_key LUT (없으면 d_yaw=0, prev=tick-20).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--a2-ckpt", required=True)
    ap.add_argument("--d-out", type=int, default=128)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--fx", type=float, default=640.0,
                    help="prebake 미커버 frame 의 fx (학습 데이터 fx=640).")
    ap.add_argument("--prev-dtick", type=int, default=20,
                    help="prebake 미커버 frame 의 prev key = curr tick - prev_dtick.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"[load] i_1 cache: {args.i1_cache}")
    z1 = np.load(args.i1_cache, allow_pickle=True)
    tok_keys = z1["keys"].tolist()
    tok_all = z1["tokens"]
    meta_i1 = json.loads(str(z1["meta"]))
    C_b = tok_all.shape[-1]
    print(f"  frames={len(tok_keys)}  shape={tok_all.shape}  C_b={C_b}")
    k2idx = {k: i for i, k in enumerate(tok_keys)}

    frame_meta = {}   # ck -> (prev_key, d_yaw, fx). prebake 커버 frame 만.
    if args.prebake:
        print(f"[load] prebake (optional d_yaw/prev LUT): {args.prebake}")
        z2 = np.load(args.prebake, allow_pickle=True)
        curr_keys = z2["curr_keys"]; prev_keys = z2["prev_keys"]
        d_yaws = z2["d_yaws"]
        sample_fx = z2["sample_fx"] if "sample_fx" in z2.files else np.full(len(curr_keys), args.fx, dtype=np.float32)
        seen = set()
        for i in range(len(curr_keys)):
            ck = str(curr_keys[i])
            if ck in seen: continue
            seen.add(ck)
            frame_meta[ck] = (str(prev_keys[i]), float(d_yaws[i]), float(sample_fx[i]))
        print(f"  unique frames in prebake: {len(frame_meta)}")
    else:
        print(f"[no prebake] d_yaw=0, prev=tick-{args.prev_dtick}, fx={args.fx} for ALL frames")

    print(f"[load] a_2 v5: {args.a2_ckpt}")
    a2 = A2Model(d_token=256, d_out=args.d_out).to(args.device).eval()
    sd = torch.load(args.a2_ckpt, map_location=args.device, weights_only=False)
    if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
    own = a2.state_dict()
    skip = [k for k in list(sd.keys()) if k in own and sd[k].shape != own[k].shape]
    for k in skip: del sd[k]
    miss, unexp = a2.load_state_dict(sd, strict=False)
    print(f"  loaded missing={len(miss)} unexpected={len(unexp)} shape-skip={len(skip)}")

    # 전 cache frame 처리 (이전 버그: prebake ∩ cache 만 처리 → sangam/spd 96% zero-fill).
    # prebake 커버 frame 은 실제 d_yaw/prev 사용, 미커버는 d_yaw=0 + prev=tick-dtick fallback.
    def _prev_from_tick(ck):
        # ck = "<run>__t<7d>". prev = "<run>__t<tick-dtick>".
        try:
            base, tk = ck.rsplit("__t", 1)
            return f"{base}__t{int(tk) - args.prev_dtick:07d}"
        except Exception:
            return ck
    frames_to_process = list(tok_keys)
    n_pb = sum(1 for k in tok_keys if k in frame_meta)
    print(f"[match] processing ALL {len(frames_to_process)} cache frames "
          f"(prebake-covered={n_pb}, fallback={len(frames_to_process)-n_pb})")

    i2_mem = np.zeros((len(tok_keys), 40, args.d_out), dtype=np.float16)
    a2_logits = np.zeros((len(tok_keys), 40, 7), dtype=np.float16)
    t0 = time.time()
    with torch.no_grad():
        for bi in range(0, len(frames_to_process), args.batch):
            batch_keys = frames_to_process[bi:bi+args.batch]
            ci = [k2idx[k] for k in batch_keys]
            tc = torch.from_numpy(tok_all[ci].astype(np.float32)).to(args.device)
            prev_idx = []; dyaws = []; fxs = []
            for k in batch_keys:
                if k in frame_meta:
                    pk, dy, fx = frame_meta[k]
                else:
                    pk, dy, fx = _prev_from_tick(k), 0.0, args.fx
                pi = k2idx.get(pk, k2idx[k])
                prev_idx.append(pi); dyaws.append(dy); fxs.append(fx)
            tp = torch.from_numpy(tok_all[prev_idx].astype(np.float32)).to(args.device)
            dy_t = torch.tensor(dyaws, dtype=torch.float32, device=args.device)
            fx_t = torch.tensor(fxs, dtype=torch.float32, device=args.device)
            tp_rot = yaw_shift_tokens_batch(tp, dy_t, fx=fx_t)
            ego_mot = torch.zeros(tc.shape[0], 2, device=args.device)
            # encoder.proj
            tc_proj = a2.encoder.proj(tc.transpose(1, 2)).transpose(1, 2)
            tp_proj = a2.encoder.proj(tp_rot.transpose(1, 2)).transpose(1, 2)
            # fusion (256)
            fi_mem_256 = a2.fusion(tc_proj, tp_proj, ego_mot)
            # fc_proj (256→128) = image_mem
            fi_mem_128 = a2.fc_proj(fi_mem_256)
            # head (128→7) = a_2 logits
            logits = a2.head(fi_mem_128)
            mem_np = fi_mem_128.cpu().numpy().astype(np.float16)
            log_np = logits.cpu().numpy().astype(np.float16)
            for j, k in enumerate(batch_keys):
                i2_mem[k2idx[k]] = mem_np[j]
                a2_logits[k2idx[k]] = log_np[j]
            if (bi // args.batch + 1) % 10 == 0:
                pct = 100 * (bi + args.batch) / len(frames_to_process)
                print(f"  {bi+args.batch}/{len(frames_to_process)}  ({pct:.1f}%, {time.time()-t0:.1f}s)")

    meta = {
        "a2_ckpt": args.a2_ckpt,
        "i1_cache_source": args.i1_cache,
        "d_out": args.d_out,
        "n_frames": len(tok_keys),
        "n_processed": len(frames_to_process),
    }
    print(f"[save] {args.out}")
    np.savez_compressed(args.out,
                        keys=np.array(tok_keys),
                        i2_mem=i2_mem,           # (N_frame, 40, 128) — image_mem
                        a2_logits=a2_logits,     # (N_frame, 40, 7) — 검증용
                        meta=json.dumps(meta))
    print(f"  done in {time.time()-t0:.1f}s")
    print(f"  channel-wise (image_mem, processed frames):")
    vals = i2_mem[:len(frames_to_process)].astype(np.float32)
    print(f"    mean={vals.mean():.3f}  std={vals.std():.3f}  range [{vals.min():.2f}, {vals.max():.2f}]")
    print(f"  obj logit (processed frames):")
    obj = a2_logits[:len(frames_to_process), :, 0].astype(np.float32)
    print(f"    mean={obj.mean():.3f}  std={obj.std():.3f}  range [{obj.min():.2f}, {obj.max():.2f}]")


if __name__ == "__main__":
    main()
