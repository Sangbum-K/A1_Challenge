"""a_2 single-frame + obj·dist-only 학습 — i_1 cache + aux_targets (raw image NFS 회피).

2026-05-26 plan: prev cross-attn(fusion)·yaw_shift·ego_mot 제거 → single-frame.
  속도 4채널(vL_abs, vY_abs, vL_rel, vY_rel) 제거 → head [obj, dist] (+옵션 prox).
  downstream 제어(distbrake)가 dist+obj 만 쓰므로 capacity 를 obj/dist 에 집중.

forward (cache path):
  i_1 cache (B, 40, C_b=256, raw backbone token) →
    a2.encoder.proj (frozen, v7b init)  →  (B, 40, 256)
    a2.sa  (lateral self-attn × n_sa, LEARNED) →  (B, 40, 256)
    a2.fc_proj (256→128, LEARNED) = i2_mem →  (B, 40, 128)
    a2.head    (128 → n_ch, LEARNED)   →  (B, 40, n_ch)  [obj, dist, (prox)]

학습:
  encoder(backbone+proj) frozen → forward only.
  sa + fc_proj + head 학습 (구조 변경이라 from scratch).
  loss = obj focal(α≈0.75) + masked L1(dist; v7b 와 동일 scale-dist=40).

사용:
  python scripts/train_a2_single_objdist.py \
      --i1-cache cache/i1_*.npz [--i1-cache more.npz ...] \
      --aux-dir aux_targets/*  [--aux-dir more/* ...] \
      --resume-from data/ckpts/a2_v7b_d128_alpha075_best_loss.pt \
      --d-out 128 --n-ch 2 --epochs 10 --batch-size 256 --lr 3e-4 --use-focal --focal-alpha 0.75
"""
from __future__ import annotations
import os, sys, json, time, argparse, glob, re
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.model.image_models.a2_model import A2ModelSingleObjDist

# aux_targets 7ch 순서 = [obj, vL_abs, vY_abs, vL_rel, vY_rel, prox, dist]
AUX_OBJ_IDX, AUX_PROX_IDX, AUX_DIST_IDX = 0, 5, 6


def _lds_weight_dist(dist_norm, obj_mask, nbins=20):
    """batch-level inverse-density weight (dist 만; single-frame obj·dist).
    dist_norm (B,40) normalized, obj_mask (B,40) bool. tail(근거리) 강조. mean≈1.
    """
    w = torch.ones(obj_mask.shape, device=dist_norm.device, dtype=torch.float32)
    m = obj_mask
    if m.sum() < 10:
        return w
    vals = dist_norm[m]
    lo = vals.min(); hi = vals.max()
    if (hi - lo) < 1e-6:
        return w
    bins = ((vals - lo) / (hi - lo) * (nbins - 1)).long().clamp(0, nbins - 1)
    cnt = torch.bincount(bins, minlength=nbins).float()
    dens = cnt[bins] / cnt.sum().clamp(min=1.0)
    cw = 1.0 / (dens + 1e-3)
    cw = (cw / cw.mean().clamp(min=1e-6)).clamp(max=10.0)
    tmp = torch.ones_like(w)
    tmp[m] = cw
    return tmp


# ============================================================================
class A2SingleCacheDataset(Dataset):
    """i_1 cache + aux_targets 매칭. single-frame → tok_curr + target([obj, dist(, prox)])."""
    def __init__(self, i1_cache_path, aux_dir, run_filter: str = None, n_ch: int = 2):
        z = np.load(i1_cache_path, allow_pickle=True)
        self.tokens = z["tokens"]   # (N, 40, C_b)
        self.keys = z["keys"].tolist()
        self.k2i = {k: i for i, k in enumerate(self.keys)}
        self.n_ch = int(n_ch)
        aux_files = sorted(glob.glob(str(Path(aux_dir) / "*.npz")))
        pat = re.compile(r".+?__([^_]+(?:_[^_]+)*?)__t(\d+)\.npz$")
        self.samples = []  # (cache_idx, aux_path)
        n_aux = 0
        for f in aux_files:
            fname = Path(f).name
            m = pat.match(fname)
            if not m: continue
            n_aux += 1
            run, tick = m.group(1), int(m.group(2))
            # run_filter(policy) 는 파일명 전체에 적용 — group(1) 은 run-token 만
            if run_filter and run_filter not in fname: continue
            key = f"{run}__t{tick:07d}"
            if key not in self.k2i: continue
            self.samples.append((self.k2i[key], f))
        print(f"[a2-single-ds] {Path(i1_cache_path).name}  →  matched {len(self.samples):,} / {len(aux_files):,} aux (cache N={len(self.keys):,})", flush=True)
        if n_aux > 0 and len(self.samples) < 0.5 * n_aux:
            raise RuntimeError(
                f"[a2-single-ds][FAIL-FAST] join {len(self.samples):,}/{n_aux:,} < 50% — "
                f"key/run-filter('{run_filter}') 오매칭 의심. cache={Path(i1_cache_path).name} aux={Path(aux_dir).name}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        idx, ap = self.samples[i]
        tok_curr = torch.from_numpy(self.tokens[idx].astype(np.float32))   # (40, C_b)
        t = np.load(ap)["target"].astype(np.float32)   # (40, 7)
        # [obj, dist] 또는 [obj, dist, prox]
        cols = [AUX_OBJ_IDX, AUX_DIST_IDX] + ([AUX_PROX_IDX] if self.n_ch >= 3 else [])
        target = torch.from_numpy(t[:, cols])           # (40, n_ch)
        return {"tok_curr": tok_curr, "target": target}


def collate(items):
    return {k: torch.stack([it[k] for it in items], dim=0) for k in items[0]}


# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i1-cache", action="append", required=True)
    ap.add_argument("--aux-dir",  action="append", required=True)
    ap.add_argument("--run-filter", action="append", default=[])
    ap.add_argument("--resume-from", default=None, help="v7b ckpt (encoder.proj init; sa/head 는 새로 학습)")
    ap.add_argument("--seg-pretrain", default=None, help="encoder backbone seg_pretrain ckpt (resume-from 없을 때)")
    ap.add_argument("--d-out", type=int, default=128)
    ap.add_argument("--n-ch", type=int, default=2, help="2=[obj,dist], 3=[obj,dist,prox]")
    ap.add_argument("--n-sa", type=int, default=2, help="lateral self-attn 층 수")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--w-obj", type=float, default=1.0)
    ap.add_argument("--w-reg", type=float, default=1.0)
    ap.add_argument("--use-focal", action="store_true")
    ap.add_argument("--focal-alpha", type=float, default=0.75)
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--scale-dist", type=float, default=40.0)
    ap.add_argument("--lds", action="store_true", help="dist inverse-density weight (tail 강조)")
    ap.add_argument("--lds-nbins", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="outputs_multi")
    ap.add_argument("--name", default="a2_single_objdist")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out_dir = Path(args.out_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / "train.log", "w")
    def logp(m): print(m, flush=True); log_f.write(m + "\n"); log_f.flush()
    logp(f"args={vars(args)}")

    if len(args.i1_cache) != len(args.aux_dir):
        raise RuntimeError("--i1-cache 와 --aux-dir 갯수 동일 필요")
    run_filters = args.run_filter + [None] * (len(args.i1_cache) - len(args.run_filter))
    ds_list = [A2SingleCacheDataset(i1p, apd, run_filter=rf, n_ch=args.n_ch)
               for i1p, apd, rf in zip(args.i1_cache, args.aux_dir, run_filters)]
    full = ConcatDataset(ds_list)
    N = len(full)
    n_val = int(N * args.val_ratio); n_tr = N - n_val
    tr_ds, va_ds = random_split(full, [n_tr, n_val], generator=torch.Generator().manual_seed(args.seed))
    logp(f"[data] total={N:,}  train={n_tr:,}  val={n_val:,}")
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                       collate_fn=collate, pin_memory=True, persistent_workers=args.workers > 0)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                       collate_fn=collate, pin_memory=True, persistent_workers=args.workers > 0)

    # === model ===
    model = A2ModelSingleObjDist(d_token=256, d_out=args.d_out, n_ch=args.n_ch, n_sa=args.n_sa).to(args.device)
    # encoder.proj init: v7b ckpt 에서 (불가하면 새로). backbone 은 timm pretrained.
    if args.resume_from:
        sd = torch.load(args.resume_from, map_location=args.device, weights_only=False)
        if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
        own = model.state_dict()
        # encoder.* 만 가져오기 (fusion/fc_proj/head 는 구조 다름)
        load = {k: v for k, v in sd.items() if k.startswith("encoder.") and k in own and v.shape == own[k].shape}
        own.update(load)
        model.load_state_dict(own)
        logp(f"[resume] {args.resume_from}  encoder keys loaded={len(load)}")
    elif args.seg_pretrain:
        r = model.load_seg_pretrain(args.seg_pretrain)
        logp(f"[seg-pretrain] {args.seg_pretrain}  {r}")
    model.freeze_encoder()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logp(f"[model] single-frame obj·dist  n_ch={args.n_ch}  n_sa={args.n_sa}  trainable={n_train/1e3:.1f}K (sa+fc_proj+head)")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    bce_fn = nn.BCEWithLogitsLoss()
    def focal_loss(logits, targets, alpha=args.focal_alpha, gamma=args.focal_gamma):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** gamma * bce).mean()
    obj_loss_fn = focal_loss if args.use_focal else bce_fn
    logp(f"[loss] obj = {'Focal(α=%.2f,γ=%.1f)' % (args.focal_alpha, args.focal_gamma) if args.use_focal else 'BCE'}; reg = L1(dist){' +LDS' if args.lds else ''}")

    def dist_reg_loss(logits, target, obj_mask):
        """masked L1 on dist (normalized by scale_dist). logits/target ch1 = dist."""
        if not obj_mask.any():
            return torch.zeros((), device=logits.device)
        pred_d = logits[..., 1]                              # (B,40)
        tgt_d = target[..., 1] / args.scale_dist             # normalized
        err = (pred_d - tgt_d).abs()
        if args.lds:
            w = _lds_weight_dist(tgt_d.detach(), obj_mask, args.lds_nbins)
            return (err[obj_mask] * w[obj_mask]).sum() / w[obj_mask].sum().clamp(min=1e-6)
        return err[obj_mask].mean()

    best_val = float("inf")
    for ep in range(1, args.epochs + 1):
        model.train(); model.encoder.eval()   # frozen encoder BN 고정
        t0 = time.time()
        for bi, b in enumerate(tr_dl):
            tok_c = b["tok_curr"].to(args.device, non_blocking=True)
            target = b["target"].to(args.device, non_blocking=True)
            with torch.no_grad():
                # cache token (B,40,C_b) → encoder.proj (frozen) → (B,40,256)
                tc = model.encoder.proj(tok_c.transpose(1, 2)).transpose(1, 2)
            logits = model.forward_tokens(tc)                # (B,40,n_ch)
            obj_logit = logits[..., 0]; obj_gt = target[..., 0]
            loss_obj = obj_loss_fn(obj_logit, obj_gt)
            obj_mask = obj_gt > 0.5
            loss_reg = dist_reg_loss(logits, target, obj_mask)
            loss = args.w_obj * loss_obj + args.w_reg * loss_reg
            optim.zero_grad(set_to_none=True)
            loss.backward(); optim.step()
            if (bi + 1) % args.log_every == 0:
                logp(f"[ep {ep:02d} | {bi+1:>5,}/{len(tr_dl):,}] total={loss:.4f}  obj={loss_obj:.4f}  reg={loss_reg:.4f}  lr={optim.param_groups[0]['lr']:.2e}")
        sched.step()
        # val
        model.eval()
        with torch.no_grad():
            v_total, v_obj, v_reg, n = 0., 0., 0., 0
            for b in va_dl:
                tok_c = b["tok_curr"].to(args.device, non_blocking=True)
                target = b["target"].to(args.device, non_blocking=True)
                tc = model.encoder.proj(tok_c.transpose(1, 2)).transpose(1, 2)
                logits = model.forward_tokens(tc)
                obj_logit = logits[..., 0]; obj_gt = target[..., 0]
                l_obj = obj_loss_fn(obj_logit, obj_gt)
                obj_mask = obj_gt > 0.5
                # val reg = plain MAE (normalized) — 버전 간 비교용
                if obj_mask.any():
                    pred_d = logits[..., 1]; tgt_d = target[..., 1] / args.scale_dist
                    l_reg = (pred_d - tgt_d).abs()[obj_mask].mean()
                else:
                    l_reg = torch.zeros((), device=logits.device)
                bsz = tok_c.size(0)
                v_total += (args.w_obj * l_obj + args.w_reg * l_reg).item() * bsz
                v_obj += l_obj.item() * bsz; v_reg += l_reg.item() * bsz; n += bsz
        v_total /= n; v_obj /= n; v_reg /= n
        dt = time.time() - t0
        logp(f"[ep {ep:02d}] val total={v_total:.4f}  obj={v_obj:.4f}  reg={v_reg:.4f}  time={dt:.1f}s")
        ckpt = {"model": model.state_dict(), "args": vars(args), "epoch": ep, "val": v_total}
        if v_total < best_val:
            best_val = v_total
            torch.save(ckpt, out_dir / "best.pt")
            logp(f"  -> saved best ({v_total:.4f})")
        torch.save(ckpt, out_dir / "last.pt")
    log_f.close()


if __name__ == "__main__":
    main()
