"""a_2 v5 학습 — i_1 cache + aux_targets 만 사용 (raw image NFS load 회피).

forward:
  i_1 cache (B, 40, C_b=256, raw backbone token) →
    a2.encoder.proj (frozen, v4)  →  (B, 40, 256)
    a2.fusion       (frozen, v4)  →  (B, 40, 256)
    a2.fc_proj      (LEARNED)     →  (B, 40, 128)
    a2.head         (LEARNED)     →  (B, 40, 7)

학습:
  encoder.proj + fusion 는 frozen → forward only (backward 안 흐름)
  fc_proj + head 만 학습. trainable ~33K params.

사용:
  python scripts/train_a2_v5_from_cache.py \
      --i1-cache cache/i1_*.npz [--i1-cache more.npz ...] \
      --aux-dir aux_targets/* [--aux-dir more/* ...] \
      --resume-from data/ckpts/a2_fx640_v4_best_loss.pt \
      --d-out 128 --epochs 10 --batch-size 256 --lr 3e-4
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
from src.model.image_models.a2_model import A2Model, mean_indices, sigma_indices
from src.model.image_models.iy_model import yaw_shift_tokens_batch


def _lds_weight(tgt_reg, obj_mask, nbins=20):
    """batch-level inverse-density weight (A1).
    tgt_reg (B,40,6) normalized, obj_mask (B,40) bool.
    vL_rel(reg idx2) + dist(reg idx5) 의 batch 히스토그램 → 1/density. tail 강조.
    returns w (B,40) — obj token 만 의미, mean≈1 정규화.
    """
    w = torch.ones(obj_mask.shape, device=tgt_reg.device, dtype=torch.float32)
    m = obj_mask
    if m.sum() < 10:
        return w
    for ch in (2, 5):   # vL_rel, dist
        vals = tgt_reg[..., ch][m]
        lo = vals.min(); hi = vals.max()
        if (hi - lo) < 1e-6:
            continue
        bins = ((vals - lo) / (hi - lo) * (nbins - 1)).long().clamp(0, nbins - 1)
        cnt = torch.bincount(bins, minlength=nbins).float()
        dens = cnt[bins] / cnt.sum().clamp(min=1.0)
        cw = 1.0 / (dens + 1e-3)
        cw = cw / cw.mean().clamp(min=1e-6)        # mean 1
        cw = cw.clamp(max=10.0)                     # 폭주 방지
        tmp = torch.ones_like(w)
        tmp[m] = cw
        w = w * tmp.pow(0.5)                         # 두 채널 기하평균
    return w


def _reg_loss(logits, target, scales, obj_mask, args):
    """v7 L1 또는 v8 Gaussian NLL + LDS. returns scalar."""
    if not obj_mask.any():
        return torch.zeros((), device=logits.device)
    pv = args.gaussian_nll
    midx = mean_indices(pv)                           # mean 채널 인덱스 (6)
    mean = logits[..., midx]                            # (B,40,6)
    tgt_reg = target[..., 1:7] / scales                 # (B,40,6) normalized
    if not pv:
        # v7/v9: L1. --lds 시 inverse-density weight (v9 = L1 + LDS, NLL 없음)
        if args.lds:
            w = _lds_weight(tgt_reg.detach(), obj_mask, args.lds_nbins)
            err = (mean - tgt_reg).abs().mean(-1)         # (B,40)
            m = obj_mask
            return (err[m] * w[m]).sum() / w[m].sum().clamp(min=1e-6)
        m = obj_mask.unsqueeze(-1).expand_as(mean)
        return F.l1_loss(mean[m], tgt_reg[m])
    # v8: Gaussian NLL
    sidx = sigma_indices()
    logvar = logits[..., sidx].clamp(-7.0, 7.0)         # (B,40,6) log σ²
    nll = 0.5 * (torch.exp(-logvar) * (mean - tgt_reg) ** 2 + logvar)   # (B,40,6)
    nll = nll.mean(-1)                                   # (B,40)
    if args.lds:
        w = _lds_weight(tgt_reg.detach(), obj_mask, args.lds_nbins)
    else:
        w = torch.ones_like(obj_mask, dtype=torch.float32)
    m = obj_mask
    return (nll[m] * w[m]).sum() / w[m].sum().clamp(min=1e-6)

# ============================================================================
class A2CacheDataset(Dataset):
    """i_1 cache + aux_targets 매칭. tok_curr + tok_prev_raw + ego_motion → target."""
    def __init__(self, i1_cache_path, aux_dir, run_filter: str = None, fx: float = 640.0):
        z = np.load(i1_cache_path, allow_pickle=True)
        self.tokens = z["tokens"]   # (N, 40, C_b)
        self.keys = z["keys"].tolist()
        self.k2i = {k: i for i, k in enumerate(self.keys)}
        self.fx = float(fx)
        # aux files (filter by run if provided)
        aux_files = sorted(glob.glob(str(Path(aux_dir) / "*.npz")))
        # extract (run, tick) from filename and match cache key
        pat = re.compile(r".+?__([^_]+(?:_[^_]+)*?)__t(\d+)\.npz$")
        self.samples = []  # list of (cache_idx, aux_path, ego_motion=0, d_yaw=0)
        n_aux = 0          # 패턴 매칭된 aux 전체
        n_filtered = 0     # run_filter 통과한 aux (== join 판정 분모)
        for f in aux_files:
            fname = Path(f).name
            m = pat.match(fname)
            if not m: continue
            n_aux += 1
            run, tick = m.group(1), int(m.group(2))
            # run_filter(policy) 는 파일명 전체에 적용 — group(1) 은 run-token 만이라 policy 누락
            if run_filter and run_filter not in fname: continue
            n_filtered += 1
            key = f"{run}__t{tick:07d}"
            if key not in self.k2i: continue
            self.samples.append((self.k2i[key], f, run, tick))
        print(f"[a2-cache-ds] {i1_cache_path}  →  matched {len(self.samples):,} / {n_filtered:,} (run-filter 통과) / {len(aux_files):,} aux files (cache N={len(self.keys):,})")
        # FAIL-FAST 는 run-filter 통과분 대비 cache-key 매칭률로 판정.
        # (run-filter 로 의도적으로 제외한 aux 는 분모에서 빠짐 — pertO/pertX 분리 시 false-positive 방지)
        if n_filtered > 0 and len(self.samples) < 0.5 * n_filtered:
            raise RuntimeError(
                f"[a2-cache-ds][FAIL-FAST] join {len(self.samples):,}/{n_filtered:,} < 50% — "
                f"key 불일치/run-filter('{run_filter}') 오매칭 의심. cache={Path(i1_cache_path).name} aux={Path(aux_dir).name}")
        # prev key 계산 (sample 별 cache index)
        # tick - 20 (PREV_DTICK) 의 key 가 cache 에 있으면 prev. 없으면 self.
        self.prev_idx = []
        for (idx, ap, run, tick) in self.samples:
            pk = f"{run}__t{tick - 20:07d}"
            self.prev_idx.append(self.k2i.get(pk, idx))

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        idx, ap, run, tick = self.samples[i]
        pidx = self.prev_idx[i]
        tok_curr = torch.from_numpy(self.tokens[idx].astype(np.float32))   # (40, C_b)
        tok_prev = torch.from_numpy(self.tokens[pidx].astype(np.float32))
        # ego_motion = 0 simplification (a_2 inference 시도 동일 가정)
        ego_motion = torch.zeros(2, dtype=torch.float32)
        # d_yaw = 0 (ego_motion 0 이면 prev rotation 도 0)
        d_yaw = torch.zeros(1, dtype=torch.float32)
        fx = torch.tensor(self.fx, dtype=torch.float32)
        # target
        t = np.load(ap)["target"].astype(np.float32)   # (40, 7)
        target = torch.from_numpy(t)
        return {
            "tok_curr": tok_curr,
            "tok_prev": tok_prev,
            "ego_motion": ego_motion,
            "d_yaw": d_yaw,
            "fx": fx,
            "target": target,
        }


def collate(items):
    return {k: torch.stack([it[k] for it in items], dim=0) for k in items[0]}


# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i1-cache", action="append", required=True, help="i_1 cache npz (반복 가능)")
    ap.add_argument("--aux-dir",  action="append", required=True, help="aux_targets dir (i1-cache 와 같은 순서)")
    ap.add_argument("--run-filter", action="append", default=[], help="cache 별 run filter (옵션, 같은 순서). empty 면 미적용")
    ap.add_argument("--fx", type=float, default=640.0)
    ap.add_argument("--resume-from", required=True, help="a_2 v4 ckpt path (encoder+fusion weights load)")
    ap.add_argument("--d-out", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--w-obj", type=float, default=1.0)
    ap.add_argument("--w-reg", type=float, default=1.0)
    ap.add_argument("--use-focal", action="store_true",
                    help="obj loss 를 focal loss 로 변경 (positive class 강조, calibration 균형)")
    ap.add_argument("--focal-alpha", type=float, default=0.25,
                    help="focal loss α (positive weight)")
    ap.add_argument("--focal-gamma", type=float, default=2.0,
                    help="focal loss γ (focusing parameter)")
    ap.add_argument("--scale-v", type=float, default=20.0)
    ap.add_argument("--scale-dist", type=float, default=40.0)
    ap.add_argument("--gaussian-nll", action="store_true",
                    help="v8: regression 을 Gaussian NLL (mean+logvar) 로. var_head 사용 — 이미지 차분 노이즈 흡수")
    ap.add_argument("--lds", action="store_true",
                    help="v8 A1: inverse-density weight (vL_rel+dist batch 히스토그램). tail(강접근/근거리) 강조 → mean collapse 완화")
    ap.add_argument("--lds-nbins", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="outputs_multi/a2")
    ap.add_argument("--name", default="v5_d128_cache")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out_dir = Path(args.out_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / "train.log", "w")
    def logp(m): print(m, flush=True); log_f.write(m + "\n"); log_f.flush()
    logp(f"args={vars(args)}")

    # === datasets ===
    if len(args.i1_cache) != len(args.aux_dir):
        raise RuntimeError("--i1-cache 와 --aux-dir 갯수 동일 필요")
    run_filters = args.run_filter + [None] * (len(args.i1_cache) - len(args.run_filter))
    ds_list = []
    for i1p, ap_dir, rf in zip(args.i1_cache, args.aux_dir, run_filters):
        ds_list.append(A2CacheDataset(i1p, ap_dir, run_filter=rf, fx=args.fx))
    full = ConcatDataset(ds_list)
    N = len(full)
    n_val = int(N * args.val_ratio)
    n_tr = N - n_val
    tr_ds, va_ds = random_split(full, [n_tr, n_val], generator=torch.Generator().manual_seed(args.seed))
    logp(f"[data] total={N:,}  train={n_tr:,}  val={n_val:,}")
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                       collate_fn=collate, pin_memory=True, persistent_workers=args.workers > 0)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                       collate_fn=collate, pin_memory=True, persistent_workers=args.workers > 0)

    # === model ===
    model = A2Model(d_token=256, d_out=args.d_out, predict_var=args.gaussian_nll).to(args.device)
    sd = torch.load(args.resume_from, map_location=args.device, weights_only=False)
    if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
    own = model.state_dict()
    skip = [k for k in list(sd.keys()) if k in own and sd[k].shape != own[k].shape]
    for k in skip: del sd[k]
    miss, unexp = model.load_state_dict(sd, strict=False)
    logp(f"[resume] {args.resume_from}  missing={len(miss)}  unexpected={len(unexp)}  shape-skip={len(skip)}")
    model.freeze_encoder_fusion()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logp(f"[model] trainable={n_train/1e3:.1f}K  (fc_proj + head 만)")

    # === optimizer (fc_proj + head 만) ===
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    # === train loop ===
    bce_fn = nn.BCEWithLogitsLoss()
    def focal_loss(logits, targets, alpha=args.focal_alpha, gamma=args.focal_gamma):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** gamma * bce).mean()
    obj_loss_fn = focal_loss if args.use_focal else bce_fn
    if args.use_focal:
        logp(f"[loss] obj = FocalLoss(α={args.focal_alpha}, γ={args.focal_gamma}); reg = L1")
    else:
        logp(f"[loss] obj = BCE; reg = L1")
    best_val = float("inf")
    for ep in range(1, args.epochs + 1):
        model.train()
        # encoder/fusion 는 frozen 이지만 eval 모드로 BN stats 안정화
        model.encoder.eval(); model.fusion.eval()
        t0 = time.time()
        for bi, b in enumerate(tr_dl):
            tok_c = b["tok_curr"].to(args.device, non_blocking=True)
            tok_p = b["tok_prev"].to(args.device, non_blocking=True)
            dy_t = b["d_yaw"].squeeze(-1).to(args.device, non_blocking=True)
            fx_t = b["fx"].to(args.device, non_blocking=True)
            ego = b["ego_motion"].to(args.device, non_blocking=True)
            target = b["target"].to(args.device, non_blocking=True)
            # forward (encoder/fusion frozen → no_grad block 으로 효율)
            with torch.no_grad():
                tok_p_rot = yaw_shift_tokens_batch(tok_p, dy_t, fx=fx_t)
                # encoder.proj (frozen): tok 의 (B, 40, C_b) → (B, 40, d_token=256)
                tc = model.encoder.proj(tok_c.transpose(1, 2)).transpose(1, 2)
                tp = model.encoder.proj(tok_p_rot.transpose(1, 2)).transpose(1, 2)
                fi_mem = model.fusion(tc, tp, ego)   # (B, 40, 256)
            # trainable: fc_proj + head
            fi_mem_out = model.fc_proj(fi_mem)        # (B, 40, 128)
            logits = model.head(fi_mem_out)            # (B, 40, 7 또는 13)
            # loss
            obj_logit = logits[..., 0]
            obj_gt = target[..., 0]
            loss_obj = obj_loss_fn(obj_logit, obj_gt)
            scales = torch.tensor([args.scale_v, args.scale_v, args.scale_v, args.scale_v,
                                    1.0, args.scale_dist], device=logits.device)
            obj_mask = obj_gt > 0.5
            loss_reg = _reg_loss(logits, target, scales, obj_mask, args)
            loss = args.w_obj * loss_obj + args.w_reg * loss_reg
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            if (bi + 1) % args.log_every == 0:
                logp(f"[ep {ep:02d} | {bi+1:>5,}/{len(tr_dl):,}] total={loss:.4f}  bce={loss_obj:.4f}  reg={loss_reg:.4f}  lr={optim.param_groups[0]['lr']:.2e}")
        sched.step()
        # val
        model.eval()
        with torch.no_grad():
            v_total, v_bce, v_reg, n = 0., 0., 0., 0
            for b in va_dl:
                tok_c = b["tok_curr"].to(args.device, non_blocking=True)
                tok_p = b["tok_prev"].to(args.device, non_blocking=True)
                dy_t = b["d_yaw"].squeeze(-1).to(args.device, non_blocking=True)
                fx_t = b["fx"].to(args.device, non_blocking=True)
                ego = b["ego_motion"].to(args.device, non_blocking=True)
                target = b["target"].to(args.device, non_blocking=True)
                tok_p_rot = yaw_shift_tokens_batch(tok_p, dy_t, fx=fx_t)
                tc = model.encoder.proj(tok_c.transpose(1, 2)).transpose(1, 2)
                tp = model.encoder.proj(tok_p_rot.transpose(1, 2)).transpose(1, 2)
                fi_mem = model.fusion(tc, tp, ego)
                fi_mem_out = model.fc_proj(fi_mem)
                logits = model.head(fi_mem_out)
                obj_logit = logits[..., 0]; obj_gt = target[..., 0]
                l_obj = obj_loss_fn(obj_logit, obj_gt)
                scales_v = torch.tensor([args.scale_v, args.scale_v, args.scale_v, args.scale_v,
                                          1.0, args.scale_dist], device=logits.device)
                obj_mask_v = obj_gt > 0.5
                # val 은 버전 간 비교 위해 항상 mean 채널의 L1 (MAE) 로 측정
                if obj_mask_v.any():
                    mean_v = logits[..., mean_indices(args.gaussian_nll)]
                    tgt_reg_v = target[..., 1:7] / scales_v
                    m_v = obj_mask_v.unsqueeze(-1).expand_as(mean_v)
                    l_reg = F.l1_loss(mean_v[m_v], tgt_reg_v[m_v])
                else:
                    l_reg = torch.zeros((), device=logits.device)
                bsz = tok_c.size(0)
                v_total += (args.w_obj * l_obj + args.w_reg * l_reg).item() * bsz
                v_bce += l_obj.item() * bsz; v_reg += l_reg.item() * bsz; n += bsz
        v_total /= n; v_bce /= n; v_reg /= n
        dt = time.time() - t0
        logp(f"[ep {ep:02d}] val total={v_total:.4f}  bce={v_bce:.4f}  reg={v_reg:.4f}  time={dt:.1f}s")
        if v_total < best_val:
            best_val = v_total
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": ep, "val": v_total},
                       out_dir / "best_loss.pt")
            logp(f"  -> saved best ({v_total:.4f})")
        torch.save({"model": model.state_dict(), "args": vars(args), "epoch": ep, "val": v_total},
                   out_dir / "last.pt")
    log_f.close()


if __name__ == "__main__":
    main()
