"""a_1(seg backbone) + a_2 end-to-end fine-tune (raw image → obj/aux GT).

2026-05-26 plan:
  기존 train_a2_multiframe.py 는 i_1 token cache 위에서 backbone frozen + fusion/head 만 학습.
  이 스크립트는 raw 이미지에서 시작해 backbone(encoder) 까지 같이 fine-tune 한다.
    - A2ModelMultiFrame(d_token=256, d_out=128, n_frames=N) 의 raw path forward(image_list, ego).
    - load_seg_pretrain 으로 backbone init, freeze_encoder() 는 호출하지 않음 → 전체 학습.
    - backbone(encoder) lr = head lr × backbone_lr_mult (기본 0.1) → seg pretrain 보존하며 fine-tune.
    - loss = focal(obj) + L1(reg). train_a2_multiframe 와 동일 (SCALE_V=20, SCALE_DIST=40).
    - AMP autocast + GradScaler (backbone forward 메모리).

데이터: aux dir glob → 파일명 {track}__{policy}__{run}__t{tick:07d}.npz.
  raw 이미지 = {raw_base}/{track}/{policy}/{run}/CAMERA_1/{tick}_Intensity.jpeg.
  prev tick = tick - offset. 없으면 가장 가까운 가용(직전 offset)으로 clamp, 최종 curr.

사용 예 (slurm/train_a2_e2e.sbatch 참조):
  python scripts/train_a2_e2e.py \
    --aux-dir data/aux_targets_sangam_lead_0520 --aux-dir ... \
    --raw-base /workspace/raw --seg-ckpt outputs_multi/seg_pretrain_fx640_v3/best_miou_backbone.pt \
    --n-frames 2 --prev-offsets 20 --epochs 8 --batch-size 16 --lr 1e-4 --backbone-lr-mult 0.1 \
    --workers 8 --out-dir outputs_multi --name a2_e2e_n2
"""
from __future__ import annotations
import os, sys, time, argparse, glob
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.model.image_models.a2_model import A2ModelMultiFrame, mean_indices, sigma_indices

# ROI / normalize — compute_i1_tok_cache.py 와 동일 ----------------------
ROI_Y0, ROI_Y1 = 235, 485
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# prev offset 기본 (frame0=curr 제외)
PREV_OFFSETS = {1: [], 2: [20], 3: [20, 40], 4: [20, 40, 60]}


def _load_roi_tensor(path: str) -> torch.Tensor:
    img = np.array(Image.open(path).convert("RGB"))
    roi = img[ROI_Y0:ROI_Y1]
    arr = roi.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())   # (3, 250, 1280)


def _parse_aux_name(fname: str):
    """'{track}__{policy}__{run}__t{tick:07d}.npz' → (track, policy, run, tick) | None."""
    stem = fname[:-4] if fname.endswith(".npz") else fname
    parts = stem.split("__")
    if len(parts) != 4:
        return None
    track, policy, run, ttok = parts
    if not (ttok.startswith("t") and ttok[1:].isdigit()):
        return None
    return track, policy, run, int(ttok[1:])


def _raw_path(raw_base, track, policy, run, tick):
    return str(Path(raw_base) / track / policy / run / "CAMERA_1" / f"{tick}_Intensity.jpeg")


class A2E2EDataset(Dataset):
    """raw image multi-frame dataset.

    aux dir glob → 각 sample = (curr + prev offset 이미지들, ego=zeros(2), aux target (40,7)).
    prev tick = tick - offset. raw 이미지 미존재 시 직전 가용 tick (최종 curr) 으로 clamp.
    """
    def __init__(self, aux_dir, raw_base, prev_offsets, run_filter=None, fx=640.0,
                 verify_exists=True):
        self.raw_base = raw_base
        self.prev_offsets = prev_offsets
        self.fx = float(fx)
        aux_files = sorted(glob.glob(str(Path(aux_dir) / "*.npz")))
        self.samples = []   # (aux_path, [curr_tick, prev_tick1, ...], track, policy, run)
        n_parse = 0; n_runfilt = 0; n_nofile = 0
        for f in aux_files:
            fname = Path(f).name
            parsed = _parse_aux_name(fname)
            if parsed is None:
                continue
            n_parse += 1
            track, policy, run, tick = parsed
            if run_filter and run_filter not in fname:
                n_runfilt += 1
                continue
            # curr 이미지 존재 검증 (없으면 sample drop)
            curr_path = _raw_path(raw_base, track, policy, run, tick)
            if verify_exists and not os.path.exists(curr_path):
                n_nofile += 1
                continue
            # prev tick: 없으면 직전 가용으로 clamp (curr 까지)
            ticks = [tick]
            last = tick
            for off in prev_offsets:
                pt = tick - off
                pp = _raw_path(raw_base, track, policy, run, pt)
                if verify_exists and not os.path.exists(pp):
                    pt = last
                ticks.append(pt)
                last = pt
            self.samples.append((f, ticks, track, policy, run))
        n_eligible = n_parse - n_runfilt
        print(f"[e2e-ds] {Path(aux_dir).name}  matched {len(self.samples):,}/{len(aux_files):,} "
              f"(parseable={n_parse:,} run-filtered={n_runfilt:,} no-rawfile={n_nofile:,}) "
              f"(N_frames={1+len(prev_offsets)})", flush=True)
        if verify_exists and n_eligible > 0 and len(self.samples) < 0.5 * n_eligible:
            raise RuntimeError(
                f"[e2e-ds][FAIL-FAST] join {len(self.samples):,}/{n_eligible:,} (eligible) < 50% — "
                f"aux↔raw 경로 불일치 의심 (no-rawfile={n_nofile:,}). "
                f"aux={Path(aux_dir).name} raw_base={raw_base} run-filter='{run_filter}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        ap, ticks, track, policy, run = self.samples[i]
        imgs = []   # frame0 = curr, 1.. = prev
        for t in ticks:
            p = _raw_path(self.raw_base, track, policy, run, t)
            try:
                imgs.append(_load_roi_tensor(p))
            except Exception:
                imgs.append(torch.zeros(3, ROI_Y1 - ROI_Y0, 1280))
        img_stack = torch.stack(imgs, dim=0)   # (Nf, 3, 250, 1280)
        ego_motion = torch.zeros(2, dtype=torch.float32)
        t = np.load(ap)["target"].astype(np.float32)   # (40, 7)
        return {
            "img_stack": img_stack,
            "ego_motion": ego_motion,
            "fx": torch.tensor(self.fx, dtype=torch.float32),
            "target": torch.from_numpy(t),
        }


def collate(items):
    return {k: torch.stack([it[k] for it in items], dim=0) for k in items[0]}


# loss — train_a2_multiframe.py 동일 ------------------------------------
def _reg_loss(logits, target, scales, obj_mask, predict_var):
    if not obj_mask.any():
        return torch.zeros((), device=logits.device)
    midx = mean_indices(predict_var)
    mean = logits[..., midx]
    tgt_reg = target[..., 1:7] / scales
    if not predict_var:
        m = obj_mask.unsqueeze(-1).expand_as(mean)
        return F.l1_loss(mean[m], tgt_reg[m])
    sidx = sigma_indices()
    logvar = logits[..., sidx].clamp(-7.0, 7.0)
    nll = 0.5 * (torch.exp(-logvar) * (mean - tgt_reg) ** 2 + logvar)
    nll = nll.mean(-1)
    return nll[obj_mask].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aux-dir", action="append", required=True)
    ap.add_argument("--run-filter", action="append", default=[])
    ap.add_argument("--raw-base", required=True,
                    help="raw 이미지 base. {raw_base}/{track}/{policy}/{run}/CAMERA_1/{tick}_Intensity.jpeg")
    ap.add_argument("--seg-ckpt", default="outputs_multi/seg_pretrain_fx640_v3/best_miou_backbone.pt",
                    help="backbone init (seg pretrain). load_seg_pretrain 로 encoder.backbone 채움.")
    ap.add_argument("--n-frames", type=int, default=2)
    ap.add_argument("--prev-offsets", type=str, default="",
                    help="comma-sep prev tick offsets (예: '20' / '20,40'). 비면 PREV_OFFSETS[n_frames]. "
                         "지정 시 n_frames = 1+len(offsets) 강제.")
    ap.add_argument("--fx", type=float, default=640.0)
    ap.add_argument("--d-token", type=int, default=256)
    ap.add_argument("--d-out", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4, help="head lr")
    ap.add_argument("--backbone-lr-mult", type=float, default=0.1,
                    help="backbone(encoder) lr = head lr × 이 값 (seg pretrain 보존)")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--w-obj", type=float, default=1.0)
    ap.add_argument("--w-reg", type=float, default=1.0)
    ap.add_argument("--use-focal", action="store_true")
    ap.add_argument("--focal-alpha", type=float, default=0.75)
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--scale-v", type=float, default=20.0)
    ap.add_argument("--scale-dist", type=float, default=40.0)
    ap.add_argument("--gaussian-nll", action="store_true")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--resume-from", default=None,
                    help="(선택) 기존 a_2 ckpt resume (encoder+fusion+head). 없으면 seg-ckpt 만 backbone init.")
    ap.add_argument("--limit-steps", type=int, default=0,
                    help=">0 이면 epoch 당 이 step 수만 돌고 break (로컬 smoke 용).")
    ap.add_argument("--no-verify-exists", action="store_true",
                    help="raw 이미지 존재 검증 skip (느린 NFS glob 회피). smoke 에선 검증 권장.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="outputs_multi")
    ap.add_argument("--name", default="a2_e2e")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out_dir = Path(args.out_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / "train.log", "w")
    def logp(m): print(m, flush=True); log_f.write(m + "\n"); log_f.flush()
    logp(f"args={vars(args)}")

    if args.prev_offsets.strip():
        prev_offsets = [int(x) for x in args.prev_offsets.split(",") if x.strip()]
        args.n_frames = 1 + len(prev_offsets)
    else:
        prev_offsets = PREV_OFFSETS[args.n_frames]
    logp(f"[frames] n_frames={args.n_frames}  prev_offsets={prev_offsets}")

    run_filters = args.run_filter + [None] * (len(args.aux_dir) - len(args.run_filter))
    ds_list = []
    for ad, rf in zip(args.aux_dir, run_filters):
        ds_list.append(A2E2EDataset(ad, args.raw_base, prev_offsets, run_filter=rf, fx=args.fx,
                                     verify_exists=not args.no_verify_exists))
    full = ConcatDataset(ds_list)
    N = len(full)
    n_val = int(N * args.val_ratio)
    n_tr = N - n_val
    tr_ds, va_ds = random_split(full, [n_tr, n_val],
                                generator=torch.Generator().manual_seed(args.seed))
    logp(f"[data] total={N:,}  train={n_tr:,}  val={n_val:,}")
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                       collate_fn=collate, pin_memory=True, persistent_workers=args.workers > 0)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                       collate_fn=collate, pin_memory=True, persistent_workers=args.workers > 0)

    # 모델: A2ModelMultiFrame raw path. freeze 안 함 → encoder(backbone+proj) 까지 학습.
    model = A2ModelMultiFrame(d_token=args.d_token, d_out=args.d_out, n_frames=args.n_frames,
                              predict_var=args.gaussian_nll).to(args.device)
    seg_info = model.load_seg_pretrain(args.seg_ckpt)
    logp(f"[seg-init] {args.seg_ckpt}  {seg_info}")
    if args.resume_from:
        sd = torch.load(args.resume_from, map_location=args.device, weights_only=False)
        if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
        own = model.state_dict()
        skip = [k for k in list(sd.keys()) if k in own and sd[k].shape != own[k].shape]
        for k in skip: del sd[k]
        miss, unexp = model.load_state_dict(sd, strict=False)
        logp(f"[resume] {args.resume_from}  missing={len(miss)} shape-skip={len(skip)}")

    # backbone lr 분리: encoder.backbone = head lr × mult, 나머지(proj/fusion/fc_proj/head) = head lr.
    backbone_ids = {id(p) for p in model.encoder.backbone.parameters()}
    bb_params = [p for p in model.parameters() if id(p) in backbone_ids and p.requires_grad]
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids and p.requires_grad]
    bb_lr = args.lr * args.backbone_lr_mult
    optim = torch.optim.AdamW(
        [{"params": bb_params, "lr": bb_lr, "name": "backbone"},
         {"params": head_params, "lr": args.lr, "name": "head"}],
        weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    n_bb = sum(p.numel() for p in bb_params); n_hd = sum(p.numel() for p in head_params)
    logp(f"[optim] backbone params={n_bb/1e6:.2f}M lr={bb_lr:.2e} | head params={n_hd/1e6:.2f}M lr={args.lr:.2e}")

    use_amp = args.amp and args.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    bce_fn = nn.BCEWithLogitsLoss()
    def focal_loss(logits, targets, alpha=args.focal_alpha, gamma=args.focal_gamma):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** gamma * bce).mean()
    obj_loss_fn = focal_loss if args.use_focal else bce_fn
    logp(f"[loss] obj={'focal' if args.use_focal else 'bce'} reg={'NLL' if args.gaussian_nll else 'L1'} amp={use_amp}")
    scales = torch.tensor([args.scale_v, args.scale_v, args.scale_v, args.scale_v,
                           1.0, args.scale_dist], device=args.device)

    def forward_batch(b):
        img_stack = b["img_stack"].to(args.device, non_blocking=True)   # (B,Nf,3,250,1280)
        ego = b["ego_motion"].to(args.device, non_blocking=True)
        target = b["target"].to(args.device, non_blocking=True)
        image_list = [img_stack[:, fi] for fi in range(img_stack.shape[1])]
        logits = model(image_list, ego)            # raw path → encoder forward (backbone 포함)
        obj_logit = logits[..., 0]; obj_gt = target[..., 0]
        loss_obj = obj_loss_fn(obj_logit, obj_gt)
        obj_mask = obj_gt > 0.5
        loss_reg = _reg_loss(logits, target, scales, obj_mask, args.gaussian_nll)
        loss = args.w_obj * loss_obj + args.w_reg * loss_reg
        return loss, loss_obj, loss_reg, logits, target, img_stack.size(0)

    best_val = float("inf")
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        for bi, b in enumerate(tr_dl):
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss, loss_obj, loss_reg, _, _, _ = forward_batch(b)
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            if (bi + 1) % args.log_every == 0:
                logp(f"[ep {ep:02d} | {bi+1:>5,}/{len(tr_dl):,}] total={loss:.4f} "
                     f"obj={loss_obj:.4f} reg={loss_reg:.4f} "
                     f"bb_lr={optim.param_groups[0]['lr']:.2e} hd_lr={optim.param_groups[1]['lr']:.2e}")
            if args.limit_steps and (bi + 1) >= args.limit_steps:
                logp(f"[limit-steps] break at {bi+1}")
                break
        sched.step()
        # val
        model.eval()
        with torch.no_grad():
            v_total, v_obj, v_reg, n = 0., 0., 0., 0
            for vi, b in enumerate(va_dl):
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss, l_obj, l_reg, _, _, bsz = forward_batch(b)
                v_total += loss.item() * bsz; v_obj += l_obj.item() * bsz
                v_reg += l_reg.item() * bsz; n += bsz
                if args.limit_steps and (vi + 1) >= args.limit_steps:
                    break
        if n > 0:
            v_total /= n; v_obj /= n; v_reg /= n
        dt = time.time() - t0
        logp(f"[ep {ep:02d}] val total={v_total:.4f} obj={v_obj:.4f} reg={v_reg:.4f} time={dt:.1f}s")
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
