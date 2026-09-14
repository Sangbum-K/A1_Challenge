"""i_1 + i_2 + a_2 학습.

사용:
  python scripts/train_a2_head.py \
    --pretrained outputs_multi/seg_pretrain_v2s_full/best_backbone.pt \
    --epochs 25 --batch-size 12 --lr 2e-4

체크포인트: outputs_multi/a2_<name>/best_loss.pt
"""
from __future__ import annotations
import os, sys, time, json, argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.datamodule.a2_pair_dataset import (
    A2PairDataset, collate_a2,
    IMAGENET_MEAN, IMAGENET_STD,
)
from src.model.image_models.a2_model import A2Model

CH_NAMES = ["obj", "vL_abs", "vY_abs", "vL_rel", "vY_rel", "prox", "dist"]


def _denorm_img(img_t):
    """(3, H, W) ImageNet-normed → (H, W, 3) uint8."""
    arr = img_t.detach().cpu().numpy().transpose(1, 2, 0)
    arr = arr * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def dump_viz_panel(model, viz_batch, device, out_path, args):
    """viz_batch: list of dataset items.
    panel rows: 2 per frame (top=image + GT bins, bot=image + Pred bins).
    bin overlay: 32px wide vertical bar at bottom of image, colored by v_rel
      (red catching up, green pulling away), label = dist.
    """
    import torch
    from PIL import Image as PILImage, ImageDraw, ImageFont
    model.eval()
    img_c = torch.stack([b["image_curr"] for b in viz_batch], dim=0).to(device)
    img_p = torch.stack([b["image_prev_rot"] for b in viz_batch], dim=0).to(device)
    ego   = torch.stack([b["ego_motion"] for b in viz_batch], dim=0).to(device)
    tgts  = torch.stack([b["target"] for b in viz_batch], dim=0).numpy()
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=args.amp):
        logits = model(img_c, img_p, ego)
    obj_prob = torch.sigmoid(logits[..., 0]).cpu().numpy()
    scales = np.array([args.scale_v, args.scale_v, args.scale_v, args.scale_v, 1.0, args.scale_dist])
    pred_reg = (logits[..., 1:7].cpu().numpy() * scales)
    H, W = 250, 1280; N = len(viz_batch); HBAR = 36
    panel_h = (H + HBAR) * 2 * N + 12 * N    # +12 between frames
    panel = np.full((panel_h, W, 3), 240, dtype=np.uint8)
    try: font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 11)
    except: font = ImageFont.load_default()
    try: font_small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 9)
    except: font_small = font

    def render_strip(img_np, obj_arr, vrel_arr, dist_arr, kind: str):
        """img_np (H, W, 3) → (H+HBAR, W, 3) with horizontal bin strip at bottom."""
        out = np.full((H + HBAR, W, 3), 240, dtype=np.uint8)
        out[:H] = img_np
        pim = PILImage.fromarray(out); draw = ImageDraw.Draw(pim)
        bin_w_px = W / 40
        # top label
        draw.rectangle([0, H, 60, H + HBAR], fill=(40, 40, 40))
        draw.text((4, H + 8), kind, fill=(255, 255, 255), font=font)
        for bi in range(40):
            p = float(obj_arr[bi])
            if kind == "GT":
                if p < 0.5: continue   # GT obj 0
            else:
                if p < 0.3: continue   # Pred threshold
            d = float(dist_arr[bi]); v = float(vrel_arr[bi])
            # color
            if v < 0:
                c = (240, int(80 + 40 * min(1, -v/15)), int(80 + 40 * min(1, -v/15)))
            else:
                c = (int(80 + 40 * min(1, v/15)), 240, int(80 + 40 * min(1, v/15)))
            x0 = int(bi * bin_w_px); x1 = int((bi+1) * bin_w_px)
            # bar in image area (overlay translucent)
            alpha = 0.18 if kind == "GT" else 0.30
            ov = np.full((H, x1-x0, 3), c, dtype=np.uint8)
            mix = (out[:H, x0:x1].astype(np.float32) * (1-alpha) + ov.astype(np.float32) * alpha)
            out[:H, x0:x1] = np.clip(mix, 0, 255).astype(np.uint8)
            # rebuild pil after image overlay
            pim = PILImage.fromarray(out); draw = ImageDraw.Draw(pim)
            # label cell
            cell_color = c
            if kind == "Pred":
                # outline color: green if matches GT obj, red if doesn't (input later)
                pass
            draw.rectangle([x0, H, x1, H+HBAR], fill=cell_color, outline=(60, 60, 60))
            draw.text((x0+2, H+2), f"{d:.0f}m", fill=(20, 20, 20), font=font_small)
            draw.text((x0+2, H+18), f"{v:+.0f}", fill=(20, 20, 20), font=font_small)
            out = np.array(pim)
        return out

    y = 0
    for i, item in enumerate(viz_batch):
        img_np = _denorm_img(item["image_curr"])
        # GT strip
        gt_strip = render_strip(img_np, tgts[i, :, 0], tgts[i, :, 3], tgts[i, :, 6], "GT")
        panel[y:y+H+HBAR] = gt_strip; y += H + HBAR
        # Pred strip
        pred_strip = render_strip(img_np, obj_prob[i], pred_reg[i, :, 2], pred_reg[i, :, 5], "Pred")
        panel[y:y+H+HBAR] = pred_strip; y += H + HBAR
        y += 12   # gap

    PILImage.fromarray(panel).save(out_path, optimize=True)


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="tf_efficientnetv2_s")
    ap.add_argument("--pretrained", default=None,
                    help="seg_pretrain best_backbone.pt path (있으면 load)")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr-encoder", type=float, default=None,
                    help="encoder LR 별도. None 이면 lr (head 와 동일)")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default="outputs_multi/a2")
    ap.add_argument("--name", default="v1")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--w-obj", type=float, default=1.0)
    ap.add_argument("--w-reg", type=float, default=1.0)
    # 회귀 scale (norm 후 L1). 안정성 위해 학습 시 target 을 이 값들로 나눠서 L1
    ap.add_argument("--scale-v",   type=float, default=20.0, help="v_*_abs / v_*_rel m/s scale")
    ap.add_argument("--scale-dist", type=float, default=40.0, help="dist m scale (prox 는 무scale, 이미 0..1)")
    ap.add_argument("--n-viz", type=int, default=6, help="정성 평가 frame 수 (val 에서)")
    ap.add_argument("--viz-every", type=int, default=1)
    # combined / 신규 통합
    ap.add_argument("--extra-meta",     action="append", default=[])
    ap.add_argument("--extra-a2-dir",   action="append", default=[])
    ap.add_argument("--extra-raw-base", action="append", default=[])
    ap.add_argument("--resume-from",    default=None, help="prior a_2 best ckpt warm-start")
    ap.add_argument("--prev-dtick",     type=int, default=20,
                    help="고정 prev_dtick (5=50ms, 20=200ms 기본). multi-dt 비활성")
    ap.add_argument("--close-lead-meta", default=None, help="close-lead 가중 활성")
    ap.add_argument("--close-thr",       type=float, default=30.0)
    ap.add_argument("--d-out", type=int, default=None,
                    help="A2Model 의 fusion 후 fc_proj output dim. None=d_token (proj 없음). 128 등 작은 값 → Linear(d_token→d_out) 추가하여 iy_model 호환 image_mem 생성.")
    ap.add_argument("--freeze-enc-fusion", action="store_true",
                    help="encoder + fusion frozen (v4 ckpt 재사용). fc_proj + head 만 학습.")
    ap.add_argument("--close-weight",    type=float, default=3.0, help="oversample weight")
    ap.add_argument("--no-default",     action="store_true",
                    help="default sangam fx320 (MultiAgnet) dataset 로드 안 함. extras 만 사용")
    ap.add_argument("--no-require-obj", action="store_true",
                    help="obj 없는 frame 도 포함 (gate 학습 negative supervision)")
    return ap.parse_args()


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def split_losses(logits, target, obj_mask, args, sample_weight=None):
    """logits (B, 40, 7); target (B, 40, 7); obj_mask (B, 40) bool.
    sample_weight (B,) or None — close-lead 가중치 per-sample.
    """
    if sample_weight is None:
        bce = F.binary_cross_entropy_with_logits(logits[..., 0], target[..., 0])
    else:
        bce_per = F.binary_cross_entropy_with_logits(logits[..., 0], target[..., 0], reduction="none")
        bce_per_sample = bce_per.mean(dim=1)                              # (B,)
        w = sample_weight.to(bce_per.device).clamp(min=1e-6)
        bce = (bce_per_sample * w).sum() / w.sum().clamp(min=1e-6)
    if obj_mask.any():
        scales = torch.tensor([args.scale_v, args.scale_v, args.scale_v, args.scale_v,
                                1.0, args.scale_dist], device=logits.device)
        pred_reg = logits[..., 1:7]
        tgt_reg  = target[..., 1:7] / scales
        if sample_weight is None:
            m = obj_mask.unsqueeze(-1).expand_as(pred_reg)
            diff = F.l1_loss(pred_reg[m], tgt_reg[m])
        else:
            # per-sample reg L1: mean over (bin × ch) where obj_mask True
            err = (pred_reg - tgt_reg).abs()                              # (B, 40, 6)
            m   = obj_mask.unsqueeze(-1).float()                          # (B, 40, 1)
            denom_per = (m.sum(dim=(1,2)) * 6).clamp(min=1.0)
            err_per   = (err * m).sum(dim=(1,2)) / denom_per              # (B,)
            w = sample_weight.to(err.device).clamp(min=1e-6)
            diff = (err_per * w).sum() / w.sum().clamp(min=1e-6)
    else:
        diff = torch.zeros((), device=logits.device)
    total = args.w_obj * bce + args.w_reg * diff
    return total, bce.detach(), diff.detach()


@torch.no_grad()
def eval_metrics(logits, target, obj_mask, args):
    """val 용: per-ch L1 (un-normalized m/s, m), obj BCE, obj accuracy."""
    obj_logit = logits[..., 0]
    obj_prob = torch.sigmoid(obj_logit)
    obj_pred = obj_prob > 0.5
    obj_gt = target[..., 0] > 0.5
    obj_acc = (obj_pred == obj_gt).float().mean().item()

    scales = torch.tensor([args.scale_v, args.scale_v, args.scale_v, args.scale_v,
                            1.0, args.scale_dist], device=logits.device)
    pred_reg = logits[..., 1:7] * scales              # → original units
    tgt_reg  = target[..., 1:7]
    per_ch_l1 = []
    if obj_mask.any():
        m = obj_mask.unsqueeze(-1)
        for c in range(6):
            mm = m[..., 0]
            l1 = (pred_reg[..., c][mm] - tgt_reg[..., c][mm]).abs().mean().item()
            per_ch_l1.append(l1)
    else:
        per_ch_l1 = [float("nan")] * 6
    return obj_acc, per_ch_l1


def main():
    args = get_args()
    seed_all(args.seed)
    out_dir = f"{args.out_dir}_{args.name}"
    os.makedirs(out_dir, exist_ok=True)

    # viz 전용: obj 있는 frame 만 골라야 의미 있는 panel
    from torch.utils.data import ConcatDataset, Subset, WeightedRandomSampler
    datasets = []
    req_obj = not args.no_require_obj
    if not args.no_default:
        datasets.append(A2PairDataset(require_obj=req_obj, prev_dtick=args.prev_dtick,
                                       close_lead_meta_path=args.close_lead_meta))
        print(f"[data] default sangam fx320: {len(datasets[0])} (prev_dtick={args.prev_dtick})")
    if args.extra_meta:
        assert len(args.extra_meta) == len(args.extra_a2_dir) == len(args.extra_raw_base), \
            "--extra-meta/--extra-a2-dir/--extra-raw-base 개수 일치"
        for m, ad, rb in zip(args.extra_meta, args.extra_a2_dir, args.extra_raw_base):
            ds = A2PairDataset(meta_path=m, a2_dir=ad, raw_base=rb, require_obj=req_obj,
                                prev_dtick=args.prev_dtick, close_lead_meta_path=args.close_lead_meta)
            datasets.append(ds)
            print(f"[data] extra: {m} → {len(ds)}")
    assert datasets, "no dataset specified — pass --extra-* or remove --no-default"
    full = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    if args.limit:
        full = Subset(full, list(range(min(args.limit, len(full)))))
    n_total = len(full); n_val = max(1, int(n_total * args.val_ratio))
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(args.seed)
    tr, va = random_split(full, [n_train, n_val], generator=g)
    print(f"[data] total={n_total}  train={n_train}  val={n_val}")

    tl = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                     num_workers=args.workers, pin_memory=True, collate_fn=collate_a2,
                     drop_last=True, persistent_workers=args.workers > 0)
    vl = DataLoader(va, batch_size=args.batch_size, shuffle=False,
                     num_workers=args.workers, pin_memory=True, collate_fn=collate_a2,
                     persistent_workers=args.workers > 0)

    # viz batch (val 의 처음 N_viz)
    n_viz = min(args.n_viz, len(va))
    viz_items = [va[i] for i in range(n_viz)]
    viz_dir = os.path.join(out_dir, "viz_progress")
    os.makedirs(viz_dir, exist_ok=True)

    model = A2Model(img_backbone=args.backbone, d_out=args.d_out).to(args.device)
    if args.pretrained:
        info = model.load_seg_pretrain(args.pretrained)
        print(f"[i_1] loaded seg_pretrain backbone: {info}")
    if args.resume_from:
        sd = torch.load(args.resume_from, map_location=args.device, weights_only=False)
        if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
        # shape mismatch key skip (예: d_token 다를 때 head/proj)
        own = model.state_dict()
        skipped = [k for k in list(sd.keys()) if k in own and sd[k].shape != own[k].shape]
        for k in skipped: del sd[k]
        m, u = model.load_state_dict(sd, strict=False)
        print(f"[resume] {args.resume_from}  missing={len(m)}  unexpected={len(u)}  shape-skip={len(skipped)}")
        if skipped: print(f"  shape-skip keys: {skipped[:5]}")
    if args.freeze_enc_fusion:
        info = model.freeze_encoder_fusion()
        print(f"[freeze] encoder + fusion frozen → {info}")
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] params={n_params/1e6:.2f} M  (trainable={n_train/1e6:.3f} M)")

    enc_lr = args.lr_encoder if args.lr_encoder is not None else args.lr
    # frozen 시 lr=0 effective. trainable params 만 optimizer 에 넣음.
    trainable_groups = []
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    if enc_params:
        trainable_groups.append({"params": enc_params, "lr": enc_lr})
    fus_params = [p for p in model.fusion.parameters() if p.requires_grad]
    if fus_params:
        trainable_groups.append({"params": fus_params, "lr": args.lr})
    other_params = []
    if hasattr(model, 'fc_proj'):
        other_params += [p for p in model.fc_proj.parameters() if p.requires_grad]
    other_params += [p for p in model.head.parameters() if p.requires_grad]
    if other_params:
        trainable_groups.append({"params": other_params, "lr": args.lr})
    optim = torch.optim.AdamW(trainable_groups, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    log_path = os.path.join(out_dir, "train.log")
    log_f = open(log_path, "w")
    def logp(m): print(m, flush=True); log_f.write(m + "\n"); log_f.flush()
    logp(f"args={vars(args)}")

    # baseline viz (ep0)
    dump_viz_panel(model, viz_items, args.device,
                    os.path.join(viz_dir, "ep00_init.png"), args)
    logp(f"  viz baseline → {viz_dir}/ep00_init.png  ({n_viz} samples)")

    best_loss = float("inf"); best_obj_acc = -1.0
    for ep in range(args.epochs):
        model.train(); t0 = time.time()
        rl = rb = rr = 0.0; rn = 0
        for it, batch in enumerate(tl):
            img_c = batch["image_curr"].to(args.device, non_blocking=True)
            img_p = batch["image_prev_rot"].to(args.device, non_blocking=True)
            ego   = batch["ego_motion"].to(args.device, non_blocking=True)
            tgt   = batch["target"].to(args.device, non_blocking=True)
            msk   = batch["obj_mask"].to(args.device, non_blocking=True)
            sw    = None
            if args.close_lead_meta and "min_dist" in batch and args.close_weight > 1.0:
                md = batch["min_dist"].to(args.device)
                sw = torch.where(md < args.close_thr,
                                  torch.full_like(md, args.close_weight),
                                  torch.ones_like(md))
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                logits = model(img_c, img_p, ego)
                total, bce, reg = split_losses(logits, tgt, msk, args, sample_weight=sw)
            scaler.scale(total).backward()
            scaler.step(optim); scaler.update()
            B = img_c.size(0)
            rl += total.item() * B; rb += bce.item() * B; rr += reg.item() * B; rn += B
            if (it + 1) % args.log_every == 0:
                logp(f"[ep {ep+1:02d} | {it+1:4d}/{len(tl)}] total={rl/rn:.4f}  "
                     f"bce={rb/rn:.4f}  reg={rr/rn:.4f}  "
                     f"lr_enc={optim.param_groups[0]['lr']:.2e}")
        sched.step()
        train_loss = rl / max(1, rn)

        # val
        model.eval()
        vl_total = vl_bce = vl_reg = 0.0; vn = 0
        accs = []; ch_l1s = [[] for _ in range(6)]
        with torch.no_grad():
            for batch in vl:
                img_c = batch["image_curr"].to(args.device, non_blocking=True)
                img_p = batch["image_prev_rot"].to(args.device, non_blocking=True)
                ego   = batch["ego_motion"].to(args.device, non_blocking=True)
                tgt   = batch["target"].to(args.device, non_blocking=True)
                msk   = batch["obj_mask"].to(args.device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=args.amp):
                    logits = model(img_c, img_p, ego)
                    total, bce, reg = split_losses(logits, tgt, msk, args)
                B = img_c.size(0)
                vl_total += total.item()*B; vl_bce += bce.item()*B; vl_reg += reg.item()*B; vn += B
                a, l1s = eval_metrics(logits, tgt, msk, args)
                accs.append(a)
                for c, l in enumerate(l1s):
                    if not np.isnan(l): ch_l1s[c].append(l)
        val_loss = vl_total / max(1, vn)
        val_bce  = vl_bce / max(1, vn)
        val_reg  = vl_reg / max(1, vn)
        obj_acc = float(np.mean(accs)) if accs else 0.0
        ch_means = [float(np.mean(c)) if c else float("nan") for c in ch_l1s]
        dt = time.time() - t0
        logp(f"[ep {ep+1:02d}] train={train_loss:.4f}  "
             f"val={val_loss:.4f} (bce={val_bce:.4f} reg={val_reg:.4f})  "
             f"obj_acc={obj_acc:.3f}  "
             f"L1[vL_abs={ch_means[0]:.2f} vL_rel={ch_means[2]:.2f} dist={ch_means[5]:.2f}m]  "
             f"time={dt:.1f}s")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(out_dir, "best_loss.pt"))
            logp(f"  -> saved best val_loss ({val_loss:.4f})")
        if obj_acc > best_obj_acc:
            best_obj_acc = obj_acc

        if (ep + 1) % args.viz_every == 0:
            vp = os.path.join(viz_dir, f"ep{ep+1:02d}.png")
            dump_viz_panel(model, viz_items, args.device, vp, args)
            logp(f"  viz → {vp}")

    logp(f"\n[done] best_val_loss={best_loss:.4f}  best_obj_acc={best_obj_acc:.3f}")
    log_f.close()


if __name__ == "__main__":
    main()
