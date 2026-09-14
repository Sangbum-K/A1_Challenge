"""SAM mask 기반 image encoder pretraining.

목적: SCC controller training 전에 LateralStripEncoder 의 backbone 을
도메인-특화 (track / vehicle / wall) feature 로 사전학습.

사용:
  ~/miniforge3/envs/ml_con/bin/python scripts/train_seg_pretrain.py \
    --backbone efficientnet_b0 --epochs 20 --batch-size 8 --lr 3e-4

체크포인트:
  outputs_multi/seg_pretrain_<name>/best_backbone.pt    (LateralStripEncoder.backbone 와 호환)
  outputs_multi/seg_pretrain_<name>/best_full.pt        (encoder + decoder 전체)
"""
from __future__ import annotations
import os, sys, json, time, argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.datamodule.seg_pretrain_dataset import (
    SegPretrainDataset, collate_seg, IGNORE_IDX,
    IMAGENET_MEAN, IMAGENET_STD,
)
from src.model.image_models.seg_pretrain_model import SegPretrainModel

# 4-class palette: bg, road, vehicle, wall  (BGR-friendly RGB)
PALETTE = np.array([
    [  0,   0,   0],   # bg     - 검정
    [ 80,  80,  80],   # road   - 회색
    [255,  80,  80],   # vehicle- 빨강
    [ 80, 200, 255],   # wall   - 하늘
], dtype=np.uint8)


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="tf_efficientnetv2_s",
                    help="timm backbone. 기본: tf_efficientnetv2_s")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default="outputs_multi/seg_pretrain")
    ap.add_argument("--name", default="b0_v1")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None, help="dataset frame 수 제한 (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true", help="mixed precision (FP16) 활성")
    ap.add_argument("--n-viz", type=int, default=8,
                    help="정성 평가용 고정 frame 수 (val 에서 sampling)")
    ap.add_argument("--viz-every", type=int, default=1, help="N epoch 마다 viz dump")
    # combined / 신규 데이터 통합용
    ap.add_argument("--extra-meta",     action="append", default=[],
                    help="추가 dataset 의 meta json. 여러 번 사용 가능 (--extra-mask-dir/--extra-raw-base 와 같은 개수)")
    ap.add_argument("--extra-mask-dir", action="append", default=[])
    ap.add_argument("--extra-raw-base", action="append", default=[])
    ap.add_argument("--resume-from",    default=None, help="prior best ckpt 에서 warm-start")
    ap.add_argument("--no-default",     action="store_true",
                    help="default sangam fx320 (MultiAgnet) dataset 로드 안 함. extras 만 사용")
    return ap.parse_args()


def _denorm_img(img_t: torch.Tensor) -> np.ndarray:
    """(3, H, W) ImageNet-normed → (H, W, 3) uint8."""
    arr = img_t.detach().cpu().numpy().transpose(1, 2, 0)
    arr = arr * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def _label_to_rgb(label: np.ndarray) -> np.ndarray:
    """(H, W) int64 → (H, W, 3) uint8 (ignore→black)."""
    rgb = np.zeros((*label.shape, 3), dtype=np.uint8)
    valid = label >= 0
    rgb[valid] = PALETTE[label[valid]]
    return rgb


def _overlay(img: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """img/mask_rgb (H, W, 3) uint8 → blended uint8."""
    return (img.astype(np.float32) * (1 - alpha) + mask_rgb.astype(np.float32) * alpha).astype(np.uint8)


def dump_viz_panel(model, viz_batch, device, out_path, epoch_label: str, amp_enabled: bool):
    """viz_batch: dataset items 의 list. 각 item 에 image, label 포함.
    panel: rows=N_viz, cols=4 (image | gt | pred | overlay).
    """
    from PIL import Image as PILImage
    model.eval()
    imgs = torch.stack([b["image"] for b in viz_batch], dim=0).to(device)
    labs = torch.stack([b["label"] for b in viz_batch], dim=0)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp_enabled):
        logits = model(imgs)
    preds = logits.argmax(dim=1).cpu().numpy()

    N = len(viz_batch); H, W = labs.shape[-2:]
    panel = np.zeros((N * H, 4 * W, 3), dtype=np.uint8)
    for i in range(N):
        img_np = _denorm_img(viz_batch[i]["image"])     # (H, W, 3)
        gt_rgb = _label_to_rgb(labs[i].numpy())
        pr_rgb = _label_to_rgb(preds[i])
        ov     = _overlay(img_np, pr_rgb, alpha=0.5)
        panel[i*H:(i+1)*H,   0:  W] = img_np
        panel[i*H:(i+1)*H,   W:2*W] = gt_rgb
        panel[i*H:(i+1)*H, 2*W:3*W] = pr_rgb
        panel[i*H:(i+1)*H, 3*W:4*W] = ov
    PILImage.fromarray(panel).save(out_path, optimize=True)


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def compute_iou(pred: torch.Tensor, label: torch.Tensor, n_classes: int) -> list[float]:
    """pred (B, H, W) int64, label (B, H, W) int64 → per-class IoU (ignore 제외)."""
    ious = []
    valid = label != IGNORE_IDX
    for c in range(n_classes):
        p = (pred == c) & valid
        g = (label == c) & valid
        inter = (p & g).sum().item()
        union = (p | g).sum().item()
        ious.append(inter / union if union > 0 else float("nan"))
    return ious


def main():
    args = get_args()
    seed_all(args.seed)
    out_dir = os.path.join(args.out_dir + "_" + args.name)
    os.makedirs(out_dir, exist_ok=True)

    from torch.utils.data import ConcatDataset, Subset
    datasets = []
    if not args.no_default:
        datasets.append(SegPretrainDataset())
        print(f"[data] sangam default: {len(datasets[0])}")
    if args.extra_meta:
        assert len(args.extra_meta) == len(args.extra_mask_dir) == len(args.extra_raw_base), \
            "--extra-meta/--extra-mask-dir/--extra-raw-base 개수 일치 필요"
        for m, md, rb in zip(args.extra_meta, args.extra_mask_dir, args.extra_raw_base):
            ds = SegPretrainDataset(meta_path=m, mask_dir=md, raw_base=rb)
            datasets.append(ds)
            print(f"[data] extra: meta={m} → {len(ds)}")
    assert datasets, "no dataset specified — pass --extra-* or remove --no-default"
    full = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    if args.limit is not None:
        idx = list(range(min(args.limit, len(full))))
        full = Subset(full, idx)
    n_total = len(full)
    n_val = max(1, int(n_total * args.val_ratio))
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(full, [n_train, n_val], generator=g)
    print(f"[data] total={n_total}  train={n_train}  val={n_val}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=True,
                               collate_fn=collate_seg, drop_last=True, persistent_workers=args.workers > 0)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                               num_workers=args.workers, pin_memory=True,
                               collate_fn=collate_seg, persistent_workers=args.workers > 0)

    # 정성 viz 용 고정 sample (val_set 의 첫 N_viz)
    n_viz = min(args.n_viz, len(val_set))
    viz_items = [val_set[i] for i in range(n_viz)]
    viz_dir = os.path.join(out_dir, "viz_progress")
    os.makedirs(viz_dir, exist_ok=True)
    # 학습 전 baseline (ImageNet pretrained only)
    pass    # baseline panel은 model 만든 후

    n_classes = 4
    model = SegPretrainModel(backbone=args.backbone, n_classes=n_classes).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {args.backbone}  params={n_params/1e6:.2f} M")
    if args.resume_from:
        sd = torch.load(args.resume_from, map_location=args.device, weights_only=False)
        if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
        m, u = model.load_state_dict(sd, strict=False)
        print(f"[resume] {args.resume_from}  missing={len(m)}  unexpected={len(u)}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE_IDX)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    best_miou = -1.0
    best_val_loss = float("inf")
    log_path = os.path.join(out_dir, "train.log")
    log_f = open(log_path, "w")
    def logp(msg):
        print(msg, flush=True); log_f.write(msg + "\n"); log_f.flush()

    logp(f"args={vars(args)}")
    logp(f"out_dir={out_dir}")

    # baseline viz (epoch 0 = ImageNet 초기값만)
    dump_viz_panel(model, viz_items, args.device,
                   os.path.join(viz_dir, "ep00_init.png"),
                   "init", args.amp)
    logp(f"  viz baseline → {viz_dir}/ep00_init.png  ({n_viz} samples)")

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0; running_n = 0
        for it, batch in enumerate(train_loader):
            img = batch["image"].to(args.device, non_blocking=True)
            lab = batch["label"].to(args.device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                logits = model(img)
                loss = ce(logits, lab)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            running_loss += loss.item() * img.size(0)
            running_n += img.size(0)
            if (it + 1) % args.log_every == 0:
                avg = running_loss / running_n
                logp(f"[ep {epoch+1:02d} | {it+1:4d}/{len(train_loader)}] loss={avg:.4f}  "
                     f"lr={optim.param_groups[0]['lr']:.2e}")
        sched.step()
        train_loss = running_loss / max(1, running_n)

        # ----- val -----
        model.eval()
        val_loss = 0.0; val_n = 0
        iou_acc = np.zeros(n_classes); iou_cnt = np.zeros(n_classes)
        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(args.device, non_blocking=True)
                lab = batch["label"].to(args.device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=args.amp):
                    logits = model(img)
                    loss = ce(logits, lab)
                val_loss += loss.item() * img.size(0); val_n += img.size(0)
                pred = logits.argmax(dim=1)
                for b in range(pred.size(0)):
                    ious = compute_iou(pred[b:b+1], lab[b:b+1], n_classes)
                    for c, v in enumerate(ious):
                        if v == v:  # not nan
                            iou_acc[c] += v; iou_cnt[c] += 1
        val_loss /= max(1, val_n)
        ious = [iou_acc[c] / iou_cnt[c] if iou_cnt[c] > 0 else 0.0 for c in range(n_classes)]
        miou = float(np.mean(ious))
        dt = time.time() - t0
        logp(f"[ep {epoch+1:02d}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
             f"mIoU={miou:.3f}  ious={[f'{v:.3f}' for v in ious]}  "
             f"time={dt:.1f}s")

        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), os.path.join(out_dir, "best_miou_full.pt"))
            torch.save(model.backbone_state_dict(), os.path.join(out_dir, "best_miou_backbone.pt"))
            logp(f"  -> saved best mIoU ({miou:.3f})")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(out_dir, "best_valloss_full.pt"))
            torch.save(model.backbone_state_dict(), os.path.join(out_dir, "best_valloss_backbone.pt"))
            logp(f"  -> saved best val_loss ({val_loss:.4f})")

        # epoch viz
        if (epoch + 1) % args.viz_every == 0:
            viz_path = os.path.join(viz_dir, f"ep{epoch+1:02d}.png")
            dump_viz_panel(model, viz_items, args.device, viz_path,
                            f"ep{epoch+1}", args.amp)
            logp(f"  viz → {viz_path}")

    logp(f"\n[done] best_mIoU={best_miou:.3f}  best_val_loss={best_val_loss:.4f}")
    logp(f"  ckpt(mIoU)    = {out_dir}/best_miou_backbone.pt")
    logp(f"  ckpt(val_loss) = {out_dir}/best_valloss_backbone.pt")
    log_f.close()


if __name__ == "__main__":
    main()
