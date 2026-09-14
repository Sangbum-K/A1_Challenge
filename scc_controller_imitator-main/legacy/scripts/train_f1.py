"""f_1 학습: i_1 (seg_pretrain) + i_2 + f_1 (Professor #654 pretrain) + o_1.

Model: SCCControllerImitatorV2  (d_model=128, num_layers=6, nhead=8)
Loads:
  - i_1.backbone : seg_pretrain best_backbone.pt (efficientnetv2_s)
  - traj_embed/state_embed/pos_encoder.pe/transformer_encoder/fc_out : Professor #654
  - image fusion / image_pos_emb / proj : random init

데이터: scc_synth_v2 의 .scc.txt (logged + random) — frame 당 최대 K=100 samples 공유 image.

사용:
  python scripts/train_f1.py \
    --seg-ckpt outputs_multi/seg_pretrain_v2s_full/best_backbone.pt \
    --professor-ckpt /home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/controller_imitator/outputs_multi/654/2026-05-14-11-13-10_*/model_0.11137903_2437_0.00189743_0.03548164.pth \
    --epochs 15 --batch-size 8

체크포인트: outputs_multi/f1_<name>/best_loss.pt
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

from src.datamodule.scc_synth_pair_dataset import (
    SCCSynthPairDataset, collate_scc_synth, IMAGENET_MEAN, IMAGENET_STD,
)
from src.model.image_models.image_scc_models_v2 import SCCControllerImitatorV2


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg-ckpt", default=None, help="seg_pretrain best_backbone.pt (i_1)")
    ap.add_argument("--professor-ckpt", default=None,
                    help="controller_imitator #654 best model .pth (traj/state/pos/transformer/fc_out)")
    ap.add_argument("--img-backbone", default="tf_efficientnetv2_s")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--num-layers", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr-encoder", type=float, default=5e-5)
    ap.add_argument("--lr-professor", type=float, default=3e-5,
                    help="pretrained Professor 부분 LR (낮게)")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default="outputs_multi/f1")
    ap.add_argument("--name", default="v1")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    # target scale (학습 안정성)
    ap.add_argument("--scale-steer", type=float, default=10.0, help="steer deg scale")
    ap.add_argument("--scale-ax",    type=float, default=10.0, help="ax m/s² scale")
    return ap.parse_args()


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def split_losses(pred, target, mask, args):
    """pred/target (B, 40, 2). mask (B, 40) bool. Scaled L1."""
    scales = torch.tensor([args.scale_steer, args.scale_ax], device=pred.device)
    pred_n = pred / scales; tgt_n = target / scales
    if not mask.any():
        return torch.zeros((), device=pred.device), {}
    m = mask.unsqueeze(-1).expand_as(pred_n)
    diff = pred_n - tgt_n
    l1_steer = diff[..., 0][mask].abs().mean()
    l1_ax    = diff[..., 1][mask].abs().mean()
    total = l1_steer + l1_ax
    return total, {"l1_steer": l1_steer.detach(), "l1_ax": l1_ax.detach()}


def main():
    args = get_args()
    seed_all(args.seed)
    out_dir = f"{args.out_dir}_{args.name}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"[data] loading SCC synth dataset ...")
    full = SCCSynthPairDataset(limit=args.limit)
    n_total = len(full); n_val = max(1, int(n_total * args.val_ratio))
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(args.seed)
    tr, va = random_split(full, [n_train, n_val], generator=g)
    print(f"  total={n_total}  train={n_train}  val={n_val}")

    tl = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=True,
                    collate_fn=collate_scc_synth, drop_last=True,
                    persistent_workers=args.workers > 0)
    vl = DataLoader(va, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True,
                    collate_fn=collate_scc_synth,
                    persistent_workers=args.workers > 0)

    # build model (d_model=128 to match Professor #654)
    model = SCCControllerImitatorV2(
        img_backbone=args.img_backbone,
        d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers,
        traj_seq_len=40, traj_dim=3, state_dim=6,    # Professor #654 == 6 (5d pad +1 zero)
        num_model_output=2,
    ).to(args.device)

    # load Professor pretrain
    if args.professor_ckpt:
        print(f"[Professor] loading {args.professor_ckpt}")
        info = model.load_controller_imitator_pretrained(args.professor_ckpt)
        print(f"  loaded={len(info[0])}  missing={len(info[1])}  skipped={len(info[2])}")

    # load seg_pretrain backbone for i_1
    if args.seg_ckpt:
        print(f"[i_1] loading seg_pretrain backbone {args.seg_ckpt}")
        sd = torch.load(args.seg_ckpt, map_location="cpu")
        own = model.img_encoder.backbone.state_dict()
        loaded = 0
        for k, v in sd.items():
            k2 = k.replace("backbone.", "")
            if k2 in own and own[k2].shape == v.shape:
                own[k2] = v; loaded += 1
        model.img_encoder.backbone.load_state_dict(own)
        print(f"  loaded {loaded} backbone keys")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params/1e6:.2f} M")

    # parameter group: i_1 backbone / image fusion+proj / Professor (pretrained) / fc_out
    professor_keys = ["traj_embed", "state_embed", "pos_encoder", "transformer_encoder", "fc_out"]
    enc_params, fusion_params, prof_params = [], [], []
    for n, p in model.named_parameters():
        if n.startswith("img_encoder.backbone"):
            enc_params.append(p)
        elif any(n.startswith(k) for k in professor_keys):
            prof_params.append(p)
        else:
            fusion_params.append(p)
    optim = torch.optim.AdamW([
        {"params": enc_params, "lr": args.lr_encoder},
        {"params": prof_params, "lr": args.lr_professor},
        {"params": fusion_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    log_path = os.path.join(out_dir, "train.log")
    log_f = open(log_path, "w")
    def logp(m): print(m, flush=True); log_f.write(m + "\n"); log_f.flush()
    logp(f"args={vars(args)}  param_groups: enc={len(enc_params)} prof={len(prof_params)} fusion={len(fusion_params)}")

    best_loss = float("inf")
    for ep in range(args.epochs):
        model.train(); t0 = time.time()
        rl = 0.0; rsteer = 0.0; rax = 0.0; rn = 0
        for it, batch in enumerate(tl):
            img_c = batch["image_curr"].to(args.device, non_blocking=True)
            img_p = batch["image_prev_rot"].to(args.device, non_blocking=True)
            ego5  = batch["ego_5d"].to(args.device, non_blocking=True)
            traj  = batch["traj_40"].to(args.device, non_blocking=True)
            tgt   = batch["target"].to(args.device, non_blocking=True)
            mask  = batch["target_mask"].to(args.device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                pred = model({
                    "image_curr": img_c, "image_prev_rot": img_p,
                    "ego_5d": ego5, "traj_30": traj,    # v2 expects key 'traj_30'
                })
                total, parts = split_losses(pred, tgt, mask, args)
            scaler.scale(total).backward()
            scaler.step(optim); scaler.update()
            B = img_c.size(0)
            rl += total.item() * B
            if parts:
                rsteer += parts["l1_steer"].item() * B
                rax    += parts["l1_ax"].item() * B
            rn += B
            if (it + 1) % args.log_every == 0:
                logp(f"[ep {ep+1:02d} | {it+1:4d}/{len(tl)}] total={rl/rn:.4f}  "
                     f"l1_steer={rsteer/rn:.4f} l1_ax={rax/rn:.4f}  "
                     f"lr_enc={optim.param_groups[0]['lr']:.2e} lr_prof={optim.param_groups[1]['lr']:.2e}")
        sched.step()
        train_loss = rl / max(1, rn)

        # val
        model.eval()
        vl_total = vsteer = vax = 0.0; vn = 0
        with torch.no_grad():
            for batch in vl:
                img_c = batch["image_curr"].to(args.device, non_blocking=True)
                img_p = batch["image_prev_rot"].to(args.device, non_blocking=True)
                ego5  = batch["ego_5d"].to(args.device, non_blocking=True)
                traj  = batch["traj_40"].to(args.device, non_blocking=True)
                tgt   = batch["target"].to(args.device, non_blocking=True)
                mask  = batch["target_mask"].to(args.device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=args.amp):
                    pred = model({
                        "image_curr": img_c, "image_prev_rot": img_p,
                        "ego_5d": ego5, "traj_30": traj,
                    })
                    total, parts = split_losses(pred, tgt, mask, args)
                B = img_c.size(0); vl_total += total.item()*B; vn += B
                if parts:
                    vsteer += parts["l1_steer"].item()*B
                    vax    += parts["l1_ax"].item()*B
        val_loss = vl_total / max(1, vn)
        dt = time.time() - t0
        logp(f"[ep {ep+1:02d}] train={train_loss:.4f}  val={val_loss:.4f} "
             f"(steer_n={vsteer/vn:.4f} ax_n={vax/vn:.4f}; un-norm: steer={vsteer/vn*args.scale_steer:.2f}°  ax={vax/vn*args.scale_ax:.2f}m/s²)  time={dt:.1f}s")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(out_dir, "best_loss.pt"))
            logp(f"  -> saved best val_loss ({val_loss:.4f})")

    logp(f"\n[done] best_val_loss={best_loss:.4f}")
    log_f.close()


if __name__ == "__main__":
    main()
