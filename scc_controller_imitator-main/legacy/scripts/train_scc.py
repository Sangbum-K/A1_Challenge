"""SCC controller imitator — baseline training loop.

Usage:
  ~/miniforge3/envs/ml_con/bin/python scripts/train_scc.py \
    --data-dir data/scc_synth_k100_v1 --prefixes fwd50 \
    --epochs 30 --batch-size 16 --lr 2e-4

Saves to outputs_multi/scc_<run_name>/.
"""
from __future__ import annotations
import os, sys, time, argparse, json
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.datamodule.scc_dataset import SCCSynthDataset, collate_scc
from src.model.image_models.image_scc_models import SCCControllerImitator
from src.model.image_models.image_scc_models_v2 import SCCControllerImitatorV2


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/scc_synth_k100_v1")
    ap.add_argument("--prefixes", nargs="+", default=["fwd50"])
    ap.add_argument("--out-dir", default="outputs_multi/scc_baseline")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--img-backbone", default="efficientnet_b0")
    ap.add_argument("--d-token", type=int, default=256)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-decoder-layers", type=int, default=2)
    ap.add_argument("--model", choices=["v1", "v2"], default="v2",
                    help="v1=our transformer, v2=controller_imitator-compatible + image stream")
    ap.add_argument("--pretrained-ckpt", default=None,
                    help="controller_imitator .pth checkpoint (v2 모델일 때만 사용)")
    ap.add_argument("--steer-weight", type=float, default=1.0)
    ap.add_argument("--accel-weight", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def _seed_all(seed: int):
    import random as _r
    _r.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _masked_smooth_l1(pred, target, mask):
    """pred/target (B, T, *), mask (B, T) bool. Returns scalar mean over True positions."""
    diff = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none")
    if diff.dim() > mask.dim():
        # broadcast mask to last dim if present
        m = mask.unsqueeze(-1).expand_as(diff)
    else:
        m = mask
    n = m.float().sum().clamp_min(1.0)
    return (diff * m.float()).sum() / n


def loop(model, loader, optim, device, weights, train=True, log_every=20):
    model.train() if train else model.eval()
    total = 0.0
    total_steer = 0.0
    total_accel = 0.0
    n = 0
    t0 = time.time()
    for i, batch in enumerate(loader):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            pred = model(batch)                  # (B, 40, 2)
            target = batch["target"]             # (B, 40, 2)
            mask = batch["target_mask"]          # (B, 40) bool
            loss_steer = _masked_smooth_l1(pred[..., 0], target[..., 0], mask)
            loss_accel = _masked_smooth_l1(pred[..., 1], target[..., 1], mask)
            loss = weights[0] * loss_steer + weights[1] * loss_accel
        if train:
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
        B = target.shape[0]
        total += float(loss.item()) * B
        total_steer += float(loss_steer.item()) * B
        total_accel += float(loss_accel.item()) * B
        n += B
        if (i + 1) % log_every == 0:
            rate = (i + 1) * loader.batch_size / max(1e-3, time.time() - t0)
            print(f"  iter {i+1}/{len(loader)} loss={loss.item():.4f} (steer={loss_steer.item():.4f} "
                  f"accel={loss_accel.item():.4f}) rate={rate:.1f}/s")
    return total / max(1, n), total_steer / max(1, n), total_accel / max(1, n)


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    _seed_all(args.seed)
    print(f"args:\n{json.dumps(vars(args), indent=2)}")

    print("loading datasets ...")
    train_ds = SCCSynthDataset(args.data_dir, tuple(args.prefixes),
                                split="train", val_ratio=args.val_ratio)
    val_ds = SCCSynthDataset(args.data_dir, tuple(args.prefixes),
                              split="val", val_ratio=args.val_ratio)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.workers, collate_fn=collate_scc,
                                pin_memory=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, collate_fn=collate_scc,
                              pin_memory=True, persistent_workers=args.workers > 0)

    print(f"build model: {args.model} {args.img_backbone} d={args.d_token}")
    if args.model == "v2":
        model = SCCControllerImitatorV2(img_backbone=args.img_backbone).to(args.device)
        if args.pretrained_ckpt:
            model.load_controller_imitator_pretrained(args.pretrained_ckpt)
    else:
        model = SCCControllerImitator(
            img_backbone=args.img_backbone,
            d_token=args.d_token, n_heads=args.n_heads,
            n_decoder_layers=args.n_decoder_layers).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params/1e6:.2f}M")

    # foreach=False 로 PyTorch 2.4 의 _foreach_sqrt CUDAGuard 버그 회피 (epoch 5+ 에서 crash).
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay, foreach=False)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    weights = (args.steer_weight, args.accel_weight)
    best_val = float("inf")
    history = []
    for ep in range(args.epochs):
        t_ep = time.time()
        print(f"\n== epoch {ep+1}/{args.epochs} ==")
        tr_loss, tr_st, tr_ac = loop(model, train_loader, optim, args.device, weights,
                                       train=True, log_every=args.log_every)
        with torch.no_grad():
            vl_loss, vl_st, vl_ac = loop(model, val_loader, optim, args.device, weights,
                                           train=False, log_every=args.log_every * 5)
        sched.step()
        dt = time.time() - t_ep
        line = (f"epoch {ep+1}: train loss={tr_loss:.4f} (st={tr_st:.4f} ac={tr_ac:.4f}) "
                f"val={vl_loss:.4f} (st={vl_st:.4f} ac={vl_ac:.4f}) {dt:.0f}s")
        print(line)
        history.append({"epoch": ep + 1, "tr": tr_loss, "vl": vl_loss,
                        "tr_st": tr_st, "tr_ac": tr_ac, "vl_st": vl_st, "vl_ac": vl_ac})
        with open(os.path.join(args.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        if vl_loss < best_val:
            best_val = vl_loss
            torch.save({"model": model.state_dict(),
                          "args": vars(args), "epoch": ep + 1,
                          "best_val": best_val},
                         os.path.join(args.out_dir, "best.pt"))
            print(f"  -> saved best (val={best_val:.4f})")
        if (ep + 1) % args.save_every == 0:
            torch.save({"model": model.state_dict(), "epoch": ep + 1},
                         os.path.join(args.out_dir, f"ckpt_ep{ep+1:03d}.pt"))
    print(f"\nDone. best val: {best_val:.4f}")


if __name__ == "__main__":
    main()
