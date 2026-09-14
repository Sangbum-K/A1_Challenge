"""f_1 학습 (cached fi_mem 버전) — 이미지 인코더 forward 생략.

학습 가능 부분:
  - traj_embed, state_embed, pos_encoder.pe (Professor pretrain)
  - transformer_encoder (Professor pretrain, finetune)
  - fc_out (Professor pretrain)
  - image_pos_emb (random init, train)

이미지 부분 (i_1 + i_2) 은 cache 시점에 frozen — 학습 안함.

사용:
  python scripts/train_f1_cached.py \
    --professor-ckpt /path/to/654/model_*.pth \
    --cache data/fi_mem_cache_v1.npz \
    --epochs 30 --batch-size 64
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

from src.datamodule.scc_synth_cached_dataset import SCCSynthCachedDataset, collate_cached
from src.datamodule.scc_synth_prebake_dataset import SCCSynthPrebakedDataset, collate_prebaked
from src.model.image_models.image_scc_models_v2 import (
    PositionalEncoding,
)


class CachedF1Model(nn.Module):
    """traj/state embed + Professor transformer + fc_out — image fusion 은 cached fi_mem.

    forward 입력: dict with traj_40 (B, 40, 3), ego_5d (B, 5 or 6), fi_mem (B, 40, D).
    """
    def __init__(self, d_model=128, nhead=8, dim_feedforward=512, num_layers=6,
                 dropout=0.1, traj_seq_len=40, traj_dim=3, state_dim=6, num_out=2):
        super().__init__()
        self.d_model = d_model
        self.traj_seq_len = traj_seq_len
        self.traj_dim = traj_dim; self.state_dim = state_dim
        self.traj_embed = nn.Linear(traj_dim, d_model)
        self.state_embed = nn.Linear(state_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout, max_len=traj_seq_len + 1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, num_out)
        self.image_pos_emb = nn.Parameter(torch.zeros(40, d_model))
        nn.init.trunc_normal_(self.image_pos_emb, std=0.02)

    def forward(self, b):
        B = b["traj_40"].size(0)
        traj = b["traj_40"][:, :, :self.traj_dim]
        state = b["ego_5d"][:, :self.state_dim]
        if state.shape[-1] < self.state_dim:
            pad = torch.zeros(B, self.state_dim - state.shape[-1],
                              dtype=state.dtype, device=state.device)
            state = torch.cat([state, pad], dim=-1)
        traj_tok = self.traj_embed(traj)
        state_tok = self.state_embed(state).unsqueeze(1)
        tokens_text = self.pos_encoder(traj_tok + state_tok)

        image_mem = b["fi_mem"] + self.image_pos_emb     # (B, 40, D)
        tokens = torch.cat([tokens_text, image_mem], dim=1)
        encoded = self.transformer_encoder(tokens)
        ctrl_tokens = encoded[:, :self.traj_seq_len]
        return self.fc_out(ctrl_tokens)

    def load_professor(self, ckpt_path: str):
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
        my_sd = self.state_dict()
        loaded = {}; skipped = []
        for k, v in sd.items():
            if k in my_sd and my_sd[k].shape == v.shape:
                loaded[k] = v
            elif k == "pos_encoder.pe":
                if k in my_sd and my_sd[k].shape[1] >= v.shape[1]:
                    new = my_sd[k].clone()
                    new[:, :v.shape[1]] = v
                    loaded[k] = new
            else:
                skipped.append((k, tuple(v.shape), tuple(my_sd[k].shape) if k in my_sd else None))
        self.load_state_dict(loaded, strict=False)
        return loaded, skipped


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--professor-ckpt", default=None)
    ap.add_argument("--cache", default="/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/fi_mem_cache_v1.npz")
    ap.add_argument("--prebake", default="/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/f1_prebake_v1.npz",
                    help="pre-baked dataset (traj+target+ego). prebake 사용 시 fast in-memory path")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-prof", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default="outputs_multi/f1_cached")
    ap.add_argument("--name", default="v1")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--scale-steer", type=float, default=10.0)
    ap.add_argument("--scale-ax",    type=float, default=10.0)
    # wandb
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="scc-controller-imitator")
    ap.add_argument("--wandb-entity",  default=None)
    ap.add_argument("--wandb-run",     default=None, help="run 이름. None 이면 args.name")
    ap.add_argument("--viz-every", type=int, default=2)
    ap.add_argument("--n-viz", type=int, default=6)
    # resume
    ap.add_argument("--resume-from", default=None,
                    help="ckpt path (.pt). model weights 만 warm-start (optim/sched 는 새로)")
    return ap.parse_args()


def build_viz_figure(model, viz_batch, device, args):
    """val 의 fixed sample N 개에 대해 pred vs GT steer/ax 곡선 plot.
    return: matplotlib.Figure
    """
    import matplotlib.pyplot as plt
    model.eval()
    b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in viz_batch.items()}
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=args.amp):
        pred = model(b).float().cpu().numpy()
    tgt = b["target"].cpu().numpy()
    msk = b["target_mask"].cpu().numpy()
    N = pred.shape[0]
    ts = np.arange(40) * 0.025
    fig, axes = plt.subplots(N, 2, figsize=(11, 1.8*N), squeeze=False)
    for i in range(N):
        m = msk[i]
        ax = axes[i, 0]
        ax.plot(ts[m], tgt[i, m, 0], 'k-', lw=1.2, label='GT')
        ax.plot(ts[m], pred[i, m, 0], 'r--', lw=1.1, label='pred')
        ax.set_xlabel('t (s)'); ax.set_ylabel('steer (deg)')
        ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
        ax.set_title(f"s{i}: steer", fontsize=9)
        ax = axes[i, 1]
        ax.plot(ts[m], tgt[i, m, 1], 'k-', lw=1.2, label='GT')
        ax.plot(ts[m], pred[i, m, 1], 'r--', lw=1.1, label='pred')
        ax.set_xlabel('t (s)'); ax.set_ylabel('ax (m/s²)')
        ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
        ax.set_title(f"s{i}: ax", fontsize=9)
    plt.tight_layout()
    return fig


def split_losses(pred, target, mask, args):
    if not mask.any():
        return torch.zeros((), device=pred.device), {}
    scales = torch.tensor([args.scale_steer, args.scale_ax], device=pred.device)
    diff = (pred - target) / scales
    l1_steer = diff[..., 0][mask].abs().mean()
    l1_ax    = diff[..., 1][mask].abs().mean()
    return l1_steer + l1_ax, {"l1_steer": l1_steer.detach(), "l1_ax": l1_ax.detach()}


def main():
    args = get_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out_dir = f"{args.out_dir}_{args.name}"
    os.makedirs(out_dir, exist_ok=True)

    # prebake 우선, 없으면 on-the-fly cached dataset
    if args.prebake and os.path.exists(args.prebake):
        full = SCCSynthPrebakedDataset(prebake_path=args.prebake, cache_path=args.cache, limit=args.limit)
        collate_fn = collate_prebaked
    else:
        full = SCCSynthCachedDataset(cache_path=args.cache, limit=args.limit)
        collate_fn = collate_cached
    n_total = len(full); n_val = max(1, int(n_total * args.val_ratio))
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(args.seed)
    tr, va = random_split(full, [n_train, n_val], generator=g)
    print(f"[data] total={n_total} train={n_train} val={n_val}")

    tl = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=True, collate_fn=collate_fn,
                    drop_last=True, persistent_workers=args.workers > 0)
    vl = DataLoader(va, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True, collate_fn=collate_fn,
                    persistent_workers=args.workers > 0)

    model = CachedF1Model(d_model=args.d_model).to(args.device)
    if args.professor_ckpt:
        loaded, skipped = model.load_professor(args.professor_ckpt)
        print(f"[Professor] loaded={len(loaded)} skipped={len(skipped)}")
    if args.resume_from:
        sd = torch.load(args.resume_from, map_location=args.device, weights_only=False)
        if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
        miss, unex = model.load_state_dict(sd, strict=False)
        print(f"[resume] loaded {args.resume_from}  missing={len(miss)} unexpected={len(unex)}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params/1e6:.2f} M")

    # parameter groups: Professor (transformer 등) 낮은 LR, image_pos_emb 일반 LR
    prof_keys = ["traj_embed", "state_embed", "pos_encoder", "transformer_encoder", "fc_out"]
    prof_params = []; other_params = []
    for n, p in model.named_parameters():
        (prof_params if any(n.startswith(k) for k in prof_keys) else other_params).append(p)
    optim = torch.optim.AdamW([
        {"params": prof_params, "lr": args.lr_prof},
        {"params": other_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)
    # 5 cycle Cosine Warm Restarts (T_0 = epochs/5)
    sched_T0 = max(1, args.epochs // 5)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optim, T_0=sched_T0, T_mult=1, eta_min=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    log_path = os.path.join(out_dir, "train.log")
    log_f = open(log_path, "w")
    def logp(m): print(m, flush=True); log_f.write(m + "\n"); log_f.flush()
    logp(f"args={vars(args)}  prof_params={len(prof_params)} other={len(other_params)}")

    # wandb
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=args.wandb_run or args.name, config=vars(args), dir=out_dir,
        )
        logp(f"[wandb] init project={args.wandb_project} run={wandb_run.name}")

    # viz batch (fixed: val 의 처음 n_viz)
    viz_batch = None
    if args.wandb and args.viz_every > 0:
        # collect a fixed val batch deterministically
        from torch.utils.data import DataLoader as _DL
        seq = _DL(va, batch_size=args.n_viz, shuffle=False,
                  num_workers=0, collate_fn=collate_fn)
        viz_batch = next(iter(seq))

    # grad norm 그룹 분류
    grad_group_keys = {
        "traj_embed": "traj_embed", "state_embed": "state_embed",
        "pos_encoder": "pos_encoder",
        "tfr.in_proj": "in_proj", "tfr.out_proj": "out_proj",
        "tfr.linear1": "linear1", "tfr.linear2": "linear2",
        "tfr.norm": "norm", "fc_out": "fc_out", "image_pos_emb": "image_pos_emb",
    }
    def collect_grad_norms():
        out = {k: 0.0 for k in grad_group_keys}; total_sq = 0.0
        layer_norms = {f"tfr.layer{L}": 0.0 for L in range(args.num_layers if hasattr(args, 'num_layers') else 6)}
        for n, p in model.named_parameters():
            if p.grad is None: continue
            g2 = p.grad.detach().pow(2).sum().item()
            total_sq += g2
            for grp, key in grad_group_keys.items():
                if key in n:
                    out[grp] += g2; break
            for L in range(6):
                if f"transformer_encoder.layers.{L}." in n:
                    layer_norms[f"tfr.layer{L}"] += g2; break
        out = {k: v**0.5 for k, v in out.items()}
        layer_norms = {k: v**0.5 for k, v in layer_norms.items()}
        out["TOTAL"] = total_sq ** 0.5
        out.update(layer_norms)
        return out

    if args.wandb:
        import wandb
        wandb.watch(model, log="gradients", log_freq=500)

    best_val = float("inf")
    for ep in range(args.epochs):
        model.train(); t0 = time.time()
        rl = rs = ra = 0.0; rn = 0
        # epoch 평균 grad norm 누적
        gn_sum = {k: 0.0 for k in list(grad_group_keys) + [f"tfr.layer{L}" for L in range(6)] + ["TOTAL"]}
        gn_n = 0
        for it, batch in enumerate(tl):
            b = {k: (v.to(args.device, non_blocking=True) if torch.is_tensor(v) else v)
                  for k, v in batch.items()}
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                pred = model(b)
                total, parts = split_losses(pred, b["target"], b["target_mask"], args)
            scaler.scale(total).backward()
            # grad norm 측정 (unscaled — scaler 적용 전 측정 위해 unscale)
            scaler.unscale_(optim)
            gns = collect_grad_norms()
            for k, v in gns.items(): gn_sum[k] += v
            gn_n += 1
            scaler.step(optim); scaler.update()
            B = b["traj_40"].size(0)
            rl += total.item() * B
            if parts:
                rs += parts["l1_steer"].item() * B
                ra += parts["l1_ax"].item() * B
            rn += B
            if (it + 1) % args.log_every == 0:
                logp(f"[ep {ep+1:02d} | {it+1:4d}/{len(tl)}] total={rl/rn:.4f}  "
                     f"l1_steer={rs/rn:.4f} l1_ax={ra/rn:.4f}  "
                     f"lr_prof={optim.param_groups[0]['lr']:.2e}")
        sched.step()
        train_loss = rl / max(1, rn)

        model.eval()
        vl_total = vs = va = 0.0; vn = 0
        with torch.no_grad():
            for batch in vl:
                b = {k: (v.to(args.device, non_blocking=True) if torch.is_tensor(v) else v)
                      for k, v in batch.items()}
                with torch.amp.autocast("cuda", enabled=args.amp):
                    pred = model(b)
                    total, parts = split_losses(pred, b["target"], b["target_mask"], args)
                B = b["traj_40"].size(0)
                vl_total += total.item()*B; vn += B
                if parts:
                    vs += parts["l1_steer"].item()*B; va += parts["l1_ax"].item()*B
        val_loss = vl_total / max(1, vn)
        dt = time.time() - t0
        logp(f"[ep {ep+1:02d}] train={train_loss:.4f} val={val_loss:.4f} "
             f"(steer={vs/vn*args.scale_steer:.2f}° ax={va/vn*args.scale_ax:.2f}m/s²)  time={dt:.1f}s")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(out_dir, "best_loss.pt"))
            logp(f"  -> saved best val_loss ({val_loss:.4f})")

        if wandb_run is not None:
            log = {
                "train/loss": train_loss, "val/loss": val_loss,
                "val/l1_steer_deg": vs/vn*args.scale_steer,
                "val/l1_ax_mps2":   va/vn*args.scale_ax,
                "lr/prof": optim.param_groups[0]['lr'],
                "lr/other": optim.param_groups[1]['lr'],
                "epoch": ep + 1, "epoch_time_s": dt,
            }
            # avg grad norms (epoch)
            for k, v in gn_sum.items():
                log[f"grad/{k}"] = v / max(gn_n, 1)
            if (ep + 1) % args.viz_every == 0 and viz_batch is not None:
                fig = build_viz_figure(model, viz_batch, args.device, args)
                log["viz/pred_vs_gt"] = wandb.Image(fig)
                import matplotlib.pyplot as plt
                plt.close(fig)
            wandb_run.log(log, step=ep + 1)

    logp(f"\n[done] best_val_loss={best_val:.4f}")
    log_f.close()
    if wandb_run is not None: wandb_run.finish()


if __name__ == "__main__":
    main()
