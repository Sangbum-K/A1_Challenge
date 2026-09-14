"""f_1 cached 학습 결과 평가 + 정성 viz.

평가:
  val_set 의 N frame 에 대해 predicted vs GT (steer, ax) 비교
  - 시간축 plot (각 sample: 40 timestep)
  - logged vs random trajectory 별 separation
  - 통계: per-step L1, instant (t=0) error

사용:
  python scripts/eval_f1_cached.py --ckpt outputs_multi/f1_cached_v1/best_loss.pt
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.datamodule.scc_synth_prebake_dataset import SCCSynthPrebakedDataset, collate_prebaked
from scripts.train_f1_cached import CachedF1Model


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prebake", default="/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/f1_prebake_v1.npz")
    ap.add_argument("--cache", default="/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/fi_mem_cache_v1.npz")
    ap.add_argument("--out", default="/tmp/f1_eval.png")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    args = get_args()
    print(f"[eval] loading {args.ckpt}")
    sd = torch.load(args.ckpt, map_location="cpu")
    model = CachedF1Model(d_model=128).to(args.device)
    model.load_state_dict(sd)
    model.eval()
    print("  model loaded")

    print(f"[data] loading prebake")
    full = SCCSynthPrebakedDataset(prebake_path=args.prebake, cache_path=args.cache)
    # validation 셋 sampling (deterministic split: last 5%)
    n_total = len(full)
    n_val = max(1, int(n_total * 0.05))
    rng = np.random.RandomState(args.seed)
    val_idx = rng.choice(n_total, size=n_val, replace=False)
    print(f"  total={n_total}, val={n_val}")

    # 통계: 전체 val 의 per-step L1
    dl = DataLoader(full, batch_size=args.batch_size, num_workers=4,
                     collate_fn=collate_prebaked,
                     sampler=torch.utils.data.SubsetRandomSampler(val_idx.tolist()))
    all_pred = []; all_tgt = []; all_mask = []
    with torch.no_grad():
        for batch in dl:
            b = {k: (v.to(args.device, non_blocking=True) if torch.is_tensor(v) else v)
                  for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=True):
                pred = model(b)
            all_pred.append(pred.float().cpu().numpy())
            all_tgt.append(b["target"].cpu().numpy())
            all_mask.append(b["target_mask"].cpu().numpy())
    pred_arr = np.concatenate(all_pred, axis=0)
    tgt_arr  = np.concatenate(all_tgt, axis=0)
    msk_arr  = np.concatenate(all_mask, axis=0)
    # per-step L1
    diff = np.abs(pred_arr - tgt_arr)
    diff[~msk_arr] = np.nan
    per_step_steer = np.nanmean(diff[:, :, 0], axis=0)
    per_step_ax    = np.nanmean(diff[:, :, 1], axis=0)
    print(f"\n[stats over {pred_arr.shape[0]} val samples]")
    print(f"  steer L1 (deg): mean={np.nanmean(per_step_steer):.3f}  "
           f"t=0: {per_step_steer[0]:.3f}  t=39: {per_step_steer[-1]:.3f}")
    print(f"  ax    L1 (m/s²): mean={np.nanmean(per_step_ax):.3f}  "
           f"t=0: {per_step_ax[0]:.3f}  t=39: {per_step_ax[-1]:.3f}")

    # ----- 정성 viz: N samples -----
    N = min(args.n_samples, pred_arr.shape[0])
    fig, axes = plt.subplots(N, 2, figsize=(14, 2.5 * N), squeeze=False)
    for i in range(N):
        idx = i  # 처음 N
        ax_s = axes[i, 0]; ax_a = axes[i, 1]
        ts = np.arange(40) * 0.025   # long_traj dt
        m = msk_arr[idx]
        ax_s.plot(ts[m], tgt_arr[idx, m, 0], 'k-', lw=1.5, label='GT')
        ax_s.plot(ts[m], pred_arr[idx, m, 0], 'r--', lw=1.2, label='pred')
        ax_s.set_xlabel("t (s)"); ax_s.set_ylabel("steer (deg)")
        ax_s.legend(fontsize=8); ax_s.grid(True, alpha=0.3)
        ax_s.set_title(f"sample {i}: steer", fontsize=9)
        ax_a.plot(ts[m], tgt_arr[idx, m, 1], 'k-', lw=1.5, label='GT')
        ax_a.plot(ts[m], pred_arr[idx, m, 1], 'r--', lw=1.2, label='pred')
        ax_a.set_xlabel("t (s)"); ax_a.set_ylabel("ax (m/s²)")
        ax_a.legend(fontsize=8); ax_a.grid(True, alpha=0.3)
        ax_a.set_title(f"sample {i}: ax", fontsize=9)
    plt.tight_layout()
    plt.savefig(args.out, dpi=85, bbox_inches="tight")
    print(f"\n[viz] saved → {args.out}")

    # per-step plot
    out_step = args.out.replace(".png", "_per_step.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(np.arange(40) * 0.025, per_step_steer, 'C0-o')
    axes[0].set_xlabel("t (s)"); axes[0].set_ylabel("L1 steer (deg)")
    axes[0].set_title("per-step steer error"); axes[0].grid(True, alpha=0.3)
    axes[1].plot(np.arange(40) * 0.025, per_step_ax, 'C3-o')
    axes[1].set_xlabel("t (s)"); axes[1].set_ylabel("L1 ax (m/s²)")
    axes[1].set_title("per-step ax error"); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_step, dpi=85, bbox_inches="tight")
    print(f"[viz] per-step → {out_step}")


if __name__ == "__main__":
    main()
