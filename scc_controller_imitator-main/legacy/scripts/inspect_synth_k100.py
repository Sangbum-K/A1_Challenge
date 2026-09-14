"""K=100 합성 데이터 quality 검증 + 시각화.

- lead 와 no-lead samples 통계 (분포, ego, gt control)
- 한 frame 의 K=100 trajectory 다양성 시각화 (top-down)
- frame 마다 다른 traj 에 대한 GT control diversity
"""
import pickle, os, glob, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

DIR = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/scc_synth_k100_v1"


def load(prefix):
    samples = []
    for fp in sorted(glob.glob(os.path.join(DIR, f"{prefix}_*.pkl"))):
        with open(fp, 'rb') as f: d = pickle.load(f)
        samples.extend(d['samples'])
    return samples


def main():
    lead = load("lead")
    nolead = load("nolead")
    print(f"lead: {len(lead)} samples, no-lead: {len(nolead)} samples")

    # Stats
    for name, S in [("LEAD", lead), ("NO-LEAD", nolead)]:
        if not S:
            continue
        egos = np.array([s["ego_5d"] for s in S])
        gt_st = np.array([s["gt_steer_30"] for s in S])
        gt_ac = np.array([s["gt_accel_30"] for s in S])
        print(f"\n=== {name} ===")
        print(f"  ego vx: mean={egos[:,0].mean():.2f}, range=[{egos[:,0].min():.2f}, {egos[:,0].max():.2f}]")
        print(f"  GT steer: p50={np.median(np.abs(gt_st)):.2f}, p95={np.percentile(np.abs(gt_st),95):.2f}, max={np.abs(gt_st).max():.1f}")
        print(f"  GT accel: p50={np.median(np.abs(gt_ac)):.2f}, p95={np.percentile(np.abs(gt_ac),95):.2f}, max={np.abs(gt_ac).max():.1f}")
        print(f"  GT accel mean[t=0]: {gt_ac[:,0].mean():.2f}")
        print(f"  GT accel mean (whole): {gt_ac.mean():.2f}")

    # frame 별 K trajectory diversity
    from collections import defaultdict
    by_frame = defaultdict(list)
    for s in lead:
        by_frame[s["frame_id"]].append(s)
    frame_counts = sorted([(fid, len(sl)) for fid, sl in by_frame.items()],
                          key=lambda x: -x[1])
    print(f"\nlead 가장 풍부한 frame top 5:")
    for fid, c in frame_counts[:5]:
        print(f"  {c} samples: {fid}")

    # 시각화: 가장 sample 많은 frame 의 K trajectory + GT control diversity
    if frame_counts:
        best_fid, best_n = frame_counts[0]
        samples_one = by_frame[best_fid]
        print(f"\nvisualizing {best_n} samples from {best_fid}")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        # (0) trajectory top-down
        ax0 = axes[0]
        for s in samples_one:
            traj = s["traj_30"]
            ax0.plot(traj[:, 0], traj[:, 1], lw=0.6, alpha=0.3, color='C0')
        # lead
        lead_info = samples_one[0]["lead_info"]
        if samples_one[0]["lead_valid"]:
            ax0.add_patch(Rectangle((lead_info[0] - lead_info[4]/2, lead_info[1] - lead_info[5]/2),
                                    lead_info[4], lead_info[5], ec='r', fc='salmon', alpha=0.5))
        ax0.add_patch(Rectangle((-2.5, -1), 5, 2, ec='k', fc='lightgray'))
        ax0.set_aspect("equal"); ax0.grid(True, alpha=0.3)
        ax0.set_xlim(-5, 80); ax0.set_ylim(-15, 15)
        ax0.set_title(f"{best_n} trajectory diversity (lead 빨강)")
        ax0.set_xlabel("x [m] forward"); ax0.set_ylabel("y [m] left")

        # (1) GT accel sequences
        ax1 = axes[1]
        for s in samples_one:
            ax1.plot(np.arange(30)*0.1, s["gt_accel_30"], lw=0.5, alpha=0.3, color='C1')
        ax1.set_title(f"GT accel diversity ({best_n} traj)")
        ax1.set_xlabel("time (s)"); ax1.set_ylabel("accel (m/s²)")
        ax1.grid(True, alpha=0.3)
        ax1.axhline(0, color='k', lw=0.5)

        # (2) GT steer sequences
        ax2 = axes[2]
        for s in samples_one:
            ax2.plot(np.arange(30)*0.1, s["gt_steer_30"], lw=0.5, alpha=0.3, color='C2')
        ax2.set_title(f"GT steer diversity ({best_n} traj)")
        ax2.set_xlabel("time (s)"); ax2.set_ylabel("steer (deg)")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        out_png = os.path.join(DIR, "inspect_k100_diversity.png")
        fig.savefig(out_png, dpi=110)
        print(f"saved: {out_png}")

    # 분포 시각화 — lead vs no-lead
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for j, (name, S) in enumerate([("LEAD", lead), ("NO-LEAD", nolead)]):
        if not S: continue
        egos = np.array([s["ego_5d"] for s in S])
        gt_st = np.array([s["gt_steer_30"] for s in S])
        gt_ac = np.array([s["gt_accel_30"] for s in S])
        axes[j, 0].hist(egos[:, 0], bins=50)
        axes[j, 0].set_title(f"{name}: ego vx (n={len(S)})")
        axes[j, 0].set_xlabel("vx (m/s)")
        axes[j, 1].hist(gt_st[:, 0], bins=80, range=(-30, 30))
        axes[j, 1].set_title(f"{name}: GT steer[0] (deg)")
        axes[j, 2].hist(gt_ac[:, 0], bins=80, range=(-30, 30))
        axes[j, 2].set_title(f"{name}: GT accel[0] (m/s²)")
        axes[j, 2].axvline(0, color='k', lw=0.5)
    plt.tight_layout()
    out_png2 = os.path.join(DIR, "inspect_k100_distributions.png")
    fig.savefig(out_png2, dpi=110)
    print(f"saved: {out_png2}")


if __name__ == "__main__":
    main()
