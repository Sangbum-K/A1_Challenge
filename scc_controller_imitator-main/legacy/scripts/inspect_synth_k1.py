"""Phase 1 K=1 합성 결과 검증.

- 통계: GT steer/accel 분포, ego speed, trajectory shape, lead 유무
- 시각화: 5 sample 의 GT control 시퀀스 + ego/lead 위치 plot
- Image path 존재 확인 sanity check
"""
import pickle, glob, os, numpy as np

DIR = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/scc_synth_v1"


def load_all():
    samples = []
    for fp in sorted(glob.glob(os.path.join(DIR, "scc_synth_k1_v1_chunk_*.pkl"))):
        with open(fp, "rb") as f:
            d = pickle.load(f)
        samples.extend(d["samples"])
    return samples


def main():
    samples = load_all()
    print(f"=== Total samples: {len(samples)} ===\n")

    # 분포 통계
    egos = np.array([s["ego_5d"] for s in samples])
    trajs = np.array([s["traj_30"] for s in samples])
    gt_steer = np.array([s["gt_steer_30"] for s in samples])
    gt_accel = np.array([s["gt_accel_30"] for s in samples])
    leads_valid = np.array([s["lead_valid"] for s in samples])

    print(f"ego_5d shape={egos.shape}")
    print(f"  vx        : mean={egos[:,0].mean():.2f}, range=[{egos[:,0].min():.2f}, {egos[:,0].max():.2f}]")
    print(f"  vy        : mean={egos[:,1].mean():.3f}, range=[{egos[:,1].min():.3f}, {egos[:,1].max():.3f}]")
    print(f"  ax_est    : mean={egos[:,2].mean():.2f}, range=[{egos[:,2].min():.2f}, {egos[:,2].max():.2f}]")
    print(f"  yaw_rate  : mean={egos[:,4].mean():.3f}, range=[{egos[:,4].min():.3f}, {egos[:,4].max():.3f}]")

    print(f"\ntraj_30 shape={trajs.shape}")
    print(f"  traj_x[last]   range=[{trajs[:,-1,0].min():.1f}, {trajs[:,-1,0].max():.1f}]")
    print(f"  traj_speed[mean]: range=[{trajs[:,:,3].mean(axis=1).min():.2f}, {trajs[:,:,3].mean(axis=1).max():.2f}]")

    print(f"\ngt_steer_30 (deg) shape={gt_steer.shape}")
    print(f"  steer[0]  : mean={gt_steer[:,0].mean():.2f}, std={gt_steer[:,0].std():.2f}, range=[{gt_steer[:,0].min():.1f}, {gt_steer[:,0].max():.1f}]")
    print(f"  steer all : mean={gt_steer.mean():.2f}, abs_max={np.abs(gt_steer).max():.2f}")
    print(f"  steer p95 : {np.percentile(np.abs(gt_steer), 95):.2f} deg")

    print(f"\ngt_accel_30 (m/s²) shape={gt_accel.shape}")
    print(f"  accel[0]  : mean={gt_accel[:,0].mean():.2f}, std={gt_accel[:,0].std():.2f}, range=[{gt_accel[:,0].min():.1f}, {gt_accel[:,0].max():.1f}]")
    print(f"  accel all : mean={gt_accel.mean():.2f}, abs_max={np.abs(gt_accel).max():.2f}")
    print(f"  accel p95 : {np.percentile(np.abs(gt_accel), 95):.2f} m/s²")

    n_with_lead = int(leads_valid.sum())
    print(f"\nLead car valid: {n_with_lead}/{len(samples)} ({100*n_with_lead/len(samples):.1f}%)")

    # Outlier check: accel |abs| > 30 m/s²
    n_outlier = int((np.abs(gt_accel).max(axis=1) > 30).sum())
    print(f"  accel outliers (|abs|>30 m/s²): {n_outlier}/{len(samples)} ({100*n_outlier/len(samples):.1f}%)")
    n_outlier_steer = int((np.abs(gt_steer).max(axis=1) > 30).sum())
    print(f"  steer outliers (|abs|>30 deg): {n_outlier_steer}/{len(samples)} ({100*n_outlier_steer/len(samples):.1f}%)")

    # Sample image path existence
    n_check = min(20, len(samples))
    print(f"\nImage path check ({n_check} random samples):")
    rng = np.random.RandomState(0)
    miss = 0
    for idx in rng.choice(len(samples), n_check, replace=False):
        s = samples[int(idx)]
        c_ok = os.path.exists(s["image_curr_path"])
        p_ok = os.path.exists(s["image_prev_path"])
        if not (c_ok and p_ok):
            miss += 1
    print(f"  {n_check - miss}/{n_check} both images exist")

    # 5 sample 디테일 출력 (with lead 우선)
    print(f"\n=== Detail: 5 samples (lead present 우선) ===")
    leads_idx = [i for i, s in enumerate(samples) if s["lead_valid"]]
    others_idx = [i for i, s in enumerate(samples) if not s["lead_valid"]]
    picks = (leads_idx[:3] if len(leads_idx) >= 3 else leads_idx) + others_idx[:5 - min(3, len(leads_idx))]
    for j, i in enumerate(picks):
        s = samples[i]
        print(f"\n--- [{j}] frame_id={s['frame_id']} lead={s['lead_valid']}")
        print(f"    ego={s['ego_5d']}")
        if s["lead_valid"]:
            print(f"    lead_info (x,y,vx,vy,L,W)={s['lead_info']}")
        print(f"    traj_first 3: {s['traj_30'][:3]}")
        print(f"    gt_steer[:8]: {np.round(s['gt_steer_30'][:8], 2)}")
        print(f"    gt_accel[:8]: {np.round(s['gt_accel_30'][:8], 2)}")

    # Visualization (matplotlib if available)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        # (0,0) steer[0] hist
        axes[0, 0].hist(gt_steer[:, 0], bins=80, color="C0")
        axes[0, 0].set_title("GT steer[0] (deg)")
        axes[0, 0].axvline(0, color='k', lw=0.5)
        # (0,1) accel[0] hist
        axes[0, 1].hist(gt_accel[:, 0], bins=80, color="C1")
        axes[0, 1].set_title("GT accel[0] (m/s²)")
        axes[0, 1].axvline(0, color='k', lw=0.5)
        # (0,2) ego vx vs accel scatter
        axes[0, 2].scatter(egos[:, 0], gt_accel[:, 5], s=3, alpha=0.4,
                            c=leads_valid.astype(int), cmap="coolwarm")
        axes[0, 2].set_xlabel("ego vx (m/s)")
        axes[0, 2].set_ylabel("GT accel[5] (~0.5s ahead)")
        axes[0, 2].set_title("vx vs accel (blue=no lead, red=lead)")
        # (1,0) sample steer sequences (with lead)
        for i in (leads_idx[:8] if leads_idx else []):
            axes[1, 0].plot(np.arange(30) * 0.1, samples[i]["gt_steer_30"], lw=1, alpha=0.6)
        axes[1, 0].set_title("GT steer sequences (lead samples, 8)")
        axes[1, 0].set_xlabel("time (s)"); axes[1, 0].set_ylabel("steer (deg)")
        axes[1, 0].grid(True)
        # (1,1) sample accel sequences (with lead)
        for i in (leads_idx[:8] if leads_idx else []):
            axes[1, 1].plot(np.arange(30) * 0.1, samples[i]["gt_accel_30"], lw=1, alpha=0.6)
        axes[1, 1].set_title("GT accel sequences (lead samples, 8)")
        axes[1, 1].set_xlabel("time (s)"); axes[1, 1].set_ylabel("accel (m/s²)")
        axes[1, 1].grid(True)
        # (1,2) ego vx hist
        axes[1, 2].hist(egos[:, 0], bins=60, color="C2")
        axes[1, 2].set_title("ego vx distribution")
        axes[1, 2].set_xlabel("vx (m/s)")
        fig.tight_layout()
        outp = os.path.join(DIR, "inspect_k1.png")
        fig.savefig(outp, dpi=110)
        print(f"\nsaved visualization: {outp}")
    except Exception as e:
        print(f"\nviz skipped: {e}")


if __name__ == "__main__":
    main()
