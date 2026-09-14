"""vinject f_2 delta(보정) 를 (거리 × closing) 평면에서 — 거리별 슬라이스 + 3D.

우려: lead 있어도 cruising/receding 이면 보정 0 이어야 하는데 f_2 가 양(+) delta(가속)로
간섭("앞차 가속 따라가기")하지 않는지. delta>0 영역(가속 보정)을 거리별로 점검.
vinject ckpt 는 미정규화 학습 → gt_vrel = closing*10 보정 주입.
"""
import os, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.image_models.prof635_side_model import Prof635SideModel
from src.datamodule.prof635_dataset import build_traj6, build_ego6
D = os.environ.get("SCC_DATA_DIR") or "/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data"


def gather():
    rows = []
    for ds, pbf in [("sangam_lead_0520", "f1_prebake_iy_sangam_lead_0520_v1_brakeaug_prev20.npz"),
                    ("sangam_lead_0521", "f1_prebake_iy_sangam_lead_0521_v1_brakeaug_prev20.npz"),
                    ("DriverB_new", "f1_prebake_iy_DriverB_new_logged_fixed_prev20_trajfix.npz")]:
        pb = np.load(f"{D}/prebake/{pbf}", allow_pickle=True); ck = pb["curr_keys"].tolist(); tr = pb["traj"]; eg = pb["ego_5d"]
        z = np.load(f"{D}/cache/i2_mem_cache_{ds}_v7b_prev20.npz", allow_pickle=True)
        k2i = {k: i for i, k in enumerate(z["keys"].tolist())}; i2a = z["i2_mem"]; a2a = z["a2_logits"]
        zg = np.load(f"{D}/cache/inlane_gt_{ds}.npz", allow_pickle=True); zk = {k: i for i, k in enumerate(zg["keys"].tolist())}
        gd = zg["inlane_dist"]; gc = zg["inlane_closing"]
        for i, k in enumerate(ck):
            gi = zk.get(k); ii = k2i.get(k)
            if gi is None or ii is None or not (np.isfinite(gd[gi]) and gd[gi] < 60): continue
            if np.abs(i2a[ii].astype(np.float32)).sum() < 1e-6: continue
            rows.append((tr[i], eg[i], i2a[ii].astype(np.float32), a2a[ii].astype(np.float32), float(gc[gi]), float(gd[gi])))
    return rows


def main():
    rows = gather(); print(f"in-lane lead frames N={len(rows)}", flush=True)
    m = Prof635SideModel(residual_gate=True, use_traj_in_side=False)
    m.load_state_dict(torch.load("outputs_multi/prof635_side_vinject_gtvrel/best.pt", map_location="cpu", weights_only=False), strict=False); m.eval()
    T6 = torch.tensor(np.stack([build_traj6(r[0], mode="speed") for r in rows]))
    E6 = torch.tensor(np.stack([build_ego6(r[1]) for r in rows]))
    I2 = torch.tensor(np.stack([r[2] for r in rows])); A2 = torch.tensor(np.stack([r[3] for r in rows]))
    clos = np.array([r[4] for r in rows], np.float32); dist = np.array([r[5] for r in rows], np.float32)
    with torch.no_grad():
        m({"traj6": T6, "ego6": E6, "i2_mem": I2, "a2_logits": A2, "gt_vrel": torch.tensor(clos * 10.0)})
    delta = m.last_delta.numpy()

    dbins = [(0, 10), (10, 20), (20, 40), (40, 60)]
    cbins = [(-30, -8), (-8, -3), (-3, -1), (-1, 1), (1, 3), (3, 8), (8, 30)]
    # Fig A: 거리별 패널 delta vs closing
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (dlo, dhi) in zip(axes.flat, dbins):
        dm = (dist >= dlo) & (dist < dhi)
        ax.axhspan(0, 2.2, color="#ffe8e8", alpha=0.6)  # delta>0 = 가속 보정(주의)
        ax.scatter(clos[dm], delta[dm], s=4, alpha=0.15, color="#005195")
        cc, mm = [], []
        for lo, hi in cbins:
            sel = dm & (clos >= lo) & (clos < hi)
            if sel.sum() < 20: continue
            cc.append((lo + hi) / 2); mm.append(delta[sel].mean())
        ax.plot(cc, mm, "o-", color="#DC291E", lw=2, label="mean")
        ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="k", lw=0.5, ls=":")
        # 우려 지표: non-approaching(closing<1) 에서 delta>0.1 (가속 간섭) 비율
        na = dm & (clos < 1)
        spur = 100 * (delta[na] > 0.1).mean() if na.sum() else 0
        ax.set_title(f"dist {dlo}-{dhi}m (N={dm.sum()}) | non-approach delta>+0.1: {spur:.0f}%")
        ax.set_xlabel("closing (m/s)"); ax.set_ylabel("f_2 delta"); ax.set_ylim(-2.1, 2.1); ax.grid(alpha=0.3)
    fig.suptitle("vinject f_2 delta — 거리별 (분홍=delta>0 가속보정 영역). 우려: cruising/receding(closing<1)에서 delta>0 간섭", fontsize=12)
    plt.tight_layout(); plt.savefig("outputs_multi/viz/vinject_delta_by_distance.png", dpi=110); plt.close()

    # Fig B: 3D (dist, closing, delta)
    fig = plt.figure(figsize=(11, 8)); ax = fig.add_subplot(111, projection="3d")
    idx = np.random.RandomState(0).choice(len(rows), min(6000, len(rows)), replace=False)
    sc = ax.scatter(dist[idx], clos[idx], delta[idx], c=delta[idx], cmap="RdBu", s=6, alpha=0.4, vmin=-1, vmax=1)
    ax.set_xlabel("distance (m)"); ax.set_ylabel("closing (m/s)"); ax.set_zlabel("f_2 delta")
    ax.set_title("vinject f_2 delta (dist × closing × delta)\n빨강=brake, 파랑=accel 보정")
    fig.colorbar(sc, shrink=0.5, label="delta")
    plt.tight_layout(); plt.savefig("outputs_multi/viz/vinject_delta_3d.png", dpi=110); plt.close()

    # 우려 정량
    na = clos < 1
    print(f"[우려 점검] non-approaching lead(closing<1) N={na.sum()}: "
          f"delta>+0.1(가속간섭) {100*(delta[na]>0.1).mean():.0f}%, "
          f"delta mean={delta[na].mean():+.3f}", flush=True)
    print("[done] vinject_delta_by_distance.png, vinject_delta_3d.png", flush=True)


if __name__ == "__main__":
    main()
