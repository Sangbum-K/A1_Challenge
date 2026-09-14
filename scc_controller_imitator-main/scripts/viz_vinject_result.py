"""vinject(GT 상대속도 주입) 결과 시각화: f_2 delta 가 closing 으로 modulate 하는가.

v5 (gt_vrel 없음, 현 배포) vs vinject (gt_vrel 주입) 의 delta-vs-closing 비교.
→ vinject 만 monotonic modulation 이면 "병목 = a_2 vL_rel 지각" 시각 증거.
주의: vinject ckpt 는 미정규화(±30 raw)로 학습됨. 현 모델 forward 는 gt_vrel*0.1 정규화 →
      보정 위해 gt_vrel = closing*10 주입 (→ *0.1 = closing raw, 학습 분포 일치).
"""
import os, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.image_models.prof635_side_model import Prof635SideModel
from src.datamodule.prof635_dataset import build_traj6, build_ego6
D = os.environ.get("SCC_DATA_DIR") or "/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data"
CK = "data/ckpts"


def load(ckpt):
    m = Prof635SideModel(residual_gate=True, use_traj_in_side=False)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False), strict=False)
    return m.eval()


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
            rows.append((tr[i], eg[i], i2a[ii].astype(np.float32), a2a[ii].astype(np.float32), float(gc[gi])))
    return rows


def deltas(model, rows, inject=False):
    """반환: final pedal (f_1 base + gate·delta, 실제 제어 출력), gtc."""
    T6 = torch.tensor(np.stack([build_traj6(r[0], mode="speed") for r in rows]))
    E6 = torch.tensor(np.stack([build_ego6(r[1]) for r in rows]))
    I2 = torch.tensor(np.stack([r[2] for r in rows])); A2 = torch.tensor(np.stack([r[3] for r in rows]))
    gtc = np.array([r[4] for r in rows], np.float32)
    b = {"traj6": T6, "ego6": E6, "i2_mem": I2, "a2_logits": A2}
    if inject:
        b["gt_vrel"] = torch.tensor(gtc * 10.0)  # 미정규화 ckpt 보정 (*0.1 → raw)
    with torch.no_grad():
        _, final = model(b)   # final pedal = clamp(base + gate·delta, -1, 1)
    return final.numpy(), gtc


def main():
    rows = gather()
    print(f"in-lane lead frames N={len(rows)}", flush=True)
    d_v5, gtc = deltas(load(f"{CK}/prof635_side_v5_best.pt"), rows, inject=False)
    d_inj, _ = deltas(load("outputs_multi/prof635_side_vinject_gtvrel/best.pt"), rows, inject=True)

    bins = [(-30, -8), (-8, -3), (-3, -1), (-1, 1), (1, 3), (3, 8), (8, 30)]
    centers, m_v5, s_v5, m_inj, s_inj = [], [], [], [], []
    for lo, hi in bins:
        mm = (gtc >= lo) & (gtc < hi); n = mm.sum()
        if n < 30: continue
        centers.append((lo + hi) / 2)
        m_v5.append(d_v5[mm].mean()); s_v5.append(d_v5[mm].std() / np.sqrt(n))   # SEM
        m_inj.append(d_inj[mm].mean()); s_inj.append(d_inj[mm].std() / np.sqrt(n))
    centers = np.array(centers)
    cr_v5 = np.corrcoef(d_v5, gtc)[0, 1]; cr_inj = np.corrcoef(d_inj, gtc)[0, 1]

    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    # (L) binned mean delta vs closing
    ax[0].errorbar(centers, m_v5, yerr=s_v5, marker="o", capsize=3, label=f"v5 (no GT vrel)  corr={cr_v5:+.2f}", color="#888")
    ax[0].errorbar(centers, m_inj, yerr=s_inj, marker="s", capsize=3, label=f"vinject (GT vrel)  corr={cr_inj:+.2f}", color="#005195")
    ax[0].axhline(0, color="k", lw=0.5); ax[0].axvline(0, color="k", lw=0.5, ls=":")
    ax[0].set_xlabel("GT closing speed (m/s)   (- receding / + approaching)")
    ax[0].set_ylabel("final pedal = f_1 base + gate·delta   (- = brake)")
    ax[0].set_title("최종 pedal vs lead closing speed\n(monotonic ↓ = correct ACC)")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[0].annotate("approaching → brake", (15, min(m_inj) - 0.05), color="#005195", ha="center", fontsize=9)
    ax[0].annotate("receding → less brake", (-15, max(m_inj) + 0.02), color="#005195", ha="center", fontsize=9)
    # (R) scatter (subsample) vinject
    idx = np.random.RandomState(0).choice(len(gtc), min(4000, len(gtc)), replace=False)
    ax[1].scatter(gtc[idx], d_inj[idx], s=3, alpha=0.25, color="#005195")
    ax[1].axhline(0, color="k", lw=0.5); ax[1].axvline(0, color="k", lw=0.5, ls=":")
    ax[1].set_xlabel("GT closing speed (m/s)"); ax[1].set_ylabel("final pedal (vinject)")
    ax[1].set_title(f"vinject final pedal scatter (corr={cr_inj:+.2f})\nGT vrel 주입 시 closing 으로 modulate")
    ax[1].grid(alpha=0.3)
    fig.suptitle("GT-vrel 주입 실험: 최종 pedal(f_1 base + gate·delta) vs closing — 좋은 상대속도 주면 ACC 제동 modulate → 병목 = a_2 vL_rel 지각", fontsize=11)
    out = "outputs_multi/viz/vinject_delta_vs_closing.png"
    os.makedirs("outputs_multi/viz", exist_ok=True)
    plt.tight_layout(); plt.savefig(out, dpi=110); plt.close()
    print(f"[done] {out}", flush=True)
    print(f"corr(delta,closing): v5={cr_v5:+.3f}  vinject={cr_inj:+.3f}", flush=True)


if __name__ == "__main__":
    main()
