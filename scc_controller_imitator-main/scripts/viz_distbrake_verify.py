"""distbrake 라벨 시각 검증: x=object distance, y=ego speed, color=acceleration.
좌: distbrake 곡선 출력(-sd, 순수 rule)  우: 최종 라벨 pedal(step0, cornering 포함).
onset(v)=DB_D0+DB_HEADWAY·v 경계선 overlay. 라벨 영문(폰트 깨짐 방지)."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DB_D0, DB_HEADWAY, DB_DFULL, DB_MINB, DB_P, MAXB = 10.0, 1.0, 7.0, 1.0, 2.0, 20.0
D = os.environ.get("SCC_DATA_DIR") or "/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data"
DSETS = {
    "sangam_lead_0520": "f1_prebake_iy_sangam_lead_0520_v1_brakeaug_prev20.npz",
    "sangam_lead_0521": "f1_prebake_iy_sangam_lead_0521_v1_brakeaug_prev20.npz",
    "DriverB_new":      "f1_prebake_iy_DriverB_new_logged_fixed_prev20_trajfix.npz",
}


def dist_brake_decel(dist, v_ego):
    onset = DB_D0 + DB_HEADWAY * v_ego
    u = np.clip((onset - dist) / np.maximum(onset - DB_DFULL, 1e-3), 0, 1)
    sd = DB_MINB + (MAXB - DB_MINB) * (u ** DB_P)
    sd = np.where(dist <= DB_DFULL, MAXB, sd)
    sd = np.where(dist >= onset, 0.0, sd)
    return sd


dist_all, vego_all, decel_all, ped_all = [], [], [], []
for ds, pbf in DSETS.items():
    pb = np.load(f"{D}/prebake/{pbf}", allow_pickle=True)
    v_ego = pb["ego_5d"][:, 0]
    zg = np.load(f"{D}/cache/inlane_gt_{ds}.npz", allow_pickle=True)
    zk = {k: i for i, k in enumerate(zg["keys"].tolist())}
    idl = zg["inlane_dist"]
    ped = np.load(f"{D}/cache/pedal_label_{ds}_distbrake.npz", allow_pickle=True)["pedal"][:, 0]
    ck = pb["curr_keys"].tolist()
    gi = np.array([zk.get(k, -1) for k in ck])
    have = gi >= 0
    dist = np.where(have, idl[np.clip(gi, 0, len(idl) - 1)], np.inf)
    m = np.isfinite(dist) & (dist < 60)   # in-lane lead frames only
    dist_all.append(dist[m]); vego_all.append(v_ego[m])
    decel_all.append(-dist_brake_decel(dist[m], v_ego[m])); ped_all.append(ped[m])
    print(f"[{ds}] lead frames {int(m.sum())}", flush=True)

dist = np.concatenate(dist_all); vego = np.concatenate(vego_all)
decel = np.concatenate(decel_all); ped = np.concatenate(ped_all)
print(f"[total] {len(dist)} lead frames | dist {dist.min():.1f}-{dist.max():.1f}m | vego {vego.min():.1f}-{vego.max():.1f}", flush=True)

vline = np.linspace(0, vego.max() + 2, 100)
onset_line = DB_D0 + DB_HEADWAY * vline

fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
order = np.argsort(np.abs(decel))   # 강한 brake 위로
for ax, c, ttl, cl, cm in [
    (axes[0], decel[order], "distbrake curve output  (-sd, pure rule)", (-20, 0), "viridis_r"),
    (axes[1], ped[order], "final label pedal @step0  (min(raw, -sd) -> pedal_map)", (-1, 1), "RdBu"),
]:
    sc = ax.scatter(dist[order], vego[order], c=c, s=4, cmap=cm, vmin=cl[0], vmax=cl[1], alpha=0.6)
    ax.plot(onset_line, vline, "k--", lw=1.4, label="onset(v)=10+v  (brake starts)")
    ax.axvline(DB_DFULL, color="r", ls=":", lw=1.2, label=f"d_full={DB_DFULL}m (full brake)")
    ax.set_xlabel("object distance [m]"); ax.set_ylabel("ego speed [m/s]")
    ax.set_title(ttl, fontsize=10); ax.set_xlim(0, 60); ax.set_ylim(-1, vego.max() + 2)
    ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.25)
    plt.colorbar(sc, ax=ax, label=("decel [m/s^2]" if ax is axes[0] else "pedal label [-1,1]"))

fig.suptitle("distbrake label verification  (in-lane lead frames: sangam_lead_0520/0521 + DriverB)", fontsize=12)
plt.tight_layout()
os.makedirs("outputs_multi/viz", exist_ok=True)
out = "outputs_multi/viz/distbrake_verify.png"
plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close()
print(f"[done] {out}")
