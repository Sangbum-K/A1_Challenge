"""gate BCE (f_2 + gate) 학습 5셋 데이터 분포 정리.
read + 분석 + plot only. GPU 미사용.

분석 단위 두 가지:
  - row 기준 = 학습 loss 가 실제 보는 분포 (brakeaug 증강 포함).
  - unique frame 기준 = 고유 주행 상황 다양성 (curr_key dedup).
지도 절대좌표/raceline s 는 prebake 에 없음 → 곡률(traj)/yaw_rate 로 코너 vs 직선 근사 (추정).
inlane lateral 40-bin 분포는 inlane_gt cache 에 없음(이미 in-lane bin16-25 로 필터된 dist 만) → 보고에 명시.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs_multi", "data_dist")
os.makedirs(OUT, exist_ok=True)

PB = {
    "sangam_lead_0520": "f1_prebake_iy_sangam_lead_0520_v1_brakeaug_prev20.npz",
    "sangam_lead_0521": "f1_prebake_iy_sangam_lead_0521_v1_brakeaug_prev20.npz",
    "sangam_pertO_v2":  "f1_prebake_iy_sangam_pertO_v2_prev20_fixed.npz",
    "sangam_pertX_v2":  "f1_prebake_iy_sangam_pertX_v2_prev20_fixed.npz",
    "DriverB_new":      "f1_prebake_iy_DriverB_new_logged_fixed_prev20_trajfix.npz",
}
GATE_TAU = 0.5
DT = 0.1  # traj40 step (s)

# distance bins
DIST_BINS = [(0, 20, "near<20"), (20, 40, "mid20-40"), (40, 60, "far40-60"), (60, 1e9, "60+")]
# closing bins (m/s)
CLOS_BINS = [(-1e9, 0, "receding<0"), (0, 0.5, "cruise0-0.5"), (0.5, 2, "appr_weak0.5-2"), (2, 1e9, "appr_strong2+")]


def curvature_from_traj(traj40):
    """(40,3)[x_fwd,y_left,speed] 의 평균 |kappa| 근사. 0.5s~1.3s 구간 사용 (예측 안정 영역).
    kappa ~ |x'y'' - y'x''| / (x'^2+y'^2)^1.5. 추정치."""
    x = traj40[:, 0].astype(np.float64); y = traj40[:, 1].astype(np.float64)
    if len(x) < 5:
        return 0.0
    dx = np.gradient(x); dy = np.gradient(y)
    ddx = np.gradient(dx); ddy = np.gradient(dy)
    denom = (dx * dx + dy * dy) ** 1.5 + 1e-6
    kap = np.abs(dx * ddy - dy * ddx) / denom
    # 0.3~1.0s window (idx 3..10) 평균
    return float(np.nanmean(kap[3:11]))


def analyze():
    rows = []  # per-dataset summary
    # for global aggregation (row-level)
    g_speed = []; g_dist = []; g_clos = []; g_gate = []; g_yaw = []; g_kap = []; g_track = []
    g_track_lead = []  # track tag aligned to lead-frame arrays (dist/clos)
    # store per-dataset arrays for plotting
    per = {}

    for nm, pb in PB.items():
        z = np.load(f"{D}/prebake/{pb}", allow_pickle=True, mmap_mode="r")
        ck = z["curr_keys"]
        ego5 = np.asarray(z["ego_5d"])          # (N,5) [vx,vy,ax,ay,yaw_rate]
        traj = z["traj"]                         # (N,40,3)
        mask0 = np.asarray(z["mask"][:, 0]).astype(bool)
        N = len(ck)
        uniq = np.unique(ck)

        # inlane_gt join (gate_valid + dist + closing)
        ig = np.load(f"{D}/cache/inlane_gt_{nm}.npz", allow_pickle=True, mmap_mode="r")
        igk = {k: i for i, k in enumerate(ig["keys"].tolist())}
        idist = np.asarray(ig["inlane_dist"]); iclos = np.asarray(ig["inlane_closing"])

        # row-level arrays
        idx = np.array([igk.get(k, -1) for k in ck], dtype=np.int64)
        valid = idx >= 0
        dist = np.full(N, np.inf, dtype=np.float32)
        clos = np.zeros(N, dtype=np.float32)
        dist[valid] = idist[idx[valid]]
        cl_raw = iclos[idx[valid]]
        clos[valid] = np.where(np.isfinite(cl_raw), cl_raw, 0.0)
        has_lead = np.isfinite(dist)              # finite dist = in-lane lead present
        gate_tgt = (has_lead & (clos > GATE_TAU)).astype(np.int8)

        vx = ego5[:, 0]; yaw = ego5[:, 4]
        speed = np.sqrt(ego5[:, 0] ** 2 + ego5[:, 1] ** 2)

        # curvature only on unique frames (cheap) then map back? expensive on rows -> compute on unique
        # 학습이 보는 건 row 지만 traj 곡률은 unique frame 마다 동일/유사. row 비용 절감 위해 unique 만.
        kap_by_key = {}
        for u in uniq:
            j = np.where(ck == u)[0][0]
            kap_by_key[u] = curvature_from_traj(np.asarray(traj[j]))
        kap = np.array([kap_by_key[k] for k in ck], dtype=np.float32)

        # track/policy from meta if available
        meta_path = f"{D}/meta_{nm}.json"
        track = "unknown"; npolicy = None
        if os.path.exists(meta_path):
            try:
                m = json.load(open(meta_path))
                if isinstance(m, list) and m:
                    tracks = set(e.get("track", "?") for e in m[:5000])
                    pols = set(e.get("policy", "?") for e in m)
                    track = "/".join(sorted(tracks))
                    npolicy = len(pols)
            except Exception:
                pass
        if track == "unknown":
            track = "DriverB" if nm.startswith("Driver") else "sangam_track"  # inferred from name

        # masks for "lead present" subset (row-level, mask0 valid)
        m0 = mask0 & valid
        lead_m = m0 & has_lead

        def binpct(vals, bins):
            tot = max(1, len(vals))
            return {lbl: float(((vals >= lo) & (vals < hi)).sum()) / tot * 100 for lo, hi, lbl in bins}

        dist_lead = dist[lead_m]
        clos_lead = clos[lead_m]
        dpct = binpct(dist_lead, DIST_BINS)
        cpct = binpct(clos_lead, CLOS_BINS)

        # far + high closing rarity (lead frames with dist>=40 and closing>2)
        far_hi = int(((dist_lead >= 40) & (clos_lead > 2.0)).sum())
        far_hi_pct = far_hi / max(1, len(dist_lead)) * 100

        row = dict(
            name=nm, track=track, n_policy=npolicy,
            rows=int(N), unique=int(len(uniq)),
            rows_valid=int(m0.sum()),
            lead_rows=int(lead_m.sum()),
            lead_pct=float(lead_m.sum()) / max(1, int(m0.sum())) * 100,
            gate_pos_pct=float(gate_tgt[m0].mean()) * 100 if m0.any() else 0.0,
            speed_mean=float(np.nanmean(speed[m0])) if m0.any() else 0.0,
            speed_p50=float(np.nanmedian(speed[m0])) if m0.any() else 0.0,
            speed_p95=float(np.nanpercentile(speed[m0], 95)) if m0.any() else 0.0,
            dist_pct=dpct, clos_pct=cpct,
            far_hi_pct=far_hi_pct, far_hi_n=far_hi,
            kap_corner_pct=float((kap[m0] > 0.01).mean()) * 100 if m0.any() else 0.0,
        )
        rows.append(row)
        per[nm] = dict(speed=speed[m0], dist=dist_lead, clos=clos_lead, kap=kap[m0],
                       gate=gate_tgt[m0], yaw=yaw[m0])

        g_speed.append(speed[m0]); g_dist.append(dist_lead); g_clos.append(clos_lead)
        g_gate.append(gate_tgt[m0]); g_yaw.append(yaw[m0]); g_kap.append(kap[m0])
        g_track += [track] * int(m0.sum())
        g_track_lead += [track] * len(dist_lead)
        print(f"[done] {nm}: rows={N} unique={len(uniq)} lead%={row['lead_pct']:.1f} "
              f"gate_pos%={row['gate_pos_pct']:.1f} far_hi%={far_hi_pct:.2f}", flush=True)

    return rows, per, dict(speed=np.concatenate(g_speed), dist=np.concatenate(g_dist),
                           clos=np.concatenate(g_clos), gate=np.concatenate(g_gate),
                           yaw=np.concatenate(g_yaw), kap=np.concatenate(g_kap),
                           track=np.array(g_track), track_lead=np.array(g_track_lead))


def make_plots(rows, per, g):
    names = list(per.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

    # 1. frames: rows vs unique (log) — augmentation imbalance
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(names)); w = 0.38
    ax.bar(x - w / 2, [r["rows"] for r in rows], w, label="prebake rows (train sees)", color="#005195")
    ax.bar(x + w / 2, [r["unique"] for r in rows], w, label="unique frames (driving situations)", color="#01A0E9")
    ax.set_yscale("log"); ax.set_ylabel("count (log)"); ax.set_title("Frame counts: rows vs unique (brakeaug imbalance)")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/01_frame_counts.png", dpi=120); plt.close(fig)

    # 2. lead% and gate_pos% per dataset
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, [r["lead_pct"] for r in rows], w, label="lead present %", color="#005195")
    ax.bar(x + w / 2, [r["gate_pos_pct"] for r in rows], w, label="gate target=1 (lead & closing>0.5) %", color="#DC291E")
    ax.set_ylabel("% of valid rows"); ax.set_title("Lead presence & gate-open target rate")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/02_lead_gate_rate.png", dpi=120); plt.close(fig)

    # 3. inlane_dist histogram per dataset (lead frames)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 80, 41)
    for i, nm in enumerate(names):
        d = per[nm]["dist"]; d = d[np.isfinite(d)]
        if len(d):
            ax.hist(d, bins=bins, histtype="step", density=True, label=nm, color=colors[i], lw=1.5)
    ax.set_xlabel("inlane_dist [m]"); ax.set_ylabel("density"); ax.set_title("In-lane lead distance distribution (lead frames)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{OUT}/03_dist_hist.png", dpi=120); plt.close(fig)

    # 4. inlane_closing histogram per dataset
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(-5, 8, 53)
    for i, nm in enumerate(names):
        c = per[nm]["clos"]
        if len(c):
            ax.hist(np.clip(c, -5, 8), bins=bins, histtype="step", density=True, label=nm, color=colors[i], lw=1.5)
    ax.axvline(GATE_TAU, color="k", ls="--", lw=1, label=f"gate_tau={GATE_TAU}")
    ax.set_xlabel("inlane_closing [m/s]  (approach>0)"); ax.set_ylabel("density")
    ax.set_title("Lead closing-speed distribution (lead frames)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{OUT}/04_closing_hist.png", dpi=120); plt.close(fig)

    # 5. ego speed histogram per dataset
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 35, 71)
    for i, nm in enumerate(names):
        s = per[nm]["speed"]; s = s[np.isfinite(s)]
        if len(s):
            ax.hist(s, bins=bins, histtype="step", density=True, label=nm, color=colors[i], lw=1.5)
    ax.set_xlabel("ego speed [m/s]"); ax.set_ylabel("density"); ax.set_title("Ego speed distribution")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{OUT}/05_speed_hist.png", dpi=120); plt.close(fig)

    # 6. dist x closing 2D (global) — far+high-closing rarity
    fig, ax = plt.subplots(figsize=(7, 5.5))
    d = g["dist"]; c = g["clos"]; m = np.isfinite(d)
    h = ax.hist2d(d[m], np.clip(c[m], -5, 8), bins=[np.linspace(0, 80, 41), np.linspace(-5, 8, 53)],
                  cmap="viridis", cmin=1)
    fig.colorbar(h[3], ax=ax, label="count")
    ax.axhline(2.0, color="r", ls="--", lw=1); ax.axvline(40, color="r", ls="--", lw=1)
    ax.set_xlabel("inlane_dist [m]"); ax.set_ylabel("inlane_closing [m/s]")
    ax.set_title("Lead dist x closing (global, lead frames)\nred box = far(>40m) & fast-approach(>2 m/s) = main-track bottleneck")
    fig.tight_layout(); fig.savefig(f"{OUT}/06_dist_closing_2d.png", dpi=120); plt.close(fig)

    # 7. DriverB vs sangam-agent aggregate comparison (all on sangam_track; differ by driver/policy)
    db_speed = per["DriverB_new"]["speed"]
    sg_speed = np.concatenate([per[n]["speed"] for n in names if n != "DriverB_new"])
    db_dist = per["DriverB_new"]["dist"]; db_dist = db_dist[np.isfinite(db_dist)]
    sg_dist = np.concatenate([per[n]["dist"] for n in names if n != "DriverB_new"]); sg_dist = sg_dist[np.isfinite(sg_dist)]
    db_clos = per["DriverB_new"]["clos"]; sg_clos = np.concatenate([per[n]["clos"] for n in names if n != "DriverB_new"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    triples = [(axes[0], db_speed, sg_speed, "ego speed [m/s]", np.linspace(0, 35, 50)),
               (axes[1], db_dist, sg_dist, "inlane_dist [m]", np.linspace(0, 80, 40)),
               (axes[2], db_clos, sg_clos, "closing [m/s]", np.linspace(-5, 8, 50))]
    for ax, vdb, vsg, xlab, bn in triples:
        vdb = vdb[np.isfinite(vdb)]; vsg = vsg[np.isfinite(vsg)]
        if len(vdb): ax.hist(np.clip(vdb, bn[0], bn[-1]), bins=bn, density=True, alpha=0.5, label="DriverB (logged)", color="#DC291E")
        if len(vsg): ax.hist(np.clip(vsg, bn[0], bn[-1]), bins=bn, density=True, alpha=0.5, label="sangam-agent (4 sets)", color="#005195")
        ax.set_xlabel(xlab); ax.legend(fontsize=8)
    axes[0].set_ylabel("density"); fig.suptitle("DriverB vs sangam-agent (both sangam_track; row-level, lead frames for dist/clos)")
    fig.tight_layout(); fig.savefig(f"{OUT}/07_driverB_vs_sangam.png", dpi=120); plt.close(fig)

    print(f"[plots] saved 7 figures to {OUT}", flush=True)


def write_table(rows):
    lines = []
    lines.append("dataset | track | n_policy | rows | unique | lead% | gate_pos% | spd_mean | spd_p95 | far+hiClos%")
    lines.append("-" * 110)
    for r in rows:
        lines.append(f"{r['name']:18s} | {r['track'][:14]:14s} | {str(r['n_policy']):>4s} | "
                     f"{r['rows']:>7d} | {r['unique']:>6d} | {r['lead_pct']:5.1f} | {r['gate_pos_pct']:6.1f} | "
                     f"{r['speed_mean']:6.1f} | {r['speed_p95']:6.1f} | {r['far_hi_pct']:5.2f}")
    lines.append("")
    lines.append("--- dist bins (% of lead frames) ---")
    lines.append("dataset | " + " | ".join(l for _, _, l in DIST_BINS))
    for r in rows:
        lines.append(f"{r['name']:18s} | " + " | ".join(f"{r['dist_pct'][l]:6.1f}" for _, _, l in DIST_BINS))
    lines.append("")
    lines.append("--- closing bins (% of lead frames) ---")
    lines.append("dataset | " + " | ".join(l for _, _, l in CLOS_BINS))
    for r in rows:
        lines.append(f"{r['name']:18s} | " + " | ".join(f"{r['clos_pct'][l]:6.1f}" for _, _, l in CLOS_BINS))
    txt = "\n".join(lines)
    print(txt, flush=True)
    open(f"{OUT}/summary_table.txt", "w").write(txt)


if __name__ == "__main__":
    rows, per, g = analyze()
    write_table(rows)
    make_plots(rows, per, g)
    print("[ALL DONE]", flush=True)
