"""SCC profile 별 v_scc 거동 차이 sanity plot (English labels).
좌: 대표 lead frame 의 v_scc(s) profile p1..p5 overlay (lead 앞 감속 시점/강도).
우: lead frame 집합에 대한 profile 별 평균 감속 깊이 / 제동 강도 막대.
사용: python scripts/viz_profile_vscc.py <smoke_base> <out_png>
"""
import sys, os, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.viz_synth_batch_result import parse_scc_txt

PROFILES = ["p1", "p2", "p3", "p4", "p5"]
LABELS = {"p1": "p1_aggr (th0.4/sm2.0)", "p2": "p2 (th0.8/sm3.0)",
          "p3": "p3_mid (th1.2/sm4.0)", "p4": "p4 (th1.6/sm5.5)",
          "p5": "p5_cons (th2.0/sm7.0)"}
COL = {"p1": "C2", "p2": "C0", "p3": "C8", "p4": "C1", "p5": "C3"}


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "/tmp/scc_profile_smoke"
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/scc_profile_smoke/profile_vscc.png"
    # pick a representative lead frame present in all profiles
    keys_per = {}
    for p in PROFILES:
        ks = {}
        for f in glob.glob(f"{base}/cmp_{p}/*logged*.scc.txt"):
            d = parse_scc_txt(f)
            if len(d["in_lane_obj_ids"]) > 0 and d["traj"]:
                ks[os.path.basename(f)] = f
        keys_per[p] = ks
    common = set(keys_per["p1"])
    for p in PROFILES[1:]:
        common &= set(keys_per[p])
    # rep frame = one where p5 brakes hardest vs p1
    best = None; best_spread = -1
    for k in common:
        d1 = parse_scc_txt(keys_per["p1"][k]); d5 = parse_scc_txt(keys_per["p5"][k])
        t1 = np.array(d1["traj"]); t5 = np.array(d5["traj"])
        sp = (t1[:, 3] - t1[:, 4]).max() - (t5[:, 3] - t5[:, 4]).max()
        if abs(sp) > best_spread:
            best_spread = abs(sp); best = k

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 5.5))
    # left: v_scc(s) overlay for rep frame
    d = parse_scc_txt(keys_per["p1"][best]); tr = np.array(d["traj"])
    axL.plot(tr[:, 1], tr[:, 3], "k--", lw=2, label="v_orig (raceline)")
    for p in PROFILES:
        dp = parse_scc_txt(keys_per[p][best]); trp = np.array(dp["traj"])
        axL.plot(trp[:, 1], trp[:, 4], COL[p], lw=2, label=LABELS[p])
    lead_s = tr[:, 5][~np.isnan(tr[:, 5])]
    if len(lead_s):
        axL.axvline(lead_s.min(), color="gray", ls=":", lw=1.5, label="nearest lead s")
    axL.set_xlabel("s (m, cumulative arc length from ego)")
    axL.set_ylabel("speed (m/s)")
    axL.set_title(f"v_scc profile vs SCC strategy (frame {best.split('__t')[1][:7]}, in-lane lead)")
    axL.legend(fontsize=8, loc="best"); axL.grid(True, alpha=0.3)

    # right: per-profile mean decel depth & brake strength over lead frames
    dv_means, ax_means = [], []
    for p in PROFILES:
        dvs, axs = [], []
        for k in common:
            dp = parse_scc_txt(keys_per[p][k]); trp = np.array(dp["traj"])
            dvs.append((trp[:, 3] - trp[:, 4]).max())
            t = trp[:, 2]; v = trp[:, 4]
            if len(t) > 2:
                with np.errstate(all="ignore"):
                    a = np.gradient(v, t)
                a = a[np.isfinite(a)]
                axs.append(a.min() if len(a) else 0.0)
        dv_means.append(np.mean(dvs)); ax_means.append(np.mean(axs))
    x = np.arange(len(PROFILES))
    axR.bar(x - 0.2, dv_means, 0.4, label="mean max decel depth v_orig-v_scc (m/s)", color="C0")
    axR2 = axR.twinx()
    axR2.bar(x + 0.2, np.abs(ax_means), 0.4, label="mean brake strength |min ax| (m/s2)", color="C3")
    axR.set_xticks(x); axR.set_xticklabels(PROFILES)
    axR.set_xlabel("SCC profile (aggressive -> conservative)")
    axR.set_ylabel("decel depth (m/s)", color="C0")
    axR2.set_ylabel("brake strength (m/s2)", color="C3")
    axR.set_title(f"Lead-approach braking by profile (N={len(common)} lead frames)")
    axR.legend(loc="upper left", fontsize=8); axR2.legend(loc="lower right", fontsize=8)
    axR.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print(f"  → {out}  (rep frame={best})")
    print(f"  dv_max means: {dict(zip(PROFILES, [round(x,2) for x in dv_means]))}")
    print(f"  ax_min means: {dict(zip(PROFILES, [round(x,2) for x in ax_means]))}")


if __name__ == "__main__":
    main()
