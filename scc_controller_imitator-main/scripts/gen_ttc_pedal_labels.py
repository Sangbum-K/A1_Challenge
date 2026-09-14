"""ttc-augment 된 pedal 라벨 생성: raw target_ax 에 ACC-style brake floor(GT inlane) 적용 후 pedal_map.
이전 pedal_label_*.npz 는 raw ax 기반 → 앞차 접근 시도 가속(라벨 버그). 이 _ttc 버전이 lead-brake 포함."""
import os
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.pedal_label import PedalLabelizer

D = os.environ.get("SCC_DATA_DIR") or ("/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data" if os.path.isdir("/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data") else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "local"))
# ACC-style ttc params (train_iy 와 동일). env override 가능 (기본값=legacy _ttc 와 동일).
# _ttc2 재설계: LBL_DECAY=8 LBL_TTC_EMERG=3 LBL_KCLO=0.8 LBL_KPURE=0.45 OUT_SUFFIX=ttc2 로 호출.
D0        = float(os.environ.get("LBL_D0", "5.0"))
HEADWAY   = float(os.environ.get("LBL_HEADWAY", "1.0"))
KGAP      = float(os.environ.get("LBL_KGAP", "0.2"))
KCLO      = float(os.environ.get("LBL_KCLO", "0.6"))
TTC_EMERG = float(os.environ.get("LBL_TTC_EMERG", "2.0"))
DECAY     = float(os.environ.get("LBL_DECAY", "5.0"))
MAXB      = float(os.environ.get("LBL_MAXB", "20.0"))
MARGIN    = float(os.environ.get("LBL_MARGIN", "2.0"))
KSTEPS    = int(os.environ.get("LBL_KSTEPS", "10"))
KPURE     = float(os.environ.get("LBL_KPURE", "0.0"))   # ungated closing-비례 floor (a_pure=KPURE*closing). 0=off(legacy)
# distance-onset brake params (closing 무관, 거리만): onset(v)=DB_D0+DB_HEADWAY·v_ego 안쪽서 brake.
# far(onset)=-1 → near(d_full)=-MAXB, convex u^DB_P. far(onset)는 속도로 이동, near(d_full) 고정.
DB_D0, DB_HEADWAY, DB_DFULL, DB_MINB, DB_P = 10.0, 1.0, 7.0, 1.0, 2.0
BRAKE_MODE = os.environ.get("BRAKE_MODE", "ttc")  # "ttc" | "distbrake"

DSETS = {  # ds: (prebake, inlane_gt, out_name)
    "sangam_lead_0520": ("f1_prebake_iy_sangam_lead_0520_v1_brakeaug_prev20.npz", "inlane_gt_sangam_lead_0520.npz"),
    "sangam_lead_0521": ("f1_prebake_iy_sangam_lead_0521_v1_brakeaug_prev20.npz", "inlane_gt_sangam_lead_0521.npz"),
    "sangam_pertO_v2":  ("f1_prebake_iy_sangam_pertO_v2_prev20_fixed.npz", "inlane_gt_sangam_pertO_v2.npz"),
    "sangam_pertX_v2":  ("f1_prebake_iy_sangam_pertX_v2_prev20_fixed.npz", "inlane_gt_sangam_pertX_v2.npz"),
    "DriverB_new":      ("f1_prebake_iy_DriverB_new_logged_fixed_prev20_trajfix.npz", "inlane_gt_DriverB_new.npz"),
    "spd_v2":           ("f1_prebake_iy_spd_streaming_brakeaug_prev20.npz", "inlane_gt_spd_v2.npz"),
    "spd_0522_40m":     ("f1_prebake_iy_spd_0522_40m_logged_prev20.npz", "inlane_gt_spd_0522_40m.npz"),
}


def acc_safe_decel(dist, closing, v_ego):
    """ACC-style desired decel (>=0). dist,closing,v_ego: (N,) arrays."""
    cl = np.maximum(closing, 0.0)
    ttc = dist / np.maximum(cl, 0.1)
    a_kin = cl ** 2 / (2.0 * np.maximum(dist - MARGIN, 0.5))
    clo_gate = np.clip((DECAY - ttc) / max(DECAY - TTC_EMERG, 1e-3), 0.0, 1.0)
    a_gap = KGAP * np.maximum(D0 + HEADWAY * v_ego - dist, 0.0) + KCLO * cl * clo_gate
    a_emerg = MAXB * np.clip(1.0 - ttc / TTC_EMERG, 0.0, 1.0)
    a_pure = KPURE * cl                              # ungated: closing 에 항상 비례 (원거리 접근도 제동)
    sd = np.maximum.reduce([a_kin, a_gap, a_emerg, a_pure])
    sd = np.where(cl > 0.1, np.minimum(sd, MAXB), 0.0)
    return sd


def dist_brake_decel(dist, v_ego):
    """거리-onset 기반 desired decel (>=0, 양수=감속 크기). closing 무관.
    onset(v)=DB_D0+DB_HEADWAY·v 안쪽서만 brake. far(onset)=DB_MINB, near(<=d_full)=MAXB.
    곡선: decel = MINB + (MAXB-MINB)·u^DB_P,  u=clip((onset-dist)/(onset-d_full),0,1).
    v=0 도 onset=DB_D0(=10m) 안쪽서 brake → 0 으로 죽지 않음."""
    onset = DB_D0 + DB_HEADWAY * v_ego                       # (N,)
    span = np.maximum(onset - DB_DFULL, 1e-3)
    u = np.clip((onset - dist) / span, 0.0, 1.0)
    sd = DB_MINB + (MAXB - DB_MINB) * (u ** DB_P)            # far(u=0)=MINB → near(u=1)=MAXB
    sd = np.where(dist <= DB_DFULL, MAXB, sd)                # 임박 full brake
    sd = np.where(dist >= onset, 0.0, sd)                    # onset 바깥 → 0 (no lead concern)
    return sd, onset


def main():
    L = PedalLabelizer()
    out_suffix = os.environ.get("OUT_SUFFIX") or ("distbrake" if BRAKE_MODE == "distbrake" else "ttc")
    print(f"[mode] BRAKE_MODE={BRAKE_MODE} → pedal_label_*_{out_suffix}.npz", flush=True)
    only = os.environ.get("ONLY_DS")  # comma-sep subset; default=all
    only_set = set(x.strip() for x in only.split(",")) if only else None
    for ds, (pbf, igf) in DSETS.items():
        if only_set is not None and ds not in only_set:
            continue
        pb = np.load(f"{D}/prebake/{pbf}", allow_pickle=True)
        tr, eg, tg = pb["traj"], pb["ego_5d"], pb["target"]
        spd = tr[:, :, 2]; ax = tg[:, :, 1].copy()
        v_ego = eg[:, 0]
        zg = np.load(f"{D}/cache/{igf}", allow_pickle=True)
        zk = {k: i for i, k in enumerate(zg["keys"].tolist())}
        idl = zg["inlane_dist"]; icl = zg["inlane_closing"]
        ck = pb["curr_keys"].tolist()
        gi = np.array([zk.get(k, -1) for k in ck])
        have = gi >= 0
        dist = np.where(have, idl[np.clip(gi, 0, len(idl) - 1)], np.inf)
        clos = np.where(have, icl[np.clip(gi, 0, len(icl) - 1)], 0.0)
        if BRAKE_MODE == "distbrake":
            sd, onset = dist_brake_decel(np.where(np.isfinite(dist), dist, 1e9), v_ego)  # (N,)
            lead = np.isfinite(dist) & (dist < onset)   # closing 무관: onset 안쪽이면 brake
        else:
            lead = np.isfinite(dist) & (clos > 0.1)
            sd = acc_safe_decel(np.where(np.isfinite(dist), dist, 1e3), clos, v_ego)  # (N,)
        # aug: 첫 KSTEPS step ax 에 floor min(raw, -sd) (lead 일 때만)
        aug = ax.copy()
        nfire = 0
        for i in np.where(lead)[0]:
            aug[i, :KSTEPS] = np.minimum(ax[i, :KSTEPS], -sd[i])
            nfire += 1
        ped = L.label(spd, aug)  # (N,40)
        np.savez_compressed(f"{D}/cache/pedal_label_{ds}_{out_suffix}.npz", keys=pb["curr_keys"], pedal=ped.astype(np.float32))
        p0 = ped[:, 0]
        print(f"[{ds}] N={len(ped)} fire={nfire} ({100*nfire/len(ped):.0f}%) | step0 pedal mean={p0.mean():+.3f} brake%={100*(p0<0).mean():.0f} | lead-step0 brake%={100*(p0[lead]<0).mean() if lead.any() else 0:.0f}", flush=True)
    print(f"[done] _{out_suffix} pedal labels 생성 완료", flush=True)


if __name__ == "__main__":
    main()
