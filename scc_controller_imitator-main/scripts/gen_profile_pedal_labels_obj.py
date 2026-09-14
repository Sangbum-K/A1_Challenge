"""SCC 5-profile pedal_label 생성: inlane_gt(obj 추정 dist/closing GT) 물리 감속 공식
+ base pedal(pedal_label_<ds>_ttc) 과 element-wise min 결합.

배경 (2026-05-27, 사용자 최종 설계):
  - synth_batch(C++ MPC) 재합성 / python ego-start recalc(EGO_INFO) 둘 다 폐기 (느림/불일치/위험).
  - obj 가 추정하는 in-lane dist/closing 만 있으면 물리 공식으로 SCC 목표 감속 A 를 직접 계산.
    ego v0 / trajectory / scc.txt / EGO_INFO / synth_batch / raw 일절 불필요 (import 금지).
  - SCC 는 "부족한 brake 만" 더한다: base(pedal_label_ttc, 기존 학습 GT, f_1 base_pedal 동급) 와
    pedal_scc 의 element-wise min(=더 음수=더 brake). base 가 이미 충분히 감속(코너 등)이면
    target=base → SCC 무개입. lead 접근 frame 에서만 target<base.

입력:
  cache/inlane_gt_<ds>.npz   { keys:(M,)<U21, inlane_dist:(M,)f32(m; no-lead=NaN/inf),
                               inlane_closing:(M,)f32(m/s, >0=접근) }
  cache/pedal_label_<ds>_ttc.npz  { keys:(N,)<U21, pedal:(N,40)f32 }  ← base_pedal
  prebake/<pbf>.npz          curr_keys(N) 가 ttc keys 와 동일 순서. traj[:,:,2] = ego 속도 v (40pt).

공식 (per frame, per profile; 일정 감속 가정 → 40 point 동일 broadcast):
  profile s_min: p1=2.0 p2=3.0 p3=4.0 p4=5.5 p5=7.0   (t_headway 미사용, s_min 만 거동 결정)
  A_MIN = -40.0 (풀제동 강화; 합성 GT 의 "최대한 세게" 의도 — 실차 4g 초과지만 거동 강조용)
  if (closing > 0) and isfinite(dist):
      denom = dist - s_min
      if denom <= 0:  A = -40.0                          # 안전거리 침범 = 충돌 임박 → 풀제동
      else:           A = clip(-(closing**2)/(2*denom), -40.0, 0.0)  # s_min 안에서 closing→0 일정감속
  else:               A = 0.0                            # no-lead/receding → SCC 무개입
  → s_min 이 분모: |A| 는 p1<p2<...<p5 (p5 가 더 일찍/세게 감속).

A → pedal_scc: PedalLabelizer(v=traj[:,:,2], ax=A broadcast).  (gen_profile_pedal_labels.py 동일 사용법)
  ※ velocity 는 prebake traj[:,:,2] 사용. target[:,:,1] 은 ax(longitudinal accel; 음수 다수)라
     velocity 가 아니며, 기존 검증된 gen_ttc_pedal_labels.py 도 spd=traj[:,:,2] 를 쓴다 (확인 2026-05-27).

target_pedal = min(base_pedal, pedal_scc)  (element-wise, 더 brake 쪽 채택; pedal 음수=brake)

출력: cache/pedal_label_<ds>_p{N}.npz  { keys:(N,)<U21=curr_keys, pedal:(N,40)f32 }
  → 기존 _ttc / _p{N} 와 동일 형식 (train_prof635_side --pedal-suffix p{N} 즉시 호환).
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # PedalLabelizer 의 pedal_map csv 가 repo-relative 경로
from src.utils.pedal_label import PedalLabelizer

D = os.environ.get("SCC_DATA_DIR", "/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data")
TARG_N = 40
A_MIN = -40.0
A_MAX = 0.0

# profile: s_min[m]. 거동은 s_min(분모) 으로만 갈림 (t_headway 미사용 — spec 대로).
PROFILES = {"p1": 3.0, "p2": 6.0, "p3": 10.0, "p4": 14.0, "p5": 18.0}

# ds: (prebake basename, inlane_gt basename, base pedal_label suffix)
DSETS = {
    "sangam_lead_0520": ("f1_prebake_iy_sangam_lead_0520_v1_brakeaug_prev20.npz", "inlane_gt_sangam_lead_0520.npz"),
    "sangam_lead_0521": ("f1_prebake_iy_sangam_lead_0521_v1_brakeaug_prev20.npz", "inlane_gt_sangam_lead_0521.npz"),
    "sangam_pertO_v2":  ("f1_prebake_iy_sangam_pertO_v2_prev20_fixed.npz", "inlane_gt_sangam_pertO_v2.npz"),
    "sangam_pertX_v2":  ("f1_prebake_iy_sangam_pertX_v2_prev20_fixed.npz", "inlane_gt_sangam_pertX_v2.npz"),
    "DriverB_new":      ("f1_prebake_iy_DriverB_new_logged_fixed_prev20_trajfix.npz", "inlane_gt_DriverB_new.npz"),
    "spd_v2":           ("f1_prebake_iy_spd_streaming_brakeaug_prev20.npz", "inlane_gt_spd_v2.npz"),
}


def compute_A(dist: np.ndarray, closing: np.ndarray, s_min: float) -> np.ndarray:
    """inlane dist/closing → SCC 목표 감속 A (m/s², <=0). 벡터화.
    NaN/inf dist 또는 closing<=0 → A=0 (no-lead/receding, 무개입)."""
    dist = np.asarray(dist, np.float64)
    closing = np.asarray(closing, np.float64)
    lead = np.isfinite(dist) & (closing > 0.0)
    A = np.zeros_like(dist)
    if not lead.any():
        return A
    d = dist[lead]; c = closing[lead]
    denom = d - s_min
    a = np.empty_like(d)
    breach = denom <= 0.0                        # 안전거리 침범 → 풀제동
    a[breach] = A_MIN
    nb = ~breach
    a[nb] = -(c[nb] ** 2) / (2.0 * denom[nb])
    a = np.clip(a, A_MIN, A_MAX)
    A[lead] = a
    return A


def load_curr_keys(pb) -> np.ndarray:
    curr_keys = np.asarray(pb["curr_keys"]).astype("U21")
    if "curr_keys" not in pb.files or (curr_keys == "").mean() > 0.5:
        sk = np.asarray(pb["sample_keys"])
        def _derive(s: str) -> str:
            for sfx in ("__logged", "__rand_"):
                idx = s.find(sfx)
                if idx >= 0:
                    return s[:idx]
            return s
        curr_keys = np.array([_derive(str(s)) for s in sk], dtype="U21")
        print(f"[info] curr_keys empty → derived from sample_keys "
              f"({(curr_keys != '').sum()}/{len(curr_keys)} non-empty)", flush=True)
    return curr_keys


def gen_one(L: PedalLabelizer, ds: str, profile: str, args) -> None:
    s_min = PROFILES[profile]
    pbf, igf = DSETS[ds]

    pbp = args.prebake or f"{D}/prebake/{pbf}"
    pb = np.load(pbp, allow_pickle=True)
    curr_keys = load_curr_keys(pb)
    N = len(curr_keys)
    vmat = np.asarray(pb["traj"][:, :, 2], np.float64)        # (N,40) ego 속도 (PedalLabelizer velocity)

    # base pedal (pedal_label_<ds>_ttc). keys 가 curr_keys 와 동일 순서여야 element-wise min 정합.
    bp = args.base or f"{D}/cache/pedal_label_{ds}_{args.base_suffix}.npz"
    zb = np.load(bp, allow_pickle=True)
    base_keys = np.asarray(zb["keys"]).astype("U21")
    base_pedal = np.asarray(zb["pedal"], np.float32)          # (N,40)
    if not (len(base_keys) == N and (base_keys == curr_keys).all()):
        raise RuntimeError(f"[{ds}/{profile}] base keys != prebake curr_keys (N {len(base_keys)} vs {N})")

    # inlane_gt (anchor/frame 단위, unique key) → frame 단위 A → curr_keys 로 broadcast(key-join).
    igp = args.inlane_gt or f"{D}/cache/{igf}"
    zv = np.load(igp, allow_pickle=True)
    ik = np.asarray(zv["keys"]).astype(str)
    idist = np.asarray(zv["inlane_dist"], np.float64)
    iclose = np.asarray(zv["inlane_closing"], np.float64)
    A_frame = compute_A(idist, iclose, s_min)                 # (M,) inlane_gt order
    k2a = {k: A_frame[i] for i, k in enumerate(ik.tolist())}
    A_curr = np.fromiter((k2a.get(str(k), 0.0) for k in curr_keys), np.float64, count=N)
    covered = int(sum(1 for k in curr_keys if str(k) in k2a))

    # ax=A broadcast(일정 감속) → pedal_scc. v = traj[:,:,2].
    amat = np.repeat(A_curr[:, None], TARG_N, axis=1)         # (N,40)
    pedal_scc = L.label(vmat, amat).astype(np.float32)        # (N,40)

    # ★ min 결합: SCC 가 감속을 요구(A<0)하는 frame 에서만 더 brake 쪽 채택.
    #   A==0(no-lead/receding)인 frame 은 PedalLabelizer(v,0)=양의 accel pedal 이라
    #   min 을 그대로 쓰면 base 의 정당한 가속을 깎는다 → A<0 일 때만 min, 아니면 base 그대로.
    lead_mask = (A_curr < 0.0)[:, None]                       # (N,1) → broadcast
    target_pedal = np.where(lead_mask, np.minimum(base_pedal, pedal_scc), base_pedal).astype(np.float32)

    out = args.out or f"{D}/cache/pedal_label_{ds}_{profile}.npz"
    np.savez_compressed(out, keys=curr_keys, pedal=target_pedal)

    lead = A_curr < 0.0
    intervened = (target_pedal < base_pedal - 1e-6).any(axis=1)   # SCC 가 base 를 끌어내린 sample
    p0 = target_pedal[:, 0]
    print(f"[{ds}/{profile}] N={N} cov={covered}({100*covered/max(N,1):.0f}%) s_min={s_min} "
          f"| lead(A<0)={int(lead.sum())}({100*lead.mean():.0f}%) "
          f"A[A<0] med={np.median(A_curr[lead]) if lead.any() else 0:+.2f} min={A_curr.min():+.2f} "
          f"| SCC 개입 sample={int(intervened.sum())}({100*intervened.mean():.1f}%) "
          f"| step0 target med={np.median(p0):+.3f} brake%={100*(p0 < 0).mean():.0f}", flush=True)
    print(f"  → {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default=None, help="단일 ds (생략 시 DSETS 전체)")
    ap.add_argument("--profile", default=None, choices=list(PROFILES), help="단일 profile (생략 시 p1..p5 전체)")
    ap.add_argument("--prebake", default=None, help="단일 ds 일 때 override")
    ap.add_argument("--inlane-gt", default=None, help="단일 ds 일 때 override")
    ap.add_argument("--base", default=None, help="단일 ds 일 때 base pedal npz override")
    ap.add_argument("--base-suffix", default="ttc", help="base pedal suffix (기본 ttc)")
    ap.add_argument("--out", default=None, help="단일 ds+profile 일 때만")
    args = ap.parse_args()

    dss = [args.ds] if args.ds else list(DSETS)
    profs = [args.profile] if args.profile else list(PROFILES)
    L = PedalLabelizer()
    for ds in dss:
        for p in profs:
            gen_one(L, ds, p, args)


if __name__ == "__main__":
    main()
