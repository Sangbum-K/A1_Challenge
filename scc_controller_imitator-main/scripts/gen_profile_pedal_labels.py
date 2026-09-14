"""SCC profile pedal label 생성: synth_batch 의 v_scc(=replanned speed) 자체를 GT 로 심는다.

배경 (confirmed 2026-05-27):
  - synth_batch .scc.txt 의 traj col4 (v_scc) 는 t_headway/s_min 에 강하게 반응 — lead 앞 감속
    시점/강도가 profile 별로 다름 (p1_aggr=거의 무감속 ~ p5_cons=일찍 깊은 감속).
  - 반면 long_traj 의 ax (기존 prebake target[:,1]) 는 profile 에 거의 무반응 → 기존
    pedal_label_*_{ttc,distbrake} 는 v_scc 를 전혀 안 쓰고 후처리 노브(acc_safe_decel 등)로
    제동을 덮어씌웠다. 따라서 profile 거동은 long_traj/raceline ax 경로로는 GT 에 안 들어간다.
  - 이 스크립트는 v_scc 프로파일을 시간미분해 ax 를 유도하고(dv/dt), (v_scc, ax) 를 그대로
    PedalLabelizer 에 통과시켜 profile 별 제동을 GT(pedal) 에 직접 심는다 — 후처리 노브 제거.

출력 형식: pedal_label_<ds>_p{N}.npz  { keys:(N,)<U21=curr_keys, pedal:(N,40) f32 }
          → 기존 pedal_label_*_ttc.npz 와 동일 (학습 즉시 호환).

사용:
  python scripts/gen_profile_pedal_labels.py --ds sangam_lead_0520 \
      --prebake .../f1_prebake_iy_sangam_lead_0520_v1_brakeaug_prev20.npz \
      --scc-dir .../scc_synth_p1_sangam_lead_0520 --profile p1
"""
from __future__ import annotations
import os, sys, argparse
from pathlib import Path
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # PedalLabelizer 의 pedal_map csv 가 repo-relative 경로
from src.utils.pedal_label import PedalLabelizer

D = os.environ.get("SCC_DATA_DIR", "/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data")
TARG_N = 40


def parse_vscc(path: Path):
    """scc.txt → (t_i (M,), v_scc (M,)). traj cols: i cum_s t_i v_orig v_scc lead_s gap."""
    t, v = [], []
    cur = None
    for line in open(path):
        line = line.rstrip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        if "=" in line and not line[0].isdigit():
            k = line.split("=", 1)[0].strip()
            cur = "traj" if k == "N" else ("stop" if k in ("long_traj_N", "lat_traj_N") else cur)
            continue
        if cur != "traj":
            continue
        p = line.split()
        try:
            nums = [float(x) for x in p]
        except ValueError:
            continue
        if len(nums) >= 5:
            t.append(nums[2]); v.append(nums[4])
    return np.asarray(t, np.float64), np.asarray(v, np.float64)


def vscc_to_ax(t: np.ndarray, v: np.ndarray) -> np.ndarray:
    """v_scc 프로파일 → ax (m/s^2). dt 불균일 + 정지구간 dt=0 (cum_s 정체) 대비:
    중심차분이되 dt=0 인 곳은 한쪽 차분으로 fallback (np.gradient 의 divide-by-zero 회피)."""
    n = len(v)
    if n < 2:
        return np.zeros_like(v)
    ax = np.zeros(n)
    for i in range(n):
        lo = max(i - 1, 0); hi = min(i + 1, n - 1)
        dt = t[hi] - t[lo]
        if dt > 1e-6:
            ax[i] = (v[hi] - v[lo]) / dt
        else:
            # 양쪽 dt=0 → 가까운 유효 이웃과의 차분 탐색
            j = i + 1
            while j < n and (t[j] - t[i]) <= 1e-6:
                j += 1
            if j < n:
                ax[i] = (v[j] - v[i]) / (t[j] - t[i])
            else:
                ax[i] = ax[i - 1] if i > 0 else 0.0
    return ax


def resample_40(t: np.ndarray, v: np.ndarray, ax: np.ndarray):
    """v_scc/ax 를 균일 0.1s 그리드(학습 target step) 40 step 으로. 부족하면 마지막 값 hold."""
    # long_traj target 은 0.1s step. v_scc traj 는 약 0.1s 간격이나 불균일 → 재샘플.
    if len(t) == 0:
        return np.zeros(TARG_N), np.zeros(TARG_N)
    tgrid = np.arange(TARG_N) * 0.1
    vg = np.interp(tgrid, t, v, left=v[0], right=v[-1])
    ag = np.interp(tgrid, t, ax, left=ax[0], right=ax[-1])
    return vg, ag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--prebake", required=True, help="기존 prebake npz (sample_keys/curr_keys 순서 기준)")
    ap.add_argument("--scc-dir", required=True, help="profile synth_batch 출력 dir (.scc.txt)")
    ap.add_argument("--profile", required=True, help="p1..p5 (출력 suffix)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None, help="디버그: 앞 N sample 만")
    args = ap.parse_args()

    pb = np.load(args.prebake, allow_pickle=True)
    sample_keys = np.asarray(pb["sample_keys"])
    curr_keys = np.asarray(pb["curr_keys"]).astype("U21")
    # 일부 prebake 는 curr_keys 가 비어있음 → sample_key 에서 suffix(__logged/__rand_NNNNN) 제거해 복원.
    if (curr_keys == "").all() or (curr_keys == "").mean() > 0.5:
        def _derive(sk: str) -> str:
            for sfx in ("__logged", "__rand_"):
                idx = sk.find(sfx)
                if idx >= 0:
                    return sk[:idx]
            return sk
        curr_keys = np.array([_derive(str(s)) for s in sample_keys], dtype="U21")
        print(f"[info] curr_keys empty → derived from sample_keys ({(curr_keys != '').sum()}/{len(curr_keys)} non-empty)")
    N = len(sample_keys) if args.limit is None else min(args.limit, len(sample_keys))
    scc_dir = Path(args.scc_dir)

    L = PedalLabelizer()
    vmat = np.zeros((N, TARG_N), np.float32)
    amat = np.zeros((N, TARG_N), np.float32)
    miss = 0
    for i in range(N):
        sp = scc_dir / f"{sample_keys[i]}.scc.txt"
        if not sp.exists():
            miss += 1
            continue
        t, v = parse_vscc(sp)
        if len(v) < 2:
            miss += 1
            continue
        ax = vscc_to_ax(t, v)
        vg, ag = resample_40(t, v, ax)
        vmat[i] = vg
        amat[i] = ag
    have = (vmat.sum(1) != 0) | (amat.sum(1) != 0)
    ped = L.label(vmat, amat).astype(np.float32)  # (N,40)
    # missing sample → pedal 0 (학습 시 mask 가능하나, 기존과 동일하게 그대로 저장)
    out = args.out or f"{D}/cache/pedal_label_{args.ds}_{args.profile}.npz"
    np.savez_compressed(out, keys=curr_keys[:N], pedal=ped)
    p0 = ped[:, 0]
    print(f"[{args.ds}/{args.profile}] N={N} miss={miss} ({100*miss/max(N,1):.0f}%) | "
          f"step0 pedal mean={p0[have].mean():+.3f} brake%={100*(p0[have] < 0).mean():.0f} | "
          f"ax_min10 mean={amat[have, :10].min(1).mean():+.2f} | "
          f"v_scc[0] mean={vmat[have, 0].mean():.1f} v_scc[39] mean={vmat[have, 39].mean():.1f}", flush=True)
    print(f"  → {out}", flush=True)


if __name__ == "__main__":
    main()
