"""SCC 5-profile pedal_label python 재계산 — C++ synth_batch/SCC replanner 재실행 없이
기존 scc.txt + EGO_INFO 만 재사용.

배경 / 근거 (confirmed 2026-05-27):
  - scc.txt traj 블록은 이미 cum_s / t_i / v_orig / v_scc / lead_s / gap 을 담는다.
    v_scc/gap 은 그 dir 합성 시 쓰인 (단일) profile 값이라 그대로는 못 쓰지만, lead_s 와
    cum_s / t_i / v_orig 은 profile 무관 — 이것만 재사용하면 5 profile 의 v_safe 를 python 으로
    재계산할 수 있다. SCC replanner 로직이 단순(v_safe=(gap-s_min)/t_headway + decel/accel pass).
  - EGO_INFO velocity[0]=vx 가 그 frame 의 실제 ego 시작속도 (vehicle frame, vy≈0). v_orig[0]/
    prebake ego_5d 는 rand 변형 trajectory 와 불일치(예: ego vx=3.56 vs v_orig[0]=7.63) → 못 씀.
    anchor v0 는 frame(curr_key) 단위로 그 frame 의 모든 rand 에 공유.

파이프라인 (per frame, per profile) — scc_speed_replanner.cpp(line 147-179) + ego-start forward:
  1) gap    = lead_s - cum_s             (no-lead 면 v_safe=+inf → SCC 무개입)
  2) v_safe = (gap - s_min)/t_headway
  3) v_scc  = min(v_orig, max(0, v_safe))
  4) backward: i=N-2..0  v[i]=min(v[i], v[i+1] + MAX_DECEL*dt_i)   (MAX_DECEL=-a_min=20)
  5) forward (★ego-start, 원 replanner 에 없는 feasibility):
       v[0]=v_ego (EGO_INFO vx)
       i=1..N-1  v[i]=clip(v[i], v[i-1]+A_MIN*dt_i, v[i-1]+A_MAX*dt_i)
  6) ax[i]  = (v[i]-v[i-1])/dt_i
  7) resample → 0.1s 균일 40 step → PedalLabelizer(v, ax) → pedal_label_<ds>_p{N}.npz

출력 형식: { keys:(N,)<U21=curr_keys, pedal:(N,40) f32 }  — 기존 pedal_label_*_ttc 와 동일.

사용:
  python scripts/gen_profile_pedal_labels_recalc.py --ds sangam_lead_0520 \
      --prebake .../f1_prebake_iy_sangam_lead_0520_v1_brakeaug_prev20.npz \
      --scc-dir .../scc_synth_v3_sangam_lead_0520 \
      --meta    .../meta_sangam_lead_0520.json \
      --profile p1
"""
from __future__ import annotations
import os, sys, json, argparse, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # PedalLabelizer 의 pedal_map csv 가 repo-relative 경로
from src.utils.pedal_label import PedalLabelizer

D = os.environ.get("SCC_DATA_DIR", "/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data")
RAW_BASE = os.environ.get("SCC_RAW_BASE", "/home/ailab-12/AILabSSD/02_Data/57_A1_E2E/00_raw")
TARG_N = 40

# 5 profile + 한계 (확정)
PROFILES = {
    "p1": dict(t_headway=0.4, s_min=2.0),
    "p2": dict(t_headway=0.8, s_min=3.0),
    "p3": dict(t_headway=1.2, s_min=4.0),
    "p4": dict(t_headway=1.6, s_min=5.5),
    "p5": dict(t_headway=2.0, s_min=7.0),
}
A_MAX = 5.0     # m/s^2  (forward 가속 상한)
A_MIN = -20.0   # m/s^2  (forward 감속 하한)
MAX_DECEL = -A_MIN  # 20, backward pass 의 양수 decel limit


def parse_scc_traj(path: Path):
    """scc.txt traj 블록 → (cum_s, t_i, v_orig, lead_s) 각 (M,) np.float64.
    header `# i  cum_s  t_i  v_orig  v_scc  lead_s  gap`. lead_s='nan' → np.nan."""
    cum_s, t_i, v_orig, lead_s = [], [], [], []
    cur = None
    for line in open(path):
        line = line.rstrip()
        if not line:
            continue
        if line.startswith("# i  cum_s"):
            cur = "traj"; continue
        if line.startswith("#"):
            continue
        if line.startswith("in_lane_obj_ids:"):
            continue
        if "=" in line and not line[0].isdigit():
            k = line.split("=", 1)[0].strip()
            # traj 블록은 `N=` 이후 시작, long_traj_N/lat_traj_N 이후 종료
            if k in ("long_traj_N", "lat_traj_N"):
                cur = None
            continue
        if cur != "traj":
            continue
        p = line.split()
        if len(p) < 7:
            continue
        try:
            cum_s.append(float(p[1])); t_i.append(float(p[2]))
            v_orig.append(float(p[3]))
            lead_s.append(float(p[5]))  # 'nan' → np.nan
        except ValueError:
            continue
    return (np.asarray(cum_s, np.float64), np.asarray(t_i, np.float64),
            np.asarray(v_orig, np.float64), np.asarray(lead_s, np.float64))


def replan_profile(cum_s, t_i, v_orig, lead_s, v_ego, t_headway, s_min):
    """5-profile SCC replan + ego-start forward. → v (M,), ax (M,)."""
    n = len(v_orig)
    if n == 0:
        return np.zeros(0), np.zeros(0)
    v = v_orig.copy()

    # 1)-3) lead-based v_safe
    has_lead = ~np.isnan(lead_s)
    if has_lead.any():
        # SCC 활성화 거리 = vx-종속 log: s_min_eff = s_min + alpha·ln(1+v_ego/v0). 빠를수록 ↑(제동 일찍), 고속 완만.
        s_min_eff = s_min + float(os.environ.get("SCC_LOG_ALPHA", 6.0)) * float(np.log1p(v_ego / float(os.environ.get("SCC_LOG_V0", 5.0))))
        gap = lead_s - cum_s
        v_safe = (gap - s_min_eff) / t_headway
        v_safe = np.maximum(0.0, v_safe)
        v[has_lead] = np.minimum(v_orig[has_lead], v_safe[has_lead])

    # 4) backward pass (미리 감속): v[i] <= v[i+1] + MAX_DECEL*dt_i
    for i in range(n - 2, -1, -1):
        dt = t_i[i + 1] - t_i[i]
        if dt > 1e-9:
            lim = v[i + 1] + MAX_DECEL * dt
            if v[i] > lim:
                v[i] = lim

    # 5) forward pass (ego-start): v[0]=v_ego, clip per-step accel/decel
    v[0] = max(0.0, float(v_ego))
    for i in range(1, n):
        dt = t_i[i] - t_i[i - 1]
        if dt > 1e-9:
            lo = v[i - 1] + A_MIN * dt
            hi = v[i - 1] + A_MAX * dt
            v[i] = min(max(v[i], lo), hi)
    v = np.maximum(0.0, v)

    # 6) ax (제어값): backward diff per dt_i. ax[0]=ax[1].
    ax = np.zeros(n)
    for i in range(1, n):
        dt = t_i[i] - t_i[i - 1]
        ax[i] = (v[i] - v[i - 1]) / dt if dt > 1e-9 else (ax[i - 1] if i > 0 else 0.0)
    if n >= 2:
        ax[0] = ax[1]
    return v, ax


def resample_40(t, v, ax):
    """v/ax 를 0.1s 균일 40 step grid 로 interp (t 는 scc.txt t_i, 불균일)."""
    if len(t) == 0:
        return np.zeros(TARG_N), np.zeros(TARG_N)
    tgrid = np.arange(TARG_N) * 0.1
    vg = np.interp(tgrid, t, v, left=v[0], right=v[-1])
    ag = np.interp(tgrid, t, ax, left=ax[0], right=ax[-1])
    return vg, ag


def _parse_ego_vx(args_tuple):
    """worker: EGO_INFO velocity[0]=vx (vehicle frame). → (curr_key, vx or None)."""
    ck, track, policy, run_id, tick = args_tuple
    ep = Path(RAW_BASE) / track / policy / run_id / "EGO_INFO" / f"{tick}.txt"
    try:
        for line in open(ep):
            if line.startswith("velocity:"):
                nums = [float(x) for x in line.split(":", 1)[1].split()]
                return (ck, nums[0] if nums else None)
    except (FileNotFoundError, OSError):
        return (ck, None)
    return (ck, None)


def load_ego_v0(meta_path: str, ds: str, workers: int = 16):
    """meta.json → {curr_key: v_ego}. curr_key = '{run_id}__t{tick:07d}'.
    EGO_INFO velocity[0]=vx (vehicle frame). frame 단위 1회 parse (rand 전부 공유).
    결과는 cache/ego_v0_<ds>.npz 에 저장 — profile 25 회 재실행 시 NFS 재read 회피."""
    cache_p = Path(D) / "cache" / f"ego_v0_{ds}.npz"
    if cache_p.exists():
        z = np.load(cache_p, allow_pickle=True)
        ego_v0 = {str(k): float(v) for k, v in zip(z["keys"], z["v0"])}
        return ego_v0, len(ego_v0), 0
    meta = json.load(open(meta_path))
    seen = {}
    for m in meta:
        ck = f"{m['run_id']}__t{int(m['tick']):07d}"
        if ck in seen:
            continue
        seen[ck] = (m["track"], m["policy"], m["run_id"], int(m["tick"]))
    args = [(ck, *seen[ck]) for ck in seen]
    ego_v0 = {}
    miss = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for ck, vx in ex.map(_parse_ego_vx, args, chunksize=64):
            if vx is None:
                miss += 1
            else:
                ego_v0[ck] = vx
    if ego_v0:
        cache_p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_p,
                            keys=np.array(list(ego_v0.keys())),
                            v0=np.array(list(ego_v0.values()), np.float32))
    return ego_v0, len(seen), miss


def derive_curr_key(sk: str) -> str:
    for sfx in ("__logged", "__rand_"):
        idx = sk.find(sfx)
        if idx >= 0:
            return sk[:idx]
    return sk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--prebake", required=True, help="prebake npz (sample_keys 순서 기준)")
    ap.add_argument("--scc-dir", required=True, help="scc.txt dir (scc_synth_v3_* / scc_synth_p1_*)")
    ap.add_argument("--meta", required=True, help="meta_<ds>.json (EGO_INFO 경로 + tick)")
    ap.add_argument("--profile", required=True, help="p1..p5")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ego-workers", type=int, default=16, help="EGO_INFO parse 병렬 (CPU 2 reserve)")
    args = ap.parse_args()
    if args.profile not in PROFILES:
        raise SystemExit(f"unknown profile {args.profile}; choose {list(PROFILES)}")
    cfg = PROFILES[args.profile]

    pb = np.load(args.prebake, allow_pickle=True)
    sample_keys = np.asarray(pb["sample_keys"]).astype(str)
    curr_keys = np.asarray(pb["curr_keys"]).astype("U21") if "curr_keys" in pb.files else None
    sample_tags = np.asarray(pb["sample_tags"]).astype(str) if "sample_tags" in pb.files else None

    # sample_keys 가 INVALID_* 면 scc.txt 파일명 복원 불가 → curr_keys + tag 로 재구성
    sk_invalid = np.char.startswith(sample_keys, "INVALID")
    if sk_invalid.any() and curr_keys is not None and sample_tags is not None:
        # rebuild sample_keys: '{curr_key}__{tag}' (rand_ tag 은 그대로 못 복원하나
        # DriverB 는 logged only → tag='logged')
        rebuilt = np.array([
            (f"{ck}__{tg}" if tg == "logged" else sk)
            for sk, ck, tg in zip(sample_keys, curr_keys, sample_tags)
        ], dtype=object)
        n_fix = int((rebuilt != sample_keys).sum())
        sample_keys = rebuilt.astype(str)
        print(f"[info] {n_fix} INVALID sample_keys rebuilt from curr_keys+tag", flush=True)

    if curr_keys is None or (curr_keys == "").mean() > 0.5:
        curr_keys = np.array([derive_curr_key(s) for s in sample_keys], dtype="U21")
        print(f"[info] curr_keys derived from sample_keys "
              f"({(curr_keys != '').sum()}/{len(curr_keys)} non-empty)", flush=True)

    N = len(sample_keys) if args.limit is None else min(args.limit, len(sample_keys))
    scc_dir = Path(args.scc_dir)

    print(f"[ego] loading EGO_INFO v0 from {RAW_BASE} via {args.meta} ...", flush=True)
    t0 = time.time()
    ego_v0, n_frames, n_miss_ego = load_ego_v0(args.meta, args.ds, workers=args.ego_workers)
    # prebake curr_keys 는 <U21 로 truncate 되어 있어, 긴 run_id frame(예: 260521_13_seg0__t0021620
    # → 260521_13_seg0__t0021) 은 full EGO key 와 매칭 안 됨. EGO key 도 U21 truncate 한
    # fallback index 추가 (충돌 시 인접 tick 이라 v0 거의 동일 → 첫 값 사용).
    ego_v0_t21 = {}
    for k, v in ego_v0.items():
        kk = k[:21]
        if kk not in ego_v0_t21:
            ego_v0_t21[kk] = v
    print(f"[ego] frames={n_frames} ego_ok={len(ego_v0)} miss={n_miss_ego} "
          f"in {time.time()-t0:.1f}s | v0 mean={np.mean(list(ego_v0.values())):.2f} "
          f"max={max(ego_v0.values()):.2f}", flush=True)

    L = PedalLabelizer()
    vmat = np.zeros((N, TARG_N), np.float32)
    amat = np.zeros((N, TARG_N), np.float32)
    miss = 0; no_ego = 0; n_lead = 0
    v0_applied = np.zeros(N, bool)
    t0 = time.time()
    for i in range(N):
        sp = scc_dir / f"{sample_keys[i]}.scc.txt"
        if not sp.exists():
            miss += 1
            continue
        cum_s, t_i, v_orig, lead_s = parse_scc_traj(sp)
        if len(v_orig) < 2:
            miss += 1
            continue
        ck = str(curr_keys[i])
        v_ego = ego_v0.get(ck)
        if v_ego is None:
            v_ego = ego_v0_t21.get(ck[:21])  # U21-truncated fallback
        if v_ego is None:
            no_ego += 1
            v_ego = v_orig[0]  # fallback (드물어야 함)
        else:
            v0_applied[i] = True
        if (~np.isnan(lead_s)).any():
            n_lead += 1
        v, ax = replan_profile(cum_s, t_i, v_orig, lead_s, v_ego, **cfg)
        vg, ag = resample_40(t_i, v, ax)
        vmat[i] = vg
        amat[i] = ag
        if (i + 1) % 20000 == 0:
            print(f"  [{i+1}/{N}] rate={(i+1)/(time.time()-t0):.0f}/s "
                  f"lead%={100*n_lead/(i+1):.0f}", flush=True)

    have = (vmat.sum(1) != 0) | (amat.sum(1) != 0)
    ped = L.label(vmat, amat).astype(np.float32)
    out = args.out or f"{D}/cache/pedal_label_{args.ds}_{args.profile}.npz"
    np.savez_compressed(out, keys=curr_keys[:N], pedal=ped)

    p0 = ped[:, 0]
    print(f"[{args.ds}/{args.profile}] N={N} miss={miss} ({100*miss/max(N,1):.0f}%) "
          f"no_ego={no_ego} lead_frames={n_lead} | v0_applied={int(v0_applied.sum())} | "
          f"step0 pedal mean={p0[have].mean():+.3f} brake%={100*(p0[have] < 0).mean():.0f} | "
          f"ax_min10 mean={amat[have, :10].min(1).mean():+.2f} | "
          f"v[0] mean={vmat[have, 0].mean():.2f} v[39] mean={vmat[have, 39].mean():.2f}", flush=True)
    print(f"  -> {out}", flush=True)


if __name__ == "__main__":
    main()
