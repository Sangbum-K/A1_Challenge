#!/usr/bin/env python
"""speedway(spd_0522_40m) profile pedal — scc.txt 없이 prebake traj 로 생성.
recalc 의 replan_profile(vx-log s_min_eff 적용됨)/resample_40/PedalLabelizer 재사용.
입력: prebake traj(40,3 [x_fwd,y_left,speed]) → cum_s(arc)/v_orig(speed)/t_i(0.1s),
      inlane_gt → lead_s(=inlane_dist const, lead 거의 정지 가정), EGO_INFO → v_ego.
출력: pedal_label_spd_0522_40m_{p}.npz  (recalc 산출과 동일 스키마 {keys, pedal(N,40)})."""
import numpy as np, sys, os
ROOT = "/home/ailab-12/git/scc_controller_imitator"
sys.path.insert(0, f"{ROOT}/scripts"); os.chdir(ROOT)
from gen_profile_pedal_labels_recalc import (
    replan_profile, resample_40, load_ego_v0, PROFILES, derive_curr_key, TARG_N, D)
from src.utils.pedal_label import PedalLabelizer

PROFILE = sys.argv[1] if len(sys.argv) > 1 else "p3"
cfg = PROFILES[PROFILE]
DS = "spd_0522_40m"
pb = np.load(f"{D}/prebake/f1_prebake_iy_{DS}_logged_prev20.npz", allow_pickle=True)
sk = pb["sample_keys"].astype(str); traj = pb["traj"]
ck = np.array([derive_curr_key(s) for s in sk], dtype="U21")
ig = np.load(f"{D}/cache/inlane_gt_{DS}.npz", allow_pickle=True)
_igk = [str(x) for x in ig["keys"]]
idist = dict(zip(_igk, np.asarray(ig["inlane_dist"], np.float64)))
iclose = dict(zip(_igk, np.asarray(ig["inlane_closing"], np.float64)))
ego_v0, nf, miss = load_ego_v0(f"{D}/meta_{DS}.json", DS)
ego_t21 = {}
for k, v in ego_v0.items():
    ego_t21.setdefault(k[:21], v)
print(f"[ego] frames={nf} ok={len(ego_v0)} miss={miss}", flush=True)

N = len(sk); L = PedalLabelizer()
vmat = np.zeros((N, TARG_N), np.float32); amat = np.zeros((N, TARG_N), np.float32)
nlead = 0; nz = 0
for i in range(N):
    t = traj[i]; x = t[:, 0].astype(np.float64); y = t[:, 1].astype(np.float64); spd = t[:, 2].astype(np.float64)
    if spd.max() <= 0:
        nz += 1; continue
    ds = np.sqrt(np.diff(x, prepend=x[0]) ** 2 + np.diff(y, prepend=y[0]) ** 2)
    cum_s = np.cumsum(ds); v_orig = spd; t_i = 0.1 * np.arange(len(spd))
    d0 = idist.get(str(ck[i]), np.inf)
    lead_s = np.full(len(spd), float(d0)) if np.isfinite(d0) else np.full(len(spd), np.nan)
    if np.isfinite(d0): nlead += 1
    ve = ego_v0.get(str(ck[i])) or ego_t21.get(str(ck[i])[:21]) or float(v_orig[0])
    v, ax = replan_profile(cum_s, t_i, v_orig, lead_s, ve, **cfg)
    vg, ag = resample_40(t_i, v, ax)
    vmat[i] = vg; amat[i] = ag
ped = L.label(vmat, amat).astype(np.float32)
out = f"{D}/cache/pedal_label_{DS}_{PROFILE}.npz"
np.savez_compressed(out, keys=ck, pedal=ped)
print(f"[{DS}/{PROFILE}] N={N} lead={nlead} zerospeed={nz} -> {out}")
print(f"  pedal[:,0] mean={ped[:,0].mean():.3f} brake%={(ped[:,0] < 0).mean()*100:.1f} "
      f"min={ped[:,0].min():.2f} max={ped[:,0].max():.2f}")
