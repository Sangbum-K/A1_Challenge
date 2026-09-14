"""Verify the DriverB_new traj fix.

(a) traj speed↔per-step-displacement corr (before vs after) — expect ~0.23 → ~0.99
(b) traj speed range — expect low/zero speeds present after fix
(c) controller_imitator #654 (SelfAttentionControlPredictor_Professor) frozen steer corr
    on the traj, before vs after. Reference: healthy sangam_lead_0520 → ~0.88.
"""
from __future__ import annotations
import os, sys, types, importlib.util
import numpy as np
import torch

DATA = "/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data"
ORIG = f"{DATA}/prebake/f1_prebake_iy_DriverB_new_logged_fixed_prev20.npz"
FIX  = f"{DATA}/prebake/f1_prebake_iy_DriverB_new_logged_fixed_prev20_trajfix.npz"
CKPT = "/home/ailab-12/git/scc_controller_imitator/data/ckpts/main_prof654.pth"
RTM  = "/home/ailab-12/git/A1_E2E/projects/mmdet3d_plugin/GenAD/control_integration/runtime_models.py"


def corr_disp_speed_full(traj, mask_valid):
    """Full-array corr: pool all (disp, speed) pairs from valid samples."""
    disps = []; sps = []
    for i in np.where(mask_valid)[0]:
        t = traj[i]
        xy = t[:, :2]
        d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        s = t[:-1, 2]
        disps.append(d); sps.append(s)
    disps = np.concatenate(disps); sps = np.concatenate(sps)
    return float(np.corrcoef(disps, sps)[0, 1])


def load_model():
    spec = importlib.util.spec_from_file_location("runtime_models", RTM)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = types.SimpleNamespace(
        model_name="SelfAttentionControlPredictor_Professor",
        tfr_d_model=128, tfr_nhead=8, tfr_num_decoder_layers=6,
        tfr_dim_feedforward=512, tfr_state_dim=6, tfr_seq_length=40,
        tfr_dropout=0.0,
        include_speed_profile=True, input_time_to_traj="none",
        target="steer_accel_brake", output_shape="control",
        pred_first_only=False,
    )
    model = mod.SelfAttentionControlPredictor_Professor(args)
    sd = torch.load(CKPT, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "model" in sd and all(not k.startswith(("traj_embed","state_embed","fc_out","transformer")) for k in sd):
        sd = sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  load_state_dict: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing: print("    missing[:5]:", missing[:5], flush=True)
    if unexpected: print("    unexpected[:5]:", unexpected[:5], flush=True)
    model.eval()
    return model


# input times for interpolating raw 0.1s traj
INTERP_T = np.array([0.05, 0.30, 0.55, 0.80, 1.05, 1.30], dtype=np.float32)


def build_traj6(traj40):
    """(N,40,3) ego [x_fwd,y_left,speed], 0.1s spacing → (N,6,3) interp at INTERP_T,
    reordered [-y_left, x_fwd, speed]."""
    N = traj40.shape[0]
    t_src = np.arange(40, dtype=np.float32) * 0.1
    out = np.zeros((N, 6, 3), dtype=np.float32)
    for j in range(3):
        # vectorized per-channel interp
        for i in range(N):
            out[i, :, j] = np.interp(INTERP_T, t_src, traj40[i, :, j])
    traj6 = np.stack([-out[:, :, 1], out[:, :, 0], out[:, :, 2]], axis=2)
    return traj6.astype(np.float32)


def build_ego6(ego5):
    """ego_5d = [vx, vy, ax, ay, yaw_rate] → ego6 = [vx,vy,ax,ay,tire,yaw_rate]
    tire = atan(2.35*yaw_rate/vx) if |vx|>=5 else 0."""
    vx, vy, ax, ay, yr = ego5[:, 0], ego5[:, 1], ego5[:, 2], ego5[:, 3], ego5[:, 4]
    tire = np.where(np.abs(vx) >= 5.0, np.arctan(2.35 * yr / np.where(np.abs(vx) < 1e-6, 1e-6, vx)), 0.0)
    return np.stack([vx, vy, ax, ay, tire, yr], axis=1).astype(np.float32)


@torch.no_grad()
def steer_corr(model, traj40, ego5, target, mask, device, max_n=40000):
    valid = mask[:, 0]  # first-step target valid
    idx = np.where(valid)[0]
    if len(idx) > max_n:
        rng = np.random.default_rng(0)
        idx = rng.choice(idx, max_n, replace=False)
        idx.sort()
    traj6 = build_traj6(traj40[idx])
    ego6 = build_ego6(ego5[idx])
    tgt = target[idx]
    preds = np.zeros(len(idx), dtype=np.float32)
    B = 2048
    for s in range(0, len(idx), B):
        e = min(s + B, len(idx))
        tj = torch.from_numpy(traj6[s:e]).to(device)
        eg = torch.from_numpy(ego6[s:e]).to(device)
        out = model(tj, eg)  # (b, 40, 2)
        preds[s:e] = (out[:, 0, 0] * 40.0).cpu().numpy()
    tgt_steer = tgt[:, 0, 0]
    return float(np.corrcoef(preds, tgt_steer)[0, 1]), len(idx)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[verify] device={device}", flush=True)

    do = np.load(ORIG, allow_pickle=True)
    df = np.load(FIX, allow_pickle=True)
    mask_valid = (do["sample_tags"] != "invalid") & np.array([bool(k) for k in do["curr_keys"]])
    print(f"  valid (non-invalid) samples: {mask_valid.sum()}", flush=True)

    # (a) disp-speed corr
    c_before = corr_disp_speed_full(do["traj"], mask_valid)
    c_after  = corr_disp_speed_full(df["traj"], mask_valid)
    print(f"\n(a) disp↔speed corr  BEFORE={c_before:.4f}  AFTER={c_after:.4f}", flush=True)

    # (b) speed range
    sb = do["traj"][mask_valid][:, :, 2]
    sa = df["traj"][mask_valid][:, :, 2]
    print(f"(b) speed range  BEFORE=[{sb.min():.2f},{sb.max():.2f}] mean={sb.mean():.2f}  "
          f"AFTER=[{sa.min():.2f},{sa.max():.2f}] mean={sa.mean():.2f}", flush=True)
    print(f"    AFTER speed<5 fraction = {(sa<5).mean():.3f}  (BEFORE {(sb<5).mean():.3f})", flush=True)

    # (c) #654 steer corr
    print("\n(c) loading #654 model ...", flush=True)
    model = load_model().to(device)
    sc_before, n_b = steer_corr(model, do["traj"], do["ego_5d"], do["target"], do["mask"], device)
    sc_after,  n_a = steer_corr(model, df["traj"], df["ego_5d"], df["target"], df["mask"], device)
    print(f"  #654 steer corr  BEFORE={sc_before:.4f} (n={n_b})  AFTER={sc_after:.4f} (n={n_a})", flush=True)
    print(f"  reference healthy sangam_lead_0520 ~0.88", flush=True)


if __name__ == "__main__":
    main()
