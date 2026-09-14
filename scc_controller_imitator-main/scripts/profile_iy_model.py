"""IyModel forward 모듈별 시간 측정.

batch_size 별 + use_obj_gate on/off 비교.
torch.cuda.Event 로 stage 별 측정 (sync).
"""
from __future__ import annotations
import os, sys, time, argparse
import torch
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.model.image_models.iy_model import IyModel, yaw_shift_tokens_batch


def time_section(events, label, device):
    """events: list[(name, start_event, end_event)]"""
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    return (label, s, e)


def profile_step(model: IyModel, b: dict, device: str):
    """run forward with per-module timing. returns dict label → ms."""
    torch.cuda.synchronize()
    times = {}

    # warm prepare batch
    fx_b = b.get("fx", model.fx)

    # i_y
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    prev_shifted = yaw_shift_tokens_batch(b["tok_prev_raw"], b["d_yaw"], fx=fx_b)
    e.record(); torch.cuda.synchronize()
    times["i_y (token roll)"] = s.elapsed_time(e)

    # img_proj curr
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    tc = model._proj_tok(b["tok_curr"])
    tp = model._proj_tok(prev_shifted)
    e.record(); torch.cuda.synchronize()
    times["img_proj (Conv1d k=1) ×2"] = s.elapsed_time(e)

    # i_2 fusion
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    ego_mot = b["ego_5d"][:, :2]
    fi_mem = model.img_fusion(tc, tp, ego_mot)
    e.record(); torch.cuda.synchronize()
    times["i_2 ImageFusion (3 attn)"] = s.elapsed_time(e)

    # i_g obj gate
    if model.use_obj_gate:
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        obj_logit = model.obj_head(fi_mem).squeeze(-1)
        gate = torch.sigmoid(obj_logit).unsqueeze(-1)
        fi_mem = fi_mem * gate
        e.record(); torch.cuda.synchronize()
        times["i_g (obj_head + gate)"] = s.elapsed_time(e)

    # traj/state embed + pos_enc
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    traj = b["traj_40"][:, :, :model.traj_dim]
    state = b["ego_5d"][:, :model.state_dim]
    if state.shape[-1] < model.state_dim:
        pad = torch.zeros(state.shape[0], model.state_dim - state.shape[-1],
                          dtype=state.dtype, device=state.device)
        state = torch.cat([state, pad], dim=-1)
    traj_tok = model.traj_embed(traj)
    state_tok = model.state_embed(state).unsqueeze(1)
    tokens_text = model.pos_encoder(traj_tok + state_tok)
    e.record(); torch.cuda.synchronize()
    times["traj+state embed + pos_enc"] = s.elapsed_time(e)

    # concat + image_pos
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    image_mem = fi_mem + model.image_pos_emb
    tokens = torch.cat([tokens_text, image_mem], dim=1)
    e.record(); torch.cuda.synchronize()
    times["concat traj+image"] = s.elapsed_time(e)

    # transformer 6L (main cost)
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    encoded = model.transformer_encoder(tokens)
    e.record(); torch.cuda.synchronize()
    times["transformer (6L self-attn)"] = s.elapsed_time(e)

    # fc_out
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    pred = model.fc_out(encoded[:, :model.traj_seq_len])
    e.record(); torch.cuda.synchronize()
    times["fc_out (D→2)"] = s.elapsed_time(e)

    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--c-b", type=int, default=256)
    ap.add_argument("--n-warmup", type=int, default=5)
    ap.add_argument("--n-trial", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device

    for B in args.batch:
        print(f"\n{'='*70}")
        print(f"BATCH SIZE = {B}, C_b = {args.c_b}")
        print('='*70)

        # build dummy batch
        b = {
            "tok_curr":     torch.randn(B, 40, args.c_b, device=device),
            "tok_prev_raw": torch.randn(B, 40, args.c_b, device=device),
            "d_yaw":        torch.randn(B, device=device) * 0.05,
            "fx":           torch.full((B,), 640.0, device=device),
            "ego_5d":       torch.randn(B, 5, device=device),
            "traj_40":      torch.randn(B, 40, 3, device=device),
        }

        for use_gate in [True, False]:
            print(f"\n--- use_obj_gate={use_gate} ---")
            model = IyModel(C_b=args.c_b, d_model=128, use_obj_gate=use_gate).to(device).eval()
            # warmup
            with torch.no_grad():
                for _ in range(args.n_warmup):
                    profile_step(model, b, device)

            agg = {}
            with torch.no_grad():
                for _ in range(args.n_trial):
                    t = profile_step(model, b, device)
                    for k, v in t.items():
                        agg.setdefault(k, []).append(v)

            total = sum(np.mean(vs) for vs in agg.values())
            print(f"  {'module':40s}  {'ms':>8s}  {'%':>6s}  {'samples/sec':>12s}")
            for k, vs in agg.items():
                mean = np.mean(vs)
                pct = mean / total * 100
                sps = B / (mean / 1000)
                print(f"  {k:40s}  {mean:8.2f}  {pct:6.1f}  {sps:12.0f}")
            print(f"  {'TOTAL':40s}  {total:8.2f}  {100.0:6.1f}  {B/(total/1000):12.0f}")


if __name__ == "__main__":
    main()
