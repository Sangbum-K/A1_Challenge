"""신규 MPC (mpc/ fork) 결과 정성 검증 — scc.txt 파일 직접 파싱.

한 frame composite (5 panel):
  (top) ROI 카메라 + ego 속도 큰 글자 표시
  (B) BEV: input trajectory (logged + N rand bank) color=speed
  (C) ax curves overlay (logged 굵게, rand 얇게)
  (D) steer curves overlay
  (E) vx curves overlay (각 trajectory 별 MPC가 추적한 속도)

사용:
  python scripts/viz_new_mpc_samples.py \
      --synth-dir /tmp/scc_synth_v3_smoke \
      --raw-base /home/ailab-12/AILabDatasetLocal/.../raw_cache \
      --track speedway_track --policy 0519_agent1_pertX_pitX \
      --n 8 --n-rand 12 \
      --out outputs_multi/viz_new_mpc
"""
from __future__ import annotations
import os, sys, json, argparse, glob, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.datamodule.scc_synth_pair_dataset import _parse_scc_txt, _parse_ego

DEG2RAD = np.pi / 180.0
ROI_Y0, ROI_Y1 = 235, 485
IMG_W = 1280
BIN_W = IMG_W / 40


def load_bank(path):
    import struct
    if not Path(path).exists(): return None
    with open(path, "rb") as f:
        K, N = struct.unpack("ii", f.read(8))
        data = np.frombuffer(f.read(K * N * 4 * 4), dtype=np.float32).reshape(K, N, 4)
    return data.copy()


def world_to_ego(pts, ex, ey, eyaw):
    dx = pts[..., 0] - ex; dy = pts[..., 1] - ey
    c, s = np.cos(-eyaw), np.sin(-eyaw)
    return np.stack([c*dx - s*dy, s*dx + c*dy], axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-dir", required=True, help="새 MPC output (scc.txt)")
    ap.add_argument("--raw-base", required=True, help="raw_cache root (track/policy/run)")
    ap.add_argument("--track",  default="speedway_track")
    ap.add_argument("--policy", default="0519_agent1_pertX_pitX")
    ap.add_argument("--bank-dir", default=None,
                    help="random K bank (.trajs.bin) 폴더 — BEV 에 input trajectories 표시")
    ap.add_argument("--aux-dir",  default=None, help="aux_targets dir for obj overlay")
    ap.add_argument("--n",       type=int, default=8)
    ap.add_argument("--n-rand",  type=int, default=8)
    ap.add_argument("--sample",  choices=["head","stride","random"], default="head",
                    help="frame sampling: head (first n) / stride (evenly spaced) / random")
    ap.add_argument("--seed",    type=int, default=0)
    ap.add_argument("--out",     required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # group scc files by frame (logged + rand_*)
    files = sorted(glob.glob(f"{args.synth_dir}/*__logged.scc.txt"))
    print(f"[viz] logged frames: {len(files)}")
    if args.sample == "stride" and len(files) > args.n:
        idx = np.linspace(0, len(files)-1, args.n).astype(int)
        files = [files[i] for i in idx]
    elif args.sample == "random" and len(files) > args.n:
        rs = np.random.RandomState(args.seed)
        idx = sorted(rs.choice(len(files), size=args.n, replace=False))
        files = [files[i] for i in idx]
    else:
        files = files[:args.n]
    frames = []
    for f in files:
        # filename pattern: {run}__t{tick:07d}__logged.scc.txt
        m = re.match(r"(.+?)__t(\d+)__logged\.scc\.txt", Path(f).name)
        if not m: continue
        run, tick = m.group(1), int(m.group(2))
        frames.append((f, run, tick))

    for fi, (logged_f, run, tick) in enumerate(frames):
        # parse logged
        r_log = _parse_scc_txt(Path(logged_f))
        if r_log is None or r_log["long_traj"] is None: continue
        long_log = r_log["long_traj"]   # (N, 8) i time vx ax steer trq pedal brake
        lat_log = r_log["lat_traj"]      # (N, 5) i time vx ax steer

        # parse random
        rand_files = sorted(glob.glob(f"{args.synth_dir}/{run}__t{tick:07d}__rand_*.scc.txt"))[:args.n_rand]
        rand_data = []
        for rf in rand_files:
            r = _parse_scc_txt(Path(rf))
            if r and r["long_traj"] is not None:
                rand_data.append((r["long_traj"], r["lat_traj"]))

        # raw image
        img_p = Path(args.raw_base) / args.track / args.policy / run / "CAMERA_1" / f"{tick}_Intensity.jpeg"
        if not img_p.exists(): print(f"  skip img {img_p}"); continue
        img = np.array(Image.open(img_p).convert("RGB"))
        roi = img[ROI_Y0:ROI_Y1]

        # raw ego (current + future for BEV)
        ego_d = Path(args.raw_base) / args.track / args.policy / run / "EGO_INFO"
        ego_curr = _parse_ego(ego_d / f"{tick}.txt")
        if ego_curr is None: print(f"  skip ego {tick}"); continue
        ex, ey = ego_curr["position"][0], ego_curr["position"][1]
        eyaw = ego_curr["orientation"][2] * DEG2RAD
        vx_init = ego_curr["velocity"][0]
        # future ego (for plot + reference speed for logged)
        future_ego_xy = []; future_ego_vx = []
        for k in range(1, 41):
            ego_k = _parse_ego(ego_d / f"{tick + k*5}.txt")
            if ego_k is None: break
            future_ego_xy.append([ego_k["position"][0], ego_k["position"][1]])
            future_ego_vx.append(ego_k["velocity"][0])
        future_ego_xy = np.array(future_ego_xy) if future_ego_xy else None
        future_ego_vx = np.array(future_ego_vx) if future_ego_vx else None

        # obj from aux
        obj = np.zeros(40)
        if args.aux_dir:
            ap2 = Path(args.aux_dir) / f"{args.track}__{args.policy}__{run}__t{tick:07d}.npz"
            if ap2.exists():
                obj = (np.load(ap2)["target"][:, 0] > 0.5).astype(float)

        # input trajectory bank (BEV)
        bank = None
        if args.bank_dir:
            bank = load_bank(Path(args.bank_dir) / f"{run}__t{tick:07d}.trajs.bin")
            if bank is not None and len(bank) > args.n_rand:
                idx = np.random.RandomState(fi).choice(len(bank), size=args.n_rand, replace=False)
                bank = bank[idx]

        # ── figure ──
        fig = plt.figure(figsize=(18, 11), facecolor="white")
        gs = fig.add_gridspec(2, 4, height_ratios=[0.9, 1.3])

        # (top) image + ego vx large title
        ax_img = fig.add_subplot(gs[0, :])
        H = ROI_Y1 - ROI_Y0
        pil = Image.fromarray(roi.copy()); draw = ImageDraw.Draw(pil)
        for b in range(40):
            if obj[b] > 0.5:
                draw.rectangle([b*BIN_W, 0, (b+1)*BIN_W-1, H-1], outline=(255, 60, 60), width=3)
        ax_img.imshow(np.array(pil)); ax_img.axis("off")
        ax_img.set_title(f"{run}__t{tick:07d}    ★ ego vx = {vx_init:.2f} m/s ({vx_init*3.6:.1f} km/h)    "
                          f"obj_bins={int(obj.sum())}", fontsize=14, fontweight="bold")

        # (B) BEV input trajectories
        ax_bev = fig.add_subplot(gs[1, 0])
        if future_ego_xy is not None:
            fe = world_to_ego(future_ego_xy, ex, ey, eyaw)
            ax_bev.plot(fe[:, 0], fe[:, 1], 'k--', linewidth=1.5, label="raw future ego", alpha=0.6)
        if bank is not None:
            for k in range(bank.shape[0]):
                be = world_to_ego(bank[k, :, :2], ex, ey, eyaw)
                speeds = bank[k, :, 3]
                pts = np.array([be[:, 0], be[:, 1]]).T.reshape(-1, 1, 2)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                lc = LineCollection(segs, cmap="viridis", linewidths=1.0, alpha=0.7)
                lc.set_array(speeds[:-1])
                ax_bev.add_collection(lc)
        ax_bev.scatter([0], [0], color="red", s=150, marker="^", zorder=10, label="ego")
        ax_bev.set_xlim(-3, 80); ax_bev.set_ylim(-25, 25)
        ax_bev.set_xlabel("x forward (m)"); ax_bev.set_ylabel("y left (m)")
        ax_bev.set_aspect("equal"); ax_bev.grid(True, alpha=0.3)
        ax_bev.set_title("input BEV (bank color=speed, dashed=raw future ego)", fontsize=10)
        ax_bev.legend(fontsize=8, loc="upper left")

        # (C) ax curves
        ax_ax = fig.add_subplot(gs[1, 1])
        for long_t, _ in rand_data:
            ax_ax.plot(long_t[:, 1], long_t[:, 3], "-", color="lightcoral", linewidth=0.8, alpha=0.5)
        ax_ax.plot(long_log[:, 1], long_log[:, 3], "r-s", markersize=3, linewidth=2, label="logged")
        ax_ax.axhline(0, color="k", linestyle=":", linewidth=0.5)
        ax_ax.axhline(15, color="g", linestyle=":", alpha=0.5, label="±15 cap")
        ax_ax.axhline(-15, color="g", linestyle=":", alpha=0.5)
        ax_ax.set_xlabel("time (s)"); ax_ax.set_ylabel("ax (m/s²)")
        ax_ax.set_title("ax (logged + rand)", fontsize=10); ax_ax.grid(True, alpha=0.3); ax_ax.legend(fontsize=7)
        ax_ax.set_ylim(-18, 18)

        # (D) steer curves
        ax_st = fig.add_subplot(gs[1, 2])
        for _, lat_t in rand_data:
            ax_st.plot(lat_t[:, 1], lat_t[:, 4], "-", color="lightblue", linewidth=0.8, alpha=0.5)
        ax_st.plot(lat_log[:, 1], lat_log[:, 4], "b-o", markersize=3, linewidth=2, label="logged")
        ax_st.axhline(0, color="k", linestyle=":", linewidth=0.5)
        ax_st.set_xlabel("time (s)"); ax_st.set_ylabel("steer (deg)")
        ax_st.set_title("steer (logged + rand)", fontsize=10); ax_st.grid(True, alpha=0.3); ax_st.legend(fontsize=7)

        # (E) vx — reference vs MPC output
        ax_vx = fig.add_subplot(gs[1, 3])
        # reference speeds:
        #   rand: from input bank speed column (interpolated to MPC time)
        #   logged: from raw future ego vx (sampled every 50ms = 5 tick)
        if bank is not None:
            for k in range(bank.shape[0]):
                # bank: 40 waypoints @ 100ms
                t_ref = np.arange(bank.shape[1]) * 0.1
                ax_vx.plot(t_ref, bank[k, :, 3], "-", color="gray", linewidth=0.6, alpha=0.35)
        # rand MPC output
        for long_t, _ in rand_data:
            ax_vx.plot(long_t[:, 1], long_t[:, 2], "-", color="lightgreen", linewidth=0.8, alpha=0.45)
        # logged reference (raw future ego)
        if future_ego_vx is not None:
            t_log_ref = np.arange(len(future_ego_vx)) * 0.05    # 5 tick = 50ms
            ax_vx.plot(t_log_ref, future_ego_vx, "k-D", markersize=3, linewidth=1.5, label="logged REF (raw future ego)", alpha=0.7)
        # logged MPC output
        ax_vx.plot(long_log[:, 1], long_log[:, 2], "g-^", markersize=3, linewidth=2, label="logged MPC out")
        ax_vx.axhline(vx_init, color="orange", linestyle="--", linewidth=1, label=f"init {vx_init:.1f}")
        ax_vx.set_xlabel("time (s)"); ax_vx.set_ylabel("vx (m/s)")
        ax_vx.set_title("vx: REF (gray=bank, black=raw future) vs MPC (green/light green)", fontsize=9)
        ax_vx.grid(True, alpha=0.3); ax_vx.legend(fontsize=6, loc="best")

        out_p = args.out / f"{fi:02d}_{run}__t{tick:07d}.png"
        fig.savefig(out_p, dpi=80, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{fi+1}/{len(frames)}] {out_p.name}  vx={vx_init:.1f} m/s")

    # html
    pngs = sorted([p.name for p in args.out.glob("*.png")])
    html = "<!doctype html><html><head><meta charset=utf-8><title>new MPC samples</title>"
    html += "<style>body{font-family:sans-serif;background:#1c1c1c;color:#ddd;margin:12px}"
    html += ".card{background:#2a2a2a;padding:6px;border-radius:6px;margin-bottom:12px}"
    html += ".card img{width:100%;display:block}</style></head><body>"
    html += "<h2>new MPC samples (ax ±15 cap)</h2><p>각 frame: image / ego vx 큰 글자 + BEV input traj + ax/steer/vx 곡선</p>"
    for p in pngs:
        html += f'<div class="card"><div style="font-size:11px;color:#888">{p}</div>'
        html += f'<img src="{p}" loading="lazy"></div>'
    html += "</body></html>"
    (args.out / "index.html").write_text(html)
    print(f"\n[viz] index → {args.out}/index.html")


if __name__ == "__main__":
    main()
