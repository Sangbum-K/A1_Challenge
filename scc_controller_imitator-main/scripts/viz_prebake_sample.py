"""prebake_iy 검토용 정성 시각화.

각 row (sample 1개) 4 panel:
  (A) curr camera image + obj_target bin overlay (red bins where obj=1)
  (B) BEV trajectory (ego frame, x-forward / y-left), **color = speed**
       + ego marker, speed colorbar
  (C) SCC GT (target) — steer (deg) over time + ax (m/s²) over time
  (D) ego state + meta info text

obj_target > 0 frame 우선 선택 (close-lead 검토).

사용:
  python scripts/viz_prebake_sample.py \
      --prebake .../f1_prebake_iy_spd_v3.npz \
      --i1-cache .../i1_tok_cache_spd_v3.npz \
      --raw-base .../raw_cache \
      --n 12 --out outputs_multi/prebake_viz_spd
"""
from __future__ import annotations
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ROI_Y0, ROI_Y1 = 235, 485
IMG_H, IMG_W = 720, 1280
BIN_W = IMG_W / 40


def load_prebake(p):
    z = np.load(p, allow_pickle=True)
    return {
        "sample_keys": z["sample_keys"].tolist(),
        "sample_tags": z["sample_tags"].tolist(),
        "curr_keys":   z["curr_keys"].tolist(),
        "d_yaws":      z["d_yaws"],
        "sample_fx":   z["sample_fx"] if "sample_fx" in z.files else None,
        "ego_5d":      z["ego_5d"],
        "traj":        z["traj"],
        "target":      z["target"],
        "mask":        z["mask"],
        "obj_target":  z["obj_target"] if "obj_target" in z.files else None,
    }


def find_image_path(raw_base, sample_key, frame_id_template=None):
    """sample_key '...' → raw image path 추측. sample_key 는 보통 curr_key 와 매칭됨.
    sample_keys 는 'run__tNNNNNNN__tag_XXX' 형태. curr_keys 가 'run__tNNNNNNN'.
    track/policy 는 raw_base 안에서 enumerate 해서 찾아야 함.
    """
    # 이 함수는 외부에서 만든 (track, policy, run, tick) lookup table 받기로
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prebake", required=True)
    ap.add_argument("--raw-base", required=True, help="raw_cache root (track/policy/run/CAMERA_1/*.jpeg)")
    ap.add_argument("--meta",    required=True, help="meta json with frame_id + track/policy/run/tick 매핑")
    ap.add_argument("--n",        type=int, default=12)
    ap.add_argument("--min-obj",  type=int, default=2, help="obj bin count >= min")
    ap.add_argument("--seed",     type=int, default=0)
    ap.add_argument("--out",      required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.prebake}")
    pb = load_prebake(args.prebake)
    N = len(pb["sample_keys"])
    print(f"  samples: {N}")
    if pb["obj_target"] is None:
        print("  [warn] obj_target column 없음 — random sample")
        obj_counts = np.zeros(N, dtype=int)
    else:
        obj_counts = (pb["obj_target"] > 0.5).sum(axis=1)

    # filter: logged + obj count
    valid = []
    for i in range(N):
        if pb["sample_tags"][i] != "logged": continue
        if obj_counts[i] < args.min_obj: continue
        valid.append((i, int(obj_counts[i])))
    print(f"  valid (logged + obj>={args.min_obj}): {len(valid)}")
    rng = np.random.RandomState(args.seed)
    rng.shuffle(valid)
    sel = valid[:args.n]
    sel.sort(key=lambda x: -x[1])

    # meta lookup: frame_id → (track, policy, run, tick)
    print(f"[load] meta {args.meta}")
    meta_list = json.load(open(args.meta))
    fid2meta = {}
    for m in meta_list:
        fid2meta[m["frame_id"]] = m
        # 또한 curr_key 같은 짧은 key 도 매핑
        run = m["run_id"]; tick = int(m["tick"])
        fid2meta[f"{run}__t{tick:07d}"] = m

    rows = []
    for idx, (i, nobj) in enumerate(sel):
        ck = pb["curr_keys"][i]   # "run__tNNNNNNN"
        if ck not in fid2meta:
            print(f"  [skip] {ck} not in meta"); continue
        m = fid2meta[ck]
        track, policy, run, tick = m["track"], m["policy"], m["run_id"], int(m["tick"])
        img_p = Path(args.raw_base) / track / policy / run / "CAMERA_1" / f"{tick}_Intensity.jpeg"
        if not img_p.exists():
            print(f"  [skip] no image {img_p}"); continue
        img = np.array(Image.open(img_p).convert("RGB"))
        roi = img[ROI_Y0:ROI_Y1]

        traj = pb["traj"][i]            # (40, 3)
        target = pb["target"][i]        # (40, 2) [steer °, ax m/s²]
        ego = pb["ego_5d"][i]           # 5d
        obj = pb["obj_target"][i] if pb["obj_target"] is not None else np.zeros(40)
        fx = float(pb["sample_fx"][i]) if pb["sample_fx"] is not None else 640.0
        d_yaw = float(pb["d_yaws"][i])

        # ── figure (4 row × 1 col, but wider) ──
        fig = plt.figure(figsize=(14, 13), facecolor="white")
        gs = fig.add_gridspec(4, 1, height_ratios=[1.1, 2.5, 1.5, 0.5], hspace=0.35)

        # (A) ROI image + obj bin overlay
        ax_img = fig.add_subplot(gs[0])
        img_pil = Image.fromarray(roi.copy())
        draw = ImageDraw.Draw(img_pil)
        H = ROI_Y1 - ROI_Y0
        for b in range(40):
            if obj[b] > 0.5:
                draw.rectangle([b*BIN_W, 0, (b+1)*BIN_W-1, H-1],
                                outline=(255, 60, 60), width=3)
        ax_img.imshow(np.array(img_pil)); ax_img.axis("off")
        ax_img.set_title(f"{ck}  fx={fx:.0f}  Δyaw={d_yaw*57.3:.2f}°  obj_bins={nobj}", fontsize=10)

        # (B) BEV trajectory colored by speed
        ax_bev = fig.add_subplot(gs[1])
        # traj[:, 0]=right (or x-forward depending on convention)
        # 현재 convention: (x_forward, y_left, speed) 인지 (right, forward, speed) 인지 확인
        # scc_synth_iy_dataset 의 traj 는 prebake 의 _world_to_ego_traj 결과 (x=forward, y=left)
        x_fwd = traj[:, 0]
        y_left = traj[:, 1]
        speed = traj[:, 2]
        pts = np.array([x_fwd, y_left]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        norm = Normalize(vmin=max(0, speed.min()-1), vmax=max(speed.max(), 5))
        lc = LineCollection(segs, cmap="viridis", norm=norm, linewidths=3)
        lc.set_array(speed[:-1])
        ax_bev.add_collection(lc)
        ax_bev.scatter([0], [0], color="red", s=80, marker="^", zorder=10, label="ego")
        ax_bev.set_xlim(-2, max(40, x_fwd.max() + 2))
        ax_bev.set_ylim(min(-15, y_left.min() - 2), max(15, y_left.max() + 2))
        ax_bev.set_xlabel("x forward (m)")
        ax_bev.set_ylabel("y left (m)")
        ax_bev.set_aspect("equal"); ax_bev.grid(True, alpha=0.3)
        cbar = plt.colorbar(lc, ax=ax_bev, fraction=0.03)
        cbar.set_label("speed (m/s)")
        ax_bev.set_title(
            f"trajectory BEV (ego frame, color=speed). ego vx={ego[0]:.1f}", fontsize=10)

        # (C) SCC GT — steer + ax over time
        ax_st = fig.add_subplot(gs[2])
        wp = np.arange(40)
        ax_st.plot(wp, target[:, 0], "b-o", markersize=3, label="steer (deg)")
        ax_st.set_xlabel("waypoint i (0.1s 간격)")
        ax_st.set_ylabel("steer (deg)", color="b")
        ax_st.tick_params(axis="y", labelcolor="b")
        ax_st.grid(True, alpha=0.3)
        ax_ax = ax_st.twinx()
        ax_ax.plot(wp, target[:, 1], "r-s", markersize=3, label="ax (m/s²)")
        ax_ax.set_ylabel("ax (m/s²)", color="r")
        ax_ax.tick_params(axis="y", labelcolor="r")
        ax_st.set_title(f"SCC GT: steer max={target[:,0].max():.1f}° / min={target[:,0].min():.1f}°  "
                         f"ax range {target[:,1].min():.2f}~{target[:,1].max():.2f} m/s²", fontsize=10)

        # (D) text info
        ax_txt = fig.add_subplot(gs[3]); ax_txt.axis("off")
        info = (
            f"track/policy/run/tick: {track} / {policy} / {run} / t{tick}\n"
            f"ego_5d: vx={ego[0]:.2f} vy={ego[1]:.2f} ax={ego[2]:.2f} ay={ego[3]:.2f} wz={ego[4]:.3f}\n"
            f"speed range over traj: {speed.min():.2f} ~ {speed.max():.2f} m/s  ({speed.mean():.2f} mean)\n"
            f"obj_bins (gate target): {[int(b) for b in np.where(obj>0.5)[0]]}"
        )
        ax_txt.text(0, 0.95, info, family="monospace", fontsize=9, va="top")

        out_p = args.out / f"{idx:02d}_obj{nobj}_{ck}.png"
        fig.savefig(out_p, dpi=80, bbox_inches="tight")
        plt.close(fig)
        rows.append((out_p.name, nobj, ck))
        print(f"  [{idx+1}/{len(sel)}] saved {out_p.name}")

    # index.html
    rows_html = "\n".join(
        f'<div class="card"><div class="cap">[{nobj}obj] {ck}</div>'
        f'<img src="{name}" loading="lazy"></div>'
        for (name, nobj, ck) in rows)
    html = f"""<!doctype html><html><head><meta charset=utf-8>
<title>Prebake sample inspection</title>
<style>
body {{ font-family: sans-serif; background:#1c1c1c; color:#ddd; margin:12px; }}
.card {{ background:#2a2a2a; padding:6px; border-radius:6px; margin-bottom:12px; }}
.cap {{ font-size:12px; color:#aaa; padding:4px; word-break:break-all; }}
.card img {{ width:100%; display:block; }}
h2 {{ font-weight:300; }}
</style></head><body>
<h2>Prebake sample inspection — {args.prebake}</h2>
<p>각 row: image + obj bin overlay / BEV (color=speed) / SCC GT (steer + ax) / meta</p>
{rows_html}
</body></html>"""
    (args.out / "index.html").write_text(html)
    print(f"\n[viz] index → {args.out}/index.html")


if __name__ == "__main__":
    main()
