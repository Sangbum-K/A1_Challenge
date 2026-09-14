"""prebake 검토 — 한 이미지(frame)에 logged + random K trajectory 모두 표시.

비교 목적: 객체 가로지르는 trajectory → SCC 가 ax 음수 (감속). 옆 지나치는 trajectory → ax 양수 (가속).

각 frame composite (4 col):
  (A) ROI 카메라 + obj_target overlay
  (B) BEV: 모든 trajectory 라인 (logged=굵은 파랑, rand=얇은 색-by-speed)
  (C) steer (deg) overlay: logged vs rand
  (D) ax (m/s²) overlay: logged vs rand (음수 = 감속)

사용:
  python scripts/viz_prebake_multi_traj.py \
      --prebake .../f1_prebake_iy_spd_v3.npz \
      --raw-base .../raw_cache --meta .../speedway_agent1_meta.json \
      --n-frames 8 --n-rand 12 \
      --out outputs_multi/prebake_viz_multi
"""
from __future__ import annotations
import os, sys, json, argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from PIL import Image, ImageDraw

ROI_Y0, ROI_Y1 = 235, 485
IMG_W = 1280
BIN_W = IMG_W / 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prebake", required=True)
    ap.add_argument("--raw-base", required=True)
    ap.add_argument("--meta",    required=True)
    ap.add_argument("--n-frames", type=int, default=8, help="객체 있는 frame 수")
    ap.add_argument("--n-no-obj", type=int, default=0, help="객체 없는 (obj=0) frame 수 — 비교용")
    ap.add_argument("--n-rand",   type=int, default=12, help="frame 당 random sample 수 (최대)")
    ap.add_argument("--min-obj",  type=int, default=2, help="obj 있는 frame 의 최소 bin 수")
    ap.add_argument("--seed",     type=int, default=0)
    ap.add_argument("--out",      required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.prebake}")
    z = np.load(args.prebake, allow_pickle=True)
    sample_keys = z["sample_keys"].tolist()
    sample_tags = z["sample_tags"].tolist()
    curr_keys   = z["curr_keys"].tolist()
    traj_arr    = z["traj"]
    target_arr  = z["target"]
    mask_arr    = z["mask"]
    obj_target  = z["obj_target"] if "obj_target" in z.files else np.zeros((len(sample_keys), 40))
    ego_5d_arr  = z["ego_5d"]
    N = len(sample_keys)
    print(f"  samples={N}")

    # group sample idx by curr_key
    print("  grouping by curr_key...")
    by_frame = defaultdict(lambda: {"logged": [], "rand": []})
    for i in range(N):
        ck = curr_keys[i]
        if not ck: continue
        if sample_tags[i] == "logged":
            by_frame[ck]["logged"].append(i)
        else:
            by_frame[ck]["rand"].append(i)
    n_frames_total = len(by_frame)
    print(f"  unique frames: {n_frames_total}")

    # candidate frames
    rng = np.random.RandomState(args.seed)
    obj_cands = []   # 객체 있는 frame
    no_obj_cands = []  # 객체 없는 frame
    for ck, d in by_frame.items():
        if not d["logged"]: continue
        i_logged = d["logged"][0]
        nobj = int((obj_target[i_logged] > 0.5).sum())
        if len(d["rand"]) < 5: continue
        if nobj >= args.min_obj:
            obj_cands.append((ck, nobj, len(d["rand"])))
        elif nobj == 0:
            no_obj_cands.append((ck, nobj, len(d["rand"])))
    print(f"  candidates: obj>={args.min_obj}={len(obj_cands)}  no-obj={len(no_obj_cands)}")
    rng.shuffle(obj_cands); rng.shuffle(no_obj_cands)
    sel = obj_cands[:args.n_frames] + no_obj_cands[:args.n_no_obj]
    sel.sort(key=lambda x: -x[1])  # obj count 큰 순 (no-obj 는 뒤)

    # meta lookup
    meta_list = json.load(open(args.meta))
    fid2meta = {}
    for m in meta_list:
        run = m["run_id"]; tick = int(m["tick"])
        fid2meta[f"{run}__t{tick:07d}"] = m

    rows = []
    for fi, (ck, nobj, nrand) in enumerate(sel):
        if ck not in fid2meta: continue
        m = fid2meta[ck]
        img_p = Path(args.raw_base) / m["track"] / m["policy"] / m["run_id"] / "CAMERA_1" / f"{int(m['tick'])}_Intensity.jpeg"
        if not img_p.exists(): continue
        img = np.array(Image.open(img_p).convert("RGB"))
        roi = img[ROI_Y0:ROI_Y1]

        i_logged = by_frame[ck]["logged"][0]
        rand_idx = by_frame[ck]["rand"]
        rng2 = np.random.RandomState(args.seed + fi)
        if len(rand_idx) > args.n_rand:
            rand_idx = list(rng2.choice(rand_idx, size=args.n_rand, replace=False))

        traj_log = traj_arr[i_logged]       # (40, 3) x_fwd, y_left, speed
        targ_log = target_arr[i_logged]
        obj      = obj_target[i_logged]
        ego      = ego_5d_arr[i_logged]

        fig = plt.figure(figsize=(20, 9), facecolor="white")
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.3], width_ratios=[1.6, 1, 1])

        # (A) image (top row, col 0) with obj overlay
        ax_img = fig.add_subplot(gs[0, :])
        H = ROI_Y1 - ROI_Y0
        pil = Image.fromarray(roi.copy())
        draw = ImageDraw.Draw(pil)
        for b in range(40):
            if obj[b] > 0.5:
                draw.rectangle([b*BIN_W, 0, (b+1)*BIN_W-1, H-1], outline=(255, 60, 60), width=3)
        ax_img.imshow(np.array(pil)); ax_img.axis("off")
        ax_img.set_title(f"{ck}  obj_bins={nobj}  ego_vx={ego[0]:.1f}m/s  (logged + {len(rand_idx)} rand)", fontsize=10)

        # speed normalization (모든 trajectory 통합)
        all_speeds = [traj_log[:, 2]] + [traj_arr[ri, :, 2] for ri in rand_idx]
        s_min = min(s.min() for s in all_speeds); s_min = max(0, s_min - 1)
        s_max = max(s.max() for s in all_speeds); s_max = max(s_max, 5)
        norm = Normalize(vmin=s_min, vmax=s_max)

        # (B) BEV — all trajectories
        ax_bev = fig.add_subplot(gs[1, 0])
        # random first (얇음)
        for ri in rand_idx:
            tr = traj_arr[ri]
            pts = np.array([tr[:, 0], tr[:, 1]]).T.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, cmap="viridis", norm=norm, linewidths=1.2, alpha=0.7)
            lc.set_array(tr[:-1, 2])
            ax_bev.add_collection(lc)
        # logged (굵음, 검정 outline + viridis)
        pts = np.array([traj_log[:, 0], traj_log[:, 1]]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="viridis", norm=norm, linewidths=4.0)
        lc.set_array(traj_log[:-1, 2])
        ax_bev.add_collection(lc)
        ax_bev.scatter([0], [0], color="red", s=100, marker="^", zorder=10)
        cbar = plt.colorbar(lc, ax=ax_bev, fraction=0.04)
        cbar.set_label("speed (m/s)")
        # 객체 위치 표시: obj_target bin → BEV 거리는 알 수 없으나, 대략 forward 5-50m 영역에 사각형 (참고용)
        x_max = max(60, max(traj_log[:, 0].max(), max(traj_arr[ri, :, 0].max() for ri in rand_idx)) + 5)
        y_lo = min(-20, min(traj_log[:, 1].min(), min(traj_arr[ri, :, 1].min() for ri in rand_idx)) - 2)
        y_hi = max(20, max(traj_log[:, 1].max(), max(traj_arr[ri, :, 1].max() for ri in rand_idx)) + 2)
        ax_bev.set_xlim(-2, x_max); ax_bev.set_ylim(y_lo, y_hi)
        ax_bev.set_xlabel("x forward (m)"); ax_bev.set_ylabel("y left (m)")
        ax_bev.set_aspect("equal"); ax_bev.grid(True, alpha=0.3)
        ax_bev.set_title(f"BEV: logged (thick) + {len(rand_idx)} random (thin)", fontsize=10)

        # (C) steer overlay
        ax_st = fig.add_subplot(gs[1, 1])
        wp = np.arange(40)
        for ri in rand_idx:
            ax_st.plot(wp, target_arr[ri, :, 0], "-", color="gray", linewidth=0.8, alpha=0.5)
        ax_st.plot(wp, targ_log[:, 0], "b-o", markersize=3, linewidth=2.0, label="logged")
        ax_st.axhline(0, color="k", linestyle=":", linewidth=0.5)
        ax_st.set_xlabel("waypoint i"); ax_st.set_ylabel("steer (deg)")
        ax_st.set_title("steer (deg)", fontsize=10); ax_st.grid(True, alpha=0.3); ax_st.legend(fontsize=8)

        # (D) ax overlay — 핵심! 객체 가로지르면 ax 음수 (감속)
        ax_ax = fig.add_subplot(gs[1, 2])
        for ri in rand_idx:
            ax_ax.plot(wp, target_arr[ri, :, 1], "-", color="lightcoral", linewidth=0.8, alpha=0.5)
        ax_ax.plot(wp, targ_log[:, 1], "r-s", markersize=3, linewidth=2.0, label="logged")
        ax_ax.axhline(0, color="k", linestyle=":", linewidth=0.5)
        ax_ax.set_xlabel("waypoint i"); ax_ax.set_ylabel("ax (m/s²)")
        ax_ax.set_title("ax (m/s²) — 객체 통과 → 음수", fontsize=10); ax_ax.grid(True, alpha=0.3); ax_ax.legend(fontsize=8)

        out_p = args.out / f"{fi:02d}_obj{nobj}_{ck}.png"
        fig.savefig(out_p, dpi=80, bbox_inches="tight")
        plt.close(fig)
        rows.append((out_p.name, nobj, ck))
        print(f"  [{fi+1}/{len(sel)}] {out_p.name}")

    rows_html = "\n".join(
        f'<div class="card"><div class="cap">[{nobj}obj] {ck}</div>'
        f'<img src="{name}" loading="lazy"></div>'
        for (name, nobj, ck) in rows)
    html = f"""<!doctype html><html><head><meta charset=utf-8>
<title>Prebake multi-traj inspection</title>
<style>
body {{ font-family: sans-serif; background:#1c1c1c; color:#ddd; margin:12px; }}
.card {{ background:#2a2a2a; padding:6px; border-radius:6px; margin-bottom:12px; }}
.cap {{ font-size:12px; color:#aaa; padding:4px; word-break:break-all; }}
.card img {{ width:100%; display:block; }}
</style></head><body>
<h2>Prebake multi-trajectory inspection — 한 image 에 logged + random K</h2>
<p>BEV: logged (굵은) + N random (얇은), color=speed.  steer/ax: logged 굵게, random 얇게 overlay.</p>
<p>객체 가로지르는 random trajectory → ax 음수 (감속). 옆 지나치는 → ax 양수.</p>
{rows_html}
</body></html>"""
    (args.out / "index.html").write_text(html)
    print(f"\n[viz] index → {args.out}/index.html")


if __name__ == "__main__":
    main()
