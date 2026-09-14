"""speedway_agent1 데이터 종합 리뷰 사이트.

각 sample frame 마다 panel:
  (1) ROI 카메라 이미지 + 'lead at d=Xm' 표시
  (2) seg mask overlay (road=green, vehicle=red, wall=blue)
  (3) a_2 GT bin viz (40 bins): obj 있는 bin 강조 + dist/vL_rel 텍스트
  (4) SCC GT (logged): future 40 step steer (deg) + ax (m/s²)
  (5) trajectory BEV: logged + 5 random K bank traj (ego frame x-forward, y-left)

index.html: 썸네일 grid, click → 큰 composite

사용:
  python scripts/build_speedway_review_site.py [--n 200] [--out dir]
"""
from __future__ import annotations
import os, sys, json, time, argparse, random, struct
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROI_Y0, ROI_Y1 = 235, 485
IMG_H, IMG_W = 720, 1280
BIN_W = IMG_W / 40
PALETTE_SEG = np.array([[0,0,0], [80,80,80], [255,80,80], [80,200,255], [128,128,128]], dtype=np.uint8)

DEFAULT_MANIFEST = "/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/scc_synth_pair_manifest_spd.json"
DEFAULT_META     = "/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/speedway_agent1_meta.json"
DEFAULT_SEG      = "/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/seg_masks_spd"
DEFAULT_A2       = "/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/aux_targets_spd_fx640"
DEFAULT_LOGGED   = "/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/scc_logged_bank_spd"
DEFAULT_BANK     = "/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/scc_traj_bank_spd"
DEFAULT_SCC      = "/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/scc_synth_v2_spd"
DEFAULT_RAW      = "/home/ailab-12/AILabSSD/02_Data/57_A1_E2E/00_raw"
DEFAULT_TRACK    = "speedway_track"
DEFAULT_POLICY   = "0519_agent1_pertX_pitX"

DEG2RAD = np.pi / 180.0


def parse_ego(p: Path):
    if not p.exists(): return None
    out = {}
    for line in p.read_text().splitlines():
        if ":" not in line: continue
        k, _, v = line.partition(":")
        k = k.strip(); v = v.strip()
        if k in ("position", "orientation", "velocity"):
            try: nums = [float(t) for t in v.split()]
            except: continue
            if len(nums) >= 3: out[k] = nums
    return out if {"position","orientation","velocity"} <= out.keys() else None


def parse_scc_txt(path: Path):
    """SCC .txt → np.ndarray (N, 7) [i, cum_s, t_i, v_orig, v_scc, lead_s, gap]
    + extract long_traj 와 lat_traj sections."""
    if not path.exists(): return None, None, None
    lines = path.read_text().splitlines()
    spd_rows = []; long_rows = []; lat_rows = []
    section = "spd"
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or ":" in line: continue
        if line.startswith("N="): continue
        if line.startswith("long_traj"): section = "long"; continue
        if line.startswith("lat_traj"):  section = "lat";  continue
        if line.startswith("predicted_state"): section = "skip"; continue
        if section == "skip": continue
        try:
            toks = [float(t) for t in line.split()]
        except ValueError: continue
        if section == "spd"  and len(toks) >= 7: spd_rows.append(toks[:7])
        if section == "long" and len(toks) >= 5: long_rows.append(toks[:5])
        if section == "lat"  and len(toks) >= 5: lat_rows.append(toks[:5])
    spd  = np.array(spd_rows,  dtype=np.float32) if spd_rows else None
    longt= np.array(long_rows, dtype=np.float32) if long_rows else None
    latt = np.array(lat_rows,  dtype=np.float32) if lat_rows else None
    return spd, longt, latt


def load_traj_bin(path: Path):
    """logged traj bin → (N_wp, 4) [x, y, yaw, speed]."""
    if not path.exists(): return None
    with open(path, "rb") as f:
        N = struct.unpack("i", f.read(4))[0]
        arr = np.frombuffer(f.read(N * 4 * 4), dtype=np.float32).reshape(N, 4)
    return arr.copy()


def load_bank(path: Path, n_pick: int = 5, seed: int = 0):
    """bank → (n_pick, N_wp, 4) random subset."""
    if not path.exists(): return None
    with open(path, "rb") as f:
        K, N = struct.unpack("ii", f.read(8))
        data = np.frombuffer(f.read(K * N * 4 * 4), dtype=np.float32).reshape(K, N, 4)
    rng = np.random.RandomState(seed)
    pick = rng.choice(K, size=min(n_pick, K), replace=False)
    return data[pick].copy()


def world_to_ego(pts_xy, ex, ey, eyaw):
    """world → ego (x-forward, y-left). pts_xy (..., 2)."""
    dx = pts_xy[..., 0] - ex
    dy = pts_xy[..., 1] - ey
    c, s = np.cos(-eyaw), np.sin(-eyaw)
    out = np.stack([c*dx - s*dy, s*dx + c*dy], axis=-1)
    return out


def build_panel(sample, out_path):
    """sample dict 으로 composite panel 생성."""
    fid    = sample["frame_id"]
    run    = sample["run_id"]; tick = sample["tick"]
    raw    = sample["raw_base"]

    # paths
    img_p  = Path(raw) / DEFAULT_TRACK / DEFAULT_POLICY / run / "CAMERA_1" / f"{tick}_Intensity.jpeg"
    ego_p  = Path(raw) / DEFAULT_TRACK / DEFAULT_POLICY / run / "EGO_INFO" / f"{tick}.txt"
    seg_p  = Path(DEFAULT_SEG) / f"{fid}.npz"
    a2_p   = Path(DEFAULT_A2)  / f"{fid}.npz"
    log_p  = Path(DEFAULT_LOGGED) / f"{run}__t{tick:07d}.bin"
    bank_p = Path(DEFAULT_BANK)   / f"{run}__t{tick:07d}.trajs.bin"
    scc_p  = Path(DEFAULT_SCC)    / f"{run}__t{tick:07d}__logged.scc.txt"

    if not img_p.exists(): return False

    img    = np.array(Image.open(img_p).convert("RGB"))
    roi    = img[ROI_Y0:ROI_Y1].copy()
    seg    = np.load(seg_p)["mask"] if seg_p.exists() else None    # (5, 250, 1280) uint8
    a2     = np.load(a2_p)["target"] if a2_p.exists() else None    # (40, 7)
    log    = load_traj_bin(log_p)
    bank   = load_bank(bank_p, n_pick=5)
    spd, long_t, lat_t = parse_scc_txt(scc_p)
    ego    = parse_ego(ego_p)
    if ego is None: return False
    ex, ey = ego["position"][0], ego["position"][1]
    eyaw   = ego["orientation"][2] * DEG2RAD
    vx     = ego["velocity"][0]

    # ── composite figure (5 rows) ──
    fig = plt.figure(figsize=(16, 17), facecolor="white")
    gs  = fig.add_gridspec(5, 2, height_ratios=[1.2, 1.2, 0.6, 1.2, 1.4],
                            hspace=0.35, wspace=0.15)

    # (1) full image with ROI box
    ax1 = fig.add_subplot(gs[0, :])
    ax1.imshow(img); ax1.axis("off")
    ax1.add_patch(Rectangle((0, ROI_Y0), IMG_W, ROI_Y1-ROI_Y0,
                              edgecolor="lime", linewidth=2, fill=False))
    ax1.set_title(f"{fid}  ego_vx={vx:.1f} m/s  yaw={eyaw*180/np.pi:.1f}°", fontsize=10)

    # (2) seg mask overlay on ROI
    ax2 = fig.add_subplot(gs[1, :])
    if seg is not None:
        # build label map (argmax over channels but use class priority)
        label = np.argmax(seg, axis=0)  # (250, 1280) 0..4
        rgb = PALETTE_SEG[label]
        blended = (roi.astype(np.float32) * 0.5 + rgb.astype(np.float32) * 0.5).astype(np.uint8)
        ax2.imshow(blended)
        ax2.set_title("seg mask overlay (road=gray, vehicle=red, wall=cyan)", fontsize=10)
    else:
        ax2.imshow(roi); ax2.set_title("ROI (no seg mask)", fontsize=10)
    ax2.axis("off")

    # (3) a_2 bin viz (40 bins × 7 ch) — heatmap
    ax3 = fig.add_subplot(gs[2, :])
    if a2 is not None:
        # heat: obj column + bin distance scatter
        ch_names = ["obj", "vL_abs", "vY_abs", "vL_rel", "vY_rel", "prox", "dist"]
        im = ax3.imshow(a2.T, aspect="auto", cmap="viridis")
        ax3.set_yticks(range(7)); ax3.set_yticklabels(ch_names, fontsize=8)
        ax3.set_xlabel("bin (lateral, 0=left .. 39=right)", fontsize=8)
        plt.colorbar(im, ax=ax3, fraction=0.02)
        # obj bins 표시
        obj_bins = np.where(a2[:, 0] > 0.5)[0]
        if len(obj_bins) > 0:
            min_dist = float(a2[obj_bins, 6].min())
            ax3.set_title(f"a_2 GT (n_bins_obj={len(obj_bins)}, min dist={min_dist:.1f} m)", fontsize=10)
        else:
            ax3.set_title("a_2 GT (no obj in FOV)", fontsize=10)
    else:
        ax3.text(0.5, 0.5, "no a_2 GT", ha="center"); ax3.axis("off")

    # (4) SCC GT plots (lat_traj.steer + long_traj.ax)
    ax4 = fig.add_subplot(gs[3, 0])
    ax5 = fig.add_subplot(gs[3, 1])
    if lat_t is not None and len(lat_t) > 0:
        ax4.plot(lat_t[:, 0], lat_t[:, 4], 'b-')   # col 4 = steer deg
        ax4.set_title("lat_traj.steer (deg)", fontsize=10); ax4.grid(True)
        ax4.set_xlabel("waypoint i")
    if long_t is not None and len(long_t) > 0:
        ax5.plot(long_t[:, 0], long_t[:, 3], 'r-')  # col 3 = ax
        ax5.set_title("long_traj.ax (m/s²)", fontsize=10); ax5.grid(True)
        ax5.set_xlabel("waypoint i")

    # (5) trajectory BEV (ego frame)
    ax6 = fig.add_subplot(gs[4, :])
    if log is not None:
        log_e = world_to_ego(log[:, :2], ex, ey, eyaw)
        ax6.plot(log_e[:, 0], log_e[:, 1], 'b-', linewidth=2, label="logged future")
    if bank is not None:
        for k in range(bank.shape[0]):
            be = world_to_ego(bank[k, :, :2], ex, ey, eyaw)
            ax6.plot(be[:, 0], be[:, 1], '-', linewidth=0.8, alpha=0.6,
                     label=f"rand_{k}" if k < 3 else None)
    ax6.scatter([0], [0], color='red', s=40, zorder=10, label="ego")
    ax6.set_xlabel("x (forward) m"); ax6.set_ylabel("y (left) m")
    ax6.set_aspect("equal"); ax6.grid(True)
    ax6.set_title("trajectory BEV — logged (blue) + 5 random bank", fontsize=10)
    ax6.legend(fontsize=8, loc="upper right")
    ax6.set_xlim(-5, 100); ax6.set_ylim(-25, 25)

    fig.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",   type=int, default=200, help="composite 수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs_multi/speedway_review_v1", type=Path)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    comp_dir = args.out / "composites"
    comp_dir.mkdir(parents=True, exist_ok=True)

    print(f"[viewer] load manifest {args.manifest}")
    mani = json.load(open(args.manifest))
    samples = [s for s in mani["samples"] if s.get("tag") == "logged"]
    print(f"  {len(samples)} logged samples")

    # filter: seg mask 있는 frame 만 (mask 가 viewer 핵심 panel 중 하나)
    seg_dir = Path(DEFAULT_SEG)
    samples = [s for s in samples if (seg_dir / f"{s['frame_id']}.npz").exists()]
    print(f"  with seg mask: {len(samples)} (filtered)")

    # uniform sample
    rng = np.random.RandomState(args.seed)
    idx = rng.choice(len(samples), size=min(args.n, len(samples)), replace=False)
    idx.sort()
    sel = [samples[i] for i in idx]
    for s in sel:
        s["raw_base"] = mani["raw_base"]

    print(f"[viewer] generating {len(sel)} composites")
    t0 = time.time(); done = 0; fail = 0
    rows_html = []
    for i, s in enumerate(sel):
        fid = s["frame_id"]
        out_p = comp_dir / f"{fid}.png"
        if not out_p.exists():
            try:
                ok = build_panel(s, out_p)
                if not ok: fail += 1; continue
            except Exception as e:
                print(f"  fail {fid}: {e}"); fail += 1; continue
        done += 1
        rows_html.append(
            f'<div class="card"><a href="composites/{fid}.png" target="_blank">'
            f'<img src="composites/{fid}.png" loading="lazy"></a>'
            f'<div class="cap">{fid}</div></div>'
        )
        if (i + 1) % 20 == 0:
            dt = time.time() - t0
            print(f"  [{i+1}/{len(sel)}] done={done} fail={fail} rate={done/dt:.1f}/s")
    print(f"\n[done] {done} composites in {time.time()-t0:.0f}s ({fail} fails)")

    html = f"""<!doctype html><html><head><meta charset=utf-8>
<title>Speedway Agent1 Review</title>
<style>
body {{ font-family: sans-serif; background: #1c1c1c; color: #ddd; margin: 12px; }}
h1 {{ font-weight: 400; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 8px; }}
.card {{ background: #2a2a2a; padding: 4px; border-radius: 6px; }}
.card img {{ width: 100%; display: block; }}
.cap {{ font-size: 10px; color: #888; padding: 4px; word-break: break-all; }}
</style></head><body>
<h1>Speedway Agent1 Review — {done} samples</h1>
<p>raw + seg mask + a_2 GT + SCC GT + trajectory BEV. click an image to view full.</p>
<div class="grid">
{''.join(rows_html)}
</div>
</body></html>"""
    (args.out / "index.html").write_text(html)
    print(f"[viewer] index → {args.out}/index.html")


if __name__ == "__main__":
    main()
