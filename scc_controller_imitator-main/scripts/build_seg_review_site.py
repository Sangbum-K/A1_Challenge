"""SAM + SCC review HTML 사이트 빌드.

각 frame composite (1 page):
  (1) full image + ROI 박스 + BEV obj 투영 (yellow=others, red=lead)
  (2) SAM mask overlay (vehicle=red, road=green, wall=blue) on ROI
  (3) BEV: ego + objects + lane(±2m) + closest in-lane lead
  (4) summary text (lead distance, n_objs, SCC active/not, etc.)

index.html:
  - thumbnails 그리드
  - filter: has-lead-in-lane, has-vehicle-mask, has-close-lead (<15m)
  - click → 큰 composite

사용:
  ~/miniforge3/envs/ml_con/bin/python scripts/build_seg_review_site.py [--once|--watch]

  --once: 한 번만 모든 available frame 처리하고 종료
  --watch: 지속적으로 새 mask 감시하며 추가 (60s 간격)

출력:
  outputs_multi/seg_review_v2/
    composites/{frame_id}.png
    index.html
"""
from __future__ import annotations
import os, sys, json, time, argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path("/home/ailab-12/git/scc_controller_imitator")
sys.path.insert(0, str(ROOT / "scripts"))
E2E_ROOT = Path("/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/jiwon_e2e/e2e_racing")
sys.path.insert(0, str(E2E_ROOT))
from src.data.ipm import _invert_extrinsic, _world_to_image
K_CAM = np.load(str(E2E_ROOT / "calibration/CAMERA_1__intrinsic.npy"))
E_CAM = np.load(str(E2E_ROOT / "calibration/CAMERA_1__extrinsic.npy"))
R_V2C, T_V2C = _invert_extrinsic(E_CAM)

META_JSON = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/multiagnet_compact_meta.json"
RAW_BASE = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/jiwon_e2e/e2e_racing/data/raw/sangam_track/MultiAgnet"
SEG_DIR = Path("/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/seg_masks_v2")

OUT_DIR = ROOT / "outputs_multi/seg_review_v2"
COMP_DIR = OUT_DIR / "composites"
SCC_DIR = Path("/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/scc_synth_v2")
A2_DIR  = Path("/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/aux_targets_fx320")
ROI_Y_TOP, ROI_Y_BOT = 235, 485
LANE_HALF = 2.0

# a_2 GT 채널: [obj, v_long_abs, v_lat_abs, v_long_rel, v_lat_rel, prox, dist]
A2_CH_NAMES = ["obj", "vL_abs", "vY_abs", "vL_rel", "vY_rel", "prox", "dist"]


def a2_path_for(meta_entry: dict) -> Path:
    return A2_DIR / f"{meta_entry['frame_id']}.npz"


def plot_a2_bins_on_image(ax, img_roi, target, fov_deg=None):
    """image (250, 1280, 3) 위에 a_2 GT 의 obj bin 을 vertical bar 로 오버레이.
    bar 색: 상대 v_long (red=catching up, green=pulling away),
    label: dist (m) + Δv_long_rel.
    """
    import matplotlib.cm as cm
    from matplotlib.patches import Rectangle
    H, W = img_roi.shape[:2]
    n_bins = target.shape[0]
    bin_w_px = W / n_bins
    ax.imshow(img_roi)
    for bi in range(n_bins):
        if target[bi, 0] < 0.5: continue
        v_rel = target[bi, 3]    # v_long_rel
        d     = target[bi, 6]
        x0 = bi * bin_w_px
        # 색: rel<0 = catching → red, >0 = pulling → green
        if v_rel < 0:
            c = (1.0, 0.2, 0.2)
        else:
            c = (0.2, 0.8, 0.2)
        ax.add_patch(Rectangle((x0, 0), bin_w_px, H, fc=c, alpha=0.18, ec='none'))
        ax.add_patch(Rectangle((x0, H - 30), bin_w_px, 30, fc=c, alpha=0.55, ec='k', lw=0.5))
        ax.text(x0 + bin_w_px/2, H - 16, f"{d:.0f}m",
                ha='center', va='center', fontsize=7, color='white', fontweight='bold')
        ax.text(x0 + bin_w_px/2, H - 4, f"{v_rel:+.0f}",
                ha='center', va='center', fontsize=6, color='white')
    ttl = f"a_2 GT: obj per-bin (color=Δv_long: red=ego catching up)"
    if fov_deg: ttl += f"   FOV {fov_deg:.0f}°"
    ax.set_title(ttl, fontsize=10)
    ax.axis("off")


def plot_a2_channels_heatmap(ax, target):
    """target (40, 7) → (7, 40) heatmap. obj 가 0 인 bin 은 mask 처리."""
    import matplotlib.colors as mcolors
    n_bins, n_ch = target.shape
    valid_mask = target[:, 0] > 0.5
    vis = target.T.copy()    # (7, 40)
    # obj 가 0 인 column 은 NaN 으로 (visual mask)
    for c in range(1, n_ch):
        vis[c, ~valid_mask] = np.nan
    # 각 ch 별 vmin/vmax 자체 normalize 위해 그냥 imshow 하지 않고
    # row 별로 표시 — quick: stretch each row to [0, 1]
    norm = np.zeros_like(vis)
    for c in range(n_ch):
        v = vis[c]
        finite = np.isfinite(v)
        if not finite.any():
            continue
        vmin, vmax = np.nanmin(v), np.nanmax(v)
        if vmax - vmin < 1e-6:
            norm[c] = 0
        else:
            norm[c] = (v - vmin) / (vmax - vmin)
    ax.imshow(norm, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(n_ch))
    ax.set_yticklabels(A2_CH_NAMES, fontsize=8)
    ax.set_xlabel("lateral bin (0=left, 39=right)")
    ax.set_title("a_2 channels (row-wise normalized; gray=no obj)", fontsize=9)
    # NaN 영역에 회색 overlay
    for bi in range(n_bins):
        if not valid_mask[bi]:
            ax.axvspan(bi - 0.5, bi + 0.5, color="lightgray", alpha=0.6)

# .scc.txt 파일 이름 패턴: {run_id}__t{tick:07d}.scc.txt
def scc_path_for(meta_entry: dict) -> Path:
    return SCC_DIR / f"{meta_entry['run_id']}__t{int(meta_entry['tick']):07d}.scc.txt"

# parser (light, no dep on viz_synth_batch_result module)
def parse_scc(path: Path) -> dict:
    out = {"header": {}, "in_lane_obj_ids": [], "traj": [],
            "cmd": {}, "long_traj": [], "lat_traj": []}
    cur = None
    for raw in path.read_text().splitlines():
        line = raw.rstrip()
        if not line: continue
        if line.startswith("# run="):
            for kv in line.lstrip("# ").split():
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    out["header"][k] = v
            continue
        if line.startswith("#"): continue
        if line.startswith("in_lane_obj_ids:"):
            out["in_lane_obj_ids"] = [int(x) for x in line.split(":",1)[1].split()]
            continue
        if "=" in line and not line[0].isdigit():
            k, v = line.split("=", 1)
            k = k.strip(); v = v.strip()
            if k == "N": out["header"]["N"] = int(v); cur = "traj"
            elif k == "long_traj_N": out["header"]["long_traj_N"] = int(v); cur = "long_traj"
            elif k == "lat_traj_N": out["header"]["lat_traj_N"] = int(v); cur = "lat_traj"
            else:
                try: out["cmd"][k] = float(v)
                except: out["cmd"][k] = v
            continue
        try:
            nums = [float(x) for x in line.split()]
        except: continue
        if cur in out:
            out[cur].append(nums)
    return out

BEV_X_MAX = 80.0; BEV_Y_MIN = -25.0; BEV_CELL = 0.5
def bbox_to_ego(b):
    cx, cy, w, h = b
    return BEV_X_MAX - cy*BEV_CELL, BEV_Y_MIN + cx*BEV_CELL, h*BEV_CELL, w*BEV_CELL

CLASS_COLORS = np.array([
    [  0,   0,   0], [ 80, 200,  80], [255,  80,  80],
    [ 80, 100, 255], [200, 200,  60]], dtype=np.uint8)
CLASS_NAMES = ["bg", "road", "vehicle", "wall", "ignore"]


def project_box(ex, ey, L, W, h=1.5):
    c = np.array([
        [ex+L/2, ey+W/2, -0.13], [ex+L/2, ey-W/2, -0.13],
        [ex-L/2, ey-W/2, -0.13], [ex-L/2, ey+W/2, -0.13],
        [ex+L/2, ey+W/2, -0.13+h], [ex+L/2, ey-W/2, -0.13+h],
        [ex-L/2, ey-W/2, -0.13+h], [ex-L/2, ey+W/2, -0.13+h]])
    return _world_to_image(c, K_CAM, R_V2C, T_V2C)


def colorize_mask(mask):
    K, H, W = mask.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    for c in [1, 4, 3, 2]:
        if c >= K: continue
        m = mask[c].astype(bool)
        out[m] = CLASS_COLORS[c]
    return out


def overlay_image(rgb, mask_rgb, alpha=0.5):
    blend = alpha * mask_rgb.astype(np.float32) + (1-alpha) * rgb.astype(np.float32)
    return np.clip(blend, 0, 255).astype(np.uint8)


def closest_lead_in_lane(objs_with_ego):
    """returns (ex, ey, L, W) of closest in-lane (|ey|<2m) lead, or None."""
    best = None
    for ex, ey, L, W in objs_with_ego:
        if not (1.5 < ex < 60 and abs(ey) <= LANE_HALF):
            continue
        if best is None or ex < best[0]:
            best = (ex, ey, L, W)
    return best


def build_composite(meta_entry, out_path) -> dict:
    """returns dict of summary (used for index filter/sort)."""
    fid = meta_entry["frame_id"]
    run = meta_entry["run_id"]; tick = int(meta_entry["tick"])
    img_path = Path(RAW_BASE) / run / "CAMERA_1" / f"{tick}_Intensity.jpeg"
    if not img_path.exists():
        return None
    mask_path = SEG_DIR / f"{fid}.npz"
    mask = np.load(mask_path)["mask"] if mask_path.exists() else None

    img_full = np.array(Image.open(img_path).convert("RGB"))
    img_roi = img_full[ROI_Y_TOP:ROI_Y_BOT]

    all_objs = []
    for o in meta_entry["objects"]:
        ex, ey, L, W = bbox_to_ego(o["bbox"])
        all_objs.append((ex, ey, L, W))
    lead = closest_lead_in_lane(all_objs)
    n_close = sum(1 for ex, ey, _, _ in all_objs if 1.5 < ex < 30 and abs(ey) < 10)

    scc_p = scc_path_for(meta_entry)
    scc_data = parse_scc(scc_p) if scc_p.exists() else None
    a2_p = a2_path_for(meta_entry)
    a2_target = None
    if a2_p.exists():
        try: a2_target = np.load(a2_p)["target"]   # (40, 7)
        except Exception: a2_target = None

    # 행 구성 동적:
    # 항상: full image (0), SAM ROI (1), BEV+summary (2)
    # +SCC 있으면: SCC speed + gap (행 추가)
    # +a_2 있으면: a_2 image-overlay + a_2 heatmap (행 추가)
    # 마지막: nav row
    has_scc = scc_data is not None and len(scc_data["traj"]) > 0
    has_a2 = a2_target is not None and a2_target[:, 0].sum() > 0
    rows = []   # list of (kind, height)
    rows.append(("img_full",  1.0)); IDX_IMG_FULL = 0
    rows.append(("sam",       0.9)); IDX_SAM = 1
    rows.append(("bev_sum",   1.2)); IDX_BEV = 2
    if has_a2:
        rows.append(("a2_img",   1.0)); IDX_A2_IMG = len(rows)-1
        rows.append(("a2_heat",  0.6)); IDX_A2_HEAT = len(rows)-1
    if has_scc:
        rows.append(("scc",      0.9)); IDX_SCC = len(rows)-1
    rows.append(("nav",       0.45)); IDX_NAV = len(rows)-1
    n_rows = len(rows)
    heights = [h for _, h in rows]
    fig = plt.figure(figsize=(16, sum(heights)*2.5))
    gs = fig.add_gridspec(n_rows, 2, height_ratios=heights,
                            width_ratios=[2, 1.1], hspace=0.30, wspace=0.15)

    # (0, :) full image + GT BEV projection
    ax = fig.add_subplot(gs[IDX_IMG_FULL, :])
    ax.imshow(img_full)
    ax.add_patch(Rectangle((0, ROI_Y_TOP), 1280, 250, ec="lime", fc="none", lw=2))
    for (ex, ey, L, W) in all_objs:
        if not (1.5 < ex < 80 and abs(ey) < 20): continue
        uv = project_box(ex, ey, L, W)
        if (uv[:, 2] <= 0).any(): continue
        is_lead = lead is not None and abs(ex-lead[0])<0.01 and abs(ey-lead[1])<0.01
        color = "red" if is_lead else "yellow"
        poly = uv[:4, :2]
        ax.plot(np.append(poly[:,0], poly[0,0]),
                  np.append(poly[:,1], poly[0,1]), color=color, lw=2 if is_lead else 1)
    ax.set_title(f"{fid}   close objs={n_close}   lead={('('+str(round(lead[0],1))+','+str(round(lead[1],1))+'m)') if lead else 'none'}",
                  fontsize=11)
    ax.axis("off")

    # (1, :) SAM mask overlay
    ax = fig.add_subplot(gs[IDX_SAM, :])
    if mask is not None:
        m_rgb = colorize_mask(mask)
        ax.imshow(overlay_image(img_roi, m_rgb, alpha=0.5))
        px_sum = " ".join(f"{CLASS_NAMES[c]}={int(mask[c].sum())}" for c in range(mask.shape[0]))
        ax.set_title(f"SAM v2 overlay  {px_sum}", fontsize=10)
    else:
        ax.imshow(img_roi)
        ax.set_title(f"⚠ no SAM mask yet", fontsize=10, color="red")
    ax.axis("off")

    # (2, 0) BEV
    ax = fig.add_subplot(gs[IDX_BEV, 0])
    ax.add_patch(Rectangle((-2.5, -1), 5, 2, ec="k", fc="gray", alpha=0.7))
    ax.axhline(+LANE_HALF, color="gray", ls=":", lw=1, alpha=0.5)
    ax.axhline(-LANE_HALF, color="gray", ls=":", lw=1, alpha=0.5)
    for (ex, ey, L, W) in all_objs:
        if not (-2 < ex < 80 and abs(ey) < 25): continue
        is_lead = lead is not None and abs(ex-lead[0])<0.01 and abs(ey-lead[1])<0.01
        ax.add_patch(Rectangle((ex-L/2, ey-W/2), L, W,
                                  ec="red" if is_lead else "gray",
                                  fc="salmon" if is_lead else "lightgray",
                                  alpha=0.8 if is_lead else 0.5))
        if 1.5 < ex < 30:
            ax.text(ex, ey+W/2+0.3, f"({ex:.0f},{ey:+.0f})",
                      ha="center", fontsize=8,
                      color="red" if is_lead else "dimgray")
    ax.set_xlim(-5, 50); ax.set_ylim(-12, 12); ax.set_aspect("equal")
    ax.grid(True, alpha=0.3); ax.set_xlabel("x [m] fwd"); ax.set_ylabel("y [m] left")
    ax.set_title("BEV  (gray dotted = lane ±2m)")

    # (2, 1) summary
    ax = fig.add_subplot(gs[IDX_BEV, 1])
    ax.axis("off")
    lines = [
        f"frame: {fid}",
        f"  i={meta_entry.get('i', '?')}  tick={tick}",
        f"",
        f"BEV objects: {len(all_objs)}",
        f"  close (<30m, |y|<10): {n_close}",
        f"",
    ]
    if lead:
        lines += [
            f"closest IN-LANE lead:",
            f"  pos=({lead[0]:.1f}, {lead[1]:+.1f}) m",
            f"  L={lead[2]:.1f} W={lead[3]:.1f}",
            f"  SCC would trigger",
        ]
    else:
        lines += [
            f"no in-lane lead (|y|>2m)",
            f"  SCC not triggered",
        ]
    if mask is not None:
        lines += ["", f"SAM v2 vehicle px: {int(mask[2].sum())}"]
    else:
        lines += ["", "no SAM mask yet"]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", fontsize=10,
              family="monospace", transform=ax.transAxes)

    # a_2 panel (있으면)
    if has_a2:
        ax = fig.add_subplot(gs[IDX_A2_IMG, :])
        plot_a2_bins_on_image(ax, img_roi, a2_target)
        ax = fig.add_subplot(gs[IDX_A2_HEAT, :])
        plot_a2_channels_heatmap(ax, a2_target)

    # SCC panel (있으면)
    if has_scc:
        tr = np.array(scc_data["traj"])  # cols: i cum_s t_i v_orig v_scc lead_s gap
        ax = fig.add_subplot(gs[IDX_SCC, 0])
        ax.plot(tr[:, 1], tr[:, 3], "k--", lw=1.2, alpha=0.6, label="v_orig")
        ax.plot(tr[:, 1], tr[:, 4], "C3-", lw=2, label="v_scc (SCC replan)")
        ax.set_xlabel("s (m)"); ax.set_ylabel("speed (m/s)")
        ax.legend(loc="best", fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_title("SCC speed profile (v_orig vs v_scc)", fontsize=10)
        ax = fig.add_subplot(gs[IDX_SCC, 1])
        valid = ~np.isnan(tr[:, 6])
        if valid.any():
            ax.plot(tr[valid, 1], tr[valid, 6], "C2-o", lw=1.5, ms=3)
            ax.axhline(3.0, color="gray", ls=":", lw=0.8, label="s_min=3m")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "no in-lane lead at any wp", ha="center",
                     va="center", transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_xlabel("s (m)"); ax.set_ylabel("gap to lead (m)")
        ax.grid(True, alpha=0.3); ax.set_title("longitudinal gap", fontsize=10)
    # navigation hint
    ax = fig.add_subplot(gs[IDX_NAV, :])
    ax.axis("off")
    extras = []
    if has_scc:
        cmd = scc_data["cmd"]
        extras.append(f"instant: lat={cmd.get('lat_steer_tire_deg', 0):+.2f}°  "
                       f"ax={cmd.get('long_ax', 0):+.2f} m/s²  "
                       f"accel={cmd.get('long_accel_pedal', 0):.2f}  "
                       f"brake={cmd.get('long_brake_pedal', 0):.2f}")
    else:
        extras.append("(SCC data 없음 — synth_batch 미실행 frame)")
    ax.text(0.5, 0.5, "  |  ".join(extras), ha="center", fontsize=10,
              color="gray", transform=ax.transAxes)

    plt.savefig(out_path, dpi=85, bbox_inches="tight")
    plt.close(fig)

    # SCC 요약 (있으면)
    scc_summary = {}
    if has_scc:
        cmd = scc_data["cmd"]
        scc_summary = {
            "has_scc": True,
            "scc_active": bool(scc_data["in_lane_obj_ids"]),
            "long_ax": float(cmd.get("long_ax", 0)),
            "brake_pedal": float(cmd.get("long_brake_pedal", 0)),
            "lat_steer_deg": float(cmd.get("lat_steer_tire_deg", 0)),
        }
    else:
        scc_summary = {"has_scc": False}

    return {
        "frame_id": fid, "i": meta_entry.get("i", -1),
        "n_close": n_close, "has_lead_in_lane": bool(lead),
        "lead_x": float(lead[0]) if lead else None,
        "has_mask": mask is not None,
        "vehicle_px": int(mask[2].sum()) if mask is not None else 0,
        **scc_summary,
    }


def write_html_index(summaries, out_html):
    """Build index.html with filterable thumbnail grid."""
    rows_json = json.dumps(summaries)
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>SAM + SCC review v2</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 20px; background: #f8f8f8; }}
h1 {{ margin: 0 0 6px; }}
.filters {{ background: #fff; padding: 12px; margin-bottom: 18px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); display: flex; gap: 14px; align-items: center; flex-wrap: wrap; }}
.filters label {{ font-size: 13px; cursor: pointer; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 14px; }}
.card {{ background: #fff; border-radius: 6px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); transition: transform 0.1s; }}
.card:hover {{ transform: translateY(-2px); box-shadow: 0 3px 12px rgba(0,0,0,0.12); }}
.card img {{ width: 100%; display: block; }}
.card .info {{ padding: 6px 9px; font-size: 11px; color: #555; }}
.card .info b {{ color: #222; }}
.badge {{ display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 10px; margin-right: 4px; }}
.badge-lead {{ background: #fdd; color: #900; }}
.badge-nomask {{ background: #eee; color: #888; }}
.badge-vehicle {{ background: #ffeecc; color: #b80; }}
.badge-scc {{ background: #cef; color: #036; }}
.badge-brake {{ background: #fda; color: #b03; }}
.summary {{ color: #666; font-size: 13px; margin-bottom: 12px; }}
</style></head><body>
<h1>SAM v2 + SCC review</h1>
<div class="summary" id="summary"></div>
<div class="filters">
  <label><input type="checkbox" id="f_lead"> in-lane lead 만</label>
  <label><input type="checkbox" id="f_vehicle"> vehicle mask 있는 것만</label>
  <label><input type="checkbox" id="f_close"> close lead (&lt;15m)</label>
  <label><input type="checkbox" id="f_scc"> SCC 데이터 있는 것만</label>
  <label><input type="checkbox" id="f_brake"> SCC brake 명령 (pedal&gt;0)</label>
  <label>sort: <select id="sort">
    <option value="i">frame i</option>
    <option value="lead_x">lead 거리 (가까운 순)</option>
    <option value="vehicle_px">vehicle px (큰 순)</option>
    <option value="brake">brake pedal 큰 순</option>
  </select></label>
  <label>page size: <select id="ps"><option>60</option><option selected>120</option><option>240</option><option>600</option></select></label>
  <span id="page_nav"></span>
</div>
<div class="grid" id="grid"></div>
<script>
const data = {rows_json};
const grid = document.getElementById("grid");
const $ = id => document.getElementById(id);
let page = 0;

function filteredSorted() {{
  let rows = data.slice();
  if ($("f_lead").checked) rows = rows.filter(r => r.has_lead_in_lane);
  if ($("f_vehicle").checked) rows = rows.filter(r => r.vehicle_px > 0);
  if ($("f_close").checked) rows = rows.filter(r => r.lead_x !== null && r.lead_x < 15);
  if ($("f_scc").checked) rows = rows.filter(r => r.has_scc);
  if ($("f_brake").checked) rows = rows.filter(r => (r.brake_pedal ?? 0) > 0);
  const k = $("sort").value;
  rows.sort((a, b) => {{
    if (k === "lead_x") return (a.lead_x ?? 999) - (b.lead_x ?? 999);
    if (k === "vehicle_px") return (b.vehicle_px ?? 0) - (a.vehicle_px ?? 0);
    if (k === "brake") return (b.brake_pedal ?? 0) - (a.brake_pedal ?? 0);
    return (a.i ?? 0) - (b.i ?? 0);
  }});
  return rows;
}}

function render() {{
  const rows = filteredSorted();
  const ps = parseInt($("ps").value);
  const start = page * ps;
  const subset = rows.slice(start, start + ps);
  grid.innerHTML = "";
  subset.forEach(r => {{
    const card = document.createElement("div");
    card.className = "card";
    const badges = [];
    if (r.has_lead_in_lane) badges.push(`<span class="badge badge-lead">lead ${{r.lead_x?.toFixed(1)}}m</span>`);
    if (r.vehicle_px > 1000) badges.push(`<span class="badge badge-vehicle">veh ${{(r.vehicle_px/1000).toFixed(0)}}k</span>`);
    if (!r.has_mask) badges.push(`<span class="badge badge-nomask">no mask</span>`);
    if (r.has_scc) badges.push(`<span class="badge badge-scc">SCC</span>`);
    if ((r.brake_pedal ?? 0) > 0.1) badges.push(`<span class="badge badge-brake">brake ${{r.brake_pedal.toFixed(2)}}</span>`);
    card.innerHTML = `<a href="composites/${{r.frame_id}}.png" target="_blank"><img src="composites/${{r.frame_id}}.png" loading="lazy"></a>
      <div class="info"><b>i=${{r.i}}</b> ${{r.frame_id.substring(0,40)}}...<br>${{badges.join(" ")}}</div>`;
    grid.appendChild(card);
  }});
  $("summary").textContent = `${{rows.length}} / ${{data.length}} frames    page ${{page+1}}/${{Math.ceil(rows.length/ps)}}    in-lane lead: ${{rows.filter(r=>r.has_lead_in_lane).length}}    mask 있음: ${{rows.filter(r=>r.has_mask).length}}`;
  const npage = Math.ceil(rows.length / ps);
  $("page_nav").innerHTML = `<button onclick="goto(${{page-1}})" ${{page<=0?'disabled':''}}>◀</button> <button onclick="goto(${{page+1}})" ${{page>=npage-1?'disabled':''}}>▶</button>`;
}}
function goto(p) {{ page = Math.max(0, p); render(); }}
["f_lead", "f_vehicle", "f_close", "f_scc", "f_brake", "sort", "ps"].forEach(id => {{
  $(id).addEventListener("change", () => {{ page = 0; render(); }});
}});
render();
</script>
</body></html>"""
    out_html.write_text(html)


def run_once(force_rebuild=False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    COMP_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(META_JSON))
    print(f"meta entries: {len(meta)}")
    summaries = []
    n_done = n_skip = n_no_mask = 0
    t0 = time.time()
    for k, m in enumerate(meta):
        fid = m["frame_id"]
        comp_path = COMP_DIR / f"{fid}.png"
        if comp_path.exists() and not force_rebuild:
            n_skip += 1
            # 기존 composite 의 summary 는 mask 변경 가능성 있으니 항상 새로 계산하지 말고 reuse?
            # 단순화: skip 도 summary 추가 (현 mask 상태 기준)
            mask_path = SEG_DIR / f"{fid}.npz"
            has_mask = mask_path.exists()
            vehicle_px = int(np.load(mask_path)['mask'][2].sum()) if has_mask else 0
            objs = [bbox_to_ego(o['bbox']) for o in m['objects']]
            lead = closest_lead_in_lane(objs)
            summaries.append({
                "frame_id": fid, "i": m.get("i", -1),
                "n_close": sum(1 for ex,ey,_,_ in objs if 1.5<ex<30 and abs(ey)<10),
                "has_lead_in_lane": bool(lead),
                "lead_x": float(lead[0]) if lead else None,
                "has_mask": has_mask, "vehicle_px": vehicle_px,
            })
            continue
        try:
            s = build_composite(m, comp_path)
            if s:
                summaries.append(s)
                n_done += 1
                if not s["has_mask"]: n_no_mask += 1
        except Exception as e:
            print(f"  fail {fid}: {e}")
        if (k + 1) % 50 == 0:
            dt = time.time() - t0
            rate = n_done / max(dt, 0.01)
            print(f"  [{k+1}/{len(meta)}] new={n_done} skip={n_skip} no_mask={n_no_mask} rate={rate:.1f}/s")
    write_html_index(summaries, OUT_DIR / "index.html")
    print(f"\ndone: new={n_done} skipped={n_skip} no_mask={n_no_mask}")
    print(f"index → {OUT_DIR / 'index.html'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true",
                    help="60s 간격 polling, 새 mask 감지 시 composite 추가")
    ap.add_argument("--once", action="store_true", help="한 번 처리하고 종료")
    ap.add_argument("--force", action="store_true", help="기존 composite 무시하고 재생성")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.watch:
        print("watch mode (Ctrl-C 종료)")
        while True:
            run_once(force_rebuild=args.force)
            print("sleep 60s ...")
            time.sleep(60)
    else:
        run_once(force_rebuild=args.force)


if __name__ == "__main__":
    main()
