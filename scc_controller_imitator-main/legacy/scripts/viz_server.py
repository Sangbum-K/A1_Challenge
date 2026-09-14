"""FastAPI + Plotly: SCC synth dataset 원격 sample browser.

실행:
  ~/miniforge3/envs/ml_con/bin/python scripts/viz_server.py
  # 또는 host 지정
  ~/miniforge3/envs/ml_con/bin/python scripts/viz_server.py --port 8000

원격 접속 (host=ailab-12):
  ssh -L 8000:localhost:8000 ailab-12@<HOST>
  # 브라우저에서 http://localhost:8000

UI: prefix 선택 → frame_id 선택 → sample (traj_tag) 선택 → 시각화.
Plotly interactive (zoom, pan, hover).
"""
from __future__ import annotations
import os, sys, glob, pickle, json, argparse, base64, io, threading
from pathlib import Path
import numpy as np
from PIL import Image

# pkl 안의 numpy.float32 자체는 ml_con (numpy 2.x) 에서 unpickle 가능
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import plotly.graph_objects as go

sys.path.insert(0, '/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/jiwon_e2e/e2e_racing')
from src.data.ipm import _invert_extrinsic, _world_to_image

OUT_DIR = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/scc_synth_k100_v1"
META_JSON = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/multiagnet_compact_meta.json"
CALIB_DIR = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/jiwon_e2e/e2e_racing/calibration"
K_CAM = np.load(f"{CALIB_DIR}/CAMERA_1__intrinsic.npy")
E_CAM = np.load(f"{CALIB_DIR}/CAMERA_1__extrinsic.npy")
R_V2C, T_V2C = _invert_extrinsic(E_CAM)

X_MAX, Y_MIN, CELL = 80.0, -25.0, 0.5
L_WB, TS = 2.7, 0.1

PREFIXES = ["fwd50", "lead", "lead2", "lead3", "nolead", "nolead2", "nolead3"]

app = FastAPI()
_lock = threading.Lock()
_cache_index = {}          # prefix → [(pkl_path, [frame_id, ...]), ...]
_cache_meta = None         # frame_id → meta dict


def _bbox_to_ego(b):
    cx, cy, _, _ = b
    return X_MAX - cy*CELL, cx*CELL + Y_MIN


def _load_meta():
    global _cache_meta
    if _cache_meta is None:
        m = json.load(open(META_JSON))
        _cache_meta = {x["frame_id"]: x for x in m}
    return _cache_meta


def _scan_prefix(prefix):
    if prefix in _cache_index: return _cache_index[prefix]
    out = []
    pkls = sorted(fp for fp in glob.glob(os.path.join(OUT_DIR, f"{prefix}_*.pkl"))
                   if os.path.getsize(fp) > 10000)
    for fp in pkls:
        try:
            d = pickle.load(open(fp, "rb"))
            fids = sorted(set(s["frame_id"] for s in d["samples"]))
            out.append((fp, fids))
        except Exception as e:
            print(f"skip {fp}: {e}")
    _cache_index[prefix] = out
    return out


def _all_frames(prefix):
    """returns dict frame_id -> first pkl_path that contains it."""
    fids = {}
    for fp, ids in _scan_prefix(prefix):
        for fid in ids:
            fids.setdefault(fid, fp)
    return fids


def _load_samples(pkl_path, frame_id):
    d = pickle.load(open(pkl_path, "rb"))
    return [s for s in d["samples"] if s["frame_id"] == frame_id]


def _forward_integrate(ego_5d, steer_deg, accel):
    T = len(steer_deg)
    out = np.zeros((T + 1, 4))
    v = float(ego_5d[0]); x, y, yaw = 0.0, 0.0, 0.0
    out[0] = [x, y, yaw, v]
    for t in range(T):
        delta = np.deg2rad(float(steer_deg[t])); a = float(accel[t])
        v_next = max(0.0, v + a * TS); v_mid = 0.5 * (v + v_next)
        x += v_mid * np.cos(yaw) * TS; y += v_mid * np.sin(yaw) * TS
        yaw += v_mid / L_WB * np.tan(delta) * TS; v = v_next
        out[t + 1] = [x, y, yaw, v]
    return out


def _project_to_cam(ex, ey, z=-0.13):
    pt = np.array([[ex, ey, z]])
    uv = _world_to_image(pt, K_CAM, R_V2C, T_V2C)[0]
    return float(uv[0]), float(uv[1]), float(uv[2])


def _img_to_base64(path):
    if not os.path.exists(path): return None
    img = Image.open(path).convert("RGB")
    img.thumbnail((1280, 720))  # already 1280×720
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _build_figure(sample, meta):
    s = sample
    fig = go.Figure()
    # image (will be on a separate plot — we'll return two figures)
    return fig


def _build_bev_fig(sample, meta):
    s = sample
    fig = go.Figure()
    # reference traj
    ref = np.asarray(s["traj_30"])
    fig.add_trace(go.Scatter(x=ref[:, 0], y=ref[:, 1], mode='lines+markers',
                              line=dict(color='black', dash='dash', width=2),
                              marker=dict(size=4),
                              name='reference traj (input)'))
    # GT-integrated traj
    sol = _forward_integrate(s["ego_5d"], s["gt_steer_30"], s["gt_accel_30"])
    fig.add_trace(go.Scatter(x=sol[:, 0], y=sol[:, 1], mode='lines+markers',
                              line=dict(color='green', width=3),
                              marker=dict(size=5),
                              name='GT-integrated solved traj'))
    # ego box
    fig.add_shape(type="rect", x0=-2.5, y0=-1, x1=2.5, y1=1,
                  line=dict(color="black"), fillcolor="lightgray", opacity=0.7)
    # all meta objects
    for o in meta.get("objects", []):
        ex, ey = _bbox_to_ego(o["bbox"])
        if not (1.5 < ex < 200 and abs(ey) < 50): continue
        L, W = 4.5, 2.0
        fig.add_shape(type="rect", x0=ex - L/2, y0=ey - W/2, x1=ex + L/2, y1=ey + W/2,
                      line=dict(color="gray"), fillcolor="lightgray", opacity=0.5)
        fig.add_annotation(x=ex, y=ey + W/2 + 0.5, text=f"({ex:.0f},{ey:.1f})",
                            showarrow=False, font=dict(color="gray", size=11))
    # lead (sample 의 pick_front_lead 결과)
    if s["lead_valid"]:
        lx, ly, vx_l, vy_l, L, W = [float(x) for x in s["lead_info"]]
        fig.add_shape(type="rect", x0=lx - L/2, y0=ly - W/2, x1=lx + L/2, y1=ly + W/2,
                      line=dict(color="red", width=3), fillcolor="salmon", opacity=0.7)
        fig.add_annotation(x=lx + vx_l, y=ly + vy_l, ax=lx, ay=ly, xref="x", yref="y",
                            axref="x", ayref="y", arrowhead=2, arrowcolor="red", arrowwidth=3,
                            text="", showarrow=True)
        speed = float(np.hypot(vx_l, vy_l))
        fig.add_annotation(x=lx, y=ly + W/2 + 1.5,
                            text=f"<b>LEAD</b><br>d={lx:.1f}m, y={ly:.1f}<br>|v|={speed:.1f}",
                            showarrow=False, font=dict(color="red", size=14))
        t_arr = np.arange(31) * TS
        fig.add_trace(go.Scatter(x=lx + vx_l * t_arr, y=ly + vy_l * t_arr,
                                  mode='lines', line=dict(color='red', dash='dot', width=2),
                                  name=f'lead const-vel ({speed:.1f}m/s)'))
    # axes
    xmax = max(60, ref[:, 0].max() + 10, sol[:, 0].max() + 10)
    ymax = max(10, abs(ref[:, 1]).max() + 3, abs(sol[:, 1]).max() + 3)
    fig.update_xaxes(range=[-5, xmax], title="x [m] (ego forward)")
    fig.update_yaxes(range=[-ymax, ymax], title="y [m] (ego left)",
                      scaleanchor="x", scaleratio=1)
    vx0 = float(s["ego_5d"][0])
    fig.update_layout(
        title=f"<b>{s['frame_id']}</b> [{s['traj_tag']}] — ego vx={vx0:.1f} m/s, lead_valid={s['lead_valid']}",
        showlegend=True, height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return fig


def _build_ctrl_fig(sample):
    t = np.arange(30) * TS
    s = sample
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=s["gt_accel_30"], mode='lines+markers',
                              line=dict(color='red', width=2), name='gt accel (m/s²)'))
    fig.add_trace(go.Scatter(x=t, y=s["gt_steer_30"], mode='lines+markers',
                              line=dict(color='blue', width=2), name='gt steer (deg)',
                              yaxis='y2'))
    fig.update_layout(
        title="GT control (MPC solver output)",
        xaxis_title='t (s)',
        yaxis_title='accel m/s²',
        yaxis2=dict(title='steer deg', overlaying='y', side='right'),
        height=350, legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return fig


def _image_with_markers_html(sample, meta):
    """Return <div> containing image with projected markers as inline SVG overlay."""
    s = sample
    img_b64 = _img_to_base64(s["image_curr_path"])
    if not img_b64:
        return "<div>image not found</div>"
    # collect markers: meta objects in/out FOV
    markers = []
    for o in meta.get("objects", []):
        ex, ey = _bbox_to_ego(o["bbox"])
        if not (1.5 < ex < 200 and abs(ey) < 50): continue
        u, v, z = _project_to_cam(ex, ey)
        in_fov = (z > 0 and 0 <= u < 1280 and 0 <= v < 720)
        markers.append((u, v, in_fov, ex, ey))
    # lead marker
    lead_marker = None
    if s["lead_valid"]:
        lx, ly = float(s["lead_info"][0]), float(s["lead_info"][1])
        u, v, z = _project_to_cam(lx, ly)
        in_fov = (z > 0 and 0 <= u < 1280 and 0 <= v < 720)
        lead_marker = (u, v, in_fov, lx, ly)

    svg_circles = []
    out_of_fov_text = []
    # ROI rect
    svg_circles.append('<rect x="0" y="235" width="1280" height="250" stroke="lime" stroke-width="3" fill="none"/>')
    for u, v, in_fov, ex, ey in markers:
        if in_fov:
            svg_circles.append(f'<circle cx="{u:.0f}" cy="{v:.0f}" r="22" stroke="yellow" stroke-width="4" fill="none"/>')
            svg_circles.append(f'<text x="{u+25:.0f}" y="{v:.0f}" fill="yellow" font-size="20" stroke="black" stroke-width="0.5">({ex:.0f},{ey:.1f})</text>')
        else:
            out_of_fov_text.append(f'({ex:.0f},{ey:.1f}) out-of-FOV')
    if lead_marker is not None:
        u, v, in_fov, lx, ly = lead_marker
        if in_fov:
            svg_circles.append(f'<circle cx="{u:.0f}" cy="{v:.0f}" r="30" stroke="red" stroke-width="5" fill="none"/>')
            svg_circles.append(f'<text x="{u+35:.0f}" y="{v:.0f}" fill="red" font-size="24" font-weight="bold" stroke="black" stroke-width="0.5">LEAD ({lx:.0f},{ly:.1f})</text>')
        else:
            out_of_fov_text.append(f'LEAD ({lx:.0f},{ly:.1f}) out-of-FOV')
    oof = "<br>".join(f'<span style="color:red">{t}</span>' for t in out_of_fov_text)
    return f"""
<div style="position:relative; width:1280px; max-width:100%;">
  <img src="data:image/jpeg;base64,{img_b64}" style="width:100%;display:block;"/>
  <svg viewBox="0 0 1280 720" style="position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;">
    {''.join(svg_circles)}
  </svg>
</div>
<div style="margin-top:5px;font-size:14px">{oof}</div>
"""


# ---------- API endpoints ----------

@app.get("/api/prefixes")
def api_prefixes():
    with _lock:
        out = {}
        for p in PREFIXES:
            cnt = len(list(glob.glob(os.path.join(OUT_DIR, f"{p}_*.pkl"))))
            out[p] = cnt
        return out


@app.get("/api/frames/{prefix}")
def api_frames(prefix: str):
    if prefix not in PREFIXES: raise HTTPException(404, "unknown prefix")
    with _lock:
        fids = sorted(_all_frames(prefix).keys())
    return {"prefix": prefix, "count": len(fids), "frames": fids}


@app.get("/api/samples/{prefix}/{frame_id}")
def api_samples(prefix: str, frame_id: str):
    with _lock:
        fids = _all_frames(prefix)
        if frame_id not in fids: raise HTTPException(404, "frame not in this prefix")
        samples = _load_samples(fids[frame_id], frame_id)
    return {"frame_id": frame_id, "n": len(samples),
             "tags": [s["traj_tag"] for s in samples]}


@app.get("/view/{prefix}/{frame_id}/{sample_idx}", response_class=HTMLResponse)
def view(prefix: str, frame_id: str, sample_idx: int):
    with _lock:
        fids = _all_frames(prefix)
        if frame_id not in fids: raise HTTPException(404, "frame")
        samples = _load_samples(fids[frame_id], frame_id)
    if sample_idx >= len(samples): raise HTTPException(404, "sample idx")
    s = samples[sample_idx]
    meta = _load_meta().get(frame_id, {})

    img_html = _image_with_markers_html(s, meta)
    bev_fig = _build_bev_fig(s, meta)
    ctrl_fig = _build_ctrl_fig(s)
    bev_html = bev_fig.to_html(full_html=False, include_plotlyjs='cdn')
    ctrl_html = ctrl_fig.to_html(full_html=False, include_plotlyjs=False)

    nav = f"""
    <div style="margin-bottom:10px">
      [<a href='/'>home</a>]
      prefix: <b>{prefix}</b> | frame: <b>{frame_id}</b> | sample: <b>{sample_idx+1}/{len(samples)}</b> ({s['traj_tag']})
      |
      <a href='/view/{prefix}/{frame_id}/{(sample_idx-1)%len(samples)}'>prev</a>
      <a href='/view/{prefix}/{frame_id}/{(sample_idx+1)%len(samples)}'>next</a>
    </div>
    """
    return f"""
<!doctype html>
<html><head><meta charset="utf-8"><title>SCC viz: {frame_id}/{sample_idx}</title>
<style>body {{ font-family: sans-serif; margin: 20px; }}</style>
</head><body>
{nav}
<h2>CAMERA_1 (with object projection)</h2>
{img_html}
<h2>BEV (ego frame): ref + solved + lead + meta objects</h2>
{bev_html}
<h2>GT control</h2>
{ctrl_html}
</body></html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    p_list = api_prefixes()
    p_html = "".join(
        f'<li><a href="/browse/{p}">{p}</a> ({c} pkl)</li>'
        for p, c in p_list.items() if c > 0
    )
    return f"""
<!doctype html><html><head><title>SCC synth viz</title>
<style>body{{font-family:sans-serif;margin:30px}}</style></head><body>
<h1>SCC synth dataset browser</h1>
<p>Prefix 선택 → frame → sample.</p>
<ul>{p_html}</ul>
</body></html>"""


@app.get("/browse/{prefix}", response_class=HTMLResponse)
def browse(prefix: str):
    with _lock:
        fids = sorted(_all_frames(prefix).keys())
    if not fids: return HTMLResponse(f"no frames for {prefix}", status_code=404)
    items = "".join(
        f'<li><a href="/view/{prefix}/{fid}/0">{fid}</a></li>' for fid in fids[:500]
    )
    return f"""
<!doctype html><html><head><title>browse {prefix}</title>
<style>body{{font-family:sans-serif;margin:30px}} li{{font-family:monospace}}</style></head><body>
<h1>{prefix} — {len(fids)} frames (showing first 500)</h1>
[<a href='/'>home</a>]
<ul>{items}</ul>
</body></html>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
