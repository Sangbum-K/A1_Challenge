"""데이터셋 분포 점검 — 기존 + 신규 4 policy 통합 시각화.

각 policy 별:
  1. close_lead 분포 (close <30m / far 30+ / no-lead)
  2. obj 분포 (frame 당 obj 수, per-bin coverage)
  3. ego vx 분포 (raw EGO_INFO sample 기반)
  4. steer/ax 분포 (.scc.txt random sample)
  5. 합성 진행률

출력: outputs_multi/viz_resynth/dataset_dist/{index.html, plots/*.png}
"""
from __future__ import annotations
import os, json, glob, random
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/ailab-12/git/scc_controller_imitator")
DATA = Path("/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data")
RAW = Path("/home/ailab-12/AILabSSD/02_Data/57_A1_E2E/00_raw")
OUT = ROOT / "outputs_multi/viz_resynth/dataset_dist"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "plots").mkdir(exist_ok=True)

random.seed(0); np.random.seed(0)


# ───────── policies ─────────
POLICIES = [
    ("기존 spd",         "speedway_track", "0519_agent1_pertX_pitX", "scc_synth_v2_spd",            "scc_frame_list_spd.txt"),
    ("기존 sangam pertO","sangam_track",   "0519_pertO_pitX",        "scc_synth_v2_sangam_pertO",    "scc_frame_list_sangam_pertO.txt"),
    ("기존 sangam pertX","sangam_track",   "0519_pertX_pitX",        "scc_synth_v2_sangam_pertX",    "scc_frame_list_sangam_pertX.txt"),
    ("신규 spd no-lead 0520", "speedway_track", "0520_agent1_pertO_pitX",  "scc_synth_v3_spd_nolead_0520",   "scc_frame_list_spd_nolead_0520.txt"),
    ("신규 spd no-lead 0521", "speedway_track", "0521_agent1_pertX_pitO",  "scc_synth_v3_spd_nolead_0521",   "scc_frame_list_spd_nolead_0521.txt"),
    ("신규 sangam lead 0520", "sangam_track",   "0520_agent1_pertX_pitX",  "scc_synth_v3_sangam_lead_0520",  "scc_frame_list_sangam_lead_0520.txt"),
    ("신규 sangam lead 0521", "sangam_track",   "0521_agent1_pertO_pitX",  "scc_synth_v3_sangam_lead_0521",  "scc_frame_list_sangam_lead_0521.txt"),
]


# ───────── close_lead_meta ─────────
print("[load] close_lead_meta_v3 …")
cz = np.load(DATA / "cache/close_lead_meta_v3.npz")
cl_map = dict(zip(cz["keys"].tolist(), cz["dists"]))


# ───────── helpers ─────────
def parse_ego_world(p):
    if not p.exists(): return None
    vx = None
    try:
        for l in open(p):
            if l.startswith("velocity:"):
                nums = [float(t) for t in l.split(":", 1)[1].split()]
                vx = nums[0]; break
    except: return None
    return vx


def parse_scc_txt(p):
    """returns (steer_deg[40], ax[40]) or (None, None)"""
    if not p.exists(): return None, None
    try:
        lines = p.read_text().splitlines()
        long_t, lat_t = [], []
        sec = None
        for l in lines:
            if l.startswith("LONG_TRAJ"): sec = "long"; continue
            if l.startswith("LAT_TRAJ"):  sec = "lat";  continue
            if l.startswith("#") or not l.strip(): continue
            toks = l.split()
            if len(toks) < 4: continue
            try:
                if sec == "long":
                    long_t.append(float(toks[3]))    # ax
                elif sec == "lat":
                    lat_t.append(float(toks[4]))     # steer (deg)
            except: pass
        if not long_t or not lat_t: return None, None
        return np.array(lat_t[:40]), np.array(long_t[:40])
    except: return None, None


def collect_policy(label, track, policy, synth_dir, fl):
    """returns dict with all stats."""
    res = dict(label=label, track=track, policy=policy)
    # frame_list size
    flp = DATA / fl
    n_fl = sum(1 for _ in open(flp)) if flp.exists() else 0
    res["frame_list"] = n_fl
    # 합성 진행
    synth = DATA / synth_dir
    n_scc = int(os.popen(f"find {synth} -name '*.scc.txt' 2>/dev/null | wc -l").read().strip() or 0) if synth.exists() else 0
    res["synth_files"] = n_scc
    # close_lead 분포 (frame_list 의 frame_key 들)
    frame_keys = []
    if flp.exists():
        for l in open(flp):
            toks = l.split()
            if len(toks) >= 2:
                run = toks[0]; tk = int(toks[1])
                frame_keys.append(f"{run}__t{tk:07d}")
    dists = np.array([cl_map.get(k, np.inf) for k in frame_keys]) if frame_keys else np.array([])
    if len(dists) > 0:
        res["close"]  = int((dists < 30).sum())
        res["far"]    = int(((dists >= 30) & np.isfinite(dists)).sum())
        res["nolead"] = int((~np.isfinite(dists)).sum())
    else:
        res["close"] = res["far"] = res["nolead"] = 0

    # ego vx 분포 (raw EGO_INFO 무작위 200 sample)
    vxs = []
    smp_keys = random.sample(frame_keys, min(200, len(frame_keys))) if frame_keys else []
    for k in smp_keys:
        run, tail = k.rsplit("__t", 1); tk = int(tail)
        p = RAW / track / policy / run / "EGO_INFO" / f"{tk}.txt"
        v = parse_ego_world(p)
        if v is not None: vxs.append(v)
    res["vxs"] = np.array(vxs)

    # steer/ax 분포 (합성된 .scc.txt 무작위 100 sample)
    if n_scc > 0:
        scc_files = []
        # logged + first 10 rand
        all_files = list(synth.glob("*__logged.scc.txt"))
        smp_files = random.sample(all_files, min(100, len(all_files)))
        steer_all = []; ax_all = []
        for f in smp_files:
            st, ax = parse_scc_txt(f)
            if st is None: continue
            steer_all.extend(st.tolist()); ax_all.extend(ax.tolist())
        res["steer"] = np.array(steer_all)
        res["ax"]    = np.array(ax_all)
    else:
        res["steer"] = np.array([]); res["ax"] = np.array([])

    return res


# ───────── collect all ─────────
print("[collect] all policies …")
ALL = []
for spec in POLICIES:
    print(f"  - {spec[0]}")
    ALL.append(collect_policy(*spec))


# ───────── plots ─────────

# A. close-lead 분포 (각 policy 별 stacked bar)
fig, ax = plt.subplots(figsize=(13, 5))
labels = [r["label"] for r in ALL]
close = [r["close"] for r in ALL]
far   = [r["far"]   for r in ALL]
nolead= [r["nolead"] for r in ALL]
x = np.arange(len(labels))
ax.bar(x, close, label="close (<30m)", color="#E74C3C")
ax.bar(x, far, bottom=close, label="far (30+m)", color="#F39C12")
ax.bar(x, nolead, bottom=np.array(close)+np.array(far), label="no-lead", color="#5D6D7E")
for i, r in enumerate(ALL):
    total = r["close"] + r["far"] + r["nolead"]
    if total > 0:
        ax.text(i, total + total*0.02, f"{total:,}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("frames"); ax.set_title("close-lead 분포 (frame_list 기준, 절대값)")
ax.legend()
plt.tight_layout(); plt.savefig(OUT / "plots/close_dist_abs.png", dpi=120); plt.close()

# B. close-lead 비율 (%)
fig, ax = plt.subplots(figsize=(13, 5))
ratios_c = []; ratios_f = []; ratios_n = []
for r in ALL:
    t = r["close"] + r["far"] + r["nolead"]
    if t == 0: ratios_c.append(0); ratios_f.append(0); ratios_n.append(0); continue
    ratios_c.append(r["close"] / t * 100)
    ratios_f.append(r["far"] / t * 100)
    ratios_n.append(r["nolead"] / t * 100)
ax.bar(x, ratios_c, label="close (<30m)", color="#E74C3C")
ax.bar(x, ratios_f, bottom=ratios_c, label="far (30+m)", color="#F39C12")
ax.bar(x, ratios_n, bottom=np.array(ratios_c)+np.array(ratios_f), label="no-lead", color="#5D6D7E")
for i, (c, n) in enumerate(zip(ratios_c, ratios_n)):
    if c > 5: ax.text(i, c/2, f"{c:.0f}%", ha="center", fontsize=9, color="white", fontweight="bold")
    if n > 5: ax.text(i, 100-n/2, f"{n:.0f}%", ha="center", fontsize=9, color="white", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("ratio (%)"); ax.set_ylim(0, 110)
ax.set_title("close-lead 비율 (%) — 신규 데이터는 분포 보강")
ax.legend()
plt.tight_layout(); plt.savefig(OUT / "plots/close_dist_pct.png", dpi=120); plt.close()

# C. ego vx 분포 (overlay histogram)
fig, ax = plt.subplots(figsize=(12, 5))
colors = plt.cm.tab10.colors
for i, r in enumerate(ALL):
    if len(r["vxs"]) == 0: continue
    ax.hist(r["vxs"], bins=30, alpha=0.45, label=f'{r["label"]} (n={len(r["vxs"])})', color=colors[i % 10])
ax.set_xlabel("ego vx (m/s)"); ax.set_ylabel("frame count (sampled 200)")
ax.set_title("ego vx 분포 (200 frame sample / policy)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUT / "plots/ego_vx_dist.png", dpi=120); plt.close()

# D. steer 분포 (overlay)
fig, ax = plt.subplots(figsize=(12, 5))
for i, r in enumerate(ALL):
    if len(r["steer"]) == 0: continue
    ax.hist(r["steer"], bins=50, alpha=0.45, label=f'{r["label"]} (n={len(r["steer"])})', color=colors[i % 10])
ax.set_xlabel("steer (deg, lat_traj GT)"); ax.set_ylabel("count")
ax.set_xlim(-30, 30)
ax.set_title("steer (lat_traj GT) 분포 — 100 logged sample × 40 steps")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUT / "plots/steer_dist.png", dpi=120); plt.close()

# E. ax 분포 (overlay)
fig, ax = plt.subplots(figsize=(12, 5))
for i, r in enumerate(ALL):
    if len(r["ax"]) == 0: continue
    ax.hist(r["ax"], bins=50, alpha=0.45, label=f'{r["label"]} (n={len(r["ax"])})', color=colors[i % 10])
ax.set_xlabel("ax (m/s², long_traj GT)"); ax.set_ylabel("count")
ax.set_xlim(-22, 22)
ax.set_title("ax (long_traj GT) 분포 — 100 logged sample × 40 steps")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUT / "plots/ax_dist.png", dpi=120); plt.close()

# F. 합성 진행률
fig, ax = plt.subplots(figsize=(11, 4))
done_pct = []
for r in ALL:
    expected = r["frame_list"] * 101    # 1 logged + 100 rand
    if expected == 0: done_pct.append(0); continue
    done_pct.append(r["synth_files"] / expected * 100)
ax.barh(labels[::-1], done_pct[::-1], color=["#E74C3C" if "신규" in l else "#27AE60" for l in labels[::-1]])
for i, p in enumerate(done_pct[::-1]):
    ax.text(p + 1, i, f"{p:.1f}%", va="center", fontsize=9)
ax.set_xlabel("합성 진행률 (%)"); ax.set_xlim(0, 110)
ax.set_title("합성 진행률 (frame_list × 101 sample/frame)")
ax.grid(alpha=0.3, axis="x")
plt.tight_layout(); plt.savefig(OUT / "plots/synth_progress.png", dpi=120); plt.close()


# ───────── HTML ─────────
html = ['<!doctype html><html><head><meta charset=utf-8>',
        '<title>데이터셋 분포 점검</title>',
        '<style>body{font-family:sans-serif;background:#1a1a2e;color:#e8e8e8;margin:18px;}',
        'h1{color:#fff;border-bottom:3px solid #C0392B;padding-bottom:8px;}',
        'h2{color:#F39C12;border-left:5px solid #F39C12;padding-left:10px;margin-top:30px;}',
        'table{border-collapse:collapse;width:100%;margin:8px 0;font-size:13px;}',
        'th{background:#34495E;color:#fff;padding:8px;text-align:left;}',
        'td{padding:6px 10px;border-bottom:1px solid #444;}',
        '.plot{margin:10px 0;background:#fff;border-radius:6px;padding:5px;}',
        '.plot img{width:100%;display:block;border-radius:4px;}',
        '.new{background:#7B241C;}',
        '</style></head><body>',
        '<h1>데이터셋 분포 점검 (기존 + 신규 4 policy)</h1>',
        f'<p>생성 시각: {os.popen("date").read().strip()}</p>']

# A: close-lead
html.append('<h2>① close-lead 분포</h2>')
html.append('<p>신규 데이터의 의도: '
            '<b>speedway no-lead</b> (전방차 없음 강화) + <b>sangam lead</b> (전방차 있음 강화).</p>')
html.append('<div class="plot"><img src="plots/close_dist_abs.png"></div>')
html.append('<div class="plot"><img src="plots/close_dist_pct.png"></div>')
html.append('<table><tr><th>policy</th><th>총 frames</th><th>close (<30m)</th><th>far (30+m)</th><th>no-lead</th></tr>')
for r in ALL:
    cls = "new" if "신규" in r["label"] else ""
    t = r["close"] + r["far"] + r["nolead"]
    cpct = r["close"]/t*100 if t else 0
    fpct = r["far"]/t*100 if t else 0
    npct = r["nolead"]/t*100 if t else 0
    html.append(f'<tr class="{cls}"><td>{r["label"]}</td><td>{t:,}</td>'
                f'<td>{r["close"]:,} ({cpct:.0f}%)</td>'
                f'<td>{r["far"]:,} ({fpct:.0f}%)</td>'
                f'<td>{r["nolead"]:,} ({npct:.0f}%)</td></tr>')
html.append('</table>')

# C: ego vx
html.append('<h2>② ego vx 분포 (속도 다양성)</h2>')
html.append('<div class="plot"><img src="plots/ego_vx_dist.png"></div>')

# D-E: steer/ax
html.append('<h2>③ steer (lat_traj GT) 분포</h2>')
html.append('<div class="plot"><img src="plots/steer_dist.png"></div>')
html.append('<h2>④ ax (long_traj GT) 분포</h2>')
html.append('<p>합성 ax cap ±20 m/s² 이라 분포는 그 범위 안. 신규 mpc HPIPM warmup 20 적용 → cold-start spike 제거.</p>')
html.append('<div class="plot"><img src="plots/ax_dist.png"></div>')

# F: 합성 진행률
html.append('<h2>⑤ 합성 진행률</h2>')
html.append('<p>기존 (초록) = 보존 + 정지 / 신규 (빨강) = 가동 중. 신규 합성 frame 당 101 sample = 1 logged + 100 random.</p>')
html.append('<div class="plot"><img src="plots/synth_progress.png"></div>')
html.append('<table><tr><th>policy</th><th>frame_list</th><th>합성된 sample</th><th>예상 total</th><th>진행률</th></tr>')
for r, p in zip(ALL, done_pct):
    cls = "new" if "신규" in r["label"] else ""
    exp = r["frame_list"] * 101
    html.append(f'<tr class="{cls}"><td>{r["label"]}</td><td>{r["frame_list"]:,}</td>'
                f'<td>{r["synth_files"]:,}</td><td>{exp:,}</td><td>{p:.1f}%</td></tr>')
html.append('</table>')

html.append('</body></html>')
(OUT / "index.html").write_text("\n".join(html))
print(f"[done] → {OUT / 'index.html'}")
