"""학습/데이터 생성 dashboard — HTML + plots 자동 생성.

매 INTERVAL 초마다 호출 (외부 loop 또는 cron):
  python scripts/dashboard_status.py

출력: outputs_multi/viz_resynth/dashboard/{index.html, plots/*.png}
HTML 은 meta http-equiv refresh=60 으로 자동 새로고침.
"""
from __future__ import annotations
import os, re, json, glob, time, subprocess
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/ailab-12/git/scc_controller_imitator")
DATA = Path("/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data")
DASH = ROOT / "outputs_multi/viz_resynth/dashboard"
DASH.mkdir(parents=True, exist_ok=True)
(DASH / "plots").mkdir(exist_ok=True)


# ───────── helpers ─────────
def sh(cmd, default=""):
    try:
        return subprocess.run(cmd, shell=True, check=False,
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return default


def file_count(pat: str) -> int:
    return int(sh(f"find {pat} -name '*.scc.txt' 2>/dev/null | wc -l") or 0)


def line_count(p: Path) -> int:
    if not p.exists(): return 0
    return int(sh(f"wc -l < {p}") or 0)


def proc_count(pattern: str) -> int:
    return int(sh(f"pgrep -fc '{pattern}' || echo 0") or 0)


# ───────── data: 합성/bank/logged/aux ─────────
def collect_data():
    rows = []
    # 기존 3 track
    for tag, sdir in [("spd (기존)", "scc_synth_v2_spd"),
                       ("sangam_pertO (기존)", "scc_synth_v2_sangam_pertO"),
                       ("sangam_pertX (기존)", "scc_synth_v2_sangam_pertX")]:
        n = file_count(str(DATA / sdir))
        rows.append((tag, sdir, n, "보존 (합성 중단)"))
    # 새 4 policy 합성 dir (있으면)
    for tag, sdir in [("spd_nolead_0520 (신규)", "scc_synth_v3_spd_nolead_0520"),
                       ("spd_nolead_0521 (신규)", "scc_synth_v3_spd_nolead_0521"),
                       ("sangam_lead_0520 (신규)", "scc_synth_v3_sangam_lead_0520"),
                       ("sangam_lead_0521 (신규)", "scc_synth_v3_sangam_lead_0521")]:
        d = DATA / sdir
        if not d.exists():
            rows.append((tag, sdir, 0, "합성 미시작"))
            continue
        n = file_count(str(d))
        rows.append((tag, sdir, n, "합성 진행 중"))
    return rows


def collect_banks():
    rows = []
    for tag, bdir, fl in [
        ("spd", "scc_traj_bank_spd", "scc_frame_list_spd.txt"),
        ("sangam_pertO", "scc_traj_bank_sangam_pertO", "scc_frame_list_sangam_pertO.txt"),
        ("sangam_pertX", "scc_traj_bank_sangam_pertX", "scc_frame_list_sangam_pertX.txt"),
        ("spd_nolead_0520 (신규)", "scc_traj_bank_spd_nolead_0520", "scc_frame_list_spd_nolead_0520.txt"),
        ("spd_nolead_0521 (신규, stride5)", "scc_traj_bank_spd_nolead_0521", "scc_frame_list_spd_nolead_0521.txt"),
        ("sangam_lead_0520 (신규)", "scc_traj_bank_sangam_lead_0520", "scc_frame_list_sangam_lead_0520.txt"),
        ("sangam_lead_0521 (신규)", "scc_traj_bank_sangam_lead_0521", "scc_frame_list_sangam_lead_0521.txt"),
    ]:
        n_bank = int(sh(f"ls {DATA}/{bdir} 2>/dev/null | wc -l") or 0)
        n_logged = int(sh(f"ls {DATA}/{bdir.replace('traj_bank','logged_bank')} 2>/dev/null | wc -l") or 0)
        n_aux = 0
        aux_pol = {
            "spd": "aux_targets_spd_fx640",
            "spd_nolead_0520 (신규)": "aux_targets_spd_nolead_0520_fx640",
            "spd_nolead_0521 (신규, stride5)": "aux_targets_spd_nolead_0521_fx640",
            "sangam_lead_0520 (신규)": "aux_targets_sangam_lead_0520_fx640",
            "sangam_lead_0521 (신규)": "aux_targets_sangam_lead_0521_fx640",
        }
        for k, v in aux_pol.items():
            if tag.startswith(k.split(" ")[0]):
                n_aux = int(sh(f"ls {DATA}/{v} 2>/dev/null | wc -l") or 0)
                break
        n_fl = line_count(DATA / fl)
        rows.append((tag, n_bank, n_logged, n_aux, n_fl))
    return rows


# ───────── 분포 (close vs no-lead) ─────────
def collect_close_dist():
    """현재 prebake 의 close_lead_meta 분포 + obj_target 활성 비율."""
    pb = DATA / "prebake/f1_prebake_iy_spd_streaming.npz"
    if not pb.exists():
        return None
    try:
        z = np.load(pb, allow_pickle=True)
        ck = z["curr_keys"].tolist()
        obj = z["obj_target"]
        cz = np.load(DATA / "cache/close_lead_meta_v3.npz")
        cl_map = dict(zip(cz["keys"].tolist(), cz["dists"]))
        dists = np.array([cl_map.get(k, np.inf) for k in ck])
        return dict(
            N=len(dists),
            close=int((dists < 30).sum()),
            far=int(((dists >= 30) & np.isfinite(dists)).sum()),
            nolead=int((~np.isfinite(dists)).sum()),
            obj_any_pct=float((obj.sum(1) > 0).mean() * 100),
            obj_mean_per_bin=float(obj.mean()),
        )
    except Exception:
        return None


# ───────── 학습 ─────────
def collect_train(log_path: Path):
    if not log_path.exists():
        return None
    recs = []; rels = []
    for line in open(log_path):
        m = re.match(r"\[ep (\d+)\] train=([\d.]+) val=([\d.]+) \(steer=([\d.]+)° ax=([\d.]+)m/s²\)", line)
        if m: recs.append((int(m.group(1)), float(m.group(2)), float(m.group(3)),
                           float(m.group(4)), float(m.group(5))))
        if "new total" in line:
            rm = re.search(r"total=(\d+)", line)
            if rm: rels.append((recs[-1][0] if recs else 0, int(rm.group(1))))
    if not recs: return None
    b = min(recs, key=lambda r: r[2])
    return dict(
        epochs=len(recs),
        reloads=len(rels),
        best=b, latest=recs[-1],
        recs=recs, rels=rels,
        log=log_path.name,
    )


# ───────── plots ─────────
def plot_train_curves(tr):
    if tr is None: return None
    eps = [r[0] for r in tr["recs"]]
    vals = [r[2] for r in tr["recs"]]
    trs = [r[1] for r in tr["recs"]]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(eps, trs, "-", color="#7F8C8D", lw=1.0, label="train")
    ax.plot(eps, vals, "-", color="#C0392B", lw=1.6, label="val")
    for ep_r, _ in tr["rels"]:
        ax.axvline(ep_r, color="#3498DB", linestyle=":", alpha=0.4, lw=1.0)
    ax.scatter([tr["best"][0]], [tr["best"][2]], color="#27AE60", s=80, marker="*",
               zorder=5, label=f"best ep{tr['best'][0]} val={tr['best'][2]:.4f}")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.grid(alpha=0.3); ax.legend()
    ax.set_title(f"train/val loss — {tr['log']} (blue dashed = stream-reload)")
    p = DASH / "plots/train_curves.png"
    plt.tight_layout(); plt.savefig(p, dpi=110); plt.close()
    return p


def plot_train_steer_ax(tr):
    if tr is None: return None
    eps = [r[0] for r in tr["recs"]]
    sts = [r[3] for r in tr["recs"]]
    axs_ = [r[4] for r in tr["recs"]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(eps, sts, "-", color="#8E44AD", lw=1.5)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("val steer MAE (°)")
    axes[0].set_title(f"steer MAE   latest={sts[-1]:.3f}°  min={min(sts):.3f}°")
    axes[0].grid(alpha=0.3)
    axes[1].plot(eps, axs_, "-", color="#16A085", lw=1.5)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val ax MAE (m/s²)")
    axes[1].set_title(f"ax MAE   latest={axs_[-1]:.3f}  min={min(axs_):.3f} m/s²")
    axes[1].grid(alpha=0.3)
    p = DASH / "plots/train_steer_ax.png"
    plt.tight_layout(); plt.savefig(p, dpi=110); plt.close()
    return p


def plot_data_growth(tr):
    if tr is None or not tr["rels"]: return None
    eps = [0] + [r[0] for r in tr["rels"]]
    sizes = [tr["rels"][0][1]] + [r[1] for r in tr["rels"]]
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.step(eps, sizes, where="post", color="#2980B9", lw=2)
    ax.fill_between(eps, sizes, step="post", alpha=0.2, color="#2980B9")
    ax.set_xlabel("epoch"); ax.set_ylabel("training samples")
    ax.set_title(f"streaming data growth  ({sizes[0]:,} → {sizes[-1]:,} samples, {len(tr['rels'])} reloads)")
    ax.grid(alpha=0.3)
    p = DASH / "plots/data_growth.png"
    plt.tight_layout(); plt.savefig(p, dpi=110); plt.close()
    return p


def plot_close_dist(cd):
    if cd is None: return None
    fig, ax = plt.subplots(figsize=(6, 6))
    labels = [f"close <30m\n{cd['close']:,}", f"far ≥30m\n{cd['far']:,}", f"no-lead\n{cd['nolead']:,}"]
    sizes  = [cd["close"], cd["far"], cd["nolead"]]
    colors = ["#E74C3C", "#F39C12", "#5D6D7E"]
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90,
           wedgeprops=dict(edgecolor="white", linewidth=2))
    ax.set_title(f"학습 data lead 분포 (N={cd['N']:,})\n"
                  f"obj_any={cd['obj_any_pct']:.1f}% (옆차 포함)")
    p = DASH / "plots/close_dist.png"
    plt.tight_layout(); plt.savefig(p, dpi=110); plt.close()
    return p


# ───────── system info ─────────
def collect_system():
    load = sh("uptime").split("load average:")[-1].strip() if "load average:" in sh("uptime") else "?"
    gpu = sh("nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | head -1")
    return dict(
        time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        load=load,
        gpu=gpu,
        synth=proc_count("synth_batch"),
        train=proc_count("scripts/train_iy"),
        prebake_loop=proc_count("streaming_prebake_loop"),
        bank_gen=proc_count("prepare_new_run_bank"),
        logged_gen=proc_count("prepare_logged_trajs"),
        aux_gen=proc_count("prepare_a2_aux_targets"),
        closed_loop=proc_count("closed_loop_node"),
    )


# ───────── render HTML ─────────
def render():
    data_rows = collect_data()
    bank_rows = collect_banks()
    cd = collect_close_dist()
    sys_info = collect_system()
    tr = collect_train(ROOT / "logs/train_spd_objgate_close.log")

    plot_train_curves(tr)
    plot_train_steer_ax(tr)
    plot_data_growth(tr)
    plot_close_dist(cd)

    html = ['<!doctype html><html><head><meta charset=utf-8>',
            '<title>SCC controller imitator — dashboard</title>',
            '<meta http-equiv="refresh" content="60">',
            '<style>',
            'body{font-family:sans-serif;background:#1a1a2e;color:#e8e8e8;margin:18px;}',
            'h1{color:#fff;border-bottom:3px solid #C0392B;padding-bottom:8px;}',
            'h2{color:#F39C12;border-left:5px solid #F39C12;padding-left:10px;margin-top:30px;}',
            'table{border-collapse:collapse;width:100%;margin:8px 0;}',
            'th{background:#34495E;color:#fff;padding:8px;text-align:left;font-size:13px;}',
            'td{padding:6px 10px;border-bottom:1px solid #444;font-size:13px;}',
            '.metric{display:inline-block;background:#2C3E50;padding:10px 16px;margin:5px;border-radius:6px;}',
            '.metric .label{font-size:11px;color:#aaa;}',
            '.metric .value{font-size:18px;font-weight:bold;color:#fff;}',
            '.metric .sub{font-size:11px;color:#888;}',
            '.plot{margin:10px 0;background:#fff;border-radius:6px;padding:5px;}',
            '.plot img{width:100%;display:block;border-radius:4px;}',
            '.proc-on{color:#27AE60;font-weight:bold;}',
            '.proc-off{color:#7F8C8D;}',
            '.row{display:flex;gap:14px;flex-wrap:wrap;}',
            '.col{flex:1;min-width:480px;}',
            '</style></head><body>',
            f'<h1>SCC controller imitator — dashboard  '
            f'<span style="font-size:13px;color:#aaa;font-weight:normal">'
            f'(refresh 60s · {sys_info["time"]})</span></h1>']

    # 상단 metric cards
    html.append('<div class="row">')
    if tr:
        html.append(f'<div class="metric"><div class="label">EPOCHS</div>'
                    f'<div class="value">{tr["epochs"]}</div>'
                    f'<div class="sub">{tr["reloads"]} stream reloads</div></div>')
        html.append(f'<div class="metric"><div class="label">BEST VAL</div>'
                    f'<div class="value" style="color:#27AE60">{tr["best"][2]:.4f}</div>'
                    f'<div class="sub">ep {tr["best"][0]} (s={tr["best"][3]:.3f}°, ax={tr["best"][4]:.3f})</div></div>')
        html.append(f'<div class="metric"><div class="label">LATEST VAL</div>'
                    f'<div class="value">{tr["latest"][2]:.4f}</div>'
                    f'<div class="sub">ep {tr["latest"][0]} (s={tr["latest"][3]:.3f}°, ax={tr["latest"][4]:.3f})</div></div>')
    if cd:
        html.append(f'<div class="metric"><div class="label">학습 DATA</div>'
                    f'<div class="value">{cd["N"]:,}</div>'
                    f'<div class="sub">close {cd["close"]/cd["N"]*100:.0f}% · no-lead {cd["nolead"]/cd["N"]*100:.0f}%</div></div>')
    html.append(f'<div class="metric"><div class="label">GPU</div>'
                f'<div class="value">{sys_info["gpu"].split(",")[2].strip() if sys_info["gpu"] else "?"}</div>'
                f'<div class="sub">{sys_info["gpu"]}</div></div>')
    html.append(f'<div class="metric"><div class="label">LOAD</div>'
                f'<div class="value">{sys_info["load"].split(",")[0]}</div>'
                f'<div class="sub">1m / 5m / 15m</div></div>')
    html.append('</div>')

    # 학습 plots
    html.append('<h2>학습 진행</h2>')
    html.append('<div class="plot"><img src="plots/train_curves.png"></div>')
    html.append('<div class="row"><div class="col"><div class="plot"><img src="plots/train_steer_ax.png"></div></div></div>')
    html.append('<div class="plot"><img src="plots/data_growth.png"></div>')

    # 학습 data 분포
    html.append('<h2>학습 data 분포</h2>')
    html.append('<div class="row">')
    html.append('<div class="col"><div class="plot"><img src="plots/close_dist.png"></div></div>')
    html.append('<div class="col">')
    if cd:
        html.append('<table><tr><th>지표</th><th>값</th></tr>')
        html.append(f'<tr><td>N samples</td><td>{cd["N"]:,}</td></tr>')
        html.append(f'<tr><td>close (<30m)</td><td>{cd["close"]:,} ({cd["close"]/cd["N"]*100:.1f}%)</td></tr>')
        html.append(f'<tr><td>far (30m+)</td><td>{cd["far"]:,} ({cd["far"]/cd["N"]*100:.1f}%)</td></tr>')
        html.append(f'<tr><td>no-lead (close_meta inf)</td><td>{cd["nolead"]:,} ({cd["nolead"]/cd["N"]*100:.1f}%)</td></tr>')
        html.append(f'<tr><td>obj_any in FOV</td><td>{cd["obj_any_pct"]:.1f}%</td></tr>')
        html.append(f'<tr><td>obj_mean (per-bin avg)</td><td>{cd["obj_mean_per_bin"]:.4f}</td></tr>')
        html.append('</table>')
    html.append('</div></div>')

    # 데이터 생성 현황
    html.append('<h2>합성 데이터 현황 (.scc.txt)</h2>')
    html.append('<table><tr><th>분류</th><th>dir</th><th>files</th><th>상태</th></tr>')
    for tag, sdir, n, status in data_rows:
        html.append(f'<tr><td>{tag}</td><td><code>{sdir}</code></td>'
                    f'<td>{n:,}</td><td>{status}</td></tr>')
    html.append('</table>')

    # bank / aux 현황
    html.append('<h2>bank / logged / aux_targets 현황</h2>')
    html.append('<table><tr><th>policy</th><th>frame_list</th><th>traj_bank</th><th>logged_bank</th><th>aux_targets</th></tr>')
    for tag, n_bank, n_logged, n_aux, n_fl in bank_rows:
        html.append(f'<tr><td>{tag}</td><td>{n_fl:,}</td>'
                    f'<td>{n_bank:,}</td><td>{n_logged:,}</td><td>{n_aux:,}</td></tr>')
    html.append('</table>')

    # 가동 process
    html.append('<h2>가동 process</h2>')
    html.append('<div class="row">')
    for label, key in [("synth_batch", "synth"), ("train_iy", "train"),
                        ("streaming_prebake_loop", "prebake_loop"),
                        ("prepare_new_run_bank", "bank_gen"),
                        ("prepare_logged_trajs", "logged_gen"),
                        ("prepare_a2_aux_targets", "aux_gen"),
                        ("closed_loop_node", "closed_loop")]:
        n = sys_info[key]
        cls = "proc-on" if n > 0 else "proc-off"
        html.append(f'<div class="metric"><div class="label">{label}</div>'
                    f'<div class="value {cls}">{n}</div></div>')
    html.append('</div>')

    html.append('</body></html>')
    (DASH / "index.html").write_text("\n".join(html))
    print(f"[dashboard] {sys_info['time']}  → {DASH / 'index.html'}")


if __name__ == "__main__":
    render()
