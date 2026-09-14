"""GT (MPC 합성) brake quality 검수.

close-lead 가 가까울 / approach 빠를 때 GT first-step ax 가 충분한 brake 결정인지 확인.

분석 차원:
  1. dist 별 GT first-step ax 분포
  2. v_long_rel (lead approach rate) 별 GT ax
  3. 2D heatmap: (dist, v_long_rel) → GT ax mean
  4. 이상 case: dist<10m + v_long_rel<-5 (highly approach) 인데 GT ax > -1 (no brake)

출력: outputs_multi/viz_resynth/gt_brake_audit/
"""
from __future__ import annotations
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data")
RAW = Path("/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/raw_cache/speedway_track/0519_agent1_pertX_pitX")
ROI_Y0, ROI_Y1 = 235, 485


def parse_key(k): r, t = k.rsplit("__t", 1); return r, int(t)


def load_roi(run, tick):
    p = RAW / run / "CAMERA_1" / f"{tick}_Intensity.jpeg"
    if not p.exists(): return None
    im = np.array(Image.open(p).convert("RGB"))
    return im[ROI_Y0:ROI_Y1].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prebake", default=str(DATA / "prebake/f1_prebake_iy_spd_streaming.npz"))
    ap.add_argument("--close-meta", default=str(DATA / "cache/close_lead_meta_v4.npz"))
    ap.add_argument("--aux-dir", default=str(DATA / "aux_targets_spd_fx640"))
    ap.add_argument("--out", default="outputs_multi/viz_resynth/gt_brake_audit")
    ap.add_argument("--n-eval", type=int, default=10000, help="sampling 수 (큰 분석)")
    ap.add_argument("--bad-show", type=int, default=12, help="이상 case 시각화 sample")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # prebake 의 sample (close-lead 만)
    z = np.load(args.prebake, allow_pickle=True)
    curr_keys = z["curr_keys"].tolist()
    target_all = z["target"]; mask_all = z["mask"]
    ego_5d = z["ego_5d"]
    sample_tags = z["sample_tags"].tolist()
    sample_keys = z["sample_keys"].tolist()

    cz = np.load(args.close_meta, allow_pickle=True)
    cl_map = dict(zip(cz["keys"].tolist(), cz["dists"]))

    rng = np.random.RandomState(args.seed)
    # close-lead sample (dist < 80m) — logged tag 만 (random rand_k 는 noise)
    cand = [i for i, ck in enumerate(curr_keys)
            if mask_all[i].any() and cl_map.get(ck, np.inf) < 80
            and sample_tags[i] == "logged"]
    print(f"[gt-audit] close-lead logged candidates: {len(cand)}")
    pick = rng.choice(cand, size=min(args.n_eval, len(cand)), replace=False).tolist()
    print(f"[gt-audit] evaluating {len(pick)} samples")

    # 각 sample 의 aux_targets 에서 v_long_rel (per-bin 의 lead 정보)
    # frame_id pattern: track__policy__run__tNNNNNNN  → aux file name
    aux_dir = Path(args.aux_dir)
    rec = []
    n_miss_aux = 0
    for idx in pick:
        ck = curr_keys[idx]
        # frame_id: speedway_track__0519_agent1_pertX_pitX__run__t...
        run, tk_str = ck.rsplit("__t", 1)
        frame_id = f"speedway_track__0519_agent1_pertX_pitX__{run}__t{int(tk_str):07d}"
        aux_f = aux_dir / f"{frame_id}.npz"
        if not aux_f.exists():
            n_miss_aux += 1; continue
        try:
            t = np.load(aux_f)["target"]
            # bin 0: obj, ch 3: v_long_rel, ch 6: dist
            # lead 정의: dist 양수 + 가장 가까운 bin 의 v_long_rel
            objs = t[:, 0] > 0.5
            if not objs.any(): continue
            valid_dists = t[objs, 6]
            valid_vrels = t[objs, 3]
            min_idx = np.argmin(valid_dists)
            lead_d = float(valid_dists[min_idx])
            lead_vrel = float(valid_vrels[min_idx])  # 음수 = ego 가 따라잡는 중
        except Exception:
            continue
        gt_ax_0 = float(target_all[idx, 0, 1])
        gt_ax_avg5 = float(target_all[idx, :5, 1].mean())
        rec.append(dict(
            ck=ck, dist=lead_d, vrel=lead_vrel, vx=float(ego_5d[idx, 0]),
            gt_ax_0=gt_ax_0, gt_ax_avg5=gt_ax_avg5,
        ))

    print(f"[gt-audit] aux missed: {n_miss_aux}, processed: {len(rec)}")
    if not rec:
        print("[gt-audit] no records — aux_dir 확인 필요"); return

    dist = np.array([r["dist"] for r in rec])
    vrel = np.array([r["vrel"] for r in rec])
    gt_a = np.array([r["gt_ax_0"] for r in rec])
    gt_5 = np.array([r["gt_ax_avg5"] for r in rec])
    vx   = np.array([r["vx"] for r in rec])

    # 통계
    print()
    print(f"=== GT first-step ax (close-lead, logged sample only, n={len(rec)}) ===")
    print(f"  GT ax mean={gt_a.mean():+.2f}  std={gt_a.std():.2f}")
    print(f"  brake (GT < 0): {(gt_a < 0).sum()} ({(gt_a < 0).mean()*100:.1f}%)")
    print(f"  strong brake (GT < -3): {(gt_a < -3).sum()} ({(gt_a < -3).mean()*100:.1f}%)")
    print(f"  cruise (|GT| <= 1): {((gt_a > -1) & (gt_a < 1)).sum()} ({((gt_a > -1) & (gt_a < 1)).mean()*100:.1f}%)")
    print(f"  positive accel (GT > 1): {(gt_a > 1).sum()} ({(gt_a > 1).mean()*100:.1f}%)")
    print()
    print(f"=== 이상 case ===")
    bad1 = (dist < 10) & (vrel < -5) & (gt_a > -1)
    print(f"  (dist<10 + vrel<-5 강한 approach) 인데 GT ax > -1 (no brake): {bad1.sum()} sample")
    bad2 = (dist < 5) & (gt_a > 0)
    print(f"  (dist<5 매우 가까움) 인데 GT ax > 0 (가속!): {bad2.sum()} sample")
    print()

    # bin 별 GT ax mean
    print(f"=== dist bin × vrel bin GT ax mean (heatmap data) ===")
    dist_bins = [0, 5, 10, 15, 20, 30, 50, 80]
    vrel_bins = [-15, -10, -5, -2, 0, 5, 15]
    H = np.full((len(dist_bins)-1, len(vrel_bins)-1), np.nan)
    Hn = np.zeros((len(dist_bins)-1, len(vrel_bins)-1), dtype=int)
    for i in range(len(dist_bins)-1):
        for j in range(len(vrel_bins)-1):
            mask = (dist >= dist_bins[i]) & (dist < dist_bins[i+1]) & \
                   (vrel >= vrel_bins[j]) & (vrel < vrel_bins[j+1])
            if mask.sum() >= 3:
                H[i, j] = gt_a[mask].mean()
                Hn[i, j] = mask.sum()
    # plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    # (a) heatmap GT ax mean
    ax = axes[0, 0]
    im = ax.imshow(H, cmap="RdYlGn_r", vmin=-10, vmax=10, aspect="auto",
                    extent=[vrel_bins[0], vrel_bins[-1], dist_bins[-1], dist_bins[0]])
    for i in range(len(dist_bins)-1):
        for j in range(len(vrel_bins)-1):
            if not np.isnan(H[i, j]):
                ax.text((vrel_bins[j]+vrel_bins[j+1])/2, (dist_bins[i]+dist_bins[i+1])/2,
                         f"{H[i,j]:+.1f}\n({Hn[i,j]})", ha="center", va="center",
                         fontsize=8, color="white" if abs(H[i,j])>3 else "black")
    ax.set_xlabel("v_long_rel (m/s, <0=approach)"); ax.set_ylabel("dist (m)")
    ax.set_title("GT first-step ax mean per (dist, vrel) bin\n빨강=brake 강함, 초록=accel. 이상적: dist↓+vrel↓ → 빨강")
    plt.colorbar(im, ax=ax, label="GT ax (m/s²)")
    # (b) scatter dist vs GT ax (colored by vrel)
    ax = axes[0, 1]
    sc = ax.scatter(dist, gt_a, c=vrel, cmap="coolwarm", s=8, alpha=0.5, vmin=-15, vmax=15)
    ax.axhline(0, color="black", alpha=0.5); ax.axhline(-3, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("lead dist (m)"); ax.set_ylabel("GT first-step ax (m/s²)")
    ax.set_title("dist 별 GT ax (color=vrel)")
    ax.grid(alpha=0.3); plt.colorbar(sc, ax=ax, label="v_long_rel")
    # (c) vrel vs GT ax
    ax = axes[1, 0]
    sc = ax.scatter(vrel, gt_a, c=dist, cmap="viridis", s=8, alpha=0.5)
    ax.axhline(0, color="black", alpha=0.5); ax.axvline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("v_long_rel (m/s, <0=approach)"); ax.set_ylabel("GT first-step ax")
    ax.set_title("approach rate 별 GT ax (color=dist)")
    ax.grid(alpha=0.3); plt.colorbar(sc, ax=ax, label="dist")
    # (d) GT ax histogram by dist 그룹
    ax = axes[1, 1]
    for lo, hi, color in [(0, 10, "#E74C3C"), (10, 30, "#F39C12"), (30, 80, "#5D6D7E")]:
        sel = (dist >= lo) & (dist < hi)
        if sel.sum() > 10:
            ax.hist(gt_a[sel], bins=40, alpha=0.5, label=f"dist {lo}-{hi}m (n={sel.sum()})", color=color)
    ax.axvline(0, color="blue", linestyle="--", alpha=0.5)
    ax.set_xlabel("GT first-step ax (m/s²)"); ax.set_ylabel("count")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("dist 그룹 별 GT ax 분포 — 가까울수록 brake 영역 확장?")
    fig.suptitle(f"GT brake audit (n={len(rec)}, close-lead logged samples)",
                  fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig(out_dir / "summary.png", dpi=110); plt.close()

    # 이상 case 시각화: dist<10 + vrel<-5 인데 brake 안 함
    bad_idx = np.where(bad1)[0][:args.bad_show]
    if len(bad_idx) > 0:
        bad_recs = [rec[i] for i in bad_idx]
        fig, axes = plt.subplots(len(bad_recs), 1, figsize=(12, len(bad_recs)*2.5))
        if len(bad_recs) == 1: axes = [axes]
        for ax, r in zip(axes, bad_recs):
            run, tk_str = r["ck"].rsplit("__t", 1)
            roi = load_roi(run, int(tk_str))
            if roi is not None: ax.imshow(roi)
            ax.set_title(f"{r['ck']}  dist={r['dist']:.1f}m  vrel={r['vrel']:+.1f}m/s  vx={r['vx']:.0f}  "
                          f"GT ax_0={r['gt_ax_0']:+.2f} ← brake 안 함?",
                          fontsize=9, color="red", fontweight="bold")
            ax.axis("off")
        fig.suptitle("이상 case: dist<10m + 강한 approach 인데 GT brake 안 함", fontsize=12, fontweight="bold")
        plt.tight_layout(); plt.savefig(out_dir / "bad_gt_cases.png", dpi=100); plt.close()
        print(f"  이상 case viz → bad_gt_cases.png")

    # html
    html = out_dir / "index.html"
    with open(html, "w") as f:
        f.write("<!doctype html><html><head><meta charset=utf-8>")
        f.write("<title>GT brake audit</title>")
        f.write("<style>body{font-family:sans-serif;background:#1c1c1c;color:#ddd;margin:12px}img{width:100%}.card{background:#2a2a2a;padding:8px;border-radius:6px;margin-bottom:14px}h2{color:#fff}</style>")
        f.write("</head><body>")
        f.write(f"<h2>GT (MPC 합성) brake 검수 — close-lead logged sample n={len(rec)}</h2>")
        f.write(f"<p>aux_dir: {aux_dir.name}<br>"
                f"GT brake ratio: {(gt_a < 0).mean()*100:.1f}% / strong brake (&lt;-3): {(gt_a < -3).mean()*100:.1f}%<br>"
                f"이상 case (dist&lt;10 + vrel&lt;-5 + GT&gt;-1): {bad1.sum()}</p>")
        f.write("<div class=card><img src=summary.png></div>")
        if len(bad_idx) > 0:
            f.write("<h2>이상 case: 강한 approach 인데 GT 가 brake 결정 안 함</h2>")
            f.write("<div class=card><img src=bad_gt_cases.png></div>")
        f.write("</body></html>")
    print(f"  → {html}")


if __name__ == "__main__":
    main()
