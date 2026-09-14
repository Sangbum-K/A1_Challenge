"""매 epoch best_loss.pt 갱신 감지 → 분류된 sample 16개 inference + 시각화.

각 epoch viz:
  - no-lead (dist=inf) 4 sample
  - close approach (dist<30, GT ax < -1.5)
  - close cruise   (dist<30, |GT ax| <= 1.5)
  - close recede   (dist<30, GT ax > 1.5)

각 category 4 sample × 4 = 16 sample. random seed 고정으로 epoch 마다 같은 sample.

학습은 중단하지 않음 — 별도 process (GPU 일시 점유, 학습 batch 1-2 일시 느림).

출력: outputs_multi/viz_resynth/eval_per_ep/{ep_NNN.png, index.html}
"""
from __future__ import annotations
import os, sys, time, json, glob, argparse, re
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import timm

ROOT = Path("/home/ailab-12/git/scc_controller_imitator")
sys.path.insert(0, str(ROOT))
from src.model.image_models.iy_model import IyModel

DATA = Path("/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data")
RAW = Path("/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/raw_cache/speedway_track/0519_agent1_pertX_pitX")
SEG = ROOT / "data/ckpts/seg_pretrain_v2s_full_best_backbone.pt"
ROI_Y0, ROI_Y1 = 235, 485
ROI_H, IMG_W = ROI_Y1 - ROI_Y0, 1280
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_key(k): r, t = k.rsplit("__t", 1); return r, int(t)


def load_roi(run, tick):
    p = RAW / run / "CAMERA_1" / f"{tick}_Intensity.jpeg"
    if not p.exists(): return None
    im = np.array(Image.open(p).convert("RGB"))
    return im[ROI_Y0:ROI_Y1].copy()


def normalize(roi):
    x = roi.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prebake", default=str(DATA / "prebake/f1_prebake_iy_spd_streaming.npz"))
    ap.add_argument("--i1-cache", default=str(DATA / "cache/i1_tok_cache_spd_v2.npz"))
    ap.add_argument("--ckpt-dir", default="outputs_multi/iy_spd_sangam_ciaug_v0237_noreload")
    ap.add_argument("--close-meta", default=str(DATA / "cache/close_lead_meta_v4.npz"))
    ap.add_argument("--out", default="outputs_multi/viz_resynth/eval_per_ep")
    ap.add_argument("--fx", type=float, default=640.0)
    ap.add_argument("--check-interval", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # backbone (한 번 init)
    cm = np.load(args.i1_cache, allow_pickle=True)
    cmeta = json.loads(str(cm["meta"]))
    C_b = int(cmeta["C_b"])
    backbone = timm.create_model(cmeta["backbone"], pretrained=False, features_only=True, out_indices=[-1])
    sd = torch.load(SEG, map_location="cpu", weights_only=False)
    own = backbone.state_dict()
    fixed = {k.replace("backbone.", ""): v for k, v in sd.items()
             if k.replace("backbone.", "") in own and own[k.replace("backbone.", "")].shape == v.shape}
    backbone.load_state_dict(fixed, strict=False)
    backbone = backbone.to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)
    print(f"[viz-loop] backbone {cmeta['backbone']} preloaded ({len(fixed)} keys)")

    # prebake + close_lead
    z = np.load(args.prebake, allow_pickle=True)
    curr_keys = z["curr_keys"].tolist(); prev_keys = z["prev_keys"].tolist()
    d_yaws = z["d_yaws"]; ego_5d = z["ego_5d"]
    traj_all = z["traj"]; target_all = z["target"]; mask_all = z["mask"]
    cz = np.load(args.close_meta, allow_pickle=True)
    cl_map = dict(zip(cz["keys"].tolist(), cz["dists"]))

    # 분류 sample 선별 (한 번)
    rng = np.random.RandomState(args.seed)
    cand_nolead, cand_app, cand_crz, cand_rcd = [], [], [], []
    for i in range(len(curr_keys)):
        if not mask_all[i].any(): continue
        ck = curr_keys[i]
        d = cl_map.get(ck, np.inf)
        ax_mean = float(target_all[i, :10, 1].mean())
        if not np.isfinite(d):
            cand_nolead.append(i)
        elif d < 30:
            if   ax_mean < -1.5: cand_app.append(i)
            elif ax_mean >  1.5: cand_rcd.append(i)
            else:                cand_crz.append(i)
    print(f"[viz-loop] candidates: nolead={len(cand_nolead)} app={len(cand_app)} crz={len(cand_crz)} rcd={len(cand_rcd)}")
    N_PER = 4
    pick = {
        "no-lead":  rng.choice(cand_nolead, size=min(N_PER, len(cand_nolead)), replace=False).tolist() if cand_nolead else [],
        "approach": rng.choice(cand_app,    size=min(N_PER, len(cand_app)),    replace=False).tolist() if cand_app    else [],
        "cruise":   rng.choice(cand_crz,    size=min(N_PER, len(cand_crz)),    replace=False).tolist() if cand_crz    else [],
        "recede":   rng.choice(cand_rcd,    size=min(N_PER, len(cand_rcd)),    replace=False).tolist() if cand_rcd    else [],
    }
    print(f"[viz-loop] picked {sum(len(v) for v in pick.values())} samples ({list(pick.keys())})")

    # ROI 미리 load (학습 중 disk I/O 줄임)
    roi_cache = {}
    for cat, idxs in pick.items():
        for idx in idxs:
            for k in (curr_keys[idx], prev_keys[idx]):
                if k in roi_cache: continue
                run, tk = parse_key(k)
                roi = load_roi(run, tk)
                if roi is not None: roi_cache[k] = roi
    print(f"[viz-loop] cached {len(roi_cache)} ROI images")

    ckpt_path = ROOT / args.ckpt_dir / "best_loss.pt"
    last_mtime = 0
    iter_n = 0

    while True:
        # poll best ckpt
        try:
            mtime = os.path.getmtime(ckpt_path)
        except FileNotFoundError:
            print(f"[viz-loop] ckpt not yet: {ckpt_path}, wait")
            time.sleep(args.check_interval); continue
        if mtime == last_mtime:
            time.sleep(args.check_interval); continue
        last_mtime = mtime; iter_n += 1
        ts = time.strftime("%H:%M:%S")
        print(f"[viz-loop] {ts}  ckpt mtime changed → run viz #{iter_n}")

        # IyModel load (매번 새 ckpt)
        _sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(_sd, dict) and "model" in _sd: _sd = _sd["model"]
        has_obj = any("obj_head" in k for k in _sd.keys())
        model = IyModel(C_b=C_b, d_model=128, nhead=8, num_layers=6, fx=args.fx,
                          use_obj_gate=has_obj).to(device).eval()
        miss, unex = model.load_state_dict(_sd, strict=False)
        # epoch 추정 (train.log 에서 lookup)
        ep_now = "?"
        log = ROOT / args.ckpt_dir / "train.log"
        if log.exists():
            best_ep = None; best_val = float("inf")
            for line in open(log):
                m = re.match(r"\[ep (\d+)\] train=[\d.]+ val=([\d.]+)", line)
                if m and float(m.group(2)) < best_val:
                    best_val = float(m.group(2)); best_ep = int(m.group(1))
            ep_now = str(best_ep) if best_ep else "?"
        # inference 각 sample
        rec = []
        with torch.no_grad():
            for cat, idxs in pick.items():
                for idx in idxs:
                    ck = curr_keys[idx]; pk = prev_keys[idx]
                    if ck not in roi_cache or pk not in roi_cache: continue
                    ic = normalize(roi_cache[ck]).unsqueeze(0).to(device)
                    ip = normalize(roi_cache[pk]).unsqueeze(0).to(device)
                    fc = F.adaptive_avg_pool2d(backbone(ic)[-1], (1, 40)).squeeze(2).transpose(1, 2)
                    fp = F.adaptive_avg_pool2d(backbone(ip)[-1], (1, 40)).squeeze(2).transpose(1, 2)
                    e5 = torch.from_numpy(ego_5d[idx]).unsqueeze(0).to(device)
                    t40 = torch.from_numpy(traj_all[idx]).unsqueeze(0).to(device)
                    d_y = torch.tensor([float(d_yaws[idx])], device=device)
                    fx_t = torch.tensor([args.fx], device=device)
                    out = model({"tok_curr": fc, "tok_prev_raw": fp, "d_yaw": d_y,
                                  "ego_5d": e5, "traj_40": t40, "fx": fx_t})
                    pred = out[0] if isinstance(out, tuple) else out
                    pred_np = pred[0].cpu().numpy()
                    rec.append(dict(cat=cat, ck=ck, dist=cl_map.get(ck, np.inf),
                                      vx=float(ego_5d[idx, 0]),
                                      pred=pred_np, gt=target_all[idx], mask=mask_all[idx],
                                      roi=roi_cache[ck]))
        # plot grid: 4 row (cat) × 4 col (sample)
        fig, axes = plt.subplots(8, 4, figsize=(20, 18),
                                  gridspec_kw=dict(height_ratios=[1.2, 0.7]*4))
        cat_order = ["no-lead", "approach", "cruise", "recede"]
        cat_colors = {"no-lead": "#5D6D7E", "approach": "#E74C3C",
                       "cruise": "#2980B9", "recede": "#27AE60"}
        for ri, cat in enumerate(cat_order):
            cat_recs = [r for r in rec if r["cat"] == cat]
            for ci in range(4):
                ax_img = axes[ri*2, ci]; ax_curve = axes[ri*2+1, ci]
                if ci >= len(cat_recs):
                    ax_img.axis("off"); ax_curve.axis("off"); continue
                r = cat_recs[ci]
                ax_img.imshow(r["roi"])
                ax_img.set_title(f"[{cat}] dist={r['dist']:.0f}m vx={r['vx']:.0f}",
                                  color=cat_colors[cat], fontsize=10, fontweight="bold")
                ax_img.axis("off")
                tt = np.arange(40) * 0.1
                ax_curve.plot(tt, r["gt"][:, 1], "k--", lw=1.0, label="GT ax")
                ax_curve.plot(tt, r["pred"][:, 1], "-", color=cat_colors[cat], lw=1.6, label="pred ax")
                ax_curve.set_xlim(0, 4); ax_curve.set_ylim(-22, 22); ax_curve.grid(alpha=0.3)
                ax_curve.set_xlabel("t(s)", fontsize=8); ax_curve.set_ylabel("ax", fontsize=8)
                if ci == 0: ax_curve.legend(fontsize=7, loc="upper right")
                ax_curve.tick_params(labelsize=7)
        fig.suptitle(f"IyModel inference per-epoch — ep{ep_now}  ckpt {time.strftime('%H:%M:%S', time.localtime(mtime))}",
                      fontsize=13, fontweight="bold")
        plt.tight_layout()
        png = out_dir / f"ep_{int(ep_now) if ep_now != '?' else iter_n:03d}.png"
        plt.savefig(png, dpi=100); plt.close()
        print(f"[viz-loop] saved → {png}")

        # rebuild index.html (newest first)
        pngs = sorted(out_dir.glob("ep_*.png"), reverse=True)
        idx = out_dir / "index.html"
        with open(idx, "w") as f:
            f.write("<!doctype html><html><head><meta charset=utf-8>")
            f.write("<title>IyModel per-epoch eval (situation grouped)</title>")
            f.write("<meta http-equiv='refresh' content='60'>")
            f.write("<style>body{font-family:sans-serif;background:#1c1c1c;color:#ddd;margin:12px}.card{background:#2a2a2a;padding:6px;border-radius:6px;margin-bottom:14px}.card img{width:100%;display:block}h2{color:#fff}</style>")
            f.write("</head><body>")
            f.write("<h2>IyModel inference per-epoch — 상황별 sample (no-lead / approach / cruise / recede)</h2>")
            f.write(f"<p>auto-refresh 60s. 각 row 4 sample × 4 category. ckpt update 시 새 row 추가.</p>")
            for p in pngs:
                f.write(f"<div class=card><div style='color:#888'>{p.name}</div><img src='{p.name}'></div>")
            f.write("</body></html>")

        # 학습 batch 와 GPU 경쟁 최소화 — small wait
        time.sleep(args.check_interval)


if __name__ == "__main__":
    main()
