"""IyModel 의 img_fusion (fi_mem) per-bin L2 norm 시각화.

obj_gate 학습 안 된 상태에서도 transformer 가 객체 없는 bin 의 image token
을 자연스럽게 작은 norm 으로 만들거나 attention 약하게 주는지 확인.

각 sample:
  1. backbone forward → tok
  2. IyModel.img_fusion forward hook → fi_mem (B, 40, D)
  3. per-bin L2 norm → 40-bar plot
  4. close-lead 있는 sample 과 없는 sample 비교

사용:
  python scripts/viz_iy_fimem_norm.py \
      --prebake .../f1_prebake_iy_spd_streaming.npz \
      --i1-cache .../i1_tok_cache_spd_v2.npz \
      --ckpt outputs_multi/iy_spd_streaming_w20/best_loss.pt \
      --close-meta .../close_lead_meta_v3.npz \
      --raw-base .../raw_cache/speedway_track/0519_agent1_pertX_pitX \
      --n-with 8 --n-without 8 \
      --out outputs_multi/viz_resynth/fimem_spd
"""
from __future__ import annotations
import os, sys, argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import timm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.model.image_models.iy_model import IyModel

ROI_Y0, ROI_Y1 = 235, 485
ROI_H, IMG_W = ROI_Y1 - ROI_Y0, 1280
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
FAR_THR = 80.0   # 객체 없음 기준 (close_lead_meta 의 dist > 이값 또는 missing)


def parse_key(k): r, t = k.rsplit("__t", 1); return r, int(t)


def load_roi(raw_base: Path, run, tick):
    p = raw_base / run / "CAMERA_1" / f"{tick}_Intensity.jpeg"
    if not p.exists(): return None
    im = np.array(Image.open(p).convert("RGB"))
    return im[ROI_Y0:ROI_Y1].copy()


def normalize(roi):
    x = roi.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prebake", required=True)
    ap.add_argument("--i1-cache", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--close-meta", required=True)
    ap.add_argument("--raw-base", required=True, type=Path)
    ap.add_argument("--seg-ckpt", default="data/ckpts/seg_pretrain_v2s_full_best_backbone.pt")
    ap.add_argument("--fx", type=float, default=640.0)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--num-layers", type=int, default=6)
    ap.add_argument("--n-with", type=int, default=8, help="close-lead 있는 sample 개수")
    ap.add_argument("--n-without", type=int, default=8, help="close-lead 없는 (멀거나 없음) sample")
    ap.add_argument("--close-thr", type=float, default=30.0)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # backbone
    cm = np.load(args.i1_cache, allow_pickle=True); cmeta = json.loads(str(cm["meta"]))
    backbone_name = cmeta["backbone"]; C_b = int(cmeta["C_b"])
    backbone = timm.create_model(backbone_name, pretrained=False, features_only=True, out_indices=[-1])
    if Path(args.seg_ckpt).exists():
        sd = torch.load(args.seg_ckpt, map_location="cpu", weights_only=False)
        own = backbone.state_dict(); n = 0
        for k, v in sd.items():
            k2 = k.replace("backbone.", "")
            if k2 in own and own[k2].shape == v.shape: own[k2] = v; n += 1
        backbone.load_state_dict(own); print(f"  seg backbone preload: {n} keys")
    backbone = backbone.to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)

    # obj_gate 가 학습된 ckpt 인지 체크 → True 로 build
    _sd_peek = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(_sd_peek, dict) and "model" in _sd_peek: _sd_peek = _sd_peek["model"]
    has_obj = any("obj_head" in k for k in _sd_peek.keys())
    print(f"  ckpt has obj_head: {has_obj}")
    model = IyModel(C_b=C_b, d_model=args.d_model, nhead=args.nhead,
                     num_layers=args.num_layers, fx=args.fx,
                     use_obj_gate=has_obj).to(device).eval()
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
    miss, unex = model.load_state_dict(sd, strict=False)
    print(f"  ckpt: missing={len(miss)} unexpected={len(unex)}")

    # img_fusion forward hook → fi_mem (gate 전)
    fi_capture = {}
    def hook(_m, _i, o): fi_capture["fi"] = o.detach()
    h = model.img_fusion.register_forward_hook(hook)
    # obj_head 가 있으면 gate prob 도 capture
    obj_capture = {}
    if has_obj:
        def hook_obj(_m, _i, o): obj_capture["logit"] = o.detach().squeeze(-1)  # (B, 40)
        h2 = model.obj_head.register_forward_hook(hook_obj)

    # samples
    z = np.load(args.prebake, allow_pickle=True)
    curr_keys = z["curr_keys"].tolist(); prev_keys = z["prev_keys"].tolist()
    d_yaws = z["d_yaws"]; ego_5d = z["ego_5d"]; traj_all = z["traj"]; mask_all = z["mask"]
    cz = np.load(args.close_meta, allow_pickle=True)
    cl_map = dict(zip(cz["keys"].tolist(), cz["dists"]))

    rng = np.random.RandomState(args.seed)
    cand_with = [i for i, ck in enumerate(curr_keys)
                 if ck in cl_map and cl_map[ck] < args.close_thr and mask_all[i].any()]
    cand_without = [i for i, ck in enumerate(curr_keys)
                    if (ck not in cl_map or cl_map[ck] > FAR_THR) and mask_all[i].any()]
    print(f"  candidates: with-close={len(cand_with)}  without={len(cand_without)}")
    pick_w = rng.choice(cand_with,    size=min(args.n_with,    len(cand_with)),    replace=False).tolist() if cand_with else []
    pick_o = rng.choice(cand_without, size=min(args.n_without, len(cand_without)), replace=False).tolist() if cand_without else []
    all_picks = [("WITH", i) for i in pick_w] + [("WITHOUT", i) for i in pick_o]

    def run_one(idx):
        ck = curr_keys[idx]; pk = prev_keys[idx]
        run_c, tk_c = parse_key(ck); run_p, tk_p = parse_key(pk)
        roi_c = load_roi(args.raw_base, run_c, tk_c)
        roi_p = load_roi(args.raw_base, run_p, tk_p)
        if roi_c is None or roi_p is None: return None
        ic = normalize(roi_c).unsqueeze(0).to(device)
        ip = normalize(roi_p).unsqueeze(0).to(device)
        with torch.no_grad():
            fc = backbone(ic)[-1]; fp = backbone(ip)[-1]
            tok_c = F.adaptive_avg_pool2d(fc, (1, 40)).squeeze(2).transpose(1, 2)
            tok_p = F.adaptive_avg_pool2d(fp, (1, 40)).squeeze(2).transpose(1, 2)
            e5 = torch.from_numpy(ego_5d[idx]).unsqueeze(0).to(device)
            t40 = torch.from_numpy(traj_all[idx]).unsqueeze(0).to(device)
            d_y = torch.tensor([float(d_yaws[idx])], device=device)
            fx_t = torch.tensor([args.fx], device=device)
            out = model({"tok_curr": tok_c, "tok_prev_raw": tok_p, "d_yaw": d_y,
                         "ego_5d": e5, "traj_40": t40, "fx": fx_t})
            pred = out[0] if isinstance(out, tuple) else out
        fi = fi_capture["fi"]                                         # (1, 40, D)  gate 전
        norm = fi[0].norm(dim=-1).cpu().numpy()                       # (40,)
        obj_prob = None; norm_after = None
        if has_obj:
            logit = obj_capture["logit"][0]                            # (40,)
            obj_prob = torch.sigmoid(logit).cpu().numpy()             # (40,) 0~1
            # gate 적용 후 norm
            fi_gated = fi[0] * torch.sigmoid(logit).unsqueeze(-1)     # (40, D)
            norm_after = fi_gated.norm(dim=-1).cpu().numpy()
        return dict(roi=roi_c, ck=ck, norm=norm, norm_after=norm_after, obj_prob=obj_prob,
                    pred=pred[0].cpu().numpy(),
                    dist=cl_map.get(ck, float("inf")), vx=float(ego_5d[idx, 0]))

    recs = []
    for ti, (tag, idx) in enumerate(all_picks):
        r = run_one(idx)
        if r is None: continue
        r["tag"] = tag; recs.append(r)
        msg = f"  [{ti+1}/{len(all_picks)}] {tag} {r['ck']} dist={r['dist']:.1f}  norm[{r['norm'].min():.2f}, {r['norm'].max():.2f}]"
        if r.get("obj_prob") is not None:
            msg += f"  obj_prob[{r['obj_prob'].min():.2f}, {r['obj_prob'].max():.2f}]  norm_after[{r['norm_after'].min():.2f}, {r['norm_after'].max():.2f}]"
        print(msg)
    h.remove()
    if has_obj: h2.remove()

    # 통계
    with_r = [r for r in recs if r["tag"] == "WITH"]
    wout_r = [r for r in recs if r["tag"] == "WITHOUT"]
    print()
    if with_r:
        mw = np.mean([r["norm"].mean() for r in with_r])
        sw = np.mean([r["norm"].std() for r in with_r])
        print(f"  WITH close-lead: mean(norm)={mw:.3f} ± {sw:.3f}")
        if with_r[0].get("obj_prob") is not None:
            mwa = np.mean([r["norm_after"].mean() for r in with_r])
            mwp = np.mean([r["obj_prob"].mean() for r in with_r])
            print(f"    after-gate norm={mwa:.3f}  mean obj_prob={mwp:.3f}")
    if wout_r:
        mo = np.mean([r["norm"].mean() for r in wout_r])
        so = np.mean([r["norm"].std() for r in wout_r])
        print(f"  WITHOUT lead:    mean(norm)={mo:.3f} ± {so:.3f}")
        if wout_r[0].get("obj_prob") is not None:
            moa = np.mean([r["norm_after"].mean() for r in wout_r])
            mop = np.mean([r["obj_prob"].mean() for r in wout_r])
            print(f"    after-gate norm={moa:.3f}  mean obj_prob={mop:.3f}")

    # per-sample plot
    for ri, r in enumerate(recs):
        fig, axes = plt.subplots(2, 1, figsize=(16, 5), gridspec_kw=dict(height_ratios=[1, 0.6]))
        axes[0].imshow(r["roi"])
        # 40 bin 을 image 너비에 매핑 (image 가로 = 40 bin × 32 px/bin)
        bin_w = IMG_W / 40
        nmax = max(r["norm"].max(), 1e-3)
        for b in range(40):
            v = r["norm"][b] / nmax
            axes[0].axvspan(b*bin_w, (b+1)*bin_w, ymin=0, ymax=v*0.3, color="red", alpha=0.4)
        axes[0].set_title(f"[{r['tag']}] {r['ck']}  dist={r['dist']:.1f}m  vx={r['vx']:.1f}m/s  "
                          f"norm range=[{r['norm'].min():.2f}, {r['norm'].max():.2f}]", fontsize=11)
        axes[0].axis("off")
        axes[1].bar(np.arange(40), r["norm"], color=("crimson" if r["tag"] == "WITH" else "steelblue"))
        axes[1].set_xlabel("bin (0=좌, 39=우)"); axes[1].set_ylabel("fi_mem L2 norm")
        axes[1].set_title("per-bin |fi_mem|  — obj_gate 학습 안 된 상태에서 모델이 자발적으로 영향 변화시키는지", fontsize=10)
        axes[1].grid(alpha=0.3); axes[1].set_xlim(-0.5, 39.5)
        fname = f"{ri:02d}_{r['tag']}_{r['ck'].replace('__','_')}.png"
        plt.tight_layout(); plt.savefig(args.out / fname, dpi=110); plt.close()

    # summary plot: aggregated per-bin norm
    fig, ax = plt.subplots(figsize=(14, 4))
    if with_r:
        avg_w = np.mean([r["norm"] for r in with_r], axis=0)
        ax.plot(avg_w, "r-", lw=2, label=f"WITH close-lead (n={len(with_r)})")
    if wout_r:
        avg_o = np.mean([r["norm"] for r in wout_r], axis=0)
        ax.plot(avg_o, "b-", lw=2, label=f"WITHOUT lead (n={len(wout_r)})")
    ax.set_xlabel("bin"); ax.set_ylabel("mean |fi_mem|")
    ax.set_title("Average fi_mem L2 norm per bin — WITH vs WITHOUT close-lead")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(args.out / "summary.png", dpi=110); plt.close()
    print(f"  summary → {args.out / 'summary.png'}")

    # html
    html = args.out / "index.html"
    with open(html, "w") as f:
        f.write("<!doctype html><html><head><meta charset=utf-8>")
        f.write("<title>IyModel fi_mem L2 norm (obj_gate=False)</title>")
        f.write("<style>body{font-family:sans-serif;background:#1c1c1c;color:#ddd;margin:12px}.card{background:#2a2a2a;padding:8px;border-radius:6px;margin-bottom:14px}.card img{width:100%;display:block}h2{color:#fff}.summary{border:2px solid #f80;padding:12px}</style>")
        f.write("</head><body>")
        f.write("<h2>IyModel fi_mem L2 norm — obj_gate <b>학습 안 된 상태</b> 에서 image token 영향 분석</h2>")
        f.write("<p>현재 streaming 학습은 <code>use_obj_gate=False</code> 라서 명시적 gate 없음. transformer 가 자체적으로 image token 의 영향을 객체 유무에 따라 다르게 만드는지 fi_mem L2 norm 으로 확인.</p>")
        f.write("<div class=card summary><img src='summary.png'></div>")
        for ri, r in enumerate(recs):
            tag = r["tag"]
            f.write(f"<div class=card><div style='font-size:12px;color:#aaa'>[{tag}] {r['ck']} | dist={r['dist']:.1f}m | norm [{r['norm'].min():.2f}, {r['norm'].max():.2f}]</div>")
            f.write(f"<img src='{ri:02d}_{tag}_{r['ck'].replace('__','_')}.png' loading=lazy></div>")
        f.write("</body></html>")
    print(f"  index → {html}")


if __name__ == "__main__":
    main()
