"""fi_mem 캐시의 정성 검증 — "전방 차량을 보고 있나?".

각 viz frame 에 대해 panel 생성:
  row 0: image + a_2 GT obj bin overlay (있는 bin = 빨강 박스, label=dist)
  row 1: fi_mem 의 per-bin norm bar chart  (norm 큰 bin = 강하게 활성)
  row 2: fi_mem PCA 1st component per-bin (색으로) — 비슷한 의미 토큰 묶임
  row 3: fi_mem cosine similarity matrix (40×40) — 같은 객체 토큰끼리 비슷한지

GT 와 fi_mem norm/PCA 의 alignment → 인코더가 차량 위치를 인식

사용:
  python scripts/viz_fi_mem_cache.py \
    --cache data/fi_mem_cache_v1.npz \
    --n-frames 8 --out /tmp/fi_mem_viz.png
"""
from __future__ import annotations
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEFAULT_CACHE = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/fi_mem_cache_v1.npz"
DEFAULT_A2    = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/aux_targets_fx320"
DEFAULT_META  = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/multiagnet_compact_meta.json"
DEFAULT_RAW   = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/jiwon_e2e/e2e_racing/data/raw"

ROI_Y0, ROI_Y1 = 235, 485
IMG_W = 1280
N_BINS = 40


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--a2-dir", default=DEFAULT_A2)
    ap.add_argument("--meta", default=DEFAULT_META)
    ap.add_argument("--raw-base", default=DEFAULT_RAW)
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--out", default="/tmp/fi_mem_viz.png")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def load_image_roi(track, policy, run, tick):
    p = Path(f"{DEFAULT_RAW}/{track}/{policy}/{run}/CAMERA_1/{tick}_Intensity.jpeg")
    if not p.exists(): return None
    img = np.array(Image.open(p).convert("RGB"))
    return img[ROI_Y0:ROI_Y1]


def main():
    args = get_args()
    rng = np.random.RandomState(args.seed)

    print(f"loading cache {args.cache}")
    z = np.load(args.cache, allow_pickle=True)
    keys = z["keys"].tolist()
    fi_mem_all = z["fi_mem"]    # (N, 40, D) float16
    meta_info = json.loads(str(z["meta"]))
    print(f"  fi_mem: {fi_mem_all.shape}  D={fi_mem_all.shape[-1]}")
    key2idx = {k: i for i, k in enumerate(keys)}

    # meta lookup
    meta = json.load(open(args.meta))
    m_by_key = {f"{m['run_id']}__t{int(m['tick']):07d}": m for m in meta}

    # pick frames: prefer those with a_2 GT having obj (가까운 차량 있음)
    a2_p = Path(args.a2_dir)
    candidates = []
    for k in keys:
        m = m_by_key.get(k)
        if m is None: continue
        ap = a2_p / f"{m['frame_id']}.npz"
        if not ap.exists(): continue
        try:
            t = np.load(ap)["target"]   # (40, 7)
            n_obj = int((t[:, 0] > 0.5).sum())
            if n_obj > 0: candidates.append((k, m, n_obj))
        except Exception: pass
    print(f"  candidates with a_2 obj: {len(candidates)}")
    # sample N frames balanced across n_obj
    rng.shuffle(candidates)
    selected = candidates[:args.n_frames]
    print(f"  selected: {len(selected)}")

    # ----- composite panel -----
    N = len(selected); H, W = 250, 1280
    fig, axes = plt.subplots(N * 3, 1, figsize=(16, 3.5 * N), squeeze=False)
    axes = axes.reshape(N, 3)

    for i, (key, m, n_obj) in enumerate(selected):
        img_roi = load_image_roi(m["track"], m["policy"], m["run_id"], int(m["tick"]))
        if img_roi is None: continue
        t_a2 = np.load(a2_p / f"{m['frame_id']}.npz")["target"]   # (40, 7)
        fi = fi_mem_all[key2idx[key]].astype(np.float32)            # (40, D)
        norms = np.linalg.norm(fi, axis=1)                          # (40,)
        # PCA 1st component
        fi_c = fi - fi.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(fi_c, full_matrices=False)
        pc1 = U[:, 0] * S[0]                                        # (40,)

        # row 0: image + a_2 GT overlay
        ax = axes[i, 0]
        ax.imshow(img_roi)
        bin_w = W / N_BINS
        for bi in range(N_BINS):
            if t_a2[bi, 0] < 0.5: continue
            d = t_a2[bi, 6]; vrel = t_a2[bi, 3]
            x0 = bi * bin_w
            ax.add_patch(Rectangle((x0, H-28), bin_w, 28, fc="red", alpha=0.55, ec="white", lw=0.5))
            ax.text(x0 + bin_w/2, H - 14, f"{d:.0f}", ha="center", va="center",
                     fontsize=7, color="white", fontweight="bold")
        ax.set_title(f"frame={m['frame_id']}  a_2 GT obj bins: {n_obj}  (red=vehicle pos)", fontsize=10)
        ax.axis("off")

        # row 1: per-bin norm bar
        ax = axes[i, 1]
        bin_centers = np.arange(N_BINS) + 0.5
        # bar 색: a_2 GT obj 있는 bin 강조
        obj_bins = (t_a2[:, 0] > 0.5)
        colors = ["red" if o else "steelblue" for o in obj_bins]
        ax.bar(bin_centers, norms, width=0.9, color=colors, edgecolor="k", lw=0.3)
        ax.set_xlim(0, N_BINS); ax.set_xlabel("bin (0=left, 39=right)")
        ax.set_ylabel("||fi_mem[i]||")
        ax.set_title("fi_mem per-bin norm  (red=GT obj bin → norm 가 비슷한 위치에 강해야 함)", fontsize=9)
        ax.grid(True, alpha=0.3)

        # row 2: PC1 + obj overlay
        ax = axes[i, 2]
        ax.bar(bin_centers, pc1, width=0.9, color=colors, edgecolor="k", lw=0.3)
        for bi in range(N_BINS):
            if obj_bins[bi]:
                ax.axvspan(bi, bi+1, color="red", alpha=0.10)
        ax.set_xlim(0, N_BINS); ax.set_xlabel("bin")
        ax.set_ylabel("PC1(fi_mem)")
        ax.set_title("fi_mem PCA 1st component per bin  (red shade=GT obj)", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=85, bbox_inches="tight")
    print(f"  saved → {args.out}")


if __name__ == "__main__":
    main()
