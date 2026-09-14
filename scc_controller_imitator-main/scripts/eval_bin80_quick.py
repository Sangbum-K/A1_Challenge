#!/usr/bin/env python
"""bin80 빠른 검증 — detected present 의 far(40-60m) vL_rel error std 1지표.
40bin(tspan) s10n10 far std(=3.0)와 비교해 bin80(stride16) 추가 가치 판정용.
val split seed0 5% (학습과 동일 held-out). in-lane center = 80bin[32:52]."""
import sys, glob, re, json, argparse
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model.image_models.a2_model import A2ModelMultiFrameHR

AUX_PAT = re.compile(r".+?__([^_]+(?:_[^_]+)*?)__t(\d+)\.npz$")
SCALE_V, AUX_VLREL, AUX_DIST = 20.0, 3, 6
L0, L1 = 32, 52   # 80bin in-lane center


def load_one(i1p, ad, offsets):
    z = np.load(i1p, allow_pickle=True)
    tok, tok16, keys = z["tokens"], z["tokens_s16"], z["keys"].tolist()
    k2i = {k: i for i, k in enumerate(keys)}
    samples = []
    for f in sorted(glob.glob(str(Path(ad) / "*.npz"))):
        m = AUX_PAT.match(Path(f).name)
        if not m:
            continue
        run, tick = m.group(1), int(m.group(2))
        if f"{run}__t{tick:07d}" not in k2i:
            continue
        ci = k2i[f"{run}__t{tick:07d}"]
        pidx, last = [], ci
        for off in offsets:
            pi = k2i.get(f"{run}__t{tick - off:07d}", last); pidx.append(pi); last = pi
        samples.append((ci, f, pidx))
    return tok, tok16, samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i1-cache", action="append", required=True)
    ap.add_argument("--aux-dir", action="append", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--offsets", required=True)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    offs = [int(x) for x in args.offsets.split(",") if x.strip()]
    nf = 1 + len(offs)
    dev = "cuda"

    model = A2ModelMultiFrameHR(d_token=256, d_out=128, n_frames=nf).to(dev).eval()
    sd = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(sd["model"] if "model" in sd else sd)

    # concat + val split (random_split 동일 = 뒤쪽 perm)
    caches = [load_one(i1p, ad, offs) for i1p, ad in zip(args.i1_cache, args.aux_dir)]
    total = sum(len(s) for _, _, s in caches)
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(total, generator=g).tolist()
    n_val = int(total * args.val_ratio)
    val_set = set(perm[total - n_val:])

    err_far, err_all = [], []   # (pred-gt) vL_rel, far(40-60) / all detected present
    gidx = 0
    with torch.no_grad():
        for (tok, tok16, samples) in caches:
            for j in range(0, len(samples), 256):
                chunk = [(k, samples[k]) for k in range(j, min(j + 256, len(samples))) if (gidx + k) in val_set]
                if not chunk:
                    continue
                idx = np.array([c[1][0] for c in chunk])
                t32 = [torch.from_numpy(tok[idx].astype(np.float32)).to(dev)]
                t16 = [torch.from_numpy(tok16[idx].astype(np.float32)).to(dev)]
                for fi in range(len(offs)):
                    pid = np.array([c[1][2][fi] for c in chunk])
                    t32.append(torch.from_numpy(tok[pid].astype(np.float32)).to(dev))
                    t16.append(torch.from_numpy(tok16[pid].astype(np.float32)).to(dev))
                ego = torch.zeros(len(chunk), 2, device=dev)
                logits = model.forward_tokens(t32, t16, ego)          # (B,80,7)
                obj = torch.sigmoid(logits[:, :, 0]).cpu().numpy()
                pv = (logits[:, :, 1 + (AUX_VLREL - 1)] * SCALE_V).cpu().numpy()  # reg ch (vL_rel) — idx 1+2=3? see below
                gts = np.stack([np.load(c[1][1])["target"] for c in chunk], 0)    # (B,80,7)
                amax = obj[:, L0:L1].max(1); bn = L0 + obj[:, L0:L1].argmax(1)
                rows = np.arange(len(chunk))
                gobj_c = gts[:, L0:L1, 0].max(1)
                det = (gobj_c > 0.5) & (amax > 0.5)
                gd = gts[rows, bn, AUX_DIST]; gv = gts[rows, bn, AUX_VLREL]
                pvb = pv[rows, bn]
                e = pvb - gv
                m_all = det
                m_far = det & (gd >= 40) & (gd < 60)
                err_all.extend(e[m_all].tolist()); err_far.extend(e[m_far].tolist())
            gidx += len(samples)
    ea, ef = np.array(err_all), np.array(err_far)
    print(f"[bin80 quick] ckpt={Path(args.ckpt).name} nf={nf} val_n={n_val}")
    print(f"  detected present n={len(ea)}  vL_rel err  all: bias {ea.mean():+.2f} std {ea.std():.2f}")
    print(f"  far(40-60m) n={len(ef)}  vL_rel err  bias {ef.mean():+.2f}  STD {ef.std():.2f}")
    print(f"  ---- 비교: 40bin(tspan) s10n10 far std = 3.0 ----")
    print(f"  bin80 far std {ef.std():.2f} vs 40bin 3.0  → {'개선' if ef.std() < 2.6 else '미미(temporal 포화)'}")


if __name__ == "__main__":
    main()
