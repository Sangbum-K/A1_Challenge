"""predicted-vrel cache 빌더 (접근 B) — 5f A2ModelMultiFrame 로 per-frame in-lane closing 예측.

목적: 신뢰권에 든 a_2 vrel(5f_10_20_30_40_50, vLrel_corr 0.836)을 f_2 에 흘리기 위해,
  prof635 dataset 의 각 curr_key 에 대해 in-lane(center bin 16-25) lead 의 predicted closing
  (= -predicted vL_rel) scalar 를 미리 계산해 cache. inlane_gt cache 와 동일 출력 형식
  (keys, inlane_dist, inlane_closing) → train_prof635_side 의 inlane_gt 주입 경로를 그대로 재사용.

aggregation 은 build_inlane_gt_cache.py 와 동일:
  center bin 16-25 중 predicted obj prob > thr 인 col 중 predicted dist 최소인 col 의
  predicted vL_rel → closing = -vL_rel. lead 없으면 dist=inf, closing=0.

i_1 cache 에서 frame0=curr + prev offsets[10,20,30,40,50] token fetch (tick-off, fallback=last).
yaw_shift off (mf_sweep 5f 와 동일), ego_motion=0.

사용:
  python scripts/build_pred_vrel_cache.py --ds sangam_lead_0520 \
    --a2-ckpt outputs_multi/a2/mf_sweep_5f_10_20_30_40_50/best_loss.pt
"""
import argparse, os, sys
from pathlib import Path
import numpy as np
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.model.image_models.a2_model import A2ModelMultiFrame

D = os.environ.get("SCC_DATA_DIR") or "/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data"

# prof635 dataset → (prof635 prebake, i1 cache(prev20))
DSETS = {
    "sangam_lead_0520": ("f1_prebake_iy_sangam_lead_0520_v1_brakeaug_prev20.npz",
                         "i1_tok_cache_sangam_lead_0520_prev20.npz"),
    "sangam_lead_0521": ("f1_prebake_iy_sangam_lead_0521_v1_brakeaug_prev20.npz",
                         "i1_tok_cache_sangam_lead_0521_prev20.npz"),
    "sangam_pertO_v2":  ("f1_prebake_iy_sangam_pertO_v2_prev20_fixed.npz",
                         "i1_tok_cache_sangam_pertO_v2_prev20.npz"),
    "sangam_pertX_v2":  ("f1_prebake_iy_sangam_pertX_v2_prev20_fixed.npz",
                         "i1_tok_cache_sangam_pertX_v2_prev20.npz"),
    "DriverB_new":      ("f1_prebake_iy_DriverB_new_logged_fixed_prev20_trajfix.npz",
                         "i1_tok_cache_DriverB_new_prev20.npz"),
}

BIN_LO, BIN_HI = 16, 25                       # center (in-lane) bins, inlane_gt 와 동일
PREV_OFFSETS = [10, 20, 30, 40, 50]           # 5f_10_20_30_40_50 (frame0=curr 제외)
SCALE_V, SCALE_DIST = 20.0, 40.0              # 학습 normalize scale
# aux/logits ch (predict_var=False): [obj, vL_abs, vY_abs, vL_rel, vY_rel, prox, dist]
CH_OBJ, CH_VLREL, CH_DIST = 0, 3, 6


def _prev_key(run, tick, off):
    return f"{run}__t{tick - off:07d}"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=list(DSETS.keys()))
    ap.add_argument("--a2-ckpt", required=True)
    ap.add_argument("--out", default=None, help="기본 cache/pred_vrel_{ds}.npz")
    ap.add_argument("--obj-thr", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    pbf, i1f = DSETS[args.ds]
    out = args.out or f"{D}/cache/pred_vrel_{args.ds}.npz"

    pb = np.load(f"{D}/prebake/{pbf}", allow_pickle=True)
    curr_keys = [str(k) for k in pb["curr_keys"].tolist()]
    print(f"[{args.ds}] prof635 prebake curr_keys={len(curr_keys)}", flush=True)

    z = np.load(f"{D}/cache/{i1f}", allow_pickle=True)
    tokens = z["tokens"]                       # (N,40,C_b)
    keys = z["keys"].tolist()
    k2i = {k: i for i, k in enumerate(keys)}
    print(f"  i1 cache frames={len(keys)} tok_shape={tokens.shape}", flush=True)

    n_no_curr = sum(1 for k in curr_keys if k not in k2i)
    print(f"  curr_keys not in i1 cache: {n_no_curr}", flush=True)

    model = A2ModelMultiFrame(d_token=256, d_out=128,
                              n_frames=1 + len(PREV_OFFSETS)).to(dev).eval()
    sd = torch.load(args.a2_ckpt, map_location=dev, weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"  [a2 5f] loaded missing={len(miss)} unexpected={len(unexp)}", flush=True)

    out_dist = np.full(len(curr_keys), np.inf, dtype=np.float32)
    out_clos = np.zeros(len(curr_keys), dtype=np.float32)
    n_lead = 0

    for bi in range(0, len(curr_keys), args.batch):
        bkeys = curr_keys[bi:bi + args.batch]
        # frame0=curr token + prev offsets (없으면 직전 offset idx, 최종 self) — 학습 dataset 규칙
        frame_idx = [[] for _ in range(1 + len(PREV_OFFSETS))]
        for k in bkeys:
            ci = k2i.get(k, None)
            if ci is None:
                # curr 미존재 → 전부 0번 frame 의 dummy (closing=0 로 남음). 안전하게 첫 idx.
                ci = 0
                frame_idx[0].append(ci)
                last = ci
                for fi in range(len(PREV_OFFSETS)):
                    frame_idx[fi + 1].append(last)
                continue
            frame_idx[0].append(ci)
            run, tk = k.rsplit("__t", 1); tick = int(tk)
            last = ci
            for fi, off in enumerate(PREV_OFFSETS):
                pi = k2i.get(_prev_key(run, tick, off), last)
                frame_idx[fi + 1].append(pi)
                last = pi
        # proj each frame (yaw_shift off, ego=0)
        ego = torch.zeros(len(bkeys), 2, device=dev)
        proj_list = []
        for fi in range(1 + len(PREV_OFFSETS)):
            tok = torch.from_numpy(tokens[frame_idx[fi]].astype(np.float32)).to(dev)
            proj_list.append(model.encoder.proj(tok.transpose(1, 2)).transpose(1, 2))
        fi_mem = model.fusion(proj_list, ego)
        logits = model.head(model.fc_proj(fi_mem))            # (B,40,7)
        obj_prob = torch.sigmoid(logits[:, :, CH_OBJ])         # (B,40)
        vlrel = logits[:, :, CH_VLREL] * SCALE_V               # (B,40) m/s (denorm)
        dist = logits[:, :, CH_DIST] * SCALE_DIST              # (B,40) m
        cen_obj = obj_prob[:, BIN_LO:BIN_HI + 1].cpu().numpy()
        cen_vl = vlrel[:, BIN_LO:BIN_HI + 1].cpu().numpy()
        cen_d = dist[:, BIN_LO:BIN_HI + 1].cpu().numpy()
        for j, k in enumerate(bkeys):
            gi = bi + j
            if k not in k2i:
                continue
            om = cen_obj[j] > args.obj_thr
            if not om.any():
                continue
            d = cen_d[j].copy()
            d[~om] = np.inf                                    # lead 인 col 만 최근접 후보
            sel = int(np.argmin(d))
            out_dist[gi] = float(cen_d[j][sel])
            out_clos[gi] = float(-cen_vl[j][sel])              # closing = -vL_rel (양수=접근)
            n_lead += 1
        if (bi // args.batch + 1) % 20 == 0:
            print(f"  {bi+len(bkeys)}/{len(curr_keys)}  lead-so-far={n_lead}", flush=True)

    keys_arr = np.array(curr_keys)
    np.savez_compressed(out, keys=keys_arr, inlane_dist=out_dist, inlane_closing=out_clos)
    m = np.isfinite(out_dist)
    print(f"[saved] {out}  N={len(curr_keys)}  in-lane lead={n_lead} ({100*n_lead/max(len(curr_keys),1):.0f}%)",
          flush=True)
    if m.any():
        print(f"  pred inlane_dist(lead): mean={out_dist[m].mean():.1f}m  "
              f"closing: mean={out_clos[m].mean():+.2f} std={out_clos[m].std():.2f} "
              f"(접근>0, range [{out_clos[m].min():+.1f},{out_clos[m].max():+.1f}])", flush=True)


if __name__ == "__main__":
    main()
