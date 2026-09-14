"""a_2 multi-frame / yaw_shift-ablation 학습 (cache-only).

2026-05-26 plan:
  P1 (yaw_shift ablation): n_frames=2 로 학습. --yaw-shift {on,off}.
    - on:  prev frame(t-20) 을 실제 누적 d_yaw 로 yaw_shift_tokens_batch 회전 후 fusion.
    - off: 회전 없이 그대로 fusion (d_yaw=0).
    실제 d_yaw 는 prebake(_a2inlane_prev20) 에서 curr_key 별 추출 (없으면 0).
  P2 (multi-frame velocity): n_frames=3 (curr,t-20,t-40) / 4 (+t-60).
    prev frame token 은 i_1 cache 에서 tick-{20,40,60} 직접 조회.
    yaw_shift 는 P1 판정대로 (--yaw-shift). on 이면 각 prev 에 누적 d_yaw·(offset/20) 근사 회전.

데이터 pairing 은 v7b/single 과 동일 (train_a2_v5_from_cache.py 와 같은 --i1-cache/--aux-dir/--run-filter).
모델: A2ModelMultiFrame (encoder frozen, fusion+fc_proj+head 학습), v7b ckpt resume.

사용 예 (launcher 참조):
  python scripts/train_a2_multiframe.py --n-frames 2 --yaw-shift off ...
"""
from __future__ import annotations
import os, sys, json, time, argparse, glob, re
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.model.image_models.a2_model import (A2ModelMultiFrame, A2ModelMultiFrameHR,
                                              mean_indices, sigma_indices)
from src.model.image_models.iy_model import yaw_shift_tokens_batch

AUX_PAT = re.compile(r".+?__([^_]+(?:_[^_]+)*?)__t(\d+)\.npz$")
PREV_OFFSETS = {2: [20], 3: [20, 40], 4: [20, 40, 60]}   # frame0=curr 제외 prev offsets
DYAW_CLAMP = 0.30   # 200ms 당 yaw 변화 물리한계 (rad). prebake d_yaw 에 ±2π wrap 미적용 outlier 차단


def _wrap_clamp_dyaw(dy: torch.Tensor) -> torch.Tensor:
    """d_yaw 를 [-pi,pi] wrap 후 ±DYAW_CLAMP clamp. prebake yaw-diff wrap 버그 방어."""
    two_pi = 2.0 * 3.141592653589793
    dy = dy - two_pi * torch.round(dy / two_pi)
    return dy.clamp(-DYAW_CLAMP, DYAW_CLAMP)


def build_dyaw_lut(prebake_paths):
    """prebake(_prev20) 들에서 curr_key→d_yaw(누적, 20tick) lut. 첫 occurrence 사용."""
    lut = {}
    for p in prebake_paths:
        if not p or not Path(p).exists():
            continue
        z = np.load(p, allow_pickle=True)
        ck = np.asarray(z["curr_keys"]); dy = np.asarray(z["d_yaws"]).astype(np.float32)
        for i in range(len(ck)):
            k = str(ck[i])
            if k not in lut:
                lut[k] = float(dy[i])
    return lut


def _lds_weight(tgt_reg, obj_mask, nbins=20):
    w = torch.ones(obj_mask.shape, device=tgt_reg.device, dtype=torch.float32)
    m = obj_mask
    if m.sum() < 10:
        return w
    for ch in (2, 5):   # vL_rel, dist
        vals = tgt_reg[..., ch][m]
        lo = vals.min(); hi = vals.max()
        if (hi - lo) < 1e-6:
            continue
        bins = ((vals - lo) / (hi - lo) * (nbins - 1)).long().clamp(0, nbins - 1)
        cnt = torch.bincount(bins, minlength=nbins).float()
        dens = cnt[bins] / cnt.sum().clamp(min=1.0)
        cw = 1.0 / (dens + 1e-3)
        cw = cw / cw.mean().clamp(min=1e-6)
        cw = cw.clamp(max=10.0)
        tmp = torch.ones_like(w)
        tmp[m] = cw
        w = w * tmp.pow(0.5)
    return w


def _reg_loss(logits, target, scales, obj_mask, args):
    if not obj_mask.any():
        return torch.zeros((), device=logits.device)
    pv = args.gaussian_nll
    midx = mean_indices(pv)
    mean = logits[..., midx]
    tgt_reg = target[..., 1:7] / scales
    if not pv:
        if args.lds:
            w = _lds_weight(tgt_reg.detach(), obj_mask, args.lds_nbins)
            err = (mean - tgt_reg).abs().mean(-1)
            m = obj_mask
            return (err[m] * w[m]).sum() / w[m].sum().clamp(min=1e-6)
        m = obj_mask.unsqueeze(-1).expand_as(mean)
        return F.l1_loss(mean[m], tgt_reg[m])
    sidx = sigma_indices()
    logvar = logits[..., sidx].clamp(-7.0, 7.0)
    nll = 0.5 * (torch.exp(-logvar) * (mean - tgt_reg) ** 2 + logvar)
    nll = nll.mean(-1)
    if args.lds:
        w = _lds_weight(tgt_reg.detach(), obj_mask, args.lds_nbins)
    else:
        w = torch.ones_like(obj_mask, dtype=torch.float32)
    m = obj_mask
    return (nll[m] * w[m]).sum() / w[m].sum().clamp(min=1e-6)


class A2MultiFrameDataset(Dataset):
    """i_1 cache + aux + N-frame prev (tick-20/-40/-60) + 실제 d_yaw."""
    def __init__(self, i1_cache_path, aux_dir, prev_offsets, dyaw_lut,
                 run_filter=None, fx=640.0):
        z = np.load(i1_cache_path, allow_pickle=True)
        self.tokens = z["tokens"]               # (N, 40, C_b)
        self.tokens16 = z["tokens_s16"] if "tokens_s16" in z.files else None  # (N, 80, C16) HR
        self.keys = z["keys"].tolist()
        self.k2i = {k: i for i, k in enumerate(self.keys)}
        self.fx = float(fx)
        self.prev_offsets = prev_offsets        # [20] / [20,40] / [20,40,60]
        self.dyaw_lut = dyaw_lut
        aux_files = sorted(glob.glob(str(Path(aux_dir) / "*.npz")))
        self.samples = []   # (curr_idx, aux_path, run, tick)
        self.prev_idx = []  # list of [pidx_off1, pidx_off2, ...]
        n_runfilt = 0; n_nokey = 0
        for f in aux_files:
            fname = Path(f).name
            m = AUX_PAT.match(fname)
            if not m:
                continue
            run, tick = m.group(1), int(m.group(2))
            # run_filter 는 policy(pertO/pertX) 구분용. AUX_PAT group(1) 은 run-token
            # (예: 260519_100) 만 잡아 policy 토큰을 버리므로, 필터는 파일명 전체에 적용.
            if run_filter and run_filter not in fname:
                n_runfilt += 1
                continue
            key = f"{run}__t{tick:07d}"
            if key not in self.k2i:
                n_nokey += 1
                continue
            ci = self.k2i[key]
            # prev frame indices (없으면 가장 가까운 가용: 직전 offset 의 idx, 최종 self)
            pidxs = []
            last = ci
            for off in prev_offsets:
                pk = f"{run}__t{tick - off:07d}"
                pi = self.k2i.get(pk, last)
                pidxs.append(pi)
                last = pi
            self.samples.append((ci, f, run, tick))
            self.prev_idx.append(pidxs)
        n_aux = sum(1 for f in aux_files if AUX_PAT.match(Path(f).name))
        print(f"[mf-ds] {Path(i1_cache_path).name}  matched {len(self.samples):,}/{len(aux_files):,} "
              f"(parseable_aux={n_aux:,} run-filtered={n_runfilt:,} no-key={n_nokey:,}) "
              f"(N_frames={1+len(prev_offsets)})", flush=True)
        # 가드: run-filter 적용 후(eligible) aux 대비 join 비율이 낮으면 silent drop 경고.
        #   denominator 는 n_aux - n_runfilt (run-filter 는 의도적 제외 — pertO/pertX 가 한 aux dir
        #   에 섞여 있어 policy 1개만 남기면 ~48%; 이건 정상). key-mismatch(no-key) 만 잡는다.
        n_eligible = n_aux - n_runfilt
        if n_eligible > 0 and len(self.samples) < 0.5 * n_eligible:
            raise RuntimeError(
                f"[mf-ds][FAIL-FAST] join {len(self.samples):,}/{n_eligible:,} (eligible) < 50% — "
                f"i1-cache↔aux key 불일치 의심 (no-key={n_nokey:,}). "
                f"cache={Path(i1_cache_path).name} aux={Path(aux_dir).name} run-filter='{run_filter}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        ci, ap, run, tick = self.samples[i]
        pidxs = self.prev_idx[i]
        toks = [torch.from_numpy(self.tokens[ci].astype(np.float32))]  # frame0 = curr
        for pi in pidxs:
            toks.append(torch.from_numpy(self.tokens[pi].astype(np.float32)))
        tok_stack = torch.stack(toks, dim=0)   # (Nf, 40, C_b)
        # 실제 d_yaw (curr 기준 t-20 누적). 더 먼 prev 는 비례 근사.
        key = f"{run}__t{tick:07d}"
        d_yaw20 = float(self.dyaw_lut.get(key, 0.0))
        ego_motion = torch.zeros(2, dtype=torch.float32)
        fx = torch.tensor(self.fx, dtype=torch.float32)
        t = np.load(ap)["target"].astype(np.float32)   # (n_bins, 7)
        out = {
            "tok_stack": tok_stack,                       # (Nf, 40, C_b)
            "d_yaw20": torch.tensor(d_yaw20, dtype=torch.float32),
            "ego_motion": ego_motion,
            "fx": fx,
            "target": torch.from_numpy(t),
        }
        if self.tokens16 is not None:
            t16 = [torch.from_numpy(self.tokens16[ci].astype(np.float32))]
            for pi in pidxs:
                t16.append(torch.from_numpy(self.tokens16[pi].astype(np.float32)))
            out["tok16_stack"] = torch.stack(t16, dim=0)  # (Nf, 80, C16)
        return out


def collate(items):
    return {k: torch.stack([it[k] for it in items], dim=0) for k in items[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i1-cache", action="append", required=True)
    ap.add_argument("--aux-dir", action="append", required=True)
    ap.add_argument("--run-filter", action="append", default=[])
    ap.add_argument("--prebake", action="append", default=[],
                    help="_prev20 prebake (d_yaw lut). 순서 무관, 합쳐서 lut 구성")
    ap.add_argument("--n-frames", type=int, default=2)
    ap.add_argument("--prev-offsets", type=str, default="",
                    help="comma-sep prev tick offsets (예: '10,20,30'). 비면 PREV_OFFSETS[n_frames]. "
                         "지정 시 n_frames = 1+len(offsets) 강제. 임의 offset/span sweep 용.")
    ap.add_argument("--yaw-shift", choices=["on", "off"], default="off",
                    help="on: prev frame 을 실제 누적 d_yaw 로 회전 후 fusion. off: d_yaw=0")
    ap.add_argument("--fx", type=float, default=640.0)
    ap.add_argument("--resume-from", required=True, help="v7b ckpt (encoder.proj weights load)")
    ap.add_argument("--d-token", type=int, default=256,
                    help="encoder.proj 출력 = fusion 내부 dim. b0 cache(C_b=192) 위 학습 시에도 "
                         "proj 이 C_b→d_token 매핑하므로 d_token 은 자유. b0 ablation 은 192 권장(공정 비교).")
    ap.add_argument("--i1-cb", type=int, default=None,
                    help="i1 cache 의 token channel(C_b). 명시 시 encoder.proj in_channels 강제. "
                         "기본=cache 에서 자동 추론.")
    ap.add_argument("--d-out", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--w-obj", type=float, default=1.0)
    ap.add_argument("--w-reg", type=float, default=1.0)
    ap.add_argument("--use-focal", action="store_true")
    ap.add_argument("--focal-alpha", type=float, default=0.75)
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    # 분포 분리 실험 (기본 0=off, 기존 동작 보존)
    ap.add_argument("--hard-neg-w", type=float, default=0.0)     # Exp1 OHEM: in-lane 최난 negative top-frac 에 BCE-to-0 가중 → absent obj↓
    ap.add_argument("--hard-neg-frac", type=float, default=0.1)
    ap.add_argument("--margin-w", type=float, default=0.0)       # Exp2: present_max - neg_max ≥ margin 강제 → gap↑
    ap.add_argument("--margin-m", type=float, default=2.0)
    ap.add_argument("--scale-v", type=float, default=20.0)
    ap.add_argument("--scale-dist", type=float, default=40.0)
    ap.add_argument("--gaussian-nll", action="store_true")
    ap.add_argument("--lds", action="store_true")
    ap.add_argument("--lds-nbins", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hr", action="store_true",
                    help="80-col stride16 1D-merge (bin80). cache 에 tokens_s16 필요 + aux GT n_bins=80. "
                         "in-lane center index 32:52, resume 시 proj→proj32 / lateral_pos 40→80 interp.")
    ap.add_argument("--out-dir", default="outputs_multi")
    ap.add_argument("--name", default="a2_multiframe")
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out_dir = Path(args.out_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / "train.log", "w")
    def logp(m): print(m, flush=True); log_f.write(m + "\n"); log_f.flush()
    logp(f"args={vars(args)}")

    if args.prev_offsets.strip():
        prev_offsets = [int(x) for x in args.prev_offsets.split(",") if x.strip()]
        args.n_frames = 1 + len(prev_offsets)
    elif args.n_frames == 1:
        prev_offsets = []   # 1f baseline: temporal 없음 (KV=[ego] 만)
    else:
        prev_offsets = PREV_OFFSETS[args.n_frames]
    dyaw_lut = build_dyaw_lut(args.prebake) if args.yaw_shift == "on" else {}
    logp(f"[dyaw] yaw_shift={args.yaw_shift}  lut_size={len(dyaw_lut)}  prev_offsets={prev_offsets}")

    if len(args.i1_cache) != len(args.aux_dir):
        raise RuntimeError("--i1-cache 와 --aux-dir 갯수 동일 필요")
    run_filters = args.run_filter + [None] * (len(args.i1_cache) - len(args.run_filter))
    ds_list = []
    for i1p, ad, rf in zip(args.i1_cache, args.aux_dir, run_filters):
        ds_list.append(A2MultiFrameDataset(i1p, ad, prev_offsets, dyaw_lut,
                                            run_filter=rf, fx=args.fx))
    full = ConcatDataset(ds_list)
    N = len(full)
    n_val = int(N * args.val_ratio)
    n_tr = N - n_val
    tr_ds, va_ds = random_split(full, [n_tr, n_val],
                                generator=torch.Generator().manual_seed(args.seed))
    logp(f"[data] total={N:,}  train={n_tr:,}  val={n_val:,}")
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                       collate_fn=collate, pin_memory=True, persistent_workers=args.workers > 0)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                       collate_fn=collate, pin_memory=True, persistent_workers=args.workers > 0)

    # i1 cache 의 C_b 추론 (b0=192, v2_s=256). encoder.proj in_channels 를 cache 에 맞춤.
    cb_cache = args.i1_cb
    if cb_cache is None:
        cb_cache = int(ds_list[0].tokens.shape[-1])
    if args.hr:
        if ds_list[0].tokens16 is None:
            raise RuntimeError("--hr 인데 cache 에 tokens_s16 없음. compute_i1_tok_cache.py --hr 로 재생성 필요")
        model = A2ModelMultiFrameHR(d_token=args.d_token, d_out=args.d_out, n_frames=args.n_frames,
                                    predict_var=args.gaussian_nll).to(args.device)
    else:
        model = A2ModelMultiFrame(d_token=args.d_token, d_out=args.d_out, n_frames=args.n_frames,
                                  predict_var=args.gaussian_nll).to(args.device)
    # cache C_b 가 d_token 과 다르면 stride32 proj 를 Conv1d(C_b -> d_token) 로 재생성
    if not args.hr and cb_cache != model.encoder.proj.in_channels:
        import torch.nn as _nn
        model.encoder.proj = _nn.Conv1d(cb_cache, args.d_token, kernel_size=1).to(args.device)
        logp(f"[proj] rebuilt encoder.proj: in={cb_cache} out={args.d_token} (cache C_b={cb_cache})")
    sd = torch.load(args.resume_from, map_location=args.device, weights_only=False)
    if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
    if args.hr:
        # v7b(A2ModelMultiFrame) → HR: proj→proj32 rename, lateral_pos 40→80 interp. proj16/alpha new(missing).
        sd2 = {}
        for k, v in sd.items():
            if k.startswith("encoder.proj."):
                sd2[k.replace("encoder.proj.", "encoder.proj32.")] = v
            elif k == "fusion.lateral_pos":
                lp = v.t().unsqueeze(0)                                  # (1,d,40)
                lp = F.interpolate(lp, size=80, mode="linear", align_corners=False)
                sd2[k] = lp.squeeze(0).t().contiguous()                  # (80,d)
            else:
                sd2[k] = v
        sd = sd2
    own = model.state_dict()
    skip = [k for k in list(sd.keys()) if k in own and sd[k].shape != own[k].shape]
    for k in skip: del sd[k]
    miss, unexp = model.load_state_dict(sd, strict=False)
    logp(f"[resume] {args.resume_from}  loaded(into own)~{len(own)-len(miss)}  "
         f"missing={len(miss)} (new fusion/head)  shape-skip={len(skip)}")
    model.freeze_encoder()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logp(f"[model] n_frames={args.n_frames}  trainable={n_train/1e3:.1f}K (fusion+fc_proj+head)")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    bce_fn = nn.BCEWithLogitsLoss()
    def focal_loss(logits, targets, alpha=args.focal_alpha, gamma=args.focal_gamma):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** gamma * bce).mean()
    obj_loss_fn = focal_loss if args.use_focal else bce_fn
    logp(f"[loss] obj={'focal' if args.use_focal else 'bce'} reg={'NLL' if args.gaussian_nll else 'L1'}"
         f"{'+LDS' if args.lds else ''}")
    scales = torch.tensor([args.scale_v, args.scale_v, args.scale_v, args.scale_v,
                           1.0, args.scale_dist], device=args.device)

    LANE0, LANE1 = (32, 52) if args.hr else (16, 26)   # in-lane center bins (80bin 이면 2배)

    def proj_frames(tok_stack, tok16_stack, d_yaw20, fx_t):
        """tok_stack (B,Nf,40,C_b) [+ tok16_stack (B,Nf,80,C16) HR] → list of Nf (B,L,d_token).
        HR: stride16 1D-merge → (B,80,d). 아니면 stride32 proj (+yaw_shift) → (B,40,d)."""
        Bsz, Nf = tok_stack.shape[0], tok_stack.shape[1]
        out = []
        if args.hr:
            for fi in range(Nf):
                out.append(model.encoder._merge(tok_stack[:, fi], tok16_stack[:, fi]))  # (B,80,d)
            return out
        d_yaw_c = _wrap_clamp_dyaw(d_yaw20)
        for fi in range(Nf):
            tok = tok_stack[:, fi]                                   # (B,40,C_b)
            if args.yaw_shift == "on" and fi >= 1:
                # prev offset = prev_offsets[fi-1], d_yaw 비례 (t-20 누적 기준 * offset/20)
                scale = prev_offsets[fi - 1] / 20.0
                tok = yaw_shift_tokens_batch(tok, d_yaw_c * scale, fx=fx_t)
            tp = model.encoder.proj(tok.transpose(1, 2)).transpose(1, 2)   # (B,40,d_token)
            out.append(tp)
        return out

    best_val = float("inf")
    for ep in range(1, args.epochs + 1):
        model.train(); model.encoder.eval()
        t0 = time.time()
        for bi, b in enumerate(tr_dl):
            tok_stack = b["tok_stack"].to(args.device, non_blocking=True)    # (B,Nf,40,C_b)
            tok16_stack = b["tok16_stack"].to(args.device, non_blocking=True) if args.hr else None
            d_yaw20 = b["d_yaw20"].to(args.device, non_blocking=True)
            fx_t = b["fx"].to(args.device, non_blocking=True)
            ego = b["ego_motion"].to(args.device, non_blocking=True)
            target = b["target"].to(args.device, non_blocking=True)
            if args.hr:
                proj_list = proj_frames(tok_stack, tok16_stack, d_yaw20, fx_t)  # proj16/alpha 학습 → grad 필요
            else:
                with torch.no_grad():
                    proj_list = proj_frames(tok_stack, None, d_yaw20, fx_t)
            fi_mem = model.fusion(proj_list, ego)
            logits = model.head(model.fc_proj(fi_mem))
            obj_logit = logits[..., 0]; obj_gt = target[..., 0]
            loss_obj = obj_loss_fn(obj_logit, obj_gt)
            obj_mask = obj_gt > 0.5
            loss_reg = _reg_loss(logits, target, scales, obj_mask, args)
            loss = args.w_obj * loss_obj + args.w_reg * loss_reg
            if args.hard_neg_w > 0:                       # Exp1 OHEM: in-lane 최난 negative 억제
                cl = obj_logit[:, LANE0:LANE1]; cg = obj_gt[:, LANE0:LANE1]
                nl = cl[cg < 0.5]
                if nl.numel() > 0:
                    k = max(1, int(args.hard_neg_frac * nl.numel()))
                    hard = torch.topk(nl, k).values
                    loss = loss + args.hard_neg_w * F.binary_cross_entropy_with_logits(hard, torch.zeros_like(hard))
            if args.margin_w > 0:                         # Exp2 margin: present_max - neg_max ≥ m
                cl = obj_logit[:, LANE0:LANE1]; cg = obj_gt[:, LANE0:LANE1]
                pm = cg > 0.5; hp = pm.any(dim=1)
                if hp.any():
                    pos_l = cl.masked_fill(~pm, -1e4).amax(dim=1)
                    neg_l = cl.masked_fill(pm, -1e4).amax(dim=1)
                    loss = loss + args.margin_w * F.relu(args.margin_m - (pos_l - neg_l))[hp].mean()
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            if (bi + 1) % args.log_every == 0:
                logp(f"[ep {ep:02d} | {bi+1:>5,}/{len(tr_dl):,}] total={loss:.4f} "
                     f"obj={loss_obj:.4f} reg={loss_reg:.4f} lr={optim.param_groups[0]['lr']:.2e}")
        sched.step()
        model.eval()
        with torch.no_grad():
            v_total, v_bce, v_reg, n = 0., 0., 0., 0
            for b in va_dl:
                tok_stack = b["tok_stack"].to(args.device, non_blocking=True)
                tok16_stack = b["tok16_stack"].to(args.device, non_blocking=True) if args.hr else None
                d_yaw20 = b["d_yaw20"].to(args.device, non_blocking=True)
                fx_t = b["fx"].to(args.device, non_blocking=True)
                ego = b["ego_motion"].to(args.device, non_blocking=True)
                target = b["target"].to(args.device, non_blocking=True)
                proj_list = proj_frames(tok_stack, tok16_stack, d_yaw20, fx_t)
                fi_mem = model.fusion(proj_list, ego)
                logits = model.head(model.fc_proj(fi_mem))
                obj_logit = logits[..., 0]; obj_gt = target[..., 0]
                l_obj = obj_loss_fn(obj_logit, obj_gt)
                obj_mask_v = obj_gt > 0.5
                if obj_mask_v.any():
                    mean_v = logits[..., mean_indices(args.gaussian_nll)]
                    tgt_reg_v = target[..., 1:7] / scales
                    m_v = obj_mask_v.unsqueeze(-1).expand_as(mean_v)
                    l_reg = F.l1_loss(mean_v[m_v], tgt_reg_v[m_v])
                else:
                    l_reg = torch.zeros((), device=logits.device)
                bsz = tok_stack.size(0)
                v_total += (args.w_obj * l_obj + args.w_reg * l_reg).item() * bsz
                v_bce += l_obj.item() * bsz; v_reg += l_reg.item() * bsz; n += bsz
        v_total /= n; v_bce /= n; v_reg /= n
        dt = time.time() - t0
        logp(f"[ep {ep:02d}] val total={v_total:.4f} obj={v_bce:.4f} reg={v_reg:.4f} time={dt:.1f}s")
        if v_total < best_val:
            best_val = v_total
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": ep, "val": v_total},
                       out_dir / "best_loss.pt")
            logp(f"  -> saved best ({v_total:.4f})")
        torch.save({"model": model.state_dict(), "args": vars(args), "epoch": ep, "val": v_total},
                   out_dir / "last.pt")
    log_f.close()


if __name__ == "__main__":
    main()
