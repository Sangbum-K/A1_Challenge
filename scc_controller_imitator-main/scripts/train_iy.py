"""i_y model 학습 (train_f1_cached 와 동일 패턴, dataset/model 만 swap)."""
from __future__ import annotations
import os, sys, time, json, argparse, random, math
import numpy as np
import torch


class _CosineDecayingMaxRestarts:
    """Cosine annealing warm restarts + 매 cycle eta_max 도 cosine 으로 감쇠.
    cycle c (0-indexed) 의 eta_max(c) = base_lr * 0.5 * (1 + cos(π·c/N))
        c=0 → base_lr (full),  c=N-1 → near zero (not exactly 0 — 마지막 cycle 도 학습 가능)
    cycle 안 t-step lr = eta_min + (eta_max(c) - eta_min) * 0.5 * (1 + cos(π·t/T_0))
    버그 fix (2026-05-24): denom 이 N-1 이면 last cycle 의 max_factor=0 → 학습 멈춤 (v25c 에서 발견).
    N 으로 변경 시 N=5: cycle factors [1.0, 0.905, 0.655, 0.345, 0.095] — 마지막 도 살아있음.
    """
    def __init__(self, optimizer, T_0: int, n_cycles: int, eta_min: float = 0.0):
        self.optimizer = optimizer
        self.T_0 = int(T_0)
        self.n_cycles = int(n_cycles)
        self.eta_min = float(eta_min)
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]
        self._step_count = 0
        self._apply()

    def _apply(self):
        c = min(self._step_count // self.T_0, self.n_cycles - 1)
        t_in_cycle = self._step_count % self.T_0
        max_factor = 0.5 * (1 + math.cos(math.pi * c / self.n_cycles)) if self.n_cycles > 1 else 1.0
        cycle_factor = 0.5 * (1 + math.cos(math.pi * t_in_cycle / self.T_0))
        for g, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            g['lr'] = self.eta_min + (base_lr * max_factor - self.eta_min) * cycle_factor

    def step(self, epoch=None):
        self._step_count += 1
        self._apply()
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler, Subset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.datamodule.scc_synth_iy_dataset import SCCSynthIyDataset, collate_iy
from src.model.image_models.iy_model import IyModel


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--professor-ckpt", default=None)
    ap.add_argument("--prebake", default="/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/prebake/f1_prebake_iy_v1.npz")
    ap.add_argument("--i1-cache", default="/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/cache/i1_tok_cache_v1.npz")
    # multi-source: 여러 (prebake, i1_cache) 쌍 받기 (ConcatDataset 통합)
    ap.add_argument("--prebake-extra", nargs="*", default=[],
                    help="추가 prebake npz path 들 (multi-policy 학습)")
    ap.add_argument("--i1-cache-extra", nargs="*", default=[],
                    help="추가 i1_cache npz path 들 (prebake-extra 와 같은 순서, 1:1 매핑)")
    ap.add_argument("--prebake-extra-limits", nargs="*", type=int, default=None,
                    help="extra prebake 별 sample limit (길이 = len(prebake-extra), 0=무제한). main prebake 는 --limit 사용.")
    ap.add_argument("--c-b", type=int, default=0, help="i_1 backbone C_b (0=auto from cache; effnetv2_s ≈ 256)")
    ap.add_argument("--fx", type=float, default=320.0, help="camera fx (i_y shift 계산)")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--num-layers", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-prof", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default="outputs_multi/iy")
    ap.add_argument("--name", default="v1")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--scale-steer", type=float, default=10.0)
    ap.add_argument("--scale-ax",    type=float, default=10.0)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="scc-controller-imitator")
    ap.add_argument("--wandb-entity",  default="hmg_e2e_controller_imitator")
    ap.add_argument("--wandb-run",     default=None)
    ap.add_argument("--resume-from", default=None)
    # streaming: prebake npz mtime 변경 감지 → Dataset/DataLoader 재build
    ap.add_argument("--reload-on-change", action="store_true",
                    help="매 epoch 시작 시 prebake npz mtime 체크 → 변경 시 Dataset 재build (streaming 학습)")
    # controller_imitator augmentation (image=0 padded, no-lead 시나리오 강화)
    ap.add_argument("--ci-aug-dirs", nargs="*", default=[],
                    help="controller_imitator v0301/v0302 등 dirs. image token 0 padded ConcatDataset 추가")
    ap.add_argument("--ci-aug-limit", type=int, default=None)
    # close-lead 가중치 옵션
    ap.add_argument("--close-lead-meta", default=None,
                    help="close_lead_meta.npz path (oversample/loss-weight 활성화)")
    ap.add_argument("--close-thr", type=float, default=30.0,
                    help="close-lead 임계 m")
    ap.add_argument("--close-weight", type=float, default=3.0,
                    help="close-lead 샘플 oversample 가중치 (1=균등)")
    ap.add_argument("--close-loss-mult", type=float, default=1.0,
                    help="close-lead 샘플 loss 가중치 multiplier (1=균등)")
    ap.add_argument("--difficulty-weight", action="store_true", default=False,
                    help="target |steer|+|ax| 기반 sampler difficulty weight (easy 0.5 / normal 1 / hard 2 / extreme 4). "
                         "close-weight 와 max() 결합.")
    # i_g (image gate) — obj BCE 보조 loss
    ap.add_argument("--use-obj-gate", action="store_true", default=False,
                    help="i_g: a_2 mini-head obj gate 적용 + BCE supervision")
    ap.add_argument("--obj-loss-w",   type=float, default=0.5,
                    help="obj BCE loss weight (main L1 와 합쳐짐)")
    # ControlNet-style side branch
    ap.add_argument("--use-side-branch", action="store_true", default=False,
                    help="ControlNet-style parallel side branch (zero-init gated α). main path 보존하면서 capacity 추가.")
    ap.add_argument("--side-num-layers", type=int, default=3, help="side branch transformer layer 수")
    ap.add_argument("--side-dim-feedforward", type=int, default=512, help="side branch FFN dim")
    ap.add_argument("--side-ax-only", action="store_true", default=False,
                    help="side branch 출력 1 채널 (ax 만 보완). steer 는 main 그대로.")
    ap.add_argument("--reset-side-alpha", action="store_true", default=False,
                    help="resume 후 side_alpha 를 0 으로 초기화 (identity start 보장)")
    # a_2 v4 features
    ap.add_argument("--use-a2-feat", action="store_true", default=False,
                    help="a_2 v4 per-bin (40,7) features 를 fi_mem 에 residual add (zero-init identity start)")
    ap.add_argument("--use-a2-late", action="store_true", default=False,
                    help="a_2 features 를 transformer 출력 후에도 residual add (ax decoder 직통 path)")
    ap.add_argument("--a2-cache", default=None, help="main dataset a_2 cache npz path")
    ap.add_argument("--i2-cache", default=None, help="v6 i_2 mem cache (a_2 v5/v6 fc_proj output) — image_mem 직접 입력")
    ap.add_argument("--i2-cache-extra", nargs="*", default=[],
                    help="i_2 cache extras (prebake-extra 와 같은 순서)")
    ap.add_argument("--inlane-gt-cache", default=None,
                    help="GT inlane (dist + -vL_rel) cache. ttc-augment 가 추론값 대신 GT 로 trigger (근거리 과대평가 회피)")
    ap.add_argument("--inlane-gt-cache-extra", nargs="*", default=[],
                    help="inlane GT cache extras (prebake-extra 와 같은 순서)")
    ap.add_argument("--a2-cache-extra", nargs="*", default=[],
                    help="extra datasets a_2 cache (prebake-extra 와 같은 순서/길이)")
    ap.add_argument("--phase1-epochs", type=int, default=5,
                    help="2-phase: phase1 main frozen (side+α만 학습), phase2 joint (main lr=main-lr-phase2)")
    ap.add_argument("--main-lr-phase2", type=float, default=None,
                    help="phase2 main 학습 lr. None 이면 lr_prof/20 자동")
    ap.add_argument("--side-lr", type=float, default=None,
                    help="side branch lr. None 이면 --lr 사용")
    # Asymmetric brake loss — under-brake (pred > gt 일 때 gt 가 강 brake) 추가 가중
    ap.add_argument("--under-brake-weight", type=float, default=1.0,
                    help="GT ax<-2 인데 pred > gt (under-brake) 시 loss multiplier. 1=symmetric, 3+=safety bias")
    ap.add_argument("--under-brake-thr", type=float, default=-2.0,
                    help="under-brake 판정 threshold (GT ax < thr). default -2 m/s²")
    # Ablation flags
    ap.add_argument("--traj-zero-after", type=int, default=None,
                    help="traj_40[:, K:, :] = 0 (K 이후 traj input 제거). 1.5s ≈ 15 step (0.1s tick 가정)")
    ap.add_argument("--target-first-only", action="store_true", default=False,
                    help="target_mask[:, 1:] = False (첫 step 만 loss 에 사용)")
    # v23 redesign
    ap.add_argument("--main-no-image", action="store_true", default=False,
                    help="main transformer 입력에서 image_mem 제외 (controller_imitator 정통 모드)")
    ap.add_argument("--side-cross-attn", action="store_true", default=False,
                    help="side branch 를 cross-attn (Q=traj_text, KV=image_mem) 으로 변경. ax 1-step 보완")
    ap.add_argument("--side-min-arb", action="store_true", default=False,
                    help="lead-gated ACC override: ax=main+g·clamp(b_side-main,max=0). additive 대신 절대 brake ceiling + a_2 obj gate")
    ap.add_argument("--side-min-arb-w", type=float, default=0.5,
                    help="b_side 직접 supervision loss weight (b_side → target ax step0)")
    # TTC-based brake target augment (label safety floor)
    ap.add_argument("--ttc-augment", action="store_true", default=False,
                    help="TTC 기반 safe brake floor 를 target ax 에 conservative 적용 (앞 K step)")
    ap.add_argument("--ttc-augment-steps", type=int, default=10,
                    help="TTC augment 적용할 target 시작 step 수 (default 10)")
    ap.add_argument("--ttc-augment-max-brake", type=float, default=20.0,
                    help="TTC augment 최대 brake 크기 (m/s², default 20)")
    ap.add_argument("--ttc-augment-dist-max", type=float, default=50.0,
                    help="TTC augment 적용 거리 임계 (m, default 50 — 이보다 멀어도 TTC<ttc-max 이면 적용)")
    ap.add_argument("--ttc-augment-ttc-max", type=float, default=4.0,
                    help="TTC augment 적용 시간 임계 (s, default 4 — TTC < 이 값이면 dist 무관 적용)")
    ap.add_argument("--ttc-augment-safety-margin", type=float, default=2.0,
                    help="lead 뒤 안전 buffer (m, default 2)")
    # ACC-style desired decel (kinematic 보다 강함, 근거리/긴급 brake 보강). 2026-05-25 ACC-A 튜닝.
    ap.add_argument("--ttc-acc-style", action="store_true", default=False,
                    help="safe_decel 을 ACC-style max(kinematic, headway-gap, TTC-emergency) 로. 근거리 강함 + 긴급 -20")
    ap.add_argument("--ttc-acc-d0", type=float, default=5.0, help="standstill gap (m)")
    ap.add_argument("--ttc-acc-headway", type=float, default=1.0, help="headway time T (s) — d_safe=d0+T·v_ego")
    ap.add_argument("--ttc-acc-kgap", type=float, default=0.2, help="gap deficit gain")
    ap.add_argument("--ttc-acc-kclo", type=float, default=0.6, help="closing rate gain")
    ap.add_argument("--ttc-acc-ttc-emerg", type=float, default=2.0, help="emergency TTC 임계 (s) — 이하면 max_brake 로 ramp")
    ap.add_argument("--ttc-acc-kclo-ttc-decay", type=float, default=5.0,
                    help="closing-brake 항을 TTC 로 감쇠 — TTC 이 값 초과(멀고 여유)면 closing brake 0. 원거리 강brake 완화")
    return ap.parse_args()


def _apply_ttc_augment(b, args):
    """TTC 기반 safe brake floor 를 target[:, :K, 1] 에 Conservative apply.
    Trigger: 접근 중 (closing>0.1) AND (dist<dist_max OR TTC<ttc_max) ← 보수적 (둘 중 하나라도)
    safe_decel = closing²/(2(dist-margin))
    dist/closing 우선순위: inlane (a_2 bin 16-25 in-lane) > 기존 close_lead_meta
    """
    if 'inlane_dist' in b and 'inlane_closing' in b:
        # 우선: in-lane (a_2 cache 기반 — 모든 dataset cover + bin 16-25 gating)
        dist = b['inlane_dist']; closing = b['inlane_closing']
    elif 'closing_rate' in b and 'min_dist' in b:
        # fallback: close_lead_meta 기반 (일부 dataset 만)
        dist = b['min_dist']; closing = b['closing_rate']
    else:
        return
    K = args.ttc_augment_steps
    margin = args.ttc_augment_safety_margin
    max_brake = args.ttc_augment_max_brake
    # TTC = dist / max(closing, 0.1)
    closing_safe = torch.clamp(closing, min=0.1)
    ttc = dist / closing_safe
    # mask: 접근 중 AND (가까움 OR TTC 짧음) — conservative
    approach = (closing > 0.1) & torch.isfinite(dist) & (
        (dist < args.ttc_augment_dist_max) | (ttc < args.ttc_augment_ttc_max)
    )
    if not approach.any(): return
    eff_d = torch.clamp(dist - margin, min=0.5)
    if getattr(args, 'ttc_acc_style', False):
        # ACC-style desired decel = max(kinematic, headway-gap, TTC-emergency).
        # 근거리 강한 brake + 긴급(TTC↓) 시 max_brake(-20) 까지. (2026-05-25 튜닝: ACC-A)
        v_ego = b['ego_5d'][:, 0].abs() if 'ego_5d' in b else torch.zeros_like(dist)
        a_kin = closing.pow(2) / (2.0 * eff_d)
        d_safe = args.ttc_acc_d0 + args.ttc_acc_headway * v_ego
        # closing 항을 TTC 로 gating — 멀어서 TTC 큰(여유) 접근은 brake 완화.
        # ttc < ttc_emerg → full, ttc > kclo_ttc_decay → 0 으로 선형 감쇠.
        decay = getattr(args, 'ttc_acc_kclo_ttc_decay', 5.0)
        clo_gate = torch.clamp((decay - ttc) / max(decay - args.ttc_acc_ttc_emerg, 1e-3), min=0.0, max=1.0)
        a_gap = args.ttc_acc_kgap * torch.clamp(d_safe - dist, min=0.0) \
                + args.ttc_acc_kclo * torch.clamp(closing, min=0.0) * clo_gate
        a_emerg = max_brake * torch.clamp(1.0 - ttc / args.ttc_acc_ttc_emerg, min=0.0, max=1.0)
        safe_decel = torch.maximum(torch.maximum(a_kin, a_gap), a_emerg)
    else:
        # 기존 kinematic min-decel-to-stop
        safe_decel = closing.pow(2) / (2.0 * eff_d)
    safe_decel = torch.clamp(safe_decel, max=max_brake)
    safe_ax = -safe_decel                                  # (B,) negative = brake
    # 적용 mask
    safe_ax = torch.where(approach, safe_ax, torch.full_like(safe_ax, float('inf')))
    # broadcast to (B, K, 1) and min with target[:, :K, 1] (conservative)
    target = b['target']                                   # (B, 40, 2)
    target_clone = target.clone()
    safe_ax_bcast = safe_ax.view(-1, 1).expand(-1, K)       # (B, K)
    target_clone[:, :K, 1] = torch.minimum(target_clone[:, :K, 1], safe_ax_bcast)
    b['target'] = target_clone


def split_losses(pred, target, mask, args, sample_weight=None):
    """sample_weight (B,) or None — None 이면 mean, 있으면 sample-weighted mean.
    Asymmetric brake: under-brake (pred > gt 이면서 gt < under_brake_thr) 시 ax loss 가중.
    """
    if not mask.any():
        return torch.zeros((), device=pred.device), {}
    scales = torch.tensor([args.scale_steer, args.scale_ax], device=pred.device)
    diff = (pred - target) / scales       # per-bin per-dim 정규화 오차
    abs_st = diff[..., 0].abs()
    abs_ax = diff[..., 1].abs()
    # asymmetric brake weight (axis ax, dim 1). under-brake = pred > gt 이고 gt < thr (강 brake).
    ubw = float(getattr(args, 'under_brake_weight', 1.0))
    ax_weight = None
    if ubw > 1.0:
        gt_ax_raw = target[..., 1]          # original scale m/s²
        pred_ax_raw = pred[..., 1]
        under = (pred_ax_raw > gt_ax_raw) & (gt_ax_raw < args.under_brake_thr)
        ax_weight = torch.where(under, torch.tensor(ubw, device=pred.device),
                                 torch.tensor(1.0, device=pred.device))
        abs_ax = abs_ax * ax_weight
    if sample_weight is None:
        l1_steer = abs_st[mask].mean()
        l1_ax    = abs_ax[mask].mean()
    else:
        # per-sample mean over time dim → weighted mean over batch
        denom = mask.sum(dim=1).clamp(min=1).float()
        per_st = (abs_st * mask).sum(dim=1) / denom
        per_ax = (abs_ax * mask).sum(dim=1) / denom
        w_sum  = sample_weight.sum().clamp(min=1e-6)
        l1_steer = (per_st * sample_weight).sum() / w_sum
        l1_ax    = (per_ax * sample_weight).sum() / w_sum
    out = {"l1_steer": l1_steer.detach(), "l1_ax": l1_ax.detach()}
    if ax_weight is not None:
        out["under_brake_rate"] = (ax_weight > 1).float().mean().detach()
    return l1_steer + l1_ax, out


def _build_dataset_and_loaders(args, prev_loaders=None):
    """SCCSynthIyDataset + train/val split + DataLoader 구성.
    streaming reload 호환: 이전 loader 가 있으면 shutdown.
    returns (full, tr, va, tl, vl, train_sampler, va_close_idx, va_far_idx)
    """
    if prev_loaders is not None:
        prev_tl, prev_vl = prev_loaders
        # persistent workers shutdown
        try: del prev_tl
        except Exception: pass
        try: del prev_vl
        except Exception: pass

    # SCCSynthIyDataset: 단일 (prebake, cache) 또는 multi (prebake-extra/i1-cache-extra)
    full_main = SCCSynthIyDataset(prebake_path=args.prebake, i1_cache_path=args.i1_cache,
                                    close_lead_meta_path=args.close_lead_meta,
                                    a2_cache_path=args.a2_cache,
                                    i2_cache_path=args.i2_cache,
                                    inlane_gt_cache_path=args.inlane_gt_cache,
                                    limit=args.limit)
    # cache backbone consistency check — multi-cache 가 다른 backbone 으로 만들어졌으면 warning
    def _check_cache_backbones(cache_paths):
        import json as _json
        bks = []
        for p in cache_paths:
            try:
                z = np.load(p, allow_pickle=True)
                m = _json.loads(str(z["meta"]))
                bks.append((p, m.get("seg_ckpt", "?"), m.get("seg_metrics", {})))
            except Exception: pass
        seg_ckpts = sorted(set(b[1] for b in bks))
        if len(seg_ckpts) > 1:
            print("⚠️ [cache-warning] multi-cache 가 다른 seg_backbone 으로 만들어짐 — token distribution 불일치 risk")
            for p, sc, met in bks:
                print(f"  {os.path.basename(p)}  ←  {sc}  {met}")
    all_caches = [args.i1_cache] + list(args.i1_cache_extra or [])
    _check_cache_backbones(all_caches)
    extras = []
    if args.prebake_extra:
        assert len(args.prebake_extra) == len(args.i1_cache_extra), \
            "--prebake-extra 와 --i1-cache-extra 갯수 동일 필요"
        # a2_cache_extra 길이 맞추기 (없으면 None 채움)
        a2_extras = list(args.a2_cache_extra) if args.a2_cache_extra else [None] * len(args.prebake_extra)
        if len(a2_extras) < len(args.prebake_extra):
            a2_extras = a2_extras + [None] * (len(args.prebake_extra) - len(a2_extras))
        i2_extras = list(args.i2_cache_extra) if args.i2_cache_extra else [None] * len(args.prebake_extra)
        if len(i2_extras) < len(args.prebake_extra):
            i2_extras = i2_extras + [None] * (len(args.prebake_extra) - len(i2_extras))
        ig_extras = list(args.inlane_gt_cache_extra) if args.inlane_gt_cache_extra else [None] * len(args.prebake_extra)
        if len(ig_extras) < len(args.prebake_extra):
            ig_extras = ig_extras + [None] * (len(args.prebake_extra) - len(ig_extras))
        # per-extra limit (없으면 args.limit fallback). 0 또는 음수면 무제한 (None).
        ex_limits = list(args.prebake_extra_limits) if args.prebake_extra_limits else [args.limit] * len(args.prebake_extra)
        if len(ex_limits) < len(args.prebake_extra):
            ex_limits = ex_limits + [args.limit] * (len(args.prebake_extra) - len(ex_limits))
        for pb_path, c_path, a2_path, i2_path, ig_path, lim in zip(args.prebake_extra, args.i1_cache_extra, a2_extras, i2_extras, ig_extras, ex_limits):
            lim_use = None if (lim is None or lim <= 0) else int(lim)
            extras.append(SCCSynthIyDataset(prebake_path=pb_path, i1_cache_path=c_path,
                                              close_lead_meta_path=args.close_lead_meta,
                                              a2_cache_path=a2_path,
                                              i2_cache_path=i2_path,
                                              inlane_gt_cache_path=ig_path,
                                              limit=lim_use))
    if extras:
        from torch.utils.data import ConcatDataset
        synth_full = ConcatDataset([full_main] + extras)
        # min_dist / valid_idx aggregate
        all_md = [full_main.min_dist[full_main.valid_idx]]
        for e in extras: all_md.append(e.min_dist[e.valid_idx])
        synth_full.min_dist = np.concatenate(all_md)
        # target aggregate (difficulty sampler 용)
        all_tgt = [full_main.target[full_main.valid_idx]]
        for e in extras: all_tgt.append(e.target[e.valid_idx])
        synth_full.target = np.concatenate(all_tgt, axis=0)
        synth_full.valid_idx = np.arange(len(synth_full), dtype=np.int64)
        synth_full.C_b = full_main.C_b
        print(f"[data] SCCSynth main={len(full_main)} + extras={[len(e) for e in extras]} = {len(synth_full)}")
        full = synth_full
    else:
        full = full_main
    full_orig_synth = full   # ci_aug 추가 전 synth-only 참조
    # controller_imitator augmentation 추가 (image=0 padded sample, no-lead 학습 강화)
    ci_aug_full = None
    if args.ci_aug_dirs:
        from src.datamodule.controller_imitator_aug_dataset import ControllerImitatorAugDataset
        from torch.utils.data import ConcatDataset
        ci_aug_full = ControllerImitatorAugDataset(
            dirs=args.ci_aug_dirs, C_b=int(full.C_b), fx=320.0,  # legacy controller_imitator fx
            limit=args.ci_aug_limit)
        full_orig = full
        full = ConcatDataset([full, ci_aug_full])
        # synth 의 min_dist (이미 valid 매핑됨) + ci_aug inf
        synth_md = full_orig.min_dist if isinstance(full_orig.min_dist, np.ndarray) else np.array(full_orig.min_dist)
        if hasattr(full_orig, 'valid_idx') and len(synth_md) > len(full_orig):
            # full_orig 가 SCCSynthIyDataset 단일 → valid_idx 매핑
            synth_md = synth_md[full_orig.valid_idx]
        n_total = len(full)
        ds_min_dist = np.full(n_total, np.inf, dtype=np.float32)
        ds_min_dist[:len(full_orig)] = synth_md[:len(full_orig)]
        full.valid_idx = np.arange(n_total, dtype=np.int64)
        full.min_dist  = ds_min_dist
        full.C_b = full_orig.C_b
        # target aggregate (ci_aug 의 target 은 별도 처리 — difficulty 계산 용)
        if hasattr(full_orig, 'target'):
            synth_tgt = full_orig.target  # (N_synth_orig, 40, 2)
            if hasattr(full_orig, 'valid_idx') and len(synth_tgt) > len(full_orig):
                synth_tgt = synth_tgt[full_orig.valid_idx]
            ds_target = np.zeros((n_total, 40, 2), dtype=np.float32)
            ds_target[:len(full_orig)] = synth_tgt[:len(full_orig)]
            # ci_aug target 추출 (image-less, cruise mostly)
            try:
                # ci_aug __getitem__ 호출 부담 — 대신 difficulty 평균 적용
                # ci_aug 는 image=0 padded real-log → cruise + 다양. easy 가정 (default 0.5)
                pass
            except: pass
            full.target = ds_target
        n_val = max(1, int(n_total * args.val_ratio))
        g = torch.Generator().manual_seed(args.seed)
        tr, va = random_split(full, [n_total - n_val, n_val], generator=g)
        print(f"[data] synth={len(full_orig)} + ci_aug={len(ci_aug_full)} = total={n_total}")
    else:
        n_total = len(full); n_val = max(1, int(n_total * args.val_ratio))
        g = torch.Generator().manual_seed(args.seed)
        tr, va = random_split(full, [n_total - n_val, n_val], generator=g)
    print(f"[data] total={n_total} train={n_total-n_val} val={n_val}")

    train_sampler = None
    # difficulty-aware weight: proxy = |steer|/scale + |ax|/scale (first-step target)
    # close-lead weight 와 max() 로 결합 (둘 다 만족 시 더 강한 쪽)
    use_difficulty = bool(getattr(args, 'difficulty_weight', False))
    if (args.close_lead_meta and args.close_weight > 1.0) or use_difficulty:
        valid_full_idx = full.valid_idx
        tr_subset_idx  = tr.indices
        tr_full_idx    = [int(valid_full_idx[i]) for i in tr_subset_idx]
        # close-lead weight
        w_close = np.ones(len(tr_full_idx), dtype=np.float64)
        if args.close_lead_meta and args.close_weight > 1.0:
            tr_dist = full.min_dist[tr_full_idx]
            w_close = np.where(tr_dist < args.close_thr, args.close_weight, 1.0)
            n_c = int((tr_dist < args.close_thr).sum())
            print(f"[sampler] close-lead weight={args.close_weight}  close/total={n_c}/{len(tr)}")
        # difficulty weight (target |steer|+|ax| 기반)
        w_diff = np.ones(len(tr_full_idx), dtype=np.float64)
        if use_difficulty and hasattr(full, 'target'):
            # full.target shape: (N_orig, 40, 2)
            tgt = full.target[tr_full_idx]   # (N_train, 40, 2)
            steer0 = np.abs(tgt[:, 0, 0])     # first-step |steer| (deg)
            ax0    = np.abs(tgt[:, 0, 1])     # first-step |ax|  (m/s²)
            proxy = steer0 / args.scale_steer + ax0 / args.scale_ax
            # bucket-based weight
            w_diff = np.where(proxy > 1.0, 4.0,         # extreme (top ~10%)
                     np.where(proxy > 0.5, 2.0,         # hard
                     np.where(proxy > 0.1, 1.0,         # normal
                              0.5)))                    # easy (cruise)
            ne, nh, ned, nez = (proxy>1.0).sum(), ((proxy>0.5)&(proxy<=1.0)).sum(), \
                               ((proxy>0.1)&(proxy<=0.5)).sum(), (proxy<=0.1).sum()
            print(f"[sampler] difficulty: extreme={ne} hard={nh} normal={ned} easy={nez}")
        elif use_difficulty:
            print(f"[sampler] difficulty SKIPPED — full.target not set (ci_aug ConcatDataset issue?)")
        w = np.maximum(w_close, w_diff)
        train_sampler = WeightedRandomSampler(weights=w.tolist(), num_samples=len(tr),
                                                replacement=True)
        print(f"[sampler] combined max(close, difficulty), mean weight={w.mean():.2f} max={w.max():.1f}")
    va_close_idx = []; va_far_idx = []
    if args.close_lead_meta:
        va_full_idx = [int(full.valid_idx[i]) for i in va.indices]
        for vi, fi in enumerate(va_full_idx):
            (va_close_idx if full.min_dist[fi] < args.close_thr else va_far_idx).append(vi)
        print(f"[val-split] close={len(va_close_idx)} far={len(va_far_idx)}")

    tl = DataLoader(tr, batch_size=args.batch_size,
                     shuffle=(train_sampler is None), sampler=train_sampler,
                     num_workers=args.workers, pin_memory=True, collate_fn=collate_iy,
                     drop_last=True, persistent_workers=args.workers > 0)
    vl = DataLoader(va, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True, collate_fn=collate_iy,
                    persistent_workers=args.workers > 0)
    return full, tr, va, tl, vl, train_sampler, va_close_idx, va_far_idx


def main():
    import numpy as _np_main  # local 재import (다른 곳 np assign 회피)
    args = get_args()
    random.seed(args.seed); _np_main.random.seed(args.seed); torch.manual_seed(args.seed)
    out_dir = f"{args.out_dir}_{args.name}"
    os.makedirs(out_dir, exist_ok=True)

    full, tr, va, tl, vl, train_sampler, va_close_idx, va_far_idx = _build_dataset_and_loaders(args)
    if args.c_b in (0, None):
        args.c_b = int(full.C_b)
        print(f"[auto] C_b={args.c_b} (from cache)")

    model = IyModel(C_b=args.c_b, d_model=args.d_model, nhead=args.nhead,
                     num_layers=args.num_layers, fx=args.fx,
                     use_obj_gate=args.use_obj_gate,
                     use_side_branch=args.use_side_branch,
                     side_num_layers=args.side_num_layers,
                     side_dim_feedforward=args.side_dim_feedforward,
                     side_ax_only=args.side_ax_only,
                     use_a2_feat=args.use_a2_feat,
                     use_a2_late=args.use_a2_late,
                     main_no_image=args.main_no_image,
                     side_cross_attn=args.side_cross_attn,
                     side_min_arb=args.side_min_arb).to(args.device)
    if args.professor_ckpt:
        loaded, skipped = model.load_professor(args.professor_ckpt)
        print(f"[Professor] loaded={len(loaded)} skipped={len(skipped)}")
    if args.resume_from:
        sd = torch.load(args.resume_from, map_location=args.device, weights_only=False)
        if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
        # shape-mismatch 키 자동 제외 (strict=False 는 missing 만 허용, size mismatch 는 error)
        own = model.state_dict()
        sd_ok = {}; sd_skip = []
        for k, v in sd.items():
            if k in own and own[k].shape == v.shape:
                sd_ok[k] = v
            else:
                sd_skip.append((k, tuple(v.shape), tuple(own[k].shape) if k in own else None))
        m, u = model.load_state_dict(sd_ok, strict=False)
        print(f"[resume] {args.resume_from}")
        print(f"  loaded={len(sd_ok)}  missing={len(m)}  unexpected={len(u)}  shape-skip={len(sd_skip)}")
        if m: print(f"  missing(sample): {m[:5]}")
        if u: print(f"  unexpected(sample): {u[:5]}")
        if sd_skip: print(f"  shape-skipped: {sd_skip[:5]}")
        if args.reset_side_alpha and hasattr(model, 'side_alpha'):
            model.side_alpha.data.zero_()
            print("[resume] side_alpha reset to 0 (identity start)")

    n_params = sum(p.numel() for p in model.parameters())
    n_side = sum(p.numel() for n, p in model.named_parameters() if n.startswith("side_"))
    print(f"[model] params={n_params/1e6:.2f} M (side={n_side/1e6:.2f} M)  fx={args.fx}  side_branch={args.use_side_branch}")

    # param grouping — main(prof/other) vs side/a2 (phase1 에도 학습되는 새 모듈)
    prof_keys = ["traj_embed", "state_embed", "pos_encoder", "transformer_encoder", "fc_out"]
    side_keys = ["side_transformer", "side_fc", "side_alpha", "a2_proj", "a2_late_proj",
                 "side_cross", "side_cross_ln", "side_fc_cross", "side_alpha_cross"]  # always-trainable
    prof_params = []; other_params = []; side_params = []
    for n, p in model.named_parameters():
        if any(n.startswith(k) for k in side_keys):
            side_params.append(p)
        elif any(n.startswith(k) for k in prof_keys):
            prof_params.append(p)
        else:
            other_params.append(p)
    side_lr = args.side_lr if args.side_lr is not None else args.lr
    main_lr_phase2 = args.main_lr_phase2 if args.main_lr_phase2 is not None else args.lr_prof / 20.0
    # has_aux: side_branch / side_cross_attn / use_a2_feat / use_a2_late 중 하나라도 활성
    _has_aux_opt = bool(args.use_side_branch or args.side_cross_attn or args.use_a2_feat or args.use_a2_late)
    pg = [
        {"params": prof_params,  "lr": args.lr_prof},   # group 0: main prof transformer/fc_out
        {"params": other_params, "lr": args.lr},        # group 1: main fusion/img_proj/...
    ]
    if _has_aux_opt and side_params:
        pg.append({"params": side_params, "lr": side_lr})  # group 2: side + α + a2_proj + cross
    optim = torch.optim.AdamW(pg, weight_decay=args.weight_decay)
    # aux 모듈 사용 시 scheduler 비활성화 (phase 전환과 충돌, lr 상수 유지)
    if _has_aux_opt:
        sched = None
    else:
        # CosineAnnealingWarmRestarts + 매 cycle eta_max cosine decay (사용자 요청)
        sched_T0 = max(1, args.epochs // 5)
        n_cycles = max(1, args.epochs // sched_T0)
        sched = _CosineDecayingMaxRestarts(optim, T_0=sched_T0, n_cycles=n_cycles, eta_min=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    log_path = os.path.join(out_dir, "train.log")
    log_f = open(log_path, "w")
    def logp(m): print(m, flush=True); log_f.write(m + "\n"); log_f.flush()
    logp(f"args={vars(args)}  prof={len(prof_params)} other={len(other_params)}")

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                                 name=args.wandb_run or args.name, config=vars(args), dir=out_dir)
        logp(f"[wandb] init {args.wandb_project} {wandb_run.name}")

    best_val = float("inf")
    best_close = float("inf")    # close-lead val (safety-critical)
    best_far   = float("inf")    # far / cruise val
    best_ca    = float("inf")    # close-approach (근거리 접근 lead) ax MAE — brake-critical
    def _prebake_mtime_sum():
        paths = [args.prebake] + list(args.prebake_extra or [])
        s = 0.0
        for p in paths:
            try: s += os.path.getmtime(p)
            except FileNotFoundError: pass
        return s
    last_prebake_mtime = _prebake_mtime_sum() if args.reload_on_change else None
    # side/aux 모듈 2-phase 학습: phase1 main frozen → phase2 joint (low main lr)
    # side_branch / side_cross_attn / use_a2_feat / use_a2_late 중 하나라도 켜져 있으면 phase 모드
    has_aux = bool(args.use_side_branch or args.side_cross_attn or args.use_a2_feat or args.use_a2_late)
    side_param_names = set()
    if has_aux:
        for n, _ in model.named_parameters():
            if any(n.startswith(k) for k in side_keys):
                side_param_names.add(n)

    def _set_phase(ep_idx):
        if not has_aux or args.phase1_epochs <= 0:
            return None
        phase = 1 if ep_idx < args.phase1_epochs else 2
        main_train = (phase == 2)
        for n, p in model.named_parameters():
            if n in side_param_names:
                p.requires_grad = True
            else:
                p.requires_grad = main_train
        # phase2 진입 시 main group lr 낮춤 (group 0=prof, 1=other)
        if phase == 2:
            optim.param_groups[0]["lr"] = main_lr_phase2
            optim.param_groups[1]["lr"] = main_lr_phase2
        return phase

    for ep in range(args.epochs):
        cur_phase = _set_phase(ep)
        if cur_phase is not None:
            alpha_v = float(model.side_alpha.detach().item()) if hasattr(model, 'side_alpha') else \
                      (float(model.side_alpha_cross.detach().item()) if hasattr(model, 'side_alpha_cross') else 0.0)
            logp(f"[phase{cur_phase}] ep={ep+1} main_train={cur_phase==2} α={alpha_v:+.4f} "
                 f"lr_prof={optim.param_groups[0]['lr']:.2e} lr_other={optim.param_groups[1]['lr']:.2e} "
                 f"lr_side={optim.param_groups[2]['lr']:.2e}" if len(optim.param_groups) > 2 else "")
        if args.reload_on_change:
            cur_mtime = _prebake_mtime_sum()
            if cur_mtime != last_prebake_mtime:
                logp(f"[stream-reload] ep={ep}: prebake mtime changed ({cur_mtime - (last_prebake_mtime or 0):.0f}s newer) → rebuild dataset")
                last_prebake_mtime = cur_mtime
                full, tr, va, tl, vl, train_sampler, va_close_idx, va_far_idx = \
                    _build_dataset_and_loaders(args, prev_loaders=(tl, vl))
                logp(f"[stream-reload] new total={len(full)} train={len(tr)} val={len(va)}")
        model.train(); t0 = time.time()
        rl = rs = ra = 0.0; rn = 0
        for it, batch in enumerate(tl):
            b = {k: (v.to(args.device, non_blocking=True) if torch.is_tensor(v) else v)
                  for k, v in batch.items()}
            # ablation: traj zero after K step
            if args.traj_zero_after is not None and 'traj_40' in b:
                b['traj_40'] = b['traj_40'].clone()
                b['traj_40'][:, args.traj_zero_after:, :] = 0.0
            # ablation: target_mask 첫 step 만
            if args.target_first_only and 'target_mask' in b:
                b['target_mask'] = b['target_mask'].clone()
                b['target_mask'][:, 1:] = False
            # TTC-based brake floor augment (target ax 첫 K step)
            if args.ttc_augment:
                _apply_ttc_augment(b, args)
            optim.zero_grad(set_to_none=True)
            # per-sample loss weight from close-lead
            sw = None
            if args.close_lead_meta and args.close_loss_mult > 1.0:
                sw = torch.where(b["min_dist"] < args.close_thr,
                                  torch.full_like(b["min_dist"], args.close_loss_mult),
                                  torch.ones_like(b["min_dist"]))
            with torch.amp.autocast("cuda", enabled=args.amp):
                pred, obj_logit = model(b)
                total, parts = split_losses(pred, b["target"], b["target_mask"], args, sample_weight=sw)
                if args.use_obj_gate and obj_logit is not None and "obj_target" in b:
                    obj_bce = torch.nn.functional.binary_cross_entropy_with_logits(
                        obj_logit, b["obj_target"])
                    total = total + args.obj_loss_w * obj_bce
                    parts["obj_bce"] = obj_bce.detach()
                # min-arb: b_side(절대 brake ceiling) 직접 supervision → target ax(step0, ttc-augmented).
                # min 이 main 을 골라도 gradient 흐름 → dead-start 회피.
                if args.side_min_arb and getattr(model, "last_b_side", None) is not None:
                    b_side = model.last_b_side.squeeze(-1)          # (B,)
                    tgt0 = b["target"][:, 0, 1]                      # (B,) m/s² (ttc-aug 후)
                    m0 = b["target_mask"][:, 0].bool() if "target_mask" in b else torch.ones_like(tgt0, dtype=torch.bool)
                    if m0.any():
                        bside_loss = F.l1_loss(b_side[m0], tgt0[m0])
                        total = total + args.side_min_arb_w * bside_loss
                        parts["bside"] = bside_loss.detach()
            # loss NaN/Inf check — sample-level outlier 가 trigger 시 batch skip
            if not torch.isfinite(total):
                optim.zero_grad(set_to_none=True)
                if (it + 1) % args.log_every == 0:
                    logp(f"[ep {ep+1:02d} | {it+1:4d}] SKIP non-finite loss")
                continue
            scaler.scale(total).backward()
            # grad clip (max_norm 10 — 너무 작으면 학습 step 작아져 수렴 느림)
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            # grad NaN/Inf 시 step skip (scaler 가 자동 처리하지만 fp32 도 안전망)
            grad_ok = True
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    grad_ok = False; break
            if not grad_ok:
                optim.zero_grad(set_to_none=True)
                continue
            scaler.step(optim); scaler.update()
            B = b["tok_curr"].size(0)
            rl += total.item() * B
            if parts:
                rs += parts["l1_steer"].item() * B; ra += parts["l1_ax"].item() * B
            rn += B
            if (it + 1) % args.log_every == 0:
                logp(f"[ep {ep+1:02d} | {it+1:4d}/{len(tl)}] total={rl/rn:.4f}  "
                     f"l1_steer={rs/rn:.4f} l1_ax={ra/rn:.4f}  lr_prof={optim.param_groups[0]['lr']:.2e}")
                if wandb_run is not None:
                    # step-level (granular) — running avg train loss + LR + grad norm
                    grad_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            grad_norm += p.grad.detach().norm(2).item() ** 2
                    grad_norm = grad_norm ** 0.5
                    global_step = ep * len(tl) + it + 1
                    wandb_run.log({
                        "step/train_loss": rl/rn,
                        "step/l1_steer": rs/rn,
                        "step/l1_ax": ra/rn,
                        "step/lr_prof": optim.param_groups[0]['lr'],
                        "step/lr_other": optim.param_groups[1]['lr'],
                        "step/grad_norm": grad_norm,
                        "step/epoch_progress": (ep + it/len(tl)),
                    }, step=global_step)
        if sched is not None:
            sched.step()
        train_loss = rl / max(1, rn)

        model.eval()
        vl_total = vs = vax = 0.0; vn = 0
        # split metrics: close vs far
        sum_st_c = sum_ax_c = sum_st_f = sum_ax_f = 0.0
        n_c = n_f = 0
        # close-approach (근거리 접근 in-lane lead) — brake-critical val
        sum_ax_ca = 0.0; n_ca = 0
        brake_dir_ok = 0; n_brake_ca = 0       # target brake 시 pred 부호 일치
        sum_axraw_ca = 0.0                      # pred ax 평균 (m/s², 가속/감속 방향 판단)
        with torch.no_grad():
            for batch in vl:
                b = {k: (v.to(args.device, non_blocking=True) if torch.is_tensor(v) else v)
                      for k, v in batch.items()}
                # ablation 동일 적용
                if args.traj_zero_after is not None and 'traj_40' in b:
                    b['traj_40'] = b['traj_40'].clone()
                    b['traj_40'][:, args.traj_zero_after:, :] = 0.0
                if args.target_first_only and 'target_mask' in b:
                    b['target_mask'] = b['target_mask'].clone()
                    b['target_mask'][:, 1:] = False
                if args.ttc_augment:
                    _apply_ttc_augment(b, args)
                with torch.amp.autocast("cuda", enabled=args.amp):
                    pred, obj_logit = model(b)
                    total, parts = split_losses(pred, b["target"], b["target_mask"], args)
                B = b["tok_curr"].size(0)
                vl_total += total.item()*B; vn += B
                if parts:
                    vs += parts["l1_steer"].item()*B; vax += parts["l1_ax"].item()*B
                # per-sample split metrics
                scales = torch.tensor([args.scale_steer, args.scale_ax], device=pred.device)
                diff = (pred - b["target"]) / scales
                mask = b["target_mask"]
                denom = mask.sum(dim=1).clamp(min=1).float()
                per_st = (diff[..., 0].abs() * mask).sum(dim=1) / denom
                per_ax = (diff[..., 1].abs() * mask).sum(dim=1) / denom
                close = (b["min_dist"] < args.close_thr)
                far   = ~close
                if close.any():
                    sum_st_c += per_st[close].sum().item(); sum_ax_c += per_ax[close].sum().item()
                    n_c += int(close.sum().item())
                if far.any():
                    sum_st_f += per_st[far].sum().item(); sum_ax_f += per_ax[far].sum().item()
                    n_f += int(far.sum().item())
                # close-approach (근거리 접근 in-lane lead): inlane_dist<20 AND closing>1
                if 'inlane_dist' in b and 'inlane_closing' in b:
                    ca = (b['inlane_dist'] < 20.0) & (b['inlane_closing'] > 1.0) & torch.isfinite(b['inlane_dist'])
                    if ca.any():
                        # step0 ax — pred/target 은 이미 physical m/s² (scale 은 loss 정규화용일 뿐)
                        ax_pred0 = pred[:, 0, 1]; ax_tgt0 = b["target"][:, 0, 1]
                        sum_ax_ca += (ax_pred0[ca] - ax_tgt0[ca]).abs().sum().item()
                        sum_axraw_ca += ax_pred0[ca].sum().item()
                        n_ca += int(ca.sum().item())
                        # brake 방향 정확도: target brake(<-1.5) 일 때 pred 도 감속(<0)
                        bm = ca & (ax_tgt0 < -1.5)
                        if bm.any():
                            brake_dir_ok += int(((ax_pred0[bm] < 0)).sum().item())
                            n_brake_ca += int(bm.sum().item())
        val_loss = vl_total / max(1, vn)
        dt = time.time() - t0
        msg = (f"[ep {ep+1:02d}] train={train_loss:.4f} val={val_loss:.4f} "
               f"(steer={vs/vn*args.scale_steer:.2f}° ax={vax/vn*args.scale_ax:.2f}m/s²) time={dt:.1f}s")
        if n_c > 0 and n_f > 0:
            msg += (f"  | close(n={n_c}) st={sum_st_c/n_c*args.scale_steer:.2f}° ax={sum_ax_c/n_c*args.scale_ax:.2f}"
                    f"  far(n={n_f}) st={sum_st_f/n_f*args.scale_steer:.2f}° ax={sum_ax_f/n_f*args.scale_ax:.2f}")
        if n_ca > 0:
            bdir = (brake_dir_ok / n_brake_ca * 100) if n_brake_ca > 0 else float('nan')
            msg += (f"  | ⭐close-approach(n={n_ca}) ax_err={sum_ax_ca/n_ca:.2f}m/s² "
                    f"pred_ax={sum_axraw_ca/n_ca:+.2f} brake_dir={bdir:.0f}%(n={n_brake_ca})")
        logp(msg)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(out_dir, "best_loss.pt"))
            logp(f"  -> saved best ({val_loss:.4f})")
        # val split-aware best 저장 (close = safety critical)
        if n_c > 0:
            val_close_loss = (sum_st_c + sum_ax_c) / n_c   # close-only combined L1 normalized
            if val_close_loss < best_close:
                best_close = val_close_loss
                torch.save(model.state_dict(), os.path.join(out_dir, "best_close.pt"))
                logp(f"  -> saved best_close ({val_close_loss:.4f}) n={n_c}")
        if n_f > 0:
            val_far_loss = (sum_st_f + sum_ax_f) / n_f
            if val_far_loss < best_far:
                best_far = val_far_loss
                torch.save(model.state_dict(), os.path.join(out_dir, "best_far.pt"))
                logp(f"  -> saved best_far ({val_far_loss:.4f}) n={n_f}")
        # close-approach (brake-critical) best — ax 오차 기준
        if n_ca > 30:
            val_ca_ax = sum_ax_ca / n_ca       # close-approach ax MAE (m/s²)
            if val_ca_ax < best_ca:
                best_ca = val_ca_ax
                torch.save(model.state_dict(), os.path.join(out_dir, "best_close_approach.pt"))
                logp(f"  -> saved best_close_approach (ax_err={val_ca_ax:.3f}) n={n_ca}")
        if wandb_run is not None:
            log = {"train/loss": train_loss, "val/loss": val_loss,
                   "val/l1_steer_deg": vs/vn*args.scale_steer,
                   "val/l1_ax_mps2":   vax/vn*args.scale_ax,
                   "lr/prof": optim.param_groups[0]['lr'], "lr/other": optim.param_groups[1]['lr'],
                   "epoch": ep + 1, "epoch_time_s": dt}
            if hasattr(model, 'side_alpha'):
                log["side/alpha"] = float(model.side_alpha.detach().item())
            if hasattr(model, 'side_alpha_cross'):
                log["side/alpha_cross"] = float(model.side_alpha_cross.detach().item())
            if len(optim.param_groups) > 2:
                log["side/lr"] = optim.param_groups[2]['lr']
            if cur_phase is not None:
                log["side/phase"] = cur_phase
            if n_c > 0:
                log["val_close/l1_steer_deg"] = sum_st_c/n_c*args.scale_steer
                log["val_close/l1_ax_mps2"]   = sum_ax_c/n_c*args.scale_ax
                log["val_close/n"] = n_c
            if n_f > 0:
                log["val_far/l1_steer_deg"] = sum_st_f/n_f*args.scale_steer
                log["val_far/l1_ax_mps2"]   = sum_ax_f/n_f*args.scale_ax
                log["val_far/n"] = n_f
            if n_ca > 0:
                log["val_close_approach/ax_err_mps2"] = sum_ax_ca/n_ca
                log["val_close_approach/pred_ax_mps2"] = sum_axraw_ca/n_ca
                log["val_close_approach/n"] = n_ca
                if n_brake_ca > 0:
                    log["val_close_approach/brake_dir_acc"] = brake_dir_ok/n_brake_ca*100
                    log["val_close_approach/n_brake"] = n_brake_ca
            # ===== ML 고수 시각화: GT vs pred scatter + error histogram =====
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import wandb
                # sample 일부 val batch 로 GT/pred 수집
                samples_gt = []; samples_pred = []
                model.eval()
                with torch.no_grad():
                    for vi, batch in enumerate(vl):
                        if vi >= 8: break  # 빠른 sampling
                        b = {k: (v.to(args.device, non_blocking=True) if torch.is_tensor(v) else v)
                              for k, v in batch.items()}
                        pred, _ = model(b)
                        samples_gt.append((b["target"]*torch.tensor([args.scale_steer, args.scale_ax],
                                                                     device=pred.device)).cpu().numpy())
                        samples_pred.append((pred*torch.tensor([args.scale_steer, args.scale_ax],
                                                               device=pred.device)).cpu().numpy())
                import numpy as _np_local
                gt = _np_local.concatenate(samples_gt, axis=0)[..., 0, :]   # (N, 2) first step
                pr = _np_local.concatenate(samples_pred, axis=0)[..., 0, :]
                # scatter
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                axes[0].scatter(gt[:, 0], pr[:, 0], s=4, alpha=0.5)
                lim = max(abs(gt[:, 0]).max(), abs(pr[:, 0]).max(), 1)
                axes[0].plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3)
                axes[0].set_xlabel("GT steer (deg)"); axes[0].set_ylabel("Pred steer (deg)")
                axes[0].set_title(f"ep{ep+1} steer (n={len(gt)})")
                axes[0].grid(alpha=0.3)
                axes[1].scatter(gt[:, 1], pr[:, 1], s=4, alpha=0.5, color='tab:red')
                lim2 = max(abs(gt[:, 1]).max(), abs(pr[:, 1]).max(), 1)
                axes[1].plot([-lim2, lim2], [-lim2, lim2], 'k--', alpha=0.3)
                axes[1].set_xlabel("GT ax (m/s²)"); axes[1].set_ylabel("Pred ax (m/s²)")
                axes[1].set_title(f"ep{ep+1} ax")
                axes[1].grid(alpha=0.3)
                plt.tight_layout()
                log["viz/gt_vs_pred"] = wandb.Image(fig)
                plt.close(fig)
                # error histogram
                err_st = pr[:, 0] - gt[:, 0]; err_ax = pr[:, 1] - gt[:, 1]
                fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
                axes2[0].hist(err_st, bins=60, color='steelblue', alpha=0.7)
                axes2[0].set_xlabel("steer error (deg)"); axes2[0].set_title(f"ep{ep+1} steer err  μ={err_st.mean():.2f} σ={err_st.std():.2f}")
                axes2[0].axvline(0, color='k', alpha=0.4); axes2[0].grid(alpha=0.3)
                axes2[1].hist(err_ax, bins=60, color='tomato', alpha=0.7)
                axes2[1].set_xlabel("ax error (m/s²)"); axes2[1].set_title(f"ep{ep+1} ax err  μ={err_ax.mean():.2f} σ={err_ax.std():.2f}")
                axes2[1].axvline(0, color='k', alpha=0.4); axes2[1].grid(alpha=0.3)
                plt.tight_layout()
                log["viz/error_hist"] = wandb.Image(fig2)
                plt.close(fig2)
                # 추가 metric
                log["val/err_steer_mean"] = float(err_st.mean())
                log["val/err_steer_std"]  = float(err_st.std())
                log["val/err_ax_mean"]    = float(err_ax.mean())
                log["val/err_ax_std"]     = float(err_ax.std())
                log["val/r2_steer"]       = float(1 - ((gt[:, 0]-pr[:, 0])**2).sum() /
                                                    max(1e-9, ((gt[:, 0]-gt[:, 0].mean())**2).sum()))
                log["val/r2_ax"]          = float(1 - ((gt[:, 1]-pr[:, 1])**2).sum() /
                                                    max(1e-9, ((gt[:, 1]-gt[:, 1].mean())**2).sum()))
            except Exception as e:
                logp(f"  [wandb viz error] {e}")
            wandb_run.log(log, step=ep + 1)

    logp(f"\n[done] best_val={best_val:.4f}")
    log_f.close()
    if wandb_run is not None: wandb_run.finish()


if __name__ == "__main__":
    main()
