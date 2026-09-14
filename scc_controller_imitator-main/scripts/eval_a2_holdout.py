"""a_2 head + fx640_v3 backbone 의 held-out 평가.

held-out: 학습에 안 들어간 policy 의 sample (예: spd_nolead_0520/0521).
metric:
  obj (binary per-token): confusion matrix, accuracy, precision/recall/F1
  dist (regression per-token, obj=True only): MAE, RMSE
  vL_rel (regression): MAE, RMSE
  token-별 분포: 어느 angular bin 에서 잘하는지

사용:
  python scripts/eval_a2_holdout.py \
    --ckpt outputs_multi/a2_fx640_v3_local/best_loss.pt \
    --meta data/meta_spd_nolead_0520.json data/meta_spd_nolead_0521.json \
    --a2-dir data/aux_targets_spd_nolead_0520 data/aux_targets_spd_nolead_0521 \
    --raw-base ~/AILabSSD/02_Data/57_A1_E2E/00_raw \
    --limit 2000 \
    --out /tmp/a2_holdout_eval.json
"""
import argparse, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.model.image_models.a2_model import A2Model
from src.datamodule.a2_pair_dataset import A2PairDataset, _img_to_tensor, _load_roi, _yaw_shift, _parse_ego


def build_dataset_for_eval(meta_path, a2_dir, raw_base, prev_dtick=20):
    """A2PairDataset 와 동일 로직이지만 단일 meta+a2_dir 받음."""
    return A2PairDataset(
        meta_path=meta_path, a2_dir=a2_dir, raw_base=raw_base,
        prev_dtick=prev_dtick, require_obj=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--meta", nargs="+", required=True)
    ap.add_argument("--a2-dir", nargs="+", required=True)
    ap.add_argument("--raw-base", default="/home/ailab-12/AILabSSD/02_Data/57_A1_E2E/00_raw")
    ap.add_argument("--prev-dtick", type=int, default=20)
    ap.add_argument("--limit", type=int, default=2000, help="평가 sample 수 (전체는 0)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--scale-v", type=float, default=20.0)
    ap.add_argument("--scale-dist", type=float, default=40.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="/tmp/a2_holdout_eval.json")
    args = ap.parse_args()

    assert len(args.meta) == len(args.a2_dir), "meta 와 a2-dir 갯수 일치 필요"

    # 모델 로드
    print(f"[load] ckpt={args.ckpt}", flush=True)
    model = A2Model().to(args.device)
    sd = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd)
    model.eval()

    # 데이터셋 (여러 meta concat)
    from torch.utils.data import ConcatDataset
    sets = []
    for m, a in zip(args.meta, args.a2_dir):
        print(f"[ds] meta={m} a2={a}", flush=True)
        ds = build_dataset_for_eval(m, a, args.raw_base, args.prev_dtick)
        print(f"   N={len(ds)}", flush=True)
        sets.append(ds)
    ds_all = ConcatDataset(sets)
    print(f"[ds] total N={len(ds_all)}", flush=True)

    # subset
    N = len(ds_all)
    if args.limit > 0 and args.limit < N:
        rng = np.random.RandomState(0)
        idx = rng.choice(N, args.limit, replace=False).tolist()
        from torch.utils.data import Subset
        ds_all = Subset(ds_all, idx)
        print(f"[ds] subset N={len(ds_all)}", flush=True)

    loader = DataLoader(ds_all, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    scales = np.array([args.scale_v]*4 + [1.0, args.scale_dist], dtype=np.float32)  # (6,)

    all_pred_logits = []
    all_gt = []
    t0 = time.time()
    seen = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            img_c = batch["image_curr"].to(args.device, non_blocking=True)
            img_p = batch["image_prev_rot"].to(args.device, non_blocking=True)
            ego   = batch["ego_motion"].to(args.device, non_blocking=True)
            target = batch["target"].cpu().numpy()  # (B, 40, 7)
            with torch.amp.autocast("cuda", enabled=True):
                logits = model(img_c, img_p, ego)  # (B, 40, 7)
            all_pred_logits.append(logits.float().cpu().numpy())
            all_gt.append(target)
            seen += target.shape[0]
            if i % 10 == 0:
                dt = time.time() - t0
                print(f"  [{seen}/{len(ds_all)}] rate={seen/max(dt,0.01):.1f} samples/s", flush=True)
    print(f"[done] {seen} samples in {time.time()-t0:.1f}s", flush=True)

    pred = np.concatenate(all_pred_logits, axis=0)  # (N, 40, 7)
    gt   = np.concatenate(all_gt, axis=0)
    print(f"pred shape={pred.shape}, gt shape={gt.shape}", flush=True)

    # obj 분류 metric
    pred_obj_logit = pred[..., 0]                  # (N, 40)
    pred_obj_prob  = 1.0 / (1.0 + np.exp(-pred_obj_logit))
    pred_obj       = pred_obj_prob > 0.5
    gt_obj         = gt[..., 0] > 0.5

    tp = int((pred_obj & gt_obj).sum())
    fp = int((pred_obj & ~gt_obj).sum())
    fn = int((~pred_obj & gt_obj).sum())
    tn = int((~pred_obj & ~gt_obj).sum())
    total = tp+fp+fn+tn
    acc  = (tp+tn) / total if total else 0
    prec = tp / (tp+fp) if (tp+fp) else 0
    rec  = tp / (tp+fn) if (tp+fn) else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0
    spec = tn / (tn+fp) if (tn+fp) else 0  # specificity (TNR)
    gt_pos_rate = (tp+fn) / total if total else 0
    print(f"\n=== obj presence (per-token binary) ===")
    print(f"  Confusion matrix:")
    print(f"    TP={tp:7d}  FN={fn:7d}    (gt+: {tp+fn})")
    print(f"    FP={fp:7d}  TN={tn:7d}    (gt-: {fp+tn})")
    print(f"  GT positive rate: {gt_pos_rate*100:.2f}%")
    print(f"  Accuracy:    {acc*100:.2f}%")
    print(f"  Precision:   {prec*100:.2f}%")
    print(f"  Recall(TPR): {rec*100:.2f}%")
    print(f"  Specificity: {spec*100:.2f}% (TN/all neg)")
    print(f"  F1:          {f1*100:.2f}%")

    # 회귀 metric (obj=True 위치 만)
    pred_reg = pred[..., 1:7] * scales            # (N, 40, 6) — denorm
    gt_reg   = gt[..., 1:7]                         # GT (a2 GT 는 normalize 됐는지 확인)
    # a2_pair_dataset 의 target 은 이미 raw scale 인지 확인 — 실험적으로 dist range 보고 결정
    mask = gt_obj                                   # (N, 40)
    print(f"\n=== 회귀 (obj=True token만) ===")
    print(f"  mask sum: {int(mask.sum())} tokens")
    if mask.sum() > 0:
        # dist (ch 6 of original = ch 5 of reg)
        d_pred = pred_reg[..., 5][mask]; d_gt = gt_reg[..., 5][mask]
        print(f"  GT dist  range=[{d_gt.min():.2f},{d_gt.max():.2f}] mean={d_gt.mean():.2f}")
        print(f"  pred dist range=[{d_pred.min():.2f},{d_pred.max():.2f}] mean={d_pred.mean():.2f}")
        # 만약 GT 가 normalize (0~1 정도) 면 scale 매칭 필요
        if d_gt.max() < 2.0:  # normalized
            d_gt_real = d_gt * args.scale_dist
            print(f"  → GT normalized 추정 → denorm: range=[{d_gt_real.min():.2f},{d_gt_real.max():.2f}]")
            d_gt = d_gt_real
        d_mae = np.abs(d_pred - d_gt).mean()
        d_rmse = np.sqrt(((d_pred - d_gt)**2).mean())
        print(f"  dist  MAE={d_mae:.3f}m  RMSE={d_rmse:.3f}m")
        # vL_rel (ch 3, idx 2 of reg)
        v_pred = pred_reg[..., 2][mask]; v_gt = gt_reg[..., 2][mask]
        if abs(v_gt.max()) < 2.0:
            v_gt = v_gt * args.scale_v
        v_mae = np.abs(v_pred - v_gt).mean()
        v_rmse = np.sqrt(((v_pred - v_gt)**2).mean())
        print(f"  vL_rel MAE={v_mae:.3f}m/s  RMSE={v_rmse:.3f}m/s")

    # token-별 accuracy
    print(f"\n=== token-별 obj accuracy ===")
    per_tok_acc = ((pred_obj == gt_obj).mean(axis=0))  # (40,)
    per_tok_gt_rate = gt_obj.mean(axis=0)
    for t in range(0, 40, 4):
        bar = '█' * int(per_tok_acc[t] * 40)
        print(f"  tok[{t:2d}]: acc={per_tok_acc[t]*100:5.1f}% gt_rate={per_tok_gt_rate[t]*100:5.1f}%  {bar}")

    result = {
        'ckpt': args.ckpt,
        'meta': args.meta,
        'N': int(seen),
        'obj': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                'accuracy': acc, 'precision': prec, 'recall': rec, 'specificity': spec, 'f1': f1,
                'gt_positive_rate': gt_pos_rate},
        'per_token_acc': per_tok_acc.tolist(),
        'per_token_gt_rate': per_tok_gt_rate.tolist(),
    }
    if mask.sum() > 0:
        result['dist'] = {'mae': float(d_mae), 'rmse': float(d_rmse),
                          'gt_range': [float(d_gt.min()), float(d_gt.max())]}
        result['vL_rel'] = {'mae': float(v_mae), 'rmse': float(v_rmse)}

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
