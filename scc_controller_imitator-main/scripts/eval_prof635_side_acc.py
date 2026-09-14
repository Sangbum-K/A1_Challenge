"""f_2(Prof635Side) ACC-modulation 평가 — closing bucket 별 brake delta.

검증 질문: f_2 의 제동 delta 가 (주입된) closing 으로 modulate 되는가?
  receding(closing<0) → 무제동(delta≈0)
  approach弱(0<closing<2) → 약제동
  approach强(closing>4) → 강제동  (monotonic)
distbrake_v1(거리만, closing 비주입) 대비 — 먼 거리 高-closing 접근에서 더 일찍/세게 제동하나.

모든 모델 동일 val-split(seed=0, ratio 0.05) holdout 에서 평가. vrel 주입 모델은
주입 cache(pred_vrel_{ds} / inlane_gt_{ds})를, distbrake 는 비주입(gt_vrel 없음).

사용:
  python scripts/eval_prof635_side_acc.py \
    --model pred outputs_multi/prof635_side_predvrel/best.pt pred \
    --model gt   outputs_multi/prof635_side_gtvrel_match/best.pt gt \
    --model distbrake outputs_multi/prof635_side_distbrake_v1/best.pt none distbrake
"""
import argparse, os, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import ConcatDataset, random_split, DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.image_models.prof635_side_model import Prof635SideModel
from src.datamodule.prof635_dataset import Prof635Dataset

D = os.environ.get("SCC_DATA_DIR") or "/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data"
DATASETS = {
    "sangam_lead_0520": "f1_prebake_iy_sangam_lead_0520_v1_brakeaug_prev20.npz",
    "sangam_lead_0521": "f1_prebake_iy_sangam_lead_0521_v1_brakeaug_prev20.npz",
    "sangam_pertO_v2":  "f1_prebake_iy_sangam_pertO_v2_prev20_fixed.npz",
    "sangam_pertX_v2":  "f1_prebake_iy_sangam_pertX_v2_prev20_fixed.npz",
    "DriverB_new":      "f1_prebake_iy_DriverB_new_logged_fixed_prev20_trajfix.npz",
}
I2_V7B = {
    "sangam_lead_0520": "i2_mem_cache_sangam_lead_0520_v7b_prev20.npz",
    "sangam_lead_0521": "i2_mem_cache_sangam_lead_0521_v7b_prev20.npz",
    "sangam_pertO_v2":  "i2_mem_cache_sangam_pertO_v2_v7b_prev20.npz",
    "sangam_pertX_v2":  "i2_mem_cache_sangam_pertX_v2_v7b_prev20.npz",
    "DriverB_new":      "i2_mem_cache_DriverB_new_v7b_prev20.npz",
}
I2_MF20 = {nm: f"i2_mem_mf20_{nm}.npz" for nm in I2_V7B}   # 20f a_2 (gate BCE 학습용)
I2 = I2_V7B   # build_dataset 가 module-global I2 참조 → main 에서 --i2-set 으로 교체
DSETS = list(DATASETS.keys())
# closing buckets (m/s, 양수=접근). receding / cruise / approach弱 / approach强
BUCKETS = [("receding(<-1)", -1e9, -1.0), ("cruise(-1~1)", -1.0, 1.0),
           ("approach弱(1~4)", 1.0, 4.0), ("approach强(>4)", 4.0, 1e9)]


def build_dataset(vrel_mode, pedal_suffix):
    """vrel_mode: 'pred'|'gt'|'a2logits'|'none'. pedal_suffix: 'ttc'|'distbrake' (label 로드용).
    'a2logits' = mf20 cache a2_logits(20f a_2) in-lane closing — a-pred, 추론=학습 일관."""
    dsets = []
    for nm in DSETS:
        pb = DATASETS[nm]; i2 = I2[nm]
        pl = f"pedal_label_{nm}_{pedal_suffix}.npz"
        if vrel_mode == "pred":
            ig = f"{D}/cache/pred_vrel_{nm}.npz"
        elif vrel_mode in ("gt", "a2logits"):
            ig = f"{D}/cache/inlane_gt_{nm}.npz"   # a2logits 도 gate target 위해 로드(vrel 만 a2 에서)
        else:
            ig = None
        d = Prof635Dataset(f"{D}/prebake/{pb}", f"{D}/cache/{i2}", f"{D}/cache/{pl}", inlane_gt_path=ig,
                           vrel_from_a2logits=(vrel_mode == "a2logits"))
        dsets.append(d)
    return ConcatDataset(dsets)


@torch.no_grad()
def collect(model, ds, dev, val_ratio=0.05, seed=0, bs=512, workers=6):
    n = len(ds); nval = int(n * val_ratio); ntr = n - nval
    _, va = random_split(ds, [ntr, nval], generator=torch.Generator().manual_seed(seed))
    dl = DataLoader(va, batch_size=bs, shuffle=False, num_workers=workers)
    deltas, finals, bases, labels, closings, mask, gates, objs = [], [], [], [], [], [], [], []
    model.eval()
    for b in dl:
        bt = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        _, final = model(bt)
        deltas.append((model.last_delta.cpu().numpy() if model.last_delta is not None
                       else np.zeros(final.shape[0])))
        finals.append(final.cpu().numpy())
        bases.append(model.last_base_pedal.cpu().numpy())
        labels.append(bt["pedal_label"].cpu().numpy())
        closings.append(bt["gt_vrel"].cpu().numpy() if "gt_vrel" in bt
                        else np.full(final.shape[0], np.nan))
        mask.append(bt["mask0"].cpu().numpy())
        gates.append(model.last_gate.cpu().numpy() if model.last_gate is not None
                     else np.ones(final.shape[0]))
        # a_2 in-lane obj_amax (bins 16-25) — no-lead / no-obj 판정용
        if "a2_logits" in bt and bt["a2_logits"] is not None:
            objs.append(bt["a2_logits"][:, 16:26, 0].amax(dim=1).cpu().numpy())
        else:
            objs.append(np.full(final.shape[0], -10.0, dtype=np.float32))
    return {k: np.concatenate(v) for k, v in
            dict(delta=deltas, final=finals, base=bases, label=labels,
                 closing=closings, mask=mask, gate=gates, obj=objs).items()}


def load_model(path, side_layers, delta_mode, dev):
    m = Prof635SideModel(side_layers=side_layers, delta_mode=delta_mode).to(dev)
    sd = torch.load(path, map_location=dev, weights_only=False)
    m.load_state_dict(sd, strict=False)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", action="append", required=True,
                    metavar="NAME PATH VREL[pred|gt|none] [PEDAL_SUFFIX]",
                    help="평가 모델: NAME PATH VREL_MODE [pedal_suffix(기본 ttc, distbrake=distbrake)]")
    ap.add_argument("--side-layers", type=int, default=4)
    ap.add_argument("--delta-mode", default="brake_only")
    ap.add_argument("--delta-scale", type=float, default=1.0,
                    help="f_2 brake delta 증폭 계수 k. final=clamp(base + gate·k·delta). "
                         "기본 1.0(거동 불변). >1 이면 제동 delta 강화(강접근 제동↑ 가설 검증).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--closing-cache", default="pred",
                    help="bucketing 에 쓸 closing 기준 cache: pred(5f a_2) | gt. "
                         "비주입(distbrake) 모델도 이 cache 로 동일 bucket 평가.")
    ap.add_argument("--i2-set", default="v7b", choices=["v7b", "mf20"],
                    help="i2_mem cache set. mf20=20f a_2 (gate BCE 학습 모델 평가 시 일관성).")
    ap.add_argument("--bucket-by-gt", action="store_true",
                    help="[ablation 공정비교] 모든 모델의 closing bucket/corr 기준을 GT closing(inlane_gt)으로 "
                         "통일. 모델 forward 엔 각자 주입 vrel 사용하되, 평가 지표 closing 은 공통 GT. "
                         "→ corr(delta, GT_closing) 3-way 동일 척도.")
    args = ap.parse_args()
    dev = args.device
    global I2
    I2 = I2_MF20 if args.i2_set == "mf20" else I2_V7B
    print(f"[i2] cache set = {args.i2_set}", flush=True)

    # 모든 모델 동일 bucketing 을 위해, distbrake(none) 도 closing 을 별도로 동일 cache 에서 가져옴.
    # → vrel_mode='pred'(or gt) dataset 의 closing 을 reference 로 한 번 구축.
    ref_mode = args.closing_cache
    print(f"[ref] closing bucket reference = {ref_mode}_vrel cache", flush=True)
    print(f"[cfg] gate=learned-NN(gate_head) delta_mode={args.delta_mode} "
          f"DELTA_SCALE={args.delta_scale}", flush=True)

    results = {}
    for spec in args.model:
        name, path = spec[0], spec[1]
        vrel = spec[2] if len(spec) > 2 else "pred"
        psuf = spec[3] if len(spec) > 3 else ("distbrake" if vrel == "none" else "ttc")
        # bucketing 은 항상 ref closing — none 모델도 ref cache 주입 dataset 으로 평가하되,
        # 모델엔 gt_vrel 이 들어가도 forward 에서 gate_mode/주입 영향만 받음.
        #   none 모델은 vrel 주입 경로(side_vrel_embed)가 학습 안 됐으니 closing 주입돼도 무시됨(weights 미사용 아님 → 영향 가능).
        #   공정성 위해: none 모델은 vrel 미주입 dataset 으로 forward(모델 그대로), bucket 은 ref closing 별도 join.
        m = load_model(path, args.side_layers, args.delta_mode, dev)
        m.delta_scale = args.delta_scale
        if vrel == "none":
            ds_model = build_dataset("none", psuf)        # 모델 forward 는 vrel 미주입
            ds_ref = build_dataset(ref_mode, psuf)        # closing 만 ref 에서
            out = collect(m, ds_model, dev)
            ref = collect(m, ds_ref, dev)                 # 동일 split → closing 정렬 동일
            out["closing"] = ref["closing"]
        else:
            ds = build_dataset(vrel, psuf)
            out = collect(m, ds, dev)
        # 공정비교: 평가 지표 closing 을 공통 GT(inlane_gt)로 통일. 모델 forward(주입)는 위에서 이미 끝남.
        if args.bucket_by_gt:
            ref_gt = collect(m, build_dataset("gt", psuf), dev)   # 동일 split → 정렬 동일
            out["closing"] = ref_gt["closing"]
        results[name] = (out, vrel, psuf)
        print(f"[collected] {name} vrel={vrel} pedal={psuf} N_val={len(out['delta'])}", flush=True)

    # === ACC modulation 표: closing bucket 별 mean delta / final / label ===
    # lead-only: closing!=0 (no-lead 는 closing=0 fill → 제외). delta 의 closing 의존성을 base 묻힘 없이 본다.
    print("\n" + "=" * 100)
    print("[LEAD-ONLY] closing(주입/ref)!=0 bucket 별 f_2 brake delta (음수=제동). monotonic 강해질수록 ACC.")
    print("=" * 100)
    for name, (out, vrel, psuf) in results.items():
        cl = out["closing"]; dl = out["delta"]; fn = out["final"]; lb = out["label"]
        bp = out["base"]; mk = out["mask"].astype(bool)
        valid = mk & np.isfinite(cl) & (np.abs(cl) > 1e-6)   # lead-only
        print(f"\n--- {name} (vrel={vrel}, pedal={psuf}) ---  N_lead={valid.sum()}")
        print(f"  {'bucket':16s} {'n':>7s} {'closing_m':>9s} {'delta':>8s} {'final':>8s} {'base':>8s} {'label':>8s}")
        for bname, lo, hi in BUCKETS:
            sel = valid & (cl >= lo) & (cl < hi)
            if sel.sum() < 5:
                print(f"  {bname:16s} {sel.sum():>7d}  (too few)")
                continue
            print(f"  {bname:16s} {sel.sum():>7d} {cl[sel].mean():>9.2f} "
                  f"{dl[sel].mean():>8.3f} {fn[sel].mean():>8.3f} {bp[sel].mean():>8.3f} {lb[sel].mean():>8.3f}")
        # delta vs closing Pearson (lead-only) — modulation 강도 단일 수치
        if valid.sum() >= 10:
            c = np.corrcoef(cl[valid], dl[valid])[0, 1]
            print(f"  >> corr(delta, closing) = {c:+.3f}  (음↓ = 접근强→delta 더 음수 = ACC modulation)")

    # === brake-region (label<=-0.3) MAE / recall ===
    print("\n" + "=" * 100)
    print("brake-region (pedal_label<=-0.3): final pedal MAE, recall(final<=-0.3 | label<=-0.3)")
    print("=" * 100)
    for name, (out, vrel, psuf) in results.items():
        lb = out["label"]; fn = out["final"]; mk = out["mask"].astype(bool)
        br = mk & (lb <= -0.3)
        if br.sum() < 5:
            print(f"  {name:24s} brake-region n={br.sum()} (too few)"); continue
        mae = np.abs(fn[br] - lb[br]).mean()
        recall = (fn[br] <= -0.3).mean()
        # no-brake region false-brake rate
        nb = mk & (lb > -0.1)
        fb = (fn[nb] <= -0.3).mean() if nb.sum() else float("nan")
        print(f"  {name:24s} brake-n={br.sum():>6d} MAE={mae:.3f} recall@-0.3={recall*100:5.1f}%  "
              f"| no-brake-n={nb.sum()} false-brake@-0.3={fb*100:.1f}%")

    # === 高-closing far-distance (ACC 장점): closing>4 & ref dist>20 에서 제동 강도 ===
    print("\n" + "=" * 100)
    print("ACC 장점 검증: 高-closing(>4) 케이스에서 distbrake 대비 final pedal (더 음수=더 제동)")
    print("=" * 100)
    for name, (out, vrel, psuf) in results.items():
        cl = out["closing"]; fn = out["final"]; dl = out["delta"]; mk = out["mask"].astype(bool)
        sel = mk & np.isfinite(cl) & (cl > 4.0)
        if sel.sum() < 5:
            print(f"  {name:24s} hi-closing n={sel.sum()} (too few)"); continue
        print(f"  {name:24s} hi-closing(>4) n={sel.sum():>6d} final_mean={fn[sel].mean():+.3f} "
              f"delta_mean={dl[sel].mean():+.3f} brake%@-0.1={100*(fn[sel]<-0.1).mean():.0f}% "
              f"brake%@-0.3={100*(fn[sel]<=-0.3).mean():.0f}%")

    # === scale↑ 부작용: cruise+approach弱 over-brake, no-lead(no in-lane obj) false-brake ===
    print("\n" + "=" * 100)
    print("scale↑ 부작용: cruise(-1~1)+approach弱(1~4) over-brake / no-lead(obj_amax<0) false-brake@-0.3")
    print("=" * 100)
    for name, (out, vrel, psuf) in results.items():
        cl = out["closing"]; fn = out["final"]; mk = out["mask"].astype(bool); ob = out["obj"]
        # cruise+approach弱 : lead 있으나 강접근 아님 (-1<closing<4)
        weak = mk & np.isfinite(cl) & (cl > -1.0) & (cl < 4.0)
        ob_w = 100 * (fn[weak] <= -0.3).mean() if weak.sum() else float("nan")
        fm_w = fn[weak].mean() if weak.sum() else float("nan")
        # no-lead: in-lane obj 미인지 (gate 가 닫혀야 정상). obj_amax<0 = no in-lane object
        nolead = mk & (ob < 0.0)
        fb_nl = 100 * (fn[nolead] <= -0.3).mean() if nolead.sum() else float("nan")
        fm_nl = fn[nolead].mean() if nolead.sum() else float("nan")
        print(f"  {name:24s} weak(-1~4) n={weak.sum():>6d} final_mean={fm_w:+.3f} over-brake@-0.3={ob_w:.1f}%"
              f"  | no-lead n={nolead.sum():>6d} final_mean={fm_nl:+.3f} false-brake@-0.3={fb_nl:.1f}%")


if __name__ == "__main__":
    main()
