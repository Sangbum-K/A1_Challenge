"""#635 frozen main(steer) + f_2 side(pedal) 학습.
f_1(#635) frozen → steer 그대로. f_2 만 학습 (pedal, L1 vs pedal_label).
"""
import argparse, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.image_models.prof635_side_model import Prof635SideModel
from src.datamodule.prof635_dataset import Prof635Dataset

D = os.environ.get("SCC_DATA_DIR") or ("/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data" if os.path.isdir("/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data") else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "local"))
# i2_cache: v7b(2f) → mf20(20f, A2ModelMultiFrame v2s_n20). gate 입력 obj logit 이 20f 라 정확.
DATASETS = {  # name: (prebake, i2_cache(mf20), pedal_label) — _ttc = ttc-augment 된 brake-inclusive 라벨
    "sangam_lead_0520": ("f1_prebake_iy_sangam_lead_0520_v1_brakeaug_prev20.npz",
                          "i2_mem_mf20_sangam_lead_0520.npz", "pedal_label_sangam_lead_0520_ttc.npz"),
    "sangam_lead_0521": ("f1_prebake_iy_sangam_lead_0521_v1_brakeaug_prev20.npz",
                          "i2_mem_mf20_sangam_lead_0521.npz", "pedal_label_sangam_lead_0521_ttc.npz"),
    "sangam_pertO_v2":  ("f1_prebake_iy_sangam_pertO_v2_prev20_fixed.npz",
                          "i2_mem_mf20_sangam_pertO_v2.npz", "pedal_label_sangam_pertO_v2_ttc.npz"),
    "sangam_pertX_v2":  ("f1_prebake_iy_sangam_pertX_v2_prev20_fixed.npz",
                          "i2_mem_mf20_sangam_pertX_v2.npz", "pedal_label_sangam_pertX_v2_ttc.npz"),
    "DriverB_new":      ("f1_prebake_iy_DriverB_new_logged_fixed_prev20_trajfix.npz",
                          "i2_mem_mf20_DriverB_new.npz", "pedal_label_DriverB_new_ttc.npz"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    ap.add_argument("--main-ckpt", default="data/ckpts/main_prof654.pth")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--cosine-restarts", action="store_true",
                    help="decaying-max cosine warm restarts lr (주기마다 max lr 감소). 길게 끝까지 학습용.")
    ap.add_argument("--cos-t0", type=int, default=20, help="첫 restart 주기(epoch)")
    ap.add_argument("--cos-tmult", type=int, default=2, help="restart 주기 배수 (T0, T0·tmult, ...)")
    ap.add_argument("--cos-gamma", type=float, default=0.6, help="restart 마다 max lr × gamma (decaying-max)")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--side-layers", type=int, default=4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--name", default="prof635_side_v1")
    ap.add_argument("--gate-bce", action="store_true",
                    help="gate(last_gate_logit) 에 BCE supervision 추가. target g* = (inlane_dist 유효=lead) "
                         "& (inlane_closing > gate_tau=접근). collapse 방지 + 규정(learned gate) 충족. "
                         "켜면 inlane_gt cache 자동 로드, gt_vrel 주입 강제 OFF (vrel 은 i2_mem 20f 암묵).")
    ap.add_argument("--gate-tau", type=float, default=0.5,
                    help="gate BCE 접근 임계 (m/s). closing>tau → g*=1. 0.5=명확한 접근만 gate open target.")
    ap.add_argument("--w-gate", type=float, default=1.0,
                    help="gate BCE loss 가중 (pedal L1 에 가산).")
    ap.add_argument("--inject-gt-vrel", action="store_true",
                    help="[진단] GT 상대속도(inlane closing)를 f_2 에 주입 — perception vs capacity 격리")
    ap.add_argument("--inject-pred-vrel", action="store_true",
                    help="[실험 B] 5f a_2 predicted closing(cache/pred_vrel_{ds}.npz)을 GT 대신 f_2 에 주입 "
                         "— 신뢰 perception vrel 로 ACC pedaling 검증. inlane_gt 와 동일 형식·동일 gt_vrel 키.")
    ap.add_argument("--gate-bce-keep-vrel", action="store_true",
                    help="[ablation] gate BCE(gate 변별) + vrel 명시 주입(f_2 closing-제동)을 둘 다 켬. "
                         "기본 gate-bce 는 vrel 주입을 강제 OFF(i2_mem 암묵) 하나, 이 플래그로 동시 허용. "
                         "vrel 출처: --vrel-from-a2logits(a-pred, mf20 20f a_2 예측) 또는 inlane_gt(a-GT). "
                         "gate target g* 은 항상 inlane_gt(GT) 에서 계산.")
    ap.add_argument("--vrel-from-a2logits", action="store_true",
                    help="[a-pred] f_2 에 주입할 vrel 을 mf20 cache 의 a2_logits(20f a_2 예측) in-lane closing "
                         "으로 계산 — 추론(closed_loop 20f a_2)=학습 일관. gate-bce-keep-vrel 과 함께 사용.")
    ap.add_argument("--pedal-suffix", default="ttc",
                    help="pedal_label_{ds}_{suffix}.npz. ttc=closing-ACC, distbrake=거리-onset 곡선")
    ap.add_argument("--gate-warm-thresh", type=float, default=0.3,
                    help="learned gate MLP warm-init: 옛 rule g=sigmoid((obj-thr)·scale) 의 thr 근사.")
    ap.add_argument("--gate-warm-scale", type=float, default=5.0,
                    help="learned gate MLP warm-init: 옛 rule 의 sharpness scale 근사 (obj 채널).")
    ap.add_argument("--gate-warm-vrel-scale", type=float, default=1.0,
                    help="learned gate MLP warm-init: closing(접근) 채널 weight. >0 이면 접근(closing>0)일 때 "
                         "gate logit↑(open), receding(closing<0)은 GELU≈0 으로 기여 없음(close). sign 정합.")
    ap.add_argument("--resume", default="",
                    help="warm-start ckpt (prior best.pt). gate_head 는 새로 추가되어 strict=False 로 로드.")
    ap.add_argument("--delta-mode", default="brake_only", choices=["signed", "brake_only", "raw"],
                    help="brake_only=delta≤0 (가속 간섭 불가). signed=±2tanh. raw=clamp 없는 head (gate 없는 버전용)")
    ap.add_argument("--no-gate", action="store_true",
                    help="gate(g) 제거 → final=base+delta. delta 가 no-lead→0/lead→brake 직접 흡수 "
                         "(gate no-lead 과개방→풀악셀 깎임 회피). --delta-mode raw 와 함께. gate BCE 는 자동 무시.")
    ap.add_argument("--randomize-base", action="store_true",
                    help="학습 시 f_2 의 base 입력을 U[-1,1] 랜덤 치환 → 추론 base=+1.0(raceline nominal)에서도 풀제동. train/infer base 불일치 해소")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = f"outputs_multi/{args.name}"; os.makedirs(out_dir, exist_ok=True)
    def log(m):
        print(m, flush=True)
        open(f"{out_dir}/train.log", "a").write(m + "\n")

    # gate-bce 면 vrel 주입은 강제 OFF (vrel 정보는 i2_mem 20f 가 암묵적으로 담음 — 명시 주입 제거).
    #   단 --gate-bce-keep-vrel(ablation) 이면 gate BCE + vrel 주입을 동시 허용.
    keep_vrel = args.gate_bce and args.gate_bce_keep_vrel
    if args.gate_bce and not args.gate_bce_keep_vrel and (args.inject_gt_vrel or args.inject_pred_vrel
                                                          or args.vrel_from_a2logits):
        log("[gate-bce] vrel 명시 주입 무시 (i2_mem 20f 암묵). inject_*_vrel / vrel_from_a2logits OFF. "
            "(동시 허용하려면 --gate-bce-keep-vrel)")
        args.inject_gt_vrel = False; args.inject_pred_vrel = False; args.vrel_from_a2logits = False
    if keep_vrel:
        src = ("a2logits(20f a_2 pred)" if args.vrel_from_a2logits else
               "pred_vrel cache(5f a_2)" if args.inject_pred_vrel else "inlane_gt(GT)")
        log(f"[gate-bce-keep-vrel] gate BCE + vrel 주입 동시 ON. vrel 출처={src}. "
            f"gate target g* 은 항상 inlane_gt(GT).")
    dsets = []
    for nm in args.datasets:
        pb, i2, _ = DATASETS[nm]
        pl = f"pedal_label_{nm}_{args.pedal_suffix}.npz"
        # inlane_gt: gate-bce 면 항상 inlane_gt(gate target 계산). 아니면 vrel 주입용(pred/GT).
        #   a-pred(vrel_from_a2logits)는 gate target 위해 inlane_gt 로드, vrel 은 a2_logits 에서 별도 계산.
        if args.gate_bce:
            ig = f"{D}/cache/inlane_gt_{nm}.npz"   # gate target g* (+ a-GT 면 vrel 도 여기서)
        elif args.inject_pred_vrel:
            ig = f"{D}/cache/pred_vrel_{nm}.npz"
        elif args.inject_gt_vrel:
            ig = f"{D}/cache/inlane_gt_{nm}.npz"
        else:
            ig = None
        d = Prof635Dataset(f"{D}/prebake/{pb}", f"{D}/cache/{i2}", f"{D}/cache/{pl}",
                           inlane_gt_path=ig, gate_tau=args.gate_tau,
                           vrel_from_a2logits=(keep_vrel and args.vrel_from_a2logits))
        vtag = (' +gate_bce' + (' +vrel(a2logits)' if (keep_vrel and args.vrel_from_a2logits)
                                else ' +vrel(GT)' if keep_vrel else '')) if args.gate_bce else (
            ' +pred_vrel' if args.inject_pred_vrel else (' +gt_vrel' if args.inject_gt_vrel else ''))
        if args.gate_bce and d.gate_tgt is not None:
            pos = float(d.gate_tgt.mean()); cov = float(d.gate_valid.mean())
            vtag += f' (g*pos={pos*100:.1f}% cov={cov*100:.0f}%)'
        log(f"[data] {nm}: {len(d)}{vtag}")
        dsets.append(d)
    full = ConcatDataset(dsets)
    nval = int(len(full) * args.val_ratio); ntr = len(full) - nval
    tr, va = random_split(full, [ntr, nval], generator=torch.Generator().manual_seed(0))
    log(f"[data] total={len(full)} train={ntr} val={nval}")
    tl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    vl = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = Prof635SideModel(side_layers=args.side_layers,
                             delta_mode=args.delta_mode, no_gate=args.no_gate,
                             randomize_base=args.randomize_base,
                             gate_warm_thresh=args.gate_warm_thresh,
                             gate_warm_scale=args.gate_warm_scale,
                             gate_warm_vrel_scale=args.gate_warm_vrel_scale).to(dev)
    miss, unexp = model.load_main_frozen(args.main_ckpt)
    # warm-start: prior best.pt 의 side(f_2) 가중치 resume. gate_head 는 신규 → strict=False.
    if args.resume and os.path.exists(args.resume):
        rsd = torch.load(args.resume, map_location="cpu", weights_only=False)
        if isinstance(rsd, dict) and "model" in rsd: rsd = rsd["model"]
        # raw delta(gate 없는 버전): pedal_head 는 의미가 달라(brake_only=-2sigmoid head vs raw=linear delta)
        #   warm 그대로 쓰면 초기 delta 폭주 → pedal_head 만 새 init(zeros,bias0=delta≈0=final≈base) 유지.
        if args.delta_mode == "raw" and "_ng" not in os.path.basename(args.resume):
            # gate(brake_only head) → raw 전환 시만 head 새 init. _ng(이미 raw head) resume 은 유지(cosine 추가 학습).
            rsd = {k: v for k, v in rsd.items() if not k.startswith("pedal_head")}
            log("[resume:raw] pedal_head 제외 로드 (gate→raw 전환 — raw head 새 init, cross-attn/embed warm)")
        rmiss, runexp = model.load_state_dict(rsd, strict=False)
        n_gate = sum(1 for k in rmiss if k.startswith("gate_head."))
        log(f"[resume] {os.path.basename(args.resume)} loaded "
            f"(missing={len(rmiss)} [gate_head 신규={n_gate}] unexpected={len(runexp)})")
    log(f"[model] main frozen (missing={list(miss)} unexpected={len(unexp)}) "
        f"trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)/1e3:.0f}K")
    n_gate_params = sum(p.numel() for p in model.gate_head.parameters())
    log(f"[model] gate_head MLP params={n_gate_params} (learned lead-presence gate, rule 제거)")
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = None
    if args.cosine_restarts:
        import math as _math
        def _cos_decay(ep):  # decaying-max cosine warm restarts: gamma^cycle · 0.5(1+cos(pi·t/T))
            t, T, cyc = ep, args.cos_t0, 0
            while t >= T:
                t -= T; cyc += 1; T *= args.cos_tmult
            return (args.cos_gamma ** cyc) * 0.5 * (1.0 + _math.cos(_math.pi * t / max(1, T)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, _cos_decay)
        log(f"[lr] cosine restarts: T0={args.cos_t0} Tmult={args.cos_tmult} gamma={args.cos_gamma} base_lr={args.lr}")

    wb = None
    if args.wandb:
        import wandb
        wb = wandb.init(project="scc-controller-imitator", entity="hmg_e2e_controller_imitator",
                        name=args.name, config=vars(args), dir=out_dir)

    best = 1e9
    for ep in range(args.epochs):
        model.train()
        t0 = time.time(); rl = rn = 0.0
        rg = 0.0; rgn = 0.0
        for b in tl:
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            if args.gate_bce and not keep_vrel:
                b.pop("gt_vrel", None)   # vrel 명시 주입 제거 (i2_mem 20f 암묵). keep_vrel 이면 유지.
            _, pedal = model(b)
            m0 = b["mask0"]
            if not m0.any(): continue
            loss = F.l1_loss(pedal[m0], b["pedal_label"][m0])
            if args.gate_bce and "gate_tgt" in b:
                # gate BCE: last_gate_logit vs g* (lead & 접근). gate_valid==1 frame 만 (inlane_gt 커버).
                gv = (b["gate_valid"] > 0.5) & m0
                if gv.any():
                    gloss = F.binary_cross_entropy_with_logits(
                        model.last_gate_logit[gv], b["gate_tgt"][gv])
                    loss = loss + args.w_gate * gloss
                    rg += gloss.item() * int(gv.sum()); rgn += int(gv.sum())
            opt.zero_grad(); loss.backward(); opt.step()
            rl += loss.item() * int(m0.sum()); rn += int(m0.sum())
        train_l = rl / max(1, rn); train_g = rg / max(1, rgn)
        # val
        model.eval(); vl_ped = vl_st = vn = 0.0
        # gate 검증: g(=sigmoid logit) 를 g*=1(lead&접근)/g*=0(no-lead 또는 정속) 으로 분리 누적.
        g_pos_sum = g_pos_n = g_neg_sum = g_neg_n = 0.0
        with torch.no_grad():
            for b in vl:
                b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
                if args.gate_bce and not keep_vrel:
                    b.pop("gt_vrel", None)
                steer, pedal = model(b); m0 = b["mask0"]
                if not m0.any(): continue
                vl_ped += F.l1_loss(pedal[m0], b["pedal_label"][m0]).item() * int(m0.sum())
                vl_st += (steer[m0]*40 - b["steer_label"][m0]*40).abs().sum().item()
                vn += int(m0.sum())
                if args.gate_bce and "gate_tgt" in b:
                    g = torch.sigmoid(model.last_gate_logit)
                    gv = b["gate_valid"] > 0.5
                    pos = gv & (b["gate_tgt"] > 0.5); neg = gv & (b["gate_tgt"] < 0.5)
                    if pos.any(): g_pos_sum += g[pos].sum().item(); g_pos_n += int(pos.sum())
                    if neg.any(): g_neg_sum += g[neg].sum().item(); g_neg_n += int(neg.sum())
        vped = vl_ped / max(1, vn); vst = vl_st / max(1, vn)
        g_pos = g_pos_sum / max(1, g_pos_n); g_neg = g_neg_sum / max(1, g_neg_n)
        gate_str = (f" gate[g|lead&closing={g_pos:.3f} g|else={g_neg:.3f} bce={train_g:.4f}]"
                    if args.gate_bce else "")
        cur_lr = opt.param_groups[0]["lr"]
        log(f"[ep {ep+1:02d}] train_pedal={train_l:.4f} val_pedal={vped:.4f} lr={cur_lr:.2e} "
            f"(steer MAE={vst:.2f}° frozen){gate_str} {time.time()-t0:.0f}s")
        if wb:
            d_wb = {"train/pedal_l1": train_l, "val/pedal_l1": vped, "val/steer_mae_deg": vst, "epoch": ep+1}
            if args.gate_bce:
                d_wb.update({"train/gate_bce": train_g, "val/g_pos": g_pos, "val/g_neg": g_neg})
            wb.log(d_wb)
        if vped < best:
            best = vped
            torch.save(model.state_dict(), f"{out_dir}/best.pt")
            log(f"  -> saved best ({vped:.4f})")
        if sched is not None:
            sched.step()
    log(f"[done] best val_pedal={best:.4f}")
    if wb: wb.finish()


if __name__ == "__main__":
    main()
