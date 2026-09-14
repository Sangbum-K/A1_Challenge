"""traj-speed shortcut ablation.

가설(user): base(f_1)는 이미지 안 봄 → lead brake 는 traj6 speed 채널(prebake 에 박힌 감속)이 원인.
inference(raceline)에선 traj-speed 가 nominal → base brake 사라짐. f_2 도 traj-speed shortcut 으로
brake 했다면 inference 에서 실패.

테스트: lead-TTC 샘플에서 traj-speed 를 flat(=ego vx, 감속명령 제거)으로 바꿔 base/f_2/final 변화 측정.
  - base 가 brake→accel 으로 바뀌면 → base 는 traj-speed artifact (가설 확정).
  - f_2 가 거의 그대로 brake → f_2 는 image 로 brake (transferable, 좋음).
  - f_2 도 brake 사라지면 → f_2 도 traj-speed shortcut (inference 실패, 재설계 필요).
"""
import os
import sys, os, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.image_models.prof635_side_model import Prof635SideModel
from src.datamodule.prof635_dataset import build_ego6, T6, TSRC

D = os.environ.get("SCC_DATA_DIR") or ("/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data" if os.path.isdir("/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data") else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "local"))


def traj6(traj40, speed_override=None):
    xi = np.interp(T6, TSRC, traj40[:, 0]); yi = np.interp(T6, TSRC, traj40[:, 1])
    ch2 = np.full(6, speed_override, np.float32) if speed_override is not None \
        else np.interp(T6, TSRC, traj40[:, 2]).astype(np.float32)
    return np.stack([-yi, xi, ch2], 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="sangam_lead_0520")
    ap.add_argument("--model-ckpt", default="outputs_multi/prof635_side_v3_654gate/best.pt")
    ap.add_argument("--main-ckpt", default="data/ckpts/main_prof654.pth")
    ap.add_argument("--residual", action="store_true", help="model residual_gate=True (v4)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    pb = np.load(f"{D}/prebake/f1_prebake_iy_{args.ds}_v1_brakeaug_prev20.npz", allow_pickle=True)
    ck = pb["curr_keys"].tolist(); tr = pb["traj"]; eg = pb["ego_5d"]
    zi = np.load(f"{D}/cache/i2_mem_cache_{args.ds}_v7b_prev20.npz", allow_pickle=True)
    k2i = {k: i for i, k in enumerate(zi["keys"].tolist())}
    i2arr = zi["i2_mem"]; a2arr = zi["a2_logits"] if "a2_logits" in zi.files else None
    zg = np.load(f"{D}/cache/inlane_gt_{args.ds}.npz", allow_pickle=True)
    zk = {k: i for i, k in enumerate(zg["keys"].tolist())}; gd = zg["inlane_dist"]; gc = zg["inlane_closing"]
    zp = np.load(f"{D}/cache/pedal_label_{args.ds}_ttc.npz", allow_pickle=True)
    k2p = {k: i for i, k in enumerate(zp["keys"].tolist())}; pedarr = zp["pedal"]

    idx = []  # (prebake_i, i2_i, label)
    lead_flag = np.zeros(len(ck), bool)
    for i, k in enumerate(ck):
        gi = zk.get(k); ii = k2i.get(k); pi = k2p.get(k)
        if gi is not None and ii is not None and np.isfinite(gd[gi]) and gd[gi] < 20 and gc[gi] > 1:
            idx.append((i, ii, float(pedarr[pi, 0]) if pi is not None else 0.0)); lead_flag[i] = True
    lab = np.array([l for _, _, l in idx])
    print(f"[lead-TTC] {len(idx)} / {len(ck)} ({100*len(idx)/len(ck):.1f}% of dataset)", flush=True)
    print(f"[LABEL] lead-TTC pedal_label: mean={lab.mean():+.3f} brake%={100*(lab<-0.05).mean():.0f}% "
          f"accel%={100*(lab>0.05).mean():.0f}% | range[{lab.min():+.2f},{lab.max():+.2f}]", flush=True)

    model = Prof635SideModel(residual_gate=args.residual).to(dev)
    model.load_main_frozen(args.main_ckpt)
    if os.path.exists(args.model_ckpt):
        model.load_state_dict(torch.load(args.model_ckpt, map_location=dev, weights_only=False), strict=False)
    model.eval()

    # traj-speed vs ego-speed (감속 박혀있나)
    v_ego = np.array([eg[i][0] for i, _, _ in idx])
    v_traj_future = np.array([np.interp(1.05, TSRC, tr[i][:, 2]) for i, _, _ in idx])  # ~1s 앞 traj speed
    print(f"[traj-speed] ego_vx mean={v_ego.mean():.1f} | traj_speed@1.05s mean={v_traj_future.mean():.1f} "
          f"| 감속(traj<ego) 비율={100*(v_traj_future < v_ego - 0.5).mean():.0f}%", flush=True)

    def run(speed_override):
        T6b, E6b, I2b, A2b = [], [], [], []
        for i, ii, _ in idx:
            T6b.append(traj6(tr[i], speed_override(eg[i][0]) if speed_override else None))
            E6b.append(build_ego6(eg[i])); I2b.append(i2arr[ii].astype(np.float32))
            A2b.append(a2arr[ii].astype(np.float32) if a2arr is not None else np.full((40, 7), -10, np.float32))
        b = {"traj6": torch.tensor(np.stack(T6b)).to(dev), "ego6": torch.tensor(np.stack(E6b)).to(dev),
             "i2_mem": torch.tensor(np.stack(I2b)).to(dev), "a2_logits": torch.tensor(np.stack(A2b)).to(dev)}
        with torch.no_grad():
            steer, final = model(b)
        return (model.last_base_pedal.cpu().numpy(), model.last_f2_pedal.cpu().numpy(),
                final.cpu().numpy(), model.last_gate.cpu().numpy())

    print("\n=== [Q1] traj-speed ablation (lead-TTC) ===")
    print("                                   base(f1)   f2       final    | brake%")
    base0 = None
    for tag, ov in [("traj-speed 원본 (감속 박힘)", None),
                    ("traj-speed flat=ego_vx (감속제거)", lambda v: float(v))]:
        base, f2, final, g = run(ov)
        if base0 is None: base0, final0 = base, final
        print(f"  {tag:32s} {base.mean():+6.3f}  {f2.mean():+6.3f}  {final.mean():+6.3f}   "
              f"{100*(final < -0.05).mean():.0f}%", flush=True)
    print(f"  gate g mean={g.mean():.3f}")

    print("\n=== [Q2] label vs prediction (lead-TTC, 원본 traj) — loss 모순 점검 ===")
    print("  idx   label    base(f1)  f2      final   |label-final|")
    for n in range(min(12, len(idx))):
        print(f"  {n:3d}  {lab[n]:+6.3f}  {base0[n]:+6.3f}  {0.0:+.0f}      {final0[n]:+6.3f}   {abs(lab[n]-final0[n]):.3f}")
    print(f"  lead-TTC  L1(final vs label) = {np.abs(final0 - lab).mean():.4f}  "
          f"(sign 불일치 {100*(np.sign(final0)!=np.sign(lab)).mean():.0f}%)")

    # 전체 데이터 L1 분해 (imbalance masking 정량화)
    print("\n=== [Q2] 전체 sangam_lead L1 분해 (global loss 가 lead 가리나) ===")
    all_lab = np.array([float(pedarr[k2p[k], 0]) if k in k2p else 0.0 for k in ck])
    has = np.array([(k in k2p and k in k2i) for k in ck])
    fin_all = np.full(len(ck), np.nan)
    bs = 4096
    valid = np.where(has)[0]
    for s in range(0, len(valid), bs):
        chunk = valid[s:s+bs]
        T6b = np.stack([traj6(tr[i]) for i in chunk]); E6b = np.stack([build_ego6(eg[i]) for i in chunk])
        I2b = np.stack([i2arr[k2i[ck[i]]].astype(np.float32) for i in chunk])
        A2b = np.stack([(a2arr[k2i[ck[i]]].astype(np.float32) if a2arr is not None else np.full((40,7),-10,np.float32)) for i in chunk])
        b = {"traj6": torch.tensor(T6b).to(dev), "ego6": torch.tensor(E6b).to(dev),
             "i2_mem": torch.tensor(I2b).to(dev), "a2_logits": torch.tensor(A2b).to(dev)}
        with torch.no_grad(): _, f = model(b)
        fin_all[chunk] = f.cpu().numpy()
    m = has & np.isfinite(fin_all)
    e = np.abs(fin_all - all_lab)
    print(f"  global L1 = {e[m].mean():.4f}  (N={m.sum()})")
    print(f"  lead-TTC L1 = {e[m & lead_flag].mean():.4f}  (N={(m&lead_flag).sum()}, {100*(m&lead_flag).sum()/m.sum():.1f}%)")
    print(f"  non-lead L1 = {e[m & ~lead_flag].mean():.4f}  (N={(m&~lead_flag).sum()})")
    print("\n해석:")
    print("  Q1: base 원본→flat 에서 brake 사라지면 traj-speed artifact 확정. f_2 가 flat 에서도 brake 유지하면 image-driven.")
    print("  Q2: global L1 낮아도 lead-TTC L1 크고 sign 불일치 높으면 → 버그 아니라 rare-event 가 평균에 묻힌 것.")


if __name__ == "__main__":
    main()
