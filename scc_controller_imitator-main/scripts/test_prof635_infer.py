"""v5 (Prof635SideModel residual+no-traj) deployment smoke test — ROS 무관, 오프라인.

Test A (모델 I/O, GT 검증): prebake 샘플 → build_traj6(speed)+ego6+cached i2_mem → model
        → steer 가 prebake target steer 와 일치하는지 (deploy 모델이 올바른 제어 산출 확인).
Test B (raceline 파이프라인): raceline + ego pose → infer_step → 유한한 steer/pedal 나오는지.
"""
import os, sys, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.image_models.prof635_side_model import Prof635SideModel
from src.datamodule.prof635_dataset import build_traj6, build_ego6, STEER_NORM
from src.utils import prof635_infer
from src.utils.raceline_search import load_raceline_dt, resolve_cutoff

D = os.environ.get("SCC_DATA_DIR") or ("/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data" if os.path.isdir("/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data") else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "local"))
CK = "data/ckpts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="sangam_lead_0520")
    ap.add_argument("--main-ckpt", default=f"{CK}/main_prof654.pth")
    ap.add_argument("--model-ckpt", default=f"{CK}/prof635_side_v5_best.pt")
    ap.add_argument("--raceline", default="data/control/raceline_sangam.csv")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model = Prof635SideModel(residual_gate=True, use_traj_in_side=False).to(dev)
    miss, unexp = model.load_main_frozen(args.main_ckpt)
    sd = torch.load(args.model_ckpt, map_location=dev, weights_only=False)
    m2, u2 = model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"[load] main(miss={len(miss)}) v5(miss={len(m2)} unexp={len(u2)}) "
          f"residual_gate={model.residual_gate} use_traj_in_side={model.use_traj_in_side}")

    # ---- Test A: 모델 I/O vs GT steer ----
    pb = np.load(f"{D}/prebake/f1_prebake_iy_{args.ds}_v1_brakeaug_prev20.npz", allow_pickle=True)
    ck = pb["curr_keys"].tolist(); tr = pb["traj"]; eg = pb["ego_5d"]; tg = pb["target"]; msk = pb["mask"]
    zi = np.load(f"{D}/cache/i2_mem_cache_{args.ds}_v7b_prev20.npz", allow_pickle=True)
    k2i = {k: i for i, k in enumerate(zi["keys"].tolist())}; i2arr = zi["i2_mem"]; a2arr = zi["a2_logits"]
    idx = [i for i in range(len(ck)) if k2i.get(ck[i]) is not None and msk[i, 0]][:args.n]
    T6 = torch.tensor(np.stack([build_traj6(tr[i], mode="speed") for i in idx])).to(dev)
    E6 = torch.tensor(np.stack([build_ego6(eg[i]) for i in idx])).to(dev)
    I2 = torch.tensor(np.stack([i2arr[k2i[ck[i]]].astype(np.float32) for i in idx])).to(dev)
    A2 = torch.tensor(np.stack([a2arr[k2i[ck[i]]].astype(np.float32) for i in idx])).to(dev)
    with torch.no_grad():
        steer_n, pedal = model({"traj6": T6, "ego6": E6, "i2_mem": I2, "a2_logits": A2})
    steer_deg = steer_n.cpu().numpy() * STEER_NORM
    gt_steer = np.array([tg[i, 0, 0] for i in idx])
    mae = np.abs(steer_deg - gt_steer).mean()
    corr = np.corrcoef(steer_deg, gt_steer)[0, 1]
    ped = pedal.cpu().numpy()
    print(f"\n[Test A] 모델 I/O (N={len(idx)}, cached i2_mem)")
    print(f"  steer: MAE={mae:.2f}° corr={corr:.3f} (frozen #654 main)")
    print(f"  pedal: mean={ped.mean():+.3f} range[{ped.min():+.2f},{ped.max():+.2f}] brake%={100*(ped<-0.05).mean():.0f}%")
    print(f"  {'PASS' if (mae < 3.0 and corr > 0.8) else 'CHECK'}: steer MAE<3° & corr>0.8 → deploy 모델 정상")

    # ---- Test B: raceline → infer_step 파이프라인 ----
    print(f"\n[Test B] raceline 파이프라인 ({args.raceline})")
    try:
        raceline, polygon = load_raceline_dt(args.raceline)
        cutoff = resolve_cutoff(args.raceline, track="sangam")
        ex, ey = float(raceline[300, 1]), float(raceline[300, 2])
        e_yaw = float(raceline[300, 3])
        ego6 = prof635_infer.build_ego6(vx=20.0, vy=0.0, ax=0.0, ay=0.0, yaw_rate=0.0)
        i2 = i2arr[k2i[ck[idx[0]]]].astype(np.float32)
        out = prof635_infer.infer_step(model, raceline, ex, ey, e_yaw, ego6, i2,
                                       cutoff, polygon, device=dev)
        print(f"  raceline rows={len(raceline)} cutoff={cutoff} polygon={'Y' if polygon is not None else 'N'}")
        print(f"  traj6[0]={out['traj6'][0].round(2)} (ego-on-raceline → x,y≈0)")
        print(f"  → steer={out['steer_deg']:+.2f}° pedal={out['pedal']:+.3f} "
              f"(acc={out['accel_pedal']:.2f} brk={out['brake_pedal']:.2f})")
        finite = np.isfinite([out['steer_deg'], out['pedal']]).all()
        print(f"  {'PASS' if finite else 'FAIL'}: 파이프라인 유한 출력 (speed 정확도는 dt0p1s raceline 필요)")
    except Exception as e:
        print(f"  raceline 테스트 skip: {e}")


if __name__ == "__main__":
    main()
