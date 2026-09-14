"""closed_loop_node 의 prof635 경로 오프라인 통합 검증 (ROS 무관).
노드와 동일하게 raceline_dt0p1s.load_map → sample_traj40 → build_traj6 → Prof635SideModel(distbrake_v1, brake_only) 확인.
sangam/speedway × mild 양 트랙. i2_mem 은 cached 1개 주입(이 테스트는 raceline→traj6→steer/pedal 링크 검증용;
i2_mem live 정합성은 verify_i2_live_vs_cache 가 별도 검증).
"""
import os, sys, math
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.image_models.prof635_side_model import Prof635SideModel
from src.datamodule.prof635_dataset import build_traj6, build_ego6
from src.utils import raceline_dt0p1s

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.environ.get("SCC_DATA_DIR") or ("/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data" if os.path.isdir("/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data") else os.path.join(REPO_ROOT, "data", "local"))
REPO_DATA = os.path.join(REPO_ROOT, "data")  # raceline 통합맵은 repo data/control (노드도 REPO_ROOT/data 사용)
CK = "data/ckpts"
MODEL_CKPT = os.environ.get("PROF635_CKPT", "prof635_side_distbrake_v1.pt")
DELTA_MODE = os.environ.get("DELTA_MODE", "brake_only")


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Prof635SideModel(residual_gate=True, use_traj_in_side=False, delta_mode=DELTA_MODE).to(dev)
    model.load_main_frozen(f"{CK}/main_prof654.pth")
    model.load_state_dict(torch.load(f"{CK}/{MODEL_CKPT}", map_location=dev, weights_only=False), strict=False)
    model.eval()
    print(f"[model] {MODEL_CKPT} delta_mode={DELTA_MODE}")
    # cached i2_mem 1개 (link 검증용)
    zi = np.load(f"{D}/cache/i2_mem_cache_sangam_lead_0520_v7b_prev20.npz", allow_pickle=True)
    i2_np = zi["i2_mem"][np.argmax(np.abs(zi["i2_mem"].astype(np.float32)).sum((1, 2)))].astype(np.float32)
    i2 = torch.from_numpy(i2_np).unsqueeze(0).to(dev)
    a2 = torch.full((1, 40, 7), -10.0, device=dev)  # no-lead

    all_ok = True
    for track in ("sangam", "speedway"):
        m = raceline_dt0p1s.load_map(track, "mild", REPO_DATA)
        P, M, N = m["P"], m["M"], m["N"]
        print(f"\n[{track}/mild] N={N} P={P} M={M} speed[{m['speed'].min():.0f},{m['speed'].max():.0f}] m/s path={os.path.basename(m['path'] or '-')}")
        ok = True
        for frac in (0.1, 0.4, 0.7):  # 주행구간 [P,M] 여러 지점에서 ego on-raceline
            i0 = int(P + frac * (M - P))
            ex, ey = float(m["xy"][i0, 0]), float(m["xy"][i0, 1])
            j = min(i0 + 1, N - 1)
            eyaw = math.atan2(m["xy"][j, 1] - m["xy"][i0, 1], m["xy"][j, 0] - m["xy"][i0, 0])
            ev = float(m["speed"][i0])
            traj40, meta = raceline_dt0p1s.sample_traj40(m, ex, ey, eyaw)
            traj6 = torch.from_numpy(build_traj6(traj40, mode="speed")).unsqueeze(0).to(dev)
            ego6 = torch.from_numpy(build_ego6(np.array([ev, 0, 0, 0, 0], np.float32))).unsqueeze(0).to(dev)
            with torch.no_grad():
                steer_n, pedal = model({"traj6": traj6, "ego6": ego6, "i2_mem": i2, "a2_logits": a2})
            sd = float(steer_n) * 40; pd = float(pedal)
            finite = math.isfinite(sd) and math.isfinite(pd)
            inrange = abs(sd) < 40 and -1.01 <= pd <= 1.01
            t0 = traj40[0]
            near0 = abs(t0[0]) < 2 and abs(t0[1]) < 2  # ego on raceline → 첫 점 ~원점
            ok &= finite and inrange and near0
            print(f"  i0={i0:5d} in_pit={meta['in_pit']} traj6[0]=[{traj6[0,0,0]:+.2f},{traj6[0,0,1]:+.2f},spd{traj6[0,0,2]:.1f}] "
                  f"→ steer={sd:+.1f}deg pedal={pd:+.2f} {'OK' if (finite and inrange and near0) else 'CHECK'}")
        print(f"  [{track}] {'PASS' if ok else 'FAIL'}")
        all_ok &= ok
    print(f"\n[overall] {'PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
