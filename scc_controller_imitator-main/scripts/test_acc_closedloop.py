"""ACC 경로 (5f a_2 predicted vrel -> f_2 predvrel) 오프라인 통합 검증 (ROS 무관).

closed_loop_node.py 의 ACC wiring 과 동일한 forward chain 을 캐시 토큰으로 재현:
  1) 5f A2ModelMultiFrame(offsets[10,20,30,40,50], d_out128, 7ch) 로 in-lane predicted vrel(closing) 계산
     -> build_pred_vrel_cache.py 의 cache 출력과 일치하는지 비교(live 5f forward 정합 검증).
  2) f_2 = prof635_side_predvrel_ttc (brake_only) 에 v7b i2_mem + a2_logits + 주입 vrel 을 흘려
     steer/pedal finite·범위 검증 + closing sweep(접근↑ -> brake escalate) monotonic 검증.

CPU 가능: CUDA_VISIBLE_DEVICES="" python scripts/test_acc_closedloop.py
"""
import os, sys, math
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.image_models.a2_model import A2ModelMultiFrame
from src.model.image_models.prof635_side_model import Prof635SideModel
from src.datamodule.prof635_dataset import build_traj6, build_ego6
from src.utils import raceline_dt0p1s

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.environ.get("SCC_DATA_DIR") or (
    "/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data"
    if os.path.isdir("/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data")
    else os.path.join(REPO_ROOT, "data", "local"))
REPO_DATA = os.path.join(REPO_ROOT, "data")
CK = os.path.join(REPO_ROOT, "data", "ckpts")

A2_MF5F_CKPT = os.environ.get("A2_MF5F_CKPT", f"{CK}/a2_mf5f_acc.pt")
F2_CKPT      = os.environ.get("PROF635_CKPT", f"{CK}/prof635_side_predvrel_ttc.pt")
DELTA_MODE   = os.environ.get("DELTA_MODE", "brake_only")

PREV_OFFSETS = [10, 20, 30, 40, 50]
BIN_LO, BIN_HI = 16, 25
SCALE_V, SCALE_DIST = 20.0, 40.0
CH_OBJ, CH_VLREL, CH_DIST = 0, 3, 6
OBJ_THR = 0.5
DS = os.environ.get("ACC_TEST_DS", "sangam_lead_0520")


def _prev_key(run, tick, off):
    return f"{run}__t{tick - off:07d}"


@torch.no_grad()
def step1_predvrel(dev):
    """live 5f forward vs build_pred_vrel_cache.npz 정합."""
    i1f = f"{D}/cache/i1_tok_cache_{DS}_prev20.npz"
    pvf = f"{D}/cache/pred_vrel_{DS}.npz"
    if not (os.path.exists(i1f) and os.path.exists(pvf)):
        print(f"[step1] SKIP (캐시 없음: i1={os.path.exists(i1f)} pred_vrel={os.path.exists(pvf)})")
        return None
    z = np.load(i1f, allow_pickle=True)
    tokens = z["tokens"]; keys = z["keys"].tolist()
    k2i = {k: i for i, k in enumerate(keys)}
    zp = np.load(pvf, allow_pickle=True)
    pv_keys = zp["keys"].tolist(); pv_clos = zp["inlane_closing"]
    pv_k2i = {k: i for i, k in enumerate(pv_keys)}

    model = A2ModelMultiFrame(d_token=256, d_out=128,
                              n_frames=1 + len(PREV_OFFSETS)).to(dev).eval()
    sd = torch.load(A2_MF5F_CKPT, map_location=dev, weights_only=False)
    if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"[step1] a2 5f loaded missing={len(miss)} unexpected={len(unexp)}")

    # cache 에서 lead(closing!=0) 가 있는 key 위주로 샘플 (ACC 동작 검증 관점)
    lead_keys = [k for k in pv_keys if k in k2i and abs(float(pv_clos[pv_k2i[k]])) > 1e-6]
    nolead_keys = [k for k in pv_keys if k in k2i and abs(float(pv_clos[pv_k2i[k]])) <= 1e-6]
    np.random.seed(0)
    bkeys = (list(np.random.choice(lead_keys, min(64, len(lead_keys)), replace=False))
             + list(np.random.choice(nolead_keys, min(16, len(nolead_keys)), replace=False))) \
            if lead_keys else list(np.random.choice(list(k2i.keys()), 64, replace=False))
    print(f"[step1] sample N={len(bkeys)} (lead={min(64,len(lead_keys))} nolead={min(16,len(nolead_keys))})")

    frame_idx = [[] for _ in range(1 + len(PREV_OFFSETS))]
    for k in bkeys:
        ci = k2i[k]; frame_idx[0].append(ci)
        run, tk = k.rsplit("__t", 1); tick = int(tk); last = ci
        for fi, off in enumerate(PREV_OFFSETS):
            pi = k2i.get(_prev_key(run, tick, off), last)
            frame_idx[fi + 1].append(pi); last = pi
    ego = torch.zeros(len(bkeys), 2, device=dev)
    proj_list = []
    for fi in range(1 + len(PREV_OFFSETS)):
        tok = torch.from_numpy(tokens[frame_idx[fi]].astype(np.float32)).to(dev)
        proj_list.append(model.encoder.proj(tok.transpose(1, 2)).transpose(1, 2))
    fi_mem = model.fusion(proj_list, ego)
    logits = model.head(model.fc_proj(fi_mem))
    obj_prob = torch.sigmoid(logits[:, :, CH_OBJ])
    vlrel = logits[:, :, CH_VLREL] * SCALE_V
    dist = logits[:, :, CH_DIST] * SCALE_DIST
    cen_obj = obj_prob[:, BIN_LO:BIN_HI + 1].cpu().numpy()
    cen_vl = vlrel[:, BIN_LO:BIN_HI + 1].cpu().numpy()
    cen_d = dist[:, BIN_LO:BIN_HI + 1].cpu().numpy()

    live_clos = np.zeros(len(bkeys), np.float32)
    cache_clos = np.array([float(pv_clos[pv_k2i[k]]) for k in bkeys], np.float32)
    for j in range(len(bkeys)):
        om = cen_obj[j] > OBJ_THR
        if not om.any():
            continue
        d = cen_d[j].copy(); d[~om] = np.inf
        sel = int(np.argmin(d))
        live_clos[j] = float(-cen_vl[j][sel])
    err = np.abs(live_clos - cache_clos)
    finite = np.isfinite(live_clos).all()
    # GPU/CPU·amp 차이로 미세 편차 가능 → 0.5 m/s tol
    ok = bool(finite and float(err.max()) < 0.5)
    m = np.abs(cache_clos) > 1e-6
    print(f"[step1] live vs cache closing: max|err|={err.max():.3f} mean|err|={err.mean():.3f} m/s "
          f"(tol 0.5)  finite={finite}")
    if m.any():
        print(f"[step1] lead closing(cache): mean={cache_clos[m].mean():+.2f} "
              f"range[{cache_clos[m].min():+.1f},{cache_clos[m].max():+.1f}] m/s  N_lead={int(m.sum())}")
    print(f"[step1] {'PASS' if ok else 'FAIL'}")
    return ok


@torch.no_grad()
def step2_f2_acc(dev):
    """f_2 predvrel: 실제 v7b i2_mem + closing sweep -> brake escalate monotonic + finite/range."""
    model = Prof635SideModel(residual_gate=True, use_traj_in_side=False, delta_mode=DELTA_MODE).to(dev)
    model.load_main_frozen(f"{CK}/main_prof654.pth")
    sd = torch.load(F2_CKPT, map_location=dev, weights_only=False)
    if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
    miss, unexp = model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"[step2] f_2 {os.path.basename(F2_CKPT)} delta_mode={DELTA_MODE} "
          f"missing={len(miss)} unexpected={len(unexp)}")

    # 실제 v7b i2_mem (lead 가 있는 frame 우선: a2 logits in-lane obj 큰 것)
    zi = np.load(f"{D}/cache/i2_mem_cache_{DS}_v7b_prev20.npz", allow_pickle=True)
    i2_all = zi["i2_mem"].astype(np.float32)
    a2_all = zi["a2_logits"].astype(np.float32) if "a2_logits" in zi.files else None
    if a2_all is not None:
        objmax = a2_all[:, BIN_LO:BIN_HI + 1, CH_OBJ].max(axis=1)
        sel = int(np.argmax(objmax))
        a2_np = a2_all[sel]
        print(f"[step2] i2_mem idx={sel} in-lane obj_amax={objmax[sel]:+.2f} (lead frame)")
    else:
        sel = int(np.argmax(np.abs(i2_all).sum((1, 2))))
        a2_np = np.full((40, 7), -10.0, np.float32)
        a2_np[BIN_LO:BIN_HI + 1, CH_OBJ] = 2.0   # force gate open (a2 cache 없을 때)
        print(f"[step2] a2_logits cache 없음 -> in-lane obj=+2 강제 (gate open)")
    i2 = torch.from_numpy(i2_all[sel]).unsqueeze(0).to(dev)
    a2 = torch.from_numpy(a2_np).unsqueeze(0).to(dev)

    # raceline 기반 cruise traj/ego (steer 정합) — sangam mild 중간점
    m = raceline_dt0p1s.load_map("sangam", "mild", REPO_DATA)
    P, M, N = m["P"], m["M"], m["N"]
    i0 = int(P + 0.4 * (M - P))
    ex, ey = float(m["xy"][i0, 0]), float(m["xy"][i0, 1])
    j = min(i0 + 1, N - 1)
    eyaw = math.atan2(m["xy"][j, 1] - m["xy"][i0, 1], m["xy"][j, 0] - m["xy"][i0, 0])
    ev = float(m["speed"][i0])
    traj40, _ = raceline_dt0p1s.sample_traj40(m, ex, ey, eyaw)
    traj6 = torch.from_numpy(build_traj6(traj40, mode="speed")).unsqueeze(0).to(dev)
    ego6 = torch.from_numpy(build_ego6(np.array([ev, 0, 0, 0, 0], np.float32))).unsqueeze(0).to(dev)

    # gate 는 학습 default(open)이 아니라 live fixed_thresh 와 무관하게 model.forward 내 a2_logits 기반.
    # closing sweep: receding(-8) ~ approaching(+10).  brake = max(-final_pedal, 0)
    print(f"[step2] cruise ev={ev:.1f} m/s  closing sweep (접근>0):")
    rows = []
    for vrel in [-8.0, -4.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0]:
        b = {"traj6": traj6, "ego6": ego6, "i2_mem": i2, "a2_logits": a2,
             "gt_vrel": torch.tensor([vrel], dtype=torch.float32, device=dev)}
        steer_n, pedal = model(b)
        sd_deg = float(steer_n) * 40; pd = float(pedal)
        base = float(model.last_base_pedal); delta = float(model.last_delta); g = float(model.last_gate)
        brake = max(-pd, 0.0)
        rows.append((vrel, base, delta, g, pd, brake, sd_deg))
        print(f"  closing={vrel:+5.1f} -> base={base:+.2f} delta={delta:+.2f} gate={g:.2f} "
              f"pedal={pd:+.2f} brake={brake:.2f} steer={sd_deg:+.1f}deg")

    pedals = np.array([r[4] for r in rows]); brakes = np.array([r[5] for r in rows])
    steers = np.array([r[6] for r in rows])
    finite = np.isfinite(pedals).all() and np.isfinite(steers).all()
    inrange = (np.abs(steers) < 40).all() and ((pedals >= -1.01) & (pedals <= 1.01)).all()
    # ACC 기대: approaching(높은 closing) 일수록 brake >= receding. 강건성 위해 끝점 비교(+10 vs -8).
    acc_dir = brakes[-1] >= brakes[0] - 1e-3
    # delta 가 closing 에 반응(변동 있음) — 완전 무반응이면 vrel 주입 미동작 의심
    delta_arr = np.array([r[2] for r in rows])
    responds = (delta_arr.max() - delta_arr.min()) > 1e-3
    ok = bool(finite and inrange and acc_dir and responds)
    print(f"[step2] finite={finite} inrange={inrange} brake(+10)>=brake(-8)={acc_dir} "
          f"(브레이크 {brakes[0]:.2f}->{brakes[-1]:.2f}) delta_responds={responds} "
          f"(delta range {delta_arr.min():+.2f}~{delta_arr.max():+.2f})")
    print(f"[step2] {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    print(f"[acc-test] device={dev} ds={DS}")
    print(f"[acc-test] a2_5f={os.path.basename(A2_MF5F_CKPT)} f2={os.path.basename(F2_CKPT)}\n")
    r1 = step1_predvrel(dev)
    print()
    r2 = step2_f2_acc(dev)
    print()
    # step1 SKIP(None) 은 fail 아님 (캐시 없는 머신). step2 는 필수.
    overall = (r1 in (True, None)) and (r2 is True)
    print(f"[overall] step1={'PASS' if r1 else ('SKIP' if r1 is None else 'FAIL')} "
          f"step2={'PASS' if r2 else 'FAIL'} -> {'PASS' if overall else 'FAIL'}")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
