"""A1E2EModel-scc 검증 (반드시 CPU: CUDA_VISIBLE_DEVICES="").

검증 항목 (규정 §1.1 단일 state_dict + §1.3 differentiable single chain):
  [1] 단일 state_dict export/reload  → missing=0, unexpected=0.
  [2] forward 단일 그래프            → 입력 requires_grad → (steer.sum()+pedal.sum()).backward()
                                        → image_curr/ego5/traj40 grad not None (seg backbone 까지 연결).
  [3] 분리경로 수치 일치             → side(no_gate raw) 단독 base+delta 추출 후 동일 g 로 손계산한
                                        base+g*delta == A1E2EModel-scc forward 의 final.

사용:
  CUDA_VISIBLE_DEVICES="" python scripts/test_a1_e2e_scc.py
"""
from __future__ import annotations
import os
import sys
import tempfile

import torch

# repo root (이 파일 = <root>/scripts/test_a1_e2e_scc.py)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.model.a1_e2e_model import A1E2EModel, build_and_load_scc, STEER_NORM  # noqa: E402

CKPT_DIR = os.path.join(ROOT, "data", "ckpts")
torch.manual_seed(0)


def make_inputs(B=2, H=250, W=1280, K=16, requires_grad=False,
                img_scale=0.05, traj_scale=0.05, obj_bias=1.0, seed=None):
    """검증 입력 생성.

    grad-flow 검증용 기본값(img_scale/traj_scale 작게, obj_bias 적당)은 base+g*delta 가
    clamp 비포화 영역 (-1,1) 에 오도록 잡는다. clamp(±1) 는 포화 시 grad=0 이 되므로(정상
    수학), 그래프 *연결성* 검증이 포화로 가려지지 않게 현실적 작은 스케일을 쓴다.
    obj_bias>0 → gate ON (g≈1) → g*delta 경로(image 의존)에 grad 가 확실히 흐름.
    """
    if seed is not None:
        torch.manual_seed(seed)
    image_curr = torch.randn(B, 3, H, W) * img_scale
    image_prev = torch.randn(B, 3, H, W) * img_scale
    d_yaw = torch.randn(B) * 0.03                 # nonzero → prev token yaw-shift 경로 활성
    ego5 = torch.zeros(B, 5)
    ego5[:, 0] = 8.0                              # vx > 5 (tire 분기 활성)
    ego5 = ego5 + torch.randn(B, 5) * traj_scale
    traj40 = torch.randn(B, 40, 3) * traj_scale
    obj_window = torch.randn(B, K, 10) * 0.1 + obj_bias    # gate ON (적당)
    if requires_grad:
        for t in (image_curr, image_prev, ego5, traj40, obj_window):
            t.requires_grad_(True)
    return dict(image_curr=image_curr, image_prev=image_prev, d_yaw=d_yaw,
                ego5=ego5, traj40=traj40, obj_window=obj_window)


def fmt(ok):
    return "PASS" if ok else "FAIL"


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES", None) == "", \
        "이 검증은 반드시 CPU 에서: CUDA_VISIBLE_DEVICES=\"\" 로 실행"
    results = {}

    print("=== build_and_load_scc ===")
    m, log = build_and_load_scc(CKPT_DIR, map_location="cpu")
    m.eval()
    print("  load log:", log)
    # side / gate 가 scc 구성인지
    print("  side.no_gate =", m.side.no_gate, "| side.delta_mode =", m.side.delta_mode,
          "| mode =", m.mode)
    assert m.side.no_gate is True and m.side.delta_mode == "raw"

    # ---------------------------------------------------------------
    # [1] 단일 state_dict export / reload
    # ---------------------------------------------------------------
    print("\n=== [1] single state_dict export/reload ===")
    with tempfile.TemporaryDirectory() as td:
        pt = os.path.join(td, "a1_e2e_scc_export.pt")
        torch.save(m.state_dict(), pt)
        # 새 모듈 생성 후 통째 reload (분리 ckpt 없이 단일 .pt 만으로)
        m2 = A1E2EModel(mode="scc", fx=640.0, gate_K=16, gate_module="mlp", delta_mode="raw")
        sd = torch.load(pt, map_location="cpu", weights_only=False)
        missing, unexpected = m2.load_state_dict(sd, strict=False)
        n_params = len(sd)
        print(f"  exported {n_params} tensors | missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("  missing sample:", list(missing)[:5])
        if unexpected:
            print("  unexpected sample:", list(unexpected)[:5])
        ok1 = (len(missing) == 0 and len(unexpected) == 0)
        results["[1] reload missing=0 unexpected=0"] = ok1
        m2.eval()
        # reload 모델이 같은 출력 내는지 (sanity)
        for p in m2.parameters():
            p.requires_grad_(False)

    # ---------------------------------------------------------------
    # [2] forward 단일 그래프 (grad flow)
    # ---------------------------------------------------------------
    print("\n=== [2] single autograd graph (grad flow) ===")
    # task 명세: 입력 requires_grad → (steer.sum()+pedal.sum()).backward()
    #   → image_curr/ego5/traj40 grad not None. clamp(±1) 는 포화 시 grad=0(정상 수학)이라
    #   그래프 *연결성* 검증이 포화로 가려지지 않게 clamp 비포화 입력(작은 scale, gate ON)을 쓴다.
    inp = make_inputs(requires_grad=True, seed=11)
    steer, pedal, aux = m(**inp)
    pre_clamp = aux["base_pedal"] + aux["gate"] * aux["delta"]
    sat = ((pre_clamp > 1.0) | (pre_clamp < -1.0))
    print(f"  pre-clamp pedal={[round(x,4) for x in pre_clamp.tolist()]} "
          f"| clamp saturated={sat.tolist()} (False 면 grad 비포화)")
    loss = steer.sum() + pedal.sum()
    loss.backward()
    grad_ok = {}
    for name in ("image_curr", "ego5", "traj40"):
        g = inp[name].grad
        grad_ok[name] = (g is not None) and torch.isfinite(g).all().item() and (g.abs().sum().item() > 0)
    print("  (steer.sum()+pedal.sum()).backward() → input grads not-None & finite & nonzero:")
    for k, v in grad_ok.items():
        gs = None if inp[k].grad is None else float(inp[k].grad.abs().sum())
        print(f"    {k:12s}: {fmt(v)}  (grad_sum={gs:.4g})")
    # 보조: raw delta 경로 그래프 연결성 (clamp/gate 무관). delta=image+a2 의존이라 image grad nonzero 여야.
    inp_d = make_inputs(requires_grad=True, seed=23)
    _, _, aux_d = m(**inp_d)
    aux_d["delta"].sum().backward()
    delta_to_img = (inp_d["image_curr"].grad is not None) and (inp_d["image_curr"].grad.abs().sum().item() > 0)
    delta_to_prev = (inp_d["image_prev"].grad is not None) and (inp_d["image_prev"].grad.abs().sum().item() > 0)
    print(f"  delta.sum().backward() → image_curr grad: {fmt(delta_to_img)} "
          f"| image_prev grad: {fmt(delta_to_prev)} (seg backbone 까지 연결)")
    # 출력 grad_fn 존재 (forward 내 no_grad/detach 없음의 증거)
    has_grad_fn = (steer.grad_fn is not None) and (pedal.grad_fn is not None)
    print(f"  steer/pedal grad_fn present: {fmt(has_grad_fn)}")
    ok2 = all(grad_ok.values()) and delta_to_img and delta_to_prev and has_grad_fn
    results["[2] grad flow (image/ego/traj → backbone, single graph)"] = ok2

    # ---------------------------------------------------------------
    # [3] 분리경로 수치 일치
    #   side(no_gate raw) 단독으로 base+delta 추출 → 동일 g 로 base+g*delta 손계산
    #   == A1E2EModel-scc forward final.
    # ---------------------------------------------------------------
    print("\n=== [3] decoupled-path numeric match ===")
    with torch.no_grad():
        inp3 = make_inputs(requires_grad=False)
        steer3, pedal3, aux3 = m(**inp3)
        g = aux3["gate"]                                  # (B,) 동일 gate
        # side 단독 경로: A1E2EModel-scc 와 동일하게 perceive→traj6/ego6→side.forward
        i2_mem, a2_logits = m._perceive(inp3["image_curr"], inp3["image_prev"], inp3["d_yaw"])
        traj6 = m._build_traj6(inp3["traj40"])
        ego6 = m._build_ego6(inp3["ego5"])
        steer_side, final_side_nogate = m.side(
            {"traj6": traj6, "ego6": ego6, "i2_mem": i2_mem, "a2_logits": a2_logits})
        base = m.side.last_base_pedal                     # (B,)
        delta = m.side.last_delta                         # (B,) raw
        # side(no_gate) 단독 final 은 g 없이 base+delta. 동일 g 로 손계산:
        final_recon = (base + g * delta).clamp(-1.0, 1.0)
        steer_side_deg = steer_side * STEER_NORM

        max_pedal_err = (final_recon - pedal3).abs().max().item()
        max_steer_err = (steer_side_deg - steer3).abs().max().item()
        # side(no_gate) 단독 final (= base+delta, g=1 효과) vs base+1*delta clamp 도 참고 출력
        print(f"  base sample      : {base.tolist()}")
        print(f"  delta sample     : {delta.tolist()}")
        print(f"  g (temporal gate): {g.tolist()}")
        print(f"  forward pedal    : {pedal3.tolist()}")
        print(f"  recon base+g*delta: {final_recon.tolist()}")
        print(f"  max |pedal err|  : {max_pedal_err:.3e}")
        print(f"  max |steer err|  : {max_steer_err:.3e}")
    ok3 = (max_pedal_err < 1e-5) and (max_steer_err < 1e-5)
    results["[3] decoupled base+g*delta == forward final"] = ok3

    # ---------------------------------------------------------------
    print("\n=== SUMMARY ===")
    allok = True
    for k, v in results.items():
        print(f"  {fmt(v)}  {k}")
        allok = allok and v
    print("\nRESULT:", "ALL PASS" if allok else "FAIL")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
