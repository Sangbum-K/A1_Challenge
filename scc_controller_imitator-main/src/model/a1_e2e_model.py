"""A1 CHALLENGE 제출용 통합 단일 모델 (A1E2EModel / A1E2EModelSCC).

목적 (규정 §1.1 단일 모델 파일 + §1.3 differentiable single chain):
  seg backbone(tokenizer) + a_2(perception) + prof635 main(steer) + f_2 side(pedal)
  + learned temporal gate 를 하나의 nn.Module.forward 로 통합.

  현 closed_loop_node 는 gate 평활을 numpy/deque glue 로 처리(`a2_logits.cpu().numpy()`
  + `deque` window + python sigmoid) → forward 밖에서 그래프가 끊김. 본 모듈은 gate window
  를 **tensor 입력**(obj_window (B,K,10))으로 받아 forward 내부에서 통과시키므로
  detach/numpy/deque/EMA 없는 단일 미분 그래프.

  A1E2EModel(beat): side = brake_only delta + per-frame gate(side 내부) 를 우회하고
    learned temporal gate(beat) 로 final = base + g*delta. (예선 구성, 참고)
  A1E2EModelSCC(acc): side = no_gate + delta_mode='raw' (clamp 없는 raw delta) 를 그대로
    써서 base + g*delta, g = learned temporal gate. delta 가 no-lead→0/lead→brake 를 직접
    흡수하므로 side 내부 gate 과개방(no-lead g=0.23~0.33 → 풀악셀 깎임) 문제를 회피.

forward 입력 (모두 tensor, batch_first):
  image_curr      (B, 3, H, W)       ROI normalized
  image_prev      (B, 3, H, W)       ROI normalized (200ms 전, yaw-shift 전 raw)
  d_yaw           (B,)               ego.yaw - prev_yaw (rad) — token-level shift
  ego5            (B, 5)             [vx, vy, ax, ay, yaw_rate]
  traj40          (B, 40, 3)         dt0p1s waypoint [x_fwd, y_left, speed]
  obj_window      (B, K, 10)         최근 K프레임 in-lane(col 16..25) a_2 obj logit window.

forward 출력:
  steer_deg       (B,)               steering (deg)
  pedal           (B,)               [-1,1], 양수=accel, 음수=brake
  aux             dict (디버그용 — gate, base_pedal, delta, a2_logits ...)

state_dict 통합:
  모든 sub-module(seg backbone / a2 / main / side / gate) 가 self 의 child 라
  단일 .pt 로 save/load 가능. build_and_load_*() 가 분리 ckpt 들을 합쳐 적재.

규정 메모:
  - §1.1: self 가 하나의 nn.Module 이고 모든 sub-module 이 child → state_dict() 단일 .pt.
  - §1.3: forward 가 입력→출력 단일 autograd 그래프. forward 내부에 no_grad/detach/
    numpy/deque/EMA 없음. 모든 params requires_grad_(False) frozen → 거동/수치 불변,
    오직 그래프 연결성(미분가능성)만 보장.

gate ckpt 메모:
  - scc(acc): scc_gate_mlp_beat.pt (TemporalMLP K=16, 예선 beat track 재학습 — flip 1.67%,
    recall 73.9%, jerk 0.0141). 동일 module/K 구조라 load_state_dict 호환.
  - closed_loop_node.py 의 GATE_CKPT 실시간 경로(base + g*delta)와 동일 gate 식.
"""
from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .image_models.a2_model import A2Model
from .image_models.prof635_side_model import Prof635SideModel
from .image_models.iy_model import yaw_shift_tokens_batch
from ..utils.temporal_gate import make_module

# build_traj6 / build_ego6 의 tensor 판본 (numpy 의존 제거 — 단일 미분 그래프).
T6 = (0.05, 0.30, 0.55, 0.80, 1.05, 1.30)
WB = 2.35
STEER_NORM = 40.0


def _interp1d_batch(t_query, t_src, y):
    """선형보간 (numpy.interp tensor 판). t_query (Q,), t_src (S,), y (B,S) → (B,Q).
    t_src 는 등간격 0,0.1,...,3.9 (TSRC) — clamp + gather 로 미분가능 보간.
    """
    B, S = y.shape
    Q = t_query.shape[0]
    dt = (t_src[-1] - t_src[0]) / (S - 1)
    pos = (t_query - t_src[0]) / dt                    # (Q,) float index
    pos = pos.clamp(0.0, S - 1 - 1e-4)
    i0 = pos.floor().long()                            # (Q,)
    i1 = (i0 + 1).clamp(max=S - 1)
    w = (pos - i0.float()).unsqueeze(0).expand(B, Q)   # (B,Q)
    y0 = y.gather(1, i0.unsqueeze(0).expand(B, Q))
    y1 = y.gather(1, i1.unsqueeze(0).expand(B, Q))
    return y0 * (1 - w) + y1 * w


class A1E2EModel(nn.Module):
    """통합 단일 모델 (공통 베이스).

    mode:
      'beat' : side = delta_mode(brake_only 기본) + side 내부 gate 우회('open') →
               final = base + g*delta, g = learned temporal gate(beat).
      'scc'  : side = no_gate + delta_mode='raw' (residual_gate=True, use_traj_in_side=False).
               side.forward 의 no_gate 분기는 g 없이 final=base+delta 라, 본 모듈은 side 의
               last_base_pedal/last_delta 만 취하고 final = base + g*delta (g=temporal gate)
               를 본 forward 에서 재구성한다. → A1E2EModel-scc.

    구성요소 ckpt:
      seg backbone : seg_pretrain_fx640_v3_backbone.pt  (tokenizer)
      a_2          : a2_v7b_d128_alpha075_best_loss.pt   (perception, d_out=128)
      main         : main_prof654.pth                    (steer, frozen)
      side         : scc_p3_ng.pt(scc) / prof635_side_distbrake_v1.pt(beat)
      gate         : scc_gate_mlp_beat.pt(scc) / learned_temporal_gate_mlp_beat.pt(beat)
                     (TemporalMLP, K=16)
    """

    def __init__(self, mode: str = "scc", backbone_name: str = "tf_efficientnetv2_s",
                 fx: float = 640.0, gate_K: int = 16, gate_module: str = "mlp",
                 delta_mode: str = "raw"):
        super().__init__()
        assert mode in ("beat", "scc")
        self.mode = mode
        self.fx = float(fx)
        self.gate_K = int(gate_K)
        self.delta_mode = delta_mode
        # seg backbone = tokenizer (features_only, last stage)
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, features_only=True, out_indices=[-1])
        # a_2 (encoder.backbone 은 prof635 경로에서 미사용 — encoder.proj/fusion/fc_proj/head 만 사용)
        self.a2 = A2Model(img_backbone=backbone_name, d_token=256, n_heads=4, n_ch=7, d_out=128)
        # prof635 main(steer, frozen) + f_2 side(pedal)
        if mode == "scc":
            # A1E2EModel-scc: no_gate + raw delta. residual_gate=True, traj 미사용.
            self.side = Prof635SideModel(residual_gate=True, use_traj_in_side=False,
                                         delta_mode=delta_mode, no_gate=True)
        else:
            self.side = Prof635SideModel(residual_gate=True, use_traj_in_side=False,
                                         delta_mode=delta_mode)
        # learned temporal gate
        self.gate = make_module(gate_module, gate_K)
        # register T6 / TSRC as buffers (device 따라감, 단일 그래프)
        self.register_buffer("_t6", torch.tensor(T6, dtype=torch.float32))
        self.register_buffer("_tsrc", torch.arange(40, dtype=torch.float32) * 0.1)

    # ---- traj6 / ego6 (tensor 판, 미분가능) ----
    def _build_traj6(self, traj40):
        """traj40 (B,40,3)[x_fwd,y_left,speed] → traj6 (B,6,3)[-y, x, speed] (T6 보간)."""
        x = _interp1d_batch(self._t6, self._tsrc, traj40[:, :, 0])   # (B,6)
        y = _interp1d_batch(self._t6, self._tsrc, traj40[:, :, 1])
        v = _interp1d_batch(self._t6, self._tsrc, traj40[:, :, 2])
        return torch.stack([-y, x, v], dim=-1)                       # (B,6,3)

    def _build_ego6(self, ego5):
        """ego5 (B,5)[vx,vy,ax,ay,yaw_rate] → ego6 (B,6)[+tire_angle(kinematic)].
        |vx|<5 → tire=0 (closed_loop build_ego6 와 동일 규약).
        """
        vx, vy, ax, ay, yr = [ego5[:, i] for i in range(5)]
        tire = torch.atan(WB * yr / vx.clamp_min(1e-3))
        tire = torch.where(vx.abs() >= 5.0, tire, torch.zeros_like(tire))
        return torch.stack([vx, vy, ax, ay, tire, yr], dim=-1)       # (B,6)

    # ---- perception: 두 이미지 → i2_mem + a2_logits ----
    def _perceive(self, image_curr, image_prev, d_yaw):
        """seg backbone tokenizer → a_2 fusion → (i2_mem (B,40,128), a2_logits (B,40,7)).
        closed_loop prof635 경로와 동일: ego_mot=0, token-level yaw-shift, adaptive_avg_pool2d(1,40).
        """
        fc = self.backbone(image_curr)[-1]                                   # (B,C,~8,~40)
        fc = F.adaptive_avg_pool2d(fc, (1, 40)).squeeze(2).transpose(1, 2)    # (B,40,C)
        fp = self.backbone(image_prev)[-1]
        fp = F.adaptive_avg_pool2d(fp, (1, 40)).squeeze(2).transpose(1, 2)
        fx_t = torch.full_like(d_yaw, self.fx)
        fp_rot = yaw_shift_tokens_batch(fp, d_yaw, fx=fx_t)
        tc_proj = self.a2.encoder.proj(fc.transpose(1, 2)).transpose(1, 2)    # Conv1d C→256
        tp_proj = self.a2.encoder.proj(fp_rot.transpose(1, 2)).transpose(1, 2)
        ego_mot = torch.zeros(image_curr.shape[0], 2, device=image_curr.device,
                              dtype=tc_proj.dtype)                            # 학습 cache 와 동일(=0)
        fi_mem = self.a2.fusion(tc_proj, tp_proj, ego_mot)                    # (B,40,256)
        i2_mem = self.a2.fc_proj(fi_mem)                                     # (B,40,128)
        a2_logits = self.a2.head(i2_mem)                                    # (B,40,7)
        return i2_mem, a2_logits

    # ---- side base+delta (단일 그래프, no_grad/detach 없음) ----
    def _side_base_delta(self, traj6, ego6, i2_mem, a2_logits):
        """side.forward 와 수치 동일하되 main no_grad 를 제거해 단일 그래프로 연결.

        side.forward(eval) 가 하는 연산을 그대로 inline:
          out_main = main(traj6, ego6)[:,0,:]  → steer, base_pedal
          q = side_state_embed(ego6) + side_pedal_embed(base_pedal)   (use_traj_in_side=False)
          side = side_dec(q, img_proj(i2_mem))
          head = pedal_head(side[:,0,:])
          delta = (raw: head | brake_only: -2σ(head) | signed: 2tanh(head))
        main params 는 frozen(requires_grad_(False)) 이라 no_grad 없이도 수치 불변,
        그래프 연결성(traj6/ego6 → steer/base_pedal)만 확보됨 (§1.3).
        """
        s = self.side
        out_main = s.main(traj6, ego6)[:, 0, :]              # (B,2) [steer/40, base_pedal]
        steer_n = out_main[:, 0]
        base_pedal = out_main[:, 1]
        base_used = base_pedal                               # eval: randomize_base 미적용
        ped_emb = s.side_pedal_embed(base_used.unsqueeze(-1)).unsqueeze(1)  # (B,1,d)
        # use_traj_in_side=False 경로 (의도): image+ego+base_pedal 만
        q = s.side_state_embed(ego6).unsqueeze(1) + ped_emb  # (B,1,d)
        kv = s.img_proj(i2_mem)                              # (B,40,d)
        side = s.side_dec(q, kv)                             # (B,1,d)
        head = s.pedal_head(side[:, 0, :]).squeeze(-1)       # (B,)
        if s.delta_mode == "brake_only":
            delta = -2.0 * torch.sigmoid(head)
        elif s.delta_mode == "raw":
            delta = head
        else:
            delta = 2.0 * torch.tanh(head)
        return steer_n, base_used, delta

    def forward(self, image_curr, image_prev, d_yaw, ego5, traj40, obj_window):
        """단일 미분 그래프 (§1.3 differentiable single chain).

        seg → a_2 → f_1(main) → f_2 → gate → (steer, pedal) 가 끊김 없는 단일 autograd
        그래프로 연결됨. forward 내부에 torch.no_grad()/`.detach()` 없음 (numpy/deque/EMA 도 없음).
        모든 sub-module params 는 requires_grad_(False)(frozen)라 거동/수치는 변하지 않으며,
        오직 입력→출력 그래프 연결성(미분가능성)만 모델 정의에 보장됨.

        ★배포 추론 시 메모리/속도: 호출측에서 `with torch.inference_mode():` 로 감쌀 것.
          미분가능성은 모델 정의(forward)에 있어야 하므로 forward 자체에는 no_grad 를 두지 않음.

        obj_window (B,K,10): caller 가 매 frame in-lane obj logit 을 tensor 로 stack 해 주입.
        gate net 출력 g 로 final_pedal = base + g*delta.
        """
        i2_mem, a2_logits = self._perceive(image_curr, image_prev, d_yaw)
        traj6 = self._build_traj6(traj40)
        ego6 = self._build_ego6(ego5)
        steer_n, base_pedal, delta = self._side_base_delta(traj6, ego6, i2_mem, a2_logits)
        # learned temporal gate (tensor window → MLP → g∈[0,1])
        g = self.gate(obj_window)                            # (B,)
        final_pedal = (base_pedal + g * delta).clamp(-1.0, 1.0)
        steer_deg = steer_n * STEER_NORM
        aux = {"gate": g, "base_pedal": base_pedal, "delta": delta,
               "a2_logits": a2_logits, "i2_mem": i2_mem}
        return steer_deg, final_pedal, aux

    # ---- 통합 state_dict 적재 (분리 ckpt → 단일 모델) ----
    def load_components(self, seg_backbone_ckpt, a2_ckpt, main_ckpt, side_ckpt,
                        gate_ckpt, map_location="cpu"):
        """분리 ckpt 들을 합쳐 적재. 이후 self.state_dict() 를 단일 .pt 로 save 가능."""
        log = {}
        # 1) seg backbone (tokenizer)
        sd = torch.load(seg_backbone_ckpt, map_location=map_location, weights_only=False)
        own = self.backbone.state_dict(); n = 0
        for k, v in sd.items():
            k2 = k.replace("backbone.", "")
            if k2 in own and own[k2].shape == v.shape:
                own[k2] = v; n += 1
        self.backbone.load_state_dict(own); log["seg_backbone_keys"] = n
        # 2) a_2 (v7b)
        sd6 = torch.load(a2_ckpt, map_location=map_location, weights_only=False)
        if isinstance(sd6, dict) and "model" in sd6:
            sd6 = sd6["model"]
        own6 = self.a2.state_dict()
        sd6 = {k: v for k, v in sd6.items() if k in own6 and own6[k].shape == v.shape}
        m6, u6 = self.a2.load_state_dict(sd6, strict=False)
        log["a2_missing"], log["a2_unexpected"] = len(m6), len(u6)
        # 3) main(frozen) — side ckpt 에 main 이 내장돼 있어도 명시 로드(일관성)
        self.side.load_main_frozen(main_ckpt)
        # 4) side(f_2) — scc_p3_ng.pt 는 main+side_*+pedal_head+gate_head 전체 포함
        sds = torch.load(side_ckpt, map_location=map_location, weights_only=False)
        if isinstance(sds, dict) and "model" in sds:
            sds = sds["model"]
        ms, us = self.side.load_state_dict(sds, strict=False)
        log["side_missing"], log["side_unexpected"] = len(ms), len(us)
        # 5) gate (temporal)
        gk = torch.load(gate_ckpt, map_location=map_location, weights_only=False)
        self.gate.load_state_dict(gk["state"])
        log["gate_module"], log["gate_K"] = gk["module"], int(gk["K"])
        return log


def build_and_load_beat(ckpt_dir, map_location="cpu", delta_mode="brake_only", fx=640.0):
    """beat 구성 분리 ckpt → 통합 A1E2EModel (eval). 단일 .pt 로 export 가능."""
    m = A1E2EModel(mode="beat", fx=fx, gate_K=16, gate_module="mlp", delta_mode=delta_mode)
    log = m.load_components(
        seg_backbone_ckpt=os.path.join(ckpt_dir, "seg_pretrain_fx640_v3_backbone.pt"),
        a2_ckpt=os.path.join(ckpt_dir, "a2_v7b_d128_alpha075_best_loss.pt"),
        main_ckpt=os.path.join(ckpt_dir, "main_prof654.pth"),
        side_ckpt=os.path.join(ckpt_dir, "prof635_side_distbrake_v1.pt"),
        gate_ckpt=os.path.join(ckpt_dir, "learned_temporal_gate_mlp_beat.pt"),
        map_location=map_location)
    for p in m.parameters():
        p.requires_grad_(False)
    m.eval()
    return m, log


def build_and_load_scc(ckpt_dir, map_location="cpu", fx=640.0,
                       side_ckpt="scc_p3_ng.pt",
                       gate_ckpt="scc_gate_mlp_beat.pt"):
    """A1E2EModel-scc 구성 분리 ckpt → 통합 모델 (eval). 단일 .pt 로 export 가능.

    seg(seg_pretrain_fx640_v3_backbone.pt) + a2(a2_v7b_d128_alpha075_best_loss.pt)
    + main(main_prof654.pth) + side(scc_p3_ng.pt: no_gate raw, main654 내장)
    + gate(scc_gate_mlp_beat.pt: TemporalMLP K=16, beat track 재학습 — flip 1.67%,
      recall 73.9%, jerk 0.0141. 동일 module/K 구조라 load_state_dict 호환) 적재.

    §1.3 단일 그래프 보장 + §1.1 단일 state_dict export.
    """
    m = A1E2EModel(mode="scc", fx=fx, gate_K=16, gate_module="mlp", delta_mode="raw")
    log = m.load_components(
        seg_backbone_ckpt=os.path.join(ckpt_dir, "seg_pretrain_fx640_v3_backbone.pt"),
        a2_ckpt=os.path.join(ckpt_dir, "a2_v7b_d128_alpha075_best_loss.pt"),
        main_ckpt=os.path.join(ckpt_dir, "main_prof654.pth"),
        side_ckpt=os.path.join(ckpt_dir, side_ckpt),
        gate_ckpt=os.path.join(ckpt_dir, gate_ckpt),
        map_location=map_location)
    # §1.3: params frozen 유지 (거동/수치 불변, 오직 그래프 연결성만 확보 — 학습 안 함).
    for p in m.parameters():
        p.requires_grad_(False)
    m.eval()
    return m, log
