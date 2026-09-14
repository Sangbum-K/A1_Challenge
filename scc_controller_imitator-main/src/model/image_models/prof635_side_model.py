"""#635 frozen main (steer) + f_2 side (pedal).

main: controller_imitator #635 (SelfAttentionControlPredictor_Professor) — FROZEN.
      입력 traj6 (6,3)[-y,x,abs_time], ego6 (6) → out (.,2): [steer/40, accel_brake].
      steer 만 사용 (out[...,0]). pred_first_only.
f_2 : pedal 전담 (종방향). traj6+ego6 (trainable embed) self-attn + i2_mem cross-attn
      → pedal scalar (tanh, -1~1). raceline pedal + lead brake 모두 학습.

최종: steer = main_steer, pedal = f_2_pedal.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class _PosEnc(nn.Module):
    def __init__(self, d, max_len=41):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class Prof635Main(nn.Module):
    """#635 구조 그대로 (frozen 로드용). traj_dim=3, state_dim=6, d=128, 6L, out 2."""
    def __init__(self, d_model=128, nhead=8, dim_ff=512, num_layers=6, dropout=0.1,
                 traj_dim=3, state_dim=6, num_out=2, seq_len=40):
        super().__init__()
        self.traj_dim = traj_dim; self.state_dim = state_dim
        self.traj_embed = nn.Linear(traj_dim, d_model)
        self.state_embed = nn.Linear(state_dim, d_model)
        self.pos_encoder = _PosEnc(d_model, max_len=seq_len + 1)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, num_out)

    def forward(self, traj, state):
        traj = traj[:, :, : self.traj_dim]
        tok = self.traj_embed(traj) + self.state_embed(state).unsqueeze(1)
        enc = self.transformer_encoder(self.pos_encoder(tok))
        return self.fc_out(enc)   # (B, 6, 2)


class GateHeadMLP(nn.Module):
    """Lead-presence + approach gate head (learned, end-to-end gradient).

    입력  = a_2 in-lane(16-25) obj logit 10-dim (per-bin lead 증거)
            + in-lane closing 10-dim (per-bin 접근 방향, closing=-vL_rel, 양수=다가옴).
            총 2*n_inlane (=20) dim.
    출력  = gate logit 1개 → sigmoid → g ∈ (0,1).
    구조  = Linear(2n→hidden) → GELU → Linear(hidden→1).

    obj 만 보던 기존(obj-only) 한계: receding(멀어지는 lead)에도 obj>0 이라 g 가 켜짐.
    closing 입력 추가 → gate 가 "앞에 lead 가 있고(obj↑) AND 다가옴(closing↑)" 일 때만 열림.
    receding(closing<0) 은 닫고 approaching(closing>0) 만 연다.

    rule(threshold/IIR) 대체. gradient 가 f_2 brake delta loss 로부터 이 MLP 까지 흐름.
    closed_loop live 도 동일 forward 만 사용(별도 평활/임계 없음) → 학습=추론 일관.

    warm-init (sign 정합):
      - 처음 n_inlane hidden 채널 = per-bin obj passthrough (obj_i, identity 대각 1).
        fc2 의 이 채널들 weight = warm_gate_scale → Σ GELU(obj) soft amax (기존 obj 게이팅 보존).
      - 다음 n_inlane hidden 채널 = per-bin closing passthrough (closing_i, identity 대각 1).
        fc2 의 이 채널들 weight = warm_vrel_scale(>0) → closing>0(접근) 일 때 GELU(closing)>0
        → logit↑ → g↑. closing<0(receding) → GELU≈0 → logit 기여 없음(닫힘).
      closing 은 m/s scale (±10) 라 0.5 로 normalize(±5) 후 입력 → GELU saturate 회피.
      bias = -scale·thr (obj 기준 옛 thr). joint 학습으로 obj·closing 결합 분화.
    """
    def __init__(self, n_inlane=10, hidden=24, warm_thresh=0.3, warm_gate_scale=5.0,
                 warm_vrel_scale=1.0):
        super().__init__()
        in_dim = 2 * n_inlane                          # obj(n) + closing(n)
        hidden = max(hidden, in_dim)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.n_inlane = n_inlane
        # warm-init: hidden[0:n]=obj passthrough, hidden[n:2n]=closing passthrough, 나머지 small random.
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        with torch.no_grad():
            eye = torch.eye(n_inlane)
            self.fc1.weight[:n_inlane, :n_inlane] = eye           # h_i = obj_i  (i<n)
            self.fc1.weight[:n_inlane, n_inlane:] = 0.0
            self.fc1.weight[n_inlane:2 * n_inlane, :n_inlane] = 0.0
            self.fc1.weight[n_inlane:2 * n_inlane, n_inlane:] = eye  # h_{n+i} = closing_i
        # fc2: obj 채널 = warm_gate_scale (soft amax obj), closing 채널 = warm_vrel_scale (접근→logit↑).
        #   bias = -scale·thr → logit ≈ scale·(Σ GELU(obj) - thr) + vrel_scale·Σ GELU(closing).
        nn.init.zeros_(self.fc2.weight)
        with torch.no_grad():
            self.fc2.weight[0, :n_inlane] = float(warm_gate_scale)
            self.fc2.weight[0, n_inlane:2 * n_inlane] = float(warm_vrel_scale)
        nn.init.constant_(self.fc2.bias, -float(warm_gate_scale) * float(warm_thresh))

    def forward(self, obj_inlane_logit, closing_inlane=None):   # (B,n), (B,n)
        if closing_inlane is None:                              # 하위호환: closing 없으면 0 (obj-only 거동)
            closing_inlane = torch.zeros_like(obj_inlane_logit)
        x = torch.cat([obj_inlane_logit, closing_inlane], dim=-1)   # (B,2n)
        h = torch.nn.functional.gelu(self.fc1(x))
        return self.fc2(h).squeeze(-1)        # (B,) gate logit


class Prof635SideModel(nn.Module):
    def __init__(self, d_model=128, nhead=8, dim_ff=512, num_layers=6,
                 side_layers=4, side_dim_ff=512, img_dim=128, dropout=0.1,
                 residual_gate=True, use_traj_in_side=False,
                 delta_mode="signed", randomize_base=False, no_gate=False,
                 gate_warm_thresh=0.3, gate_warm_scale=5.0, gate_warm_vrel_scale=1.0):
        super().__init__()
        self.main = Prof635Main(d_model, nhead, dim_ff, num_layers, dropout)
        self.residual_gate = residual_gate   # True: f_2 = base+delta (skip), gate→ final=base+g*delta
        self.no_gate = no_gate   # True: gate(g) 제거 → final=base+delta. delta 가 no-lead→0/lead→brake 직접 흡수
                                 #   (gate no-lead 과개방 → 풀악셀 깎임 문제 회피. delta_mode='raw' 와 함께 사용)
        self.use_traj_in_side = use_traj_in_side  # False(의도): f_2 는 traj 안 봄 (image+ego+base_pedal 만)
        # delta_scale: final = clamp(base + g * delta_scale * delta). 기본 1.0(거동 불변).
        #   >1 이면 f_2 brake delta 를 증폭(강접근 제동↑ 가설 검증용 노브). clamp 유지.
        self.delta_scale = 1.0
        # f_2: (ego, f_1 pedal) query × image(i2_mem) cross-attn → pedal delta
        self.side_traj_embed = nn.Linear(3, d_model)    # use_traj_in_side=True 일 때만 사용
        self.side_state_embed = nn.Linear(6, d_model)
        self.side_pedal_embed = nn.Linear(1, d_model)   # f_1 base_pedal(-1~1) 을 f_2 입력으로 conditioning
        self.side_vrel_embed = nn.Linear(1, d_model)    # [진단용] GT 상대속도(closing) 주입 — batch 에 gt_vrel 있을 때만
        self.side_pos = _PosEnc(d_model, max_len=8)
        self.img_proj = nn.Linear(img_dim, d_model)
        dec = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=side_dim_ff,
                                         dropout=dropout, batch_first=True)
        self.side_dec = nn.TransformerDecoder(dec, num_layers=side_layers)
        # delta_mode: 'signed'(기본) delta=2·tanh ∈[-2,2]; 'brake_only' delta=-2·sigmoid ∈(-2,0]
        #   (f_2 가 가속 못하게 — receding/cruising lead 에 가속 간섭 원천 차단).
        self.delta_mode = delta_mode
        self.randomize_base = randomize_base   # 학습 시 base 를 U[-1,1] 로 치환 → f_2 가 base 무관 절대 target 학습
        self.pedal_head = nn.Linear(d_model, 1)
        if residual_gate:   # warm start delta≈0 → final≈base (#654)
            nn.init.zeros_(self.pedal_head.weight)
            # signed: bias 0 (tanh0=0). brake_only: bias -4 (sigmoid(-4)≈0.018 → delta≈-0.04≈0, gradient 살아있음)
            nn.init.constant_(self.pedal_head.bias, -4.0 if delta_mode == "brake_only" else 0.0)
        # lead-presence + approach gate = learned NN MLP (rule threshold/IIR 제거, 규정-clean).
        #   입력 = a_2 in-lane(16-25) obj logit (10) + in-lane closing(=-vL_rel) (10) → gate logit → sigmoid.
        #   obj(lead 유무) AND closing(접근 방향) 둘 다 봄 → receding 닫고 approaching 만 연다.
        #   f_2 brake delta loss + gate BCE 의 gradient 가 이 MLP 까지 흐름 (joint train).
        self.gate_head = GateHeadMLP(n_inlane=10, hidden=24,
                                     warm_thresh=gate_warm_thresh, warm_gate_scale=gate_warm_scale,
                                     warm_vrel_scale=gate_warm_vrel_scale)
        # a_2 logit ch: [obj, vL_abs, vY_abs, vL_rel, vY_rel, prox, dist]. closing = -vL_rel.
        #   vL_rel raw 는 SCALE_V(20)로 normalize 돼 있음(prof635_dataset A2_SCALE_V). gate 입력 closing 은
        #   denorm 후(m/s) 0.5 로 재정규화(±10m/s→±5) — warm-init GELU saturate 회피, dataset _a2logits_to_closing 와 단위 일관.
        self.GATE_VLREL_CH = 3
        self.GATE_VREL_DENORM = 20.0      # a_2 SCALE_V (vL_rel raw → m/s)
        self.GATE_CLOSING_NORM = 0.5      # gate 입력 closing(m/s) → ±5 정규화
        self.last_gate = None; self.last_base_pedal = None; self.last_f2_pedal = None
        self.last_delta = None; self.last_gate_logit = None

    def load_main_frozen(self, ckpt_path: str):
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        sd = {k: v for k, v in sd.items() if not k.startswith("pos_encoder.pe")}
        miss, unexp = self.main.load_state_dict(sd, strict=False)
        for p in self.main.parameters():
            p.requires_grad_(False)
        self.main.eval()
        return miss, unexp

    def forward(self, b):
        traj, ego = b["traj6"], b["ego6"]
        with torch.no_grad():
            out_main = self.main(traj, ego)[:, 0, :]          # (B,2) [steer/40, base_pedal]
        steer = out_main[:, 0]                                # f_1 steer (frozen)
        base_pedal = out_main[:, 1]                           # f_1 base accel_brake (frozen, raceline)
        # 학습 시 base 를 U[-1,1] 로 랜덤 치환 → f_2 가 "어떤 base 든 label 까지 가는 delta"를 학습.
        # 학습엔 lead+base=+1(가속) 케이스가 없어(녹화 traj 가 감속) f_2 가 약하게 배움 → 추론 base=+1.0 에서
        # 제동 못함(final≈0). random base 로 train/infer base 불일치 해소. (ped_emb·final 모두 같은 base 사용)
        if self.training and self.randomize_base:
            base_used = torch.empty_like(base_pedal).uniform_(-1.0, 1.0)
        else:
            base_used = base_pedal
        # f_2 query: (ego + base_pedal) [+ traj if use_traj_in_side]
        ped_emb = self.side_pedal_embed(base_used.unsqueeze(-1)).unsqueeze(1)  # (B,1,d)
        if self.use_traj_in_side:
            q = self.side_pos(self.side_traj_embed(traj)
                              + self.side_state_embed(ego).unsqueeze(1) + ped_emb)
        else:                                                 # 의도: image+ego+base_pedal 만, traj 제외
            q = self.side_state_embed(ego).unsqueeze(1) + ped_emb  # (B,1,d) 단일 query
        if "gt_vrel" in b and b["gt_vrel"] is not None:       # [진단] GT 상대속도 주입 (0.1 스케일 정규화: ±30→±3)
            q = q + self.side_vrel_embed((b["gt_vrel"] * 0.1).unsqueeze(-1)).unsqueeze(1)
        kv = self.img_proj(b["i2_mem"])                       # (B,40,d)
        side = self.side_dec(q, kv)                           # (B,6,d)
        head = self.pedal_head(side[:, 0, :]).squeeze(-1)     # (B,)
        # lead-presence + approach gate = learned NN MLP (입력: a_2 in-lane obj logit 16-25 + closing).
        #   obj = lead 유무, closing(=-vL_rel) = 접근 방향(양수=다가옴). 둘 다 보고 게이팅.
        #   gradient: f_2 brake delta loss + gate BCE → final_pedal/g → gate_head MLP 까지 흐름.
        #   a_2 logits 없으면(=no perception) obj=-10 floor, closing=0 → g≈sigmoid(MLP) (학습된 no-lead 응답).
        if "a2_logits" in b and b["a2_logits"] is not None:
            a2 = b["a2_logits"]
            obj_inlane = a2[:, 16:26, 0]                                      # (B,10) in-lane obj logit
            # closing(m/s, 양수=접근) = -vL_rel*denorm. gate 입력은 0.5 재정규화(±5).
            #   sign 정합: 접근(closing>0) → 입력>0 → GELU 통과 → gate logit↑ → g↑ (제동). receding 닫힘.
            # ★ vL_rel 은 obj<0(no-object) bin 에서 noise(garbage) — obj prob 로 mask 해 lead bin closing 만 전달.
            #   _a2logits_to_closing(dataset) 의 obj>thr 선택 aggregation 과 정합. no-lead → closing≈0.
            obj_prob = torch.sigmoid(obj_inlane)                             # (B,10)
            closing_inlane = (-a2[:, 16:26, self.GATE_VLREL_CH] * self.GATE_VREL_DENORM
                              * self.GATE_CLOSING_NORM) * obj_prob           # (B,10) obj-masked closing
        else:
            obj_inlane = torch.full((base_pedal.shape[0], 10), -10.0,
                                    dtype=base_pedal.dtype, device=base_pedal.device)
            closing_inlane = torch.zeros_like(obj_inlane)
        gate_logit = self.gate_head(obj_inlane, closing_inlane)   # (B,)
        self.last_gate_logit = gate_logit
        # g ∈ (0,2) = 2·sigmoid. g=1(full delta)이 logit 0 에서 도달 — 기존 sigmoid 는 g=1 이 asymptote 라
        #   open 에서도 g≈0.6 에 머물러 full delta(=full 제동)를 못 냄. 2배로 풀어 open(lead&접근) → g→2,
        #   no-lead → g→0. BCE 는 sigmoid(logit) vs g*∈{0,1} 그대로(logit 기반) — open logit↑ → g→2.
        g = 2.0 * torch.sigmoid(gate_logit)
        if self.residual_gate:
            if self.delta_mode == "brake_only":
                delta = -2.0 * torch.sigmoid(head)            # ∈(-2,0], 가속 불가 (brake-only override)
            elif self.delta_mode == "raw":                    # gate 없는 버전: clamp 없는 raw delta (unbounded)
                delta = head                                  # f_2 가 no-lead→0 / lead→음수 brake 를 직접 (gate 흡수)
            else:
                delta = 2.0 * torch.tanh(head)                # signed, bound ±2
            f2_pedal = base_used + delta                      # skip connection
            if self.no_gate:                                  # gate(g) 제거 → final=base+delta (sim 위해 final 만 clamp)
                final_pedal = (base_used + self.delta_scale * delta).clamp(-1.0, 1.0)
            else:
                # delta_scale: brake delta 증폭 노브(기본 1.0). final=clamp(base + g·scale·delta).
                final_pedal = (base_used + g * self.delta_scale * delta).clamp(-1.0, 1.0)
            self.last_delta = delta
        else:
            f2_pedal = torch.tanh(head)                       # absolute pedal -1~1
            final_pedal = (1.0 - g) * base_used + g * self.delta_scale * f2_pedal
        self.last_gate = g; self.last_base_pedal = base_used; self.last_f2_pedal = f2_pedal
        return steer, final_pedal
