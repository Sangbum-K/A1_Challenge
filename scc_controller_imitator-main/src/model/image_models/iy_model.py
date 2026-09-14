"""i_y refactor model — i_1 token cache 입력 + i_y token shift + i_2 fusion + Professor transformer.

Input dict:
  tok_curr       (B, 40, C_b)   i_1 token cache (frozen i_1)
  tok_prev_raw   (B, 40, C_b)   prev image 의 i_1 token (yaw warp 안됨)
  d_yaw          (B,)            curr_yaw - prev_yaw (rad)
  ego_5d         (B, 5)
  traj_40        (B, 40, 3)

Forward:
  1) proj(C_b → D=128) on tok_curr / tok_prev_raw
  2) i_y: shift prev tokens by dx = round(fx · d_yaw / bin_w)
  3) i_2 (ImageFusionBlock — random init or trained): (curr, prev_shifted, ego_motion) → fi_mem
  4) Professor transformer + fc_out
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---- Professor pos encoder (image_scc_models_v2 동일 형식) ----
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                              (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ---- i_y: token-level integer bin shift ----
def yaw_shift_tokens_batch(prev_tok: torch.Tensor, d_yaw: torch.Tensor,
                             fx, bin_w_px: float = 32.0) -> torch.Tensor:
    """prev_tok (B, N, D), d_yaw (B,) rad → shifted prev_tok (B, N, D).
    각 batch element 마다 dx_b = round(fx_b · d_yaw_b / bin_w) bin 만큼 (circular) roll.

    fx: scalar 또는 (B,) tensor.

    **벡터화**: torch.gather 로 한 번에 (Python loop / .item() sync 제거).
    이전 구현은 매 batch element 마다 GPU→CPU sync 로 batch 256 시 138ms 차지했음.

    수식: out[b, n, d] = prev_tok[b, (n - dx[b]) mod N, d]
    """
    B, N, D = prev_tok.shape
    if torch.is_tensor(fx):
        fx_t = fx.to(d_yaw.device).type_as(d_yaw)
    else:
        fx_t = torch.full_like(d_yaw, float(fx))
    dx = torch.round(fx_t * d_yaw / bin_w_px).long()       # (B,)
    arange = torch.arange(N, device=prev_tok.device).unsqueeze(0).expand(B, N)  # (B, N)
    idx = (arange - dx.unsqueeze(1)) % N                    # (B, N)
    idx_e = idx.unsqueeze(-1).expand(B, N, D)               # (B, N, D)
    return torch.gather(prev_tok, dim=1, index=idx_e)


# ---- i_2: ImageFusionBlock (image_scc_models_v2 동일 형식, 가벼움) ----
class AttnBlock(nn.Module):
    def __init__(self, d, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d*2), nn.GELU(), nn.Linear(d*2, d))
        self.ln2 = nn.LayerNorm(d)

    def forward(self, x, kv=None):
        kv = x if kv is None else kv
        h, _ = self.attn(x, kv, kv, need_weights=False)
        x = self.ln1(x + h)
        x = self.ln2(x + self.ffn(x))
        return x


class ImageFusionBlock(nn.Module):
    """Step 0 pre-SA shared + Step 1 cross-attn (Q=curr', KV=[ego_mot, prev']) + Step 2 post-SA."""
    def __init__(self, d=128, n_heads=4):
        super().__init__()
        self.lateral_pos = nn.Parameter(torch.zeros(40, d))
        nn.init.trunc_normal_(self.lateral_pos, std=0.02)
        self.pre_sa = AttnBlock(d, n_heads)
        self.cross  = AttnBlock(d, n_heads)
        self.post_sa = AttnBlock(d, n_heads)
        self.ego_proj = nn.Linear(2, d)

    def forward(self, tok_curr, tok_prev, ego_motion):
        # tok_curr/prev (B, 40, D), ego_motion (B, 2) [vx, wz]
        pos = self.lateral_pos
        c = self.pre_sa(tok_curr + pos)
        p = self.pre_sa(tok_prev + pos)
        fe_mot = self.ego_proj(ego_motion).unsqueeze(1)  # (B, 1, D)
        kv = torch.cat([fe_mot, p], dim=1)
        m = self.cross(c, kv=kv)
        m = self.post_sa(m)
        return m


# ---- 전체 모델 ----
class IyModel(nn.Module):
    """i_y refactored model. Professor #654 ckpt 와 호환 (traj/state/transformer/fc_out).

    v3 변경: i_g (image gate) 추가 — fi_mem 에 a_2 mini-head 예측 obj_prob per-bin gate.
    forward 가 (pred, obj_logit) 반환. obj_logit (B, 40) 은 train-time BCE supervision.
    """

    def __init__(self,
                 C_b: int = 320,        # backbone out channel (tf_efficientnetv2_s = 320)
                 d_model: int = 128, nhead: int = 8, dim_feedforward: int = 512,
                 dropout: float = 0.1, num_layers: int = 6,
                 traj_seq_len: int = 40, traj_dim: int = 3, state_dim: int = 6,
                 num_model_output: int = 2, fx: float = 320.0,
                 use_obj_gate: bool = True,
                 # ControlNet-style parallel side branch (zero-init gated)
                 use_side_branch: bool = False,
                 side_num_layers: int = 3,
                 side_dim_feedforward: int = 512,
                 side_ax_only: bool = False,  # True 면 side branch 가 ax 채널만 보완
                 # a_2 v4 (per-bin obj/dist/vL_rel 등 7-ch) features 주입
                 use_a2_feat: bool = False,
                 a2_n_ch: int = 7,
                 use_a2_late: bool = False,  # True: transformer 출력 후에도 inject (ax decoder 직통)
                 # v23 redesign: main = controller_imitator 스타일 (이미지 없음) + side = cross-attn (Q=traj, KV=img)
                 main_no_image: bool = False,
                 side_cross_attn: bool = False,
                 # min-arbitration: side 가 절대 brake ceiling 출력 → ax = min(main_ax, b_side)
                 side_min_arb: bool = False):
        super().__init__()
        self.fx = float(fx)
        self.side_min_arb = side_min_arb
        self.last_b_side = None   # min-arb 시 직접 supervision 용 stash (B,1)
        self.last_gate = None     # lead-presence gate g (B,1)
        self.d_model = d_model
        self.traj_seq_len = traj_seq_len
        self.traj_dim = traj_dim; self.state_dim = state_dim
        self.use_obj_gate = use_obj_gate
        self.use_side_branch = use_side_branch
        self.side_ax_only = side_ax_only
        self.use_a2_feat = use_a2_feat
        self.use_a2_late = use_a2_late
        self.main_no_image = main_no_image
        self.side_cross_attn_mode = side_cross_attn

        # i_1 proj: C_b → d_model (i_1 backbone 은 이미 cache, 여기는 Conv1d k=1 만)
        self.img_proj = nn.Conv1d(C_b, d_model, kernel_size=1)

        # i_2
        self.img_fusion = ImageFusionBlock(d_model, n_heads=nhead)

        # i_g: a_2 mini-head (D → 1) — fi_mem 마다 obj_logit
        if use_obj_gate:
            self.obj_head = nn.Linear(d_model, 1)
            # v6: a2_logits (i_2 cache) 사용 시 learnable bias 추가 (calibration shift 흡수)
            self.obj_gate_bias = nn.Parameter(torch.zeros(1))

        # Professor core
        self.traj_embed  = nn.Linear(traj_dim, d_model)
        self.state_embed = nn.Linear(state_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout, max_len=traj_seq_len + 1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, num_model_output)

        self.image_pos_emb = nn.Parameter(torch.zeros(40, d_model))
        nn.init.trunc_normal_(self.image_pos_emb, std=0.02)

        # a_2 v4 features (per-bin 7-ch: obj, vL_abs, vY_abs, vL_rel, vY_rel, prox, dist)
        # → fi_mem 에 residual add. zero-init weight 로 identity start 보장.
        if use_a2_feat:
            self.a2_proj = nn.Linear(a2_n_ch, d_model)
            nn.init.zeros_(self.a2_proj.weight)
            nn.init.zeros_(self.a2_proj.bias)
        # a_2 LATE inject — transformer 출력 후 (fc_out 직전) traj-encoded 토큰에 residual add
        # ax decoder 직통 path. zero-init 으로 identity start.
        if use_a2_late:
            self.a2_late_proj = nn.Linear(a2_n_ch, d_model)
            nn.init.zeros_(self.a2_late_proj.weight)
            nn.init.zeros_(self.a2_late_proj.bias)

        # v23 side cross-attn: Q=traj_text (B, traj_seq_len, D), KV=image_mem (B, 40, D)
        # → side_out (B, traj_seq_len, D) → first token pool → side_fc(D,1) → ax 보완
        if side_cross_attn:
            # side_num_layers ≥ 2 면 TransformerDecoder stack (self-attn + cross-attn + FFN × N)
            # side_num_layers == 1 (default-ish) 면 단일 MultiheadAttention (legacy)
            self.side_cross_n_layers = side_num_layers
            if side_num_layers >= 2:
                decoder_layer = nn.TransformerDecoderLayer(
                    d_model, nhead, dim_feedforward=side_dim_feedforward,
                    dropout=dropout, batch_first=True)
                self.side_cross = nn.TransformerDecoder(decoder_layer, num_layers=side_num_layers)
                self.side_cross_ln = nn.Identity()   # TransformerDecoder 가 자체 LN
            else:
                self.side_cross = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
                self.side_cross_ln = nn.LayerNorm(d_model)
            self.side_fc_cross = nn.Linear(d_model, 1)  # default kaiming init (NOT zero — v18 degenerate 회피)
            self.side_alpha_cross = nn.Parameter(torch.zeros(1))  # α=0 만으로 identity start 충분 (additive 모드)
            if side_min_arb:
                # b_side = ACC brake ceiling (절대 ax). bias 높게 시작 → 초기엔 brake_corr=0 (cruise 보존).
                # dead-start 는 train 의 직접 b_side supervision 으로 회피.
                self.side_brake_bias = nn.Parameter(torch.tensor([20.0]))
                # lead-presence gate bias (g = sigmoid(a_2 in-lane obj max + bias)). 0 init = a_2 신뢰 그대로.
                self.side_gate_bias = nn.Parameter(torch.zeros(1))

        # ControlNet-style parallel side branch.
        # 디자인: side_fc 는 NORMAL init (nn.Linear default), α 만 0 init.
        # 이유: α=0 만으로 identity start 충족 (α·side_pred = 0·anything = 0).
        # 만약 side_fc 도 0 으로 두면 두 grad 가 서로 gate → 영원히 0 (degenerate).
        # α grad = ∂L/∂pred · side_pred 는 side_pred(random init) 가 비어있지 않아 흐름.
        # α 가 움직이기 시작하면 side_fc grad 도 자동 열림.
        if use_side_branch:
            side_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=side_dim_feedforward,
                dropout=dropout, batch_first=True)
            self.side_transformer = nn.TransformerEncoder(side_layer, num_layers=side_num_layers)
            side_out_dim = 1 if side_ax_only else num_model_output  # ax-only mode: 1 channel
            self.side_fc = nn.Linear(d_model, side_out_dim)  # default init (kaiming-uniform)
            self.side_alpha = nn.Parameter(torch.zeros(1))

    def _proj_tok(self, tok_raw: torch.Tensor) -> torch.Tensor:
        # tok_raw (B, 40, C_b) → (B, 40, D)
        x = tok_raw.transpose(1, 2)    # (B, C_b, 40)
        x = self.img_proj(x)
        return x.transpose(1, 2)

    def forward(self, b: dict):
        """returns (pred (B, 40, 2), obj_logit (B, 40) or None)."""
        # v6: i_2 cache 가 batch 에 있으면 image_mem 직접 사용 (img_proj+img_fusion 우회)
        # — a_2 v5/v6 의 fc_proj 후 학습된 d=128 representation
        if "i2_mem" in b and b["i2_mem"] is not None:
            fi_mem = b["i2_mem"]   # (B, 40, D=128) — 이미 fusion 통과한 cache
        else:
            # i_y: token level Δyaw shift on prev (legacy path)
            fx_b = b.get("fx", self.fx)
            prev_shifted = yaw_shift_tokens_batch(b["tok_prev_raw"], b["d_yaw"], fx=fx_b)
            # proj (C_b → D)
            tc = self._proj_tok(b["tok_curr"])
            tp = self._proj_tok(prev_shifted)
            ego_mot = b["ego_5d"][:, :2]   # [vx, wz=yaw_vel]
            fi_mem  = self.img_fusion(tc, tp, ego_mot)
        # a_2 v4 풍부 features (per-bin 7-ch) residual add — zero-init 으로 시작
        if self.use_a2_feat and ("a2_feat" in b):
            fi_mem = fi_mem + self.a2_proj(b["a2_feat"])
        # i_g: per-bin obj gate
        obj_logit = None
        if self.use_obj_gate:
            # v6: a2_logits 직접 사용 (i_2 cache 의 a_2 v6 head output) + learnable bias
            if "a2_logits" in b and b["a2_logits"] is not None:
                obj_logit = b["a2_logits"][..., 0]                    # (B, 40) — a_2 v6 의 obj logit
                gate = torch.sigmoid(obj_logit + self.obj_gate_bias).unsqueeze(-1)
            else:
                # legacy: predict from fi_mem
                obj_logit = self.obj_head(fi_mem).squeeze(-1)
                gate = torch.sigmoid(obj_logit).unsqueeze(-1)
            fi_mem = fi_mem * gate
        # traj/state tokens
        traj  = b["traj_40"][:, :, :self.traj_dim]
        state = b["ego_5d"][:, :self.state_dim]
        if state.shape[-1] < self.state_dim:
            pad = torch.zeros(state.shape[0], self.state_dim - state.shape[-1],
                              dtype=state.dtype, device=state.device)
            state = torch.cat([state, pad], dim=-1)
        traj_tok = self.traj_embed(traj)
        state_tok = self.state_embed(state).unsqueeze(1)
        tokens_text = self.pos_encoder(traj_tok + state_tok)
        # main transformer 입력 — main_no_image 면 text 만, 아니면 text + image_mem concat
        image_mem = fi_mem + self.image_pos_emb
        if self.main_no_image:
            tokens = tokens_text                                       # (B, traj_seq_len, D)
        else:
            tokens = torch.cat([tokens_text, image_mem], dim=1)         # (B, 80, D)
        encoded = self.transformer_encoder(tokens)
        encoded_traj = encoded[:, :self.traj_seq_len]
        # a_2 LATE inject: traj-encoded 토큰에 a_2 features residual add (fc_out 직전)
        if self.use_a2_late and ("a2_feat" in b):
            encoded_traj = encoded_traj + self.a2_late_proj(b["a2_feat"])
        pred = self.fc_out(encoded_traj)                                # (B, traj_seq_len, 2)

        # v23 side cross-attn: Q=traj_text, KV=image_mem (a_2 augmented fi_mem 사용)
        # ax 채널만 보완 (1 step). 첫 traj token 출력 → scalar
        if self.side_cross_attn_mode:
            # min-arb: traj query 제거 (in-lane 은 a_2 obj gate 가 담당) → ego state token 만 query.
            #   b_side = ego속도 + image_mem(closing) 기반 절대 brake demand.
            # additive: 기존대로 traj_text query.
            side_q = state_tok if self.side_min_arb else tokens_text     # (B, 1 or traj_seq_len, D)
            if isinstance(self.side_cross, nn.TransformerDecoder):
                # multi-layer decoder stack: self-attn + cross-attn + FFN per layer
                side_out = self.side_cross(side_q, image_mem)            # (B, traj_seq_len, D) — 내부 residual/LN
            else:
                # single MultiheadAttention layer (legacy)
                side_out, _ = self.side_cross(side_q, image_mem, image_mem, need_weights=False)
                side_out = self.side_cross_ln(side_out + side_q)         # residual + norm
            if self.side_min_arb:
                # lead-gated ACC override: side 가 절대 brake ceiling b_side 출력.
                # ax_step0 = main_ax + g·clamp(b_side - main_ax, max=0)  (downward only, lead 인지 시에만)
                b_side = self.side_fc_cross(side_out[:, 0, :]) + self.side_brake_bias  # (B, 1) 절대 ax
                self.last_b_side = b_side                                    # train 직접 supervision 용
                main_ax0 = pred[:, 0, 1:2]                                   # (B, 1)
                if "a2_logits" in b and b["a2_logits"] is not None:
                    obj_inlane = b["a2_logits"][:, 16:26, 0]                 # (B, 10) in-lane obj logit
                    g = torch.sigmoid(obj_inlane.amax(dim=1, keepdim=True) + self.side_gate_bias)  # (B,1)
                else:
                    g = torch.ones_like(main_ax0)
                self.last_gate = g
                brake_corr = torch.clamp(b_side - main_ax0, max=0.0)         # ≤0 (downward only)
                ax_col = pred[..., 1:2].clone()
                ax_col[:, 0, :] = main_ax0 + g * brake_corr
                pred = torch.cat([pred[..., :1], ax_col], dim=-1)
            else:
                side_ax_1 = self.side_fc_cross(side_out[:, 0:1])             # (B, 1, 1) — first traj token
                # broadcast 0-step ax 에 add (target_first_only 시 첫 step 만 loss 사용)
                ax_add = torch.zeros_like(pred[..., 1:2])
                ax_add[:, 0:1, :] = self.side_alpha_cross * side_ax_1
                pred = torch.cat([pred[..., :1], pred[..., 1:2] + ax_add], dim=-1)
        # ControlNet side branch: same tokens 입력, 별도 transformer, α gate
        if self.use_side_branch:
            side_encoded = self.side_transformer(tokens)
            side_pred = self.side_fc(side_encoded[:, :self.traj_seq_len])
            if self.side_ax_only:
                # ax 채널 (index 1) 만 side 추가, steer 는 main 그대로
                ax_with_side = pred[:, :, 1:2] + self.side_alpha * side_pred  # (B, 40, 1)
                pred = torch.cat([pred[:, :, :1], ax_with_side], dim=-1)
            else:
                pred = pred + self.side_alpha * side_pred
        return pred, obj_logit

    def load_professor(self, ckpt_path: str):
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
        my_sd = self.state_dict()
        loaded = {}; skipped = []
        for k, v in sd.items():
            if k in my_sd and my_sd[k].shape == v.shape:
                loaded[k] = v
            elif k == "pos_encoder.pe":
                if k in my_sd and my_sd[k].shape[1] >= v.shape[1]:
                    new = my_sd[k].clone()
                    new[:, :v.shape[1]] = v
                    loaded[k] = new
            else:
                skipped.append((k, tuple(v.shape), tuple(my_sd[k].shape) if k in my_sd else None))
        self.load_state_dict(loaded, strict=False)
        return loaded, skipped
