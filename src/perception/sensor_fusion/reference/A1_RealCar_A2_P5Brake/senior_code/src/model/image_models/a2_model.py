"""i_1 (LateralStripEncoder) + i_2 (ImageFusionBlock) + a_2 (AuxHead) 단일 모델.

목적: a_2 GT (40, 7) [obj, vL_abs, vY_abs, vL_rel, vY_rel, prox, dist] 회귀 학습.
입력:
  image_curr     (B, 3, 250, 1280)
  image_prev_rot (B, 3, 250, 1280)
  ego_motion     (B, 2)            [vx, wz]
출력:
  logits         (B, 40, 7)        ch 0 = obj logit, ch 1..6 = regression
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# i_1 ===============================================================
class LateralStripEncoder(nn.Module):
    """siamese — out_indices=[-1] only (a_2 는 spatial_feat 불필요)."""
    def __init__(self, backbone: str = "tf_efficientnetv2_s", d_token: int = 256):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=True, features_only=True, out_indices=[-1])
        C_b = self.backbone.feature_info.channels()[-1]
        self.proj = nn.Conv1d(C_b, d_token, kernel_size=1)
        self.d_token = d_token

    def load_pretrained_backbone(self, state_dict: dict):
        """seg_pretrain SegPretrainModel.backbone_state_dict 호환 load."""
        own = self.backbone.state_dict()
        loaded = {k.replace("backbone.", ""): v
                  for k, v in state_dict.items() if k.startswith("backbone.")}
        # 누락/추가 키 출력
        missing = [k for k in own if k not in loaded]
        extra   = [k for k in loaded if k not in own]
        own.update({k: v for k, v in loaded.items() if k in own})
        self.backbone.load_state_dict(own)
        return {"missing": len(missing), "extra": len(extra), "loaded": len(loaded)}

    def forward(self, x):
        feat = self.backbone(x)[-1]                              # (B, C_b, ~8, ~40)
        feat = F.adaptive_avg_pool2d(feat, (1, 40)).squeeze(2)   # (B, C_b, 40)
        feat = self.proj(feat)                                   # (B, D, 40)
        return feat.transpose(1, 2)                              # (B, 40, D)


# i_2 ===============================================================
class _AttnBlock(nn.Module):
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
    """i_2a Step 0 (shared SA) + i_2b Step 1 (cross-attn w/ ego prepend) + i_2c Step 2 (post SA)."""
    def __init__(self, d=256, n_heads=4):
        super().__init__()
        self.lateral_pos = nn.Parameter(torch.zeros(40, d))
        nn.init.trunc_normal_(self.lateral_pos, std=0.02)
        self.pre_sa  = _AttnBlock(d, n_heads)
        self.ego_proj = nn.Linear(2, d)
        self.frame_type_emb = nn.Embedding(3, d)   # 0=Q curr, 1=KV prev, 2=KV ego
        self.cross   = _AttnBlock(d, n_heads)
        self.post_sa = _AttnBlock(d, n_heads)
        self.d = d

    def forward(self, tok_curr, tok_prev, ego_motion):
        """
        tok_curr/prev (B, 40, D), ego_motion (B, 2) → fi_mem (B, 40, D).
        """
        pos = self.lateral_pos
        c = self.pre_sa(tok_curr + pos)
        p = self.pre_sa(tok_prev + pos)

        # ego_motion → fe_mot (B, 1, D)
        fe_mot = self.ego_proj(ego_motion).unsqueeze(1)
        # KV = [ego, prev] (41, D), with frame_type embedding
        kv = torch.cat([fe_mot, p], dim=1)
        type_ids = torch.cat([
            torch.full((1,), 2, dtype=torch.long, device=kv.device),  # ego
            torch.full((40,), 1, dtype=torch.long, device=kv.device),  # prev
        ])
        kv = kv + self.frame_type_emb(type_ids)[None]
        # Q = curr + type 0
        q = c + self.frame_type_emb(torch.tensor(0, device=c.device))

        m = self.cross(q, kv=kv)
        m = self.post_sa(m)
        return m


# a_2 ===============================================================
# a_2 출력 채널 의미 ----------------------------------------------------
# legacy (predict_var=False), 7 ch: [obj, vL_abs, vY_abs, vL_rel, vY_rel, prox, dist]
# v8 interleaved (predict_var=True), 13 ch:
#   [obj, vL_abs, vL_abs_σ, vY_abs, vY_abs_σ, vL_rel, vL_rel_σ, vY_rel, vY_rel_σ, prox, prox_σ, dist, dist_σ]
#   → 각 회귀값 바로 뒤에 그 log-variance. obj 는 분류라 σ 없음.
REG_NAMES = ["vL_abs", "vY_abs", "vL_rel", "vY_rel", "prox", "dist"]
def mean_idx(i: int, predict_var: bool) -> int:
    """reg 채널 i (0..5) 의 mean 채널 인덱스."""
    return (1 + 2 * i) if predict_var else (1 + i)
def sigma_idx(i: int) -> int:
    """reg 채널 i (0..5) 의 log_var 채널 인덱스 (interleaved)."""
    return 2 + 2 * i
# 편의: 전체 mean 채널 인덱스 (obj 포함 안 함)
def mean_indices(predict_var: bool):
    return [mean_idx(i, predict_var) for i in range(6)]
def sigma_indices():
    return [sigma_idx(i) for i in range(6)]


class AuxHead(nn.Module):
    """Linear(D → n_out).
    legacy: 7 ch [obj, 6 mean].
    v8 (predict_var): 13 ch interleaved [obj, mean0, σ0, mean1, σ1, ...].
    """
    def __init__(self, d=256, n_ch=7, predict_var: bool = False):
        super().__init__()
        self.predict_var = predict_var
        n_reg = n_ch - 1                       # 6
        self.n_out = 1 + (2 * n_reg if predict_var else n_reg)
        self.fc = nn.Linear(d, self.n_out)

    def forward(self, x):   # x (B, 40, D)
        return self.fc(x)   # (B, 40, n_out)  7 또는 13


# 합본 ==============================================================
class A2Model(nn.Module):
    """a_2 model.

    d_out 가 d_token 보다 작으면 fusion 출력 (B, 40, d_token) → fc_proj Linear → (B, 40, d_out)
    적용 후 AuxHead 입력으로 사용. 이 형태가 iy_model 의 d_model (예: 128) 과 호환되는
    image_mem 생성에 사용 (v5 — 2026-05-25 plan).

    d_out 가 None (또는 d_token 과 같음) 이면 기존 동작 (proj 없음).
    """
    def __init__(self,
                 img_backbone: str = "tf_efficientnetv2_s",
                 d_token: int = 256, n_heads: int = 4, n_ch: int = 7,
                 d_out: int = None, predict_var: bool = False):
        super().__init__()
        self.encoder = LateralStripEncoder(img_backbone, d_token)
        self.fusion  = ImageFusionBlock(d_token, n_heads)
        self.d_token = d_token
        self.d_out = int(d_out) if d_out is not None else int(d_token)
        if self.d_out != self.d_token:
            self.fc_proj = nn.Linear(self.d_token, self.d_out)
        else:
            self.fc_proj = nn.Identity()
        self.predict_var = predict_var
        # v8: interleaved [obj, mean, σ, ...] 13ch 출력 (predict_var). mean 은 head 가 자체 포함.
        self.head    = AuxHead(self.d_out, n_ch, predict_var=predict_var)

    def forward(self, image_curr, image_prev_rot, ego_motion, return_mem: bool = False):
        tok_c = self.encoder(image_curr)         # (B, 40, d_token)
        tok_p = self.encoder(image_prev_rot)
        fi_mem = self.fusion(tok_c, tok_p, ego_motion)   # (B, 40, d_token)
        fi_mem_out = self.fc_proj(fi_mem)                 # (B, 40, d_out)
        logits = self.head(fi_mem_out)                    # (B, 40, 7 또는 13 interleaved)
        if return_mem:
            return logits, fi_mem_out
        return logits

    def freeze_encoder_fusion(self):
        """fc_proj + head 만 학습 가능하도록. v4 ckpt 재사용 + 새 layer 만 학습."""
        for p in self.encoder.parameters(): p.requires_grad = False
        for p in self.fusion.parameters():  p.requires_grad = False
        return {"trainable_modules": ["fc_proj", "head"]}

    def load_seg_pretrain(self, ckpt_path: str):
        sd = torch.load(ckpt_path, map_location="cpu")
        return self.encoder.load_pretrained_backbone(sd)


# single-frame obj·dist =================================================
# 2026-05-26 plan: prev cross-attn(fusion)·yaw_shift·ego_mot 제거.
#   curr i_1 token (40, d_token) → lateral self-attn (1~2층) → fc_proj(→d_out=128)
#   = i2_mem (f_2 호환) → head [obj, dist] (속도 4채널 제거).
#   다운스트림 제어(distbrake)가 dist+obj 만 쓰므로 capacity 를 obj/dist 에 집중.
class _SelfAttnBlock(nn.Module):
    """lateral self-attention block (col 간 문맥, prev cross-attn 없음)."""
    def __init__(self, d, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))
        self.ln2 = nn.LayerNorm(d)

    def forward(self, x):
        h, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + h)
        x = self.ln2(x + self.ffn(x))
        return x


# multi-frame (velocity) ================================================
# 2026-05-26 plan P2: N 프레임 i_1 stack 으로 vL_rel(상대속도) 추정 개선 + flicker 완화.
#   N=3: curr, t-20, t-40 / N=4: + t-60 (20tick=200ms).
#   학습된 temporal attention: Q=curr (40 tok), KV=[ego, 전 프레임 (N*40 tok)]
#     + lateral pos (col) + temporal pos (frame age) embedding.
#   yaw_shift 는 P1 판정대로 (각 prev frame 에 누적 d_yaw 로 적용 / OFF).
#   head = 7ch (v7b 동일) — obj, vL_abs, vY_abs, vL_rel, vY_rel, prox, dist.
class _MultiFrameFusion(nn.Module):
    """N-frame temporal fusion.

    입력: tok_proj_list — N 개의 (B, 40, d) (frame 0 = curr, 1 = oldest 까지).
          ego_motion (B, 2).
    구조: pre_sa (shared, lateral_pos) → Q=curr, KV=[ego, prev frames (각 temporal_emb)]
          → cross → post_sa.
    N=2 일 때 ImageFusionBlock 과 동등 (단, KV 에 temporal_emb 추가).
    """
    def __init__(self, d=256, n_heads=4, n_frames=3):
        super().__init__()
        self.n_frames = int(n_frames)
        self.lateral_pos = nn.Parameter(torch.zeros(40, d))
        nn.init.trunc_normal_(self.lateral_pos, std=0.02)
        self.pre_sa  = _AttnBlock(d, n_heads)
        self.ego_proj = nn.Linear(2, d)
        # temporal embedding: frame 0 = curr (Q), 1..N-1 = prev (age 순). +1 = ego slot.
        self.temporal_emb = nn.Embedding(self.n_frames + 1, d)  # idx 0..N-1 frames, idx N = ego
        self.cross   = _AttnBlock(d, n_heads)
        self.post_sa = _AttnBlock(d, n_heads)
        self.d = d

    def forward(self, tok_list, ego_motion):
        """tok_list: list of N (B,40,d). tok_list[0]=curr. → (B,40,d)."""
        assert len(tok_list) == self.n_frames, f"expect {self.n_frames} frames, got {len(tok_list)}"
        pos = self.lateral_pos
        B = tok_list[0].size(0)
        ego_idx = self.n_frames
        # Q = curr + lateral + temporal(0)
        c = self.pre_sa(tok_list[0] + pos)
        q = c + self.temporal_emb(torch.zeros(1, dtype=torch.long, device=c.device))
        # KV = [ego, prev_1 ... prev_{N-1}]  (각 프레임 temporal_emb age)
        fe_mot = self.ego_proj(ego_motion).unsqueeze(1)                 # (B,1,d)
        fe_mot = fe_mot + self.temporal_emb(
            torch.full((1,), ego_idx, dtype=torch.long, device=c.device))
        kv_parts = [fe_mot]
        for fi in range(1, self.n_frames):
            p = self.pre_sa(tok_list[fi] + pos)
            p = p + self.temporal_emb(
                torch.full((1,), fi, dtype=torch.long, device=c.device))
            kv_parts.append(p)
        kv = torch.cat(kv_parts, dim=1)                                 # (B, 1+(N-1)*40, d)
        m = self.cross(q, kv=kv)
        m = self.post_sa(m)
        return m


class A2ModelMultiFrame(nn.Module):
    """multi-frame a_2 (velocity 포함, 7ch head, fc_proj→d_out).

    입력 (cache path): tok_list — N 개 (B,40,C_b) raw backbone token (frame0=curr).
                      ego_motion (B,2).
    경로: 각 frame → encoder.proj (Conv1d C_b→d_token) → _MultiFrameFusion
          → fc_proj (d_token→d_out=128) → head[7ch].
    v7b 와 동일 head/fc_proj → 직접 비교. n_frames=2 면 v7b 와 (거의) 동등 구조.
    """
    def __init__(self,
                 img_backbone: str = "tf_efficientnetv2_s",
                 d_token: int = 256, n_heads: int = 4, n_ch: int = 7,
                 d_out: int = 128, n_frames: int = 3, predict_var: bool = False):
        super().__init__()
        self.encoder = LateralStripEncoder(img_backbone, d_token)
        self.fusion  = _MultiFrameFusion(d_token, n_heads, n_frames=n_frames)
        self.d_token = d_token
        self.n_frames = int(n_frames)
        self.d_out = int(d_out) if d_out is not None else int(d_token)
        if self.d_out != self.d_token:
            self.fc_proj = nn.Linear(self.d_token, self.d_out)
        else:
            self.fc_proj = nn.Identity()
        self.predict_var = predict_var
        self.head = AuxHead(self.d_out, n_ch, predict_var=predict_var)

    def _proj_tokens(self, tok):
        """tok (B,40,C_b) → (B,40,d_token) via encoder.proj (Conv1d)."""
        return self.encoder.proj(tok.transpose(1, 2)).transpose(1, 2)

    def forward_tokens(self, tok_list, ego_motion, return_mem: bool = False):
        """tok_list: N 개 (B,40,C_b) (encoder.proj 전 raw). frame0=curr."""
        proj_list = [self._proj_tokens(t) for t in tok_list]
        fi_mem = self.fusion(proj_list, ego_motion)
        fi_mem_out = self.fc_proj(fi_mem)
        logits = self.head(fi_mem_out)
        if return_mem:
            return logits, fi_mem_out
        return logits

    def forward(self, image_list, ego_motion, return_mem: bool = False):
        """raw image path: image_list = N 개 (B,3,250,1280)."""
        tok_list = [self.encoder(im) for im in image_list]  # (B,40,d_token) (proj 포함)
        # encoder() 가 이미 proj 적용 → forward_tokens 우회, fusion 직접
        fi_mem = self.fusion(tok_list, ego_motion)
        fi_mem_out = self.fc_proj(fi_mem)
        logits = self.head(fi_mem_out)
        if return_mem:
            return logits, fi_mem_out
        return logits

    def freeze_encoder(self):
        """encoder(backbone+proj) frozen — fusion+fc_proj+head 학습.
        (v7b 의 frozen encoder/fusion 와 달리 multi-frame fusion 은 새 구조라 학습)."""
        for p in self.encoder.parameters(): p.requires_grad = False
        return {"trainable_modules": ["fusion", "fc_proj", "head"]}

    def load_seg_pretrain(self, ckpt_path: str):
        sd = torch.load(ckpt_path, map_location="cpu")
        return self.encoder.load_pretrained_backbone(sd)


# high-res (80-col) stride16 1D-merge ===================================
# 2026-05-26 plan: far lateral 해상도. stride32 native W=40 → lateral 양자화 한계.
#   stride16 native W=80 을 H-pool 해 80-col 실측 + stride32(40)→interp80 semantic 을 add.
#   merge = pool 후 1D (사용자안): p32_80 + alpha(=0 init)·p16_80. α=0 시작 = 40→80 interp baseline
#   (v7b readout 의 upsample) → 학습으로 stride16 기여를 켜감 (warm-start 호환).
#   cache path: tok32 (B,40,C32) + tok16 (B,80,C16) 둘 입력 (proj 전 raw pool 결과).
N_LAT_HR = 80
class LateralStripEncoderHR(nn.Module):
    """stride16(80) 실측 + stride32(40→interp80) semantic 의 1D residual merge → (B,80,d)."""
    def __init__(self, backbone: str = "tf_efficientnetv2_s", d_token: int = 256):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=True, features_only=True, out_indices=[-2, -1])
        chs = self.backbone.feature_info.channels()
        self.C32 = chs[-1]; self.C16 = chs[-2]
        self.proj32 = nn.Conv1d(self.C32, d_token, kernel_size=1)   # v7b encoder.proj 호환(이름만 다름)
        self.proj16 = nn.Conv1d(self.C16, d_token, kernel_size=1)
        self.alpha  = nn.Parameter(torch.zeros(1))                  # stride16 residual gate (0 = pure interp)
        self.d_token = d_token

    def load_pretrained_backbone(self, state_dict: dict):
        own = self.backbone.state_dict()
        loaded = {k.replace("backbone.", ""): v
                  for k, v in state_dict.items() if k.startswith("backbone.")}
        missing = [k for k in own if k not in loaded]
        extra   = [k for k in loaded if k not in own]
        own.update({k: v for k, v in loaded.items() if k in own})
        self.backbone.load_state_dict(own)
        return {"missing": len(missing), "extra": len(extra), "loaded": len(loaded)}

    def _merge(self, tok32, tok16):
        """tok32 (B,40,C32), tok16 (B,80,C16) → (B,80,d). proj 후 1D residual merge."""
        p32 = self.proj32(tok32.transpose(1, 2))                   # (B,d,40)
        p32 = F.interpolate(p32, size=N_LAT_HR, mode="linear", align_corners=False)  # (B,d,80)
        p16 = self.proj16(tok16.transpose(1, 2))                   # (B,d,80)
        out = p32 + self.alpha * p16                               # (B,d,80)
        return out.transpose(1, 2)                                 # (B,80,d)

    def pool_tokens(self, x):
        """raw image (B,3,250,1280) → (tok32 (B,40,C32), tok16 (B,80,C16)) — cache 저장용 pool 결과."""
        f16, f32 = self.backbone(x)                                # [-2]=stride16, [-1]=stride32
        tok32 = F.adaptive_avg_pool2d(f32, (1, 40)).squeeze(2).transpose(1, 2)   # (B,40,C32)
        tok16 = F.adaptive_avg_pool2d(f16, (1, N_LAT_HR)).squeeze(2).transpose(1, 2)  # (B,80,C16)
        return tok32, tok16

    def forward(self, x):
        tok32, tok16 = self.pool_tokens(x)
        return self._merge(tok32, tok16)                           # (B,80,d)


class A2ModelMultiFrameHR(nn.Module):
    """multi-frame + 80-col stride16 merge a_2. A2ModelMultiFrame 의 HR 버전.

    cache path: forward_tokens(tok32_list, tok16_list, ego) — frame 별 (B,40,C32)+(B,80,C16).
    raw  path: forward(image_list, ego).
    fusion lateral = 80. head 는 per-token 이라 무변(출력 (B,80,n_ch)).
    """
    def __init__(self,
                 img_backbone: str = "tf_efficientnetv2_s",
                 d_token: int = 256, n_heads: int = 4, n_ch: int = 7,
                 d_out: int = 128, n_frames: int = 3, predict_var: bool = False):
        super().__init__()
        self.encoder = LateralStripEncoderHR(img_backbone, d_token)
        self.fusion  = _MultiFrameFusion(d_token, n_heads, n_frames=n_frames)
        # fusion.lateral_pos 를 80 으로 교체 (기본 40)
        self.fusion.lateral_pos = nn.Parameter(torch.zeros(N_LAT_HR, d_token))
        nn.init.trunc_normal_(self.fusion.lateral_pos, std=0.02)
        self.d_token = d_token
        self.n_frames = int(n_frames)
        self.d_out = int(d_out) if d_out is not None else int(d_token)
        if self.d_out != self.d_token:
            self.fc_proj = nn.Linear(self.d_token, self.d_out)
        else:
            self.fc_proj = nn.Identity()
        self.predict_var = predict_var
        self.head = AuxHead(self.d_out, n_ch, predict_var=predict_var)

    def forward_tokens(self, tok32_list, tok16_list, ego_motion, return_mem: bool = False):
        """tok32_list/tok16_list: N 개 (B,40,C32)/(B,80,C16) raw pool. frame0=curr."""
        merged = [self.encoder._merge(t32, t16) for t32, t16 in zip(tok32_list, tok16_list)]  # N×(B,80,d)
        fi_mem = self.fusion(merged, ego_motion)
        fi_mem_out = self.fc_proj(fi_mem)
        logits = self.head(fi_mem_out)
        if return_mem:
            return logits, fi_mem_out
        return logits

    def forward(self, image_list, ego_motion, return_mem: bool = False):
        merged = [self.encoder(im) for im in image_list]            # N×(B,80,d)
        fi_mem = self.fusion(merged, ego_motion)
        fi_mem_out = self.fc_proj(fi_mem)
        logits = self.head(fi_mem_out)
        if return_mem:
            return logits, fi_mem_out
        return logits

    def freeze_encoder(self):
        """backbone+proj32 frozen, proj16/alpha/fusion/fc_proj/head 학습."""
        for p in self.encoder.backbone.parameters(): p.requires_grad = False
        for p in self.encoder.proj32.parameters():   p.requires_grad = False
        return {"trainable_modules": ["proj16", "alpha", "fusion", "fc_proj", "head"]}

    def load_seg_pretrain(self, ckpt_path: str):
        sd = torch.load(ckpt_path, map_location="cpu")
        return self.encoder.load_pretrained_backbone(sd)


# single-frame obj·dist =================================================
class A2ModelSingleObjDist(nn.Module):
    """single-frame + obj·dist-only a_2.

    입력:  i_1 token (B, 40, d_token)  — encoder.proj 적용된 curr token (cache path)
           또는 image_curr (B, 3, 250, 1280) — raw path (forward_image).
    경로:  curr token (B,40,d_token) + lateral_pos → self-attn × n_sa층
           → fc_proj (d_token → d_out=128) = i2_mem → head.
    head:  n_ch 채널. ★ch0 = obj logit 고정★ (closed_loop gate a2_logits[...,0]).
           기본 n_ch=2 → [obj, dist]. n_ch=3 → [obj, dist, prox].

    prev 프레임·fusion(cross-attn)·yaw_shift·ego_mot 전부 제거 (single-frame).
    """
    def __init__(self,
                 img_backbone: str = "tf_efficientnetv2_s",
                 d_token: int = 256, n_heads: int = 4, n_ch: int = 2,
                 d_out: int = 128, n_sa: int = 2):
        super().__init__()
        self.encoder = LateralStripEncoder(img_backbone, d_token)
        self.d_token = d_token
        self.d_out = int(d_out) if d_out is not None else int(d_token)
        self.n_ch = int(n_ch)
        # lateral self-attn stack (curr token 간 문맥)
        self.lateral_pos = nn.Parameter(torch.zeros(40, d_token))
        nn.init.trunc_normal_(self.lateral_pos, std=0.02)
        self.sa = nn.ModuleList([_SelfAttnBlock(d_token, n_heads) for _ in range(n_sa)])
        # fc_proj (d_token → d_out, f_2 호환) = i2_mem
        if self.d_out != self.d_token:
            self.fc_proj = nn.Linear(self.d_token, self.d_out)
        else:
            self.fc_proj = nn.Identity()
        # head: linear (d_out → n_ch). ch0 = obj logit, ch1.. = regression.
        self.head = nn.Linear(self.d_out, self.n_ch)

    def _trunk(self, tok):
        """tok (B, 40, d_token) (encoder.proj 적용 후) → i2_mem (B, 40, d_out)."""
        x = tok + self.lateral_pos
        for blk in self.sa:
            x = blk(x)
        return self.fc_proj(x)

    def forward_tokens(self, tok_proj, return_mem: bool = False):
        """tok_proj (B, 40, d_token) — encoder.proj 적용 후 token (cache path)."""
        i2_mem = self._trunk(tok_proj)
        logits = self.head(i2_mem)
        if return_mem:
            return logits, i2_mem
        return logits

    def forward(self, image_curr, return_mem: bool = False):
        """raw image path: image_curr (B, 3, 250, 1280) → logits."""
        tok = self.encoder(image_curr)              # (B, 40, d_token) (proj 포함)
        return self.forward_tokens(tok, return_mem=return_mem)

    def freeze_encoder(self):
        """encoder(backbone+proj) frozen, sa+fc_proj+head 학습."""
        for p in self.encoder.parameters(): p.requires_grad = False
        return {"trainable_modules": ["lateral_pos", "sa", "fc_proj", "head"]}

    def load_seg_pretrain(self, ckpt_path: str):
        sd = torch.load(ckpt_path, map_location="cpu")
        return self.encoder.load_pretrained_backbone(sd)
