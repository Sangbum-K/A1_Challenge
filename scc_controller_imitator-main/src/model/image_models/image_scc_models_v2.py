"""SCC controller imitator v2 — image + controller_imitator pretrained backbone.

Reuses `SelfAttentionControlPredictor_Professor` (controller_imitator #654)
core (traj_embed + state_embed + pos_encoder + 6-layer TransformerEncoder + fc_out)
and concatenates `image_memory` (40 lateral tokens) into the self-attn input.

Layout (forward):
  - traj_tokens (B, 40, D)  ← traj_embed(traj[:, :, :3])
  - state_token (B, 1, D)   ← state_embed(state[:, :6]) → broadcast added to traj_tokens
  - tokens_text (B, 40, D)  = traj_tokens + state_token   (controller_imitator 동작 그대로)
  - image_tok  (B, 40, D)   ← LateralStripEncoder(image_curr, image_prev_rot) → ImageFusionBlock
  - tokens      (B, 80, D)  = [tokens_text, image_tok]   + pos_encoder + image_pos_emb
  - encoded     (B, 80, D)  ← transformer_encoder (pretrained 6 layers, finetune)
  - control_preds (B, 40, D_out) ← fc_out(encoded[:, :40])  (image positions discarded)

Pretrained keys loaded: traj_embed, state_embed, pos_encoder.pe (first 41),
transformer_encoder.*, fc_out.
Random init: image encoder (timm pretrained on ImageNet), fusion block, image_pos_emb.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math


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
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LateralStripEncoder(nn.Module):
    def __init__(self, backbone: str = "efficientnet_b0", d_token: int = 128):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=True, features_only=True, out_indices=[-1])
        C_b = self.backbone.feature_info.channels()[-1]
        self.proj = nn.Conv1d(C_b, d_token, kernel_size=1)
        self.d_token = d_token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)[-1]
        feat = F.adaptive_avg_pool2d(feat, (1, 40)).squeeze(2)
        feat = self.proj(feat)
        return feat.transpose(1, 2)


class AttnBlock(nn.Module):
    def __init__(self, d: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))
        self.ln2 = nn.LayerNorm(d)

    def forward(self, x, kv=None):
        kv = x if kv is None else kv
        h, _ = self.attn(x, kv, kv, need_weights=False)
        x = self.ln1(x + h)
        x = self.ln2(x + self.ffn(x))
        return x


class ImageFusionBlock(nn.Module):
    def __init__(self, d: int = 128, n_heads: int = 4):
        super().__init__()
        self.lateral_pos = nn.Parameter(torch.zeros(40, d))
        nn.init.trunc_normal_(self.lateral_pos, std=0.02)
        self.pre_sa = AttnBlock(d, n_heads)
        self.cross = AttnBlock(d, n_heads)
        self.post_sa = AttnBlock(d, n_heads)

    def forward(self, tok_curr, tok_prev):
        pos = self.lateral_pos
        c = self.pre_sa(tok_curr + pos)
        p = self.pre_sa(tok_prev + pos)
        m = self.cross(c, kv=p)
        return self.post_sa(m)


class SCCControllerImitatorV2(nn.Module):
    """Pretrained-compatible SCC controller.

    Pretrained core hyperparams (from controller_imitator #654 args.json):
        traj_dim=3, state_dim=6, d_model=128, nhead=8, dim_feedforward=512,
        dropout=0.1, num_layers=6, seq_length=40 (so pos max_len=41).
    """

    def __init__(self,
                 img_backbone: str = "efficientnet_b0",
                 d_model: int = 128,
                 nhead: int = 8,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 num_layers: int = 6,
                 traj_seq_len: int = 40,
                 traj_dim: int = 3,
                 state_dim: int = 6,
                 num_model_output: int = 2,
                 image_pos_init_std: float = 0.02):
        super().__init__()
        self.d_model = d_model
        self.traj_seq_len = traj_seq_len
        self.traj_dim = traj_dim
        self.state_dim = state_dim
        self.num_model_output = num_model_output

        # ---- pretrained-compatible core ----
        self.traj_embed = nn.Linear(self.traj_dim, d_model)
        self.state_embed = nn.Linear(self.state_dim, d_model)
        # max_len ≥ 41 (controller_imitator seq_length+1). We allocate larger so we
        # can load pe of length 41 cleanly into first 41 entries.
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout, max_len=traj_seq_len + 1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, num_model_output)

        # ---- image stream ----
        self.img_encoder = LateralStripEncoder(img_backbone, d_token=d_model)
        self.img_fusion = ImageFusionBlock(d_model, n_heads=nhead)
        # image_memory 40 token 의 별도 positional embedding (sin-cos 가 아니라 learnable)
        self.image_pos_emb = nn.Parameter(torch.zeros(40, d_model))
        nn.init.trunc_normal_(self.image_pos_emb, std=image_pos_init_std)

    def forward(self, batch: dict) -> torch.Tensor:
        """Returns control_preds (B, 40, num_model_output)."""
        B = batch["traj_30"].shape[0] if "traj_30" in batch else batch["image_curr"].shape[0]
        # ---- traj / state tokens (controller_imitator 동작 그대로) ----
        traj = batch["traj_30"][:, :, :self.traj_dim]            # (B, 40, 3)
        state = batch["ego_5d"][:, :self.state_dim]              # (B, 5 or 6) — pad if needed
        if state.shape[-1] < self.state_dim:
            pad = torch.zeros(B, self.state_dim - state.shape[-1],
                              dtype=state.dtype, device=state.device)
            state = torch.cat([state, pad], dim=-1)
        traj_tokens = self.traj_embed(traj)                      # (B, 40, D)
        state_token = self.state_embed(state).unsqueeze(1)       # (B, 1, D)
        tokens_text = traj_tokens + state_token                  # broadcast → (B, 40, D)
        tokens_text = self.pos_encoder(tokens_text)              # + first-40 pos

        # ---- image tokens ----
        tok_curr = self.img_encoder(batch["image_curr"])
        tok_prev = self.img_encoder(batch["image_prev_rot"])
        image_mem = self.img_fusion(tok_curr, tok_prev)          # (B, 40, D)
        image_mem = image_mem + self.image_pos_emb               # (B, 40, D)

        # ---- concat + self-attn ----
        tokens = torch.cat([tokens_text, image_mem], dim=1)      # (B, 80, D)
        encoded = self.transformer_encoder(tokens)
        ctrl_tokens = encoded[:, :self.traj_seq_len]             # (B, 40, D)
        return self.fc_out(ctrl_tokens)                          # (B, 40, num_out)

    def load_controller_imitator_pretrained(self, ckpt_path: str, strict: bool = False):
        """Load weights from controller_imitator SelfAttentionControlPredictor_Professor ckpt."""
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # state_dict 만 들어있다면 그대로, dict 면 'model' key
        if "model" in sd: sd = sd["model"]

        # Map keys (controller_imitator → ours; same names already mostly)
        # pos_encoder.pe (1, 41, 128) → register_buffer (1, 41, 128). Direct copy if shapes match.
        my_sd = self.state_dict()
        loaded = {}
        skipped = []
        for k, v in sd.items():
            if k in my_sd and my_sd[k].shape == v.shape:
                loaded[k] = v
            elif k == "pos_encoder.pe":
                # if my max_len > 41, copy first 41
                if k in my_sd and my_sd[k].shape[1] >= v.shape[1]:
                    new = my_sd[k].clone()
                    new[:, :v.shape[1]] = v
                    loaded[k] = new
                else:
                    skipped.append((k, tuple(v.shape), tuple(my_sd[k].shape) if k in my_sd else None))
            else:
                skipped.append((k, tuple(v.shape), tuple(my_sd[k].shape) if k in my_sd else None))
        missing, unexpected = self.load_state_dict(loaded, strict=False)
        print(f"loaded {len(loaded)} keys from {ckpt_path}")
        if missing: print(f"missing ({len(missing)}):", missing[:6], "..." if len(missing) > 6 else "")
        if skipped: print(f"skipped ({len(skipped)}):", skipped[:6], "..." if len(skipped) > 6 else "")
        return loaded, missing, skipped
