"""SCC controller imitator — image+ego+traj → control model.

DESIGN.md §4 simplified baseline:
- LateralStripEncoder (timm efficientnet_b0, features_only) → (B, 40, D)
- ImageFusionBlock (Step 0 SA shared + Step 1 cross-attn curr Q × prev KV + Step 2 SA)
- ego_5d → 1 token
- traj_30 → 30 tokens + learned pos
- Decoder: 30 control queries × cross-attn (image_memory + ego + traj_tokens)
- Head: (B, 30, 2) = [steer/40, accel]

ROI=(250, 1280), 40 lateral tokens, D=256.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class LateralStripEncoder(nn.Module):
    def __init__(self, backbone: str = "efficientnet_b0", d_token: int = 256):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=True, features_only=True, out_indices=[-1]
        )
        C_b = self.backbone.feature_info.channels()[-1]
        self.proj = nn.Conv1d(C_b, d_token, kernel_size=1)
        self.d_token = d_token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, 3, 250, 1280) → (B, 40, D)"""
        feat = self.backbone(x)[-1]                              # (B, C_b, ~8, ~40)
        feat = F.adaptive_avg_pool2d(feat, (1, 40)).squeeze(2)   # (B, C_b, 40)
        feat = self.proj(feat)                                   # (B, D, 40)
        return feat.transpose(1, 2)                              # (B, 40, D)


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
    """Step 0 SA shared + Step 1 cross-attn curr Q × prev KV + Step 2 SA → 40 tokens."""

    def __init__(self, d: int = 256, n_heads: int = 4):
        super().__init__()
        self.lateral_pos = nn.Parameter(torch.zeros(40, d))
        nn.init.trunc_normal_(self.lateral_pos, std=0.02)
        self.pre_sa = AttnBlock(d, n_heads)
        self.cross = AttnBlock(d, n_heads)
        self.post_sa = AttnBlock(d, n_heads)

    def forward(self, tok_curr: torch.Tensor, tok_prev: torch.Tensor) -> torch.Tensor:
        pos = self.lateral_pos
        c = self.pre_sa(tok_curr + pos)
        p = self.pre_sa(tok_prev + pos)
        m = self.cross(c, kv=p)
        m = self.post_sa(m)
        return m


class SCCControllerImitator(nn.Module):
    def __init__(self,
                 img_backbone: str = "efficientnet_b0",
                 d_token: int = 256,
                 n_heads: int = 4,
                 traj_horizon: int = 30,
                 traj_feat_dim: int = 4,
                 ego_dim: int = 5,
                 n_decoder_layers: int = 2,
                 out_dim: int = 2):
        super().__init__()
        self.encoder = LateralStripEncoder(img_backbone, d_token)
        self.fusion = ImageFusionBlock(d_token, n_heads)
        self.ego_proj = nn.Sequential(
            nn.Linear(ego_dim, d_token), nn.GELU(), nn.Linear(d_token, d_token))
        self.traj_proj = nn.Sequential(
            nn.Linear(traj_feat_dim, d_token), nn.GELU(), nn.Linear(d_token, d_token))
        self.traj_pos = nn.Parameter(torch.zeros(traj_horizon, d_token))
        nn.init.trunc_normal_(self.traj_pos, std=0.02)
        self.ctrl_queries = nn.Parameter(torch.zeros(traj_horizon, d_token))
        nn.init.trunc_normal_(self.ctrl_queries, std=0.02)
        self.dec_layers = nn.ModuleList([
            AttnBlock(d_token, n_heads) for _ in range(n_decoder_layers)
        ])
        self.head = nn.Linear(d_token, out_dim)
        self.traj_horizon = traj_horizon

    def forward(self, batch: dict) -> torch.Tensor:
        B = batch["image_curr"].shape[0]
        tok_curr = self.encoder(batch["image_curr"])
        tok_prev = self.encoder(batch["image_prev_rot"])
        image_mem = self.fusion(tok_curr, tok_prev)              # (B, 40, D)
        ego_tok = self.ego_proj(batch["ego_5d"]).unsqueeze(1)    # (B, 1, D)
        traj_tok = self.traj_proj(batch["traj_30"]) + self.traj_pos  # (B, 30, D)
        memory = torch.cat([image_mem, ego_tok, traj_tok], dim=1)    # (B, 71, D)
        q = self.ctrl_queries.unsqueeze(0).expand(B, -1, -1)         # (B, 30, D)
        for layer in self.dec_layers:
            q = layer(q, kv=memory)
        return self.head(q)                                          # (B, 30, 2)
