"""Learned temporal gate filter modules (규정-clean EMA 대체).

입력 = 최근 K 프레임 in-lane obj(10-bin) 의 명시적 window (B,K,10). 모듈 내부에서
per-frame amax → temporal filter → smoothed gate ∈[0,1]. cross-call 누적 state 없음
(window 는 매 프레임 새로 stack) → §1.3 differentiable single-chain 만족.

ckpt 포맷: {"module": name, "K": K, "state": state_dict}.
side-effect 없음 (closed_loop_node 가 import 해도 CUDA/thread 설정 안 건드림).
"""
from __future__ import annotations
import torch
import torch.nn as nn

DEP_THRESH = 0.3
DEP_SCALE = 5.0


class LearnableIIR(nn.Module):
    """window 위에서 learnable alpha 로 EMA 를 unroll (NN 이 EMA 재현 — distill sanity)."""
    def __init__(self):
        super().__init__()
        self.alpha_logit = nn.Parameter(torch.tensor(0.85))
        self.thresh = nn.Parameter(torch.tensor(DEP_THRESH))
        self.scale = nn.Parameter(torch.tensor(DEP_SCALE))

    def forward(self, X):                            # X (B,K,10)
        amax = X.max(dim=2).values                   # (B,K)
        graw = torch.sigmoid((amax - self.thresh) * self.scale)
        a = torch.sigmoid(self.alpha_logit)
        g = torch.zeros(X.shape[0], device=X.device)
        for k in range(X.shape[1]):
            g = a * g + (1.0 - a) * graw[:, k]
        return g                                     # (B,)


class TemporalConv1d(nn.Module):
    """K-tap causal conv over per-frame amax → MLP → gate."""
    def __init__(self, K, in_ch=1, hidden=8):
        super().__init__()
        self.in_ch = in_ch
        self.conv = nn.Conv1d(in_ch, hidden, kernel_size=K, padding=0, bias=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, X):                            # X (B,K,10)
        if self.in_ch == 1:
            feat = X.max(dim=2).values.unsqueeze(1)  # (B,1,K)
        else:
            feat = X.transpose(1, 2)                 # (B,10,K)
        h = torch.relu(self.conv(feat)).squeeze(-1)  # (B,hidden)
        return torch.sigmoid(self.fc(h)).squeeze(-1) # (B,)


class TemporalMLP(nn.Module):
    """window per-frame amax 시퀀스 (K,) → tiny MLP → gate (적응형)."""
    def __init__(self, K, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(K, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, X):                            # X (B,K,10)
        amax = X.max(dim=2).values                   # (B,K)
        return torch.sigmoid(self.net(amax)).squeeze(-1)


def make_module(name, K):
    if name == "iir":
        return LearnableIIR()
    if name == "conv":
        return TemporalConv1d(K, in_ch=1, hidden=8)
    if name == "mlp":
        return TemporalMLP(K, hidden=16)
    raise ValueError(name)


def load_gate_filter(ckpt_path, map_location="cpu"):
    """ckpt → (module eval-mode, K). closed_loop 추론용 편의 로더."""
    ck = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    net = make_module(ck["module"], ck["K"])
    net.load_state_dict(ck["state"])
    net.eval()
    return net, int(ck["K"])
