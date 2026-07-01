"""
Temporal Encoder — GRU / LSTM / Mamba 공통 인터페이스
=====================================================
입력: [B, T, input_dim]
출력: [B, hidden_dim]

use_attention=True 시 GRU/LSTM 전체 hidden states에 attention을 적용해
어느 시점이 중요한지 학습 (단순 last-state 대비 시계열 정보 손실 감소).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalEncoder(nn.Module):
    """
    Args:
        input_dim    : 입력 피처 차원 (노드=6, 엣지=5)
        hidden_dim   : 출력 은닉 차원
        encoder_type : 'gru' | 'lstm' | 'mamba'
        num_layers   : RNN 레이어 수 (기본 1)
        dropout      : 레이어 간 dropout (num_layers > 1일 때만 적용)
        use_attention: True면 전체 hidden states에 scaled dot-product attention 적용
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        encoder_type: str  = 'gru',
        num_layers: int    = 1,
        dropout: float     = 0.0,
        use_attention: bool = True,
    ):
        super().__init__()
        self.encoder_type  = encoder_type.lower()
        self.hidden_dim    = hidden_dim
        self.num_layers    = num_layers
        self.use_attention = use_attention

        if self.encoder_type == 'gru':
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif self.encoder_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif self.encoder_type == 'mamba':
            self.rnn = _build_mamba(input_dim, hidden_dim)
        else:
            raise ValueError(f"지원하지 않는 encoder_type: {encoder_type}")

        # temporal attention: hidden states → scalar score per timestep
        if use_attention and self.encoder_type in ('gru', 'lstm'):
            self.attn_proj = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, input_dim]
        반환: [B, hidden_dim]
        """
        if self.encoder_type == 'gru':
            output, _ = self.rnn(x)          # output: [B, T, hidden_dim]
            return self._aggregate(output)

        elif self.encoder_type == 'lstm':
            output, _ = self.rnn(x)          # output: [B, T, hidden_dim]
            return self._aggregate(output)

        elif self.encoder_type == 'mamba':
            return self.rnn(x)               # Mamba wrapper → [B, hidden_dim]

    def _aggregate(self, output: torch.Tensor) -> torch.Tensor:
        """
        output: [B, T, hidden_dim]
        use_attention=True : attention-weighted sum over T
        use_attention=False: 마지막 타임스텝
        """
        if self.use_attention:
            score = self.attn_proj(output)           # [B, T, 1]
            alpha = F.softmax(score, dim=1)          # [B, T, 1]
            return (alpha * output).sum(dim=1)       # [B, hidden_dim]
        else:
            return output[:, -1, :]                  # [B, hidden_dim]


def _build_mamba(input_dim: int, hidden_dim: int) -> nn.Module:
    try:
        from mamba_ssm import Mamba
        return _MambaWrapper(input_dim, hidden_dim)
    except ImportError:
        import warnings
        warnings.warn(
            "mamba_ssm 미설치 — GRU로 fallback.",
            stacklevel=3,
        )
        return nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)


class _MambaWrapper(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        from mamba_ssm import Mamba
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.mamba   = Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2)
        self.norm    = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        x = self.mamba(x)
        x = self.norm(x)
        return x[:, -1, :]
