"""
Temporal Encoder — GRU / LSTM / Mamba / Transformer 공통 인터페이스
===================================================================
입력: [B, T, input_dim]
출력: [B, hidden_dim]

use_attention=True 시 GRU/LSTM 전체 hidden states에 attention을 적용해
어느 시점이 중요한지 학습 (단순 last-state 대비 시계열 정보 손실 감소).

encoder_type='transformer' (2026-07-12, §실험현황 "20260712 추가 실험계획" ④ 전용):
comparison/transformer_tsem과 동일 패턴(2-layer nn.TransformerEncoder, sinusoidal PE,
causal mask)을 재사용 — "W=10처럼 짧은 시퀀스에서는 GRU와 Transformer 차이가 묻힐 수 있다"는
가설을 W를 늘려가며(W=10/20/30) 검증하기 위한 대조군. causal mask를 쓰는 이유는 GRU의
"과거만 보고 현재까지 요약" 성질과 동등하게 맞추기 위함(미래 프레임 누설 방지).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


class TemporalEncoder(nn.Module):
    """
    Args:
        input_dim    : 입력 피처 차원 (노드=6, 엣지=5)
        hidden_dim   : 출력 은닉 차원
        encoder_type : 'gru' | 'lstm' | 'mamba' | 'transformer'
        num_layers   : RNN 레이어 수 (기본 1) — transformer는 항상 2-layer 고정
        dropout      : 레이어 간 dropout (num_layers > 1일 때만 적용, transformer는 항상 적용)
        use_attention: True면 전체 hidden states에 scaled dot-product attention 적용
                       (transformer는 causal self-attention이 이미 있어 별도 pooling attention도
                       True/False 둘 다 지원 — True면 GRU/LSTM과 동일한 방식으로 가중합)
    """

    MAX_LEN = 64  # W=30까지 커버(여유 포함) — sinusoidal PE 사전 계산 길이

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
        elif self.encoder_type == 'transformer':
            assert hidden_dim % 4 == 0, 'transformer encoder는 hidden_dim이 4(nhead)의 배수여야 함'
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.register_buffer('pos_enc', _sinusoidal_pe(self.MAX_LEN, hidden_dim), persistent=False)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 4,
                dropout=dropout, batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=2)
            # nn.TransformerEncoder의 fused fast-path(_transformer_encoder_layer_fwd)는
            # SDPA(및 아래 math 커널 강제)를 우회하는 별도 CUDA 커널을 쓰는데, _encode_nodes가
            # 2-hop까지 펼친 대배치(특히 DataParallel replica)에서 CUBLAS_STATUS_EXECUTION_FAILED로
            # 실패한다. fast-path를 끄면 항상 표준 SDPA 경로를 타 math 커널 강제가 적용된다.
            torch.backends.mha.set_fastpath_enabled(False)
        else:
            raise ValueError(f"지원하지 않는 encoder_type: {encoder_type}")

        # temporal attention: hidden states → scalar score per timestep
        if use_attention and self.encoder_type in ('gru', 'lstm', 'transformer'):
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

        elif self.encoder_type == 'transformer':
            B, T, _ = x.shape
            h = self.input_proj(x) + self.pos_enc[:T].unsqueeze(0)   # [B,T,hidden_dim]
            mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)  # causal
            # _encode_nodes가 2-hop 이웃까지 펼쳐(B*K1*K2 ≈ 1e5) SDPA에 넘기면 flash/mem-efficient
            # 커널의 grid 차원 한계를 넘어 "CUDA error: invalid configuration argument"로 실패한다.
            # T가 작아 비용이 무시할 만하므로 math 커널을 강제해 이 한계를 회피한다.
            with sdpa_kernel([SDPBackend.MATH]):
                output = self.transformer(h, mask=mask, is_causal=True)  # [B,T,hidden_dim]
            return self._aggregate(output)

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


def _sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    """표준 sinusoidal positional encoding [max_len, d_model] (comparison/transformer_tsem과 동일)."""
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


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
