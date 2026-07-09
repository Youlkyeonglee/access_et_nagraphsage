"""
DecompTemporalEncoder — 주파수 분해(저/고주파) 시간 인코더 (기여 프로토타입 ①)
====================================================================================
"GRU를 적용"이 아니라 "무엇을 인코딩하나"의 신규성:
  입력 시퀀스를 저주파(경로추종/링-따라가기)와 고주파(기동/차선변경)로 분해 후 인코딩.
    low  = 학습형 depthwise 저역통과(이동평균 초기화)  ← path-following 추세
    high = x - low                                      ← maneuver 잔차
    z    = [low ; high]  (채널 2배) → 기존 TemporalEncoder
목적: LC↔Normal 경계(school 혼동행렬 병목)를 입력 표현 단계에서 분리.
기존 temporal_encoder.py 불변 — 이를 감싸기만 함.
"""
import torch
import torch.nn as nn
from .temporal_encoder import TemporalEncoder


class DecompTemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, encoder_type='gru', use_attention=True,
                 kernel=5, learnable=True, num_layers=1, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.kernel = kernel
        # 저역통과: 채널별(depthwise) 1D conv. 이동평균(1/k)으로 초기화 후 학습.
        self.lowpass = nn.Conv1d(input_dim, input_dim, kernel,
                                 padding=kernel // 2, groups=input_dim, bias=False)
        with torch.no_grad():
            self.lowpass.weight.fill_(1.0 / kernel)
        if not learnable:
            self.lowpass.weight.requires_grad_(False)
        # 분해된 [low;high] (2*input_dim)를 기존 인코더로
        self.enc = TemporalEncoder(2 * input_dim, hidden_dim, encoder_type=encoder_type,
                                   num_layers=num_layers, dropout=dropout,
                                   use_attention=use_attention)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        xt = x.transpose(1, 2)                 # [B, C, T]
        low = self.lowpass(xt)
        # padding으로 길이 유지되지만 짝수 커널 대비 안전하게 자름
        low = low[..., :xt.shape[-1]].transpose(1, 2)   # [B, T, C]
        high = x - low                          # 고주파 잔차 = 기동 성분
        z = torch.cat([low, high], dim=-1)      # [B, T, 2C]
        return self.enc(z)
