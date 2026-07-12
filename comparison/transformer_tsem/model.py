"""
Transformer-adapted — TSEM-SAGE 비교용 순수 아키텍처 대조군
=========================================
원본 GitHub repo 없음. 이 baseline은 특정 논문 재구현이 아니라 "표준 시간 인코더(Transformer)는
있지만 학습된 공간/그래프 상호작용 모듈이 전혀 없을 때 얼마나 되는가"를 보기 위한 순수 대조군이다.

comparison/lstm_tsem/(다른 에이전트가 병행 작성)과 짝을 이루는 실험 설계다 — 두 baseline은
**공간 처리 부분을 의도적으로 동일하게**(masked mean-pooling, 그래프/attention 전혀 없음) 맞추고
**시간 인코더만 LSTM(lstm_tsem) vs Transformer(여기)로 다르게** 해서, "시간 인코더를
LSTM→Transformer로 바꾸면(공간 모듈 없이) 뭐가 달라지는가"라는 단일 축 대조가 가능하도록 설계했다.
HiVT-adapted(comparison/hivt_tsem/)와 달리 agent-agent attention이나 global interactor 같은
학습된 공간 모듈이 전혀 없다 — 순수하게 "시간축만 Transformer로 인코딩 + 이웃은 단순 평균".

파이프라인:
  1. scene 1회 정렬 — ego의 anchor(t=W-1) 위치·헤딩 기준으로 ego+이웃(K=6) 전체를 회전·평행이동
     (CRAT-Pred/HiVT-adapted의 _build_graph와 동일 관례).
  2. 노드 입력 6D = [Δpos_x, Δpos_z(회전된 프레임간 변위), speed, dir_x, dir_z, accel]
     — HiVT-adapted와 동일한 정보량(원시 6D 전 채널).
  3. 표준 nn.TransformerEncoder(sinusoidal positional encoding, causal mask) — ego와 이웃에
     동일 가중치를 공유하는 하나의 인코더를 배치×에이전트(B*N)를 합쳐 한 번에 forward한다.
     causal mask를 써서 마지막 timestep(t=W-1)의 출력이 "그 시점까지의 표현"이 되도록 한다
     (LSTM의 마지막 hidden state와 대응되는 선택 — CRAT-Pred의 EncoderLstm이 out[:,-1,:]을
     쓰는 것과 동일한 아이디어).
  4. 공간 처리 — **의도적으로 없음**. 그래프도, attention도, GNN도 쓰지 않고 유효한 이웃
     임베딩을 단순 masked mean-pooling한다(무효 슬롯은 nbr_mask로 제외).
  5. 분류 헤드 — concat([ego_embed, pooled_nbr_embed]) -> MLP -> num_classes.

주요 변경점(README.md 참조):
  - 원본 repo가 없으므로 "표준 관행에서 벗어난 부분"이라는 개념 자체가 없다 — 설계 전체가
    이 프로젝트를 위해 새로 정의됐다(단, Transformer 시간 인코더 구성은 HiVT-adapted의
    TemporalEncoder에서 causal mask·sinusoidal-position-embedding 관례를 참고했다).
  - 2-hop 이웃 미사용 — ego + 1-hop(K=6)까지만 사용(다른 5개 baseline과 동일 관례).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """표준 sinusoidal positional encoding — 시간축(t=0..W-1) 위치 정보를 입력 임베딩에 더한다.
    학습 파라미터 없음(고정 buffer), Vaswani et al. 2017 수식 그대로."""

    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                              (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)[:, : pe[:, 1::2].size(1)]
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, T, D] (batch_first) -> x + PE[:, :T, :]"""
        return x + self.pe[:, : x.size(1), :]


def rotate2(vec: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """vec: [N,2], rot: [N,2,2] -> [N,2] (comparison/hivt_tsem, cratpred_tsem/model.py와 동일)."""
    return torch.bmm(vec.unsqueeze(-2), rot).squeeze(-2)


class TransformerTSEMAdapted(nn.Module):
    """Transformer 시간 인코더 + masked mean-pooling 공간 처리 — batch dict(TSEM dataloader
    그대로) -> logits[B,num_classes].

    lstm_tsem(다른 에이전트가 작성 중)과 공간 처리 설계를 반드시 동일하게 맞춘 대조 실험 —
    시간 인코더만 Transformer로 교체했을 때의 효과를 보기 위함.
    """

    def __init__(self, W: int = 10, K: int = 6, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 128, dropout: float = 0.1,
                 num_classes: int = 3):
        super().__init__()
        self.W = W
        self.K = K
        self.N = 1 + K  # ego + K neighbors
        self.d_model = d_model

        self.input_proj = nn.Linear(6, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=max(64, W + 1))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='relu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # causal mask: 위치 t는 <=t만 볼 수 있음(LSTM의 순차적 hidden state와 대응)
        causal_mask = torch.triu(torch.full((W, W), float('-inf')), diagonal=1)
        self.register_buffer('causal_mask', causal_mask)

        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def _build_inputs(self, batch: dict):
        """batch -> (x_seq[B*N,W,6], valid_agent[B,N])
        scene 1회 정렬(ego anchor 위치·헤딩 기준 회전+평행이동) — CRAT-Pred/HiVT-adapted와 동일."""
        device = batch['node_seq'].device
        B = batch['node_seq'].size(0)
        N, W = self.N, self.W

        raw = torch.cat([batch['node_seq'].unsqueeze(1), batch['nbr_node_seqs']], dim=1)  # [B,N,W,6]
        nbr_mask = batch.get('nbr_mask')
        if nbr_mask is not None:
            valid_agent = torch.cat(
                [torch.ones(B, 1, device=device, dtype=torch.bool), nbr_mask.bool()], dim=1)  # [B,N]
        else:
            valid_agent = raw.abs().sum(dim=(-1, -2)) > 0

        present = raw.abs().sum(dim=-1) > 0  # [B,N,W]

        ego_pos_anchor = batch['node_seq'][:, -1, 0:2]  # [B,2]
        ego_dir_anchor = batch['node_seq'][:, -1, 3:5]  # [B,2]
        theta = torch.atan2(ego_dir_anchor[:, 1], ego_dir_anchor[:, 0])
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        rot = torch.stack([torch.stack([cos_t, -sin_t], -1), torch.stack([sin_t, cos_t], -1)], -2)  # [B,2,2]
        rot_flat = rot.repeat_interleave(N, dim=0)  # [B*N,2,2]
        origin_flat = ego_pos_anchor.repeat_interleave(N, dim=0)  # [B*N,2]

        pos_raw = raw[..., 0:2].reshape(B * N, W, 2)
        dir_raw = raw[..., 3:5].reshape(B * N, W, 2)
        present_flat = present.reshape(B * N, W)

        pos_centered = pos_raw - origin_flat.unsqueeze(1)
        rot_expand = rot_flat.unsqueeze(1).expand(-1, W, -1, -1).reshape(-1, 2, 2)
        positions = torch.bmm(pos_centered.reshape(-1, 1, 2), rot_expand).reshape(B * N, W, 2)
        dir_rot = torch.bmm(dir_raw.reshape(-1, 1, 2), rot_expand).reshape(B * N, W, 2)
        positions = torch.where(present_flat.unsqueeze(-1), positions, torch.zeros_like(positions))
        dir_rot = torch.where(present_flat.unsqueeze(-1), dir_rot, torch.zeros_like(dir_rot))

        speed = raw[..., 2].reshape(B * N, W)
        accel = raw[..., 5].reshape(B * N, W)
        speed = torch.where(present_flat, speed, torch.zeros_like(speed))
        accel = torch.where(present_flat, accel, torch.zeros_like(accel))

        # frame-to-frame displacement — 양쪽 프레임 다 유효할 때만(t=0은 0)
        pos_disp = torch.zeros_like(positions)
        both_valid = present_flat[:, 1:] & present_flat[:, :-1]
        pos_disp[:, 1:] = torch.where(both_valid.unsqueeze(-1), positions[:, 1:] - positions[:, :-1],
                                       torch.zeros_like(positions[:, 1:]))

        x_seq = torch.cat([pos_disp, speed.unsqueeze(-1), dir_rot, accel.unsqueeze(-1)], dim=-1)  # [B*N,W,6]
        return x_seq, valid_agent

    def forward(self, batch: dict) -> torch.Tensor:
        x_seq, valid_agent = self._build_inputs(batch)  # x_seq:[B*N,W,6], valid_agent:[B,N]
        B, N = valid_agent.shape
        device = x_seq.device

        h = self.input_proj(x_seq)               # [B*N, W, D]
        h = self.pos_encoding(h)                  # + sinusoidal PE
        h = self.transformer(h, mask=self.causal_mask.to(dtype=h.dtype))  # [B*N, W, D]
        agent_embed = h[:, -1, :]                 # 마지막 timestep(anchor 시점) 출력만 사용

        agent_embed = agent_embed.view(B, N, self.d_model)
        ego_embed = agent_embed[:, 0, :]           # ego는 항상 slot 0
        nbr_embed = agent_embed[:, 1:, :]           # [B, K, D]
        nbr_valid = valid_agent[:, 1:].float()      # [B, K]

        nbr_sum = (nbr_embed * nbr_valid.unsqueeze(-1)).sum(dim=1)  # [B, D]
        nbr_count = nbr_valid.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1] — 0-division 방지
        pooled_nbr = nbr_sum / nbr_count

        combined = torch.cat([ego_embed, pooled_nbr], dim=-1)  # [B, 2D]
        return self.classifier(combined)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
