"""
ET-NAGraphSAGE + Supervised Contrastive Loss Head
==================================================
et_nagraphsage.py 대비 변경점:
  - projection head 추가: h_ego_updated → MLP → L2-normalized embedding z
  - forward(batch, return_embeddings=False)
      return_embeddings=True  → (logits, z)   ← 학습 시
      return_embeddings=False → logits         ← 추론/평가 시

SupCon 핵심 아이디어:
  같은 클래스(LC끼리, Normal끼리)는 임베딩 공간에서 가깝게,
  다른 클래스(LC vs Normal)는 멀게 만들어 CE만으로는 분리되지 않던
  LC/Normal 경계를 명시적으로 벌린다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .et_nagraphsage import ETSAGELayer, _masked_softmax
from .temporal_encoder import TemporalEncoder


class ETNAGraphSAGESupCon(nn.Module):
    """
    ET-NAGraphSAGE with projection head for Supervised Contrastive Learning.

    Args:
        node_dim     : 노드 피처 차원 (6)
        edge_dim     : 엣지 피처 차원 (5)
        hidden_dim   : 공간 레이어 은닉 차원
        d_e          : 엣지 인코더 출력 차원
        T            : 시계열 길이
        encoder_type : 'gru' | 'lstm' | 'mamba'
        use_attention: temporal attention 활성화
        use_2hop     : 2-hop 집계 활성화
        num_classes  : 분류 클래스 수 (3)
        dropout      : dropout 확률
        proj_dim     : projection head 출력 차원 (SupCon 임베딩 차원)
    """

    def __init__(
        self,
        node_dim:     int   = 6,
        edge_dim:     int   = 5,
        hidden_dim:   int   = 128,
        d_e:          int   = 32,
        T:            int   = 10,
        encoder_type: str   = 'gru',
        use_attention: bool = True,
        use_2hop:     bool  = False,
        num_classes:  int   = 3,
        dropout:      float = 0.3,
        proj_dim:     int   = 64,
    ):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.d_e         = d_e
        self.T           = T
        self.use_2hop    = use_2hop

        enc_kwargs = dict(encoder_type=encoder_type, use_attention=use_attention)
        self.node_encoder = TemporalEncoder(node_dim, hidden_dim, **enc_kwargs)
        self.edge_encoder = TemporalEncoder(edge_dim, d_e,        **enc_kwargs)

        if use_2hop:
            self.layer_2hop = ETSAGELayer(hidden_dim, hidden_dim, d_e, dropout)
        self.layer_1hop = ETSAGELayer(hidden_dim, hidden_dim, d_e, dropout)

        self.classifier = nn.Linear(hidden_dim, num_classes)

        # Projection head for SupCon: hidden_dim → proj_dim (L2 normalized)
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def _encode(self, batch: dict) -> torch.Tensor:
        """공유 인코딩 파이프라인 → h_ego_updated [B, hidden_dim]."""
        node_seq       = batch['node_seq']
        nbr_node_seqs  = batch['nbr_node_seqs']
        edge_seqs      = batch['edge_seqs']
        nbr_mask       = batch['nbr_mask']
        nbr2_node_seqs = batch['nbr2_node_seqs']
        nbr2_edge_seqs = batch['nbr2_edge_seqs']
        nbr2_mask      = batch['nbr2_mask']

        B, K1, T, _ = nbr_node_seqs.shape

        h_ego = self.node_encoder(node_seq)
        h_nbr = self.node_encoder(
            nbr_node_seqs.view(B * K1, T, -1)
        ).view(B, K1, -1)
        h_nbr = h_nbr * nbr_mask.unsqueeze(-1)

        e1 = self.edge_encoder(
            edge_seqs.view(B * K1, T, -1)
        ).view(B, K1, -1)
        e1 = e1 * nbr_mask.unsqueeze(-1)

        if self.use_2hop and nbr2_node_seqs.shape[2] > 0:
            K2 = nbr2_node_seqs.shape[2]
            h_nbr2 = self.node_encoder(
                nbr2_node_seqs.view(B * K1 * K2, T, -1)
            ).view(B, K1, K2, -1)
            h_nbr2 = h_nbr2 * nbr2_mask.unsqueeze(-1)

            e2 = self.edge_encoder(
                nbr2_edge_seqs.view(B * K1 * K2, T, -1)
            ).view(B, K1, K2, -1)
            e2 = e2 * nbr2_mask.unsqueeze(-1)

            h_nbr_updated = self.layer_2hop(
                h_ego    = h_nbr.view(B * K1, -1),
                h_nbr    = h_nbr2.view(B * K1, K2, -1),
                e_temp   = e2.view(B * K1, K2, -1),
                nbr_mask = nbr2_mask.view(B * K1, K2),
            ).view(B, K1, -1)
            h_nbr_updated = h_nbr_updated * nbr_mask.unsqueeze(-1)
        else:
            h_nbr_updated = h_nbr

        return self.layer_1hop(h_ego, h_nbr_updated, e1, nbr_mask)

    def forward(
        self,
        batch: dict,
        return_embeddings: bool = False,
    ):
        """
        Args:
            batch             : 배치 딕셔너리
            return_embeddings : True → (logits, z_normalized) 반환
                                False → logits 만 반환 (추론/평가용)
        """
        h = self._encode(batch)                          # [B, hidden_dim]
        logits = self.classifier(h)                      # [B, num_classes]

        if return_embeddings:
            z = F.normalize(self.proj_head(h), dim=-1)  # [B, proj_dim]
            return logits, z

        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
