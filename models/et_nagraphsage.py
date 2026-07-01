"""
ET-NAGraphSAGE — Edge-Temporal Neighbor-Aware GraphSAGE (2-hop)
================================================================
배치 입력 (data_manager.py 출력 기준):
  node_seq        : [B, T, 6]
  nbr_node_seqs   : [B, K1, T, 6]
  edge_seqs       : [B, K1, T, 5]
  nbr_mask        : [B, K1]
  nbr2_node_seqs  : [B, K1, K2, T, 6]
  nbr2_edge_seqs  : [B, K1, K2, T, 5]
  nbr2_mask       : [B, K1, K2]

3단계 파이프라인:
  STAGE 1  : Temporal Encoder (node/edge 각각 GRU+Attention)
             → h_ego[B,d], h_nbr[B,K1,d], e1[B,K1,d_e]
               h_nbr2[B,K1,K2,d], e2[B,K1,K2,d_e]

  STAGE 2  : 2-hop Spatial (ETSAGELayer × 2)
    Layer 1 (2→1 hop): 각 1-hop 이웃이 자신의 2-hop 이웃을 집계
             → h_nbr_updated[B,K1,d]
    Layer 2 (1→0 hop): ego가 업데이트된 1-hop 이웃을 집계
             → h_ego_updated[B,d]

  STAGE 3  : Classifier → logits[B,3]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .temporal_encoder import TemporalEncoder


# ─────────────────────────────────────────────────────────────────────────────
# 보조 함수
# ─────────────────────────────────────────────────────────────────────────────

def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    scores: [B, K]  mask: [B, K] (1=real, 0=pad)
    반환  : [B, K]  실제 이웃끼리만 softmax, 패딩은 0
    """
    scores = scores.masked_fill(mask == 0, -1e4)   # fp16 호환: -1e9 대신 -1e4
    return F.softmax(scores, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 단일 ET-SAGE 레이어 (C1·C2·C3)
# ─────────────────────────────────────────────────────────────────────────────

class ETSAGELayer(nn.Module):
    """
    하나의 공간 집계 레이어. ego ← aggregate(neighbors).

    C1 (Vector Gate)   : α = sigmoid(W_α · e_temp) ∈ ℝ^d
                         h̃_nbr = α ⊙ h_nbr
    C2 (Temporal Msg)  : m = ReLU(Linear(cat(h̃_nbr, e_temp)))
    C3 (β Aggregation) : β = softmax(w_β · e_temp) ∈ ℝ¹
                         h_N = Σ_j β_j · m_j
    Update             : h_new = ReLU(Linear(cat(h_ego, h_N)))

    Args:
        in_dim  : ego 입력 차원
        out_dim : 출력 차원
        d_e     : 엣지 인코더 출력 차원
        dropout : dropout 확률
    """

    def __init__(self, in_dim: int, out_dim: int, d_e: int, dropout: float = 0.3):
        super().__init__()
        self.lin_gate   = nn.Linear(d_e, in_dim)          # C1
        self.lin_msg    = nn.Linear(in_dim + d_e, out_dim) # C2
        self.lin_beta   = nn.Linear(d_e, 1)               # C3
        self.lin_update = nn.Linear(in_dim + out_dim, out_dim)
        self.bn         = nn.BatchNorm1d(out_dim)
        self.dropout    = nn.Dropout(dropout)

    def forward(
        self,
        h_ego:    torch.Tensor,  # [B, in_dim]
        h_nbr:    torch.Tensor,  # [B, K, in_dim]
        e_temp:   torch.Tensor,  # [B, K, d_e]
        nbr_mask: torch.Tensor,  # [B, K]
    ) -> torch.Tensor:
        # C1: Vector Gate
        alpha       = torch.sigmoid(self.lin_gate(e_temp))     # [B, K, in_dim]
        h_nbr_gated = alpha * h_nbr                            # [B, K, in_dim]

        # C2: Message
        msg = F.relu(self.lin_msg(
            torch.cat([h_nbr_gated, e_temp], dim=-1)           # [B, K, in_dim+d_e]
        ))                                                      # [B, K, out_dim]
        msg = msg * nbr_mask.unsqueeze(-1)                     # 패딩 마스킹

        # C3: β Aggregation
        beta = _masked_softmax(
            self.lin_beta(e_temp).squeeze(-1), nbr_mask        # [B, K]
        )
        h_N = (beta.unsqueeze(-1) * msg).sum(dim=1)           # [B, out_dim]

        # 이웃 없는 ego 처리
        no_nbr = (nbr_mask.sum(dim=1, keepdim=True) == 0).float()
        h_N    = h_N * (1.0 - no_nbr)

        # Node Update
        h_new = F.relu(self.lin_update(
            torch.cat([h_ego, h_N], dim=-1)                    # [B, in_dim+out_dim]
        ))
        h_new = self.bn(h_new)
        h_new = self.dropout(h_new)
        return h_new


# ─────────────────────────────────────────────────────────────────────────────
# 전체 모델
# ─────────────────────────────────────────────────────────────────────────────

class ETNAGraphSAGE(nn.Module):
    """
    ET-NAGraphSAGE with true 2-hop spatial aggregation.

    Spatial pipeline:
      Layer 1 (2→1): h_nbr_i ← ETSAGELayer(h_nbr_i, h_nbr2_ij, e2_ij)
      Layer 2 (1→0): h_ego   ← ETSAGELayer(h_ego,   h_nbr_i,   e1_i )

    Args:
        node_dim     : 노드 피처 차원 (6)
        edge_dim     : 엣지 피처 차원 (5)
        hidden_dim   : 공간 레이어 은닉 차원
        d_e          : 엣지 인코더 출력 차원
        T            : 시계열 길이
        encoder_type : 'gru' | 'lstm' | 'mamba'
        use_attention: temporal attention 활성화
        use_2hop     : 2-hop 집계 활성화 (False면 Layer 1 skip)
        num_classes  : 분류 클래스 수 (3)
        dropout      : dropout 확률
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
        use_2hop:     bool  = True,
        num_classes:  int   = 3,
        dropout:      float = 0.3,
    ):
        super().__init__()
        self.hidden_dim   = hidden_dim
        self.d_e          = d_e
        self.T            = T
        self.use_2hop     = use_2hop

        # ── STAGE 1: Temporal Encoders ─────────────────────────────────────
        enc_kwargs = dict(encoder_type=encoder_type, use_attention=use_attention)
        self.node_encoder = TemporalEncoder(node_dim, hidden_dim, **enc_kwargs)
        self.edge_encoder = TemporalEncoder(edge_dim, d_e,        **enc_kwargs)

        # ── STAGE 2: Spatial Layers ────────────────────────────────────────
        # Layer 1: 2-hop → 1-hop  (nbr acts as ego, nbr2 as neighbors)
        if use_2hop:
            self.layer_2hop = ETSAGELayer(hidden_dim, hidden_dim, d_e, dropout)

        # Layer 2: 1-hop → ego
        self.layer_1hop = ETSAGELayer(hidden_dim, hidden_dim, d_e, dropout)

        # ── STAGE 3: Classifier ────────────────────────────────────────────
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch: dict) -> torch.Tensor:
        node_seq       = batch['node_seq']        # [B, T, 6]
        nbr_node_seqs  = batch['nbr_node_seqs']   # [B, K1, T, 6]
        edge_seqs      = batch['edge_seqs']        # [B, K1, T, 5]
        nbr_mask       = batch['nbr_mask']         # [B, K1]
        nbr2_node_seqs = batch['nbr2_node_seqs']  # [B, K1, K2, T, 6]
        nbr2_edge_seqs = batch['nbr2_edge_seqs']  # [B, K1, K2, T, 5]
        nbr2_mask      = batch['nbr2_mask']        # [B, K1, K2]

        B, K1, T, _ = nbr_node_seqs.shape

        # ── STAGE 1: Temporal Encoding ─────────────────────────────────────
        # ego
        h_ego = self.node_encoder(node_seq)                     # [B, d]

        # 1-hop neighbors
        h_nbr = self.node_encoder(
            nbr_node_seqs.view(B * K1, T, -1)                  # [B*K1, T, 6]
        ).view(B, K1, -1)                                       # [B, K1, d]
        h_nbr = h_nbr * nbr_mask.unsqueeze(-1)

        # 1-hop edges (ego → nbr)
        e1 = self.edge_encoder(
            edge_seqs.view(B * K1, T, -1)
        ).view(B, K1, -1)                                       # [B, K1, d_e]
        e1 = e1 * nbr_mask.unsqueeze(-1)

        # ── STAGE 2-Layer 1: 2-hop → 1-hop ───────────────────────────────
        if self.use_2hop and nbr2_node_seqs.shape[2] > 0:
            K2 = nbr2_node_seqs.shape[2]

            # 2-hop neighbor temporal encoding
            h_nbr2 = self.node_encoder(
                nbr2_node_seqs.view(B * K1 * K2, T, -1)
            ).view(B, K1, K2, -1)                              # [B, K1, K2, d]
            h_nbr2 = h_nbr2 * nbr2_mask.unsqueeze(-1)

            # 2-hop edge temporal encoding (nbr → nbr2)
            e2 = self.edge_encoder(
                nbr2_edge_seqs.view(B * K1 * K2, T, -1)
            ).view(B, K1, K2, -1)                              # [B, K1, K2, d_e]
            e2 = e2 * nbr2_mask.unsqueeze(-1)

            # ETSAGELayer: each 1-hop nbr aggregates its 2-hop nbrs
            # reshape: treat K1 neighbors as independent "ego" batch
            h_nbr_updated = self.layer_2hop(
                h_ego    = h_nbr.view(B * K1, -1),            # [B*K1, d]
                h_nbr    = h_nbr2.view(B * K1, K2, -1),       # [B*K1, K2, d]
                e_temp   = e2.view(B * K1, K2, -1),           # [B*K1, K2, d_e]
                nbr_mask = nbr2_mask.view(B * K1, K2),        # [B*K1, K2]
            ).view(B, K1, -1)                                  # [B, K1, d]

            # 패딩 이웃은 업데이트 무효화
            h_nbr_updated = h_nbr_updated * nbr_mask.unsqueeze(-1)
        else:
            h_nbr_updated = h_nbr

        # ── STAGE 2-Layer 2: 1-hop → ego ─────────────────────────────────
        h_ego_updated = self.layer_1hop(h_ego, h_nbr_updated, e1, nbr_mask)

        # ── STAGE 3: Classifier ────────────────────────────────────────────
        return self.classifier(h_ego_updated)                  # [B, num_classes]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
