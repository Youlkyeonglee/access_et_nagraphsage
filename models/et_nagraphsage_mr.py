"""
ET-NAGraphSAGE-MR — Multi-Relational edge 확장 (Step 1)
========================================================
기존 et_nagraphsage.py는 불변. 이 파일은 그 복사본에 **C2(메시지 함수)만** 다중관계로
수정한 버전이다. GraphSAGE 약점("모든 엣지를 같은 타입으로 취급")을 R-GCN식으로 보완.

■ 무엇을 바꿨나 (기존 대비 유일한 구조 변경)
  - 기존 ETSAGELayer.C2:  m = ReLU( Linear(cat(h_nbr_gated, e)) )      ← 단일 메시지 weight
  - 신규 ETSAGELayerMR.C2: 관계타입 r∈{ahead,behind,left,right} 별로 **별도 메시지 weight**
        m = ReLU( Linear_r(cat(h_nbr_gated, e)) )   (엣지의 기하 관계타입 r로 선택)
  - 관계타입 r은 **마지막 프레임(t) ego heading 기준 상대위치**로 결정:
        lon = rel·heading, lat = rel·perp
        ahead(0): |lat|<=w & lon>=0 | behind(1): |lat|<=w & lon<0
        left(2):  lat<-w            | right(3): lat>=w
  - C1(게이트)·C3(β어텐션)·업데이트·temporal encoder는 **기존과 100% 동일**.

■ 주의: 관계타입은 raw 좌표 기하에 의존 → 공업탑 base(비표준화)에서 정확.
        (DRIFT는 ego_relative+z-score라 좌표가 왜곡되므로 이 in-model 타이핑엔 부적합.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .temporal_encoder import TemporalEncoder


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    scores = scores.masked_fill(mask == 0, -1e4)
    return F.softmax(scores, dim=-1)


def relation_type(center: torch.Tensor, nbr: torch.Tensor,
                  lat_w: float = 2.5, num_rel: int = 4) -> torch.Tensor:
    """center[...,6], nbr[...,K,6] (마지막프레임 노드피처) → 관계타입 [...,K] (long).
    노드피처 = [pos_x, pos_z, speed, dir_x, dir_z, accel]. heading=dir.
      0 ahead | 1 behind | 2 left | 3 right  (num_rel=4 기준)
    num_rel=2면 ahead/behind만(좌우를 앞뒤로 흡수: lon 부호). num_rel=6이면 앞/뒤 각각 좌/우 세분.
    """
    ego_pos = center[..., :2].unsqueeze(-2)                 # [...,1,2]
    ego_dir = center[..., 3:5]                              # [...,2]
    h = ego_dir / (ego_dir.norm(dim=-1, keepdim=True) + 1e-6)
    perp = torch.stack([-h[..., 1], h[..., 0]], dim=-1)     # [...,2] (좌수직)
    rel = nbr[..., :2] - ego_pos                            # [...,K,2]
    lon = (rel * h.unsqueeze(-2)).sum(-1)                   # [...,K] 전방(+)/후방(-)
    lat = (rel * perp.unsqueeze(-2)).sum(-1)                # [...,K] 좌(-)/우(+)
    same = lat.abs() <= lat_w
    t = torch.zeros_like(lon, dtype=torch.long)
    if num_rel <= 2:
        # 앞/뒤만
        t = torch.where(lon >= 0, torch.zeros_like(t), torch.ones_like(t))
    else:
        t = torch.where(same & (lon >= 0), torch.zeros_like(t), t)   # 0 ahead
        t = torch.where(same & (lon < 0),  torch.ones_like(t),  t)   # 1 behind
        t = torch.where((~same) & (lat < 0), torch.full_like(t, 2), t)  # 2 left
        t = torch.where((~same) & (lat >= 0), torch.full_like(t, 3), t) # 3 right
    return t.clamp(0, num_rel - 1)


class ETSAGELayerMR(nn.Module):
    """C2만 다중관계(R-GCN식)로 확장한 ET-SAGE 레이어. 나머지는 기존과 동일."""

    def __init__(self, in_dim: int, out_dim: int, d_e: int,
                 num_relations: int = 4, dropout: float = 0.3):
        super().__init__()
        self.R = num_relations
        self.out_dim = out_dim
        self.lin_gate   = nn.Linear(d_e, in_dim)                       # C1 (동일)
        self.lin_msg    = nn.Linear(in_dim + d_e, num_relations * out_dim)  # C2 (관계별)
        self.lin_beta   = nn.Linear(d_e, 1)                           # C3 (동일)
        self.lin_update = nn.Linear(in_dim + out_dim, out_dim)
        self.bn         = nn.BatchNorm1d(out_dim)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, h_ego, h_nbr, e_temp, nbr_mask, rel_type):
        B, K, _ = h_nbr.shape
        # C1 게이트 (기존 동일)
        alpha = torch.sigmoid(self.lin_gate(e_temp))
        h_nbr_gated = alpha * h_nbr

        # C2 관계별 메시지: R개 메시지 계산 후 엣지 관계타입으로 선택
        msg_all = self.lin_msg(torch.cat([h_nbr_gated, e_temp], dim=-1))  # [B,K,R*out]
        msg_all = msg_all.view(B, K, self.R, self.out_dim)               # [B,K,R,out]
        idx = rel_type.clamp(0, self.R - 1).unsqueeze(-1).unsqueeze(-1).expand(B, K, 1, self.out_dim)
        msg = msg_all.gather(2, idx).squeeze(2)                          # [B,K,out]
        msg = F.relu(msg)
        msg = msg * nbr_mask.unsqueeze(-1)

        # C3 β 집계 (기존 동일)
        beta = _masked_softmax(self.lin_beta(e_temp).squeeze(-1), nbr_mask)
        h_N = (beta.unsqueeze(-1) * msg).sum(dim=1)
        no_nbr = (nbr_mask.sum(dim=1, keepdim=True) == 0).float()
        h_N = h_N * (1.0 - no_nbr)

        # 업데이트 (기존 동일)
        h_new = F.relu(self.lin_update(torch.cat([h_ego, h_N], dim=-1)))
        h_new = self.bn(h_new)
        h_new = self.dropout(h_new)
        return h_new


class ETNAGraphSAGEMR(nn.Module):
    """ET-NAGraphSAGE with multi-relational edges (Step 1). 인터페이스는 기존과 동일."""

    def __init__(self, node_dim=6, edge_dim=5, hidden_dim=128, d_e=32, T=10,
                 encoder_type='gru', use_attention=True, use_2hop=True,
                 num_classes=3, dropout=0.3, temporal_target='both',
                 num_relations=4, lat_w=2.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.d_e = d_e
        self.T = T
        self.use_2hop = use_2hop
        self.num_relations = num_relations
        self.lat_w = lat_w

        assert temporal_target in ('both', 'node', 'edge')
        self.temporal_target = temporal_target
        self.temporal_node = temporal_target in ('both', 'node')
        self.temporal_edge = temporal_target in ('both', 'edge')

        enc_kwargs = dict(encoder_type=encoder_type, use_attention=use_attention)
        self.node_encoder = TemporalEncoder(node_dim, hidden_dim, **enc_kwargs)
        self.edge_encoder = TemporalEncoder(edge_dim, d_e, **enc_kwargs)

        if use_2hop:
            self.layer_2hop = ETSAGELayerMR(hidden_dim, hidden_dim, d_e, num_relations, dropout)
        self.layer_1hop = ETSAGELayerMR(hidden_dim, hidden_dim, d_e, num_relations, dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch: dict) -> torch.Tensor:
        node_seq       = batch['node_seq']
        nbr_node_seqs  = batch['nbr_node_seqs']
        edge_seqs      = batch['edge_seqs']
        nbr_mask       = batch['nbr_mask']
        nbr2_node_seqs = batch['nbr2_node_seqs']
        nbr2_edge_seqs = batch['nbr2_edge_seqs']
        nbr2_mask      = batch['nbr2_mask']

        B, K1, T, _ = nbr_node_seqs.shape

        # ── 관계타입 계산 (마지막 프레임 = frame t, raw 노드피처 기준) ──
        rel1 = relation_type(node_seq[:, -1, :], nbr_node_seqs[:, :, -1, :],
                             self.lat_w, self.num_relations)            # [B,K1]
        if self.use_2hop and nbr2_node_seqs.shape[2] > 0:
            K2 = nbr2_node_seqs.shape[2]
            rel2 = relation_type(nbr_node_seqs[:, :, -1, :],
                                 nbr2_node_seqs[:, :, :, -1, :],
                                 self.lat_w, self.num_relations)        # [B,K1,K2]
        else:
            K2 = 0

        # ── Ablation C 슬라이싱 (기존 동일) ──
        if not self.temporal_node:
            node_seq       = node_seq[:, -1:, :]
            nbr_node_seqs  = nbr_node_seqs[:, :, -1:, :]
            nbr2_node_seqs = nbr2_node_seqs[:, :, :, -1:, :]
        if not self.temporal_edge:
            edge_seqs      = edge_seqs[:, :, -1:, :]
            nbr2_edge_seqs = nbr2_edge_seqs[:, :, :, -1:, :]
        Tn = node_seq.shape[1]; Te = edge_seqs.shape[2]

        # ── STAGE 1: Temporal Encoding (기존 동일) ──
        h_ego = self.node_encoder(node_seq)
        h_nbr = self.node_encoder(nbr_node_seqs.view(B * K1, Tn, -1)).view(B, K1, -1)
        h_nbr = h_nbr * nbr_mask.unsqueeze(-1)
        e1 = self.edge_encoder(edge_seqs.view(B * K1, Te, -1)).view(B, K1, -1)
        e1 = e1 * nbr_mask.unsqueeze(-1)

        # ── STAGE 2 Layer1: 2-hop→1-hop (MR) ──
        if self.use_2hop and K2 > 0:
            h_nbr2 = self.node_encoder(
                nbr2_node_seqs.view(B * K1 * K2, Tn, -1)).view(B, K1, K2, -1)
            h_nbr2 = h_nbr2 * nbr2_mask.unsqueeze(-1)
            e2 = self.edge_encoder(
                nbr2_edge_seqs.view(B * K1 * K2, Te, -1)).view(B, K1, K2, -1)
            e2 = e2 * nbr2_mask.unsqueeze(-1)
            h_nbr_updated = self.layer_2hop(
                h_ego=h_nbr.view(B * K1, -1),
                h_nbr=h_nbr2.view(B * K1, K2, -1),
                e_temp=e2.view(B * K1, K2, -1),
                nbr_mask=nbr2_mask.view(B * K1, K2),
                rel_type=rel2.reshape(B * K1, K2),
            ).view(B, K1, -1)
            h_nbr_updated = h_nbr_updated * nbr_mask.unsqueeze(-1)
        else:
            h_nbr_updated = h_nbr

        # ── STAGE 2 Layer2: 1-hop→ego (MR) ──
        h_ego_updated = self.layer_1hop(h_ego, h_nbr_updated, e1, nbr_mask, rel1)
        return self.classifier(h_ego_updated)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
