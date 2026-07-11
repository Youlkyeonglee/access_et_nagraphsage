"""
QCNet-adapted — TSEM-SAGE 비교용 재구현
=========================================
원본: comparison/QCNet/ (Zikang Zhou, CVPR 2023, Apache-2.0)
  - layers/fourier_embedding.py    -> FourierEmbedding (수식 동일, 아래 재구현)
  - layers/attention_layer.py      -> AttentionLayer (수식 동일, 아래 재구현 — bipartite=False만 사용)
  - modules/qcnet_agent_encoder.py -> QCNetAgentEncoder (아래 QCNetTSEMAdapted._build_graph + forward)
  - modules/qcnet_decoder.py       -> 미사용(회귀 전용 DETR류 디코더) — 분류 헤드로 교체

이 환경(tna_research)에는 torch_geometric·torch_cluster가 없어 MessagePassing/radius_graph를
순수 PyTorch(scatter_reduce_/scatter_add_ + 직접 계산한 거리 필터)로 다시 구현했다.

HiVT-adapted와의 핵심 차이(설계 철학 자체가 다름, comparison/README.md 참조):
  - HiVT는 노드별 rotate_mat(회전행렬)을 명시적으로 적용해 방향 불변성을 얻지만, QCNet은
    **상대 각도(angle_between_2d_vectors)를 특징으로 직접 사용**해 회전 불변성을 얻는다 —
    회전행렬 bmm이 아예 필요 없다. 그래서 이 구현은 scene 중심 정렬·회전 전처리를 하지 않고
    원시 world 좌표를 그대로 쓴다(모든 입력이 상대 거리/각도 차이이기 때문에 절대 좌표계와 무관).
  - HiVT는 "1-hop 그래프(공간) + temporal transformer(시간)"를 분리 처리하지만, QCNet은
    **매 layer마다 시간축 self-attention(t_attn) → 공간축 self-attention(a2a_attn)을 번갈아
    반복**한다 — 매 timestep마다 별도의 공간 그래프를 만든다는 점도 다르다(HiVT는 anchor
    시점에서만 global interactor를 1회 적용).
  - map-agent(pl2a) attention은 원본에 있으나 맵 데이터가 없어 완전히 제거했다.

주요 변경점(README.md 참조):
  - pl2a(맵) attention 제거 — 맵 데이터 없음
  - x_a 입력 4D: 원본은 [motion 크기, motion-heading 각도, velocity 크기, velocity-heading 각도]인데
    velocity가 Argoverse 센서 고유 피처라 우리 데이터엔 없음(우리 방향벡터=heading 자체라 각도가
    항상 0으로 퇴화) — 대신 [motion 크기, motion-heading 각도, speed, accel]로 대체(개수는 4D 유지,
    speed·accel은 우리 데이터의 진짜 독립 신호).
  - agent-type 카테고리 임베딩 제거 — 우리 데이터는 전부 같은 타입(차량)이라 불필요.
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn


def weight_init(m: nn.Module) -> None:
    """comparison/QCNet/utils/weight_init.py 와 동일(이 모델에서 실제 쓰이는 서브모듈만)."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def angle_between_2d_vectors(ctr_vector: torch.Tensor, nbr_vector: torch.Tensor) -> torch.Tensor:
    """comparison/QCNet/utils/geometry.py::angle_between_2d_vectors 와 동일."""
    return torch.atan2(
        ctr_vector[..., 0] * nbr_vector[..., 1] - ctr_vector[..., 1] * nbr_vector[..., 0],
        (ctr_vector[..., :2] * nbr_vector[..., :2]).sum(dim=-1))


def wrap_angle(angle: torch.Tensor, min_val: float = -math.pi, max_val: float = math.pi) -> torch.Tensor:
    """comparison/QCNet/utils/geometry.py::wrap_angle 와 동일."""
    return min_val + (angle + max_val) % (max_val - min_val)


def segment_softmax(alpha: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """torch_geometric.utils.softmax(alpha, index) 재구현 — comparison/hivt_tsem/model.py와 동일 함수."""
    H = alpha.size(1)
    idx = index.unsqueeze(-1).expand(-1, H)
    node_max = torch.full((num_nodes, H), float('-inf'), device=alpha.device, dtype=alpha.dtype)
    node_max.scatter_reduce_(0, idx, alpha, reduce='amax', include_self=True)
    node_max = torch.nan_to_num(node_max, neginf=0.0)
    alpha = (alpha - node_max[index]).exp()
    denom = torch.zeros((num_nodes, H), device=alpha.device, dtype=alpha.dtype)
    denom.scatter_add_(0, idx, alpha)
    return alpha / denom[index].clamp(min=1e-12)


def scatter_sum_nodes(messages: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    D = messages.size(-1)
    out = torch.zeros((num_nodes, D), device=messages.device, dtype=messages.dtype)
    out.scatter_add_(0, index.unsqueeze(-1).expand(-1, D), messages)
    return out


class FourierEmbedding(nn.Module):
    """comparison/QCNet/layers/fourier_embedding.py 와 동일(연속형 입력만 사용, categorical 없음
    — 우리 데이터는 agent type이 전부 같아 categorical_embs가 필요 없다)."""

    def __init__(self, input_dim: int, hidden_dim: int, num_freq_bands: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.freqs = nn.Embedding(input_dim, num_freq_bands)
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_freq_bands * 2 + 1, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim))
            for _ in range(input_dim)])
        self.to_out = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim))
        self.apply(weight_init)

    def forward(self, continuous_inputs: torch.Tensor) -> torch.Tensor:
        x = continuous_inputs.unsqueeze(-1) * self.freqs.weight * 2 * math.pi  # [N, input_dim, num_freq_bands]
        x = torch.cat([x.cos(), x.sin(), continuous_inputs.unsqueeze(-1)], dim=-1)
        outs = [self.mlps[i](x[:, i]) for i in range(self.input_dim)]
        return self.to_out(torch.stack(outs).sum(dim=0))


class AttentionLayer(nn.Module):
    """comparison/QCNet/layers/attention_layer.py 대응 — bipartite=False(자기-자신 그래프)만 지원
    (pl2a의 bipartite=True 케이스는 맵 제거로 불필요)."""

    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int, dropout: float, has_pos_emb: bool):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.has_pos_emb = has_pos_emb
        self.scale = head_dim ** -0.5

        self.to_q = nn.Linear(hidden_dim, head_dim * num_heads)
        self.to_k = nn.Linear(hidden_dim, head_dim * num_heads, bias=False)
        self.to_v = nn.Linear(hidden_dim, head_dim * num_heads)
        if has_pos_emb:
            self.to_k_r = nn.Linear(hidden_dim, head_dim * num_heads, bias=False)
            self.to_v_r = nn.Linear(hidden_dim, head_dim * num_heads)
        self.to_s = nn.Linear(hidden_dim, head_dim * num_heads)
        self.to_g = nn.Linear(head_dim * num_heads + hidden_dim, head_dim * num_heads)
        self.to_out = nn.Linear(head_dim * num_heads, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.ff_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim))
        self.attn_prenorm_x = nn.LayerNorm(hidden_dim)
        if has_pos_emb:
            self.attn_prenorm_r = nn.LayerNorm(hidden_dim)
        self.attn_postnorm = nn.LayerNorm(hidden_dim)
        self.ff_prenorm = nn.LayerNorm(hidden_dim)
        self.ff_postnorm = nn.LayerNorm(hidden_dim)
        self.apply(weight_init)

    def forward(self, x: torch.Tensor, r: Optional[torch.Tensor], edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        x_norm = self.attn_prenorm_x(x)
        r_norm = self.attn_prenorm_r(r) if (self.has_pos_emb and r is not None) else r
        x = x + self.attn_postnorm(self._attn_block(x_norm, r_norm, edge_index, num_nodes))
        x = x + self.ff_postnorm(self.ff_mlp(self.ff_prenorm(x)))
        return x

    def _attn_block(self, x_norm: torch.Tensor, r: Optional[torch.Tensor],
                    edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        src, tgt = edge_index[0], edge_index[1]
        H, d = self.num_heads, self.head_dim
        q_all = self.to_q(x_norm).view(-1, H, d)
        k_all = self.to_k(x_norm).view(-1, H, d)
        v_all = self.to_v(x_norm).view(-1, H, d)
        q_i, k_j, v_j = q_all[tgt], k_all[src], v_all[src]
        if self.has_pos_emb:
            k_j = k_j + self.to_k_r(r).view(-1, H, d)
            v_j = v_j + self.to_v_r(r).view(-1, H, d)
        sim = (q_i * k_j).sum(dim=-1) * self.scale
        attn = segment_softmax(sim, tgt, num_nodes)
        attn = self.attn_drop(attn)
        msg = (v_j * attn.unsqueeze(-1)).reshape(-1, H * d)
        agg = scatter_sum_nodes(msg, tgt, num_nodes)

        g = torch.sigmoid(self.to_g(torch.cat([agg, x_norm], dim=-1)))
        gated = agg + g * (self.to_s(x_norm) - agg)
        return self.to_out(gated)


class QCNetTSEMAdapted(nn.Module):
    """QCNet 어댑터 최상위 모듈 — batch dict(TSEM dataloader 그대로) -> logits[B,num_classes].

    그래프: 샘플당 1+K개 노드(ego+1-hop) × T 시점 = (1+K)*T개 (node,time) 슬롯. 두 종류의
    edge_index를 매 layer 공통으로 재사용한다:
      - edge_index_t   : 같은 노드의 서로 다른 시점끼리(시간축 self-attention)
      - edge_index_a2a : 같은 시점의 서로 다른 노드끼리(공간축 self-attention, a2a_radius로 거리 제한)
    """

    def __init__(self, W: int = 10, K: int = 6, hidden_dim: int = 64, num_heads: int = 8,
                 head_dim: int = 8, num_layers: int = 3, num_freq_bands: int = 32,
                 time_span: Optional[int] = None, a2a_radius: float = 20.0, dropout: float = 0.1,
                 num_classes: int = 3):
        super().__init__()
        self.W = W
        self.K = K
        self.N = 1 + K
        self.hidden_dim = hidden_dim
        self.time_span = time_span if time_span is not None else (W - 1)
        self.a2a_radius = a2a_radius

        self.x_a_emb = FourierEmbedding(input_dim=4, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_t_emb = FourierEmbedding(input_dim=4, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_a2a_emb = FourierEmbedding(input_dim=3, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.t_attn_layers = nn.ModuleList([
            AttentionLayer(hidden_dim, num_heads, head_dim, dropout, has_pos_emb=True)
            for _ in range(num_layers)])
        self.a2a_attn_layers = nn.ModuleList([
            AttentionLayer(hidden_dim, num_heads, head_dim, dropout, has_pos_emb=True)
            for _ in range(num_layers)])
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # 슬롯 id: (node n, time t) -> n*T + t (샘플 내 고정 크기 N*T, 배치 시 offset만 더함)
        NT = self.N * self.W
        n_idx, t_idx = torch.meshgrid(torch.arange(self.N), torch.arange(self.W), indexing='ij')
        slot = (n_idx * self.W + t_idx).reshape(-1)  # [N*T], slot[n*T+t]=n*T+t (자기 자신 참조용 인덱스)

        # 시간축 템플릿: 같은 n, 서로 다른 t1!=t2, |t1-t2|<=time_span
        src_t, tgt_t = [], []
        for n in range(self.N):
            for t1 in range(self.W):
                for t2 in range(self.W):
                    if t1 != t2 and abs(t1 - t2) <= self.time_span:
                        src_t.append(n * self.W + t1)
                        tgt_t.append(n * self.W + t2)
        self.register_buffer('template_edge_t', torch.tensor([src_t, tgt_t], dtype=torch.long))

        # 공간축 템플릿: 같은 t, 서로 다른 n1!=n2 (거리 필터는 forward에서 동적으로 적용)
        src_a, tgt_a = [], []
        for t in range(self.W):
            for n1 in range(self.N):
                for n2 in range(self.N):
                    if n1 != n2:
                        src_a.append(n1 * self.W + t)
                        tgt_a.append(n2 * self.W + t)
        self.register_buffer('template_edge_a2a', torch.tensor([src_a, tgt_a], dtype=torch.long))
        self.register_buffer('_NT', torch.tensor(NT))

    def _build_flat(self, batch: dict):
        """batch -> (x_a[NTtot,4], pos[NTtot,2], head_vec[NTtot,2], head_angle[NTtot], t_val[NTtot],
        valid[NTtot], edge_index_t[2,Et], edge_index_a2a[2,Ea])"""
        device = batch['node_seq'].device
        B = batch['node_seq'].size(0)
        N, T = self.N, self.W

        raw = torch.cat([batch['node_seq'].unsqueeze(1), batch['nbr_node_seqs']], dim=1)  # [B,N,T,6]
        present = raw.abs().sum(dim=-1) > 0  # [B,N,T]
        raw_flat = raw.reshape(-1, 6)  # [B*N*T, 6]
        valid = present.reshape(-1)

        pos = raw_flat[:, 0:2]
        speed = raw_flat[:, 2]
        head_vec = raw_flat[:, 3:5]  # (dir_x, dir_z) — heading 단위벡터
        accel = raw_flat[:, 5]

        # motion vector: 같은 노드의 t-1 -> t 변위 (t=0은 0)
        raw_bt = raw.reshape(B * N, T, 6)
        motion = torch.zeros(B * N, T, 2, device=device, dtype=raw.dtype)
        both_valid = present.reshape(B * N, T)
        both_valid = both_valid[:, 1:] & both_valid[:, :-1]
        motion[:, 1:] = torch.where(both_valid.unsqueeze(-1),
                                     raw_bt[:, 1:, 0:2] - raw_bt[:, :-1, 0:2],
                                     torch.zeros_like(raw_bt[:, 1:, 0:2]))
        motion = motion.reshape(-1, 2)

        motion_norm = torch.norm(motion, p=2, dim=-1)
        motion_angle = angle_between_2d_vectors(head_vec, motion)
        x_a_feat = torch.stack([motion_norm, motion_angle, speed, accel], dim=-1)  # [NTtot,4]
        x_a_feat = torch.where(valid.unsqueeze(-1), x_a_feat, torch.zeros_like(x_a_feat))

        head_angle = torch.atan2(head_vec[:, 1], head_vec[:, 0])
        head_angle = torch.where(valid, head_angle, torch.zeros_like(head_angle))
        t_val = torch.arange(T, device=device).repeat(B * N).float()

        b_idx = torch.arange(B, device=device) * (N * T)
        edge_t = (self.template_edge_t.unsqueeze(0) + b_idx.view(-1, 1, 1)).permute(1, 0, 2).reshape(2, -1)
        edge_a2a = (self.template_edge_a2a.unsqueeze(0) + b_idx.view(-1, 1, 1)).permute(1, 0, 2).reshape(2, -1)

        keep_t = valid[edge_t[0]] & valid[edge_t[1]]
        edge_t = edge_t[:, keep_t]
        keep_a2a = valid[edge_a2a[0]] & valid[edge_a2a[1]]
        edge_a2a = edge_a2a[:, keep_a2a]
        # a2a_radius 거리 필터 (원본 radius_graph에 대응)
        dist_a2a = torch.norm(pos[edge_a2a[0]] - pos[edge_a2a[1]], p=2, dim=-1)
        edge_a2a = edge_a2a[:, dist_a2a < self.a2a_radius]

        return x_a_feat, pos, head_vec, head_angle, t_val, valid, edge_t, edge_a2a

    @staticmethod
    def _rel_feat_t(pos, head_vec, head_angle, t_val, edge_index):
        src, tgt = edge_index[0], edge_index[1]
        rel_pos = pos[src] - pos[tgt]
        dist = torch.norm(rel_pos, p=2, dim=-1)
        angle = angle_between_2d_vectors(head_vec[tgt], rel_pos)
        rel_head = wrap_angle(head_angle[src] - head_angle[tgt])
        delta_t = t_val[src] - t_val[tgt]
        return torch.stack([dist, angle, rel_head, delta_t], dim=-1)

    @staticmethod
    def _rel_feat_a2a(pos, head_vec, head_angle, edge_index):
        src, tgt = edge_index[0], edge_index[1]
        rel_pos = pos[src] - pos[tgt]
        dist = torch.norm(rel_pos, p=2, dim=-1)
        angle = angle_between_2d_vectors(head_vec[tgt], rel_pos)
        rel_head = wrap_angle(head_angle[src] - head_angle[tgt])
        return torch.stack([dist, angle, rel_head], dim=-1)

    def forward(self, batch: dict) -> torch.Tensor:
        x_a_feat, pos, head_vec, head_angle, t_val, valid, edge_t, edge_a2a = self._build_flat(batch)
        NTtot = x_a_feat.size(0)
        device = x_a_feat.device

        x_a = self.x_a_emb(x_a_feat)  # [NTtot, D]
        r_t = self.r_t_emb(self._rel_feat_t(pos, head_vec, head_angle, t_val, edge_t)) if edge_t.numel() else None
        r_a2a = self.r_a2a_emb(self._rel_feat_a2a(pos, head_vec, head_angle, edge_a2a)) if edge_a2a.numel() else None

        for i in range(len(self.t_attn_layers)):
            if edge_t.numel():
                x_a = self.t_attn_layers[i](x_a, r_t, edge_t)
            if edge_a2a.numel():
                x_a = self.a2a_attn_layers[i](x_a, r_a2a, edge_a2a)

        B = batch['node_seq'].size(0)
        ego_idx = torch.arange(B, device=device) * (self.N * self.W) + (self.W - 1)  # (n=0, t=T-1)
        ego_embed = x_a[ego_idx]
        return self.classifier(ego_embed)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
