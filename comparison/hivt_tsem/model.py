"""
HiVT-adapted — TSEM-SAGE 비교용 재구현
=========================================
원본: comparison/HiVT/ (Zikang Zhou, CVPR 2022, Apache-2.0)
  - models/embedding.py       -> SingleInputEmbedding, MultipleInputEmbedding (수식 동일, 아래 재구현)
  - models/local_encoder.py   -> AAEncoder, TemporalEncoder(+Layer)          (아래 TSEMAAEncoder/TemporalEncoder)
  - models/global_interactor.py -> GlobalInteractor(+Layer)                  (아래 TSEMGlobalInteractorLayer)

이 환경(tna_research)에는 torch_geometric이 없어 MessagePassing/softmax(그룹별)를
순수 PyTorch(scatter_reduce_/scatter_add_)로 다시 구현했다 — attention 수식·게이트
업데이트·MLP 구조는 원본과 동일, 그래프 배치 방식만 다르다(원본: PyG Batch/Data,
여기: 고정 크기(1+K) ego-이웃 그래프를 배치 차원에 맞춰 offset한 flat edge_index).

주요 변경점(README.md 참조):
  - ALEncoder(차선 융합) 제거 — 맵 데이터 없음
  - node_dim 2 -> 6 (pos_disp 2 + speed 1 + dir 2 + accel 1), 회전은 pos/dir 쌍에만 적용
  - num_modes(다중 미래 가설) 제거 — 회귀 전용 개념
  - 2-hop 이웃 미사용, edge_dim=2(rel_pos)만 사용 — 원본 설계 그대로
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m: nn.Module) -> None:
    """comparison/HiVT/utils.py::init_weights 와 동일(Linear/LayerNorm 부분만 — 이 모델에서
    실제로 쓰이는 서브모듈만 포함)."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            bound = (6.0 / (2 * m.embed_dim)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)


def segment_softmax(alpha: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """torch_geometric.utils.softmax(alpha, index) 재구현 — index로 묶인 그룹별 softmax.
    alpha: [E, H], index(target node id): [E] -> [E, H]"""
    H = alpha.size(1)
    idx = index.unsqueeze(-1).expand(-1, H)
    node_max = torch.full((num_nodes, H), float('-inf'), device=alpha.device, dtype=alpha.dtype)
    node_max.scatter_reduce_(0, idx, alpha, reduce='amax', include_self=True)
    node_max = torch.nan_to_num(node_max, neginf=0.0)  # 고립 노드(들어오는 edge 없음) 대비
    alpha = (alpha - node_max[index]).exp()
    denom = torch.zeros((num_nodes, H), device=alpha.device, dtype=alpha.dtype)
    denom.scatter_add_(0, idx, alpha)
    return alpha / denom[index].clamp(min=1e-12)


def scatter_sum_nodes(messages: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """messages: [E, D] -> [num_nodes, D], index(target)로 scatter-add."""
    D = messages.size(-1)
    out = torch.zeros((num_nodes, D), device=messages.device, dtype=messages.dtype)
    out.scatter_add_(0, index.unsqueeze(-1).expand(-1, D), messages)
    return out


def rotate2(vec: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """vec: [N,2], rot: [N,2,2] -> [N,2]  (원본의 torch.bmm(x.unsqueeze(-2), rot).squeeze(-2))"""
    return torch.bmm(vec.unsqueeze(-2), rot).squeeze(-2)


def rotate6(x: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """x: [N,6] = [dx,dz,speed,dir_x,dir_z,accel], rot: [N,2,2].
    위치변위쌍[0:2]·방향쌍[3:5]만 회전, speed[2]/accel[5]는 스칼라라 그대로 통과."""
    pos = rotate2(x[:, 0:2], rot)
    dirn = rotate2(x[:, 3:5], rot)
    return torch.cat([pos[:, 0:1], pos[:, 1:2], x[:, 2:3], dirn[:, 0:1], dirn[:, 1:2], x[:, 5:6]], dim=-1)


class SingleInputEmbedding(nn.Module):
    """comparison/HiVT/models/embedding.py::SingleInputEmbedding 와 동일."""

    def __init__(self, in_channel: int, out_channel: int) -> None:
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_channel, out_channel), nn.LayerNorm(out_channel), nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel), nn.LayerNorm(out_channel), nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel), nn.LayerNorm(out_channel))
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


class MultipleInputEmbedding(nn.Module):
    """comparison/HiVT/models/embedding.py::MultipleInputEmbedding 와 동일."""

    def __init__(self, in_channels: list, out_channel: int) -> None:
        super().__init__()
        self.module_list = nn.ModuleList([
            nn.Sequential(nn.Linear(c, out_channel), nn.LayerNorm(out_channel), nn.ReLU(inplace=True),
                          nn.Linear(out_channel, out_channel))
            for c in in_channels])
        self.aggr_embed = nn.Sequential(
            nn.LayerNorm(out_channel), nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel), nn.LayerNorm(out_channel))
        self.apply(init_weights)

    def forward(self, continuous_inputs: list) -> torch.Tensor:
        outs = [m(v) for m, v in zip(self.module_list, continuous_inputs)]
        return self.aggr_embed(torch.stack(outs).sum(dim=0))


class TSEMAAEncoder(nn.Module):
    """원본 AAEncoder(local_encoder.py) 대응 — agent-agent attention, 매 timestep 독립 호출.
    node_dim=6(회전은 pos/dir 쌍에만), edge_dim=2(rel_pos)."""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.center_embed = SingleInputEmbedding(in_channel=6, out_channel=embed_dim)
        self.nbr_embed = MultipleInputEmbedding(in_channels=[6, 2], out_channel=embed_dim)
        self.lin_q = nn.Linear(embed_dim, embed_dim)
        self.lin_k = nn.Linear(embed_dim, embed_dim)
        self.lin_v = nn.Linear(embed_dim, embed_dim)
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout))
        self.bos_token = nn.Parameter(torch.Tensor(embed_dim))
        nn.init.normal_(self.bos_token, mean=0., std=.02)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                bos_mask: torch.Tensor, rotate_mat: torch.Tensor) -> torch.Tensor:
        """x:[N,6] rot-invariant 미적용 원시 입력, edge_index:[2,E](row=src j, col=tgt i),
        edge_attr:[E,2] rel_pos(=pos_j-pos_i, 원본과 동일 부호), bos_mask:[N] bool,
        rotate_mat:[N,2,2] 노드별(수신자 기준) 회전행렬."""
        center_embed = self.center_embed(rotate6(x, rotate_mat))
        center_embed = torch.where(bos_mask.unsqueeze(-1), self.bos_token, center_embed)
        center_embed = center_embed + self._mha_block(
            self.norm1(center_embed), x, edge_index, edge_attr, rotate_mat)
        center_embed = center_embed + self.mlp(self.norm2(center_embed))
        return center_embed

    def _mha_block(self, center_embed: torch.Tensor, x: torch.Tensor, edge_index: torch.Tensor,
                   edge_attr: torch.Tensor, rotate_mat: torch.Tensor) -> torch.Tensor:
        src, tgt = edge_index[0], edge_index[1]
        num_nodes = x.size(0)
        tgt_rot = rotate_mat[tgt]  # 수신자(tgt) 기준 회전
        x_j = rotate6(x[src], tgt_rot)
        edge_attr_rot = rotate2(edge_attr, tgt_rot)
        nbr_embed = self.nbr_embed([x_j, edge_attr_rot])

        query = self.lin_q(center_embed[tgt]).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key = self.lin_k(nbr_embed).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value = self.lin_v(nbr_embed).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = (query * key).sum(dim=-1) / scale  # [E, H]
        alpha = segment_softmax(alpha, tgt, num_nodes)
        alpha = self.attn_drop(alpha)
        msg = (value * alpha.unsqueeze(-1)).reshape(-1, self.embed_dim)  # [E, D]
        agg = scatter_sum_nodes(msg, tgt, num_nodes)  # [N, D]

        gate = torch.sigmoid(self.lin_ih(agg) + self.lin_hh(center_embed))
        out = agg + gate * (self.lin_self(center_embed) - agg)
        return self.proj_drop(self.out_proj(out))


class TemporalEncoderLayer(nn.Module):
    """comparison/HiVT/models/local_encoder.py::TemporalEncoderLayer 와 동일."""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = src
        x = x + self.dropout1(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x),
                                              attn_mask=attn_mask, need_weights=False)[0])
        x = x + self.dropout2(self.linear2(self.dropout(F.relu_(self.linear1(self.norm2(x))))))
        return x


class TemporalEncoder(nn.Module):
    """comparison/HiVT/models/local_encoder.py::TemporalEncoder 와 동일(historical_steps=W)."""

    def __init__(self, historical_steps: int, embed_dim: int, num_heads: int = 8,
                 num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [TemporalEncoderLayer(embed_dim, num_heads, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.padding_token = nn.Parameter(torch.Tensor(historical_steps, 1, embed_dim))
        self.cls_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.Tensor(historical_steps + 1, 1, embed_dim))
        attn_mask = self._causal_mask(historical_steps + 1)
        self.register_buffer('attn_mask', attn_mask)
        nn.init.normal_(self.padding_token, mean=0., std=.02)
        nn.init.normal_(self.cls_token, mean=0., std=.02)
        nn.init.normal_(self.pos_embed, mean=0., std=.02)
        self.apply(init_weights)

    @staticmethod
    def _causal_mask(seq_len: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """x:[T,N,D], padding_mask:[N,T] (True=missing)"""
        x = torch.where(padding_mask.t().unsqueeze(-1), self.padding_token, x)
        cls = self.cls_token.expand(-1, x.shape[1], -1)
        x = torch.cat((x, cls), dim=0) + self.pos_embed
        for layer in self.layers:
            x = layer(x, attn_mask=self.attn_mask)
        return self.norm(x)[-1]  # [N, D]


class TSEMGlobalInteractorLayer(nn.Module):
    """원본 GlobalInteractorLayer(global_interactor.py) 대응 — 전역(1-hop 그래프 전체) 상호작용."""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.lin_q_node = nn.Linear(embed_dim, embed_dim)
        self.lin_k_node = nn.Linear(embed_dim, embed_dim)
        self.lin_k_edge = nn.Linear(embed_dim, embed_dim)
        self.lin_v_node = nn.Linear(embed_dim, embed_dim)
        self.lin_v_edge = nn.Linear(embed_dim, embed_dim)
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout))
        self.apply(init_weights)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        x = x + self._mha_block(self.norm1(x), edge_index, edge_attr)
        x = x + self.mlp(self.norm2(x))
        return x

    def _mha_block(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        src, tgt = edge_index[0], edge_index[1]
        num_nodes = x.size(0)
        query = self.lin_q_node(x[tgt]).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key_node = self.lin_k_node(x[src]).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key_edge = self.lin_k_edge(edge_attr).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value_node = self.lin_v_node(x[src]).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value_edge = self.lin_v_edge(edge_attr).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = (query * (key_node + key_edge)).sum(dim=-1) / scale
        alpha = segment_softmax(alpha, tgt, num_nodes)
        alpha = self.attn_drop(alpha)
        msg = ((value_node + value_edge) * alpha.unsqueeze(-1)).reshape(-1, self.embed_dim)
        agg = scatter_sum_nodes(msg, tgt, num_nodes)
        gate = torch.sigmoid(self.lin_ih(agg) + self.lin_hh(x))
        out = agg + gate * (self.lin_self(x) - agg)
        return self.proj_drop(self.out_proj(out))


class HiVTTSEMAdapted(nn.Module):
    """HiVT 어댑터 최상위 모듈 — batch dict(TSEM dataloader 그대로) -> logits[B,num_classes].

    그래프 구성: 샘플당 정확히 1+K개 노드(ego=0, 1-hop 이웃 슬롯 K개, 무효 슬롯도 패딩 노드로
    유지) — 크기가 고정이라 배치 전체를 하나의 flat 그래프(offset만 다른 반복 topology)로
    벡터화해서 구성한다(파이썬 루프로 PyG Data를 만들지 않음).
    """

    def __init__(self, W: int = 10, K: int = 6, embed_dim: int = 64, num_heads: int = 8,
                 num_temporal_layers: int = 4, num_global_layers: int = 3, dropout: float = 0.1,
                 num_classes: int = 3):
        super().__init__()
        self.W = W
        self.K = K
        self.N = 1 + K  # ego + K neighbors
        self.embed_dim = embed_dim

        self.aa_encoder = TSEMAAEncoder(embed_dim, num_heads, dropout)
        self.temporal_encoder = TemporalEncoder(W, embed_dim, num_heads, num_temporal_layers, dropout)
        # GlobalInteractor.rel_embed(원본) — rel_pos(2D)와 (cos,sin) 상대헤딩(2D) 두 스트림을
        # 합쳐 embed_dim으로 투영. MultipleInputEmbedding은 리스트로 별도 스트림을 받는다.
        self.rel_embed = MultipleInputEmbedding(in_channels=[2, 2], out_channel=embed_dim)
        self.global_layers = nn.ModuleList(
            [TSEMGlobalInteractorLayer(embed_dim, num_heads, dropout) for _ in range(num_global_layers)])
        self.global_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

        # 그래프 topology(1+K개 노드의 완전연결, 자기자신 제외)는 배치 전체에서 고정 —
        # 템플릿을 한 번만 만들어 두고 배치마다 offset만 더한다.
        idx = torch.arange(self.N)
        src, tgt = torch.meshgrid(idx, idx, indexing='ij')
        mask = src != tgt
        template_edge_index = torch.stack([src[mask], tgt[mask]], dim=0)  # [2, N*(N-1)]
        self.register_buffer('template_edge_index', template_edge_index)

    def _build_graph(self, batch: dict):
        """batch(TSEMFutureStateDataset collate 결과) -> (x[T,Ntot,6], positions[Ntot,T,2],
        rotate_angles[Ntot], padding_mask[Ntot,T], bos_mask[Ntot,T], edge_index[2,E_tot])"""
        device = batch['node_seq'].device
        B = batch['node_seq'].size(0)
        K, N, T = self.K, self.N, self.W

        # [B,N,T,6] — ego(slot0) + 1-hop 이웃(slot1..K)
        raw = torch.cat([batch['node_seq'].unsqueeze(1), batch['nbr_node_seqs']], dim=1)  # [B,N,T,6]
        present = raw.abs().sum(dim=-1) > 0  # [B,N,T] — 결측 프레임(all-zero)이면 False
        # ego는 anchor 프레임에 항상 존재(데이터셋 구성상 보장); 이웃 슬롯이 nbr_mask=0이면
        # present도 전부 False가 되어 자연히 padding 처리된다.
        padding_mask = ~present  # [B,N,T]

        Ntot = B * N
        raw_flat = raw.reshape(Ntot, T, 6)
        padding_flat = padding_mask.reshape(Ntot, T)

        # --- 씬(=샘플) 단위 원점/헤딩 정렬: ego의 anchor(t=T-1) 위치·방향 기준 ---
        ego_pos_anchor = batch['node_seq'][:, -1, 0:2]  # [B,2]
        ego_dir_anchor = batch['node_seq'][:, -1, 3:5]  # [B,2] (dir_x, dir_z)
        theta = torch.atan2(ego_dir_anchor[:, 1], ego_dir_anchor[:, 0])
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        scene_rot = torch.stack([torch.stack([cos_t, -sin_t], dim=-1),
                                  torch.stack([sin_t, cos_t], dim=-1)], dim=-2)  # [B,2,2]
        scene_rot_flat = scene_rot.repeat_interleave(N, dim=0)  # [Ntot,2,2]
        origin_flat = ego_pos_anchor.repeat_interleave(N, dim=0)  # [Ntot,2]

        pos_raw = raw_flat[..., 0:2]  # [Ntot,T,2]
        dir_raw = raw_flat[..., 3:5]
        pos_centered = pos_raw - origin_flat.unsqueeze(1)
        positions = torch.bmm(pos_centered.reshape(-1, 1, 2),
                               scene_rot_flat.unsqueeze(1).expand(-1, T, -1, -1).reshape(-1, 2, 2)
                               ).reshape(Ntot, T, 2)
        dir_rot = torch.bmm(dir_raw.reshape(-1, 1, 2),
                             scene_rot_flat.unsqueeze(1).expand(-1, T, -1, -1).reshape(-1, 2, 2)
                             ).reshape(Ntot, T, 2)
        positions = torch.where(padding_flat.unsqueeze(-1), torch.zeros_like(positions), positions)
        dir_rot = torch.where(padding_flat.unsqueeze(-1), torch.zeros_like(dir_rot), dir_rot)

        speed = torch.where(padding_flat, torch.zeros_like(raw_flat[..., 2]), raw_flat[..., 2])
        accel = torch.where(padding_flat, torch.zeros_like(raw_flat[..., 5]), raw_flat[..., 5])

        # displacement (원본 x[:,t]=pos[t]-pos[t-1], t=0은 0) — 양쪽 프레임 다 유효할 때만
        pos_disp = torch.zeros_like(positions)
        both_valid = (~padding_flat[:, 1:]) & (~padding_flat[:, :-1])
        pos_disp[:, 1:] = torch.where(both_valid.unsqueeze(-1), positions[:, 1:] - positions[:, :-1],
                                       torch.zeros_like(positions[:, 1:]))

        x_seq = torch.cat([pos_disp, speed.unsqueeze(-1), dir_rot, accel.unsqueeze(-1)], dim=-1)  # [Ntot,T,6]

        # bos_mask: t가 valid고 t-1이 invalid(또는 t=0인데 valid)
        bos_mask = torch.zeros_like(padding_flat)
        bos_mask[:, 0] = ~padding_flat[:, 0]
        bos_mask[:, 1:] = padding_flat[:, :-1] & ~padding_flat[:, 1:]

        # per-node heading(anchor 시점 회전된 방향벡터의 각도) — AA/Global 내부 국소 프레임 정렬용
        rotate_angles = torch.atan2(dir_rot[:, -1, 1], dir_rot[:, -1, 0])
        rotate_angles = torch.nan_to_num(rotate_angles, nan=0.0)

        # edge_index: 배치별 템플릿 offset
        b_idx = torch.arange(B, device=device) * N
        edge_index = (self.template_edge_index.unsqueeze(0) + b_idx.view(-1, 1, 1)).permute(1, 0, 2)
        edge_index = edge_index.reshape(2, -1)  # [2, B*N*(N-1)]

        return x_seq, positions, rotate_angles, padding_flat, bos_mask, edge_index

    def forward(self, batch: dict) -> torch.Tensor:
        x_seq, positions, rotate_angles, padding_mask, bos_mask, full_edge_index = self._build_graph(batch)
        Ntot = x_seq.size(0)
        device = x_seq.device

        rotate_mat = torch.empty(Ntot, 2, 2, device=device, dtype=x_seq.dtype)
        cos_a, sin_a = torch.cos(rotate_angles), torch.sin(rotate_angles)
        rotate_mat[:, 0, 0], rotate_mat[:, 0, 1] = cos_a, -sin_a
        rotate_mat[:, 1, 0], rotate_mat[:, 1, 1] = sin_a, cos_a

        # --- Stage: AA encoder, timestep별(비패딩 노드만 남긴 subgraph) ---
        outs = []
        for t in range(self.W):
            valid = ~padding_mask[:, t]
            e_idx = full_edge_index
            keep = valid[e_idx[0]] & valid[e_idx[1]]
            e_idx_t = e_idx[:, keep]
            edge_attr_t = positions[e_idx_t[0], t] - positions[e_idx_t[1], t]
            out_t = self.aa_encoder(x_seq[:, t], e_idx_t, edge_attr_t, bos_mask[:, t], rotate_mat)
            outs.append(out_t)
        local_seq = torch.stack(outs, dim=0)  # [T, Ntot, D]

        # --- Stage: Temporal encoder ---
        local_embed = self.temporal_encoder(local_seq, padding_mask)  # [Ntot, D]

        # --- Stage: Global interactor (anchor 시점 그래프, ALEncoder 없음) ---
        valid_anchor = ~padding_mask[:, -1]
        keep = valid_anchor[full_edge_index[0]] & valid_anchor[full_edge_index[1]]
        g_edge_index = full_edge_index[:, keep]
        rel_pos = positions[g_edge_index[0], -1] - positions[g_edge_index[1], -1]
        rel_pos = rotate2(rel_pos, rotate_mat[g_edge_index[1]])
        rel_theta = rotate_angles[g_edge_index[0]] - rotate_angles[g_edge_index[1]]
        rel_theta_vec = torch.stack([torch.cos(rel_theta), torch.sin(rel_theta)], dim=-1)  # [E,2]
        rel_embed = self.rel_embed([rel_pos, rel_theta_vec])

        g = local_embed
        for layer in self.global_layers:
            g = layer(g, g_edge_index, rel_embed)
        g = self.global_norm(g)

        ego_idx = torch.arange(batch['node_seq'].size(0), device=device) * self.N  # slot0 = ego
        ego_embed = g[ego_idx]
        return self.classifier(ego_embed)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
