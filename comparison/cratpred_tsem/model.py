"""
CRAT-Pred-adapted — TSEM-SAGE 비교용 재구현
=========================================
원본: comparison/crat-pred/ (Schmidt et al., ICRA 2022, 별도 명시 라이선스 없음 — repo 그대로 참조용)
  - model/crat_pred.py::EncoderLstm            -> TSEMEncoderLstm (수식 동일, 그대로 재사용 가능한
                                                   순수 nn.LSTM이라 재구현 불필요)
  - model/crat_pred.py::AgentGnn(CGConv x2)     -> TSEMCGConv (아래 재구현 — torch_geometric.nn.conv.CGConv
                                                   의존성 제거)
  - model/crat_pred.py::MultiheadSelfAttention  -> nn.MultiheadAttention 그대로 재사용(PyG 의존성 없음)
  - model/crat_pred.py::DecoderResidual         -> 미사용(다중모달 회귀 전용) — 분류 헤드로 교체

이 환경(tna_research)에는 torch_geometric이 없어 CGConv(Crystal Graph Convolution)를 순수
PyTorch(scatter_add_)로 다시 구현했다. CGConv 자체는 QCNet/HiVT의 attention과 달리 softmax가
없는 단순 gated-sum 집계라 재구현 난이도가 낮다 — 이게 CRAT-Pred를 재구현 최우선 후보로 꼽은
이유이기도 하다(§관련연구(A) 표 참조 — "맵 없이 차량 간 관계만으로 상호작용을 모델링").

HiVT/QCNet-adapted와의 구조적 차이:
  - HiVT: rotate_mat(회전행렬), QCNet: 상대각도(angle_between) — CRAT-Pred는 **원 논문 자체가
    scene 전체를 한 번(ego 헤딩 기준) 회전시키는 전역 정렬만 쓰고, 그 이후로는 회전 불변성 장치가
    없다**(HiVT/QCNet처럼 노드별/엣지별 추가 회전·각도 인코딩이 없음) — 그래서 이 어댑터도 원본과
    동일하게 "scene 1회 정렬"만 적용한다.
  - 시간축 처리: HiVT/QCNet은 transformer/attention이지만, CRAT-Pred는 **단순 LSTM**(에이전트별
    독립) — 셋 중 가장 단순한 시간 인코더라 "복잡한 시간 인코더가 꼭 필요한가"를 검증하는 대조군
    역할을 한다.
  - 공간축 처리: CGConv(그래프 conv, 2-layer) → MultiheadSelfAttention(1-layer) 순서 — HiVT/QCNet의
    반복형 다층 attention과 달리 "GNN 먼저, attention은 마지막에 한 번"이라는 얕은 구조.

주요 변경점(README.md 참조):
  - DecoderResidual(다중모달 회귀, mod_steps 앙상블) 제거 — 분류 헤드(Linear→num_classes)로 교체
  - LSTM 입력 3D(Δx,Δy,valid flag)는 원본과 100% 동일 — 우리 데이터의 speed/dir/accel 등 추가
    채널은 쓰지 않는다(원본이 애초에 "정말 최소한의 입력(변위+유효플래그)만으로 얼마나 되는가"를
    보여주는 게 이 모델의 존재 의의라 채널을 늘리면 CRAT-Pred라는 비교 포인트 자체가 흐려짐 —
    HiVT/QCNet과 다른 선택이며 의도적임).
  - agents_per_sample(가변 개수) 대신 고정 크기(1+K) + valid_agent 마스크로 그래프 구성(HiVT/QCNet과
    동일한 배치 벡터화 패턴).
  - 2-hop 이웃 미사용 — 원본 CRAT-Pred 자체가 1-hop(전체 관측 에이전트)만 쓰는 구조.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def scatter_sum_nodes(messages: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """comparison/hivt_tsem, qcnet_tsem의 model.py와 동일 함수."""
    D = messages.size(-1)
    out = torch.zeros((num_nodes, D), device=messages.device, dtype=messages.dtype)
    out.scatter_add_(0, index.unsqueeze(-1).expand(-1, D), messages)
    return out


class TSEMCGConv(nn.Module):
    """comparison/crat-pred/model/crat_pred.py::AgentGnn의 CGConv 1개 레이어 대응.
    원본(torch_geometric.nn.conv.CGConv) 정의: message(x_i,x_j,e) = sigmoid(W_f[x_i;x_j;e]) *
    softplus(W_s[x_i;x_j;e]), aggr='add'(sum), out = BatchNorm(sum) + x_dst(residual).
    x_i=target(edge_index[1]), x_j=source(edge_index[0]) — PyG MessagePassing 관례."""

    def __init__(self, channels: int, edge_dim: int):
        super().__init__()
        in_dim = 2 * channels + edge_dim
        self.lin_f = nn.Linear(in_dim, channels)
        self.lin_s = nn.Linear(in_dim, channels)
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        src, tgt = edge_index[0], edge_index[1]
        z = torch.cat([x[tgt], x[src], edge_attr], dim=-1)
        msg = torch.sigmoid(self.lin_f(z)) * F.softplus(self.lin_s(z))
        agg = scatter_sum_nodes(msg, tgt, num_nodes)
        agg = self.bn(agg)
        return agg + x


class TSEMEncoderLstm(nn.Module):
    """comparison/crat-pred/model/crat_pred.py::EncoderLstm 와 동일 — 순수 nn.LSTM,
    PyG 의존성이 전혀 없어 재구현 없이 원본 그대로 사용."""

    def __init__(self, latent_size: int):
        super().__init__()
        self.latent_size = latent_size
        self.lstm = nn.LSTM(input_size=3, hidden_size=latent_size, num_layers=1, batch_first=True)

    def forward(self, displ: torch.Tensor) -> torch.Tensor:
        """displ: [num_agents, T-1, 3] -> [num_agents, latent_size] (마지막 timestep hidden state)"""
        h0 = torch.zeros(1, displ.size(0), self.latent_size, device=displ.device, dtype=displ.dtype)
        c0 = torch.zeros(1, displ.size(0), self.latent_size, device=displ.device, dtype=displ.dtype)
        out, _ = self.lstm(displ, (h0, c0))
        return out[:, -1, :]


class CratPredTSEMAdapted(nn.Module):
    """CRAT-Pred 어댑터 최상위 모듈 — batch dict(TSEM dataloader 그대로) -> logits[B,num_classes].

    파이프라인: EncoderLstm(에이전트별 독립 LSTM) -> TSEMCGConv x2(완전연결 그래프, edge_attr=
    anchor 시점 상대 center) -> MultiheadSelfAttention(1layer) -> ego 노드 추출 -> 분류 헤드.
    """

    def __init__(self, W: int = 10, K: int = 6, latent_size: int = 64, num_heads: int = 4,
                 num_classes: int = 3):
        super().__init__()
        self.W = W
        self.K = K
        self.N = 1 + K
        self.latent_size = latent_size

        self.encoder_lstm = TSEMEncoderLstm(latent_size)
        self.gcn1 = TSEMCGConv(latent_size, edge_dim=2)
        self.gcn2 = TSEMCGConv(latent_size, edge_dim=2)
        self.self_attn = nn.MultiheadAttention(latent_size, num_heads)
        self.classifier = nn.Linear(latent_size, num_classes)

        # 완전연결(자기자신 제외) 그래프 템플릿 — 배치별 offset만 다름(HiVT/QCNet과 동일 패턴)
        idx = torch.arange(self.N)
        src, tgt = torch.meshgrid(idx, idx, indexing='ij')
        mask = src != tgt
        template_edge_index = torch.stack([src[mask], tgt[mask]], dim=0)
        self.register_buffer('template_edge_index', template_edge_index)

    def _build_graph(self, batch: dict):
        device = batch['node_seq'].device
        B = batch['node_seq'].size(0)
        N, W = self.N, self.W

        raw = torch.cat([batch['node_seq'].unsqueeze(1), batch['nbr_node_seqs']], dim=1)  # [B,N,W,6]
        nbr_mask = batch.get('nbr_mask')
        if nbr_mask is not None:
            valid_agent = torch.cat(
                [torch.ones(B, 1, device=device, dtype=torch.bool), nbr_mask.bool()], dim=1)  # [B,N]
        else:
            valid_agent = raw.abs().sum(dim=(-1, -2)) > 0  # fallback: 전 구간 전부 0이면 무효

        present = raw.abs().sum(dim=-1) > 0  # [B,N,W] — 프레임 단위 유효성(원본 valid flag와 동일 역할)

        # --- scene 1회 정렬(원본: AGENT 자신의 마지막 두 관측 스텝 헤딩 기준 회전) ---
        # 우리는 이미 제공되는 anchor 시점 ego 방향벡터를 그대로 사용(HiVT-adapted와 동일 근거로
        # finite-difference보다 노이즈가 적음).
        ego_pos_anchor = batch['node_seq'][:, -1, 0:2]  # [B,2]
        ego_dir_anchor = batch['node_seq'][:, -1, 3:5]  # [B,2]
        theta = torch.atan2(ego_dir_anchor[:, 1], ego_dir_anchor[:, 0])
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        rot = torch.stack([torch.stack([cos_t, -sin_t], -1), torch.stack([sin_t, cos_t], -1)], -2)  # [B,2,2]
        rot_flat = rot.repeat_interleave(N, dim=0)  # [B*N,2,2]
        origin_flat = ego_pos_anchor.repeat_interleave(N, dim=0)  # [B*N,2]

        pos_raw = raw[..., 0:2].reshape(B * N, W, 2)
        pos_centered = pos_raw - origin_flat.unsqueeze(1)
        pos = torch.bmm(pos_centered.reshape(-1, 1, 2),
                        rot_flat.unsqueeze(1).expand(-1, W, -1, -1).reshape(-1, 2, 2)).reshape(B * N, W, 2)
        present_flat = present.reshape(B * N, W)
        pos = torch.where(present_flat.unsqueeze(-1), pos, torch.zeros_like(pos))

        # --- displ: [dx,dy,valid] (원본 get_displ과 동일 정의) ---
        both_valid = present_flat[:, 1:] & present_flat[:, :-1]
        diff = pos[:, 1:] - pos[:, :-1]
        diff = torch.where(both_valid.unsqueeze(-1), diff, torch.zeros_like(diff))
        displ = torch.cat([diff, both_valid.unsqueeze(-1).float()], dim=-1)  # [B*N, W-1, 3]

        centers = pos[:, -1, :]  # anchor 시점 위치 [B*N, 2]

        # --- 그래프: valid_agent끼리만 완전연결 ---
        b_idx = torch.arange(B, device=device) * N
        edge_index = (self.template_edge_index.unsqueeze(0) + b_idx.view(-1, 1, 1)).permute(1, 0, 2).reshape(2, -1)
        valid_flat = valid_agent.reshape(-1)
        keep = valid_flat[edge_index[0]] & valid_flat[edge_index[1]]
        edge_index = edge_index[:, keep]
        edge_attr = centers[edge_index[1]] - centers[edge_index[0]]  # target - source (원본 정의와 동일)

        return displ, centers, valid_agent, edge_index, edge_attr

    def forward(self, batch: dict) -> torch.Tensor:
        displ, centers, valid_agent, edge_index, edge_attr = self._build_graph(batch)
        B, N = valid_agent.shape
        device = displ.device

        h = self.encoder_lstm(displ)  # [B*N, D]
        h = F.relu(self.gcn1(h, edge_index, edge_attr))
        h = F.relu(self.gcn2(h, edge_index, edge_attr))

        # MultiheadSelfAttention: (seq=N, batch=B, D), key_padding_mask로 무효 에이전트 제외
        h_seq = h.view(B, N, self.latent_size).transpose(0, 1)  # [N,B,D]
        key_padding_mask = ~valid_agent  # [B,N] True=무시
        # 전부 무효인 배치(이론상 ego는 항상 valid라 발생 안 함)는 NaN 방지를 위해 자기 자신만 살림
        attn_out, _ = self.self_attn(h_seq, h_seq, h_seq, key_padding_mask=key_padding_mask)
        attn_out = attn_out.transpose(0, 1)  # [B,N,D]

        ego_embed = attn_out[:, 0, :]  # ego는 항상 slot 0
        return self.classifier(ego_embed)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
