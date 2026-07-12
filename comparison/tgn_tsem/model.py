"""
TGN-adapted — TSEM-SAGE 비교용 재구현
=========================================
원본: comparison/TGN/ (Rossi et al., "Temporal Graph Networks for Deep Learning on Dynamic
Graphs", ICML 2020 Workshop on Graph Representation Learning, twitter-research/tgn, MIT License
— repo 그대로 참조용, 수정하지 않음).

원본 파일 -> 이 adapter 대응표:
  - model/tgn.py::TGN.update_memory / get_updated_memory / get_raw_messages
        -> TGNTSEMAdapted._step_memory() (아래, 프레임 단위로 재구현)
  - modules/memory.py::Memory
        -> 별도 클래스 없이 텐서 [B,N,memory_dim]로 직접 관리(전역 n_nodes 테이블이 아니라
           "샘플(윈도우)마다 독립된 zero-init memory" — 아래 설계 이탈 근거 참조)
  - modules/message_function.py::MLPMessageFunction
        -> TSEMMessageMLP (원본과 동일한 2-layer MLP: Linear->ReLU->Linear)
  - modules/message_aggregator.py::MeanMessageAggregator
        -> masked_mean() (원본은 파이썬 defaultdict+for문으로 가변개수 메시지를 모으지만, 우리
           그래프는 매 프레임 ego가 valid neighbor 전원과 "동시에" 상호작용하는 고정 크기(N=1+K)
           구조라 dense mask-mean으로 등가 재구현 — README 원칙과 동일하게 "핵심 메커니즘(평균
           집계)은 보존, 자료구조만 우리 배치 형태에 맞춤")
  - modules/memory_updater.py::GRUMemoryUpdater
        -> nn.GRUCell(message_dim, memory_dim) 그대로 사용(원본도 nn.GRUCell 그대로 씀 — 재구현
           불필요, TSEMMemoryUpdater 래퍼로 valid-mask 처리만 추가)
  - model/time_encoding.py::TimeEncode
        -> TimeEncode 그대로 재현(cos(Linear(t)) 수식·초기화 100% 동일, PyG/의존성 없어 재구현
           아님 — 원본 파일을 그대로 복사한 것에 가까움)
  - model/temporal_attention.py::TemporalAttentionLayer
        -> TSEMTemporalAttention (동일 수식: query=[node_feat;time_feat], key=[nbr_feat;edge_feat;
           edge_time_feat], nn.MultiheadAttention + key_padding_mask + "고립 노드는 첫 슬롯 강제
           unmask"까지 동일하게 재현)
  - utils/utils.py::MergeLayer
        -> TSEMMergeLayer(원본과 동일: Linear->ReLU->Linear, xavier_normal 초기화)
  - modules/embedding_module.py::GraphAttentionEmbedding.compute_embedding(n_layers=1 케이스)
        -> forward() 마지막 부분(anchor 프레임에서 1회만 attention 적용 — 아래 참조)

=========================================================================================
핵심 설계 이탈과 그 근거 (다른 5개 baseline과 동일한 "핵심 메커니즘 보존, 필요한 만큼만 변경"
원칙을 적용했지만, TGN은 이 프로젝트에서 원본과 가장 근본적인 전제가 다른 모델이라 가장 큰 설계
변경이 필요했다):

**원본 TGN 전제**: 전체 데이터셋에 걸친 하나의 비동기 연속시간 이벤트 스트림(예: 위키피디아 편집
로그, 소셜 상호작용) 위에서 동작한다. `Memory`는 데이터셋 내 **모든** 노드(`n_nodes`, 수만~수십만
개)에 대해 전역적으로 하나만 존재하고, 한 에포크 내내(또는 그 이상) 영속하며 미니배치가 지나갈
때마다 계속 갱신된다. 노드 임베딩을 구할 때는 그 노드의 "현재까지 누적된" memory를 사용한다.

**우리 과제와의 근본적 불일치**: TSEM 분류는 W=10 고정 길이 윈도우 단위의 **지도학습**이다 — 각
학습 샘플은 "ego 차량 + 그 순간의 이웃들, 그 순간 이전 W프레임"으로 완결된 독립 단위이고, 서로
다른 샘플(=다른 시각, 다른 ego)이 같은 memory를 공유할 근거가 없다(공유하면 정보 누수 —
`build_tsem_dataloaders`가 train/val/test를 파일·시간대 단위로 분리하는 전제와 충돌).
전역 영속 memory는 "누가 몇 번 슬롯인지"가 데이터셋 전체에서 고정돼야 성립하는데, 우리 그래프는
KNN 기반이라 이웃 구성 자체가 샘플마다 다르다(슬롯 1..K가 매 샘플 다른 실제 차량).

**해결**: memory를 "데이터셋 전역"이 아니라 "**샘플(윈도우) 단위로 리셋**"한다. 각 학습 샘플의
에이전트 슬롯(ego=slot 0, 이웃=slot 1..K, N=1+K개)마다 그 샘플의 W프레임 시작 시점에
zero-initialized memory를 두고, W프레임을 순서대로 "이벤트 배치"(원본의 미니배치 순회와
동일한 역할)로 처리하며 매 프레임 memory를 갱신한다. 즉 원본의 "전체 데이터셋=하나의 스트림"을
"각 샘플=자기 완결적인 미니 스트림(길이 W)"으로 축소했다 — memory 갱신 규칙(message->aggregate
->GRUCell) 자체는 원본과 동일하게 유지하고, memory의 "생애주기(scope)"만 바꿨다는 것이 이 설계
이탈의 정확한 성격이다.

**메시지**: 원본은 실제 상호작용(edge event)이 발생한 (source,destination,edge_feat,edge_time)
튜플에서 raw_message = cat([source_memory, destination_memory, edge_features, Δt_encoding])를
만든다. 우리는 매 프레임 t에서 "ego가 valid한 모든 이웃과 동시에 상호작용한다"고 보고, ego<->이웃
쌍마다 이미 계산돼 있는 edge_seqs[...,t,:]([rel_speed,rel_accel,rel_dir_x,rel_dir_z,distance])를
edge feature로 그대로 쓴다. Δt는 항상 1프레임 간격으로 고정이지만(연속시간이 아니라 등간격
샘플링이므로), 원본처럼 TimeEncode를 그대로 통과시켜 "형태"는 보존한다(원본 수식을 임의로
간소화하지 않기 위함).

**메시지 집계**: 원본 MeanMessageAggregator는 같은 미니배치 안에서 한 노드에 여러 메시지가
몰릴 수 있어 평균을 낸다. 우리 그래프는 매 프레임 ego가 최대 K개 이웃과 "동시에" 상호작용하므로
ego의 mailbox에 K개(중 valid한 것만) 메시지가 동시에 들어온다 — 원본의 mean aggregator를
그대로 쓰되, N=1+K가 소규모(<=7)라 파이썬 dict+for문 대신 dense mask-mean으로 구현했다(수학적으로
동일한 연산, 자료구조만 배치-벡터화).

**Memory updater**: `nn.GRUCell(message_dim, memory_dim)` — 원본과 100% 동일(원본도 결국
nn.GRUCell 그대로 사용).

**Embedding module**: 원본 GraphAttentionEmbedding은 임의의 query 시점에서 최근 이웃과의
temporal graph attention으로 최종 임베딩을 계산한다(주로 n_layers=1~2). 우리는 매 프레임마다
계산하지 않고 **마지막 프레임(anchor 시점, t=W-1)에서만** 1회 적용한다 — 원본도 실제로는
"쿼리하는 시점"에서만 임베딩을 계산하지(매 이벤트마다 임베딩을 뽑는 게 아니라 memory만 갱신),
우리의 "예측 시점"이 정확히 anchor 프레임이므로 이게 원본 의도에 가장 가깝다.

**나머지 4개 adapter와 공통인 부분**: scene 1회 정렬(ego anchor 위치·헤딩 기준 회전, 다른 4개와
동일), 노드 입력 6D raw 그대로(HiVT와 동일 채널), 1-hop 전용(2-hop 미사용, 원본 TGN도 이웃 홉수는
n_neighbors 하이퍼파라미터일 뿐 우리 그래프처럼 애초에 1-hop 완비 그래프면 추가 홉 의미 없음),
회귀 디코더(affinity_score, link prediction) 제거 -> 분류 헤드(Linear->num_classes) 교체(공통
패턴).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int, eps: float = 1e-8) -> torch.Tensor:
    """x: [..., K, D], mask: [..., K] (True=valid) -> [..., D]. 원본 MeanMessageAggregator와
    동일 연산(같은 타겟으로 들어오는 메시지의 평균)을 dense mask-mean으로 재구현."""
    mask_f = mask.unsqueeze(-1).to(x.dtype)
    summed = (x * mask_f).sum(dim=dim)
    count = mask_f.sum(dim=dim).clamp_min(eps)
    return summed / count


class TimeEncode(nn.Module):
    """comparison/TGN/model/time_encoding.py::TimeEncode 그대로 재현(수식·초기화 100% 동일).
    PyG 등 외부 의존성이 없는 순수 nn.Linear+cos이라 재구현이 아니라 그대로 복제."""

    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self.w = nn.Linear(1, dimension)
        self.w.weight = nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension))).float().reshape(dimension, -1)
        )
        self.w.bias = nn.Parameter(torch.zeros(dimension).float())

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: 임의의 leading shape (채널 없음) -> [..., dimension]."""
        return torch.cos(self.w(t.unsqueeze(-1)))


class TSEMMessageMLP(nn.Module):
    """modules/message_function.py::MLPMessageFunction 그대로 재현
    (raw_message_dim -> raw_message_dim//2 -> message_dim, ReLU)."""

    def __init__(self, raw_message_dim: int, message_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(raw_message_dim, raw_message_dim // 2),
            nn.ReLU(),
            nn.Linear(raw_message_dim // 2, message_dim),
        )

    def forward(self, raw_messages: torch.Tensor) -> torch.Tensor:
        return self.mlp(raw_messages)


class TSEMMergeLayer(nn.Module):
    """utils/utils.py::MergeLayer 그대로 재현(Linear->ReLU->Linear, xavier_normal 초기화)."""

    def __init__(self, dim1: int, dim2: int, dim3: int, dim4: int):
        super().__init__()
        self.fc1 = nn.Linear(dim1 + dim2, dim3)
        self.fc2 = nn.Linear(dim3, dim4)
        self.act = nn.ReLU()
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x1, x2], dim=-1)
        h = self.act(self.fc1(x))
        return self.fc2(h)


class TSEMTemporalAttention(nn.Module):
    """model/temporal_attention.py::TemporalAttentionLayer 재현 — query=ego 1개,
    key/value=이웃 N개(고정 크기, key_padding_mask로 무효 이웃 제외). "고립 노드는 첫 슬롯을 강제로
    unmask해 attention이 NaN을 내지 않도록" 하는 원본의 안전장치까지 그대로 재현."""

    def __init__(self, n_node_features: int, n_edge_features: int, time_dim: int,
                 output_dimension: int, n_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.query_dim = n_node_features + time_dim
        self.key_dim = n_node_features + n_edge_features + time_dim
        self.merger = TSEMMergeLayer(self.query_dim, n_node_features, n_node_features, output_dimension)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.query_dim, kdim=self.key_dim, vdim=self.key_dim,
            num_heads=n_heads, dropout=dropout,
        )

    def forward(self, src_node_features: torch.Tensor, src_time_features: torch.Tensor,
                neighbor_features: torch.Tensor, neighbor_time_features: torch.Tensor,
                edge_features: torch.Tensor, neighbor_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        src_node_features: [B, D]     src_time_features: [B, time_dim]
        neighbor_features: [B, K, D]  neighbor_time_features: [B, K, time_dim]
        edge_features: [B, K, edge_dim]   neighbor_padding_mask: [B, K] True=무시
        -> [B, output_dimension]
        """
        query = torch.cat([src_node_features.unsqueeze(1), src_time_features.unsqueeze(1)], dim=-1)  # [B,1,Dq]
        key = torch.cat([neighbor_features, edge_features, neighbor_time_features], dim=-1)  # [B,K,Dk]

        query = query.transpose(0, 1)  # [1,B,Dq]
        key = key.transpose(0, 1)  # [K,B,Dk]

        neighbor_padding_mask = neighbor_padding_mask.clone()
        invalid_all = neighbor_padding_mask.all(dim=1)  # [B] 이웃이 전부 무효인 샘플
        if invalid_all.any():
            neighbor_padding_mask[invalid_all, 0] = False  # 원본과 동일: 첫 슬롯 강제 unmask

        attn_out, _ = self.attn(query, key, key, key_padding_mask=neighbor_padding_mask)
        attn_out = attn_out.squeeze(0)  # [B, Dq]
        attn_out = torch.where(invalid_all.unsqueeze(-1), torch.zeros_like(attn_out), attn_out)

        return self.merger(attn_out, src_node_features)


class TGNTSEMAdapted(nn.Module):
    """TGN 어댑터 최상위 모듈 — batch dict(TSEM dataloader 그대로) -> logits[B,num_classes].

    파이프라인: scene 1회 정렬 -> [프레임 순회: 노드 원시피처 인코딩 -> 메시지 생성(memory+edge_feat
    +Δt) -> masked-mean 집계 -> GRUCell로 memory 갱신] x W -> anchor 프레임에서 1회
    TSEMTemporalAttention -> ego 임베딩 -> 분류 헤드.

    memory는 샘플(윈도우) 단위로 매 forward 호출마다 zero-init된다(위 docstring "핵심 설계 이탈"
    참조) — 원본처럼 여러 배치·에포크에 걸쳐 영속하지 않는다.
    """

    def __init__(self, W: int = 10, K: int = 6, memory_dim: int = 64, message_dim: int = 64,
                 time_dim: int = 16, num_heads: int = 2, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.W = W
        self.K = K
        self.N = 1 + K
        self.memory_dim = memory_dim
        self.message_dim = message_dim
        self.time_dim = time_dim
        node_raw_dim = 6
        edge_dim = 5

        self.node_encoder = nn.Linear(node_raw_dim, memory_dim)  # 원본 node_raw_features 역할
        self.time_encoder = TimeEncode(time_dim)

        raw_message_dim = 2 * memory_dim + edge_dim + time_dim
        self.message_fn = TSEMMessageMLP(raw_message_dim, message_dim)
        self.memory_updater = nn.GRUCell(message_dim, memory_dim)

        self.embedding_attn = TSEMTemporalAttention(
            n_node_features=memory_dim, n_edge_features=edge_dim, time_dim=time_dim,
            output_dimension=memory_dim, n_heads=num_heads, dropout=dropout,
        )
        self.classifier = nn.Linear(memory_dim, num_classes)

    def _align_scene(self, batch: dict):
        """comparison/cratpred_tsem, hivt_tsem 등과 동일한 scene 1회 정렬(ego anchor 위치·헤딩
        기준 평행이동+회전). node_seq/nbr_node_seqs의 pos(0:2)·dir(3:5) 채널만 회전, speed/accel은
        스칼라라 불변."""
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
        pos_centered = pos_raw - origin_flat.unsqueeze(1)
        pos = torch.bmm(
            pos_centered.reshape(-1, 1, 2),
            rot_flat.unsqueeze(1).expand(-1, W, -1, -1).reshape(-1, 2, 2),
        ).reshape(B, N, W, 2)

        dir_raw = raw[..., 3:5].reshape(B * N, W, 2)
        dir_rot = torch.bmm(
            dir_raw.reshape(-1, 1, 2),
            rot_flat.unsqueeze(1).expand(-1, W, -1, -1).reshape(-1, 2, 2),
        ).reshape(B, N, W, 2)

        aligned = raw.clone()
        aligned[..., 0:2] = pos
        aligned[..., 3:5] = dir_rot
        aligned = torch.where(present.unsqueeze(-1), aligned, torch.zeros_like(aligned))

        return aligned, valid_agent, present

    def forward(self, batch: dict) -> torch.Tensor:
        device = batch['node_seq'].device
        aligned, valid_agent, present = self._align_scene(batch)  # [B,N,W,6], [B,N], [B,N,W]
        B, N, W = aligned.size(0), self.N, self.W
        K = self.K

        edge_seqs = batch['edge_seqs']  # [B,K,W,5] (ego 기준 이웃 상대 피처, 원본 그대로 사용)
        nbr_mask = valid_agent[:, 1:]  # [B,K]

        node_feat = self.node_encoder(aligned)  # [B,N,W,memory_dim]

        memory = torch.zeros(B, N, self.memory_dim, device=device, dtype=node_feat.dtype)
        dt1 = torch.ones(B, K, device=device, dtype=node_feat.dtype)
        time_enc_dt1 = self.time_encoder(dt1)  # [B,K,time_dim]  (Δt=1 고정, 형태만 원본과 동일하게 보존)

        for t in range(W):
            present_t = present[:, :, t]  # [B,N]
            ego_present_t = present_t[:, 0]  # [B]
            nbr_present_t = present_t[:, 1:]  # [B,K]
            pair_valid = nbr_mask & nbr_present_t & ego_present_t.unsqueeze(-1)  # [B,K]

            memory_ego = memory[:, 0, :]  # [B,D]
            memory_nbrs = memory[:, 1:, :]  # [B,K,D]
            edge_feat_t = edge_seqs[:, :, t, :]  # [B,K,5]

            # --- ego 방향 메시지: source=ego, destination=이웃(원본 get_raw_messages 대칭 구성) ---
            raw_msg_ego = torch.cat([
                memory_ego.unsqueeze(1).expand(-1, K, -1), memory_nbrs, edge_feat_t, time_enc_dt1,
            ], dim=-1)  # [B,K,raw_dim]
            msg_ego_per_nbr = self.message_fn(raw_msg_ego)  # [B,K,message_dim]
            agg_msg_ego = masked_mean(msg_ego_per_nbr, pair_valid, dim=1)  # [B,message_dim]

            has_any_valid = pair_valid.any(dim=1)  # [B]
            new_memory_ego = self.memory_updater(agg_msg_ego, memory_ego)
            memory_ego_next = torch.where(has_any_valid.unsqueeze(-1), new_memory_ego, memory_ego)

            # --- 이웃 방향 메시지: source=이웃, destination=ego ---
            raw_msg_nbr = torch.cat([
                memory_nbrs, memory_ego.unsqueeze(1).expand(-1, K, -1), edge_feat_t, time_enc_dt1,
            ], dim=-1)  # [B,K,raw_dim]  (이웃당 메시지 1개뿐이므로 집계는 그 자체가 항등)
            msg_nbr = self.message_fn(raw_msg_nbr.reshape(B * K, -1)).reshape(B, K, -1)
            new_memory_nbrs = self.memory_updater(
                msg_nbr.reshape(B * K, -1), memory_nbrs.reshape(B * K, -1)
            ).reshape(B, K, -1)
            memory_nbrs_next = torch.where(pair_valid.unsqueeze(-1), new_memory_nbrs, memory_nbrs)

            memory = torch.cat([memory_ego_next.unsqueeze(1), memory_nbrs_next], dim=1)

        # --- anchor 프레임(t=W-1)에서 1회 embedding 계산 (원본 "쿼리 시점에만 임베딩" 취지) ---
        last = W - 1
        node_feat_last = node_feat[:, :, last, :]  # [B,N,D]
        source_node_features = memory[:, 0, :] + node_feat_last[:, 0, :]  # [B,D] (원본: memory+raw feat)
        neighbor_features = memory[:, 1:, :] + node_feat_last[:, 1:, :]  # [B,K,D]

        zero_t = torch.zeros(B, device=device, dtype=node_feat.dtype)
        src_time_features = self.time_encoder(zero_t)  # [B,time_dim] (쿼리 노드는 time span=0)
        nbr_time_features = self.time_encoder(zero_t).unsqueeze(1).expand(-1, K, -1)  # 동일 프레임=Δt 0

        edge_feat_last = edge_seqs[:, :, last, :]  # [B,K,5]
        neighbor_padding_mask = ~(nbr_mask & present[:, 1:, last])  # [B,K] True=무시

        ego_embed = self.embedding_attn(
            source_node_features, src_time_features, neighbor_features, nbr_time_features,
            edge_feat_last, neighbor_padding_mask,
        )
        return self.classifier(ego_embed)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
