"""
SIMPL-adapted — TSEM-SAGE 비교용 재구현
=========================================
원본: comparison/SIMPL/ (Zhang et al., IEEE RA-L 2024)
  - simpl/simpl.py::ActorNet                        -> TSEMActorNet (1D CNN ResNet+FPN, 수식 동일)
  - simpl/simpl.py::SftLayer/SymmetricFusionTransformer -> TSEMSftLayer/TSEMSceneFusion (아래 재구현)
  - simpl/simpl.py::LaneNet, MLPDecoder             -> 미사용(맵 인코더 · 다중모달 회귀 디코더 제거)
  - simpl/av1_dataset.py::_get_rpe                  -> build_rpe (수식 동일, 아래 재구현)

이 환경(tna_research)에는 torch_geometric이 없지만, SIMPL 원본은 애초에 그래프 라이브러리에
의존하지 않는다(순수 nn.Conv1d/GroupNorm/nn.MultiheadAttention) — 그래서 이 어댑터는 HiVT/QCNet
어댑터와 달리 attention 자체를 scatter로 재구현할 필요가 없다. SftLayer의 핵심 아이디어는
"각 타깃 토큰마다 (edge, source, target) 정보를 합친 전용 memory 텐서를 만들고, 그 memory를
key/value로 삼아 PyTorch 내장 nn.MultiheadAttention을 호출"하는 것 — attention 수식 자체는
신뢰할 수 있는 내장 모듈에 위임되므로, 우리가 검증해야 할 건 memory 텐서를 올바르게 만드는
브로드캐스트 로직뿐이다(comparison/simpl_tsem/test_model.py에서 브루트포스 대조).

HiVT/QCNet/CRAT-Pred-adapted와의 구조적 차이:
  - 시간 인코더가 **1D CNN(ResNet+FPN, 다중 스케일)** — 셋 중 유일하게 transformer도 LSTM도 아님.
  - 공간 융합이 "각 타깃마다 (edge+source+target) 결합 memory → attention"이라는 독특한 패턴 —
    QCNet의 "key/value에 상대위치를 더하는" 방식과도, HiVT의 "회전행렬로 정렬 후 표준 attention"
    방식과도 다르다.
  - 회전 불변성은 QCNet과 유사하게 **상대 각도(cos/sin)를 직접 특징으로 사용**하지만, 코사인·사인을
    바로 특징 벡터에 넣는다는 점에서 QCNet의 atan2 기반 각도값과는 표현이 다르다(연속성 측면에서
    cos/sin 표현이 각도 wrap-around 불연속을 피할 수 있어 오히려 더 매끄러움).

주요 변경점(README.md 참조):
  - LaneNet(맵 인코더) 제거 — 맵 데이터 없음. RPE(상대 위치 인코딩)도 에이전트끼리만 계산.
  - MLPDecoder(다중모달 베지어/모노미얼 회귀) 제거 — 분류 헤드로 교체.
  - ActorNet 입력은 원본과 동일 3D([Δx,Δy,valid flag]) — CRAT-Pred와 같은 이유로 채널을 늘리지
    않음(원본 설계를 그대로 시험).
  - n_fpn_scale 4→3(기본값) — 원본은 T=20(historical_steps) 기준으로 설계됐고 우리는 T=10이라,
    스케일을 하나 줄여 다운샘플링이 과도해지는 것을 방지. FPN 업샘플도 원본의
    `scale_factor=2`(길이가 정확히 2배씩 줄어든다고 가정) 대신 `size=`를 명시적으로 지정해 임의
    T에서도 길이 불일치 없이 안전하게 동작하도록 강건화했다(수식·구조는 동일, 견고성만 개선).
  - 2-hop 이웃 미사용 — 원본도 1-hop(관측된 모든 에이전트)만 사용.
"""
from __future__ import annotations

from math import gcd
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────
# ActorNet — comparison/SIMPL/simpl/simpl.py::Conv1d/Res1d/ActorNet 와 동일(순수 CNN,
# PyG 의존성 없음). FPN 업샘플만 scale_factor 대신 size= 로 강건화(주석 참조).
# ────────────────────────────────────────────────────────────────
class Conv1dBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, ng=1, act=True):
        super().__init__()
        self.conv = nn.Conv1d(n_in, n_out, kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2, stride=stride, bias=False)
        self.norm = nn.GroupNorm(gcd(ng, n_out), n_out)
        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        out = self.norm(self.conv(x))
        return self.relu(out) if self.act else out


class Res1dBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, ng=1, act=True):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.conv2 = nn.Conv1d(n_out, n_out, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn1 = nn.GroupNorm(gcd(ng, n_out), n_out)
        self.bn2 = nn.GroupNorm(gcd(ng, n_out), n_out)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or n_out != n_in:
            self.downsample = nn.Sequential(
                nn.Conv1d(n_in, n_out, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(gcd(ng, n_out), n_out))
        else:
            self.downsample = None
        self.act = act

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            x = self.downsample(x)
        out = out + x
        return self.relu(out) if self.act else out


class TSEMActorNet(nn.Module):
    """comparison/SIMPL/simpl/simpl.py::ActorNet 대응 — 다중 스케일 Res1d + FPN lateral 결합.
    n_fpn_scale=3(원본 4에서 축소, T=10 대응) — docstring 상단 참조."""

    def __init__(self, n_in: int = 3, hidden_size: int = 64, n_fpn_scale: int = 3):
        super().__init__()
        n_out = [2 ** (5 + s) for s in range(n_fpn_scale)]  # [32, 64, 128, ...]
        groups = []
        cur_in = n_in
        for i in range(n_fpn_scale):
            stride = 1 if i == 0 else 2
            block = nn.Sequential(
                Res1dBlock(cur_in, n_out[i], stride=stride),
                Res1dBlock(n_out[i], n_out[i]))
            groups.append(block)
            cur_in = n_out[i]
        self.groups = nn.ModuleList(groups)
        self.lateral = nn.ModuleList([Conv1dBlock(n_out[i], hidden_size, act=False) for i in range(n_fpn_scale)])
        self.output = Res1dBlock(hidden_size, hidden_size)

    def forward(self, actors: torch.Tensor) -> torch.Tensor:
        """actors: [num_agents, n_in, T] -> [num_agents, hidden_size]"""
        out = actors
        outputs = []
        for group in self.groups:
            out = group(out)
            outputs.append(out)

        out = self.lateral[-1](outputs[-1])
        for i in range(len(outputs) - 2, -1, -1):
            # 원본은 scale_factor=2로 업샘플하지만, T가 정확히 2의 배수로 줄어든다는 보장이 없어
            # (특히 T=10처럼 짧은 경우) 목표 길이를 직접 지정해 길이 불일치를 방지한다.
            out = F.interpolate(out, size=outputs[i].shape[-1], mode='linear', align_corners=False)
            out = out + self.lateral[i](outputs[i])

        out = self.output(out)[:, :, -1]
        return out


# ────────────────────────────────────────────────────────────────
# RPE(상대 위치 인코딩) — comparison/SIMPL/simpl/av1_dataset.py::_get_rpe 와 동일 수식.
# 순수 dense 브로드캐스트라 scatter/그래프 라이브러리 없이도 원본과 동일하게 구현 가능.
# ────────────────────────────────────────────────────────────────
def _get_cos(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    v1n, v2n = v1.norm(dim=-1), v2.norm(dim=-1)
    return (v1[..., 0] * v2[..., 0] + v1[..., 1] * v2[..., 1]) / (v1n * v2n + 1e-10)


def _get_sin(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    v1n, v2n = v1.norm(dim=-1), v2.norm(dim=-1)
    return (v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]) / (v1n * v2n + 1e-10)


def build_rpe(ctrs: torch.Tensor, vecs: torch.Tensor, radius: float = 100.0) -> torch.Tensor:
    """ctrs,vecs: [N,2] (배치 없는 단일 그래프 기준) -> rpe[N,N,5]
    (원본은 [5,N,N] 반환 후 permute — 여기서는 바로 [N,N,5]로 반환해 사용처에서 permute 불필요)"""
    d_pos = (ctrs.unsqueeze(0) - ctrs.unsqueeze(1)).norm(dim=-1) * 2.0 / radius  # [N,N]
    cos_a1 = _get_cos(vecs.unsqueeze(0), vecs.unsqueeze(1))
    sin_a1 = _get_sin(vecs.unsqueeze(0), vecs.unsqueeze(1))
    v_pos = ctrs.unsqueeze(0) - ctrs.unsqueeze(1)
    cos_a2 = _get_cos(vecs.unsqueeze(0), v_pos)
    sin_a2 = _get_sin(vecs.unsqueeze(0), v_pos)
    return torch.stack([cos_a1, sin_a1, cos_a2, sin_a2, d_pos], dim=-1)  # [N,N,5]


def build_rpe_batched(ctrs: torch.Tensor, vecs: torch.Tensor, radius: float = 100.0) -> torch.Tensor:
    """ctrs,vecs: [B,N,2] -> rpe[B,N,N,5] (build_rpe의 배치 버전, 동일 수식)"""
    d_pos = (ctrs.unsqueeze(1) - ctrs.unsqueeze(2)).norm(dim=-1) * 2.0 / radius  # [B,N,N]
    v1 = vecs.unsqueeze(1)  # [B,1,N,2]
    v2 = vecs.unsqueeze(2)  # [B,N,1,2]

    def cos(a, b):
        return (a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-10)

    def sin(a, b):
        return (a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-10)

    cos_a1 = cos(v1.expand(-1, ctrs.size(1), -1, -1), v2.expand(-1, -1, ctrs.size(1), -1))
    sin_a1 = sin(v1.expand(-1, ctrs.size(1), -1, -1), v2.expand(-1, -1, ctrs.size(1), -1))
    v_pos = ctrs.unsqueeze(1) - ctrs.unsqueeze(2)  # [B,N,N,2]
    cos_a2 = cos(v1.expand(-1, ctrs.size(1), -1, -1), v_pos)
    sin_a2 = sin(v1.expand(-1, ctrs.size(1), -1, -1), v_pos)
    return torch.stack([cos_a1, sin_a1, cos_a2, sin_a2, d_pos], dim=-1)  # [B,N,N,5]


class TSEMSftLayer(nn.Module):
    """comparison/SIMPL/simpl/simpl.py::SftLayer 대응 — 배치 벡터화 버전.
    memory[b,s,i] = proj_memory(cat[edge[b,s,i], node[b,i](src), node[b,s](tgt)]) — 원본과
    동일한 src_x/tar_x 브로드캐스트 규칙(주석 상단 파일 docstring 참조). 실제 attention은
    PyTorch 내장 nn.MultiheadAttention에 위임."""

    def __init__(self, d_model: int, d_edge: int, n_head: int, dropout: float, update_edge: bool):
        super().__init__()
        self.update_edge = update_edge
        self.proj_memory = nn.Sequential(
            nn.Linear(d_model + d_model + d_edge, d_model), nn.LayerNorm(d_model), nn.ReLU(inplace=True))
        if update_edge:
            self.proj_edge = nn.Sequential(
                nn.Linear(d_model, d_edge), nn.LayerNorm(d_edge), nn.ReLU(inplace=True))
            self.norm_edge = nn.LayerNorm(d_edge)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head, dropout=dropout, batch_first=False)
        self.linear1 = nn.Linear(d_model, d_model * 2)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 2, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, node: torch.Tensor, edge: torch.Tensor, key_padding_mask: torch.Tensor):
        """node:[B,N,D], edge:[B,N,N,d_edge], key_padding_mask:[B,N](True=무시할 소스)
        -> (node_new[B,N,D], edge_new[B,N,N,d_edge] or None)"""
        B, N, D = node.shape
        src_x = node.unsqueeze(1).expand(-1, N, -1, -1)  # [B,N(tgt-broadcast),N,D] -> [b,i,j]=node[b,j]
        tar_x = node.unsqueeze(2).expand(-1, -1, N, -1)  # [b,i,j]=node[b,i]
        memory = self.proj_memory(torch.cat([edge, src_x, tar_x], dim=-1))  # [B,N,N,D] ([b,i,j])

        if self.update_edge:
            edge = self.norm_edge(edge + self.proj_edge(memory))

        # attention batch index (b,i) -> combined B*N; query seq_len=1, key/value seq_len=N(over j)
        # 원본(_mha_block)의 attention은 query=node[q](배치), key/value 시퀀스=p(소스 후보)를
        # memory[p,q]에서 뽑는다(memory[p,q]=f(edge[p,q],node[q]=src,node[p]=tar), 파일 상단 docstring
        # 참조). 우리 표기로는 p=i(소스/시퀀스), q=j(타깃/배치) — 즉 배치=j, 시퀀스=i여야 한다.
        # (2026-07-11: 최초 구현 시 배치/시퀀스를 반대로(i=배치,j=시퀀스) 잡는 실수가 있었고,
        # comparison/simpl_tsem/test_model.py의 브루트포스 대조에서 발견·수정함.)
        x = node.reshape(1, B * N, D)
        # memory[b,i,j] -> mem_combined[i, b*N+j, D] (seq=i, batch=b*N+j)
        mem = memory.permute(1, 0, 2, 3).reshape(N, B * N, D)
        # key_padding_mask[b,i] -> combined[b*N+j, i] (j에 무관하게 소스 i의 유효성만 반영)
        kpm = key_padding_mask.unsqueeze(1).expand(-1, N, -1).reshape(B * N, N)

        attn_out, _ = self.mha(x, mem, mem, key_padding_mask=kpm, need_weights=False)  # [1,B*N,D]
        attn_out = self.dropout2(attn_out)
        node_flat = node.reshape(1, B * N, D)
        x = self.norm2(node_flat + attn_out).reshape(B * N, D)
        ff = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = self.norm3(x + self.dropout3(ff))
        node_new = x.reshape(B, N, D)

        return node_new, (edge if self.update_edge else None)


class SIMPLTSEMAdapted(nn.Module):
    """SIMPL 어댑터 최상위 모듈 — batch dict(TSEM dataloader 그대로) -> logits[B,num_classes].
    ActorNet(1D CNN) -> RPE 계산 -> TSEMSftLayer x n_layer -> ego 노드 추출 -> 분류 헤드."""

    def __init__(self, W: int = 10, K: int = 6, hidden_size: int = 64, n_fpn_scale: int = 3,
                 d_rpe: int = 32, n_layer: int = 4, n_head: int = 8, dropout: float = 0.1,
                 rpe_radius: float = 100.0, num_classes: int = 3):
        super().__init__()
        self.W = W
        self.K = K
        self.N = 1 + K
        self.hidden_size = hidden_size
        self.rpe_radius = rpe_radius

        self.actor_net = TSEMActorNet(n_in=3, hidden_size=hidden_size, n_fpn_scale=n_fpn_scale)
        self.proj_rpe = nn.Sequential(nn.Linear(5, d_rpe), nn.LayerNorm(d_rpe), nn.ReLU(inplace=True))
        self.layers = nn.ModuleList([
            TSEMSftLayer(hidden_size, d_rpe, n_head, dropout, update_edge=(i != n_layer - 1))
            for i in range(n_layer)])
        self.classifier = nn.Linear(hidden_size, num_classes)

    def _build_inputs(self, batch: dict):
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

        # scene 1회 정렬(CRAT-Pred-adapted와 동일 근거: ego anchor 헤딩 기준)
        ego_pos_anchor = batch['node_seq'][:, -1, 0:2]
        ego_dir_anchor = batch['node_seq'][:, -1, 3:5]
        theta = torch.atan2(ego_dir_anchor[:, 1], ego_dir_anchor[:, 0])
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        rot = torch.stack([torch.stack([cos_t, -sin_t], -1), torch.stack([sin_t, cos_t], -1)], -2)  # [B,2,2]
        rot_flat = rot.repeat_interleave(N, dim=0)
        origin_flat = ego_pos_anchor.repeat_interleave(N, dim=0)

        pos_raw = raw[..., 0:2].reshape(B * N, W, 2)
        pos_centered = pos_raw - origin_flat.unsqueeze(1)
        pos = torch.bmm(pos_centered.reshape(-1, 1, 2),
                        rot_flat.unsqueeze(1).expand(-1, W, -1, -1).reshape(-1, 2, 2)).reshape(B * N, W, 2)
        dir_raw = raw[..., 3:5].reshape(B * N, W, 2)
        dir_rot = torch.bmm(dir_raw.reshape(-1, 1, 2),
                            rot_flat.unsqueeze(1).expand(-1, W, -1, -1).reshape(-1, 2, 2)).reshape(B * N, W, 2)
        present_flat = present.reshape(B * N, W)
        pos = torch.where(present_flat.unsqueeze(-1), pos, torch.zeros_like(pos))
        dir_rot = torch.where(present_flat.unsqueeze(-1), dir_rot, torch.zeros_like(dir_rot))

        both_valid = present_flat[:, 1:] & present_flat[:, :-1]
        diff = pos[:, 1:] - pos[:, :-1]
        diff = torch.where(both_valid.unsqueeze(-1), diff, torch.zeros_like(diff))
        # ActorNet 입력: [B*N, 3(=dx,dy,valid), W] — 원본과 동일하게 첫 프레임 변위는 0
        vel = torch.zeros(B * N, W, 2, device=device, dtype=raw.dtype)
        vel[:, 1:] = diff
        actor_in = torch.cat([vel, present_flat.unsqueeze(-1).float()], dim=-1)  # [B*N,W,3]
        actor_in = actor_in.transpose(1, 2)  # [B*N,3,W] (Conv1d channel-first)

        centers = pos[:, -1, :].reshape(B, N, 2)
        # heading 벡터: anchor 시점 방향벡터(회전 반영됨) — 무효 노드는 0으로 두되 norm+1e-10로 안전 처리
        head_vecs = dir_rot[:, -1, :].reshape(B, N, 2)

        return actor_in, centers, head_vecs, valid_agent

    def forward(self, batch: dict) -> torch.Tensor:
        actor_in, centers, head_vecs, valid_agent = self._build_inputs(batch)
        B, N = valid_agent.shape

        node = self.actor_net(actor_in).view(B, N, self.hidden_size)
        rpe = build_rpe_batched(centers, head_vecs, radius=self.rpe_radius)  # [B,N,N,5]
        edge = self.proj_rpe(rpe)  # [B,N,N,d_rpe]

        key_padding_mask = ~valid_agent  # [B,N]
        for layer in self.layers:
            node, edge_new = layer(node, edge, key_padding_mask)
            if edge_new is not None:
                edge = edge_new

        ego_embed = node[:, 0, :]  # ego는 항상 slot 0
        return self.classifier(ego_embed)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
