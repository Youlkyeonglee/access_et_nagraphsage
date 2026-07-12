"""
DCRNN-adapted — TSEM-SAGE 비교용 재구현
=========================================
원본: comparison/DCRNN/ (chnsh/DCRNN_PyTorch — Li et al., ICLR 2018 "Diffusion Convolutional
Recurrent Neural Network"의 PyTorch 포트. 공식 원저자 구현은 TensorFlow(liyaguang/DCRNN)라
PyTorch 재구현이 쉬운 이 포트를 clone했다. LICENSE는 원본 그대로 보존.)
  - model/pytorch/dcrnn_cell.py::DCGRUCell._gconv   -> TSEMDCGRUCell._diffuse (아래 재구현 —
                                                        torch.sparse.mm 기반 원본을 이 프로젝트의
                                                        소규모 dense 그래프(N=7)에 맞춰 dense
                                                        bmm 기반으로 재구현)
  - model/pytorch/dcrnn_cell.py::DCGRUCell.forward  -> TSEMDCGRUCell.forward (GRU 게이트 수식은
                                                        동일, gconv만 교체)
  - model/pytorch/dcrnn_model.py::EncoderModel      -> TSEMDCRNNEncoder (인코더 스택만 재사용,
                                                        디코더는 아래 참조)
  - model/pytorch/dcrnn_model.py::DecoderModel      -> 미사용(seq2seq 다중스텝 회귀 전용) —
                                                        분류 헤드로 교체
  - lib/utils.py::calculate_random_walk_matrix      -> _build_diffusion_supports (아래 재구현 —
                                                        scipy sparse 대신 dense torch 연산)

## 핵심 설계 차이 (원본 대비 무엇을 바꿨는지, 왜)

**1. sparse -> dense.** 원본 DCRNN은 교통 센서 그래프(207~325개 노드)를 대상으로 하기 때문에
`torch.sparse.mm` + scipy 희소행렬로 인접행렬을 다룬다. 우리 그래프는 ego+1-hop 이웃(K=6) =
N=7 고정 크기의 매우 작은 dense 그래프라, `torch.sparse` 없이 `torch.bmm`으로 직접 P_f^k, P_b^k를
반복 행렬곱하는 게 더 단순하고 빠르다 — CRAT-Pred/HiVT-adapted가 `torch_geometric` 없이
`scatter_add_`로 재구현한 것과 같은 이유(이 환경에 `torch_geometric`이 없다는 제약과는 별개로,
그래프 규모 자체가 dense가 자연스러운 크기라는 점도 있다).

**2. Chebyshev 재귀 대신 원 논문 Eq.(2)의 직접적인 P^k 정의를 그대로 구현.** 원본
`dcrnn_cell.py::_gconv`는 `filter_type`(laplacian/random_walk/dual_random_walk) 관계없이 항상
`T_k = 2·P·T_{k-1} - T_{k-2}`(Chebyshev 다항식 재귀, `_concat` 참조)를 사용한다. 이 재귀는
`laplacian`(고유값을 [-1,1]로 스케일)일 때만 정확히 Chebyshev 근사가 되고, `random_walk`/
`dual_random_walk`(스케일 없는 전이행렬 P)에 그대로 적용하면 원 논문 Eq.(2)의
`sum_{k=0}^{K} theta_k P^k x`와 정확히 같지 않다(원본 저장소 자체의 알려진 근사/관행이며 버그는
아니지만, 우리는 이 프로젝트 CLAUDE.md 지침대로 "N=7이라 P^k를 매 스텝 반복 행렬곱(dense)으로
계산"하는 **원 논문 수식 그대로**(`P_f^k`, `P_b^k`를 진짜 k번 행렬곱)를 채택했다 — 소규모
그래프라 Chebyshev 근사로 계산량을 줄일 필요가 없고, 논문 정의에 더 충실하다.

**3. 인접행렬 구성.** 원본은 도로망의 실측 거리 기반 Gaussian kernel(`exp(-dist^2/sigma^2)`,
임계값 이하는 0)로 정적 인접행렬을 한 번 만들어 전체 학습에 재사용한다(도로망 자체가 고정이므로).
우리는 매 샘플(scene)마다 에이전트 구성이 달라지므로, **anchor 시점(윈도우 마지막 프레임, t)의
차량 간 유클리드 거리**로 동일한 Gaussian kernel을 샘플별로 구성한다(CRAT-Pred-adapted의
"scene 1회 정렬" 패턴과 동일하게, 그래프는 anchor 시점 1회만 구성하고 W개 타임스텝 전체에
재사용 — 원본 DCRNN의 "정적 그래프, 동적 노드 신호"라는 설계 그대로). sigma는 배치 내 유효
쌍별 거리의 표준편차로 매 배치 적응적으로 정한다(원본은 데이터셋 전체 통계로 고정값을 쓰지만,
우리는 배치마다 에이전트 수·거리 분포가 다양해 적응적 스케일이 더 안정적). 무효 에이전트
(`nbr_mask=0`)는 인접행렬에서 행/열 전부 0으로 마스킹한다.
  - 이 그래프가 (커널 자체가 대칭이라) 사실상 무방향이므로 forward/backward 전이행렬
    `P_f=D_O^-1 A`, `P_b=D_I^-1 A^T`이 수치적으로 같아지지만, 원 설계(양방향 확산, 각 방향마다
    독립된 학습 가능 가중치)는 그대로 유지한다 — 향후 방향성 있는 인접행렬(예: 차선 진행방향
    기준 비대칭 그래프)로 바꿀 여지를 남겨둔다.

**4. seq2seq -> 인코더 전용.** 원본은 인코더(과거 관측 압축) -> 디코더(미래 다중스텝 좌표 회귀,
scheduled sampling 포함)의 seq2seq 구조다. 우리 과제는 3클래스 분류(H스텝 뒤 단일 라벨)라
디코더가 필요 없다 — 인코더만 사용해 W프레임을 순차 처리한 뒤 마지막 hidden state 중 ego
노드(slot 0) 표현을 분류 헤드(Linear->num_classes)로 보낸다(다른 5개 어댑터와 동일한
공통 패턴: "회귀 디코더 제거 -> 분류 헤드 교체").

**5. 노드 입력 6D.** 원 논문은 센서(도로 구간)당 스칼라 값(교통량/속도) 하나만 입력으로 쓰지만,
HiVT-adapted와 동일한 이유로 우리 데이터의 원시 6D 운동학 벡터
(`pos_x,pos_z,speed,dir_x,dir_z,accel`)를 전 채널 사용한다 — "baseline이 정보를 적게 받아서
불리하다"는 혼입을 피하기 위해서다. scene 1회 정렬(ego anchor 헤딩 기준 회전)은 위치쌍
`[pos_x,pos_z]`와 방향쌍 `[dir_x,dir_z]`에만 적용하고 speed·accel은 회전과 무관한 스칼라라
그대로 통과시킨다(CRAT-Pred/HiVT-adapted와 동일 규약).

**6. 2-hop 이웃 미사용.** ego+1-hop(K=6)의 diffusion convolution 자체가 K-hop 확산으로 이미
그래프 전체를 커버하므로(K_diffusion>=2면 1-hop 그래프 안에서도 2번 이상 확산됨), 별도의 2-hop
원시 이웃 데이터는 필요 없다 — 나머지 4개 어댑터와 동일한 근거.

**7. num_rnn_layers, filter_type='dual_random_walk' 유지** — 원 논문 기본 설정(양방향 확산)을
그대로 채택.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────
# 그래프 구성: anchor 시점 위치로 Gaussian kernel 인접행렬 -> P_f, P_b
# ────────────────────────────────────────────────────────────────
def build_diffusion_supports(
    pos_anchor: torch.Tensor,      # [B, N, 2]
    valid_agent: torch.Tensor,     # [B, N] bool
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """anchor 시점(윈도우 마지막 프레임) 위치로 Gaussian kernel 인접행렬 A를 만들고,
    원 논문 정의 그대로 forward P_f=D_O^-1 A, backward P_b=D_I^-1 A^T 전이행렬을 반환.

    원본(`lib/utils.py::calculate_random_walk_matrix`)은 도로망 실측 거리로 고정된 A를 데이터셋
    전체에서 한 번 계산하지만, 우리는 샘플(scene)마다 에이전트 구성이 달라 anchor 시점 상대
    거리로 배치마다 A를 새로 만든다 — CRAT-Pred-adapted의 "scene 1회 정렬"과 같은 지점(anchor
    시점 1회)에서 그래프를 구성하고 W개 타임스텝 전체에 재사용한다(원본의 "정적 그래프, 동적
    신호" 설계를 그대로 유지).
    """
    B, N, _ = pos_anchor.shape
    diff = pos_anchor.unsqueeze(2) - pos_anchor.unsqueeze(1)      # [B,N,N,2]
    dist = diff.norm(dim=-1)                                      # [B,N,N]

    valid_pair = valid_agent.unsqueeze(2) & valid_agent.unsqueeze(1)  # [B,N,N]
    eye = torch.eye(N, device=pos_anchor.device, dtype=torch.bool).unsqueeze(0)
    off_diag = valid_pair & (~eye)

    # 배치별 적응 sigma — 유효 쌍이 거의 없으면(고립 그래프) 1.0으로 안전하게 폴백
    masked_dist = torch.where(off_diag, dist, torch.zeros_like(dist))
    denom = off_diag.float().sum(dim=(1, 2)).clamp(min=1.0)
    mean_d = masked_dist.sum(dim=(1, 2)) / denom
    var_d = ((masked_dist - mean_d.view(-1, 1, 1)) ** 2 * off_diag.float()).sum(dim=(1, 2)) / denom
    sigma = var_d.clamp(min=eps).sqrt().clamp(min=1.0)  # [B]

    A = torch.exp(-(dist ** 2) / (sigma.view(-1, 1, 1) ** 2))
    A = torch.where(off_diag, A, torch.zeros_like(A))

    d_out = A.sum(dim=2, keepdim=True).clamp(min=eps)   # [B,N,1]
    P_f = A / d_out                                      # D_O^-1 A

    A_t = A.transpose(1, 2)
    d_in = A_t.sum(dim=2, keepdim=True).clamp(min=eps)   # [B,N,1] (열합 = 노드의 in-degree)
    P_b = A_t / d_in                                      # D_I^-1 A^T

    return P_f, P_b


class TSEMDCGRUCell(nn.Module):
    """comparison/DCRNN/model/pytorch/dcrnn_cell.py::DCGRUCell 대응.
    gconv(diffusion convolution)만 dense P^k 반복행렬곱으로 재구현, GRU 게이트 수식(r,u,c,
    new_state=u*h+(1-u)*c)은 원본과 100% 동일하게 유지."""

    def __init__(self, input_dim: int, hidden_dim: int, K: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.K = K
        num_matrices = 1 + 2 * K  # x 자신 + forward K-hop + backward K-hop (K=0이면 x 자신뿐)
        in_size = (input_dim + hidden_dim) * num_matrices
        self.gate_lin = nn.Linear(in_size, 2 * hidden_dim)   # r,u 동시 계산 (원본과 동일)
        self.cand_lin = nn.Linear(in_size, hidden_dim)       # candidate c

    def _diffuse(self, z: torch.Tensor, P_f: torch.Tensor, P_b: torch.Tensor) -> torch.Tensor:
        """z: [B,N,D] -> [B,N,D*(1+2K)] = concat([z, P_f z, P_f^2 z, ..., P_b z, P_b^2 z, ...]).
        원 논문 Eq.(2) sum_k theta_k P^k z 를, "채널을 이어붙인 뒤 하나의 Linear로 project"하는
        방식으로 구현 — concat 후 Linear는 각 P^k 항에 서로 다른 부분행렬(=서로 다른 theta_k)을
        곱해 더하는 것과 수학적으로 동치다(원본 `_gconv`와 동일한 구현 관용구)."""
        feats: List[torch.Tensor] = [z]
        if self.K > 0:
            x = z
            for _ in range(self.K):
                x = torch.bmm(P_f, x)
                feats.append(x)
            x = z
            for _ in range(self.K):
                x = torch.bmm(P_b, x)
                feats.append(x)
        return torch.cat(feats, dim=-1)

    def forward(self, x: torch.Tensor, h: torch.Tensor,
                P_f: torch.Tensor, P_b: torch.Tensor) -> torch.Tensor:
        """x: [B,N,input_dim], h: [B,N,hidden_dim] -> new_h: [B,N,hidden_dim]"""
        xh = torch.cat([x, h], dim=-1)
        diffused = self._diffuse(xh, P_f, P_b)
        ru = torch.sigmoid(self.gate_lin(diffused))
        r, u = ru.chunk(2, dim=-1)

        xh_r = torch.cat([x, r * h], dim=-1)
        diffused_c = self._diffuse(xh_r, P_f, P_b)
        c = torch.tanh(self.cand_lin(diffused_c))

        new_h = u * h + (1.0 - u) * c
        return new_h


class TSEMDCRNNEncoder(nn.Module):
    """comparison/DCRNN/model/pytorch/dcrnn_model.py::EncoderModel 대응 — DCGRUCell을
    num_layers만큼 쌓아 W프레임을 순차 처리. 원본과 달리 seq2seq 디코더는 없음(인코더 전용)."""

    def __init__(self, input_dim: int, hidden_dim: int, K: int, num_layers: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cells = nn.ModuleList([
            TSEMDCGRUCell(input_dim if i == 0 else hidden_dim, hidden_dim, K)
            for i in range(num_layers)
        ])

    def forward(self, x_seq: torch.Tensor, P_f: torch.Tensor, P_b: torch.Tensor) -> torch.Tensor:
        """x_seq: [B,N,W,input_dim] -> 마지막 layer의 마지막 timestep hidden [B,N,hidden_dim]"""
        B, N, W, _ = x_seq.shape
        h = [torch.zeros(B, N, self.hidden_dim, device=x_seq.device, dtype=x_seq.dtype)
             for _ in range(self.num_layers)]
        for t in range(W):
            inp = x_seq[:, :, t, :]
            for l, cell in enumerate(self.cells):
                h[l] = cell(inp, h[l], P_f, P_b)
                inp = h[l]
        return h[-1]


class DCRNNTSEMAdapted(nn.Module):
    """DCRNN 어댑터 최상위 모듈 — batch dict(TSEM dataloader 그대로) -> logits[B,num_classes].

    파이프라인: scene 1회 정렬(ego anchor 헤딩 기준 회전) -> anchor 시점 위치로 P_f,P_b 구성
    -> TSEMDCRNNEncoder(DCGRU num_layers층, W프레임 순차) -> ego 노드(slot 0) 추출 -> 분류 헤드.
    """

    def __init__(self, W: int = 10, K_nbr: int = 6, hidden_dim: int = 64, K_diffusion: int = 2,
                 num_layers: int = 2, num_classes: int = 3):
        super().__init__()
        self.W = W
        self.K_nbr = K_nbr
        self.N = 1 + K_nbr
        self.hidden_dim = hidden_dim
        self.node_dim = 6  # [pos_x,pos_z,speed,dir_x,dir_z,accel]

        self.encoder = TSEMDCRNNEncoder(self.node_dim, hidden_dim, K_diffusion, num_layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)

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
            valid_agent = raw.abs().sum(dim=(-1, -2)) > 0

        present = raw.abs().sum(dim=-1) > 0  # [B,N,W]

        # --- scene 1회 정렬(anchor=윈도우 마지막 프레임 ego 위치·헤딩 기준 회전, CRAT-Pred/HiVT와 동일) ---
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

        dir_raw = raw[..., 3:5].reshape(B * N, W, 2)
        dir_rot = torch.bmm(dir_raw.reshape(-1, 1, 2),
                            rot_flat.unsqueeze(1).expand(-1, W, -1, -1).reshape(-1, 2, 2)).reshape(B * N, W, 2)

        present_flat = present.reshape(B * N, W)
        pos = torch.where(present_flat.unsqueeze(-1), pos, torch.zeros_like(pos))
        dir_rot = torch.where(present_flat.unsqueeze(-1), dir_rot, torch.zeros_like(dir_rot))

        speed = raw[..., 2:3].reshape(B * N, W, 1)
        accel = raw[..., 5:6].reshape(B * N, W, 1)
        speed = torch.where(present_flat.unsqueeze(-1), speed, torch.zeros_like(speed))
        accel = torch.where(present_flat.unsqueeze(-1), accel, torch.zeros_like(accel))

        x_seq = torch.cat([pos, speed, dir_rot, accel], dim=-1)  # [B*N,W,6]
        x_seq = x_seq.reshape(B, N, W, self.node_dim)

        pos_anchor = x_seq[:, :, -1, 0:2]  # [B,N,2] — anchor 시점(window 마지막 프레임) 위치
        P_f, P_b = build_diffusion_supports(pos_anchor, valid_agent)

        return x_seq, valid_agent, P_f, P_b

    def forward(self, batch: dict) -> torch.Tensor:
        x_seq, valid_agent, P_f, P_b = self._build_graph(batch)
        h_final = self.encoder(x_seq, P_f, P_b)   # [B,N,hidden_dim]
        ego_embed = h_final[:, 0, :]               # ego는 항상 slot 0
        return self.classifier(ego_embed)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
