"""
STGCN-adapted — TSEM-SAGE 비교용 재구현
=========================================
원본: comparison/STGCN/ (hazdzz/STGCN, Yu et al. IJCAI 2018 "Spatio-Temporal Graph Convolutional
Networks: A Deep Learning Framework for Traffic Forecasting" 의 커뮤니티 재구현, LGPL-2.1)
  - model/layers.py::Align              -> TSEMAlign            (1x1 Conv2d 채널 정합, 수식 동일)
  - model/layers.py::CausalConv2d       -> TSEMCausalConv2d      (좌측 zero-pad 후 valid conv, 수식 동일.
                                            단 enable_padding=True로 고정 — 아래 "설계 결정 1" 참조)
  - model/layers.py::TemporalConvLayer  -> TSEMTemporalConvLayer (GLU 게이트, 수식 동일)
  - model/layers.py::GraphConv          -> TSEMGraphConv         (1st-order GCN 근사, 수식 동일.
                                            단 gso가 배치별로 다름 — 아래 "설계 결정 2" 참조)
  - model/layers.py::GraphConvLayer     -> TSEMGraphConvLayer    (Align 잔차 결합까지 동일)
  - model/layers.py::STConvBlock        -> TSEMSTConvBlock       ('TGTND' 구조 동일: 시간conv1→그래프conv→
                                            시간conv2→LayerNorm→Dropout)
  - model/models.py::STGCNGraphConv     -> STGCNTSEMAdapted      (아래 최상위 모듈)
  - script/utility.py::calc_gso(gso_type='sym_renorm_adj')
                                         -> _build_gso()          (D^-1/2(A+I)D^-1/2 정규화, 수식 동일)

이 환경(tna_research)에는 torch_geometric이 없지만, 원래 STGCN 자체가 pure PyTorch(einsum 기반
dense adjacency)라 애초에 그래프 라이브러리 의존성이 없다 — 5개 어댑터 중 유일하게 "PyG 대체"가
필요 없는 재구현이었다(CRAT-Pred는 CGConv만 PyG였고, HiVT/QCNet은 MessagePassing 전체가 PyG,
SIMPL/Forecast-MAE는 애초에 PyG 미사용).

Kt(시간 커널) / graph_conv_type: 원본 저장소는 ChebGraphConv(Ks차 체비셰프)와 1st-order GraphConv
두 방식을 모두 제공하는데, 이 어댑터는 GraphConv(1st-order GCN 근사, GCN(Kipf&Welling) 스타일)만
채택한다 — 원 논문(IJCAI 2018) 본문의 1st-order approximation 섹션이 이미 이 방식을 "충분히 좋고
더 가볍다"고 제안하며, N=7(ego+이웃6)의 초소형 그래프에서 Ks>=2 체비셰프 다항 확장은 사실상
GraphConv와 표현력 차이가 거의 없어 불필요한 파라미터만 늘린다(과적합 위험 ↑, ETSAGELayer/CGConv
등 나머지 baseline도 전부 1-hop만 쓰는 것과 일관성 유지).

주요 변경점(README.md 참조):
  - OutputBlock(다중 미래 시점 회귀, 'TNFF' 구조) 제거 — 분류 헤드(Linear→num_classes)로 교체.
    2-block 'TGTND TGTND' 스택까지는 원본과 동일하게 유지하고, 그 뒤에 바로 ego 노드 최종 시점
    표현을 뽑아 classifier에 넣는다(HiVT/CRAT-Pred와 동일한 "회귀 디코더 제거 → 분류 헤드" 패턴).
  - n_vertex(원본: 고정 도로/센서 그래프의 노드 수, 보통 수백~수천) -> N=7(ego+K=6 이웃)로 축소,
    샘플마다 다른 소규모 완전연결(마스킹 포함) 그래프를 배치 벡터화로 구성(CRAT-Pred/HiVT와 동일
    패턴 — `_build_graph`에서 anchor 시점 ego 위치·헤딩 기준 scene 1회 정렬).
  - node_dim: 원본은 센서당 스칼라 1개(교통 속도)뿐이었으나, 이 데이터는 raw 6D
    [pos_x,pos_z,speed,dir_x,dir_z,accel] 전 채널을 c_in으로 사용(HiVT와 동일 근거: 원본이 "센서당
    스칼라 1개"만 쓴 것 자체가 traffic-sensor 데이터의 한계였을 뿐, 원 논문 수식 자체는 c_in을
    임의 채널 수로 일반화하는 데 아무 제약이 없다 — 오히려 6D를 쓰는 게 원 논문 프레임워크를 있는
    그대로 더 풍부하게 활용하는 것이지 별도 설계 이탈이 아니다).

설계 결정 1 — CausalConv2d를 enable_padding=True로 고정(원본 기본값은 False):
  원본(TemporalConvLayer)은 항상 enable_padding=False로 CausalConv2d를 호출해 valid-conv로
  시간축을 블록마다 2*(Kt-1)씩 줄인다(원본의 n_his가 보통 12~48로 길어 여러 블록을 거치며 다운샘플
  하는 게 자연스러운 설계). 우리 W=10은 원본 대비 이미 훨씬 짧고(historical_steps가 원본의 1/3
  이하), 2-block 스택만으로 valid-conv를 쓰면 Kt=3 기준 T=10→6→2로 줄어 마지막 표현이 시간 정보를
  거의 다 잃는다. 그래서 CausalConv2d가 원래 제공하는 enable_padding=True 옵션(좌측만 zero-pad,
  `F.pad(x, (0,0,Kt-1,0))`)을 켜서 T=10을 전 구간 유지한다 — 이건 원본에 없는 코드를 새로 만든 게
  아니라 원본 클래스가 이미 지원하는 옵션 분기를 켠 것뿐이고(`model/layers.py::CausalConv2d`),
  오히려 인과성(causality)은 더 엄격해진다: 좌측 padding 방식은 "출력 시점 t가 입력 시점 <=t만
  본다"를 엄밀히 보장하는 표준 TCN(causal conv) 정의 그 자체인 반면, 원본의 valid-conv 방식은
  출력이 줄어들며 각 출력 위치가 자기보다 뒤 인덱스의 입력까지 함께 보는 로컬 윈도우 방식이라
  엄밀한 "출력 t는 입력 <=t만" 인과성은 아니다(전체 관측 구간 W 밖의 미래를 보진 않지만, 축소된
  각 출력 슬롯 단위로는 아니라는 뜻). 검증 스크립트(test_model.py #3)는 이 엄격한 causal 정의를
  기준으로 검증한다.

설계 결정 2 — 그래프 인접행렬: Gaussian 거리 커널(이진 인접 대신) 채택, 이유:
  원 논문(STGCN)이 실제로 쓰는 그래프도 "이진 연결 여부"가 아니라 **도로망 거리 기반 가중
  인접행렬**이다(원 논문 §3.1 "we construct the adjacency matrix ... a weighted graph based on the
  distance", `script/dataloader.py`가 로드하는 `PeMSD7 W_25/W_228.csv`도 실측 도로 거리를 가우시안
  커널로 변환한 값이지 0/1 이진 행렬이 아니다 — `gso_type='sym_renorm_adj'`가 정규화 방식이고
  edge weight 자체는 이미 가중치다). 즉 이 어댑터가 채택한 Gaussian 커널 가중치
  `A_ij = exp(-dist_ij^2 / sigma^2)`는 원 논문을 단순화한 게 아니라, 오히려 이 프로젝트의 다른
  4개 baseline(전부 이진 완전연결 그래프만 씀)보다 **원 논문의 실제 그래프 구성 방식에 더 충실한
  선택**이다 — 이 데이터셋엔 고정 도로망 거리表가 없으니 각 샘플의 실측 상대거리(두 차량 간
  유클리드 거리, anchor 시점)로 그 자리를 대신한다. sigma(=`adj_sigma`)는 `graph.radius`(이웃
  탐색 반경, 기본 20m)의 절반 근방으로 설정해 "반경 절반 거리에서 가중치 ~0.6, 반경 끝에서
  ~0.1" 정도로 완만하게 감쇠하도록 잡았다(원 논문 PeMSD7 세팅의 sigma 선택 관행과 동일한 방식 —
  "네트워크 규모에 맞춰 적당히 감쇠"를 그대로 재현).
  이후 정규화는 원본과 100% 동일한 `sym_renorm_adj`: Â = D^-1/2 (A+I) D^-1/2
  (`script/utility.py::calc_gso`).
"""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────
# model/layers.py 대응 재구현
# ────────────────────────────────────────────────────────────────
class TSEMAlign(nn.Module):
    """comparison/STGCN/model/layers.py::Align 과 동일 — 1x1 Conv2d로 채널 수만 맞춤."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.c_in > self.c_out:
            return self.align_conv(x)
        if self.c_in < self.c_out:
            b, _, t, v = x.shape
            pad = torch.zeros(b, self.c_out - self.c_in, t, v, device=x.device, dtype=x.dtype)
            return torch.cat([x, pad], dim=1)
        return x


class TSEMCausalConv2d(nn.Conv2d):
    """comparison/STGCN/model/layers.py::CausalConv2d 와 동일한 좌측 zero-pad 방식.
    이 어댑터는 항상 enable_padding=True로 사용(설계 결정 1, 위 모듈 docstring 참조) — 시간축
    kernel_size=(Kt,1)에서 시간축만 좌측으로 (Kt-1)만큼 pad, vertex축(kernel=1)은 pad 없음."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size, bias: bool = True):
        kernel_size = nn.modules.utils._pair(kernel_size)
        self._causal_padding = [kernel_size[0] - 1, kernel_size[1] - 1]
        super().__init__(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.pad 순서: (마지막축 좌, 마지막축 우, 끝에서2번째축 좌, 끝에서2번째축 우)
        # x: [B,C,T,V] -> V(마지막축) pad 없음, T(시간축) 좌측만 pad
        x = F.pad(x, (self._causal_padding[1], 0, self._causal_padding[0], 0))
        return super().forward(x)


class TSEMTemporalConvLayer(nn.Module):
    """comparison/STGCN/model/layers.py::TemporalConvLayer 와 동일 — GLU 게이트만 지원
    (원본은 gtu/relu/silu도 있으나 원 논문 §Fig.2/Table 2가 GLU를 기본·최고 성능으로 채택하므로
    이 어댑터도 GLU만 재구현). enable_padding=True 고정(T축 길이 보존, 설계 결정 1).

    구조(원본 주석 그대로 재현):
        |--------------------------------| * residual connection *
        |                                |
        |    |--->--- causal_conv ------ + -------|
    ----|----|                                    (x) ----->
             |--->--- causal_conv --- sigmoid -----|
    """

    def __init__(self, Kt: int, c_in: int, c_out: int):
        super().__init__()
        self.c_out = c_out
        self.align = TSEMAlign(c_in, c_out)
        self.causal_conv = TSEMCausalConv2d(c_in, 2 * c_out, kernel_size=(Kt, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.align(x)  # enable_padding=True라 T가 보존되므로 원본과 달리 슬라이싱 불필요
        x_causal = self.causal_conv(x)
        x_p = x_causal[:, :self.c_out]
        x_q = x_causal[:, self.c_out:]
        return (x_p + x_in) * torch.sigmoid(x_q)  # GLU


class TSEMGraphConv(nn.Module):
    """comparison/STGCN/model/layers.py::GraphConv(1st-order GCN 근사) 와 동일 수식.
    원본과의 유일한 차이: gso가 [V,V](전체 배치 공유, 고정 도로망)가 아니라 [B,V,V](샘플별 동적
    소규모 그래프) — einsum을 'hi,btij->bthj'에서 'bhi,btij->bthj'로만 바꿔 배치 차원을 추가."""

    def __init__(self, c_in: int, c_out: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(c_in, c_out))
        self.bias = nn.Parameter(torch.empty(c_out)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, gso: torch.Tensor) -> torch.Tensor:
        """x: [B,C,T,V], gso: [B,V,V] (D^-1/2(A+I)D^-1/2, 배치별). 원본과 동일하게
        x -> [B,T,V,C] 로 permute 후 vertex축(V)에 gso를 곱하고, 채널축(C)에 weight를 곱한다."""
        x = x.permute(0, 2, 3, 1)  # [B,T,V,C]
        first_mul = torch.einsum('bhi,btij->bthj', gso, x)       # 그래프 전파 (vertex축)
        second_mul = torch.einsum('bthi,ij->bthj', first_mul, self.weight)  # 채널 변환
        if self.bias is not None:
            second_mul = second_mul + self.bias
        return second_mul  # [B,T,V,C_out]


class TSEMGraphConvLayer(nn.Module):
    """comparison/STGCN/model/layers.py::GraphConvLayer 와 동일 — Align 잔차 결합까지 포함.
    graph_conv_type='graph_conv'(1st-order GCN)만 지원(설계: 위 모듈 docstring 참조)."""

    def __init__(self, c_in: int, c_out: int, bias: bool = True):
        super().__init__()
        self.align = TSEMAlign(c_in, c_out)
        self.graph_conv = TSEMGraphConv(c_out, c_out, bias)

    def forward(self, x: torch.Tensor, gso: torch.Tensor) -> torch.Tensor:
        x_in = self.align(x)  # [B,C,T,V]
        x_gc = self.graph_conv(x_in, gso)          # [B,T,V,C]
        x_gc = x_gc.permute(0, 3, 1, 2)             # [B,C,T,V]
        return x_gc + x_in


class TSEMSTConvBlock(nn.Module):
    """comparison/STGCN/model/layers.py::STConvBlock 와 동일한 'TGTND' 구조:
    시간conv1(GLU) -> 그래프conv(1st-order GCN) -> ReLU -> 시간conv2(GLU) -> LayerNorm -> Dropout."""

    def __init__(self, Kt: int, n_vertex: int, c_in: int, channels: List[int], droprate: float,
                 bias: bool = True):
        super().__init__()
        self.tmp_conv1 = TSEMTemporalConvLayer(Kt, c_in, channels[0])
        self.graph_conv = TSEMGraphConvLayer(channels[0], channels[1], bias)
        self.tmp_conv2 = TSEMTemporalConvLayer(Kt, channels[1], channels[2])
        self.ln = nn.LayerNorm([n_vertex, channels[2]], eps=1e-12)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x: torch.Tensor, gso: torch.Tensor) -> torch.Tensor:
        x = self.tmp_conv1(x)
        x = self.graph_conv(x, gso)
        x = self.relu(x)
        x = self.tmp_conv2(x)
        x = self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # LN은 (V,C) 축 기준(원본과 동일)
        x = self.dropout(x)
        return x


def _build_gso(centers: torch.Tensor, valid_agent: torch.Tensor, sigma: float) -> torch.Tensor:
    """샘플별 정규화 그래프 시프트 연산자(gso) 구성 — script/utility.py::calc_gso(gso_type=
    'sym_renorm_adj')와 동일한 수식: Â = D^-1/2 (A+I) D^-1/2.
    centers: [B,N,2] anchor 시점 위치(scene 정렬 후), valid_agent: [B,N] bool.
    A_ij(가중치, i!=j, 둘 다 valid일 때만) = exp(-dist_ij^2 / sigma^2) — 설계 결정 2 참조."""
    B, N, _ = centers.shape
    diff = centers.unsqueeze(2) - centers.unsqueeze(1)  # [B,N,N,2]
    dist2 = (diff ** 2).sum(-1)  # [B,N,N]
    A = torch.exp(-dist2 / (sigma ** 2))
    eye_mask = torch.eye(N, device=centers.device, dtype=torch.bool).unsqueeze(0)
    A = A.masked_fill(eye_mask, 0.0)  # 자기자신 가중치는 아래서 renorm 시 I로 별도 추가
    pair_valid = valid_agent.unsqueeze(2) & valid_agent.unsqueeze(1)  # [B,N,N]
    A = A * pair_valid.float()

    eye = torch.eye(N, device=centers.device, dtype=centers.dtype).unsqueeze(0).expand(B, -1, -1)
    A_hat = A + eye  # sym_renorm_adj: adj = adj + I (원본 calc_gso와 동일)

    deg = A_hat.sum(-1)  # [B,N]
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt = torch.where(torch.isinf(deg_inv_sqrt), torch.zeros_like(deg_inv_sqrt), deg_inv_sqrt)
    gso = deg_inv_sqrt.unsqueeze(-1) * A_hat * deg_inv_sqrt.unsqueeze(-2)  # D^-.5 A_hat D^-.5
    return gso


class STGCNTSEMAdapted(nn.Module):
    """STGCN 어댑터 최상위 모듈 — batch dict(TSEM dataloader 그대로) -> logits[B,num_classes].

    파이프라인: scene 1회 정렬(ego anchor 위치·헤딩 기준) -> 샘플별 gso 구성(Gaussian 거리 커널
    + sym_renorm_adj) -> TSEMSTConvBlock x2('TGTND TGTND', T=W 유지) -> ego 노드(slot 0)의 anchor
    시점(t=-1) 표현 추출 -> 분류 헤드.
    """

    def __init__(self, W: int = 10, K: int = 6, Kt: int = 3,
                 block_channels: List[List[int]] = None, droprate: float = 0.1,
                 adj_sigma: float = 10.0, num_classes: int = 3, bias: bool = True):
        super().__init__()
        self.W = W
        self.K = K
        self.N = 1 + K
        self.Kt = Kt
        self.adj_sigma = adj_sigma
        if block_channels is None:
            block_channels = [[16, 16, 16], [16, 16, 16]]
        self.block_channels = block_channels

        c_in = 6  # raw node dim [pos_x,pos_z,speed,dir_x,dir_z,accel] — 설계 결정: 모듈 docstring 참조
        blocks = nn.ModuleList()
        last_c = c_in
        for ch in block_channels:
            blocks.append(TSEMSTConvBlock(Kt, self.N, last_c, ch, droprate, bias))
            last_c = ch[-1]
        self.st_blocks = blocks
        self.final_channels = last_c
        self.classifier = nn.Linear(self.final_channels, num_classes)

    def _build_graph(self, batch: dict):
        """CRAT-Pred/HiVT-adapted와 동일한 scene 1회 정렬(ego anchor 위치+헤딩 기준 center+rotate).
        반환: x_seq[B,6,W,N](conv2d 입력 규격), centers[B,N,2](anchor 시점, gso 구성용),
        valid_agent[B,N]."""
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
        pos_centered = pos_raw - origin_flat.unsqueeze(1)
        rot_expand = rot_flat.unsqueeze(1).expand(-1, W, -1, -1).reshape(-1, 2, 2)
        pos = torch.bmm(pos_centered.reshape(-1, 1, 2), rot_expand).reshape(B * N, W, 2)
        dir_rot = torch.bmm(dir_raw.reshape(-1, 1, 2), rot_expand).reshape(B * N, W, 2)

        present_flat = present.reshape(B * N, W)
        pos = torch.where(present_flat.unsqueeze(-1), pos, torch.zeros_like(pos))
        dir_rot = torch.where(present_flat.unsqueeze(-1), dir_rot, torch.zeros_like(dir_rot))
        speed = torch.where(present_flat, raw[..., 2].reshape(B * N, W), torch.zeros(1, device=device))
        accel = torch.where(present_flat, raw[..., 5].reshape(B * N, W), torch.zeros(1, device=device))

        x_seq = torch.cat([pos, speed.unsqueeze(-1), dir_rot, accel.unsqueeze(-1)], dim=-1)  # [B*N,W,6]
        x_seq = x_seq.reshape(B, N, W, 6)
        # 무효 노드(node_mask=0)는 명시적으로 0으로 고정 — TSEMSTConvBlock의 LayerNorm이
        # `[n_vertex, channels]` 축 전체를 함께 정규화하는 원본(STGCN) 구조라(script/layers.py::
        # STConvBlock, self.ln = LayerNorm([n_vertex, channels[2]])), 그래프conv 단계에서 gso로
        # 무효 노드의 "기여"는 이미 차단되더라도(_build_gso 참조) 무효 노드 자신의 raw 값이 이
        # LayerNorm 통계(평균/분산)에는 여전히 섞여 들어가 valid 노드 출력에 미세하게 새어나갈 수
        # 있다(원본은 traffic 센서가 항상 전부 존재한다고 가정해 이 문제가 없었음). 여기서 값 자체를
        # 0으로 고정해 "무효 슬롯에 어떤 값이 들어있든 항상 동일한 상수(0)"가 되게 만들어 이 경로의
        # 정보 누수를 원천 차단한다(test_model.py::test_invalid_neighbor_no_influence_on_output 참조).
        x_seq = x_seq * valid_agent.view(B, N, 1, 1).to(x_seq.dtype)
        x_seq = x_seq.permute(0, 3, 2, 1).contiguous()  # [B,6,W,N] (conv2d 규격)

        centers = pos.reshape(B, N, W, 2)[:, :, -1, :]  # anchor 시점 위치 [B,N,2]
        return x_seq, centers, valid_agent

    def forward(self, batch: dict) -> torch.Tensor:
        x_seq, centers, valid_agent = self._build_graph(batch)
        gso = _build_gso(centers, valid_agent, self.adj_sigma)  # [B,N,N]

        x = x_seq
        for block in self.st_blocks:
            x = block(x, gso)  # [B,C,W,N] (enable_padding=True라 W 유지)

        ego_final = x[:, :, -1, 0]  # [B,C] — ego(slot0) anchor 시점(t=-1) 표현
        return self.classifier(ego_final)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
