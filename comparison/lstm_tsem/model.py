"""
LSTM-adapted — TSEM-SAGE 비교용 순수 아키텍처 대조군 (원본 GitHub repo 없음)
========================================================================
`comparison/README.md` §"5개 어댑터"가 다루는 HiVT/QCNet/CRAT-Pred/SIMPL/Forecast-MAE는 전부
**학습되는 공간/그래프 상호작용 모듈**(attention, CGConv, SftLayer 등)을 갖는다. 이 6번째
baseline은 그 반대 극단이다 — "시퀀스 인코더는 있지만 상호작용을 학습하는 모듈이 전혀 없을 때
얼마나 되는가"를 재는 순수 대조군이라, 대응할 원 논문/원본 구현이 없다(직접 설계).

**이 baseline이 검증하는 축**: 위 5개 baseline이 전부 이기는 이유가 "상호작용 모듈 자체" 때문인지,
아니면 단순히 "시간축 인코딩 + ego 주변 정보를 어떻게든 합쳤기 때문"인지를 분리한다. 이 모델이
5개보다 뚜렷이 떨어지면 학습되는 상호작용 모듈에 실질적 가치가 있다는 근거가 되고, 큰 차이가
없다면 이 데이터셋(ego+1-hop, K=6, 완전관측 이웃)에서는 상호작용 모듈 자체보다 "이웃이 있다는
사실"만으로 대부분의 이득이 난다는 뜻이 된다.

**설계 (comparison/README.md §"5개 baseline 전부 ego-anchor 정규화" 관행을 그대로 따름 — 6번째도
예외 없음)**:
  - **노드 입력 6D 전체** — HiVT-adapted와 동일하게 raw 6D 전 채널
    `[Δpos_x,Δpos_z(ego-anchor 기준 정렬 후 프레임간 변위), speed, dir_x, dir_z, accel]`을 그대로
    사용해 "baseline에게 정보량을 덜 준다"는 오해를 없앤다(CRAT-Pred/SIMPL처럼 3D로 의도적으로
    줄이는 것과 다른 선택 — 이 모델의 핵심 비교 포인트는 "입력 정보량"이 아니라 "상호작용 모듈
    유무"이므로 입력은 최대치로 맞춘다).
  - **scene 1회 정렬** — `comparison/cratpred_tsem/model.py::CratPredTSEMAdapted._build_graph`와
    동일한 방식(ego 마지막 프레임 위치·헤딩 기준 회전+평행이동)을 그대로 재사용. 위치쌍[Δx,Δz]과
    방향쌍[dir_x,dir_z]만 회전하고 speed·accel은 스칼라라 그대로 통과(HiVT-adapted의 `rotate6`과
    동일 규약).
  - **시간 인코더** — ego와 각 이웃에 **동일 가중치를 공유하는 단일 `nn.LSTM`**
    (`TSEMEncoderLstm6`, `CratPredTSEMAdapted.TSEMEncoderLstm`과 동일 구조, input_size만 3→6).
    배치×에이전트(B*N)를 합쳐 한 번에 forward하고 마지막 timestep hidden state를 에이전트
    임베딩으로 쓴다.
  - **공간 처리: 의도적으로 없음.** 그래프도, attention도, GNN도 쓰지 않는다 — 유효한 이웃
    (`nbr_mask`)의 임베딩을 단순 **masked mean-pooling**으로 합친다. 이게 이 baseline의 핵심
    포인트다: 학습되는 파라미터가 전혀 없는 pooling으로 "상호작용을 어떻게든 요약"할 뿐, 이웃들
    사이의 관계(거리·상대속도 등)나 ego와의 관계를 학습하는 가중치가 없다.
  - **분류 헤드** — `concat([ego_embed, pooled_nbr_embed])` → 2-layer MLP → `num_classes`.
  - **회귀 디코더 자체가 없음** — 원래부터 만들 필요 없음(다른 5개는 원 논문의 다중모달 회귀
    디코더를 분류 헤드로 교체하지만, 이 baseline은 애초에 자체 설계라 회귀 디코더가 없었다).

데이터로더·loss·스케줄러·평가는 `train_tsem.py`/`modules/tsem_eval.py`와 100% 동일
(`train_lstm_tsem.py`가 5개 baseline의 학습 스크립트와 동일 패턴으로 재사용).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate2(vec: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """vec: [N,2], rot: [N,2,2] -> [N,2]. comparison/hivt_tsem/model.py::rotate2와 동일."""
    return torch.bmm(vec.unsqueeze(-2), rot).squeeze(-2)


class TSEMEncoderLstm6(nn.Module):
    """comparison/cratpred_tsem/model.py::TSEMEncoderLstm과 동일 구조 — input_size만 3→6
    (valid flag 대신 speed,dir_x,dir_z,accel 전 채널 사용). 에이전트 종류(ego/이웃) 구분 없이
    동일 가중치를 공유하는 단일 LSTM."""

    def __init__(self, latent_size: int, input_size: int = 6):
        super().__init__()
        self.latent_size = latent_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=latent_size, num_layers=1, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [num_agents, T, input_size] -> [num_agents, latent_size] (마지막 timestep hidden state)"""
        h0 = torch.zeros(1, x.size(0), self.latent_size, device=x.device, dtype=x.dtype)
        c0 = torch.zeros(1, x.size(0), self.latent_size, device=x.device, dtype=x.dtype)
        out, _ = self.lstm(x, (h0, c0))
        return out[:, -1, :]


class LstmTSEMAdapted(nn.Module):
    """LSTM baseline 어댑터 최상위 모듈 — batch dict(TSEM dataloader 그대로) -> logits[B,num_classes].

    파이프라인: TSEMEncoderLstm6(ego/이웃 공유 가중치) -> masked mean-pooling(학습 파라미터 없음,
    공간/그래프 상호작용 모듈 의도적 부재) -> concat(ego, pooled_nbr) -> MLP 분류 헤드.
    """

    def __init__(self, W: int = 10, K: int = 6, latent_size: int = 64, hidden_size: int = 64,
                 num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.W = W
        self.K = K
        self.N = 1 + K
        self.latent_size = latent_size

        self.encoder_lstm = TSEMEncoderLstm6(latent_size, input_size=6)
        self.classifier = nn.Sequential(
            nn.Linear(2 * latent_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def _build_inputs(self, batch: dict):
        """CratPredTSEMAdapted._build_graph와 동일한 scene 1회 정렬(ego anchor 위치·헤딩 기준
        회전+평행이동)이지만, 3D displ(Δx,Δy,valid) 대신 6D 전 채널을 반환한다는 점만 다르다."""
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

        Ntot = B * N
        raw_flat = raw.reshape(Ntot, W, 6)
        present_flat = present.reshape(Ntot, W)

        pos_raw = raw_flat[..., 0:2]
        dir_raw = raw_flat[..., 3:5]
        pos_centered = pos_raw - origin_flat.unsqueeze(1)
        rot_expand = rot_flat.unsqueeze(1).expand(-1, W, -1, -1).reshape(-1, 2, 2)
        pos = torch.bmm(pos_centered.reshape(-1, 1, 2), rot_expand).reshape(Ntot, W, 2)
        dir_rot = torch.bmm(dir_raw.reshape(-1, 1, 2), rot_expand).reshape(Ntot, W, 2)
        pos = torch.where(present_flat.unsqueeze(-1), pos, torch.zeros_like(pos))
        dir_rot = torch.where(present_flat.unsqueeze(-1), dir_rot, torch.zeros_like(dir_rot))

        speed = torch.where(present_flat, raw_flat[..., 2], torch.zeros_like(raw_flat[..., 2]))
        accel = torch.where(present_flat, raw_flat[..., 5], torch.zeros_like(raw_flat[..., 5]))

        # 프레임간 변위(원 위치 그대로가 아니라 Δ) — CRAT-Pred/HiVT-adapted와 동일 관례,
        # 두 프레임 다 유효할 때만 diff, 그 외엔 0
        both_valid = present_flat[:, 1:] & present_flat[:, :-1]
        pos_disp = torch.zeros_like(pos)
        pos_disp[:, 1:] = torch.where(both_valid.unsqueeze(-1), pos[:, 1:] - pos[:, :-1],
                                       torch.zeros_like(pos[:, 1:]))

        x_seq = torch.cat(
            [pos_disp, speed.unsqueeze(-1), dir_rot, accel.unsqueeze(-1)], dim=-1
        )  # [B*N, W, 6]

        return x_seq, valid_agent

    def forward(self, batch: dict) -> torch.Tensor:
        x_seq, valid_agent = self._build_inputs(batch)
        B, N = valid_agent.shape

        h = self.encoder_lstm(x_seq)  # [B*N, D]
        h = h.view(B, N, self.latent_size)

        ego_embed = h[:, 0, :]  # ego는 항상 slot 0

        nbr_h = h[:, 1:, :]  # [B, K, D]
        nbr_valid = valid_agent[:, 1:].float()  # [B, K]
        denom = nbr_valid.sum(dim=1, keepdim=True).clamp(min=1.0)  # 이웃 0명이어도 NaN 방지
        pooled_nbr = (nbr_h * nbr_valid.unsqueeze(-1)).sum(dim=1) / denom  # [B, D]

        combined = torch.cat([ego_embed, pooled_nbr], dim=-1)  # [B, 2D]
        return self.classifier(combined)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
