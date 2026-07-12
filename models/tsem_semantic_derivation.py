"""
TSEM Stage A — maneuver semantic 채널 (world 좌표, §5.3)
========================================================
raw 6D [pos_x, pos_z, speed, dir_x, dir_z, accel] → semantic 8D per frame:
  종방향: v, a, j
  횡방향: ω, d_lat(초기 heading 기준 횡변위), κ
  로터리 극좌표 (2026-07-07 추가, world 6채널을 대체 아닌 추가):
    Δρ  — 로터리 중심 대비 반경 변화율. lane_change가 point 정의로는 8.5배 과소포착됐던 것과 달리,
          Δρ는 차선변경 순간 실측으로 lane_change=0.585 vs normal=0.349로 분리력 확인됨.
    접선 — dθ·ρ (각속도 × 반경, 원운동 공식). 로터리를 도는 속도 성분을 지도 좌표로 분해한 값.
          journal/augment_test.py에서 world에 [ρ,Δρ,접선,|Δρ|] augment 시 AUC 유의미 개선 확인
          (탐지 +0.015, 예측 +0.010, p<0.001) — 단 '대체'가 아니라 '추가'일 때만.
  로터리 중심 C=(72.86, -13.45)는 journal/map_features.py::build_map_world()로 4개 CSV에서
  독립 계산해 오차 <0.02m로 검증된 고정 상수 (HD 차선지도 기반, 공업탑 전용).

  variant='invariant' (2026-07-11, comparison/ baseline 5개와의 공정 비교용 신설):
  Δρ·접선 2채널은 로터리 중심(world 고정 상수)을 참조해야 계산 가능한 site-specific 신호임을
  코드 검사로 확인(comparison/README.md "semantic 8D 채널별 site-specific 여부 재점검" 참조).
  나머지 6채널(v,a,j,ω,d_lat,κ)은 자기 궤적의 시간 미분/차분만 사용하는 순수 운동학이라 baseline
  5개(HiVT/QCNet/CRAT-Pred/SIMPL/Forecast-MAE)의 ego-relative 불변성과 동급이다. variant='invariant'는
  이 6채널만 반환 — raw_append='none'과 함께 쓰면(TSEMSAGE) 랜드마크 참조 신호가 전혀 없는 완전
  ego/rotation-invariant 입력이 되어 baseline과 apples-to-apples 비교가 가능하다.

  variant='position_only' (2026-07-12, 위치 암기 가설 검증② 전용):
  운동학 유도채널(v,a,j,ω,d_lat,κ)도 랜드마크 참조채널(Δρ,접선)도 전혀 계산하지 않고
  raw world 좌표 [pos_x,pos_z] 그대로 2D만 반환한다. 속도·방향·가속도 정보를 완전히 제거해
  "궤적이 어떻게 움직였는가"는 모델이 전혀 볼 수 없고 "지금 세계좌표 어디인가"만 남는다.
  TSEMSAGE(semantic_variant='position_only', raw_append='none')로 감싸면, 나머지 아키텍처
  (시간 인코더+공간 GNN)는 10차-2 최종모델과 동일하게 유지한 채 입력만 2D로 제한한
  "position-only 신경망" ablation이 된다 — journal/position_memorization_entropy.py의 정적
  좌표 룩업(Persist baseline 수준, ~64%)보다 신경망이 궤적형태까지 학습해 더 높은 성능을
  내는지 확인하는 게 목적(CLAUDE.md "위치 암기 가설" §1 참조).

  variant='invariant_rot' (2026-07-12, §실험현황 "20260712 추가 실험계획" ① 전용):
  variant='invariant' 6채널(v,a,j,ω,d_lat,κ)에 "윈도우 시작 시점(h0) 대비 지금까지 얼마나
  돌았는가"를 나타내는 cos(heading_t-h0), sin(heading_t-h0) 2채널을 추가한 8D. 절대
  heading(dir_x,dir_z 그대로)은 이 사이트 카메라 좌표계에 묶인 site-calibration-specific
  신호라 위치 암기와 같은 방식으로 새어 들어갈 위험이 있지만(§데이터 설계 "방향벡터 추가 검토"
  참조), 이 채널은 d_lat과 동일하게 "윈도우 자기 자신의 시작 시점"만 기준으로 삼는 self-relative
  값이라 site-invariant가 유지된다. omega(순간 회전율)와 상호보완적 — omega는 그 순간의 회전
  "속도", 이 cos/sin쌍은 시작점 대비 누적된 회전 "변위"를 담는다(v↔d_lat 관계와 동형).
"""
import torch
import torch.nn as nn


def _wrap_angle(d: torch.Tensor) -> torch.Tensor:
    return (d + torch.pi) % (2 * torch.pi) - torch.pi


class SemanticDerivation(nn.Module):
    """
    입력: node_seq [..., T, 6]
    출력: semantic_seq [..., T, 8] (variant='full', 기본) 또는 [..., T, 6] (variant='invariant',
          Δρ·접선 제외 — v,a,j,ω,d_lat,κ만) 또는 [..., T, 2] (variant='position_only',
          raw pos_x,pos_z 그대로 — 운동학 채널 전무) 또는 [..., T, 8] (variant='invariant_rot',
          invariant 6D + cos(Δheading),sin(Δheading))
    """

    SEM_DIM = 8
    SEM_DIM_INVARIANT = 6
    SEM_DIM_POSITION_ONLY = 2
    SEM_DIM_INVARIANT_ROT = 8
    # 공업탑 로터리 중심 (world 좌표) — journal/map_features.py 검증값, 공업탑 데이터 전용 상수
    CENTER_X = 72.86
    CENTER_Z = -13.45

    def __init__(self, variant: str = 'full'):
        super().__init__()
        assert variant in ('full', 'invariant', 'position_only', 'invariant_rot')
        self.variant = variant
        if variant == 'invariant':
            self.SEM_DIM = self.SEM_DIM_INVARIANT
        elif variant == 'position_only':
            self.SEM_DIM = self.SEM_DIM_POSITION_ONLY
        elif variant == 'invariant_rot':
            self.SEM_DIM = self.SEM_DIM_INVARIANT_ROT
        else:
            self.SEM_DIM = 8

    def forward(self, node_seq: torch.Tensor) -> torch.Tensor:
        if self.variant == 'position_only':
            # 운동학 유도채널 전부 생략 — raw world 좌표 [pos_x,pos_z]만 그대로 반환.
            # 결측(0 패딩) 프레임도 이미 0이라 별도 마스킹 불필요.
            return node_seq[..., 0:2]

        v = node_seq[..., 2]
        a = node_seq[..., 5]

        j = torch.zeros_like(a)
        if a.shape[-1] > 1:
            j[..., 1:] = a[..., 1:] - a[..., :-1]

        heading = torch.atan2(node_seq[..., 4], node_seq[..., 3] + 1e-8)
        omega = torch.zeros_like(heading)
        if heading.shape[-1] > 1:
            omega[..., 1:] = _wrap_angle(heading[..., 1:] - heading[..., :-1])

        h0 = heading[..., :1]
        dx = node_seq[..., 0] - node_seq[..., :1, 0]
        dz = node_seq[..., 1] - node_seq[..., :1, 1]
        d_lat = -torch.sin(h0) * dx + torch.cos(h0) * dz

        kappa = omega / (v.abs() + 0.1)

        # 결측(0 패딩) 프레임: 두 프레임 다 유효할 때만 diff 계산 (미관측 0을 실좌표로 오인 방지)
        present = (node_seq.abs().sum(dim=-1) > 0).float()
        valid_diff = torch.zeros_like(present)
        if present.shape[-1] > 1:
            valid_diff[..., 1:] = present[..., 1:] * present[..., :-1]

        if self.variant == 'invariant':
            # 로터리 중심(world 고정 상수) 참조 채널(Δρ, 접선)은 계산조차 하지 않는다 — site-specific
            # 신호가 전혀 섞이지 않는 완전 ego/rotation-invariant 6D.
            return torch.stack([v, a, j, omega, d_lat, kappa], dim=-1)

        if self.variant == 'invariant_rot':
            # d_lat과 동일한 원칙(윈도우 자기 시작 시점만 기준) — 절대 heading이 아니라
            # h0(윈도우 시작 heading) 대비 "지금까지 얼마나 돌았는가"만 담아 site-invariant 유지.
            dtheta_from_start = _wrap_angle(heading - h0)
            cos_rot = torch.cos(dtheta_from_start) * present
            sin_rot = torch.sin(dtheta_from_start) * present
            return torch.stack([v, a, j, omega, d_lat, kappa, cos_rot, sin_rot], dim=-1)

        cx = node_seq[..., 0] - self.CENTER_X
        cz = node_seq[..., 1] - self.CENTER_Z
        rho = torch.sqrt(cx * cx + cz * cz + 1e-8)
        theta = torch.atan2(cz, cx)

        drho = torch.zeros_like(rho)
        dtheta = torch.zeros_like(theta)
        if rho.shape[-1] > 1:
            drho[..., 1:] = rho[..., 1:] - rho[..., :-1]
            dtheta[..., 1:] = _wrap_angle(theta[..., 1:] - theta[..., :-1])
        drho = drho * valid_diff
        dtheta = dtheta * valid_diff
        tangent = dtheta * rho * valid_diff

        return torch.stack([v, a, j, omega, d_lat, kappa, drho, tangent], dim=-1)


class LearnableSemanticResidual(nn.Module):
    """§실험현황 "20260712 추가 실험계획" ② 전용 (2026-07-12).

    SemanticDerivation은 파라미터 0개인 고정 손공학 변환이라 압축 과정에서 버려지는 미세한
    궤적 정보(정확한 곡선 형태, 고차 미분 등)가 있을 수 있다는 가설을 검증하기 위한 보강 모듈.
    raw 6D 중 위치·절대방향과 무관한 두 스칼라(speed, accel)만 프레임별로 받아 작은 MLP로
    residual_dim개의 학습된 채널을 뽑는다 — speed·accel은 그 자체로 이미 ego-relative/invariant
    스칼라라(세계좌표계·카메라 방향과 무관) 위치 암기 경로를 새로 열지 않는다. 결측(0-padding)
    프레임은 present 마스크로 0 처리(다른 semantic 채널과 동일 컨벤션).
    """

    def __init__(self, residual_dim: int = 4, hidden: int = 16, dropout: float = 0.1):
        super().__init__()
        self.residual_dim = residual_dim
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, residual_dim),
        )

    def forward(self, node_seq: torch.Tensor) -> torch.Tensor:
        """raw [..., T, 6] -> learned residual [..., T, residual_dim]."""
        speed = node_seq[..., 2:3]
        accel = node_seq[..., 5:6]
        present = (node_seq.abs().sum(dim=-1, keepdim=True) > 0).float()
        x = torch.cat([speed, accel], dim=-1)
        return self.mlp(x) * present
