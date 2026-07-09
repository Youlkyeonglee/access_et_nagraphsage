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
"""
import torch
import torch.nn as nn


def _wrap_angle(d: torch.Tensor) -> torch.Tensor:
    return (d + torch.pi) % (2 * torch.pi) - torch.pi


class SemanticDerivation(nn.Module):
    """
    입력: node_seq [..., T, 6]
    출력: semantic_seq [..., T, 8]
    """

    SEM_DIM = 8
    # 공업탑 로터리 중심 (world 좌표) — journal/map_features.py 검증값, 공업탑 데이터 전용 상수
    CENTER_X = 72.86
    CENTER_Z = -13.45

    def forward(self, node_seq: torch.Tensor) -> torch.Tensor:
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
