"""
TSEM-SAGE 데이터 augmentation — 9차(2026-07-08) 신설.

7·8차(500 epoch 연장) 실험에서 train loss는 계속 내려가는데 val_acc는 epoch~90 부근에서
정점을 찍고 이후 계속 하락하는 과적합이 관측됨 — 원인 중 하나로 지목된 것이 "매 epoch 완전히
동일한 입력"(augmentation 전무)이었음. 이 모듈은 train split에만 적용되는 조합 가능한
augmentation 4종을 제공한다(val/test는 항상 원본 그대로 — 평가 재현성 유지).

조합 번호(docs/TSEM_journal_design.html 실험 표와 매칭):
  1) 가우시안 노이즈 — 속도·가속도·위치·방향에 소량 노이즈 (센서/검출 오차 흉내)
  2) 이웃 랜덤 드롭아웃 — 존재하는 이웃 슬롯 일부를 학습 시에만 랜덤하게 결측 처리
  3) 프레임 랜덤 드롭아웃 — 과거 W프레임 중 일부(앵커 t 제외)를 랜덤 결측 처리
  4) 좌표 회전 — world 좌표 전체를 **로터리 중심(CENTER_X, CENTER_Z) 기준**으로 소각도 회전
     (9차엔 rotate_deg=0으로 미적용)

적용 시점: 캐시에서 로드한(또는 비캐시 조립한) node_seq/nbr_node_seqs/nbr2_node_seqs에
_recompute_edges() 호출 *이전*에 적용 — 그래야 rel_speed 등 edge feature가 노이즈가 반영된
값으로부터 다시 계산되어 내부적으로 일관성이 유지된다.

4번(회전)의 회전축 주의사항: models/tsem_semantic_derivation.py::SemanticDerivation이 Δρ·접선
채널을 "공업탑 로터리 중심(CENTER_X, CENTER_Z)"이라는 **고정된 절대 world 좌표** 기준으로 계산한다.
만약 이 augmentation이 좌표를 원점(0,0) 기준으로 회전시키면, 회전된(가짜) 위치와 회전 안 된(실제)
로터리 중심 사이 거리를 계산하게 되어 ρ·θ·Δρ·접선이 전부 물리적으로 의미 없는 값이 된다. 그래서
아래 _rotate()는 반드시 SemanticDerivation과 **동일한 중심점**을 기준으로 회전한다 — 이러면 로터리
중심까지의 거리(ρ)는 회전에 불변으로 보존되고, 각도(θ)만 회전각만큼 일관되게 이동한다(= 로터리를
다른 방위각에서 관측한 것과 동일한 물리적 의미).
"""
from __future__ import annotations

import numpy as np

from models.tsem_semantic_derivation import SemanticDerivation

# NODE_COLS 순서: position_x, position_z, speed, direction_x, direction_z, acceleration
_NODE_STD = np.array([0.1, 0.1, 0.15, 0.02, 0.02, 0.1], dtype=np.float32)
_ROTARY_CENTER = np.array(
    [SemanticDerivation.CENTER_X, SemanticDerivation.CENTER_Z], dtype=np.float32
)


class TsemAugment:
    def __init__(
        self,
        noise_std: float = 0.0,
        neighbor_dropout_p: float = 0.0,
        frame_dropout_p: float = 0.0,
        rotate_deg: float = 0.0,
    ):
        self.noise_std = float(noise_std)
        self.neighbor_dropout_p = float(neighbor_dropout_p)
        self.frame_dropout_p = float(frame_dropout_p)
        self.rotate_deg = float(rotate_deg)

    @property
    def enabled(self) -> bool:
        return (
            self.noise_std > 0
            or self.neighbor_dropout_p > 0
            or self.frame_dropout_p > 0
            or self.rotate_deg > 0
        )

    def describe(self) -> str:
        parts = []
        if self.noise_std > 0:
            parts.append(f'noise(σ×{self.noise_std})')
        if self.neighbor_dropout_p > 0:
            parts.append(f'neighbor_dropout(p={self.neighbor_dropout_p})')
        if self.frame_dropout_p > 0:
            parts.append(f'frame_dropout(p={self.frame_dropout_p})')
        if self.rotate_deg > 0:
            parts.append(f'rotate(±{self.rotate_deg}°)')
        return ' + '.join(parts) if parts else 'off'

    @staticmethod
    def _present_mask(arr: np.ndarray) -> np.ndarray:
        return (np.abs(arr).sum(axis=-1, keepdims=True) > 0).astype(np.float32)

    def _add_noise(self, arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        present = self._present_mask(arr)
        noise = rng.normal(0.0, 1.0, size=arr.shape).astype(np.float32) * (_NODE_STD * self.noise_std)
        return arr + noise * present

    def _frame_dropout(self, node_seq, nbr_node_seqs, nbr2_node_seqs, rng: np.random.Generator):
        """과거 T프레임 중 일부를 결측 처리. _recompute_edges()가 이웃(dst)쪽 결측만으로
        엣지를 0-마스킹하기 때문에(ego/src쪽은 확인 안 함), ego 프레임만 지우면 그 프레임의
        엣지가 '0 - 실제값' 형태의 엉터리 값으로 남는다 — 그래서 같은 시간 인덱스를 ego·1hop·2hop
        전부에서 동시에 지워 "그 프레임엔 아무도 관측 안 됨" 상태로 일관되게 맞춘다."""
        T = node_seq.shape[0]
        drop = rng.random(T) < self.frame_dropout_p
        drop[-1] = False  # 앵커 프레임(t)은 그래프 구성에 쓰이므로 항상 유지
        node_seq = node_seq.copy()
        node_seq[drop] = 0.0
        nbr_node_seqs = nbr_node_seqs.copy()
        nbr_node_seqs[:, drop] = 0.0
        nbr2_node_seqs = nbr2_node_seqs.copy()
        nbr2_node_seqs[:, :, drop] = 0.0
        return node_seq, nbr_node_seqs, nbr2_node_seqs

    def _neighbor_dropout(self, nbr_node_seqs, nbr_mask, rng):
        K = nbr_mask.shape[0]
        present = nbr_mask > 0
        drop = (rng.random(K) < self.neighbor_dropout_p) & present
        nbr_node_seqs = nbr_node_seqs.copy()
        nbr_node_seqs[drop] = 0.0
        nbr_mask = nbr_mask.copy()
        nbr_mask[drop] = 0.0
        return nbr_node_seqs, nbr_mask

    def _neighbor2_dropout(self, nbr2_node_seqs, nbr2_mask, rng):
        shape = nbr2_mask.shape
        present = nbr2_mask > 0
        drop = (rng.random(shape) < self.neighbor_dropout_p) & present
        nbr2_node_seqs = nbr2_node_seqs.copy()
        nbr2_node_seqs[drop] = 0.0
        nbr2_mask = nbr2_mask.copy()
        nbr2_mask[drop] = 0.0
        return nbr2_node_seqs, nbr2_mask

    def _rotate(self, node_seq, nbr_node_seqs, nbr2_node_seqs, rng):
        """4번(좌표 회전) — 9차엔 미적용(rotate_deg=0). 로터리 중심(_ROTARY_CENTER) 기준으로
        회전해야 SemanticDerivation의 ρ(중심까지 거리)가 보존된다 — 원점 기준으로 돌리면 ρ·θ·
        Δρ·접선이 전부 깨진다(모듈 docstring 참조)."""
        theta = np.deg2rad(rng.uniform(-self.rotate_deg, self.rotate_deg))
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)

        def _apply(arr):
            pos = arr[..., 0:2]
            present = self._present_mask(arr)
            rotated = (pos - _ROTARY_CENTER) @ rot.T + _ROTARY_CENTER  # 중심 기준 회전 → ρ 보존
            arr = arr.copy()
            arr[..., 0:2] = np.where(present > 0, rotated, pos)
            dirn = arr[..., 3:5]
            arr[..., 3:5] = np.where(present > 0, dirn @ rot.T, dirn)  # 방향벡터는 이동 불변이라 중심 무관
            return arr

        return _apply(node_seq), _apply(nbr_node_seqs), _apply(nbr2_node_seqs)

    def apply(
        self,
        node_seq: np.ndarray,
        nbr_node_seqs: np.ndarray,
        nbr_mask: np.ndarray,
        nbr2_node_seqs: np.ndarray,
        nbr2_mask: np.ndarray,
        rng: np.random.Generator,
    ):
        """[T,6], [K,T,6], [K], [K,K2,T,6], [K,K2] → 증강된 동일 shape 튜플."""
        if self.rotate_deg > 0:
            node_seq, nbr_node_seqs, nbr2_node_seqs = self._rotate(
                node_seq, nbr_node_seqs, nbr2_node_seqs, rng
            )
        if self.noise_std > 0:
            node_seq = self._add_noise(node_seq, rng)
            nbr_node_seqs = self._add_noise(nbr_node_seqs, rng)
            nbr2_node_seqs = self._add_noise(nbr2_node_seqs, rng)
        if self.frame_dropout_p > 0:
            node_seq, nbr_node_seqs, nbr2_node_seqs = self._frame_dropout(
                node_seq, nbr_node_seqs, nbr2_node_seqs, rng
            )
        if self.neighbor_dropout_p > 0:
            nbr_node_seqs, nbr_mask = self._neighbor_dropout(nbr_node_seqs, nbr_mask, rng)
            nbr2_node_seqs, nbr2_mask = self._neighbor2_dropout(nbr2_node_seqs, nbr2_mask, rng)
        return node_seq, nbr_node_seqs, nbr_mask, nbr2_node_seqs, nbr2_mask
