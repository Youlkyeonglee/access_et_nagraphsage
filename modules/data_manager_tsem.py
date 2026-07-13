"""
TSEM Future-State Dataloader
============================
ET-NAGraphSAGE `data_manager.py`와 분리된 파이프라인.

차이점:
  - 정답: y = instant state(t+H)  (CSV category / ±6 smear 미사용)
  - 관측: 과거 W프레임 [t-W+1 … t] (기존과 동일 구조)
  - 캐시: cache/tsem/ (기존 cache/ 와 분리)

출력 샘플 키는 data_manager와 동일 + y_persist (Persist 베이스라인용 state(t)).
"""
from __future__ import annotations

import bisect
import hashlib
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from modules.data_manager import (
    EDGE_DIM,
    NODE_COLS,
    NODE_DIM,
    _SANITIZE_CLIP,
    _compute_edge_feat,
    _recompute_edges,
    _sanitize,
    _standardize_present,
)
from modules.tsem_instant_label import (
    CLASS_NAMES,
    LANE_CHANGE,
    NORMAL,
    STOP,
    build_instant_labels_from_dataframe,
)
from modules.tsem_augment import TsemAugment

CACHE_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'cache',
    'tsem',
)

EDGE_DIM_BEARING = 7  # [rel_speed, rel_accel, dist, cos φ, sin φ, cos Δθ, sin Δθ] — rel_dx/rel_dz 제외

def _recompute_edges_bearing(node_seq: np.ndarray, nbr_node_seqs: np.ndarray,
                              nbr2_node_seqs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """TSEM 전용 edge feature (2026-07-13) — bearing(cos φ,sin φ) + Δheading(cos Δθ,sin Δθ) 포함,
    rel_dx/rel_dz(heading 벡터 단순 차, 방향 정보 없음) 제외. modules/data_manager.py::_recompute_edges와
    별도 구현 — 그 파일은 ET-NAGraphSAGE 컨퍼런스 트랙과 공유·수정 금지(CLAUDE.md §4)라 TSEM 전용
    opt-in으로 병행 구현한다. 도식·근거: docs/TSEM_journal_design.html §data-neighbor "③ edge feature에
    방위각(bearing)이 빠져 있다".

    bearing: rel_pos=(dst_x-src_x, dst_z-src_z)를 src의 heading(dir_x,dir_z, 단위벡터 가정)
    기준 로컬 프레임으로 투영 — atan2 없이 내적/외적만으로 cos φ, sin φ를 직접 얻는다
    (φ의 wrap-around 문제 자체가 발생하지 않음).
      cos φ = (rel_pos · src_dir) / dist,  sin φ = (src_dir × rel_pos) / dist
    Δheading: 두 heading 단위벡터의 내적/외적 = cos/sin(Δθ) (product-to-sum 항등식, atan2 불필요).
      cos Δθ = dst_dir · src_dir,  sin Δθ = src_dir × dst_dir
    """
    ns = node_seq.astype(np.float32)          # [T,6]
    nbr = nbr_node_seqs.astype(np.float32)      # [K,T,6]

    def _edges(src, dst):                       # [...,6],[...,6] → [...,7]
        rs = dst[..., 2] - src[..., 2]
        ra = dst[..., 5] - src[..., 5]
        rx = dst[..., 0] - src[..., 0]
        rz = dst[..., 1] - src[..., 1]
        dist = np.sqrt(rx ** 2 + rz ** 2)
        dist_safe = dist + 1e-8

        src_dx, src_dz = src[..., 3], src[..., 4]
        src_norm = np.sqrt(src_dx ** 2 + src_dz ** 2) + 1e-8
        sdx, sdz = src_dx / src_norm, src_dz / src_norm   # src heading 단위벡터

        dst_dx, dst_dz = dst[..., 3], dst[..., 4]
        dst_norm = np.sqrt(dst_dx ** 2 + dst_dz ** 2) + 1e-8
        ddx, ddz = dst_dx / dst_norm, dst_dz / dst_norm   # dst heading 단위벡터

        cos_phi = (rx * sdx + rz * sdz) / dist_safe        # rel_pos·src_dir / |rel_pos|
        sin_phi = (sdx * rz - sdz * rx) / dist_safe        # src_dir×rel_pos / |rel_pos|
        cos_dtheta = ddx * sdx + ddz * sdz                  # dst_dir·src_dir
        sin_dtheta = sdx * ddz - sdz * ddx                  # src_dir×dst_dir

        return np.stack(
            [rs, ra, dist, cos_phi, sin_phi, cos_dtheta, sin_dtheta], axis=-1
        ).astype(np.float32)

    src1 = np.broadcast_to(ns[None], nbr.shape)                 # [K,T,6]
    edge_seqs = _edges(src1, nbr)                                # [K,T,7]
    edge_seqs *= (np.abs(nbr).sum(-1, keepdims=True) > 0)        # 결측 0 유지

    if nbr2_node_seqs is not None and nbr2_node_seqs.size:
        nbr2 = nbr2_node_seqs.astype(np.float32)                # [K,K2,T,6]
        src2 = np.broadcast_to(nbr[:, None], nbr2.shape)        # [K,K2,T,6]
        nbr2_edge_seqs = _edges(src2, nbr2)                      # [K,K2,T,7]
        nbr2_edge_seqs *= (np.abs(nbr2).sum(-1, keepdims=True) > 0)
    else:
        nbr2_edge_seqs = np.zeros((nbr.shape[0], 0, nbr.shape[1], EDGE_DIM_BEARING),
                                  dtype=np.float32)
    return edge_seqs, nbr2_edge_seqs


EDGE_DIM_FOURIER_NUM_FREQ = 4
EDGE_DIM_FOURIER = 7 + 2 * EDGE_DIM_FOURIER_NUM_FREQ  # 15 = bearing 7D + dist Fourier(4주파수×sin/cos)
_FOURIER_DIST_SCALE = 20.0  # graph.radius(20.0) 기준 정규화 — 관측 가능한 최대 거리로 스케일 맞춤


def _recompute_edges_fourier(node_seq: np.ndarray, nbr_node_seqs: np.ndarray,
                              nbr2_node_seqs: np.ndarray,
                              num_freq: int = EDGE_DIM_FOURIER_NUM_FREQ) -> Tuple[np.ndarray, np.ndarray]:
    """TSEM 전용 edge feature (2026-07-13) — _recompute_edges_bearing(7D)에 dist의 Fourier
    위치 인코딩(HiVT/QCNet 스타일, sin/cos 다중 주파수)을 추가한 15D 변형. bearing/Δheading은
    그대로 두고 스칼라 dist 하나만 표현력을 확장 — "raw dist 하나로는 거리 스케일을 세밀하게
    구분 못 할 수 있다"는 §data-neighbor "③" 가설의 연장. modules/data_manager.py는 여전히
    미접촉(CLAUDE.md §4 patttern 그대로 따름).

    dist_norm = dist / radius(20.0),  주파수 f_k = 2^k (k=0..num_freq-1)
    fourier(dist) = [sin(2π f_0 dist_norm), cos(2π f_0 dist_norm), ..., sin(2π f_{K-1} dist_norm), cos(...)]
    """
    ns = node_seq.astype(np.float32)
    nbr = nbr_node_seqs.astype(np.float32)
    freqs = (2.0 ** np.arange(num_freq)).astype(np.float32)  # [K]

    def _edges(src, dst):
        rs = dst[..., 2] - src[..., 2]
        ra = dst[..., 5] - src[..., 5]
        rx = dst[..., 0] - src[..., 0]
        rz = dst[..., 1] - src[..., 1]
        dist = np.sqrt(rx ** 2 + rz ** 2)
        dist_safe = dist + 1e-8

        src_dx, src_dz = src[..., 3], src[..., 4]
        src_norm = np.sqrt(src_dx ** 2 + src_dz ** 2) + 1e-8
        sdx, sdz = src_dx / src_norm, src_dz / src_norm

        dst_dx, dst_dz = dst[..., 3], dst[..., 4]
        dst_norm = np.sqrt(dst_dx ** 2 + dst_dz ** 2) + 1e-8
        ddx, ddz = dst_dx / dst_norm, dst_dz / dst_norm

        cos_phi = (rx * sdx + rz * sdz) / dist_safe
        sin_phi = (sdx * rz - sdz * rx) / dist_safe
        cos_dtheta = ddx * sdx + ddz * sdz
        sin_dtheta = sdx * ddz - sdz * ddx

        dist_norm = dist / _FOURIER_DIST_SCALE               # [...]
        angle = 2.0 * np.pi * dist_norm[..., None] * freqs   # [...,K]
        fourier = np.concatenate([np.sin(angle), np.cos(angle)], axis=-1)  # [...,2K]

        base = np.stack(
            [rs, ra, dist, cos_phi, sin_phi, cos_dtheta, sin_dtheta], axis=-1
        ).astype(np.float32)
        return np.concatenate([base, fourier.astype(np.float32)], axis=-1)

    src1 = np.broadcast_to(ns[None], nbr.shape)
    edge_seqs = _edges(src1, nbr)
    edge_seqs *= (np.abs(nbr).sum(-1, keepdims=True) > 0)

    if nbr2_node_seqs is not None and nbr2_node_seqs.size:
        nbr2 = nbr2_node_seqs.astype(np.float32)
        src2 = np.broadcast_to(nbr[:, None], nbr2.shape)
        nbr2_edge_seqs = _edges(src2, nbr2)
        nbr2_edge_seqs *= (np.abs(nbr2).sum(-1, keepdims=True) > 0)
    else:
        nbr2_edge_seqs = np.zeros((nbr.shape[0], 0, nbr.shape[1], EDGE_DIM_FOURIER),
                                  dtype=np.float32)
    return edge_seqs, nbr2_edge_seqs


_EDGE_FN_BY_VARIANT = {
    'legacy': _recompute_edges,
    'bearing': _recompute_edges_bearing,
    'fourier': _recompute_edges_fourier,
}


_CACHE_KEYS = [
    'node_seq', 'nbr_node_seqs', 'edge_seqs', 'nbr_mask',
    'nbr2_node_seqs', 'nbr2_edge_seqs', 'nbr2_mask', 'y', 'y_persist', 'stop_conf',
]
_CACHE_KEYS_STORED = [
    'node_seq', 'nbr_node_seqs', 'nbr_mask',
    'nbr2_node_seqs', 'nbr2_mask', 'y', 'y_persist', 'stop_conf',
]
_CACHE_FMT = 'tsem_f16ne_v4'  # v4: stop_conf 추가(8차 불확실성 인지 loss용, docs §12.12)


class _TsemFileData:
    """CSV 로드 — kinematics + lane_id, instant 라벨은 category 무시."""

    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        if 'lane_id' not in df.columns:
            raise ValueError(
                f'{csv_path}: lane_id 열 필요 (TSEM instant LC 라벨). '
                'category는 사용하지 않습니다.'
            )
        for col in NODE_COLS:
            if col not in df.columns:
                raise ValueError(f'{csv_path}: 필수 열 없음 {col}')

        df = df.sort_values(['frame', 'object_id']).reset_index(drop=True)
        self.csv_path = csv_path
        self.instant_labels = build_instant_labels_from_dataframe(df)

        self.sorted_frames = sorted(df['frame'].unique().tolist())
        self.n_frames = len(self.sorted_frames)
        diffs = np.diff(self.sorted_frames)
        self.frame_step = int(np.median(diffs)) if len(diffs) > 0 else 1
        self.max_gap = self.frame_step * 2

        self.frame_node: Dict[int, Dict[int, np.ndarray]] = {}
        for frame, grp in df.groupby('frame'):
            node_dict = {}
            for _, row in grp.iterrows():
                oid = int(row['object_id'])
                node_dict[oid] = np.array(
                    [row[c] for c in NODE_COLS], dtype=np.float32
                )
            self.frame_node[int(frame)] = node_dict

        self.obj_frames: Dict[int, List[int]] = defaultdict(list)
        for frame, grp in df.groupby('frame'):
            for oid in grp['object_id'].unique():
                self.obj_frames[int(oid)].append(int(frame))
        for oid in self.obj_frames:
            self.obj_frames[oid].sort()

        self._kdtree_cache: Dict[int, Tuple] = {}

    def get_kdtree(self, frame: int) -> Tuple:
        if frame not in self._kdtree_cache:
            nd = self.frame_node[frame]
            obj_ids = np.array(list(nd.keys()), dtype=np.int64)
            positions = np.array(
                [nd[oid][:2] for oid in obj_ids], dtype=np.float32
            )
            tree = cKDTree(positions)
            self._kdtree_cache[frame] = (tree, obj_ids, positions)
        return self._kdtree_cache[frame]

    def instant_state(self, frame: int, object_id: int) -> int:
        return self.instant_labels[(int(frame), int(object_id))]

    def _stop_window_votes(self, future_frame: int, object_id: int, delta: int) -> Tuple[int, int]:
        """future_frame 중심 ±delta 구간의 (STOP 프레임 수 k, 전체 프레임 수 n)."""
        frames_obj = self.obj_frames[object_id]
        i = bisect.bisect_left(frames_obj, future_frame)
        near = frames_obj[max(0, i - delta): i + delta + 1]
        n = len(near)
        k = sum(1 for f in near if self.instant_state(f, object_id) == STOP)
        return k, n

    def _stop_persist_state(self, future_frame: int, object_id: int, delta: int) -> int:
        """B안 (2026-07-07): stop을 future_frame 한 점이 아니라, 그 주변 대칭 소구간
        [future_frame-delta, future_frame+delta]의 다수결로 판정. speed<=1.0 임계값 경계에서
        프레임 단위로 뒤집히는 flicker(정답=normal의 29.8%가 future_frame ±2프레임 내 STOP 상태와
        인접, journal/stop_error_analysis.py 실측)를 제거하기 위함. lane_change 판정에는 영향 없음
        (이 값이 STOP이 아니면 기존 로직 그대로 진행)."""
        k, n = self._stop_window_votes(future_frame, object_id, delta)
        return STOP if k * 2 > n else NORMAL

    def stop_confidence(self, future_frame: int, object_id: int, delta: int = 2) -> float:
        """8차 (2026-07-08): ±delta 다수결 stop 판정의 신뢰도. Beta(1,1) 사전 + Beta-Binomial
        사후분산을 n에서 실제 도달 가능한 최댓값(1/(4(n+3)))으로 정규화 — n->inf 극한에서
        1-4·p̂(1-p̂)로 수렴함(docs/TSEM_journal_design.html §12.12 유도 참조)."""
        k, n = self._stop_window_votes(future_frame, object_id, delta)
        return 1.0 - 4.0 * (k + 1) * (n - k + 1) / ((n + 2) ** 2)

    def future_window_state(
        self, anchor: int, future_frame: int, object_id: int, stop_persist_delta: int = 2,
    ) -> int:
        """B안 (2026-07-07): stop은 future_frame 중심 ±stop_persist_delta 다수결(지속성),
        lane_change는 (anchor, future_frame] 구간 내 발생 여부(occurrence, A안 그대로 유지).
        bisect로 프레임 인덱스 조회 (obj_frames는 정렬되어 있음)."""
        pt = self._stop_persist_state(future_frame, object_id, stop_persist_delta)
        if pt == STOP:
            return STOP
        frames_obj = self.obj_frames[object_id]
        ia = bisect.bisect_left(frames_obj, anchor)
        ifut = bisect.bisect_left(frames_obj, future_frame)
        for f in frames_obj[ia + 1: ifut + 1]:
            if self.instant_state(f, object_id) == LANE_CHANGE:
                return LANE_CHANGE
        return NORMAL


_FILE_CACHE: Dict[str, _TsemFileData] = {}


def _get_file_data(csv_path: str) -> _TsemFileData:
    if csv_path not in _FILE_CACHE:
        _FILE_CACHE[csv_path] = _TsemFileData(csv_path)
    return _FILE_CACHE[csv_path]


def _build_sample_index(
    fd: _TsemFileData,
    W: int,
    H: int,
    split: str,
    train_ratio: float,
    val_ratio: float,
) -> List[Tuple[str, int, List[int], int]]:
    """
    샘플: (csv_path, ego_id, window_frames, future_frame)
    """
    n = fd.n_frames
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    if split == 'train':
        valid_frame_set = set(fd.sorted_frames[:train_end])
    elif split == 'val':
        valid_frame_set = set(fd.sorted_frames[train_end:val_end])
    else:
        valid_frame_set = set(fd.sorted_frames[val_end:])

    samples = []
    for oid, frames in fd.obj_frames.items():
        if len(frames) < W + H:
            continue
        for i in range(W - 1, len(frames) - H):
            window = frames[i - W + 1: i + 1]
            future_frame = frames[i + H]
            anchor = window[-1]
            if anchor not in valid_frame_set:
                continue
            gaps = [window[k + 1] - window[k] for k in range(len(window) - 1)]
            if any(g > fd.max_gap for g in gaps):
                continue
            gap_future = future_frame - anchor
            if gap_future > H * fd.max_gap:
                continue
            if (anchor, oid) not in fd.instant_labels:
                continue
            if (future_frame, oid) not in fd.instant_labels:
                continue
            samples.append((fd.csv_path, oid, window, future_frame))
    return samples


class TSEMFutureStateDataset(Dataset):
    """
    TSEM 미래 상태 예측 데이터셋.

    Args:
        W: 과거 관측 창 (프레임 수)
        H: 예측 지평 — ego 궤적상 H스텝 뒤 프레임의 instant state
    """

    def __init__(
        self,
        csv_files: List[str],
        W: int = 10,
        H: int = 10,
        radius: float = 20.0,
        K_max: int = 6,
        K_max2: int = 4,
        split: str = 'train',
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        neighbor_mode: str = 'hybrid',
        ego_relative: bool = False,
        verbose: bool = True,
        use_cache: bool = True,
        augment: 'TsemAugment' = None,
        augment_seed: int = 12345,
        stop_persist_delta: int = 2,
        with_v_future: bool = False,
        edge_feat_variant: str = 'legacy',
    ):
        self.W = W
        self.H = H
        self.T = W
        self.radius = radius
        self.K = K_max
        self.K2 = K_max2
        self.use_2hop = K_max2 > 0
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.ego_relative = ego_relative
        assert neighbor_mode in ('hybrid', 'count', 'radius')
        self.neighbor_mode = neighbor_mode
        # edge feature 변형 (2026-07-13): 'legacy'(기본, 하위호환)는 기존 _recompute_edges(5D,
        # rel_dx/rel_dz 포함) 그대로. 'bearing'은 _recompute_edges_bearing(7D, bearing cos/sinφ +
        # Δheading cos/sinΔθ, rel_dx/rel_dz 제외) — docs/TSEM_journal_design.html §data-neighbor
        # "③ edge feature에 방위각이 빠져 있다" 참조. 'fourier'는 bearing 7D + dist Fourier
        # 위치인코딩(15D, _recompute_edges_fourier) — §exp-20260713-plan ②. 모델 쪽
        # edge_dim(config model.edge_dim)을 각각 7/15로 맞춰야 shape이 맞는다.
        assert edge_feat_variant in ('legacy', 'bearing', 'fourier')
        self.edge_feat_variant = edge_feat_variant
        # 9차 (2026-07-08): augmentation은 train split에만 적용, val/test는 항상 원본 그대로
        self.augment = augment if (augment is not None and split == 'train') else None
        self._augment_seed = augment_seed
        # 학교서버 실험 (2026-07-09): stop ±δ 다수결의 δ를 노출 — 기본 2(B안), 3/4 sweep용
        self.stop_persist_delta = stop_persist_delta
        # 제안1 (2026-07-09): v(t+H) 회귀 보조헤드용 미래 속도 라벨. opt-in — 켜면 캐시 폴더에
        # v_future.npy 사이드카를 생성/재사용(캐시 키는 안 바뀜). 첫 생성 시 CSV 파싱 필요.
        self.with_v_future = with_v_future
        self._v_future: np.ndarray = None
        self._aug_rng = None

        self.samples: List[Tuple[str, int, List[int], int]] = []
        self._cache: Dict[str, np.ndarray] = None
        self._norm = None

        cache_dir = (
            os.path.join(CACHE_ROOT, self._cache_key(csv_files))
            if use_cache
            else None
        )

        if cache_dir is not None and os.path.exists(
            os.path.join(cache_dir, '.done')
        ):
            with open(os.path.join(cache_dir, 'samples.pkl'), 'rb') as f:
                self.samples = pickle.load(f)
            self._cache = {
                k: np.load(os.path.join(cache_dir, f'{k}.npy'))
                for k in _CACHE_KEYS_STORED
            }
            if verbose:
                print(
                    f'[TSEM/{split}] 캐시 로드: {len(self.samples):,}개  ({cache_dir})'
                )
            if self.with_v_future:
                self._v_future = self._load_or_build_v_future(cache_dir, verbose)
            self._setup_norm(cache_dir, verbose)
            return

        for csv_path in (
            tqdm(csv_files, desc=f'[TSEM/{split}] 인덱스')
            if verbose
            else csv_files
        ):
            fd = _get_file_data(csv_path)
            self.samples += _build_sample_index(
                fd, W, H, split, train_ratio, val_ratio
            )

        if verbose:
            hop = f', K2={K_max2}' if self.use_2hop else ''
            print(
                f'[TSEM/{split}] 샘플 {len(self.samples):,}개  '
                f'(W={W}, H={H}, r={radius}m, K={K_max}{hop})'
            )

        if cache_dir is not None:
            self._build_cache(cache_dir, verbose)
        if self.with_v_future:
            self._v_future = self._load_or_build_v_future(cache_dir, verbose)
        self._setup_norm(cache_dir, verbose)

    def _cache_key(self, csv_files: List[str]) -> str:
        h = hashlib.md5()
        for p in sorted(csv_files):
            st = os.stat(p)
            h.update(
                f'{os.path.basename(p)}:{st.st_size}:{int(st.st_mtime)}'.encode()
            )
        sig = (
            f'W{self.W}_H{self.H}_r{self.radius}_K{self.K}-{self.K2}'
            f'_{self.neighbor_mode}_{_CACHE_FMT}'
            f'{"_egorel" if self.ego_relative else ""}'
            # δ≠2일 때만 키에 포함 — 기존 v4 캐시(δ=2)의 폴더명·재사용을 그대로 유지하기 위함
            f'{f"_sd{self.stop_persist_delta}" if self.stop_persist_delta != 2 else ""}'
            f'_tr{self.train_ratio}_vr{self.val_ratio}'
        )
        h.update(sig.encode())
        return f'{self.split}_{sig}_{h.hexdigest()[:10]}'

    def _cache_shapes(self, N: int) -> Dict[str, Tuple]:
        K, K2, T = self.K, self.K2, self.T
        return {
            'node_seq': (N, T, NODE_DIM),
            'nbr_node_seqs': (N, K, T, NODE_DIM),
            'nbr_mask': (N, K),
            'nbr2_node_seqs': (N, K, K2, T, NODE_DIM),
            'nbr2_mask': (N, K, K2),
            'y': (N,),
            'y_persist': (N,),
            'stop_conf': (N,),
        }

    def _build_cache(self, cache_dir: str, verbose: bool):
        os.makedirs(cache_dir, exist_ok=True)
        N = len(self.samples)
        shapes = self._cache_shapes(N)
        mm = {}
        for k in _CACHE_KEYS_STORED:
            dtype = np.int64 if k in ('y', 'y_persist') else np.float16
            mm[k] = np.lib.format.open_memmap(
                os.path.join(cache_dir, f'{k}.npy'),
                mode='w+',
                dtype=dtype,
                shape=shapes[k],
            )
        it = tqdm(range(N), desc=f'[TSEM/{self.split}] 캐시') if verbose else range(N)
        for i in it:
            s = self._assemble_np(i)
            for k in _CACHE_KEYS_STORED:
                mm[k][i] = s[k]
        for k in _CACHE_KEYS_STORED:
            mm[k].flush()
        with open(os.path.join(cache_dir, 'samples.pkl'), 'wb') as f:
            pickle.dump(self.samples, f)
        open(os.path.join(cache_dir, '.done'), 'w').close()
        self._cache = {
            k: np.load(os.path.join(cache_dir, f'{k}.npy'))
            for k in _CACHE_KEYS_STORED
        }
        if verbose:
            print(f'[TSEM/{self.split}] 캐시 저장: {cache_dir}')

    def _load_or_build_v_future(self, cache_dir, verbose: bool) -> np.ndarray:
        """샘플별 v(t+H) (raw speed, NODE_COLS[2]) — 캐시 폴더 사이드카로 저장/재사용.
        기존 캐시 스키마·키는 건드리지 않음. 결측(미래 프레임에 ego 없음)은 0.0 —
        라벨이 존재하는 샘플만 인덱스에 들어오므로 실제로는 거의 발생 안 함."""
        path = os.path.join(cache_dir, 'v_future.npy') if cache_dir else None
        if path and os.path.exists(path):
            return np.load(path)
        v = np.zeros(len(self.samples), dtype=np.float32)
        it = (
            tqdm(range(len(self.samples)), desc=f'[TSEM/{self.split}] v_future')
            if verbose else range(len(self.samples))
        )
        for i in it:
            csv_path, ego_id, _, future_frame = self.samples[i]
            fd = _get_file_data(csv_path)
            node = fd.frame_node.get(future_frame, {}).get(ego_id)
            v[i] = float(node[2]) if node is not None else 0.0
        if path:
            np.save(path, v)
        return v

    def _setup_norm(self, cache_dir, verbose: bool):
        if not self.ego_relative:
            self._norm = None
            return
        stats_path = os.path.join(cache_dir, 'norm_stats.npz') if cache_dir else None
        if stats_path and os.path.exists(stats_path):
            z = np.load(stats_path)
            self._norm = {k: z[k].astype(np.float32) for k in z.files}
            return
        # ego_relative 미사용이 기본 — DRIFT 확장 시 train.py 패턴 재사용 가능
        self._norm = None

    def __len__(self) -> int:
        return len(self.samples)

    def _get_aug_rng(self) -> np.random.Generator:
        """워커 프로세스별로 독립된 RNG를 지연 생성 — DataLoader가 fork한 워커들이
        전부 같은 난수를 뽑는 걸 방지(num_workers>0일 때 흔한 함정)."""
        if self._aug_rng is None:
            info = torch.utils.data.get_worker_info()
            wid = info.id if info is not None else 0
            self._aug_rng = np.random.default_rng(self._augment_seed + wid)
        return self._aug_rng

    def __getitem__(self, idx: int) -> dict:
        if self._cache is not None:
            c = self._cache
            node_seq = np.asarray(c['node_seq'][idx], dtype=np.float32)
            nbr_node_seqs = np.asarray(c['nbr_node_seqs'][idx], dtype=np.float32)
            nbr_mask = np.asarray(c['nbr_mask'][idx], dtype=np.float32)
            nbr2_node_seqs = np.asarray(c['nbr2_node_seqs'][idx], dtype=np.float32)
            nbr2_mask = np.asarray(c['nbr2_mask'][idx], dtype=np.float32)
            if self.augment is not None:
                node_seq, nbr_node_seqs, nbr_mask, nbr2_node_seqs, nbr2_mask = self.augment.apply(
                    node_seq, nbr_node_seqs, nbr_mask, nbr2_node_seqs, nbr2_mask,
                    rng=self._get_aug_rng(),
                )
            edge_fn = _EDGE_FN_BY_VARIANT[self.edge_feat_variant]
            edge_seqs, nbr2_edge_seqs = edge_fn(
                node_seq, nbr_node_seqs, nbr2_node_seqs
            )
            csv_path, ego_id, window_frames, future_frame = self.samples[idx]
            out = {
                'node_seq': torch.from_numpy(node_seq),
                'nbr_node_seqs': torch.from_numpy(nbr_node_seqs),
                'edge_seqs': torch.from_numpy(edge_seqs),
                'nbr_mask': torch.from_numpy(nbr_mask),
                'nbr2_node_seqs': torch.from_numpy(nbr2_node_seqs),
                'nbr2_edge_seqs': torch.from_numpy(nbr2_edge_seqs),
                'nbr2_mask': torch.from_numpy(nbr2_mask),
                'y': torch.tensor(int(c['y'][idx]), dtype=torch.long),
                'y_persist': torch.tensor(int(c['y_persist'][idx]), dtype=torch.long),
                'stop_conf': torch.tensor(float(c['stop_conf'][idx]), dtype=torch.float32),
                **({'v_future': torch.tensor(float(self._v_future[idx]), dtype=torch.float32)}
                   if self._v_future is not None else {}),
                'meta': {
                    'object_id': ego_id,
                    'frame': window_frames[-1],
                    'future_frame': future_frame,
                    'H': self.H,
                    'W': self.W,
                    'csv_path': os.path.basename(csv_path),
                    'window_frames': window_frames,
                },
            }
            return out

        s = self._assemble_np(idx)
        if self.augment is not None:
            s['node_seq'], s['nbr_node_seqs'], s['nbr_mask'], s['nbr2_node_seqs'], s['nbr2_mask'] = (
                self.augment.apply(
                    s['node_seq'], s['nbr_node_seqs'], s['nbr_mask'],
                    s['nbr2_node_seqs'], s['nbr2_mask'], rng=self._get_aug_rng(),
                )
            )
        edge_fn = _EDGE_FN_BY_VARIANT[self.edge_feat_variant]
        edge_seqs, nbr2_edge_seqs = edge_fn(
            s['node_seq'], s['nbr_node_seqs'], s['nbr2_node_seqs']
        )
        tensor_keys = [k for k in _CACHE_KEYS_STORED if k not in ('y', 'y_persist', 'stop_conf')]
        out = {k: torch.from_numpy(s[k]) for k in tensor_keys}
        out['edge_seqs'] = torch.from_numpy(edge_seqs)
        out['nbr2_edge_seqs'] = torch.from_numpy(nbr2_edge_seqs)
        out['y'] = torch.tensor(int(s['y']), dtype=torch.long)
        out['y_persist'] = torch.tensor(int(s['y_persist']), dtype=torch.long)
        out['stop_conf'] = torch.tensor(float(s['stop_conf']), dtype=torch.float32)
        if self._v_future is not None:
            out['v_future'] = torch.tensor(float(self._v_future[idx]), dtype=torch.float32)
        out['meta'] = s['meta']
        return out

    def _assemble_np(self, idx: int) -> dict:
        csv_path, ego_id, window_frames, future_frame = self.samples[idx]
        fd = _get_file_data(csv_path)
        t = window_frames[-1]

        node_seq = np.zeros((self.W, NODE_DIM), dtype=np.float32)
        for ti, frame in enumerate(window_frames):
            if ego_id in fd.frame_node.get(frame, {}):
                node_seq[ti] = fd.frame_node[frame][ego_id]

        tree, obj_ids, positions = fd.get_kdtree(t)
        ego_pos = fd.frame_node[t][ego_id][:2]

        if self.neighbor_mode == 'count':
            k_query = min(self.K + 1, len(obj_ids))
            _, knn_idx = tree.query(ego_pos, k=k_query)
            knn_idx = np.atleast_1d(knn_idx)
            candidate_indices = [int(i) for i in knn_idx]
        else:
            candidate_indices = tree.query_ball_point(ego_pos, r=self.radius)

        nbr_candidates = []
        for idx_val in candidate_indices:
            oid = int(obj_ids[idx_val])
            if oid == ego_id:
                continue
            d = float(np.sqrt(np.sum((positions[idx_val] - ego_pos) ** 2)))
            nbr_candidates.append((d, oid, idx_val))
        nbr_candidates.sort(key=lambda x: x[0])
        nbr_candidates = nbr_candidates[: self.K]

        nbr_ids = [oid for _, oid, _ in nbr_candidates]
        nbr_pos_idx = [idx_val for _, _, idx_val in nbr_candidates]
        n_real = len(nbr_ids)

        nbr_node_seqs = np.zeros((self.K, self.W, NODE_DIM), dtype=np.float32)
        edge_seqs = np.zeros((self.K, self.W, EDGE_DIM), dtype=np.float32)
        for ki, nbr_id in enumerate(nbr_ids):
            for ti, frame in enumerate(window_frames):
                fn = fd.frame_node.get(frame, {})
                if nbr_id in fn and ego_id in fn:
                    nbr_node_seqs[ki, ti] = fn[nbr_id]
                    edge_seqs[ki, ti] = _compute_edge_feat(fn[ego_id], fn[nbr_id])

        nbr_mask = np.zeros(self.K, dtype=np.float32)
        nbr_mask[:n_real] = 1.0

        if self.use_2hop:
            nbr2_node_seqs = np.zeros(
                (self.K, self.K2, self.W, NODE_DIM), dtype=np.float32
            )
            nbr2_edge_seqs = np.zeros(
                (self.K, self.K2, self.W, EDGE_DIM), dtype=np.float32
            )
            nbr2_mask = np.zeros((self.K, self.K2), dtype=np.float32)
            for ki, (nbr_id, pos_idx) in enumerate(zip(nbr_ids, nbr_pos_idx)):
                nbr_pos = positions[pos_idx]
                tree2, obj_ids2, pos2 = fd.get_kdtree(t)
                if self.neighbor_mode == 'count':
                    k2 = min(self.K2 + 2, len(obj_ids2))
                    _, idx2 = tree2.query(nbr_pos, k=k2)
                    idx2 = np.atleast_1d(idx2)
                    cand2 = [int(i) for i in idx2]
                else:
                    cand2 = tree2.query_ball_point(nbr_pos, r=self.radius)
                nbr2_cands = []
                for j in cand2:
                    oid2 = int(obj_ids2[j])
                    if oid2 in (ego_id, nbr_id):
                        continue
                    d2 = float(np.sqrt(np.sum((pos2[j] - nbr_pos) ** 2)))
                    nbr2_cands.append((d2, oid2))
                nbr2_cands.sort(key=lambda x: x[0])
                nbr2_cands = nbr2_cands[: self.K2]
                for k2i, (_, oid2) in enumerate(nbr2_cands):
                    nbr2_mask[ki, k2i] = 1.0
                    for ti, frame in enumerate(window_frames):
                        fn = fd.frame_node.get(frame, {})
                        if oid2 in fn and nbr_id in fn:
                            nbr2_node_seqs[ki, k2i, ti] = fn[oid2]
                            nbr2_edge_seqs[ki, k2i, ti] = _compute_edge_feat(
                                fn[nbr_id], fn[oid2]
                            )
        else:
            nbr2_node_seqs = np.zeros(
                (self.K, 0, self.W, NODE_DIM), dtype=np.float32
            )
            nbr2_edge_seqs = np.zeros(
                (self.K, 0, self.W, EDGE_DIM), dtype=np.float32
            )
            nbr2_mask = np.zeros((self.K, 0), dtype=np.float32)

        if self.ego_relative:
            ref = node_seq[-1, :2].copy()

            def _rel(arr):
                present = np.abs(arr).sum(-1, keepdims=True) > 0
                shift = np.zeros(arr.shape[-1], dtype=np.float32)
                shift[0], shift[1] = ref[0], ref[1]
                return arr - shift * present

            node_seq = _rel(node_seq)
            nbr_node_seqs = _rel(nbr_node_seqs)
            nbr2_node_seqs = _rel(nbr2_node_seqs)

        y = fd.future_window_state(t, future_frame, ego_id, stop_persist_delta=self.stop_persist_delta)
        y_persist = fd.instant_state(t, ego_id)
        stop_conf = fd.stop_confidence(future_frame, ego_id)

        return {
            'node_seq': node_seq,
            'nbr_node_seqs': nbr_node_seqs,
            'edge_seqs': edge_seqs,
            'nbr_mask': nbr_mask,
            'nbr2_node_seqs': nbr2_node_seqs,
            'nbr2_edge_seqs': nbr2_edge_seqs,
            'nbr2_mask': nbr2_mask,
            'y': np.int64(y),
            'y_persist': np.int64(y_persist),
            'stop_conf': np.array(stop_conf, dtype=np.float32),
            'meta': {
                'object_id': ego_id,
                'frame': t,
                'future_frame': future_frame,
                'H': self.H,
                'W': self.W,
                'csv_path': os.path.basename(csv_path),
                'n_neighbors': n_real,
                'window_frames': window_frames,
            },
        }


def _collate_fn(batch: list) -> dict:
    keys_tensor = [
        'node_seq', 'nbr_node_seqs', 'edge_seqs', 'nbr_mask',
        'nbr2_node_seqs', 'nbr2_edge_seqs', 'nbr2_mask', 'y', 'y_persist', 'stop_conf',
    ]
    out = {k: torch.stack([b[k] for b in batch]) for k in keys_tensor}
    if 'v_future' in batch[0]:
        out['v_future'] = torch.stack([b['v_future'] for b in batch])
    out['meta'] = [b['meta'] for b in batch]
    return out


def build_tsem_dataloaders(
    csv_files: List[str],
    W: int = 10,
    H: int = 10,
    radius: float = 20.0,
    K_max: int = 6,
    K_max2: int = 4,
    batch_size: int = 512,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    num_workers: int = 4,
    neighbor_mode: str = 'hybrid',
    ego_relative: bool = False,
    verbose: bool = True,
    use_cache: bool = True,
    augment: 'TsemAugment' = None,
    stop_persist_delta: int = 2,
    with_v_future: bool = False,
    edge_feat_variant: str = 'legacy',
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    kwargs = dict(
        W=W, H=H, radius=radius, K_max=K_max, K_max2=K_max2,
        train_ratio=train_ratio, val_ratio=val_ratio,
        neighbor_mode=neighbor_mode, ego_relative=ego_relative,
        verbose=verbose, use_cache=use_cache, augment=augment,
        stop_persist_delta=stop_persist_delta, with_v_future=with_v_future,
        edge_feat_variant=edge_feat_variant,
    )
    train_ds = TSEMFutureStateDataset(csv_files, split='train', **kwargs)
    val_ds = TSEMFutureStateDataset(csv_files, split='val', **kwargs)
    test_ds = TSEMFutureStateDataset(csv_files, split='test', **kwargs)
    loader_kwargs = dict(
        batch_size=batch_size,
        collate_fn=_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return (
        DataLoader(train_ds, shuffle=True, **loader_kwargs),
        DataLoader(val_ds, shuffle=False, **loader_kwargs),
        DataLoader(test_ds, shuffle=False, **loader_kwargs),
    )


def summarize_label_distribution(
    csv_files: List[str], W: int, H: int, stop_persist_delta: int = 2,
) -> dict:
    """instant state(t+H) 클래스 비율 (train split 샘플)."""
    from collections import Counter

    ds = TSEMFutureStateDataset(
        csv_files, W=W, H=H, split='train', use_cache=False, verbose=False,
        stop_persist_delta=stop_persist_delta,
    )
    cnt = Counter()
    if ds._cache is not None:
        for i in range(len(ds)):
            cnt[int(ds._cache['y'][i])] += 1
    else:
        for csv_path, ego_id, _window_frames, future_frame in ds.samples:
            fd = _get_file_data(csv_path)
            anchor = _window_frames[-1]
            cnt[fd.future_window_state(anchor, future_frame, ego_id,
                                       stop_persist_delta=stop_persist_delta)] += 1
    total = sum(cnt.values()) or 1
    return {
        name: {'count': cnt[i], 'ratio': cnt[i] / total}
        for i, name in enumerate(CLASS_NAMES)
    }
