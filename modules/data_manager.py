"""
ET-NAGraphSAGE Temporal Dataloader
===================================
CSV(frame, object_id, position_x/z, speed, dir_x/z, accel, category) 파일을
슬라이딩 윈도우 방식으로 읽어 시계열 샘플을 생성한다.

출력 샘플 구조 (한 차량 at frame t):
  node_seq        : [T, 6]       - ego 차량의 T프레임 노드 피처
  nbr_node_seqs   : [K1, T, 6]   - K1 이웃 차량들의 T프레임 노드 피처
  edge_seqs       : [K1, T, 5]   - K1 이웃과의 T프레임 엣지 피처
  nbr_mask        : [K1]         - 실제 이웃 여부 (1=real, 0=padded)
  nbr2_node_seqs  : [K1, K2, T, 6] - 2-hop 이웃 노드 피처
  nbr2_edge_seqs  : [K1, K2, T, 5] - 2-hop 엣지 피처 (nbr→nbr2 기준)
  nbr2_mask       : [K1, K2]     - 2-hop 실제 이웃 여부
  y               : int          - frame t 시점 레이블 (0=stop, 1=lane_change, 2=normal)
  meta            : dict         - 디버깅용

피처 정의:
  노드 피처 (6D): [position_x, position_z, speed, direction_x, direction_z, acceleration]
  엣지 피처 (5D): [rel_speed, rel_accel, rel_dir_x, rel_dir_z, distance]

이웃 선택: 현재 프레임 t에서 KNN(position_x, position_z) 기준
  1-hop: ego 반경 내 K1개
  2-hop: 각 1-hop 이웃의 반경 내 K2개 (ego 및 해당 이웃 제외)

시간 분할 (Temporal Split):
  각 CSV 파일 내 프레임을 시간 순서로 분할 → train/val/test 범위가 겹치지 않음
"""

import os
import hashlib
import pickle
from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial import cKDTree
from tqdm import tqdm

LABEL_MAP = {'stop': 0, 'lane_change': 1, 'normal_driving': 2}
NODE_COLS  = ['position_x', 'position_z', 'speed', 'direction_x', 'direction_z', 'acceleration']
NODE_DIM   = 6
EDGE_DIM   = 5

# 조립된 샘플 텐서 캐시 루트 (프로젝트 루트/cache). git 동기화 제외(.gitignore).
CACHE_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache')

# 조립 샘플의 전체 텐서 키 (모델 입력 / 비캐시 경로)
_CACHE_KEYS = [
    'node_seq', 'nbr_node_seqs', 'edge_seqs', 'nbr_mask',
    'nbr2_node_seqs', 'nbr2_edge_seqs', 'nbr2_mask', 'y',
]

# 디스크 효율화: 엣지(edge_seqs, nbr2_edge_seqs)는 노드에서 파생되므로 저장하지 않고
# 로드 시 재계산한다. 저장 텐서는 float16(y만 int64). fp16(2×)+엣지제거(~1.8×) = ~3.6× 절감.
_CACHE_KEYS_STORED = [
    'node_seq', 'nbr_node_seqs', 'nbr_mask',
    'nbr2_node_seqs', 'nbr2_mask', 'y',
]
_CACHE_FMT = 'f16ne'   # float16 + no-edge. 캐시 키에 포함해 기존 fp32 캐시와 분리.


# ─────────────────────────────────────────────────────────────────────────────
# 파일 단위 전처리 (캐시 보관)
# ─────────────────────────────────────────────────────────────────────────────

class _FileData:
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        df = df[df['category'].isin(LABEL_MAP)].copy()
        df['label'] = df['category'].map(LABEL_MAP).astype(np.int64)
        df = df.sort_values(['frame', 'object_id']).reset_index(drop=True)

        self.csv_path      = csv_path
        self.sorted_frames = sorted(df['frame'].unique().tolist())
        self.n_frames      = len(self.sorted_frames)

        diffs = np.diff(self.sorted_frames)
        self.frame_step = int(np.median(diffs)) if len(diffs) > 0 else 1
        self.max_gap    = self.frame_step * 2

        self.frame_node: Dict[int, Dict[int, np.ndarray]] = {}
        self.frame_label: Dict[int, Dict[int, int]]       = {}

        for frame, grp in df.groupby('frame'):
            node_dict  = {}
            label_dict = {}
            for _, row in grp.iterrows():
                oid = int(row['object_id'])
                node_dict[oid]  = np.array(
                    [row[c] for c in NODE_COLS], dtype=np.float32)
                label_dict[oid] = int(row['label'])
            self.frame_node[frame]  = node_dict
            self.frame_label[frame] = label_dict

        self.obj_frames: Dict[int, List[int]] = defaultdict(list)
        for frame, grp in df.groupby('frame'):
            for oid in grp['object_id'].unique():
                self.obj_frames[int(oid)].append(frame)
        for oid in self.obj_frames:
            self.obj_frames[oid].sort()

        self._kdtree_cache: Dict[int, Tuple] = {}

    def get_kdtree(self, frame: int) -> Tuple:
        if frame not in self._kdtree_cache:
            nd = self.frame_node[frame]
            obj_ids   = np.array(list(nd.keys()), dtype=np.int64)
            positions = np.array([nd[oid][:2] for oid in obj_ids], dtype=np.float32)
            tree = cKDTree(positions)
            self._kdtree_cache[frame] = (tree, obj_ids, positions)
        return self._kdtree_cache[frame]


_FILE_CACHE: Dict[str, _FileData] = {}

def _get_file_data(csv_path: str) -> _FileData:
    if csv_path not in _FILE_CACHE:
        _FILE_CACHE[csv_path] = _FileData(csv_path)
    return _FILE_CACHE[csv_path]


# ─────────────────────────────────────────────────────────────────────────────
# 샘플 인덱스 빌드
# ─────────────────────────────────────────────────────────────────────────────

def _build_sample_index(
    fd: _FileData,
    T: int,
    split: str,
    train_ratio: float,
    val_ratio: float,
) -> List[Tuple[str, int, List[int]]]:
    n = fd.n_frames
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))

    if split == 'train':
        valid_frame_set = set(fd.sorted_frames[:train_end])
    elif split == 'val':
        valid_frame_set = set(fd.sorted_frames[train_end:val_end])
    else:
        valid_frame_set = set(fd.sorted_frames[val_end:])

    samples = []
    for oid, frames in fd.obj_frames.items():
        if len(frames) < T:
            continue
        for i in range(T - 1, len(frames)):
            window = frames[i - T + 1: i + 1]
            if window[-1] not in valid_frame_set:
                continue
            gaps = [window[k+1] - window[k] for k in range(len(window)-1)]
            if any(g > fd.max_gap for g in gaps):
                continue
            if oid not in fd.frame_label.get(window[-1], {}):
                continue
            samples.append((fd.csv_path, oid, window))
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 엣지 피처 계산
# ─────────────────────────────────────────────────────────────────────────────

def _compute_edge_feat(src_node: np.ndarray, dst_node: np.ndarray) -> np.ndarray:
    """
    src_node → dst_node 방향 엣지 피처.
    [rel_speed, rel_accel, rel_dir_x, rel_dir_z, distance]
    """
    rel_speed = dst_node[2] - src_node[2]
    rel_accel = dst_node[5] - src_node[5]
    rel_dx    = dst_node[3] - src_node[3]
    rel_dz    = dst_node[4] - src_node[4]
    dist      = float(np.sqrt((dst_node[0]-src_node[0])**2 +
                               (dst_node[1]-src_node[1])**2))
    return np.array([rel_speed, rel_accel, rel_dx, rel_dz, dist], dtype=np.float32)


def _recompute_edges(node_seq: np.ndarray, nbr_node_seqs: np.ndarray,
                     nbr2_node_seqs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """저장된 노드 시퀀스에서 엣지 피처를 벡터화 재계산 (fp32).
    정의는 _compute_edge_feat와 동일. 결측(all-zero) 이웃 슬롯은 0으로 유지한다."""
    ns  = node_seq.astype(np.float32)          # [T,6]
    nbr = nbr_node_seqs.astype(np.float32)      # [K,T,6]

    def _edges(src, dst):                       # [...,6],[...,6] → [...,5]
        rs = dst[..., 2] - src[..., 2]
        ra = dst[..., 5] - src[..., 5]
        rx = dst[..., 3] - src[..., 3]
        rz = dst[..., 4] - src[..., 4]
        di = np.sqrt((dst[..., 0]-src[..., 0])**2 + (dst[..., 1]-src[..., 1])**2)
        return np.stack([rs, ra, rx, rz, di], axis=-1).astype(np.float32)

    # 1-hop: ego(broadcast) → 이웃
    src1 = np.broadcast_to(ns[None], nbr.shape)                 # [K,T,6]
    edge_seqs = _edges(src1, nbr)                                # [K,T,5]
    edge_seqs *= (np.abs(nbr).sum(-1, keepdims=True) > 0)        # 결측 0 유지

    # 2-hop: 1-hop 이웃(broadcast) → 2-hop 이웃
    if nbr2_node_seqs is not None and nbr2_node_seqs.size:
        nbr2 = nbr2_node_seqs.astype(np.float32)                # [K,K2,T,6]
        src2 = np.broadcast_to(nbr[:, None], nbr2.shape)        # [K,K2,T,6]
        nbr2_edge_seqs = _edges(src2, nbr2)                      # [K,K2,T,5]
        nbr2_edge_seqs *= (np.abs(nbr2).sum(-1, keepdims=True) > 0)
    else:
        nbr2_edge_seqs = np.zeros((nbr.shape[0], 0, nbr.shape[1], EDGE_DIM),
                                  dtype=np.float32)
    return edge_seqs, nbr2_edge_seqs


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TemporalVehicleDataset(Dataset):
    """
    ET-NAGraphSAGE용 시계열 차량 데이터셋 (2-hop 지원).

    Args:
        csv_files  : CSV 파일 경로 목록
        T          : 시퀀스 길이 (프레임 수)
        radius     : 이웃 탐색 반경 (m)
        K_max      : 1-hop 최대 이웃 수
        K_max2     : 2-hop 최대 이웃 수 (0이면 2-hop 비활성화)
        split      : 'train' | 'val' | 'test'
        train_ratio, val_ratio : 시간 분할 비율
        verbose    : 진행 메시지 출력 여부
    """

    def __init__(
        self,
        csv_files: List[str],
        T: int             = 10,
        radius: float      = 20.0,
        K_max: int         = 6,
        K_max2: int        = 4,
        split: str         = 'train',
        train_ratio: float = 0.70,
        val_ratio: float   = 0.15,
        neighbor_mode: str = 'hybrid',   # 'hybrid' | 'count' | 'radius' (Ablation E)
        verbose: bool      = True,
        use_cache: bool    = True,       # 조립 텐서 디스크 캐시 사용
    ):
        self.T             = T
        self.radius        = radius
        self.K             = K_max
        self.K2            = K_max2
        self.use_2hop      = (K_max2 > 0)
        self.split         = split
        self.train_ratio   = train_ratio
        self.val_ratio     = val_ratio
        # Ablation E: 이웃 선택 정책
        #   'hybrid' (기본): 반경 r 내에서 최근접 K대 (radius 필터 + count 상한)
        #   'count'        : 반경 무관, 최근접 K대 (NAGraphSAGE 최고기록 방식)
        #   'radius'       : 반경 r 내 전부 (K_max로 상한, 개수 가변)
        assert neighbor_mode in ('hybrid', 'count', 'radius')
        self.neighbor_mode = neighbor_mode

        self.samples: List[Tuple[str, int, List[int]]] = []
        self._cache: Dict[str, np.ndarray] = None  # {key: memmap} (캐시 로드 시)

        cache_dir = os.path.join(CACHE_ROOT, self._cache_key(csv_files)) if use_cache else None

        # ── 캐시가 이미 있으면: CSV/그래프 계산 없이 즉시 로드 ────────────────
        if cache_dir is not None and os.path.exists(os.path.join(cache_dir, '.done')):
            with open(os.path.join(cache_dir, 'samples.pkl'), 'rb') as f:
                self.samples = pickle.load(f)
            self._cache = {k: np.load(os.path.join(cache_dir, f'{k}.npy'), mmap_mode='r')
                           for k in _CACHE_KEYS_STORED}
            if verbose:
                print(f'[{split}] 캐시 로드(fp16, 엣지 파생): {len(self.samples):,}개  ({cache_dir})')
            return

        # ── 캐시 없음: 기존 방식으로 샘플 인덱스 빌드 ─────────────────────────
        for csv_path in (tqdm(csv_files, desc=f'[{split}] 인덱스 빌드') if verbose else csv_files):
            fd = _get_file_data(csv_path)
            self.samples += _build_sample_index(fd, T, split, train_ratio, val_ratio)

        if verbose:
            hop_str = f', K_max2={K_max2}' if self.use_2hop else ''
            print(f'[{split}] 총 샘플: {len(self.samples):,}개  '
                  f'(파일 {len(csv_files)}개, T={T}, radius={radius}m, K_max={K_max}{hop_str})')

        # ── 캐시 생성 (조립 텐서를 1회 계산해 디스크에 저장) ─────────────────
        if cache_dir is not None:
            self._build_cache(cache_dir, split, verbose)

    # ── 캐시 유틸 ────────────────────────────────────────────────────────────
    def _cache_key(self, csv_files: List[str]) -> str:
        """샘플 조립 결과를 결정하는 모든 요소로 캐시 키 생성.
        CSV 목록/크기/수정시각 + 그래프·분할 파라미터 → 하나라도 바뀌면 새 캐시."""
        h = hashlib.md5()
        for p in sorted(csv_files):
            st = os.stat(p)
            h.update(f'{os.path.basename(p)}:{st.st_size}:{int(st.st_mtime)}'.encode())
        sig = (f'T{self.T}_r{self.radius}_K{self.K}-{self.K2}'
               f'_{self.neighbor_mode}_{_CACHE_FMT}'
               f'_tr{self.train_ratio}_vr{self.val_ratio}')
        h.update(sig.encode())
        return f'{self.split}_{sig}_{h.hexdigest()[:10]}'

    def _cache_shapes(self, N: int) -> Dict[str, Tuple]:
        """저장 텐서(_CACHE_KEYS_STORED)의 shape. 엣지는 저장 안 함(파생계산)."""
        K, K2, T = self.K, self.K2, self.T
        return {
            'node_seq':       (N, T, NODE_DIM),
            'nbr_node_seqs':  (N, K, T, NODE_DIM),
            'nbr_mask':       (N, K),
            'nbr2_node_seqs': (N, K, self.K2, T, NODE_DIM),
            'nbr2_mask':      (N, K, self.K2),
            'y':              (N,),
        }

    def _build_cache(self, cache_dir: str, split: str, verbose: bool):
        os.makedirs(cache_dir, exist_ok=True)
        N = len(self.samples)
        shapes = self._cache_shapes(N)
        mm = {}
        for k in _CACHE_KEYS_STORED:
            dtype = np.int64 if k == 'y' else np.float16   # 부동소수 텐서는 fp16 저장
            mm[k] = np.lib.format.open_memmap(
                os.path.join(cache_dir, f'{k}.npy'), mode='w+',
                dtype=dtype, shape=shapes[k])

        it = tqdm(range(N), desc=f'[{split}] 캐시 생성') if verbose else range(N)
        for i in it:
            s = self._assemble_np(i)
            for k in _CACHE_KEYS_STORED:
                mm[k][i] = s[k]                              # float16으로 자동 캐스팅
        for k in _CACHE_KEYS_STORED:
            mm[k].flush()

        with open(os.path.join(cache_dir, 'samples.pkl'), 'wb') as f:
            pickle.dump(self.samples, f)
        open(os.path.join(cache_dir, '.done'), 'w').close()

        # 이후 __getitem__은 memmap 슬라이스 + 엣지 재계산만 → CPU 부하 최소화
        self._cache = {k: np.load(os.path.join(cache_dir, f'{k}.npy'), mmap_mode='r')
                       for k in _CACHE_KEYS_STORED}
        if verbose:
            print(f'[{split}] 캐시 저장 완료: {cache_dir}')

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        csv_path, ego_id, window_frames = self.samples[idx]
        t = window_frames[-1]

        # ── 캐시가 있으면 memmap 슬라이스(fp16→fp32) + 엣지 재계산 ──
        if self._cache is not None:
            c = self._cache
            node_seq       = np.asarray(c['node_seq'][idx],       dtype=np.float32)
            nbr_node_seqs  = np.asarray(c['nbr_node_seqs'][idx],  dtype=np.float32)
            nbr_mask       = np.asarray(c['nbr_mask'][idx],       dtype=np.float32)
            nbr2_node_seqs = np.asarray(c['nbr2_node_seqs'][idx], dtype=np.float32)
            nbr2_mask      = np.asarray(c['nbr2_mask'][idx],      dtype=np.float32)
            edge_seqs, nbr2_edge_seqs = _recompute_edges(
                node_seq, nbr_node_seqs, nbr2_node_seqs)
            out = {
                'node_seq':       torch.from_numpy(node_seq),
                'nbr_node_seqs':  torch.from_numpy(nbr_node_seqs),
                'edge_seqs':      torch.from_numpy(edge_seqs),
                'nbr_mask':       torch.from_numpy(nbr_mask),
                'nbr2_node_seqs': torch.from_numpy(nbr2_node_seqs),
                'nbr2_edge_seqs': torch.from_numpy(nbr2_edge_seqs),
                'nbr2_mask':      torch.from_numpy(nbr2_mask),
                'y':              torch.tensor(int(c['y'][idx]), dtype=torch.long),
                'meta': {
                    'object_id':     ego_id,
                    'frame':         t,
                    'csv_path':      os.path.basename(csv_path),
                    'n_neighbors':   int(nbr_mask.sum()),
                    'window_frames': window_frames,
                },
            }
            return out

        # 캐시 미사용/미생성 시: 즉석 조립 후 torch 변환
        s = self._assemble_np(idx)
        out = {k: torch.from_numpy(s[k]) for k in _CACHE_KEYS if k != 'y'}
        out['y']    = torch.tensor(int(s['y']), dtype=torch.long)
        out['meta'] = s['meta']
        return out

    def _assemble_np(self, idx: int) -> dict:
        """샘플 하나의 조립 텐서를 numpy로 계산 (KD-tree 이웃 탐색 + 피처 구성).
        캐시 생성 및 캐시 미사용 경로에서 호출된다."""
        csv_path, ego_id, window_frames = self.samples[idx]
        fd = _get_file_data(csv_path)
        t  = window_frames[-1]

        # ── 1. ego 노드 시퀀스 [T, 6] ──────────────────────────────────────
        node_seq = np.zeros((self.T, NODE_DIM), dtype=np.float32)
        for ti, frame in enumerate(window_frames):
            if ego_id in fd.frame_node.get(frame, {}):
                node_seq[ti] = fd.frame_node[frame][ego_id]

        # ── 2. 현재 프레임 t에서 KD-tree로 1-hop 이웃 탐색 ──────────────────
        tree, obj_ids, positions = fd.get_kdtree(t)
        ego_pos = fd.frame_node[t][ego_id][:2]

        # ── Ablation E: 이웃 후보 선택 ────────────────────────────────────
        if self.neighbor_mode == 'count':
            # 반경 무관 최근접 K대 (자기 자신 제외 위해 K+1 질의)
            k_query = min(self.K + 1, len(obj_ids))
            _, knn_idx = tree.query(ego_pos, k=k_query)
            knn_idx = np.atleast_1d(knn_idx)
            candidate_indices = [int(i) for i in knn_idx]
        else:
            # 'hybrid' / 'radius' : 반경 r 내 후보
            candidate_indices = tree.query_ball_point(ego_pos, r=self.radius)

        nbr_candidates = []
        for idx_val in candidate_indices:
            oid = int(obj_ids[idx_val])
            if oid == ego_id:
                continue
            d = float(np.sqrt(np.sum((positions[idx_val] - ego_pos) ** 2)))
            nbr_candidates.append((d, oid, idx_val))
        nbr_candidates.sort(key=lambda x: x[0])
        # 'radius'는 반경 내 전부(K_max 상한), 'hybrid'/'count'는 최근접 K대
        nbr_candidates = nbr_candidates[:self.K]

        nbr_ids        = [oid      for _, oid, _   in nbr_candidates]
        nbr_dists_at_t = [d        for d,   _, _   in nbr_candidates]
        nbr_pos_idx    = [idx_val  for _, _,  idx_val in nbr_candidates]

        n_real = len(nbr_ids)

        # ── 3. 1-hop 이웃 시퀀스 [K1, T, 6] + 엣지 [K1, T, 5] ───────────
        nbr_node_seqs = np.zeros((self.K, self.T, NODE_DIM), dtype=np.float32)
        edge_seqs     = np.zeros((self.K, self.T, EDGE_DIM), dtype=np.float32)

        for ki, nbr_id in enumerate(nbr_ids):
            for ti, frame in enumerate(window_frames):
                fn = fd.frame_node.get(frame, {})
                if nbr_id in fn and ego_id in fn:
                    nbr_node_seqs[ki, ti] = fn[nbr_id]
                    edge_seqs[ki, ti]     = _compute_edge_feat(fn[ego_id], fn[nbr_id])

        # ── 4. 마스크 [K1] ──────────────────────────────────────────────────
        nbr_mask = np.zeros(self.K, dtype=np.float32)
        nbr_mask[:n_real] = 1.0

        # ── 5. 2-hop 이웃 탐색 및 시퀀스 구성 ──────────────────────────────
        if self.use_2hop:
            nbr2_node_seqs = np.zeros((self.K, self.K2, self.T, NODE_DIM), dtype=np.float32)
            nbr2_edge_seqs = np.zeros((self.K, self.K2, self.T, EDGE_DIM), dtype=np.float32)
            nbr2_mask      = np.zeros((self.K, self.K2), dtype=np.float32)

            for ki, (nbr_id, pos_idx) in enumerate(zip(nbr_ids, nbr_pos_idx)):
                nbr_pos = positions[pos_idx]

                # nbr_id 주변 이웃 탐색 (같은 KD-tree 재사용)
                cand2 = tree.query_ball_point(nbr_pos, r=self.radius)
                nbr2_candidates = []
                for idx_val2 in cand2:
                    oid2 = int(obj_ids[idx_val2])
                    if oid2 == ego_id or oid2 == nbr_id:
                        continue
                    d2 = float(np.sqrt(np.sum((positions[idx_val2] - nbr_pos) ** 2)))
                    nbr2_candidates.append((d2, oid2))
                nbr2_candidates.sort(key=lambda x: x[0])
                nbr2_candidates = nbr2_candidates[:self.K2]

                nbr2_ids = [oid2 for _, oid2 in nbr2_candidates]
                nbr2_mask[ki, :len(nbr2_ids)] = 1.0

                for ki2, nbr2_id in enumerate(nbr2_ids):
                    for ti, frame in enumerate(window_frames):
                        fn = fd.frame_node.get(frame, {})
                        if nbr2_id in fn and nbr_id in fn:
                            nbr2_node_seqs[ki, ki2, ti] = fn[nbr2_id]
                            nbr2_edge_seqs[ki, ki2, ti] = _compute_edge_feat(
                                fn[nbr_id], fn[nbr2_id])
        else:
            # 2-hop 비활성화 시 빈 텐서
            nbr2_node_seqs = np.zeros((self.K, 0, self.T, NODE_DIM), dtype=np.float32)
            nbr2_edge_seqs = np.zeros((self.K, 0, self.T, EDGE_DIM), dtype=np.float32)
            nbr2_mask      = np.zeros((self.K, 0), dtype=np.float32)

        # ── 6. 레이블 ──────────────────────────────────────────────────────
        y = fd.frame_label[t][ego_id]

        return {
            'node_seq':       node_seq,
            'nbr_node_seqs':  nbr_node_seqs,
            'edge_seqs':      edge_seqs,
            'nbr_mask':       nbr_mask,
            'nbr2_node_seqs': nbr2_node_seqs,
            'nbr2_edge_seqs': nbr2_edge_seqs,
            'nbr2_mask':      nbr2_mask,
            'y':              np.int64(y),
            'meta': {
                'object_id':     ego_id,
                'frame':         t,
                'csv_path':      os.path.basename(csv_path),
                'n_neighbors':   n_real,
                'nbr_ids':       nbr_ids,
                'nbr_dists':     nbr_dists_at_t,
                'window_frames': window_frames,
            }
        }


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader 팩토리
# ─────────────────────────────────────────────────────────────────────────────

def _collate_fn(batch: list) -> dict:
    keys_tensor = [
        'node_seq', 'nbr_node_seqs', 'edge_seqs', 'nbr_mask',
        'nbr2_node_seqs', 'nbr2_edge_seqs', 'nbr2_mask', 'y',
    ]
    out = {k: torch.stack([b[k] for b in batch]) for k in keys_tensor}
    out['meta'] = [b['meta'] for b in batch]
    return out


def build_dataloaders(
    csv_files: List[str],
    T: int             = 10,
    radius: float      = 20.0,
    K_max: int         = 6,
    K_max2: int        = 4,
    batch_size: int    = 512,
    train_ratio: float = 0.70,
    val_ratio: float   = 0.15,
    num_workers: int   = 4,
    neighbor_mode: str = 'hybrid',
    verbose: bool      = True,
    use_cache: bool    = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    kwargs = dict(T=T, radius=radius, K_max=K_max, K_max2=K_max2,
                  train_ratio=train_ratio, val_ratio=val_ratio,
                  neighbor_mode=neighbor_mode, verbose=verbose,
                  use_cache=use_cache)
    train_ds = TemporalVehicleDataset(csv_files, split='train', **kwargs)
    val_ds   = TemporalVehicleDataset(csv_files, split='val',   **kwargs)
    test_ds  = TemporalVehicleDataset(csv_files, split='test',  **kwargs)

    loader_kwargs = dict(
        batch_size=batch_size,
        collate_fn=_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),  # 에포크마다 워커 재생성 방지
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
