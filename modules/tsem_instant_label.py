"""
TSEM 순간 상태 라벨 — §1.2 instant state(f)
============================================
CSV `category`(±6 smear) 미사용. speed·lane_id로 프레임별 3-class 정의.

클래스: 0=stop, 1=lane_change, 2=normal
우선순위: stop > lane_change > normal
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ET-NAGraphSAGE와 동일한 클래스 인덱스 (비교·혼동 방지)
STOP = 0
LANE_CHANGE = 1
NORMAL = 2
CLASS_NAMES = ('stop', 'lane_change', 'normal')

SPEED_STOP_THRESH = 1.0


def instant_state(
    speed: float,
    lane_id: str,
    prev_lane_id: Optional[str],
) -> int:
    """단일 프레임 순간 상태."""
    if speed <= SPEED_STOP_THRESH:
        return STOP
    if (
        prev_lane_id is not None
        and lane_id
        and prev_lane_id
        and lane_id != prev_lane_id
    ):
        return LANE_CHANGE
    return NORMAL


def _norm_lane(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ''
    s = str(x).strip()
    return '' if s.lower() in ('nan', 'none') else s


def build_instant_labels_for_track(
    frames: Sequence[int],
    speeds: Sequence[float],
    lane_ids: Sequence,
) -> Dict[int, int]:
    """한 차량 궤적에 대해 frame → state(f) 맵."""
    out: Dict[int, int] = {}
    prev_lane: Optional[str] = None
    for fr, spd, lid in zip(frames, speeds, lane_ids):
        lane = _norm_lane(lid)
        out[int(fr)] = instant_state(float(spd), lane, prev_lane)
        if lane:
            prev_lane = lane
    return out


def build_instant_labels_from_dataframe(df: pd.DataFrame) -> Dict[Tuple[int, int], int]:
    """
    DataFrame(frame, object_id, speed, lane_id) → {(frame, object_id): state}.
  lane_id 열이 없으면 LC는 항상 0으로 잡히지 않고 normal만 가능(경고는 호출부).
    """
    required = {'frame', 'object_id', 'speed'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'필수 열 없음: {missing}')
    has_lane = 'lane_id' in df.columns

    labels: Dict[Tuple[int, int], int] = {}
    for oid, g in df.groupby('object_id'):
        g = g.sort_values('frame')
        frames = g['frame'].to_numpy()
        speeds = g['speed'].to_numpy(dtype=np.float64)
        lanes = g['lane_id'].to_numpy() if has_lane else [''] * len(g)
        track = build_instant_labels_for_track(frames, speeds, lanes)
        for fr, st in track.items():
            labels[(int(fr), int(oid))] = st
    return labels


def state_at_future(
    labels: Dict[Tuple[int, int], int],
    object_id: int,
    future_frame: int,
) -> Optional[int]:
    return labels.get((int(future_frame), int(object_id)))


def future_window_state(
    labels: Dict[Tuple[int, int], int],
    object_id: int,
    future_frame: int,
    window_frames: Sequence[int],
) -> Optional[int]:
    """
    미래 상태 라벨 (A안, 2026-07-07) — maneuver anticipation 문헌 표준 적용.
    stop: future_frame 시점 그대로 (지속 상태라 point 정의로 충분, 실측 point/window 배율 1.46x).
    lane_change: window_frames(=(t, t+H], 앵커 t 다음 프레임부터 future_frame까지) 구간 내
      전이가 한 번이라도 발생하면 LC (순간 이벤트라 point 정의로는 8.5x 과소포착됨).
    우선순위: stop > lane_change > normal (기존 instant_state와 동일 순서).
    """
    pt = labels.get((int(future_frame), int(object_id)))
    if pt is None:
        return None
    if pt == STOP:
        return STOP
    oid = int(object_id)
    for f in window_frames:
        if labels.get((int(f), oid)) == LANE_CHANGE:
            return LANE_CHANGE
    return NORMAL


def count_class_distribution(
    labels: Dict[Tuple[int, int], int],
) -> Dict[int, int]:
    from collections import Counter
    c = Counter(labels.values())
    return {k: c.get(k, 0) for k in (STOP, LANE_CHANGE, NORMAL)}
