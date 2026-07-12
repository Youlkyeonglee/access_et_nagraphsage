"""
openDD 위치 암기 분석 — position_memorization_entropy.py(공업탑)의 openDD 버전 (2026-07-12)
================================================================================
공업탑에서 쓴 것과 동일한 방법론(격자 셀 엔트로피 + train 셀 다수결을 test에 그대로 적용하는
"위치 룩업" baseline)을 openDD rdb1(Wolfsburg, 가장 큰 사이트)에 적용한다.

openDD는 아직 TSEM 학습 파이프라인(캐시·lane_id·저널 라벨)이 없으므로, 이 스크립트 안에서
직접 만든다:
  1. lane_id 부여 — nearest-lane(최단거리) 방식. §데이터 설계 3단계 확장검증에서 확인한 대로
     거리 임계값(THRESH_M) 밖 포인트는 "차선 미배정"으로 버리고, 신규 차선이 HOLD_FRAMES 프레임
     이상 유지될 때만 실제 차선 전이로 인정(플래핑 노이즈 필터, Argoverse2와 동일 방법론).
  2. stop 라벨 — 공업탑과 동일하게 speed<=1.0 순간 판정을 ±2 다수결(persistence, B안)로.
  3. lane_change 라벨 — 공업탑과 동일하게 (t, t+H] 윈도우 내 차선 전이 발생 여부(A안).
     H=30프레임(29.97fps 기준 약 1초, 공업탑 H=10프레임@10fps와 동일한 시간 지평).

범위 제한(명시): 계산량 때문에 rdb1의 처음 N_RECORDINGS개 레코딩만 사용 — 전체 153개가 아님.
이 스크립트는 "openDD도 공업탑처럼 위치만으로 라벨이 새는가?"를 빠르게 확인하는 것이 목적이며,
실제 학습 파이프라인 이식(lane_id 배치 스크립트)과는 별개다.

출력: journal/paper_figs/opendd/position_memorization_entropy_cell{N}m_{ko,en}.png
      journal/paper_figs/opendd/position_memorization_entropy_cell{N}m_summary.json
      journal/paper_figs/opendd/position_memorization_entropy_sweep_{ko,en}.png
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/oem/TNA_research')

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

ROOT = '/home/oem/data/TII_data/opendd_dataset'
SITE = 'rdb1'
N_RECORDINGS = 15          # 범위 제한 — rdb1 153개 레코딩 중 처음 15개만 사용
SPEED_STOP_THRESH = 1.0    # m/s, 공업탑과 동일
STOP_PERSIST_DELTA = 2     # B안과 동일 ±2 다수결
FPS = 29.97
H_FRAMES = round(FPS * 1.0)   # 공업탑 H=10프레임@10fps와 동일한 1초 지평
LANE_DIST_THRESH_M = 3.0   # 3단계 확장검증의 p95(대부분 1.4~4.5m) 참고, 이 밖은 "차선 미배정"
LANE_HOLD_FRAMES = 10      # Argoverse2와 동일한 신규 차선 유지 조건(플래핑 필터)
VEHICLE_CLASSES = {'Bus', 'Car', 'Medium Vehicle', 'Heavy Vehicle', 'Motorcycle'}
OUT_DIR = '/home/oem/TNA_research/journal/paper_figs/opendd'
NUM_CLASSES = 3
CLASS_NAMES = ('stop', 'lane_change', 'normal')
CLASS_NAMES_EN = ('stop', 'lane_change', 'normal')
CLASS_COLORS = ['#2563eb', '#dc2626', '#16a34a']
STOP, LANE_CHANGE, NORMAL = 0, 1, 2

# 공업탑 결과와 나란히 비교 (라벨 정의가 다르므로 직접비교 금지 — §하네스 원칙, 참고용으로만 표기)
GONGEOPTAP_PERSIST_ACC, GONGEOPTAP_PERSIST_F1 = 63.17, 50.55
GONGEOPTAP_FINAL_ACC, GONGEOPTAP_FINAL_F1 = 81.91, 81.95


def parse_linestring(ls):
    s = ls.replace('LINESTRING (', '').replace(')', '')
    return np.array([[float(v) for v in pair.split()] for pair in s.split(', ')], dtype=np.float64)


def load_lanes():
    d = os.path.join(ROOT, f'opendd_v3-{SITE}', SITE)
    mcon = sqlite3.connect(os.path.join(d, f'map_{SITE}', f'map_{SITE}.sqlite'))
    lanes = pd.read_sql(f"SELECT id, geometry FROM {SITE} WHERE type='trafficLane'", mcon)
    mcon.close()
    polys = [parse_linestring(g) for g in lanes['geometry']]
    ids = lanes['id'].to_numpy()
    return ids, polys


def assign_lane_ids(points: np.ndarray, lane_ids: np.ndarray, polys: list, thresh: float, chunk: int = 150_000):
    """points: [N,2] -> (lane_id 또는 -1, dist)"""
    n = len(points)
    best_dist = np.full(n, np.inf, dtype=np.float32)
    best_lane = np.full(n, -1, dtype=np.int64)
    pts = points.astype(np.float32)
    for lid, poly in zip(lane_ids, polys):
        poly = poly.astype(np.float32)
        a, b = poly[:-1], poly[1:]
        ab = b - a
        denom = (ab * ab).sum(1)
        denom[denom < 1e-9] = 1e-9
        for start in range(0, n, chunk):
            sl = slice(start, start + chunk)
            p = pts[sl]
            ap = p[:, None, :] - a[None, :, :]
            t = np.clip((ap * ab[None, :, :]).sum(2) / denom[None, :], 0, 1)
            proj = a[None, :, :] + t[:, :, None] * ab[None, :, :]
            d = np.linalg.norm(proj - p[:, None, :], axis=2).min(1)
            mask = d < best_dist[sl]
            idxs = np.nonzero(mask)[0] + start
            best_dist[idxs] = d[mask]
            best_lane[idxs] = lid
    best_lane[best_dist > thresh] = -1
    return best_lane, best_dist


def hold_filter(lane_seq: np.ndarray, hold: int) -> np.ndarray:
    """짧게(< hold 프레임) 유지된 차선 값을 직전 안정 차선으로 대체(플래핑 제거)."""
    out = lane_seq.copy()
    n = len(out)
    i = 0
    stable = out[0]
    while i < n:
        j = i
        while j < n and out[j] == out[i]:
            j += 1
        run_len = j - i
        if run_len < hold and i > 0:
            out[i:j] = stable
        else:
            stable = out[i]
        i = j
    return out


def build_labels_for_site():
    lane_ids, polys = load_lanes()
    lane_center = np.vstack(polys)
    center = lane_center.mean(0)

    d = os.path.join(ROOT, f'opendd_v3-{SITE}', SITE)
    tcon = sqlite3.connect(os.path.join(d, f'trajectories_{SITE}_v3.sqlite'))
    tables = [r[0] for r in tcon.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    tables = tables[:N_RECORDINGS]

    all_pos, all_y, all_meta = [], [], []
    for ti, t in enumerate(tables):
        df = pd.read_sql(f"SELECT TIMESTAMP, OBJID, UTM_X, UTM_Y, V, CLASS FROM {t}", tcon)
        df = df[df['CLASS'].isin(VEHICLE_CLASSES)].sort_values(['OBJID', 'TIMESTAMP'])
        if df.empty:
            continue
        pts = df[['UTM_X', 'UTM_Y']].to_numpy()
        lane_assign, _ = assign_lane_ids(pts, lane_ids, polys, LANE_DIST_THRESH_M)
        df = df.assign(lane=lane_assign)

        for oid, g in df.groupby('OBJID', sort=False):
            g = g.reset_index(drop=True)
            n = len(g)
            if n < H_FRAMES + STOP_PERSIST_DELTA + 1:
                continue
            v = g['V'].to_numpy()
            lane = hold_filter(g['lane'].to_numpy(), LANE_HOLD_FRAMES)
            is_stop_instant = v <= SPEED_STOP_THRESH

            # B안 stop persistence (±delta 다수결)
            stop_persist = np.zeros(n, dtype=bool)
            for i in range(n):
                lo, hi = max(0, i - STOP_PERSIST_DELTA), min(n, i + STOP_PERSIST_DELTA + 1)
                stop_persist[i] = is_stop_instant[lo:hi].mean() >= 0.5

            # A안 lane_change window: (t, t+H] 내 lane 값이 바뀌고 둘 다 유효(-1 아님)한 전이가 있는지
            lc_window = np.zeros(n, dtype=bool)
            valid_lane = lane >= 0
            for i in range(n - 1):
                hi = min(n, i + 1 + H_FRAMES)
                seg = lane[i:hi]
                segv = valid_lane[i:hi]
                base = lane[i] if valid_lane[i] else -1
                changed = (seg != base) & segv & (base >= 0)
                lc_window[i] = changed.any()

            y = np.full(n, NORMAL, dtype=np.int64)
            y[stop_persist] = STOP
            y[lc_window & ~stop_persist] = LANE_CHANGE  # stop 우선순위는 공업탑과 동일 순서

            # anchor(t)만 사용 (미래 H프레임을 볼 수 없는 마지막 구간 제외)
            keep = np.arange(n - H_FRAMES)
            all_pos.append(np.stack([g['UTM_X'].to_numpy()[keep], g['UTM_Y'].to_numpy()[keep]], axis=1))
            all_y.append(y[keep])
            all_meta.append(np.full(len(keep), oid, dtype=np.int64))
        print(f'  [{t}] {ti+1}/{len(tables)} 처리 완료 (누적 anchor={sum(len(a) for a in all_y):,})')

    tcon.close()
    pos = np.concatenate(all_pos, axis=0).astype(np.float32)
    y = np.concatenate(all_y, axis=0)
    obj = np.concatenate(all_meta, axis=0)
    return pos, y, obj, center


def _entropy_bits(counts):
    n = counts.sum()
    if n == 0:
        return float('nan')
    p = counts / n
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def build_grid(pos, y, cell, bounds):
    xmin, xmax, zmin, zmax = bounds
    nx = int(np.ceil((xmax - xmin) / cell)) + 1
    nz = int(np.ceil((zmax - zmin) / cell)) + 1
    ix = np.clip(((pos[:, 0] - xmin) / cell).astype(np.int64), 0, nx - 1)
    iz = np.clip(((pos[:, 1] - zmin) / cell).astype(np.int64), 0, nz - 1)
    counts = np.zeros((nx, nz, NUM_CLASSES), dtype=np.int64)
    np.add.at(counts, (ix, iz, y), 1)
    return counts, nx, nz


def analyze_cell(cell, min_count, train_pos, train_y, test_pos, test_y, bounds):
    xmin, xmax, zmin, zmax = bounds
    counts, nx, nz = build_grid(train_pos, train_y, cell, bounds)
    n_per_cell = counts.sum(axis=2)
    valid = n_per_cell >= min_count
    entropy_map = np.full((nx, nz), np.nan, dtype=np.float32)
    purity_map = np.full((nx, nz), np.nan, dtype=np.float32)
    majority_map = np.full((nx, nz), -1, dtype=np.int64)
    vx, vz = np.nonzero(valid)
    for xi, zi in zip(vx, vz):
        c = counts[xi, zi]
        n = c.sum()
        entropy_map[xi, zi] = _entropy_bits(c)
        purity_map[xi, zi] = c.max() / n
        majority_map[xi, zi] = int(c.argmax())

    unconditional_entropy = _entropy_bits(np.bincount(train_y, minlength=NUM_CLASSES))
    w = n_per_cell[valid].astype(np.float64)
    weighted_cond_entropy = float((entropy_map[valid] * w).sum() / w.sum()) if w.sum() > 0 else float('nan')
    mutual_info = unconditional_entropy - weighted_cond_entropy
    weighted_purity = float((purity_map[valid] * w).sum() / w.sum()) if w.sum() > 0 else float('nan')
    max_entropy = float(np.log2(NUM_CLASSES))

    test_ix = np.clip(((test_pos[:, 0] - xmin) / cell).astype(np.int64), 0, nx - 1)
    test_iz = np.clip(((test_pos[:, 1] - zmin) / cell).astype(np.int64), 0, nz - 1)
    train_majority_all = counts.reshape(-1, NUM_CLASSES)
    flat_idx = test_ix * nz + test_iz
    cell_has_train = train_majority_all.sum(axis=1) > 0
    global_majority = int(np.bincount(train_y, minlength=NUM_CLASSES).argmax())

    pred = np.full(len(test_y), global_majority, dtype=np.int64)
    seen_mask = cell_has_train[flat_idx]
    pred[seen_mask] = train_majority_all[flat_idx[seen_mask]].argmax(axis=1)
    unseen_frac = 1.0 - seen_mask.mean()

    acc = float((pred == test_y).mean())
    per_class = []
    for c in range(NUM_CLASSES):
        tp = int(((pred == c) & (test_y == c)).sum())
        fp = int(((pred == c) & (test_y != c)).sum())
        fn = int(((pred != c) & (test_y == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class.append({'class': CLASS_NAMES[c], 'recall': rec, 'precision': prec, 'f1': f1})
    macro_f1 = float(np.mean([p['f1'] for p in per_class]))

    return dict(
        cell=cell, min_count=min_count, bounds=bounds, nx=nx, nz=nz,
        n_per_cell=n_per_cell, valid=valid, entropy_map=entropy_map,
        purity_map=purity_map, majority_map=majority_map,
        unconditional_entropy=unconditional_entropy, weighted_cond_entropy=weighted_cond_entropy,
        mutual_info=mutual_info, max_entropy=max_entropy, weighted_purity=weighted_purity,
        acc=acc, macro_f1=macro_f1, unseen_frac=unseen_frac, per_class=per_class,
        n_train=len(train_y),
    )


def print_report(r, class_dist):
    cell = r['cell']
    print(f'\n=== openDD {SITE} (train, cell={cell}m, 최소 {r["min_count"]}개/셀) ===')
    print(f'클래스 분포 (전체): stop={class_dist[0]*100:.1f}% lane_change={class_dist[1]*100:.1f}% normal={class_dist[2]*100:.1f}%')
    print(f'H(y) = {r["unconditional_entropy"]:.4f} bits (최대 {r["max_entropy"]:.4f})')
    print(f'H(y|cell) = {r["weighted_cond_entropy"]:.4f} bits, I(y;cell) = {r["mutual_info"]:.4f} bits '
          f'({100*r["mutual_info"]/r["unconditional_entropy"]:.1f}% 감소)')
    print(f'순도 가중평균 = {r["weighted_purity"]*100:.2f}%')
    print(f'위치룩업 test acc = {r["acc"]*100:.2f}%  macro-F1 = {r["macro_f1"]*100:.2f}%  '
          f'(unseen cell {r["unseen_frac"]*100:.2f}%)')
    print(f'(참고, 라벨정의 달라 직접비교 금지) 공업탑 Persist={GONGEOPTAP_PERSIST_ACC:.2f}% '
          f'/ 공업탑 최종모델={GONGEOPTAP_FINAL_ACC:.2f}%')


def visualize_cell(r, lang, out_path):
    bounds = r['bounds']
    xmin, xmax, zmin, zmax = bounds
    extent = [xmin, xmax, zmin, zmax]
    cell, min_count, max_entropy = r['cell'], r['min_count'], r['max_entropy']
    names = CLASS_NAMES if lang == 'ko' else CLASS_NAMES_EN

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    density = np.log10(r['n_per_cell'].T + 1)
    im0 = axes[0].imshow(density, origin='lower', extent=extent, aspect='equal', cmap='magma')
    axes[0].set_title(f'(a) sample density log10 (cell={cell}m)' if lang == 'en' else f'(a) 샘플 밀도 log10 (cell={cell}m)')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    cmap_cls = ListedColormap(CLASS_COLORS)
    maj = np.ma.masked_where(~r['valid'].T, r['majority_map'].T)
    im1 = axes[1].imshow(maj, origin='lower', extent=extent, aspect='equal', cmap=cmap_cls, vmin=0, vmax=2)
    axes[1].set_title(f'(b) majority label (n>={min_count})' if lang == 'en' else f'(b) 셀별 다수결 라벨 (n>={min_count})')
    cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, ticks=[0, 1, 2])
    cbar1.ax.set_yticklabels(names)

    ent = np.ma.masked_where(~r['valid'].T, r['entropy_map'].T)
    im2 = axes[2].imshow(ent, origin='lower', extent=extent, aspect='equal', cmap='viridis_r', vmin=0, vmax=max_entropy)
    axes[2].set_title(f'(c) label entropy (0=deterministic)' if lang == 'en' else '(c) 라벨 엔트로피 (0=완전확정)')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.scatter([r['center'][0]], [r['center'][1]], marker='+', s=140, c='white', linewidths=2, zorder=5)
        ax.set_xlabel('UTM_X (m)')
        ax.set_ylabel('UTM_Y (m)')

    suptitle = (
        f'openDD {SITE} position-memorization (train N={r["n_train"]:,}, cell={cell}m) — '
        f'I(y;position)={r["mutual_info"]:.3f}/{max_entropy:.3f} bits, position-lookup acc={r["acc"]*100:.1f}%'
        if lang == 'en' else
        f'openDD {SITE} 위치 암기 분석 (train N={r["n_train"]:,}, cell={cell}m) — '
        f'I(y;position)={r["mutual_info"]:.3f}/{max_entropy:.3f} bits, 위치룩업 acc={r["acc"]*100:.1f}%'
    )
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def visualize_sweep(results, lang, out_path):
    cells = [r['cell'] for r in results]
    mi = [r['mutual_info'] for r in results]
    acc = [r['acc'] * 100 for r in results]
    f1 = [r['macro_f1'] * 100 for r in results]
    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax2 = ax1.twinx()
    l1, = ax1.plot(cells, mi, 'o-', color='#7c3aed', label='I(y;cell)')
    ax1.set_xlabel('grid cell size (m)')
    ax1.set_ylabel('I(y;position) bits', color='#7c3aed')
    l2, = ax2.plot(cells, acc, 's-', color='#dc2626', label='position-lookup acc')
    l3, = ax2.plot(cells, f1, '^-', color='#f97316', label='position-lookup macro-F1')
    ax2.set_ylabel('position-lookup (%)', color='#dc2626')
    ax2.set_ylim(0, 100)
    ax1.set_title(f'openDD {SITE} — grid cell size sensitivity')
    lines = [l1, l2, l3]
    ax1.legend(lines, [ln.get_label() for ln in lines], loc='center right', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', type=float, nargs='+', default=[1.0, 2.0, 5.0])
    ap.add_argument('--min_count', type=int, default=20)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f'openDD {SITE} 라벨 생성 중 (레코딩 {N_RECORDINGS}개, 범위 제한)...')
    pos, y, obj, center = build_labels_for_site()
    print(f'anchor 총 {len(y):,}개  클래스분포: stop={100*np.mean(y==STOP):.1f}% '
          f'lane_change={100*np.mean(y==LANE_CHANGE):.1f}% normal={100*np.mean(y==NORMAL):.1f}%')

    rng = np.random.default_rng(42)
    uniq_obj = np.unique(obj)
    rng.shuffle(uniq_obj)
    n_test_obj = max(1, int(len(uniq_obj) * 0.2))
    test_objs = set(uniq_obj[:n_test_obj].tolist())
    is_test = np.array([o in test_objs for o in obj])
    train_pos, train_y = pos[~is_test], y[~is_test]
    test_pos, test_y = pos[is_test], y[is_test]
    print(f'train N={len(train_y):,}  test N={len(test_y):,} (object_id 기준 80/20 분할)')

    class_dist = np.bincount(y, minlength=NUM_CLASSES) / len(y)
    all_pos = np.concatenate([train_pos, test_pos], axis=0)
    bounds = (float(all_pos[:, 0].min()), float(all_pos[:, 0].max()),
              float(all_pos[:, 1].min()), float(all_pos[:, 1].max()))

    results = []
    for cell in args.cells:
        r = analyze_cell(cell, args.min_count, train_pos, train_y, test_pos, test_y, bounds)
        r['center'] = center
        print_report(r, class_dist)
        results.append(r)
        tag = f'cell{cell:g}m'
        json_path = os.path.join(OUT_DIR, f'position_memorization_entropy_{tag}_summary.json')
        with open(json_path, 'w') as f:
            json.dump(dict(
                site=SITE, n_recordings=N_RECORDINGS, cell_m=cell, min_count=args.min_count,
                class_dist=class_dist.tolist(),
                unconditional_entropy_bits=r['unconditional_entropy'],
                weighted_conditional_entropy_bits=r['weighted_cond_entropy'],
                mutual_info_bits=r['mutual_info'],
                mutual_info_pct_of_max=100 * r['mutual_info'] / r['max_entropy'],
                weighted_purity_pct=r['weighted_purity'] * 100,
                position_lookup_test_accuracy_pct=r['acc'] * 100,
                position_lookup_test_macro_f1_pct=r['macro_f1'] * 100,
                position_lookup_unseen_cell_frac_pct=r['unseen_frac'] * 100,
                per_class=r['per_class'],
            ), f, indent=2, ensure_ascii=False)
        for lang in ('ko', 'en'):
            png_path = os.path.join(OUT_DIR, f'position_memorization_entropy_{tag}_{lang}.png')
            visualize_cell(r, lang, png_path)
        print(f'저장: {json_path}')

    if len(results) > 1:
        for lang in ('ko', 'en'):
            sweep_path = os.path.join(OUT_DIR, f'position_memorization_entropy_sweep_{lang}.png')
            visualize_sweep(results, lang, sweep_path)
            print(f'민감도 요약: {sweep_path}')


if __name__ == '__main__':
    main()
