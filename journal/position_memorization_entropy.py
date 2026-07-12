"""
위치 암기 분석 ① — 모델 학습 없이 데이터 자체로 증명 (2026-07-12)
================================================================
"세계좌표만 알아도 라벨을 거의 안다"를 모델 없이 직접 보인다.

방법:
  1. anchor(t) 위치를 격자(cell_size)로 쪼개, 각 셀 안의 라벨 분포 엔트로피/순도(purity)를 계산.
     엔트로피가 낮을수록(=순도가 높을수록) "위치만 알아도 라벨을 거의 안다"는 뜻.
  2. train split으로 만든 "셀 -> 다수결 라벨" 룩업테이블을 test split에 그대로 적용해
     정확도/macro-F1을 계산 — 학습(gradient descent) 없이 좌표 하나만으로 얼마나 맞히는지
     Persist baseline(63.17%/50.55%) · 10차-2 최종모델(81.91%/81.95%)과 직접 비교 가능한
     숫자로 만든다.
  3. 셀 크기(--cells)별로 밀도/다수결라벨/엔트로피 3-패널 히트맵을 한글·영문 두 벌 저장하고,
     셀 크기에 따른 정보량·룩업정확도 민감도 요약 그래프도 함께 만든다.

데이터: 기존 캐시(cache/tsem/*_W10_H10_r20.0_K6-4_hybrid_tsem_f16ne_v4_tr0.7_vr0.15_*) 그대로 재사용
  — comparison/ baseline들과 완전히 동일한 train/val/test split, 재학습·재계산 불필요.
anchor 위치 = node_seq[:, -1, 0:2] (window 마지막 프레임 t, raw_append='pos'가 넣는 것과 동일 지점).

사용: python journal/position_memorization_entropy.py [--cells 0.5 1.0 2.0 3.0 5.0] [--min_count 20]
출력: journal/paper_figs/position_memorization_entropy_cell{N}m_{ko,en}.png (셀 크기별 히트맵)
      journal/paper_figs/position_memorization_entropy_cell{N}m_summary.json (셀 크기별 수치)
      journal/paper_figs/position_memorization_entropy_sweep_{ko,en}.png (셀 크기 민감도 요약)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, '/home/oem/TNA_research')

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

plt.rcParams['font.family'] = 'Noto Sans CJK JP'  # ttc에 한글 글리프도 포함(matplotlib이 JP로 등록)
plt.rcParams['axes.unicode_minus'] = False

from modules.tsem_instant_label import CLASS_NAMES  # noqa: E402

CACHE_ROOT = '/home/oem/TNA_research/cache/tsem'
CACHE_SIG = 'W10_H10_r20.0_K6-4_hybrid_tsem_f16ne_v4_tr0.7_vr0.15'
CENTER_X, CENTER_Z = 72.86, -13.45  # models/tsem_semantic_derivation.py와 동일 로터리 중심 상수
OUT_DIR = '/home/oem/TNA_research/journal/paper_figs/gongeoptap'
NUM_CLASSES = 3
PERSIST_ACC, PERSIST_F1 = 63.17, 50.55
FINAL10D_ACC, FINAL10D_F1 = 81.91, 81.95
CLASS_NAMES_EN = ('stop', 'lane_change', 'normal')
CLASS_COLORS = ['#2563eb', '#dc2626', '#16a34a']  # stop=파랑, lane_change=빨강, normal=초록


def _find_cache_dir(split: str) -> str:
    matches = glob.glob(os.path.join(CACHE_ROOT, f'{split}_{CACHE_SIG}_*'))
    if not matches:
        raise FileNotFoundError(f'{split} 캐시를 찾을 수 없음: {CACHE_ROOT}/{split}_{CACHE_SIG}_*')
    return matches[0]


def _load_split(split: str):
    d = _find_cache_dir(split)
    node_seq = np.load(os.path.join(d, 'node_seq.npy'), mmap_mode='r')  # [N, T, 6] fp16
    y = np.load(os.path.join(d, 'y.npy'), mmap_mode='r')  # [N] int64
    anchor_pos = np.asarray(node_seq[:, -1, 0:2], dtype=np.float32)  # [N, 2] (pos_x, pos_z)
    y = np.asarray(y, dtype=np.int64)
    print(f'[{split}] 캐시={os.path.basename(d)}  N={len(y):,}')
    return anchor_pos, y


def _entropy_bits(counts: np.ndarray) -> float:
    n = counts.sum()
    if n == 0:
        return float('nan')
    p = counts / n
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def build_grid(anchor_pos: np.ndarray, y: np.ndarray, cell: float, bounds):
    xmin, xmax, zmin, zmax = bounds
    nx = int(np.ceil((xmax - xmin) / cell)) + 1
    nz = int(np.ceil((zmax - zmin) / cell)) + 1
    ix = np.clip(((anchor_pos[:, 0] - xmin) / cell).astype(np.int64), 0, nx - 1)
    iz = np.clip(((anchor_pos[:, 1] - zmin) / cell).astype(np.int64), 0, nz - 1)
    counts = np.zeros((nx, nz, NUM_CLASSES), dtype=np.int64)
    np.add.at(counts, (ix, iz, y), 1)
    return counts, nx, nz


def analyze_cell(cell: float, min_count: int, train_pos, train_y, test_pos, test_y, bounds) -> dict:
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
    weighted_cond_entropy = float((entropy_map[valid] * w).sum() / w.sum())
    mutual_info = unconditional_entropy - weighted_cond_entropy
    weighted_purity = float((purity_map[valid] * w).sum() / w.sum())
    max_entropy = float(np.log2(NUM_CLASSES))

    # train 셀 다수결 -> test 그대로 적용 (재학습 없는 "위치 룩업")
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


def print_report(r: dict):
    cell = r['cell']
    print(f'\n=== 데이터 자체 (train split, cell={cell}m, 최소 {r["min_count"]}개/셀) ===')
    print(f'전체(무조건) 라벨 엔트로피 H(y)            = {r["unconditional_entropy"]:.4f} bits  '
          f'(최대 {r["max_entropy"]:.4f} = log2(3), 균등분포)')
    print(f'위치로 조건화한 평균 엔트로피 H(y|cell)     = {r["weighted_cond_entropy"]:.4f} bits '
          f'(샘플수 가중평균, 유효 셀 {int(r["valid"].sum()):,}/{r["nx"]*r["nz"]:,}개)')
    print(f'위치가 알려주는 정보량 I(y;cell) = H(y)-H(y|cell) = {r["mutual_info"]:.4f} bits '
          f'({100*r["mutual_info"]/r["unconditional_entropy"]:.1f}% 감소)')
    print(f'셀 내 다수결 클래스 비율(순도) 가중평균     = {r["weighted_purity"]*100:.2f}%  '
          f'(3클래스 무작위 추측 순도 기준 33.3%)')
    for p in r['per_class']:
        print(f'  [위치룩업] {p["class"]:>12s}  recall={p["recall"]*100:5.2f}%  '
              f'precision={p["precision"]*100:5.2f}%  F1={p["f1"]*100:5.2f}%')
    print(f'\n=== "위치 룩업" baseline (train 셀 다수결 -> test 그대로 적용, 재학습 없음) ===')
    print(f'test 중 train에서 한 번도 안 본 셀 비율 = {r["unseen_frac"]*100:.2f}% '
          f'(이 경우 전체 최빈클래스로 대체)')
    print(f'Accuracy = {r["acc"]*100:.2f}%   Macro-F1 = {r["macro_f1"]*100:.2f}%')
    print(f'(비교) Persist baseline               = {PERSIST_ACC:.2f}% / {PERSIST_F1:.2f}%')
    print(f'(비교) 10차-2 TSEM-SAGE 최종(10D, 100ep) = {FINAL10D_ACC:.2f}% / {FINAL10D_F1:.2f}%')


def save_summary_json(r: dict, out_path: str):
    summary = {
        'cell_m': r['cell'], 'min_count': r['min_count'],
        'unconditional_entropy_bits': r['unconditional_entropy'],
        'weighted_conditional_entropy_bits': r['weighted_cond_entropy'],
        'mutual_info_bits': r['mutual_info'],
        'mutual_info_pct_of_max': 100 * r['mutual_info'] / r['max_entropy'],
        'weighted_purity_pct': r['weighted_purity'] * 100,
        'n_valid_cells': int(r['valid'].sum()), 'n_total_cells': int(r['nx'] * r['nz']),
        'position_lookup_test_accuracy_pct': r['acc'] * 100,
        'position_lookup_test_macro_f1_pct': r['macro_f1'] * 100,
        'position_lookup_unseen_cell_frac_pct': r['unseen_frac'] * 100,
        'per_class': r['per_class'],
        'persist_baseline_accuracy_pct': PERSIST_ACC, 'persist_baseline_macro_f1_pct': PERSIST_F1,
        'tsem_sage_final_accuracy_pct': FINAL10D_ACC, 'tsem_sage_final_macro_f1_pct': FINAL10D_F1,
    }
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


_TXT = {
    'ko': dict(
        panel_a='(a) 샘플 밀도 (log10, cell={cell}m)',
        panel_b='(b) 셀별 다수결 라벨 (n≥{min_count})',
        panel_c='(c) 셀별 라벨 엔트로피 (0=완전확정, {maxent:.2f}=균등)',
        cbar_density='log10(count+1)', cbar_entropy='bits',
        xlabel='position_x (m)', ylabel='position_z (m)',
        class_names=list(CLASS_NAMES),
        suptitle=('위치 암기 분석① — 좌표만으로 라벨을 얼마나 아는가 (train split, N={n:,}, cell={cell}m)\n'
                   'I(y;position) = {mi:.3f} / {maxent:.3f} bits ({mipct:.0f}%) · '
                   '위치룩업 test acc={acc:.1f}% (Persist {persist:.1f}% · 10D최종 {final10d:.1f}%)'),
        sweep_title='셀 크기 민감도 — 위치가 라벨을 얼마나 알려주는가',
        sweep_left_ylabel='I(y;position) 정보량 (bits)',
        sweep_right_ylabel='위치룩업 정확도 (%)',
        sweep_xlabel='격자 셀 크기 (m)',
        sweep_legend_mi='정보량 I(y;cell)',
        sweep_legend_acc='위치룩업 test accuracy',
        sweep_legend_f1='위치룩업 test macro-F1',
        sweep_legend_persist='Persist baseline acc',
        sweep_legend_final='10D 최종모델 acc',
    ),
    'en': dict(
        panel_a='(a) Sample density (log10, cell={cell}m)',
        panel_b='(b) Majority label per cell (n>={min_count})',
        panel_c='(c) Label entropy per cell (0=deterministic, {maxent:.2f}=uniform)',
        cbar_density='log10(count+1)', cbar_entropy='bits',
        xlabel='position_x (m)', ylabel='position_z (m)',
        class_names=list(CLASS_NAMES_EN),
        suptitle=('Position-memorization analysis 1 -- how much does raw coordinate reveal the label? '
                   '(train split, N={n:,}, cell={cell}m)\n'
                   'I(y;position) = {mi:.3f} / {maxent:.3f} bits ({mipct:.0f}%) . '
                   'position-lookup test acc={acc:.1f}% (Persist {persist:.1f}% . 10D-final {final10d:.1f}%)'),
        sweep_title='Grid-cell size sensitivity -- how much does position reveal the label?',
        sweep_left_ylabel='I(y;position) mutual info (bits)',
        sweep_right_ylabel='Position-lookup accuracy (%)',
        sweep_xlabel='Grid cell size (m)',
        sweep_legend_mi='Mutual info I(y;cell)',
        sweep_legend_acc='Position-lookup test accuracy',
        sweep_legend_f1='Position-lookup test macro-F1',
        sweep_legend_persist='Persist baseline acc',
        sweep_legend_final='10D final-model acc',
    ),
}


def visualize_cell(r: dict, lang: str, out_path: str):
    t = _TXT[lang]
    bounds = r['bounds']
    xmin, xmax, zmin, zmax = bounds
    extent = [xmin, xmax, zmin, zmax]
    cell, min_count, max_entropy = r['cell'], r['min_count'], r['max_entropy']

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))

    density = np.log10(r['n_per_cell'].T + 1)
    im0 = axes[0].imshow(density, origin='lower', extent=extent, aspect='equal', cmap='magma')
    axes[0].set_title(t['panel_a'].format(cell=cell))
    plt.colorbar(im0, ax=axes[0], fraction=0.046, label=t['cbar_density'])

    cmap_cls = ListedColormap(CLASS_COLORS)
    maj = np.ma.masked_where(~r['valid'].T, r['majority_map'].T)
    im1 = axes[1].imshow(maj, origin='lower', extent=extent, aspect='equal', cmap=cmap_cls, vmin=0, vmax=2)
    axes[1].set_title(t['panel_b'].format(min_count=min_count))
    cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, ticks=[0, 1, 2])
    cbar1.ax.set_yticklabels(t['class_names'])

    ent = np.ma.masked_where(~r['valid'].T, r['entropy_map'].T)
    im2 = axes[2].imshow(ent, origin='lower', extent=extent, aspect='equal', cmap='viridis_r',
                          vmin=0, vmax=max_entropy)
    axes[2].set_title(t['panel_c'].format(maxent=max_entropy))
    plt.colorbar(im2, ax=axes[2], fraction=0.046, label=t['cbar_entropy'])

    for ax in axes:
        ax.scatter([CENTER_X], [CENTER_Z], marker='+', s=140, c='white', linewidths=2, zorder=5)
        ax.set_xlabel(t['xlabel'])
        ax.set_ylabel(t['ylabel'])

    fig.suptitle(
        t['suptitle'].format(
            n=r['n_train'], cell=cell, mi=r['mutual_info'], maxent=max_entropy,
            mipct=100 * r['mutual_info'] / max_entropy, acc=r['acc'] * 100,
            persist=PERSIST_ACC, final10d=FINAL10D_ACC,
        ),
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def visualize_sweep(results: list, lang: str, out_path: str):
    t = _TXT[lang]
    cells = [r['cell'] for r in results]
    mi = [r['mutual_info'] for r in results]
    acc = [r['acc'] * 100 for r in results]
    f1 = [r['macro_f1'] * 100 for r in results]

    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax2 = ax1.twinx()

    l1, = ax1.plot(cells, mi, 'o-', color='#7c3aed', label=t['sweep_legend_mi'])
    ax1.set_xlabel(t['sweep_xlabel'])
    ax1.set_ylabel(t['sweep_left_ylabel'], color='#7c3aed')
    ax1.tick_params(axis='y', labelcolor='#7c3aed')

    l2, = ax2.plot(cells, acc, 's-', color='#dc2626', label=t['sweep_legend_acc'])
    l3, = ax2.plot(cells, f1, '^-', color='#f97316', label=t['sweep_legend_f1'])
    l4 = ax2.axhline(PERSIST_ACC, color='gray', linestyle='--', linewidth=1, label=t['sweep_legend_persist'])
    l5 = ax2.axhline(FINAL10D_ACC, color='black', linestyle=':', linewidth=1.5, label=t['sweep_legend_final'])
    ax2.set_ylabel(t['sweep_right_ylabel'], color='#dc2626')
    ax2.tick_params(axis='y', labelcolor='#dc2626')
    ax2.set_ylim(0, 100)  # 왼쪽(bits) 축과 스케일이 우연히 겹쳐 보이는 것을 막기 위해 고정

    ax1.set_title(t['sweep_title'])
    lines = [l1, l2, l3, l4, l5]
    ax1.legend(lines, [ln.get_label() for ln in lines], loc='center right', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', type=float, nargs='+', default=[0.5, 1.0, 2.0, 3.0, 5.0],
                     help='격자 셀 크기(m) 목록 — 각 크기마다 별도 그림 생성')
    ap.add_argument('--min_count', type=int, default=20, help='엔트로피 계산에 포함할 최소 셀 샘플 수')
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    train_pos, train_y = _load_split('train')
    test_pos, test_y = _load_split('test')

    all_pos = np.concatenate([train_pos, test_pos], axis=0)
    xmin, xmax = float(all_pos[:, 0].min()), float(all_pos[:, 0].max())
    zmin, zmax = float(all_pos[:, 1].min()), float(all_pos[:, 1].max())
    bounds = (xmin, xmax, zmin, zmax)
    print(f'좌표 범위: x=[{xmin:.1f},{xmax:.1f}]  z=[{zmin:.1f},{zmax:.1f}]  '
          f'로터리중심=({CENTER_X},{CENTER_Z})  cells={args.cells}')

    results = []
    for cell in args.cells:
        r = analyze_cell(cell, args.min_count, train_pos, train_y, test_pos, test_y, bounds)
        print_report(r)
        results.append(r)

        tag = f'cell{cell:g}m'
        json_path = os.path.join(OUT_DIR, f'position_memorization_entropy_{tag}_summary.json')
        save_summary_json(r, json_path)

        for lang in ('ko', 'en'):
            png_path = os.path.join(OUT_DIR, f'position_memorization_entropy_{tag}_{lang}.png')
            visualize_cell(r, lang, png_path)
            print(f'그림 저장[{lang}]: {png_path}')
        print(f'요약 JSON: {json_path}')

    if len(results) > 1:
        for lang in ('ko', 'en'):
            sweep_path = os.path.join(OUT_DIR, f'position_memorization_entropy_sweep_{lang}.png')
            visualize_sweep(results, lang, sweep_path)
            print(f'민감도 요약 그림 저장[{lang}]: {sweep_path}')


if __name__ == '__main__':
    main()
