"""
ET-NAGraphSAGE 데이터로더 검증 스크립트
=========================================
실행:
  cd /home/oem/TNA_research
  python scripts/verify_dataloader.py [--dataset gongeoptap|drift_a|all] [--T 10] [--K 7]

무엇을 확인하는가:
  1. 샘플 수 (train / val / test)
  2. 출력 텐서 shape
  3. 실제 예시 샘플 값 (피처, 레이블, 이웃 거리)
  4. 클래스 분포 (split별)
  5. 시계열 연속성 (프레임 간격 일정한가)
  6. Temporal split 겹침 없음
  7. 엣지 피처 정상성 (거리, 상대 속도 범위)
  8. 이웃 마스크 정확성 (실제 이웃 수 == mask.sum())
  9. 배치 shape 확인
"""

import sys
import os
import glob
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.data_manager import TemporalVehicleDataset, build_dataloaders, LABEL_MAP

LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}
SEP  = '=' * 68
SEP2 = '-' * 68


def section(title: str):
    print(f'\n{SEP}')
    print(f'  {title}')
    print(SEP)


def check(cond: bool, msg: str):
    status = '✓' if cond else '✗ FAIL'
    print(f'  [{status}] {msg}')
    if not cond:
        print('        *** 위 항목 실패 — 데이터 구조를 확인하세요 ***')


# ─────────────────────────────────────────────────────────────────────────────

def verify_single_sample(ds: TemporalVehicleDataset, idx: int = 0):
    """샘플 하나를 꺼내 모든 피처 값을 출력."""
    sample = ds[idx]
    meta   = sample['meta']

    section(f'[1] 예시 샘플 #{idx}  (object_id={meta["object_id"]}, '
            f'frame={meta["frame"]}, file={meta["csv_path"]})')

    T = sample['node_seq'].shape[0]
    K = sample['nbr_node_seqs'].shape[0]

    print(f'\n  레이블  : {LABEL_NAMES[sample["y"].item()]} ({sample["y"].item()})')
    print(f'  이웃 수 : {meta["n_neighbors"]}개 (mask 합 = {int(sample["nbr_mask"].sum().item())})')
    print(f'  윈도우  : {meta["window_frames"]}')

    # Shape 검사
    print(f'\n  --- Shape ---')
    print(f'  node_seq      : {tuple(sample["node_seq"].shape)}   (기대: ({T}, 6))')
    print(f'  nbr_node_seqs : {tuple(sample["nbr_node_seqs"].shape)}  (기대: ({K}, {T}, 6))')
    print(f'  edge_seqs     : {tuple(sample["edge_seqs"].shape)}  (기대: ({K}, {T}, 5))')
    print(f'  nbr_mask      : {tuple(sample["nbr_mask"].shape)}      (기대: ({K},))')

    check(sample['node_seq'].shape      == (T, 6),     f'node_seq shape == ({T}, 6)')
    check(sample['nbr_node_seqs'].shape == (K, T, 6),  f'nbr_node_seqs shape == ({K},{T},6)')
    check(sample['edge_seqs'].shape     == (K, T, 5),  f'edge_seqs shape == ({K},{T},5)')
    check(sample['nbr_mask'].shape      == (K,),        f'nbr_mask shape == ({K},)')

    # ego 노드 시퀀스 출력
    print(f'\n  --- ego 노드 시퀀스 (처음 3 프레임, 마지막 프레임) ---')
    ns = sample['node_seq'].numpy()
    col_names = ['pos_x', 'pos_z', 'speed', 'dir_x', 'dir_z', 'accel']
    header = '  {:>6s}  ' + '  '.join([f'{n:>9s}' for n in col_names])
    print(header.format('frame'))
    frames_to_show = list(range(min(3, T))) + ([-1] if T > 3 else [])
    for ti in frames_to_show:
        label = f't{ti if ti>=0 else T+ti}'
        row = '  {:>6s}  '.format(label)
        row += '  '.join([f'{v:9.3f}' for v in ns[ti]])
        print(row)

    # 이웃 정보 출력
    print(f'\n  --- 이웃 요약 (현재 프레임 t 기준) ---')
    for ki in range(meta['n_neighbors']):
        nbr_id   = meta['nbr_ids'][ki]
        nbr_dist = meta['nbr_dists'][ki]
        ef_t     = sample['edge_seqs'][ki, -1].numpy()  # 현재 프레임 엣지
        nf_t     = sample['nbr_node_seqs'][ki, -1].numpy()  # 현재 프레임 이웃 노드
        print(f'  이웃[{ki}] obj={nbr_id:5d}  dist={nbr_dist:6.2f}m'
              f'  speed={nf_t[2]:6.2f}km/h'
              f'  rel_speed={ef_t[0]:7.3f}  rel_accel={ef_t[1]:8.3f}'
              f'  edge_dist={ef_t[4]:6.2f}m')

    # 패딩 이웃 (mask=0인 것)
    n_pad = K - meta['n_neighbors']
    if n_pad > 0:
        pad_feats = sample['edge_seqs'][meta['n_neighbors']:].numpy()
        check(np.all(pad_feats == 0.0), f'패딩 이웃 {n_pad}개 엣지 피처 = 0')

    # nbr_mask 일치 검사
    check(int(sample['nbr_mask'].sum().item()) == meta['n_neighbors'],
          f'nbr_mask.sum() == n_real_neighbors ({meta["n_neighbors"]})')


def verify_temporal_continuity(ds: TemporalVehicleDataset, n_check: int = 500):
    """T 프레임 간격이 일정한지 무작위 샘플에서 확인."""
    section('[2] 시계열 연속성 검사')
    from modules.data_manager import _FILE_CACHE

    fail_count = 0
    indices = np.random.choice(len(ds), size=min(n_check, len(ds)), replace=False)
    step_set = set()

    for idx in indices:
        _, _, window = ds.samples[idx]
        csv_path = ds.samples[idx][0]
        fd = _FILE_CACHE[csv_path]
        gaps = [window[i+1] - window[i] for i in range(len(window)-1)]
        for g in gaps:
            if g > fd.max_gap:
                fail_count += 1
        step_set.update(gaps)

    check(fail_count == 0,
          f'{n_check}개 샘플 중 프레임 간격 이상 없음 (관측된 step 값: {sorted(step_set)})')

    # frame_step 정보 출력
    for csv_path, fd in list(_FILE_CACHE.items())[:3]:
        print(f'  {os.path.basename(csv_path)}: frame_step={fd.frame_step}, '
              f'max_gap={fd.max_gap}')


def verify_temporal_split(csv_files, T, K, train_ratio, val_ratio):
    """train / val / test 프레임 범위가 겹치지 않는지 확인."""
    section('[3] Temporal Split 겹침 검사')

    from modules.data_manager import _FILE_CACHE, _get_file_data

    for csv_path in csv_files[:3]:
        fd = _get_file_data(csv_path)
        n  = fd.n_frames
        te = int(n * train_ratio)
        ve = int(n * (train_ratio + val_ratio))

        train_frames = set(fd.sorted_frames[:te])
        val_frames   = set(fd.sorted_frames[te:ve])
        test_frames  = set(fd.sorted_frames[ve:])

        overlap_tv = train_frames & val_frames
        overlap_vt = val_frames   & test_frames
        overlap_tt = train_frames & test_frames

        fname = os.path.basename(csv_path)
        check(len(overlap_tv) == 0,
              f'{fname}: train∩val 겹침 없음 (train {len(train_frames)}f, val {len(val_frames)}f)')
        check(len(overlap_vt) == 0,
              f'{fname}: val∩test 겹침 없음 (test {len(test_frames)}f)')
        check(len(overlap_tt) == 0,
              f'{fname}: train∩test 겹침 없음')

        print(f'  {fname}  train:{min(train_frames)}~{max(train_frames)}'
              f'  val:{min(val_frames) if val_frames else "-"}~{max(val_frames) if val_frames else "-"}'
              f'  test:{min(test_frames) if test_frames else "-"}~{max(test_frames) if test_frames else "-"}')


def verify_class_distribution(
        train_ds: TemporalVehicleDataset,
        val_ds:   TemporalVehicleDataset,
        test_ds:  TemporalVehicleDataset,
        n_check:  int = 2000):
    """각 split의 클래스 분포 출력 (전체 순회 대신 샘플링)."""
    section('[4] 클래스 분포')

    from modules.data_manager import _FILE_CACHE

    for split_name, ds in [('train', train_ds), ('val', val_ds), ('test', test_ds)]:
        counts = {0: 0, 1: 0, 2: 0}
        n = min(n_check, len(ds))
        idxs = np.random.choice(len(ds), size=n, replace=False)
        for idx in idxs:
            csv_path, oid, window = ds.samples[idx]
            fd = _FILE_CACHE[csv_path]
            y = fd.frame_label[window[-1]][oid]
            counts[y] += 1
        total = sum(counts.values())
        dist  = {LABEL_NAMES[k]: f'{100*v/total:.1f}%' for k, v in counts.items()}
        print(f'  {split_name:5s} (n={n:,})  {dist}')
        print(f'         절대수: {counts}')


def verify_edge_feature_range(ds: TemporalVehicleDataset, n_check: int = 1000):
    """엣지 피처 값 범위가 물리적으로 타당한지 확인."""
    section('[5] 엣지 피처 값 범위 정상성')

    all_rel_speed = []
    all_rel_accel = []
    all_distance  = []

    idxs = np.random.choice(len(ds), size=min(n_check, len(ds)), replace=False)
    for idx in idxs:
        sample  = ds[idx]
        mask    = sample['nbr_mask'].numpy().astype(bool)
        ef      = sample['edge_seqs'].numpy()  # [K, T, 5]
        for ki in range(len(mask)):
            if not mask[ki]:
                continue
            # 현재 프레임 t (마지막 프레임)
            rel_speed = ef[ki, -1, 0]
            rel_accel = ef[ki, -1, 1]
            dist      = ef[ki, -1, 4]
            all_rel_speed.append(rel_speed)
            all_rel_accel.append(rel_accel)
            all_distance.append(dist)

    rs = np.array(all_rel_speed)
    ra = np.array(all_rel_accel)
    d  = np.array(all_distance)

    print(f'  rel_speed (km/h): min={rs.min():.2f}  max={rs.max():.2f}'
          f'  mean={rs.mean():.2f}  std={rs.std():.2f}')
    print(f'  rel_accel (m/s²): min={ra.min():.2f}  max={ra.max():.2f}'
          f'  mean={ra.mean():.2f}  std={ra.std():.2f}')
    print(f'  distance  (m)   : min={d.min():.2f}  max={d.max():.2f}'
          f'  mean={d.mean():.2f}  std={d.std():.2f}')

    check(d.min() >= 0.0,          '거리 >= 0 (물리적으로 타당)')
    check(d.max() < 500.0,         '거리 < 500m (이상치 없음)')
    check(abs(rs.mean()) < 80.0,   '|mean rel_speed| < 80 km/h')
    check(abs(ra.mean()) < 100.0,  '|mean rel_accel| < 100 m/s²')


def verify_batch_shape(train_loader, T: int, K: int):
    """첫 번째 배치의 shape 확인."""
    section('[6] 배치 Shape 검사')

    batch = next(iter(train_loader))
    B = batch['node_seq'].shape[0]
    print(f'  batch_size (실제): {B}')
    print(f'  node_seq      : {tuple(batch["node_seq"].shape)}      (기대: [B, {T}, 6])')
    print(f'  nbr_node_seqs : {tuple(batch["nbr_node_seqs"].shape)} (기대: [B, {K}, {T}, 6])')
    print(f'  edge_seqs     : {tuple(batch["edge_seqs"].shape)}     (기대: [B, {K}, {T}, 5])')
    print(f'  nbr_mask      : {tuple(batch["nbr_mask"].shape)}      (기대: [B, {K}])')
    print(f'  y             : {tuple(batch["y"].shape)}             (기대: [B])')

    check(batch['node_seq'].shape      == (B, T, 6),     f'batch node_seq == (B,{T},6)')
    check(batch['nbr_node_seqs'].shape == (B, K, T, 6),  f'batch nbr_node_seqs == (B,{K},{T},6)')
    check(batch['edge_seqs'].shape     == (B, K, T, 5),  f'batch edge_seqs == (B,{K},{T},5)')
    check(batch['nbr_mask'].shape      == (B, K),         f'batch nbr_mask == (B,{K})')
    check(batch['y'].shape             == (B,),            f'batch y == (B,)')

    y_vals = batch['y'].tolist()[:10]
    print(f'  y 첫 10개: {y_vals}  ({[LABEL_NAMES[v] for v in y_vals]})')


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='ET-NAGraphSAGE 데이터로더 검증')
    parser.add_argument('--dataset', default='drift_a',
                        choices=['gongeoptap', 'drift_a', 'drift_b', 'all'],
                        help='검증할 데이터셋')
    parser.add_argument('--T',      type=int,   default=10,   help='시퀀스 길이')
    parser.add_argument('--radius', type=float, default=20.0, help='이웃 탐색 반경 (m): 10 / 20 / 30')
    parser.add_argument('--K_max',  type=int,   default=6,    help='이웃 상한: 4 / 5 / 6')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_files',  type=int, default=5,
                        help='사용할 파일 수 (빠른 검증용)')
    args = parser.parse_args()

    print('\n' + SEP)
    print('  ET-NAGraphSAGE 데이터로더 검증 스크립트')
    print(f'  dataset={args.dataset}, T={args.T}, radius={args.radius}m, K_max={args.K_max}')
    print(SEP)

    # ── 파일 목록 선택 ─────────────────────────────────────────────────────
    DATA_ROOT = '/home/oem/data/TII_data'

    if args.dataset == 'gongeoptap':
        csv_files = sorted(glob.glob(f'{DATA_ROOT}/Gongeoptap/*.csv'))
    elif args.dataset == 'drift_a':
        csv_files = sorted(glob.glob(f'{DATA_ROOT}/DRIFT_csv/DRIFT_A_*.csv'))
    elif args.dataset == 'drift_b':
        csv_files = sorted(glob.glob(f'{DATA_ROOT}/DRIFT_csv/DRIFT_B_*.csv'))
    else:  # all
        csv_files = (
            sorted(glob.glob(f'{DATA_ROOT}/Gongeoptap/*.csv')) +
            sorted(glob.glob(f'{DATA_ROOT}/DRIFT_csv/*.csv'))
        )

    csv_files = csv_files[:args.max_files]
    print(f'\n  사용 파일 ({len(csv_files)}개):')
    for f in csv_files:
        print(f'    {f}')

    # ── Dataset 생성 ───────────────────────────────────────────────────────
    print()
    train_ds = TemporalVehicleDataset(csv_files, T=args.T, radius=args.radius, K_max=args.K_max, split='train')
    val_ds   = TemporalVehicleDataset(csv_files, T=args.T, radius=args.radius, K_max=args.K_max, split='val')
    test_ds  = TemporalVehicleDataset(csv_files, T=args.T, radius=args.radius, K_max=args.K_max, split='test')

    section('[0] 샘플 수 요약')
    total = len(train_ds) + len(val_ds) + len(test_ds)
    print(f'  train : {len(train_ds):>9,}  ({100*len(train_ds)/total:.1f}%)')
    print(f'  val   : {len(val_ds):>9,}  ({100*len(val_ds)/total:.1f}%)')
    print(f'  test  : {len(test_ds):>9,}  ({100*len(test_ds)/total:.1f}%)')
    print(f'  total : {total:>9,}')

    check(len(train_ds) > 0, 'train 샘플 존재')
    check(len(val_ds)   > 0, 'val 샘플 존재')
    check(len(test_ds)  > 0, 'test 샘플 존재')
    check(len(train_ds) > len(val_ds), 'train > val')

    # ── 개별 검증 ──────────────────────────────────────────────────────────
    verify_single_sample(train_ds, idx=0)
    verify_single_sample(train_ds, idx=len(train_ds) // 2)

    verify_temporal_continuity(train_ds)
    verify_temporal_split(csv_files, args.T, args.K_max, 0.70, 0.15)
    verify_class_distribution(train_ds, val_ds, test_ds)
    verify_edge_feature_range(train_ds)

    # ── 배치 검증 ──────────────────────────────────────────────────────────
    from torch.utils.data import DataLoader
    from modules.data_manager import _collate_fn

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, collate_fn=_collate_fn, num_workers=0)
    verify_batch_shape(train_loader, args.T, args.K_max)

    print(f'\n{SEP}')
    print('  검증 완료.')
    print(SEP)


if __name__ == '__main__':
    main()
