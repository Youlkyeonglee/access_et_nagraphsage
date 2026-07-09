#!/usr/bin/env python3
"""TSEM dataloader·instant 라벨 스모크 테스트."""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_manager_tsem import build_tsem_dataloaders, summarize_label_distribution
from modules.tsem_instant_label import CLASS_NAMES, instant_state


def main():
    p = argparse.ArgumentParser(description='TSEM dataloader verify')
    p.add_argument('--data_dir', default='/home/oem/data/TII_data/Gongeoptap')
    p.add_argument('--W', type=int, default=10)
    p.add_argument('--H', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--no_cache', action='store_true')
    args = p.parse_args()

    csv_files = sorted(glob.glob(os.path.join(args.data_dir, '*.csv')))
    if not csv_files:
        print(f'CSV 없음: {args.data_dir}')
        sys.exit(1)
    print(f'CSV {len(csv_files)}개')

    # instant 라벨 단위 테스트
    assert instant_state(0.5, '3-1', '3-2') == 0
    assert instant_state(5.0, '3-2', '3-1') == 1
    assert instant_state(5.0, '3-1', '3-1') == 2
    print('instant_state 규칙 OK')

    dist = summarize_label_distribution(csv_files[:1], args.W, args.H)
    print('label dist (1 file, train):', dist)

    train_loader, _, _ = build_tsem_dataloaders(
        csv_files,
        W=args.W,
        H=args.H,
        batch_size=args.batch_size,
        num_workers=0,
        use_cache=not args.no_cache,
        verbose=True,
    )
    batch = next(iter(train_loader))
    print('batch keys:', sorted(batch.keys()))
    print('node_seq', batch['node_seq'].shape)
    print('y (future)', batch['y'][:8].tolist())
    print('y_persist (anchor)', batch['y_persist'][:8].tolist())
    print('class names:', CLASS_NAMES)
    print('OK')


if __name__ == '__main__':
    main()
