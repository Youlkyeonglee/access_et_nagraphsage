"""
[2026-07-10] 제안1(speed-reg 보조헤드) 출발임박 하위그룹 평가.
10차-2(기준) vs speedreg03/10 체크포인트를 같은 test셋에서 평가하고,
GT=normal 샘플을 앵커 속도 v(t) 구간별로 나눠 recall을 비교한다.
하위그룹: 출발임박 v(t)<=1 / 서행 1<v<=3 / 중간 3<v<=8 / 순항 v>8  (raw speed, node_seq[-1,2])
"""
import glob
import os
import sys

sys.path.insert(0, '/home/oem/TNA_research')

import numpy as np
import torch
import yaml

from models.tsem_sage import TSEMSAGE
from modules.data_manager_tsem import TSEMFutureStateDataset, _collate_fn
from torch.utils.data import DataLoader

DATA_DIR = '/home/oem/data/TII_data/Gongeoptap'
CKPTS = {
    '10차-2 (기준)': 'checkpoints/tsem/tsem_sage_w10_h10_sem_pos_10d/best.pt',
    'speedreg λ=0.3': 'checkpoints/tsem/tsem_sage_10d_speedreg03/best.pt',
    'speedreg λ=1.0': 'checkpoints/tsem/tsem_sage_10d_speedreg10/best.pt',
}
CLASS_NAMES = ['stop', 'lane_change', 'normal']
NORMAL = 2

def load_model(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ck['cfg']
    mcfg = cfg['model']
    model = TSEMSAGE(
        hidden_dim=mcfg['hidden_dim'], d_e=mcfg['d_e'], T=cfg['tsem']['W'],
        encoder_type=mcfg['encoder_type'], use_attention=mcfg['use_attention'],
        use_2hop=mcfg.get('use_2hop', True), use_spatial=mcfg.get('use_spatial', True),
        raw_append=mcfg.get('raw_append', 'none'), num_classes=mcfg['num_classes'],
        dropout=mcfg['dropout'], decomp_kernel=mcfg.get('decomp_kernel', 5),
        decomp_learnable=mcfg.get('decomp_learnable', True),
        use_speed_head=any(k.startswith('head_speed') for k in ck['model']),
    )
    model.load_state_dict(ck['model'])
    return model.to(device).eval()

def main():
    device = torch.device('cuda:0')
    csvs = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
    ds = TSEMFutureStateDataset(csvs, split='test', W=10, H=10)
    loader = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=4,
                        collate_fn=_collate_fn, pin_memory=True)

    # v(t) = 앵커 프레임 raw speed — 예측과 같은 순서로 1회만 수집
    vts, ys = [], []
    preds = {k: [] for k in CKPTS}
    models = {k: load_model(p, device) for k, p in CKPTS.items()}
    with torch.no_grad():
        for batch in loader:
            vts.append(batch['node_seq'][:, -1, 2].numpy())
            ys.append(batch['y'].numpy())
            g = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            for name, m in models.items():
                preds[name].append(m(g).argmax(-1).cpu().numpy())
    vt = np.concatenate(vts); y = np.concatenate(ys)
    preds = {k: np.concatenate(v) for k, v in preds.items()}

    groups = [('출발임박 v<=1', vt <= 1.0), ('서행 1<v<=3', (vt > 1) & (vt <= 3)),
              ('중간 3<v<=8', (vt > 3) & (vt <= 8)), ('순항 v>8', vt > 8)]
    print(f'test N={len(y):,}, normal N={(y == NORMAL).sum():,}')
    for name, p in preds.items():
        acc = (p == y).mean() * 100
        nr = (p[y == NORMAL] == NORMAL).mean() * 100
        print(f'\n=== {name}  (전체 acc {acc:.2f}, normal recall {nr:.2f})')
        for gname, gmask in groups:
            m = gmask & (y == NORMAL)
            if m.sum() == 0:
                continue
            rec = (p[m] == NORMAL).mean() * 100
            to_stop = (p[m] == 0).mean() * 100
            print(f'  {gname:16s} N={m.sum():6,} ({m.sum()/(y==NORMAL).sum()*100:4.1f}%)  '
                  f'recall {rec:5.1f}%  (→stop {to_stop:5.1f}%)')

if __name__ == '__main__':
    main()
