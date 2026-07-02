"""
데이터 캐시 프리빌드 — 학습 전 조립 텐서 캐시를 1회 순차 생성.
동시 학습 시 같은 캐시를 여러 프로세스가 쓰는 race를 방지한다.

사용: python scripts/prebuild_cache.py --config <yaml> [--K_max N --K_max2 N ...]
"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
from train import get_csv_files
from modules.data_manager import build_dataloaders

p = argparse.ArgumentParser()
p.add_argument('--config', required=True)
p.add_argument('--T', type=int, default=None)
p.add_argument('--K_max', type=int, default=None)
p.add_argument('--K_max2', type=int, default=None)
p.add_argument('--radius', type=float, default=None)
p.add_argument('--neighbor_mode', type=str, default=None)
a = p.parse_args()

cfg = yaml.safe_load(open(a.config))
if a.T is not None:            cfg['graph']['T']      = a.T
if a.K_max is not None:        cfg['graph']['K_max']  = a.K_max
if a.K_max2 is not None:       cfg['graph']['K_max2'] = a.K_max2
if a.radius is not None:       cfg['graph']['radius'] = a.radius
if a.neighbor_mode is not None: cfg['graph']['neighbor_mode'] = a.neighbor_mode

g = cfg['graph']
print(f"프리빌드: T={g['T']} r={g['radius']} K={g['K_max']}-{g['K_max2']} "
      f"mode={g.get('neighbor_mode','hybrid')}")

# build_dataloaders 호출 → train/val/test 3개 split 캐시 생성
build_dataloaders(
    csv_files=get_csv_files(cfg),
    T=g['T'], radius=g['radius'], K_max=g['K_max'], K_max2=g.get('K_max2', 0),
    batch_size=cfg['train']['batch_size'],
    train_ratio=cfg['data']['train_ratio'], val_ratio=cfg['data']['val_ratio'],
    num_workers=0,
    neighbor_mode=g.get('neighbor_mode', 'hybrid'),
    use_cache=True,
)
print("캐시 프리빌드 완료.")
