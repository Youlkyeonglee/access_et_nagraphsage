"""
[신규 2026-07-09] 교차 장소 검증 — 공업탑 학습 체크포인트를 DRIFT에서 평가만.
목적: 10차 계열에서 확정된 "절대좌표 +3.9%p 이득"이 공업탑 한정 위치 암기인지 확인.
  - 위치 암기라면: position 사용 모델(10차 raw, 10차-2 10D)이 DRIFT에서 급락
  - semantic 8D(9차)는 상대/미분량만 쓰므로 상대적으로 유지되어야 함
학습 없음 — forward 평가만. Persist baseline도 같이 계산해 "그 장소의 난이도" 기준선 제공.

사용: python journal/cross_location_eval.py [--loc A] [--gpus 0]
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, '/home/oem/TNA_research')

import torch

from torch.utils.data import DataLoader  # noqa: E402

from models.tsem_sage import TSEMSAGE  # noqa: E402
from modules.data_manager_tsem import TSEMFutureStateDataset, _collate_fn  # noqa: E402
from modules.tsem_eval import evaluate_tsem  # noqa: E402
from modules.tsem_instant_label import CLASS_NAMES  # noqa: E402

CKPTS = {
    '9차 semantic 8D': ('tsem_sage_w10_h10_labelB_unc_aug123_e500', dict(use_semantic=True, raw_append='none')),
    '10차 raw 6D': ('tsem_sage_w10_h10_nagraphsage_adapted', dict(use_semantic=False, raw_append='none')),
    '10차-2 sem+pos 10D': ('tsem_sage_w10_h10_sem_pos_10d', dict(use_semantic=True, raw_append='pos')),
}


class _ZeroPolar(torch.nn.Module):
    """Δρ·접선(semantic 7·8번째 채널)을 0으로 마스킹 — 공업탑 중심 상수 기준이라 다른 장소에서
    의미가 오염되는 채널을 '결측(0)' 컨벤션으로 무력화한 변형 평가용."""

    def __init__(self, sem):
        super().__init__()
        self.sem = sem

    def forward(self, x):
        out = self.sem(x).clone()
        out[..., 6:8] = 0.0
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--loc', default='A', help='DRIFT 장소 (A/B/C/D/E/I)')
    ap.add_argument('--gpus', default='0')
    ap.add_argument('--max_files', type=int, default=6, help='평가에 쓸 CSV 수 (캐시 시간 절약)')
    ap.add_argument('--zero_polar', action='store_true',
                    help='semantic 모델의 Δρ·접선 채널을 0 마스킹 (공업탑 중심 오염 격리)')
    args = ap.parse_args()
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', args.gpus)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    csvs = sorted(glob.glob(f'/home/oem/data/TII_data/Drift/{args.loc}/*.csv'))[: args.max_files]
    print(f'[교차평가] DRIFT/{args.loc} CSV {len(csvs)}개 (max_files={args.max_files})')

    # train_ratio=0, val_ratio=0 → test split = 전체 프레임 (순수 평가용, 학습 없음)
    # build_tsem_dataloaders는 빈 train split에서 실패하므로 test dataset만 직접 생성
    test_ds = TSEMFutureStateDataset(
        csvs, W=10, H=10, radius=20.0, K_max=6, K_max2=4,
        split='test', train_ratio=0.0, val_ratio=0.0, verbose=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=4096, shuffle=False,
        collate_fn=_collate_fn, num_workers=6, pin_memory=True,
    )
    n = len(test_ds)
    print(f'[교차평가] 평가 샘플 {n:,}개')

    persist_m, _ = evaluate_tsem(None, test_loader, dev, class_names=list(CLASS_NAMES), persist_baseline=True)
    print(f"\n=== Persist baseline (DRIFT/{args.loc}) ===")
    print(f"  acc={persist_m['accuracy']*100:.2f}%  macroF1={persist_m['macro_f1']*100:.2f}%")

    for name, (exp, mkw) in CKPTS.items():
        ck_path = f'/home/oem/TNA_research/checkpoints/tsem/{exp}/best.pt'
        ck = torch.load(ck_path, map_location=dev, weights_only=False)
        cfg = ck['cfg']
        m = cfg['model']
        model = TSEMSAGE(
            node_dim=6, edge_dim=5, hidden_dim=m['hidden_dim'], d_e=m['d_e'], T=cfg['tsem']['W'],
            encoder_type=m['encoder_type'], use_attention=m['use_attention'], use_2hop=m['use_2hop'],
            use_spatial=m['use_spatial'], num_classes=m['num_classes'], dropout=m['dropout'],
            decomp_kernel=m['decomp_kernel'], decomp_learnable=m['decomp_learnable'], **mkw,
        ).to(dev)
        model.load_state_dict(ck['model'])
        if args.zero_polar and mkw.get('use_semantic', True):
            model.semantic = _ZeroPolar(model.semantic)
            name = name + ' [Δρ·접선=0]'
        model.eval()
        tm, _ = evaluate_tsem(model, test_loader, dev, class_names=list(CLASS_NAMES))
        print(f"\n=== {name} (best@{ck['epoch']}) → DRIFT/{args.loc} ===")
        print(f"  acc={tm['accuracy']*100:.2f}%  macroF1={tm['macro_f1']*100:.2f}%")
        for c, v in tm['per_class'].items():
            print(f"  {c}: recall={v['recall']*100:.2f} precision={v['precision']*100:.2f}")


if __name__ == '__main__':
    main()
