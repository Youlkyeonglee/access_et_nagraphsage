"""
TSEM-SAGE 학습 스크립트
=======================
ET-NAGraphSAGE `train.py`와 분리. TSEM 전용 데이터·모델만 사용.

사용법:
  python train_tsem.py --config configs/tsem_sage.yaml
  python train_tsem.py --config configs/tsem_sage.yaml --W 20 --H 10
  python train_tsem.py --config configs/tsem_sage.yaml --model tsem_semantic_only
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

from models.tsem_sage import TSEMSAGE, TSEMSemanticOnly, TSEMNAGraphSAGEAdapted
from modules.data_manager_tsem import build_tsem_dataloaders, summarize_label_distribution
from modules.tsem_eval import evaluate_tsem
from modules.tsem_instant_label import CLASS_NAMES
from modules.tsem_augment import TsemAugment


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer('weight', weight)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor,
        sample_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1)
        p = log_p.exp()
        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -((1 - pt) ** self.gamma) * log_pt
        if self.weight is not None:
            loss = loss * self.weight[targets]
        if sample_weight is not None:
            loss = loss * sample_weight
        return loss.mean()


def stop_uncertainty_weight(stop_conf: torch.Tensor) -> torch.Tensor:
    """8차 (2026-07-08, 안 1): 라벨 불확실성 가중치. stop_conf(배치별 Beta-Binomial 사후분산
    기반 confidence, docs/TSEM_journal_design.html §12.12)를 배치 평균으로 재정규화해
    전체 loss 스케일은 유지하면서 애매한 샘플(경계 flicker)의 loss 기여만 상대적으로 낮춘다."""
    return stop_conf / stop_conf.mean().clamp(min=1e-8)


def kl_uniform_loss(logits: torch.Tensor) -> torch.Tensor:
    """KL(softmax(logits) ∥ Uniform) — train.py(ET-NAGraphSAGE)와 동일한 정의 (출력 확신도 균등화).
    지금까지 configs/tsem_sage.yaml에 kl_weight 필드는 있었으나 train_tsem.py에 배선되어 있지 않았음."""
    p = F.softmax(logits, dim=-1)
    num = p.shape[-1]
    return (p * (p * num).clamp(min=1e-8).log()).sum(dim=-1).mean()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_csv_files(cfg: dict) -> list:
    data_dir = cfg['data']['data_dir']
    dataset = cfg['data']['dataset']
    if dataset == 'gongeoptap':
        pattern = os.path.join(data_dir, 'Gongeoptap/*.csv')
    elif dataset == 'drift':
        pattern = os.path.join(data_dir, '*.csv')
        files = sorted(glob.glob(pattern))
        if not files:
            pattern = os.path.join(data_dir, 'Drift/**/*.csv')
            files = sorted(glob.glob(pattern, recursive=True))
        if not files:
            raise FileNotFoundError(f'CSV 없음: {data_dir}')
        return files
    else:
        patterns = [
            os.path.join(data_dir, 'Gongeoptap/*.csv'),
            os.path.join(data_dir, 'Drift/**/*.csv'),
        ]
        files = []
        for p in patterns:
            files += glob.glob(p, recursive=True)
        files = sorted(files)
        if not files:
            raise FileNotFoundError(f'CSV 없음: {data_dir}')
        return files
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f'CSV 없음: {pattern}')
    return files


def build_model(cfg: dict, T: int):
    mcfg = cfg['model']
    name = mcfg.get('name', 'tsem_sage')
    kw = dict(
        hidden_dim=mcfg['hidden_dim'],
        d_e=mcfg['d_e'],
        T=T,
        encoder_type=mcfg['encoder_type'],
        use_attention=mcfg['use_attention'],
        use_2hop=mcfg.get('use_2hop', True),
        use_spatial=mcfg.get('use_spatial', True),
        raw_append=mcfg.get('raw_append', 'none'),
        num_classes=mcfg['num_classes'],
        dropout=mcfg['dropout'],
        decomp_kernel=mcfg.get('decomp_kernel', 5),
        decomp_learnable=mcfg.get('decomp_learnable', True),
    )
    if name == 'tsem_semantic_only':
        return TSEMSemanticOnly(**kw)
    if name == 'tsem_nagraphsage_adapted':
        return TSEMNAGraphSAGEAdapted(**kw)
    return TSEMSAGE(**kw)


def class_weights_from_distribution(dist: dict, device, power: float = 1.0) -> torch.Tensor:
    """역빈도 가중치의 power제곱 완화판. power=1.0 원래 역빈도, 0.5 sqrt 완화, 0.0 균등."""
    counts = [max(dist[n]['count'], 1) for n in CLASS_NAMES]
    total = sum(counts)
    w = torch.tensor([(total / (3.0 * c)) ** power for c in counts], dtype=torch.float32)
    print(
        f'TSEM 클래스 가중치 (state@t+H, power={power}): '
        + '  '.join(f'{n}={w[i]:.3f}' for i, n in enumerate(CLASS_NAMES))
    )
    return w.to(device)


@torch.no_grad()
def run_eval(model, loader, device):
    metrics, _ = evaluate_tsem(model, loader, device, class_names=list(CLASS_NAMES))
    return metrics


def train(cfg: dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpus = torch.cuda.device_count()
    set_seed(cfg.get('seed', 42))

    W = cfg['tsem']['W']
    H = cfg['tsem']['H']
    stop_delta = cfg['tsem'].get('stop_persist_delta', 2)
    csv_files = get_csv_files(cfg)

    print(f'[TSEM] CSV {len(csv_files)}개  W={W}  H={H}  stop_persist_delta={stop_delta}')
    label_dist = summarize_label_distribution(csv_files, W, H, stop_persist_delta=stop_delta)
    print('[TSEM] train split state(t+H) 분포:', json.dumps(label_dist, indent=2))

    gcfg = cfg['graph']
    tcfg = cfg['train']
    acfg = cfg.get('augment', {})
    augment = TsemAugment(
        noise_std=acfg.get('noise_std', 0.0),
        neighbor_dropout_p=acfg.get('neighbor_dropout_p', 0.0),
        frame_dropout_p=acfg.get('frame_dropout_p', 0.0),
        rotate_deg=acfg.get('rotate_deg', 0.0),
    )
    print(f'[TSEM] augmentation (train split만 적용): {augment.describe()}')
    train_loader, val_loader, test_loader = build_tsem_dataloaders(
        csv_files,
        W=W,
        H=H,
        radius=gcfg['radius'],
        K_max=gcfg['K_max'],
        K_max2=gcfg['K_max2'],
        batch_size=tcfg['batch_size'],
        train_ratio=cfg['data']['train_ratio'],
        val_ratio=cfg['data']['val_ratio'],
        num_workers=tcfg['num_workers'],
        augment=augment if augment.enabled else None,
        stop_persist_delta=stop_delta,
    )

    raw_model = build_model(cfg, T=W).to(device)
    print(f'모델: {cfg["model"].get("name", "tsem_sage")}  params={raw_model.count_parameters():,}')
    model = raw_model
    if n_gpus > 1:
        print(f'[TSEM] {n_gpus}개 GPU 사용 (nn.DataParallel, 배치 {cfg["train"]["batch_size"]}를 GPU당 '
              f'{cfg["train"]["batch_size"] // n_gpus}로 분할)')
        model = nn.DataParallel(raw_model)

    lcfg = cfg['loss']
    weight = None
    if lcfg.get('use_class_weights', True):
        weight = class_weights_from_distribution(
            label_dist, device, power=lcfg.get('class_weight_power', 1.0)
        )
    if lcfg['type'] == 'focal':
        loss_fn = FocalLoss(gamma=lcfg['gamma'], weight=weight)
    else:
        loss_fn = nn.CrossEntropyLoss(weight=weight)

    def apply_loss(logits, targets, sw):
        if isinstance(loss_fn, FocalLoss):
            return loss_fn(logits, targets, sample_weight=sw)
        return loss_fn(logits, targets)

    lambda_temporal = lcfg.get('lambda_temporal', 0.0)
    lambda_spatial = lcfg.get('lambda_spatial', 0.0)
    kl_weight = lcfg.get('kl_weight', 0.0)
    use_aux = lambda_temporal > 0 or lambda_spatial > 0 or kl_weight > 0
    use_uncertainty_weight = lcfg.get('use_uncertainty_weight', False)
    print(
        f'TSEM loss 구성: L_main + {lambda_temporal}*L_temporal + '
        f'{lambda_spatial}*L_spatial + {kl_weight}*KL(main_logits‖Uniform)'
        + ('  [8차: stop_conf 기반 불확실성 가중치 적용]' if use_uncertainty_weight else '')
    )

    opt = torch.optim.AdamW(
        model.parameters(), lr=tcfg['lr'], weight_decay=tcfg['weight_decay']
    )
    steps = len(train_loader) * tcfg['num_epochs']
    if tcfg.get('scheduler') == 'onecycle':
        sched = OneCycleLR(opt, max_lr=tcfg['lr'], total_steps=steps)
    elif tcfg.get('scheduler') == 'cosine':
        sched = CosineAnnealingLR(opt, T_max=steps)
    else:
        sched = None

    save_dir = Path(cfg['save_dir']) / cfg['experiment']
    save_dir.mkdir(parents=True, exist_ok=True)
    scaler = GradScaler(enabled=device.type == 'cuda')

    patience = tcfg.get('early_stop_patience', 0)
    print(f'[TSEM] early stopping: {"patience=" + str(patience) + " epoch" if patience > 0 else "미사용"}')

    best_val = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    for epoch in range(1, tcfg['num_epochs'] + 1):
        model.train()
        t0 = time.time()
        total_loss = n = 0
        for batch in train_loader:
            batch_gpu = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            opt.zero_grad(set_to_none=True)
            sw = stop_uncertainty_weight(batch_gpu['stop_conf']) if use_uncertainty_weight else None
            with autocast(enabled=device.type == 'cuda'):
                if use_aux:
                    out = model(batch_gpu, return_aux=True)
                    logits = out['logits']
                    loss = apply_loss(logits, batch_gpu['y'], sw)
                    if lambda_temporal > 0:
                        loss = loss + lambda_temporal * apply_loss(out['logits_temporal'], batch_gpu['y'], sw)
                    if lambda_spatial > 0 and out['logits_spatial'] is not None:
                        loss = loss + lambda_spatial * apply_loss(out['logits_spatial'], batch_gpu['y'], sw)
                    if kl_weight > 0:
                        loss = loss + kl_weight * kl_uniform_loss(logits)
                else:
                    logits = model(batch_gpu)
                    loss = apply_loss(logits, batch_gpu['y'], sw)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if sched is not None:
                sched.step()
            total_loss += loss.item() * batch_gpu['y'].size(0)
            n += batch_gpu['y'].size(0)

        val_m = run_eval(model, val_loader, device)
        elapsed = time.time() - t0
        print(
            f'Epoch {epoch:03d}  loss={total_loss/n:.4f}  '
            f'val_acc={val_m["accuracy"]:.4f}  macro_f1={val_m["macro_f1"]:.4f}  '
            f'LC_recall={val_m["per_class"]["lane_change"]["recall"]:.4f}  '
            f'({elapsed:.1f}s)'
        )
        if val_m['accuracy'] > best_val:
            best_val = val_m['accuracy']
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {'model': raw_model.state_dict(), 'cfg': cfg, 'epoch': epoch},
                save_dir / 'best.pt',
            )
        else:
            epochs_no_improve += 1
            if patience > 0 and epochs_no_improve >= patience:
                print(
                    f'[TSEM] early stopping: val_acc가 {patience} epoch 연속 '
                    f'best({best_val:.4f}@epoch{best_epoch})를 갱신 못 해 epoch {epoch}에서 중단'
                )
                break

    # 최종 평가는 마지막 epoch가 아니라 best.pt(최고 val_acc) 가중치로 — 과적합 구간까지
    # 돈 경우 마지막 epoch 가중치는 best보다 나쁠 수 있음(9차에서 실측: peak 78.05%대 vs 후반 정체 79%대
    # val_acc이지만 test는 다시 재보고해야 정확 — 항상 best.pt 기준으로 report한다).
    raw_model.load_state_dict(torch.load(save_dir / 'best.pt', map_location=device)['model'])
    print(f'[TSEM] 최종 평가는 best.pt(epoch {best_epoch}, val_acc={best_val:.4f}) 가중치로 수행')
    test_m = run_eval(model, test_loader, device)
    persist_m, _ = evaluate_tsem(
        None, test_loader, device, class_names=list(CLASS_NAMES), persist_baseline=True
    )
    results = {
        'test': test_m,
        'persist_baseline_test': persist_m,
        'label_distribution_train': label_dist,
        'W': W,
        'H': H,
    }
    with open(save_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('[TSEM] test:', json.dumps(test_m, indent=2))
    print('[TSEM] Persist baseline test acc:', persist_m['accuracy'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/tsem_sage.yaml')
    parser.add_argument('--W', type=int, default=None)
    parser.add_argument('--H', type=int, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--encoder_type', type=str, default=None)
    parser.add_argument('--experiment', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    # 학교서버 sweep용 오버라이드 (2026-07-09, docs/20260709school.md 참조)
    parser.add_argument('--stop_delta', type=int, default=None, help='stop ±δ 다수결의 δ (기본 2)')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--radius', type=float, default=None, help='이웃 반경 (기본 20.0)')
    parser.add_argument('--raw_append', type=str, default=None,
                        help="semantic에 추가할 raw 채널: none|pos|pos_dir|polar")
    parser.add_argument('--decomp_kernel', type=int, default=None, help='low-pass 커널 크기 (기본 5)')
    parser.add_argument(
        '--gpus', type=str, default=None,
        help='콤마로 구분된 GPU id (예: "0,1,2,3") — 지정 시 CUDA_VISIBLE_DEVICES를 덮어쓰고, '
             '2개 이상이면 nn.DataParallel로 단일 실행 내 멀티 GPU 사용. 미지정 시 기존처럼 '
             '셸의 CUDA_VISIBLE_DEVICES(단일 GPU 병렬 실행 방식) 그대로 따름.',
    )
    args = parser.parse_args()
    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.W is not None:
        cfg['tsem']['W'] = args.W
    if args.H is not None:
        cfg['tsem']['H'] = args.H
    if args.model is not None:
        cfg['model']['name'] = args.model
    if args.encoder_type is not None:
        cfg['model']['encoder_type'] = args.encoder_type
    if args.experiment is not None:
        cfg['experiment'] = args.experiment
    if args.epochs is not None:
        cfg['train']['num_epochs'] = args.epochs
    if args.stop_delta is not None:
        cfg['tsem']['stop_persist_delta'] = args.stop_delta
    if args.seed is not None:
        cfg['seed'] = args.seed
    if args.radius is not None:
        cfg['graph']['radius'] = args.radius
    if args.raw_append is not None:
        cfg['model']['raw_append'] = args.raw_append
    if args.decomp_kernel is not None:
        cfg['model']['decomp_kernel'] = args.decomp_kernel
    train(cfg)


if __name__ == '__main__':
    main()
