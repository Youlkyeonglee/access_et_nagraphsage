"""
CRAT-Pred-adapted 학습 스크립트 — TSEM-SAGE와 동일 데이터·loss·평가로 공정 비교.

comparison/hivt_tsem, qcnet_tsem/train_*.py와 완전히 동일한 구조 — train_tsem.py의 핵심 로직을
그대로 재사용하고 모델 클래스만 CratPredTSEMAdapted로 교체.

사용법:
  cd /home/oem/TNA_research
  python comparison/cratpred_tsem/train_cratpred_tsem.py --config comparison/cratpred_tsem/config.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from modules.data_manager_tsem import build_tsem_dataloaders, summarize_label_distribution  # noqa: E402
from modules.tsem_eval import evaluate_tsem  # noqa: E402
from modules.tsem_instant_label import CLASS_NAMES  # noqa: E402
from modules.tsem_augment import TsemAugment  # noqa: E402
from train_tsem import (  # noqa: E402  (프로젝트 루트의 기존 학습 유틸 재사용)
    FocalLoss, class_weights_from_distribution, get_csv_files,
    kl_uniform_loss, set_seed, stop_uncertainty_weight,
)

from model import CratPredTSEMAdapted  # noqa: E402


def train(cfg: dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpus = torch.cuda.device_count()
    set_seed(cfg.get('seed', 42))

    W = cfg['tsem']['W']
    H = cfg['tsem']['H']
    stop_delta = cfg['tsem'].get('stop_persist_delta', 2)
    csv_files = get_csv_files(cfg)

    print(f'[CRAT-Pred-TSEM] CSV {len(csv_files)}개  W={W}  H={H}  stop_persist_delta={stop_delta}')
    label_dist = summarize_label_distribution(csv_files, W, H, stop_persist_delta=stop_delta)
    print('[CRAT-Pred-TSEM] train split state(t+H) 분포:', json.dumps(label_dist, indent=2))

    gcfg = cfg['graph']
    tcfg = cfg['train']
    acfg = cfg.get('augment', {})
    augment = TsemAugment(
        noise_std=acfg.get('noise_std', 0.0),
        neighbor_dropout_p=acfg.get('neighbor_dropout_p', 0.0),
        frame_dropout_p=acfg.get('frame_dropout_p', 0.0),
        rotate_deg=acfg.get('rotate_deg', 0.0),
    )
    print(f'[CRAT-Pred-TSEM] augmentation (train split만 적용): {augment.describe()}')
    train_loader, val_loader, test_loader = build_tsem_dataloaders(
        csv_files, W=W, H=H, radius=gcfg['radius'], K_max=gcfg['K_max'], K_max2=gcfg['K_max2'],
        batch_size=tcfg['batch_size'], train_ratio=cfg['data']['train_ratio'],
        val_ratio=cfg['data']['val_ratio'], num_workers=tcfg['num_workers'],
        augment=augment if augment.enabled else None, stop_persist_delta=stop_delta,
    )

    mcfg = cfg['model']
    raw_model = CratPredTSEMAdapted(
        W=W, K=gcfg['K_max'], latent_size=mcfg['latent_size'], num_heads=mcfg['num_heads'],
        num_classes=mcfg['num_classes'],
    ).to(device)
    print(f'모델: CRAT-Pred-adapted  params={raw_model.count_parameters():,}')
    model = raw_model
    if n_gpus > 1:
        print(f'[CRAT-Pred-TSEM] {n_gpus}개 GPU 사용 (nn.DataParallel)')
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

    kl_weight = lcfg.get('kl_weight', 0.0)
    use_uncertainty_weight = lcfg.get('use_uncertainty_weight', False)
    print(f'[CRAT-Pred-TSEM] loss 구성: L_focal + {kl_weight}*KL(logits‖Uniform)'
          + ('  [stop_conf 불확실성 가중치 적용]' if use_uncertainty_weight else ''))

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg['lr'], weight_decay=tcfg['weight_decay'])
    steps = len(train_loader) * tcfg['num_epochs']
    if tcfg.get('scheduler') == 'onecycle':
        sched = OneCycleLR(opt, max_lr=tcfg['lr'], total_steps=steps)
    elif tcfg.get('scheduler') == 'cosine':
        sched = CosineAnnealingLR(opt, T_max=steps)
    else:
        sched = None

    save_dir = Path(cfg['save_dir']) / cfg['experiment']
    save_dir.mkdir(parents=True, exist_ok=True)
    use_amp = device.type == 'cuda' and tcfg.get('use_amp', True)
    scaler = GradScaler(enabled=use_amp)

    patience = tcfg.get('early_stop_patience', 0)
    grad_clip = tcfg.get('grad_clip_norm', 0)
    print(f'[CRAT-Pred-TSEM] early stopping: {"patience=" + str(patience) if patience > 0 else "미사용"}'
          f'  grad_clip: {"max_norm=" + str(grad_clip) if grad_clip > 0 else "미사용"}')

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
            with autocast(enabled=use_amp):
                logits = model(batch_gpu)
                loss = apply_loss(logits, batch_gpu['y'], sw)
                if kl_weight > 0:
                    loss = loss + kl_weight * kl_uniform_loss(logits)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            if sched is not None:
                sched.step()
            total_loss += loss.item() * batch_gpu['y'].size(0)
            n += batch_gpu['y'].size(0)

        val_m, _ = evaluate_tsem(model, val_loader, device, class_names=list(CLASS_NAMES))
        elapsed = time.time() - t0
        print(f'Epoch {epoch:03d}  loss={total_loss/n:.4f}  val_acc={val_m["accuracy"]:.4f}  '
              f'macro_f1={val_m["macro_f1"]:.4f}  '
              f'LC_recall={val_m["per_class"]["lane_change"]["recall"]:.4f}  ({elapsed:.1f}s)')
        if val_m['accuracy'] > best_val:
            best_val = val_m['accuracy']
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({'model': raw_model.state_dict(), 'cfg': cfg, 'epoch': epoch}, save_dir / 'best.pt')
        else:
            epochs_no_improve += 1
            if patience > 0 and epochs_no_improve >= patience:
                print(f'[CRAT-Pred-TSEM] early stopping: {patience} epoch 연속 미갱신, epoch {epoch}에서 중단')
                break

    raw_model.load_state_dict(torch.load(save_dir / 'best.pt', map_location=device)['model'])
    print(f'[CRAT-Pred-TSEM] 최종 평가는 best.pt(epoch {best_epoch}, val_acc={best_val:.4f})로 수행')
    test_m, _ = evaluate_tsem(model, test_loader, device, class_names=list(CLASS_NAMES))
    persist_m, _ = evaluate_tsem(None, test_loader, device, class_names=list(CLASS_NAMES), persist_baseline=True)
    results = {
        'model': 'CRAT-Pred-adapted', 'test': test_m, 'persist_baseline_test': persist_m,
        'label_distribution_train': label_dist, 'W': W, 'H': H,
        'params': raw_model.count_parameters(),
    }
    with open(save_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('[CRAT-Pred-TSEM] test:', json.dumps(test_m, indent=2))
    print('[CRAT-Pred-TSEM] Persist baseline test acc:', persist_m['accuracy'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(Path(__file__).resolve().parent / 'config.yaml'))
    parser.add_argument('--experiment', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--gpus', type=str, default=None,
                        help='콤마 구분 GPU id — 지정 시 CUDA_VISIBLE_DEVICES 덮어씀 (train_tsem.py와 동일 관례)')
    args = parser.parse_args()
    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.experiment is not None:
        cfg['experiment'] = args.experiment
    if args.epochs is not None:
        cfg['train']['num_epochs'] = args.epochs
    if args.seed is not None:
        cfg['seed'] = args.seed
    train(cfg)


if __name__ == '__main__':
    main()
