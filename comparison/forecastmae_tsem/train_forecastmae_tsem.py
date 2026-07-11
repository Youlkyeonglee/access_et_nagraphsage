"""
Forecast-MAE-adapted 학습 스크립트 — 2단계 프로토콜(원본 그대로).

  1) 사전학습 (--mode pretrain): 레이블 미사용, 마스킹된 에이전트의 과거 궤적 복원 loss만으로
     인코더(TSEMMAEEncoder)를 학습. 완료 시 인코더 가중치를 별도 체크포인트로 저장.
  2) 미세조정 (--mode finetune): 1)에서 저장한 인코더 가중치를 로드해 분류 헤드를 얹고,
     TSEM-SAGE와 동일한 FocalLoss/class weight/OneCycleLR/early stopping/평가로 지도학습.

두 단계 모두 데이터로더·augmentation은 build_tsem_dataloaders(train_tsem.py와 동일)를 그대로
쓴다 — 사전학습 단계도 배치 dict의 'y'만 무시할 뿐 나머지는 미세조정과 완전히 같은 파이프라인.

사용법:
  cd /home/oem/TNA_research
  python comparison/forecastmae_tsem/train_forecastmae_tsem.py --config comparison/forecastmae_tsem/config.yaml --mode pretrain
  python comparison/forecastmae_tsem/train_forecastmae_tsem.py --config comparison/forecastmae_tsem/config.yaml --mode finetune \
      --pretrained_ckpt checkpoints/comparison/forecastmae_tsem_w10_h10/pretrain_encoder.pt
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
from train_tsem import (  # noqa: E402
    FocalLoss, class_weights_from_distribution, get_csv_files,
    kl_uniform_loss, set_seed, stop_uncertainty_weight,
)

from model import TSEMMAEPretrain, TSEMMAEFinetune  # noqa: E402


def _build_loaders(cfg: dict, batch_size: int, num_workers: int):
    W, H = cfg['tsem']['W'], cfg['tsem']['H']
    stop_delta = cfg['tsem'].get('stop_persist_delta', 2)
    csv_files = get_csv_files(cfg)
    gcfg = cfg['graph']
    acfg = cfg.get('augment', {})
    augment = TsemAugment(
        noise_std=acfg.get('noise_std', 0.0),
        neighbor_dropout_p=acfg.get('neighbor_dropout_p', 0.0),
        frame_dropout_p=acfg.get('frame_dropout_p', 0.0),
        rotate_deg=acfg.get('rotate_deg', 0.0),
    )
    label_dist = summarize_label_distribution(csv_files, W, H, stop_persist_delta=stop_delta)
    train_loader, val_loader, test_loader = build_tsem_dataloaders(
        csv_files, W=W, H=H, radius=gcfg['radius'], K_max=gcfg['K_max'], K_max2=gcfg['K_max2'],
        batch_size=batch_size, train_ratio=cfg['data']['train_ratio'],
        val_ratio=cfg['data']['val_ratio'], num_workers=num_workers,
        augment=augment if augment.enabled else None, stop_persist_delta=stop_delta,
    )
    return train_loader, val_loader, test_loader, label_dist, W, H


def pretrain(cfg: dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpus = torch.cuda.device_count()
    set_seed(cfg.get('seed', 42))

    tcfg = cfg['pretrain']
    print(f'[MAE-Pretrain] augmentation은 train split에 동일 적용 (마스킹 복원 loss만 사용, 레이블 미사용)')
    train_loader, val_loader, _, _, W, H = _build_loaders(cfg, tcfg['batch_size'], tcfg['num_workers'])

    mcfg = cfg['model']
    raw_model = TSEMMAEPretrain(
        W=W, K=cfg['graph']['K_max'], embed_dim=mcfg['embed_dim'], encoder_depth=mcfg['encoder_depth'],
        decoder_depth=mcfg['decoder_depth'], num_heads=mcfg['num_heads'], mask_ratio=mcfg['mask_ratio'],
        dropout=mcfg['dropout'],
    ).to(device)
    print(f'모델: Forecast-MAE-adapted(사전학습)  params={raw_model.count_parameters():,}')
    model = raw_model
    if n_gpus > 1:
        print(f'[MAE-Pretrain] {n_gpus}개 GPU 사용 (nn.DataParallel)')
        model = nn.DataParallel(raw_model)

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg['lr'], weight_decay=tcfg['weight_decay'])
    steps = len(train_loader) * tcfg['num_epochs']
    sched = OneCycleLR(opt, max_lr=tcfg['lr'], total_steps=steps) if tcfg.get('scheduler') == 'onecycle' else None

    save_dir = Path(cfg['save_dir']) / cfg['experiment']
    save_dir.mkdir(parents=True, exist_ok=True)
    use_amp = device.type == 'cuda' and tcfg.get('use_amp', True)
    scaler = GradScaler(enabled=use_amp)
    grad_clip = tcfg.get('grad_clip_norm', 0)

    best_val = float('inf')
    for epoch in range(1, tcfg['num_epochs'] + 1):
        model.train()
        t0 = time.time()
        total_loss = n = 0
        for batch in train_loader:
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                loss = model(batch_gpu)
                if loss.dim() > 0:  # DataParallel이 GPU별 스칼라를 모아 반환하는 경우 평균
                    loss = loss.mean()
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            if sched is not None:
                sched.step()
            bs = batch_gpu['y'].size(0)
            total_loss += loss.item() * bs
            n += bs

        model.eval()
        val_loss = val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                vloss = model(batch_gpu)
                if vloss.dim() > 0:
                    vloss = vloss.mean()
                bs = batch_gpu['y'].size(0)
                val_loss += vloss.item() * bs
                val_n += bs
        val_loss /= max(val_n, 1)
        elapsed = time.time() - t0
        print(f'[Pretrain] Epoch {epoch:03d}  train_recon_loss={total_loss/n:.4f}  '
              f'val_recon_loss={val_loss:.4f}  ({elapsed:.1f}s)')
        if val_loss < best_val:
            best_val = val_loss
            torch.save({'encoder': raw_model.encoder.state_dict(), 'cfg': cfg, 'epoch': epoch},
                      save_dir / 'pretrain_encoder.pt')

    print(f'[MAE-Pretrain] 완료 — best val_recon_loss={best_val:.4f}, '
          f'인코더 저장: {save_dir / "pretrain_encoder.pt"}')


def finetune(cfg: dict, pretrained_ckpt: str = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpus = torch.cuda.device_count()
    set_seed(cfg.get('seed', 42))

    tcfg = cfg['finetune']
    train_loader, val_loader, test_loader, label_dist, W, H = _build_loaders(
        cfg, tcfg['batch_size'], tcfg['num_workers'])
    print('[MAE-Finetune] train split state(t+H) 분포:', json.dumps(label_dist, indent=2))

    mcfg = cfg['model']
    raw_model = TSEMMAEFinetune(
        W=W, K=cfg['graph']['K_max'], embed_dim=mcfg['embed_dim'], encoder_depth=mcfg['encoder_depth'],
        num_heads=mcfg['num_heads'], dropout=mcfg['dropout'], num_classes=mcfg['num_classes'],
    ).to(device)

    if pretrained_ckpt:
        missing, unexpected = raw_model.load_pretrained_encoder(pretrained_ckpt, map_location=device)
        print(f'[MAE-Finetune] 사전학습 인코더 로드: {pretrained_ckpt} '
              f'(missing={missing}, unexpected={unexpected})')
    else:
        print('[MAE-Finetune] ⚠️ --pretrained_ckpt 미지정 — 사전학습 없이 랜덤 초기화로 학습 '
              '(Forecast-MAE 원 설계와 어긋남, ablation 비교용으로만 사용할 것)')

    print(f'모델: Forecast-MAE-adapted(미세조정)  params={raw_model.count_parameters():,}')
    model = raw_model
    if n_gpus > 1:
        print(f'[MAE-Finetune] {n_gpus}개 GPU 사용 (nn.DataParallel)')
        model = nn.DataParallel(raw_model)

    lcfg = cfg['loss']
    weight = None
    if lcfg.get('use_class_weights', True):
        weight = class_weights_from_distribution(label_dist, device, power=lcfg.get('class_weight_power', 1.0))
    loss_fn = FocalLoss(gamma=lcfg['gamma'], weight=weight) if lcfg['type'] == 'focal' else nn.CrossEntropyLoss(weight=weight)

    def apply_loss(logits, targets, sw):
        if isinstance(loss_fn, FocalLoss):
            return loss_fn(logits, targets, sample_weight=sw)
        return loss_fn(logits, targets)

    kl_weight = lcfg.get('kl_weight', 0.0)
    use_uncertainty_weight = lcfg.get('use_uncertainty_weight', False)

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg['lr'], weight_decay=tcfg['weight_decay'])
    steps = len(train_loader) * tcfg['num_epochs']
    sched = OneCycleLR(opt, max_lr=tcfg['lr'], total_steps=steps) if tcfg.get('scheduler') == 'onecycle' else \
        (CosineAnnealingLR(opt, T_max=steps) if tcfg.get('scheduler') == 'cosine' else None)

    save_dir = Path(cfg['save_dir']) / cfg['experiment']
    save_dir.mkdir(parents=True, exist_ok=True)
    use_amp = device.type == 'cuda' and tcfg.get('use_amp', True)
    scaler = GradScaler(enabled=use_amp)
    patience = tcfg.get('early_stop_patience', 0)
    grad_clip = tcfg.get('grad_clip_norm', 0)

    best_val = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    for epoch in range(1, tcfg['num_epochs'] + 1):
        model.train()
        t0 = time.time()
        total_loss = n = 0
        for batch in train_loader:
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
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
        print(f'[Finetune] Epoch {epoch:03d}  loss={total_loss/n:.4f}  val_acc={val_m["accuracy"]:.4f}  '
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
                print(f'[MAE-Finetune] early stopping: {patience} epoch 연속 미갱신, epoch {epoch}에서 중단')
                break

    raw_model.load_state_dict(torch.load(save_dir / 'best.pt', map_location=device)['model'])
    print(f'[MAE-Finetune] 최종 평가는 best.pt(epoch {best_epoch}, val_acc={best_val:.4f})로 수행')
    test_m, _ = evaluate_tsem(model, test_loader, device, class_names=list(CLASS_NAMES))
    persist_m, _ = evaluate_tsem(None, test_loader, device, class_names=list(CLASS_NAMES), persist_baseline=True)
    results = {
        'model': 'Forecast-MAE-adapted', 'pretrained_ckpt': pretrained_ckpt,
        'test': test_m, 'persist_baseline_test': persist_m,
        'label_distribution_train': label_dist, 'W': W, 'H': H,
        'params': raw_model.count_parameters(),
    }
    with open(save_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('[MAE-Finetune] test:', json.dumps(test_m, indent=2))
    print('[MAE-Finetune] Persist baseline test acc:', persist_m['accuracy'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(Path(__file__).resolve().parent / 'config.yaml'))
    parser.add_argument('--mode', choices=['pretrain', 'finetune'], required=True)
    parser.add_argument('--pretrained_ckpt', type=str, default=None,
                        help='finetune 모드에서 로드할 사전학습 인코더 체크포인트 경로')
    parser.add_argument('--experiment', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--gpus', type=str, default=None)
    args = parser.parse_args()
    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.experiment is not None:
        cfg['experiment'] = args.experiment
    if args.epochs is not None:
        cfg[args.mode]['num_epochs'] = args.epochs
    if args.seed is not None:
        cfg['seed'] = args.seed

    if args.mode == 'pretrain':
        pretrain(cfg)
    else:
        default_ckpt = str(Path(cfg['save_dir']) / cfg['experiment'] / 'pretrain_encoder.pt')
        ckpt = args.pretrained_ckpt or (default_ckpt if os.path.exists(default_ckpt) else None)
        finetune(cfg, pretrained_ckpt=ckpt)


if __name__ == '__main__':
    main()
