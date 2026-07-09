"""
ET-NAGraphSAGE-MR 학습 (기존 train.py 불변, 헬퍼 재사용 + MR 모델만 교체).
사용:
  python train_mr.py --config configs/et_nagraphsage_2hop_base_ep500.yaml \
      --num_relations 4 --experiment MR-both --temporal_target both
"""
import argparse, os, sys, time, yaml
from pathlib import Path
import torch, torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

from modules.data_manager import build_dataloaders
from modules.run_io import Tee, save_results
from models.et_nagraphsage_mr import ETNAGraphSAGEMR
# 기존 train.py의 헬퍼 재사용 (모델 비의존)
from train import (FocalLoss, compute_class_weights, kl_uniform_loss,
                   set_seed, get_csv_files, evaluate, per_class_acc)


def train(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(cfg.get('seed', 42))
    csv_files = get_csv_files(cfg)
    print(f"CSV 파일 {len(csv_files)}개 로드")

    train_loader, val_loader, test_loader = build_dataloaders(
        csv_files=csv_files, T=cfg['graph']['T'], radius=cfg['graph']['radius'],
        K_max=cfg['graph']['K_max'], K_max2=cfg['graph'].get('K_max2', 0),
        batch_size=cfg['train']['batch_size'], train_ratio=cfg['data']['train_ratio'],
        val_ratio=cfg['data']['val_ratio'], num_workers=cfg['train']['num_workers'],
        neighbor_mode=cfg['graph'].get('neighbor_mode', 'hybrid'),
        ego_relative=cfg['graph'].get('ego_relative', False),
        use_cache=cfg['train'].get('use_cache', True))

    m = cfg['model']
    model = ETNAGraphSAGEMR(
        node_dim=m['node_dim'], edge_dim=m['edge_dim'], hidden_dim=m['hidden_dim'],
        d_e=m['d_e'], T=cfg['graph']['T'], encoder_type=m['encoder_type'],
        use_attention=m.get('use_attention', True), use_2hop=m.get('use_2hop', True),
        num_classes=m['num_classes'], dropout=m['dropout'],
        temporal_target=m.get('temporal_target', 'both'),
        num_relations=m.get('num_relations', 4), lat_w=m.get('lat_w', 2.5)).to(device)
    print(f"[MR] num_relations={m.get('num_relations',4)}  파라미터: {model.count_parameters():,}개  |  {device}")

    loss_cfg = cfg['loss']; kl_weight = loss_cfg.get('kl_weight', 0.0)
    cls_weight = compute_class_weights(csv_files, device) if loss_cfg.get('use_class_weights', False) else None
    ls = loss_cfg.get('label_smoothing', 0.0)
    if loss_cfg['type'] == 'focal':
        loss_fn = FocalLoss(gamma=loss_cfg.get('gamma', 2.0), weight=cls_weight)
    else:
        loss_fn = nn.CrossEntropyLoss(weight=cls_weight, label_smoothing=ls)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'],
                                 weight_decay=cfg['train']['weight_decay'])
    num_epochs = cfg['train']['num_epochs']; sched_t = cfg['train'].get('scheduler', 'cosine')
    if sched_t == 'onecycle':
        scheduler = OneCycleLR(optimizer, max_lr=cfg['train']['lr'],
                               steps_per_epoch=len(train_loader), epochs=num_epochs)
    elif sched_t == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    else:
        scheduler = None

    exp_name = cfg.get('experiment', 'MR-exp')
    save_dir = Path(cfg.get('save_dir', 'checkpoints')) / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val = 0.0; best_ckpt = save_dir / 'best.pt'
    patience = cfg['train'].get('patience', 50); no_improve = 0
    use_amp = cfg['train'].get('use_amp', True); scaler = GradScaler(enabled=use_amp)
    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
    tee = Tee(save_dir / 'train.log'); sys.stdout = tee
    print(f"\n{'─'*60}\n실험(MR): {exp_name} | R={m.get('num_relations',4)} | T={cfg['graph']['T']}\n{'─'*60}")

    for epoch in range(1, num_epochs + 1):
        model.train(); t0 = time.time()
        for batch in train_loader:
            bg = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                logits = model(bg); targets = bg['y']
                loss = loss_fn(logits, targets)
                if kl_weight > 0: loss = loss + kl_weight * kl_uniform_loss(logits)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            if sched_t == 'onecycle': scheduler.step()
        if sched_t == 'cosine' and scheduler: scheduler.step()
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, kl_weight, device)
        print(f"Epoch {epoch:03d}/{num_epochs} | Val Acc {val_acc:.4f} | Val Loss {val_loss:.4f} | {time.time()-t0:.1f}s")
        if val_acc > best_val:
            best_val = val_acc; no_improve = 0
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'val_acc': val_acc, 'cfg': cfg}, best_ckpt)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}"); break

    print(f"\nBest Val Acc: {best_val:.4f}")
    ckpt = torch.load(best_ckpt, map_location=device); model.load_state_dict(ckpt['model_state'])
    _, test_acc = evaluate(model, test_loader, loss_fn, kl_weight, device)
    ca = per_class_acc(model, test_loader, device)
    print(f"Test Acc (State_Acc): {test_acc:.4f}")
    for lbl, a in zip(['Stop', 'LaneChange', 'Normal'], ca): print(f"  {lbl}: {a:.4f}")
    save_results(save_dir, {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'experiment': exp_name, 'script': 'train_mr.py',
        'test_acc': round(test_acc, 4), 'acc_stop': round(ca[0], 4),
        'acc_lanechange': round(ca[1], 4), 'acc_normal': round(ca[2], 4),
        'best_val_acc': round(best_val, 4), 'encoder': m.get('encoder_type'), 'T': cfg['graph']['T'],
        'hidden_dim': m.get('hidden_dim'), 'temporal_target': m.get('temporal_target'),
        'num_relations': m.get('num_relations', 4), 'seed': cfg.get('seed')})
    tee.close(); return test_acc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/et_nagraphsage_2hop_base_ep500.yaml')
    p.add_argument('--num_relations', type=int, default=None)
    p.add_argument('--lat_w', type=float, default=None)
    p.add_argument('--temporal_target', type=str, default=None, choices=['both', 'node', 'edge'])
    p.add_argument('--hidden_dim', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--num_epochs', type=int, default=None)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--experiment', type=str, default=None)
    p.add_argument('--data_dir', type=str, default=None)
    return p.parse_args()


if __name__ == '__main__':
    a = parse_args()
    with open(a.config) as f: cfg = yaml.safe_load(f)
    if a.num_relations is not None: cfg['model']['num_relations'] = a.num_relations
    if a.lat_w is not None: cfg['model']['lat_w'] = a.lat_w
    if a.temporal_target is not None: cfg['model']['temporal_target'] = a.temporal_target
    if a.hidden_dim is not None: cfg['model']['hidden_dim'] = a.hidden_dim
    if a.batch_size is not None: cfg['train']['batch_size'] = a.batch_size
    if a.num_epochs is not None: cfg['train']['num_epochs'] = a.num_epochs
    if a.seed is not None: cfg['seed'] = a.seed
    if a.experiment is not None: cfg['experiment'] = a.experiment
    if a.data_dir is not None: cfg['data']['data_dir'] = a.data_dir
    train(cfg)
