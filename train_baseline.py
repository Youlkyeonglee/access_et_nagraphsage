"""Train comparison baselines (STGCN / DCRNN / TGN) under the SAME protocol as
the proposed model. Reuses train.py's data, evaluation, and result-saving so the
comparison is apples-to-apples (same data/split/budget/metric)."""
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

from train import get_csv_files, evaluate, per_class_acc, set_seed
from modules.data_manager import build_dataloaders
from modules.run_io import Tee, save_results
from models.baselines import build_baseline


def run(cfg):
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
    )

    m = cfg['model']
    model = build_baseline(
        cfg['baseline'], node_dim=m['node_dim'], edge_dim=m['edge_dim'],
        hidden_dim=m['hidden_dim'], T=cfg['graph']['T'],
        num_classes=m['num_classes'], dropout=m['dropout'],
    ).to(device)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"baseline={cfg['baseline']}  파라미터: {n_param:,}개  |  device: {device}")

    ls = cfg['loss'].get('label_smoothing', 0.0)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=ls)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'],
                                 weight_decay=cfg['train']['weight_decay'])
    num_epochs = cfg['train']['num_epochs']
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    use_amp = cfg['train'].get('use_amp', True)
    scaler = GradScaler(enabled=use_amp)

    exp_name = cfg.get('experiment', 'baseline')
    save_dir = Path(cfg.get('save_dir', 'checkpoints')) / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)
    import sys
    tee = Tee(save_dir / 'train.log'); sys.stdout = tee
    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

    best_val, best_ckpt, patience, no_imp = 0.0, save_dir / 'best.pt', cfg['train'].get('patience', 150), 0
    print(f"\n{'─'*60}\n실험: {exp_name}  |  baseline={cfg['baseline']}  |  T={cfg['graph']['T']}\n{'─'*60}")

    for epoch in range(1, num_epochs + 1):
        model.train(); t0 = time.time(); tl = tc = tt = 0
        for batch in train_loader:
            bg = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                logits = model(bg); targets = bg['y']
                loss = loss_fn(logits, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            tc += (logits.argmax(-1) == targets).sum().item(); tt += targets.size(0)
            tl += loss.item() * targets.size(0)
        scheduler.step()
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, 0.0, device)
        print(f"Epoch {epoch:03d}/{num_epochs} | Train Acc {tc/tt:.4f} | Val Acc {val_acc:.4f} | "
              f"Val Loss {val_loss:.4f} | {time.time()-t0:.1f}s")
        if val_acc > best_val:
            best_val, no_imp = val_acc, 0
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'val_acc': val_acc, 'cfg': cfg}, best_ckpt)
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"Early stopping at epoch {epoch}"); break

    print(f"\n{'─'*60}\nBest Val Acc: {best_val:.4f}")
    ck = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ck['model_state'])
    _, test_acc = evaluate(model, test_loader, loss_fn, 0.0, device)
    caccs = per_class_acc(model, test_loader, device)
    labels = ['Stop', 'LaneChange', 'Normal']
    print(f"Test Acc (State_Acc): {test_acc:.4f}")
    for lbl, a in zip(labels, caccs):
        print(f"  {lbl}: {a:.4f}")

    save_results(save_dir, {
        'experiment': exp_name, 'script': 'train_baseline.py', 'test_acc': test_acc,
        'best_val_acc': best_val, 'acc_stop': caccs[0], 'acc_lanechange': caccs[1],
        'acc_normal': caccs[2], 'encoder': cfg['baseline'], 'T': cfg['graph']['T'],
        'hidden_dim': m['hidden_dim'], 'temporal_target': '-',
        'neighbor_mode': cfg['graph'].get('neighbor_mode'),
        'num_epochs': cfg['train']['num_epochs'], 'seed': cfg.get('seed'),
    })
    tee.close()
    return test_acc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/et_nagraphsage_2hop_base_ep500.yaml')
    p.add_argument('--baseline', required=True, choices=['stgcn', 'dcrnn', 'tgn'])
    p.add_argument('--hidden_dim', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--num_epochs', type=int, default=None)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--experiment', type=str, default=None)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg['baseline'] = args.baseline
    if args.hidden_dim is not None: cfg['model']['hidden_dim'] = args.hidden_dim
    if args.batch_size is not None: cfg['train']['batch_size'] = args.batch_size
    if args.num_epochs is not None: cfg['train']['num_epochs'] = args.num_epochs
    if args.seed       is not None: cfg['seed'] = args.seed
    cfg['experiment'] = args.experiment or f"baseline_{args.baseline}"
    run(cfg)
