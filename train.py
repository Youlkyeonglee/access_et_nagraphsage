"""
ET-NAGraphSAGE 학습 스크립트
=============================
사용법:
  python train.py --config configs/et_nagraphsage.yaml
  python train.py --config configs/et_nagraphsage.yaml --T 1   # T=1 baseline
  python train.py --config configs/et_nagraphsage.yaml --encoder_type lstm
"""

import argparse
import glob
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

from modules.data_manager import build_dataloaders
from models.et_nagraphsage import ETNAGraphSAGE


# ─────────────────────────────────────────────────────────────────────────────
# 손실 함수
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, num_classes: int = 3,
                 weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        self.register_buffer('weight', weight)  # [num_classes] or None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p  = F.log_softmax(logits, dim=-1)
        p      = log_p.exp()
        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt     = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss   = -((1 - pt) ** self.gamma) * log_pt
        if self.weight is not None:
            w    = self.weight[targets]
            loss = loss * w
        return loss.mean()


def compute_class_weights(csv_files: list, device: torch.device) -> torch.Tensor:
    """전체 CSV에서 클래스 분포를 계산해 역수 가중치를 반환."""
    import pandas as pd
    from collections import Counter
    label_map = {'stop': 0, 'lane_change': 1, 'normal_driving': 2}
    counts = Counter()
    for f in csv_files:
        df = pd.read_csv(f)
        counts.update(df['category'].value_counts().to_dict())
    total = sum(counts.values())
    n = [counts.get(k, 1) for k in ['stop', 'lane_change', 'normal_driving']]
    weights = torch.tensor([total / (3.0 * ni) for ni in n], dtype=torch.float32)
    print(f"클래스 가중치: Stop={weights[0]:.3f}  LaneChange={weights[1]:.3f}  Normal={weights[2]:.3f}")
    return weights.to(device)


def kl_uniform_loss(logits: torch.Tensor) -> torch.Tensor:
    """KL(softmax(logits) ∥ Uniform) — attention 균등화."""
    p    = F.softmax(logits, dim=-1)
    num  = p.shape[-1]
    return (p * (p * num).log()).sum(dim=-1).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_csv_files(cfg: dict) -> list:
    data_dir = cfg['data']['data_dir']
    dataset  = cfg['data']['dataset']

    if dataset == 'gongeoptap':
        pattern = os.path.join(data_dir, 'Gongeoptap/*.csv')
    elif dataset == 'drift':
        pattern = os.path.join(data_dir, 'Drift/**/*.csv')
    else:  # both
        patterns = [
            os.path.join(data_dir, 'Gongeoptap/*.csv'),
            os.path.join(data_dir, 'Drift/**/*.csv'),
        ]
        files = []
        for p in patterns:
            files += glob.glob(p, recursive=True)
        files = sorted(files)
        if not files:
            raise FileNotFoundError(f"CSV 파일 없음: {data_dir}")
        return files

    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"CSV 파일 없음: {pattern}")
    return files


@torch.no_grad()
def evaluate(model, loader, loss_fn, kl_weight, device):
    model.eval()
    total_loss = correct = total = 0

    for batch in loader:
        batch_gpu = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        logits  = model(batch_gpu)
        targets = batch_gpu['y']

        ce_loss = loss_fn(logits, targets)
        kl_loss = kl_uniform_loss(logits) if kl_weight > 0 else torch.tensor(0.0)
        loss    = ce_loss + kl_weight * kl_loss

        preds   = logits.argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total   += targets.size(0)
        total_loss += loss.item() * targets.size(0)

    return total_loss / total, correct / total


def per_class_acc(model, loader, device, num_classes=3):
    """클래스별 정확도 계산."""
    model.eval()
    correct = torch.zeros(num_classes)
    counts  = torch.zeros(num_classes)

    with torch.no_grad():
        for batch in loader:
            batch_gpu = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            preds   = model(batch_gpu).argmax(dim=-1).cpu()
            targets = batch['y']
            for c in range(num_classes):
                mask = targets == c
                correct[c] += (preds[mask] == targets[mask]).sum()
                counts[c]  += mask.sum()

    return (correct / counts.clamp(min=1)).tolist()


# ─────────────────────────────────────────────────────────────────────────────
# 학습 루프
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(cfg.get('seed', 42))

    # ── 데이터 ──────────────────────────────────────────────────────────────
    csv_files = get_csv_files(cfg)
    print(f"CSV 파일 {len(csv_files)}개 로드")

    train_loader, val_loader, test_loader = build_dataloaders(
        csv_files=csv_files,
        T=cfg['graph']['T'],
        radius=cfg['graph']['radius'],
        K_max=cfg['graph']['K_max'],
        K_max2=cfg['graph'].get('K_max2', 0),
        batch_size=cfg['train']['batch_size'],
        train_ratio=cfg['data']['train_ratio'],
        val_ratio=cfg['data']['val_ratio'],
        num_workers=cfg['train']['num_workers'],
    )

    # ── 모델 ────────────────────────────────────────────────────────────────
    m = cfg['model']
    model = ETNAGraphSAGE(
        node_dim=m['node_dim'],
        edge_dim=m['edge_dim'],
        hidden_dim=m['hidden_dim'],
        d_e=m['d_e'],
        T=cfg['graph']['T'],
        encoder_type=m['encoder_type'],
        use_attention=m.get('use_attention', True),
        use_2hop=m.get('use_2hop', True),
        num_classes=m['num_classes'],
        dropout=m['dropout'],
    ).to(device)

    print(f"파라미터: {model.count_parameters():,}개  |  device: {device}")

    # ── 손실 / 옵티마이저 ────────────────────────────────────────────────────
    loss_cfg   = cfg['loss']
    kl_weight  = loss_cfg.get('kl_weight', 0.0)
    use_cw     = loss_cfg.get('use_class_weights', False)
    cls_weight = compute_class_weights(csv_files, device) if use_cw else None

    label_smoothing = loss_cfg.get('label_smoothing', 0.0)

    if loss_cfg['type'] == 'focal':
        loss_fn = FocalLoss(gamma=loss_cfg.get('gamma', 2.0), weight=cls_weight)
    else:
        loss_fn = nn.CrossEntropyLoss(weight=cls_weight, label_smoothing=label_smoothing)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg['train']['lr'],
        weight_decay=cfg['train']['weight_decay'],
    )

    num_epochs = cfg['train']['num_epochs']
    scheduler_type = cfg['train'].get('scheduler', 'onecycle')

    if scheduler_type == 'onecycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=cfg['train']['lr'],
            steps_per_epoch=len(train_loader),
            epochs=num_epochs,
        )
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    else:
        scheduler = None

    # ── 저장 경로 ────────────────────────────────────────────────────────────
    exp_name  = cfg.get('experiment', 'exp')
    save_dir  = Path(cfg.get('save_dir', 'checkpoints')) / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc  = 0.0
    best_ckpt     = save_dir / 'best.pt'
    patience      = cfg['train'].get('patience', 50)
    no_improve    = 0
    use_amp       = cfg['train'].get('use_amp', True)
    scaler        = GradScaler(enabled=use_amp)

    # config 파일 저장
    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

    print(f"\n{'─'*60}")
    print(f"실험: {exp_name}  |  T={cfg['graph']['T']}  |  encoder={m['encoder_type']}")
    print(f"{'─'*60}")

    for epoch in range(1, num_epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = train_correct = train_total = 0

        for batch in train_loader:
            batch_gpu = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                logits  = model(batch_gpu)
                targets = batch_gpu['y']
                ce_loss = loss_fn(logits, targets)
                kl_loss = kl_uniform_loss(logits) if kl_weight > 0 else torch.tensor(0.0, device=device)
                loss    = ce_loss + kl_weight * kl_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            if scheduler_type == 'onecycle':
                scheduler.step()

            preds          = logits.argmax(dim=-1)
            train_correct += (preds == targets).sum().item()
            train_total   += targets.size(0)
            train_loss    += loss.item() * targets.size(0)

        if scheduler_type == 'cosine' and scheduler:
            scheduler.step()

        train_acc = train_correct / train_total
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, kl_weight, device)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{num_epochs} | "
            f"Train Acc {train_acc:.4f} | "
            f"Val Acc {val_acc:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve   = 0
            torch.save({
                'epoch':      epoch,
                'model_state': model.state_dict(),
                'val_acc':    val_acc,
                'cfg':        cfg,
            }, best_ckpt)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    # ── Test 평가 ────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Best Val Acc: {best_val_acc:.4f}")

    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state'])

    _, test_acc = evaluate(model, test_loader, loss_fn, kl_weight, device)
    class_accs  = per_class_acc(model, test_loader, device)
    labels      = ['Stop', 'LaneChange', 'Normal']

    print(f"Test Acc (State_Acc): {test_acc:.4f}")
    for i, (lbl, acc) in enumerate(zip(labels, class_accs)):
        print(f"  {lbl}: {acc:.4f}")
    print(f"{'─'*60}")

    return test_acc


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/et_nagraphsage.yaml')
    # config 값을 CLI에서 덮어쓸 수 있는 인자들
    parser.add_argument('--T',            type=int,   default=None)
    parser.add_argument('--encoder_type', type=str,   default=None)
    parser.add_argument('--d_e',          type=int,   default=None)
    parser.add_argument('--num_layers',   type=int,   default=None)
    parser.add_argument('--kl_weight',    type=float, default=None)
    parser.add_argument('--gamma',        type=float, default=None)
    parser.add_argument('--seed',         type=int,   default=None)
    parser.add_argument('--experiment',   type=str,   default=None)
    parser.add_argument('--scheduler',         type=str,   default=None)
    parser.add_argument('--patience',          type=int,   default=None)
    parser.add_argument('--num_epochs',        type=int,   default=None)
    parser.add_argument('--batch_size',        type=int,   default=None)
    parser.add_argument('--use_class_weights', action='store_true', default=None)
    parser.add_argument('--label_smoothing',   type=float, default=None)
    parser.add_argument('--dropout',           type=float, default=None)
    parser.add_argument('--weight_decay',      type=float, default=None)
    parser.add_argument('--loss_type',         type=str,   default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI 인자로 config 값 덮어쓰기
    if args.T            is not None: cfg['graph']['T']              = args.T
    if args.encoder_type is not None: cfg['model']['encoder_type']   = args.encoder_type
    if args.d_e          is not None: cfg['model']['d_e']            = args.d_e
    if args.num_layers   is not None: cfg['model']['num_layers']     = args.num_layers
    if args.kl_weight    is not None: cfg['loss']['kl_weight']       = args.kl_weight
    if args.gamma        is not None: cfg['loss']['gamma']           = args.gamma
    if args.seed         is not None: cfg['seed']                    = args.seed
    if args.experiment   is not None: cfg['experiment']              = args.experiment
    if args.scheduler         is not None: cfg['train']['scheduler']             = args.scheduler
    if args.patience          is not None: cfg['train']['patience']              = args.patience
    if args.num_epochs        is not None: cfg['train']['num_epochs']            = args.num_epochs
    if args.batch_size        is not None: cfg['train']['batch_size']            = args.batch_size
    if args.use_class_weights            : cfg['loss']['use_class_weights']      = True
    if args.label_smoothing  is not None: cfg['loss']['label_smoothing']        = args.label_smoothing
    if args.dropout          is not None: cfg['model']['dropout']               = args.dropout
    if args.weight_decay     is not None: cfg['train']['weight_decay']          = args.weight_decay
    if args.loss_type        is not None: cfg['loss']['type']                   = args.loss_type

    # experiment 이름 자동 생성 (미지정 시)
    if cfg.get('experiment') in (None, 'baseline_gru_t10'):
        enc = cfg['model']['encoder_type']
        T   = cfg['graph']['T']
        cfg['experiment'] = f"et_nagraphsage_{enc}_T{T}"

    train(cfg)
