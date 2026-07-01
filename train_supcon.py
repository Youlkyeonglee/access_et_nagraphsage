"""
ET-NAGraphSAGE Supervised Contrastive Learning 학습 스크립트 (Exp-B)
====================================================================
train.py 대비 변경점:
  - ETNAGraphSAGESupCon 모델 사용 (projection head 포함)
  - SupConLoss 추가: L = L_CE + λ_supcon * L_SupCon(z, labels)
  - 학습 시 return_embeddings=True로 z 추출, 평가 시 False

SupCon 핵심: LC/Normal 임베딩을 명시적으로 분리
  - 같은 클래스 샘플 → 임베딩 공간에서 가깝게
  - 다른 클래스 샘플 → 멀게 (특히 LC ↔ Normal)

사용법:
  python train_supcon.py --config configs/et_nagraphsage_supcon.yaml
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
from models.et_nagraphsage_supcon import ETNAGraphSAGESupCon


# ─────────────────────────────────────────────────────────────────────────────
# 손실 함수
# ─────────────────────────────────────────────────────────────────────────────

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., 2020).
    같은 클래스 쌍의 임베딩은 당기고, 다른 클래스 쌍은 밀어낸다.

    Args:
        temperature : τ (기본 0.07)
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features : [B, D] — L2 정규화된 임베딩
            labels   : [B]    — 클래스 레이블
        """
        B      = features.size(0)
        device = features.device

        # [B, B] cosine similarity matrix / τ
        sim = torch.matmul(features, features.T) / self.temperature

        # 수치 안정성: 각 행의 최대값 제거
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Positive mask: same class & different sample
        labels_col = labels.view(-1, 1)
        pos_mask   = (labels_col == labels_col.T).float()
        pos_mask.fill_diagonal_(0)                          # self 제거

        # Negative 분모: self 제외한 모든 쌍
        self_mask = torch.eye(B, device=device)
        exp_sim   = torch.exp(sim) * (1 - self_mask)       # [B, B]

        # Log-softmax 형태
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # 각 anchor의 positive 쌍에 대한 평균 log-prob
        n_pos = pos_mask.sum(dim=1).clamp(min=1)
        loss  = -(pos_mask * log_prob).sum(dim=1) / n_pos

        return loss.mean()


def kl_uniform_loss(logits: torch.Tensor) -> torch.Tensor:
    p   = F.softmax(logits, dim=-1)
    num = p.shape[-1]
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
    else:
        files = []
        for p in [os.path.join(data_dir, 'Gongeoptap/*.csv'),
                  os.path.join(data_dir, 'Drift/**/*.csv')]:
            files += glob.glob(p, recursive=True)
        return sorted(files)
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"CSV 파일 없음: {pattern}")
    return files


@torch.no_grad()
def evaluate(model, loader, loss_fn, kl_weight, device):
    model.eval()
    total_loss = correct = total = 0
    for batch in loader:
        batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
        logits  = model(batch_gpu, return_embeddings=False)
        targets = batch_gpu['y']
        ce_loss = loss_fn(logits, targets)
        kl_loss = kl_uniform_loss(logits) if kl_weight > 0 else torch.tensor(0.0)
        loss    = ce_loss + kl_weight * kl_loss
        preds   = logits.argmax(dim=-1)
        correct    += (preds == targets).sum().item()
        total      += targets.size(0)
        total_loss += loss.item() * targets.size(0)
    return total_loss / total, correct / total


def per_class_acc(model, loader, device, num_classes=3):
    model.eval()
    correct = torch.zeros(num_classes)
    counts  = torch.zeros(num_classes)
    with torch.no_grad():
        for batch in loader:
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            preds   = model(batch_gpu, return_embeddings=False).argmax(dim=-1).cpu()
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

    m = cfg['model']
    model = ETNAGraphSAGESupCon(
        node_dim=m['node_dim'],
        edge_dim=m['edge_dim'],
        hidden_dim=m['hidden_dim'],
        d_e=m['d_e'],
        T=cfg['graph']['T'],
        encoder_type=m['encoder_type'],
        use_attention=m.get('use_attention', True),
        use_2hop=m.get('use_2hop', False),
        num_classes=m['num_classes'],
        dropout=m['dropout'],
        proj_dim=m.get('proj_dim', 64),
    ).to(device)

    print(f"파라미터: {model.count_parameters():,}개  |  device: {device}")

    loss_cfg      = cfg['loss']
    kl_weight     = loss_cfg.get('kl_weight', 0.0)
    supcon_weight = loss_cfg.get('supcon_weight', 0.1)
    supcon_temp   = loss_cfg.get('supcon_temp', 0.07)

    ce_loss_fn     = nn.CrossEntropyLoss(
        label_smoothing=loss_cfg.get('label_smoothing', 0.0))
    supcon_loss_fn = SupConLoss(temperature=supcon_temp)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg['train']['lr'],
        weight_decay=cfg['train']['weight_decay'])

    num_epochs     = cfg['train']['num_epochs']
    scheduler_type = cfg['train'].get('scheduler', 'cosine')

    if scheduler_type == 'onecycle':
        scheduler = OneCycleLR(optimizer, max_lr=cfg['train']['lr'],
                               steps_per_epoch=len(train_loader), epochs=num_epochs)
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    else:
        scheduler = None

    exp_name = cfg.get('experiment', 'exp')
    save_dir = Path(cfg.get('save_dir', 'checkpoints')) / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

    best_val_acc = 0.0
    best_ckpt    = save_dir / 'best.pt'
    patience     = cfg['train'].get('patience', 30)
    no_improve   = 0
    use_amp      = cfg['train'].get('use_amp', True)
    scaler       = GradScaler(enabled=use_amp)

    print(f"\n{'─'*60}")
    print(f"실험: {exp_name}  |  T={cfg['graph']['T']}  |  encoder={m['encoder_type']}")
    print(f"SupCon weight={supcon_weight}  temperature={supcon_temp}  proj_dim={m.get('proj_dim',64)}")
    print(f"{'─'*60}")

    for epoch in range(1, num_epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = train_correct = train_total = 0

        for batch in train_loader:
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            targets = batch_gpu['y']
            optimizer.zero_grad()

            with autocast(enabled=use_amp):
                logits, z = model(batch_gpu, return_embeddings=True)
                ce_loss   = ce_loss_fn(logits, targets)
                sc_loss   = supcon_loss_fn(z, targets)
                kl_loss   = kl_uniform_loss(logits) if kl_weight > 0 else torch.tensor(0.0, device=device)
                loss      = ce_loss + supcon_weight * sc_loss + kl_weight * kl_loss

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
        val_loss, val_acc = evaluate(model, val_loader, ce_loss_fn, kl_weight, device)
        elapsed = time.time() - t0

        print(f"Epoch {epoch:03d}/{num_epochs} | "
              f"Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f} | "
              f"Val Loss {val_loss:.4f} | {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve   = 0
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'val_acc': val_acc, 'cfg': cfg}, best_ckpt)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    print(f"\n{'─'*60}")
    print(f"Best Val Acc: {best_val_acc:.4f}")

    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state'])

    _, test_acc = evaluate(model, test_loader, ce_loss_fn, kl_weight, device)
    class_accs  = per_class_acc(model, test_loader, device)

    print(f"Test Acc (State_Acc): {test_acc:.4f}")
    for lbl, acc in zip(['Stop', 'LaneChange', 'Normal'], class_accs):
        print(f"  {lbl}: {acc:.4f}")
    print(f"{'─'*60}")

    return test_acc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',       default='configs/et_nagraphsage_supcon.yaml')
    parser.add_argument('--T',            type=int,   default=None)
    parser.add_argument('--encoder_type', type=str,   default=None)
    parser.add_argument('--scheduler',    type=str,   default=None)
    parser.add_argument('--patience',     type=int,   default=None)
    parser.add_argument('--num_epochs',   type=int,   default=None)
    parser.add_argument('--batch_size',   type=int,   default=None)
    parser.add_argument('--dropout',      type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--supcon_weight',type=float, default=None)
    parser.add_argument('--supcon_temp',  type=float, default=None)
    parser.add_argument('--experiment',   type=str,   default=None)
    parser.add_argument('--gpu',          type=int,   default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.T             is not None: cfg['graph']['T']              = args.T
    if args.encoder_type  is not None: cfg['model']['encoder_type']   = args.encoder_type
    if args.scheduler     is not None: cfg['train']['scheduler']      = args.scheduler
    if args.patience      is not None: cfg['train']['patience']       = args.patience
    if args.num_epochs    is not None: cfg['train']['num_epochs']     = args.num_epochs
    if args.batch_size    is not None: cfg['train']['batch_size']     = args.batch_size
    if args.dropout       is not None: cfg['model']['dropout']        = args.dropout
    if args.weight_decay  is not None: cfg['train']['weight_decay']   = args.weight_decay
    if args.supcon_weight is not None: cfg['loss']['supcon_weight']   = args.supcon_weight
    if args.supcon_temp   is not None: cfg['loss']['supcon_temp']     = args.supcon_temp
    if args.experiment    is not None: cfg['experiment']              = args.experiment

    train(cfg)
