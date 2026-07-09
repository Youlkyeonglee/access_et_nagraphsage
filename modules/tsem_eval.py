"""
TSEM 평가 지표 — Future State Acc, per-class recall, macro-F1
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch


def confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 3,
) -> torch.Tensor:
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(targets.view(-1), preds.view(-1)):
        cm[int(t), int(p)] += 1
    return cm


def metrics_from_confusion(
    cm: torch.Tensor,
    class_names: Optional[List[str]] = None,
) -> Dict[str, object]:
    n = cm.shape[0]
    names = class_names or [str(i) for i in range(n)]
    support = cm.sum(dim=1).float()
    pred_count = cm.sum(dim=0).float()
    tp = cm.diag().float()
    recall = tp / support.clamp(min=1)
    precision = tp / pred_count.clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
    acc = tp.sum() / cm.sum().clamp(min=1)
    macro_f1 = f1.mean()
    per_class = {
        names[i]: {
            'recall': float(recall[i]),
            'precision': float(precision[i]),
            'f1': float(f1[i]),
            'support': int(support[i]),
        }
        for i in range(n)
    }
    return {
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'per_class': per_class,
        'confusion': cm.cpu().numpy().tolist(),
    }


@torch.no_grad()
def evaluate_tsem(
    model: Optional[torch.nn.Module],
    loader,
    device: torch.device,
    num_classes: int = 3,
    class_names: Optional[List[str]] = None,
    persist_baseline: bool = False,
) -> Tuple[Dict[str, object], float]:
    """
    persist_baseline=True 이면 model 무시, y_persist로 평가.
    반환: (metrics dict, mean loss placeholder 0)
    """
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    if model is not None and not persist_baseline:
        model.eval()

    for batch in loader:
        targets = batch['y']
        if persist_baseline:
            preds = batch['y_persist']
        else:
            batch_gpu = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            preds = model(batch_gpu).argmax(dim=-1).cpu()
        cm += confusion_matrix(preds, targets, num_classes)

    return metrics_from_confusion(cm, class_names), 0.0
