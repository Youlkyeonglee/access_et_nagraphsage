"""저장된 best.pt 체크포인트로 test 세트 State_Acc를 재계산.
학습 프로세스가 test 평가 전에 종료된 경우 사후 평가용."""
import sys
from pathlib import Path
import torch

from train import get_csv_files, evaluate, per_class_acc
from modules.data_manager import build_dataloaders
from modules.run_io import save_results
from models.et_nagraphsage import ETNAGraphSAGE
import torch.nn as nn

EXPS = sys.argv[1:] or ['C-node', 'C-edge', 'D-h192', 'D-h256']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
labels = ['Stop', 'LaneChange', 'Normal']
rows = []

for exp in EXPS:
    ckpt_path = Path('checkpoints') / exp / 'best.pt'
    if not ckpt_path.exists():
        print(f"[{exp}] best.pt 없음 — 스킵"); continue
    ck = torch.load(ckpt_path, map_location=device)
    cfg = ck['cfg']; m = cfg['model']

    _, _, test_loader = build_dataloaders(
        csv_files=get_csv_files(cfg),
        T=cfg['graph']['T'], radius=cfg['graph']['radius'],
        K_max=cfg['graph']['K_max'], K_max2=cfg['graph'].get('K_max2', 0),
        batch_size=cfg['train']['batch_size'],
        train_ratio=cfg['data']['train_ratio'], val_ratio=cfg['data']['val_ratio'],
        num_workers=cfg['train']['num_workers'],
        neighbor_mode=cfg['graph'].get('neighbor_mode', 'hybrid'),
    )

    model = ETNAGraphSAGE(
        node_dim=m['node_dim'], edge_dim=m['edge_dim'], hidden_dim=m['hidden_dim'],
        d_e=m['d_e'], T=cfg['graph']['T'], encoder_type=m['encoder_type'],
        use_attention=m.get('use_attention', True), use_2hop=m.get('use_2hop', True),
        num_classes=m['num_classes'], dropout=m['dropout'],
        temporal_target=m.get('temporal_target', 'both'),
    ).to(device)
    model.load_state_dict(ck['model_state'])

    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg['loss'].get('label_smoothing', 0.0))
    _, test_acc = evaluate(model, test_loader, loss_fn, 0.0, device)
    caccs = per_class_acc(model, test_loader, device)

    print(f"\n===== {exp} =====")
    print(f"best epoch {ck['epoch']} | Val {ck['val_acc']:.4f} | Test(State_Acc) {test_acc:.4f}")
    for lbl, a in zip(labels, caccs):
        print(f"  {lbl:11s}: {a:.4f}")

    save_results(Path('checkpoints') / exp, {
        'experiment': exp, 'script': 'eval_ckpt.py',
        'test_acc': test_acc, 'best_val_acc': ck['val_acc'],
        'acc_stop': caccs[0], 'acc_lanechange': caccs[1], 'acc_normal': caccs[2],
        'encoder': m.get('encoder_type'), 'T': cfg['graph']['T'],
        'hidden_dim': m.get('hidden_dim'), 'temporal_target': m.get('temporal_target'),
        'neighbor_mode': cfg['graph'].get('neighbor_mode'),
        'num_epochs': cfg['train']['num_epochs'], 'seed': cfg.get('seed'),
    })
    rows.append((exp, ck['val_acc'], test_acc, caccs))

print("\n\n======== 요약 ========")
print(f"{'실험':10s} {'Val':>7s} {'Test':>7s} {'Stop':>7s} {'LC':>7s} {'Normal':>7s}")
for exp, v, t, c in rows:
    print(f"{exp:10s} {v:7.4f} {t:7.4f} {c[0]:7.4f} {c[1]:7.4f} {c[2]:7.4f}")
