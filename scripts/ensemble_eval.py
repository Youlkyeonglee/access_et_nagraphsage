"""
Seed 앙상블 평가 (추론 전용, 학습 없음).
K1=10 500ep 4개 seed 체크포인트를 불러와 test set에서 softmax 확률을 평균 →
앙상블 Test Acc + 클래스별 정확도. 개별 seed도 함께 출력해 비교.

용도: LaneChange 분산을 잡으면 성능이 어디까지 오르는지 "천장" 확인 (진단).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from train import get_csv_files
from models.et_nagraphsage import ETNAGraphSAGE
from modules.data_manager import build_dataloaders

CKPTS = {
    'seed42':  'checkpoints/D-k10-ep500/best.pt',
    'seed846': 'checkpoints/D-k10-s846/best.pt',
    'seed862': 'checkpoints/D-k10-s862/best.pt',
    'seed995': 'checkpoints/D-k10-s995/best.pt',
}
LABELS = ['Stop', 'LaneChange', 'Normal']
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def build_model(cfg):
    m, g = cfg['model'], cfg['graph']
    model = ETNAGraphSAGE(
        node_dim=m['node_dim'], edge_dim=m['edge_dim'], hidden_dim=m['hidden_dim'],
        d_e=m['d_e'], T=g['T'], encoder_type=m['encoder_type'],
        use_attention=m.get('use_attention', True), use_2hop=m.get('use_2hop', True),
        num_classes=m['num_classes'], dropout=m['dropout'],
        temporal_target=m.get('temporal_target', 'both'),
    ).to(device)
    model.eval()
    return model


@torch.no_grad()
def collect_probs(model, loader):
    """test set 전체에 대한 softmax 확률 [N,C] 와 타깃 [N] 반환."""
    probs, ys = [], []
    for batch in loader:
        bg = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        p = F.softmax(model(bg), dim=-1).cpu()
        probs.append(p); ys.append(batch['y'])
    return torch.cat(probs), torch.cat(ys)


def metrics(pred, y):
    acc = (pred == y).float().mean().item()
    per = []
    for c in range(3):
        mask = y == c
        per.append((pred[mask] == c).float().mean().item() if mask.any() else float('nan'))
    return acc, per


def main():
    # 첫 체크포인트의 cfg로 데이터로더 구성 (4개 seed 모두 동일 데이터 설정)
    ckpt0 = torch.load(CKPTS['seed42'], map_location=device)
    cfg = ckpt0['cfg']
    g = cfg['graph']
    _, _, test_loader = build_dataloaders(
        csv_files=get_csv_files(cfg), T=g['T'], radius=g['radius'],
        K_max=g['K_max'], K_max2=g.get('K_max2', 0),
        batch_size=cfg['train']['batch_size'],
        train_ratio=cfg['data']['train_ratio'], val_ratio=cfg['data']['val_ratio'],
        num_workers=4, neighbor_mode=g.get('neighbor_mode', 'hybrid'),
        use_cache=True, verbose=False,
    )

    all_probs, y_ref = [], None
    print(f"\n{'모델':<12} {'Test Acc':>9} {'Stop':>7} {'LaneChange':>11} {'Normal':>8}")
    print('─' * 52)
    for name, path in CKPTS.items():
        ck = torch.load(path, map_location=device)
        model = build_model(ck['cfg'])
        model.load_state_dict(ck['model_state'])
        probs, y = collect_probs(model, test_loader)
        if y_ref is None: y_ref = y
        all_probs.append(probs)
        acc, per = metrics(probs.argmax(-1), y)
        print(f"{name:<12} {acc*100:>8.2f}% {per[0]*100:>6.2f}% {per[1]*100:>10.2f}% {per[2]*100:>7.2f}%")

    # 개별 평균±std
    accs = np.array([metrics(p.argmax(-1), y_ref)[0] for p in all_probs]) * 100
    lcs  = np.array([metrics(p.argmax(-1), y_ref)[1][1] for p in all_probs]) * 100
    print('─' * 52)
    print(f"{'개별 avg±std':<12} {accs.mean():>7.2f}±{accs.std(ddof=1):.2f} "
          f"| LaneChange {lcs.mean():.2f}±{lcs.std(ddof=1):.2f}")

    # 앙상블: 확률 평균
    ens = torch.stack(all_probs).mean(0)
    acc, per = metrics(ens.argmax(-1), y_ref)
    print('═' * 52)
    print(f"{'★ 앙상블':<11} {acc*100:>8.2f}% {per[0]*100:>6.2f}% {per[1]*100:>10.2f}% {per[2]*100:>7.2f}%")
    print(f"\n앙상블 vs 개별평균: Test {acc*100 - accs.mean():+.2f}%p, "
          f"LaneChange {per[1]*100 - lcs.mean():+.2f}%p")
    print(f"앙상블 vs best-seed(42): Test {acc*100 - accs.max():+.2f}%p")


if __name__ == '__main__':
    main()
