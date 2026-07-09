"""
[신규] 체크포인트 → test 클래스별 precision/recall/F1 + 혼동행렬 + Macro-F1.
사용: python journal/perclass_eval.py <exp1> [exp2 ...]   (기존 코드 불변)
data_dir는 실제 경로로 강제(체크포인트 cfg의 yklee 경로가 이 서버엔 없음).
"""
import sys, glob, os, numpy as np, torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
sys.path.insert(0, '/home/oem/TNA_research')
from modules.data_manager import build_dataloaders
from models.et_nagraphsage import ETNAGraphSAGE

DATA_DIR = '/home/oem/data/TII_data/'
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
LABELS = ['Stop', 'LaneChange', 'Normal']

def csvs(cfg):
    ds = cfg['data']['dataset']
    sub = 'Gongeoptap' if ds == 'gongeoptap' else ''
    fs = sorted(glob.glob(os.path.join(DATA_DIR, sub, '*.csv')))
    return fs

for exp in sys.argv[1:]:
    ck = f'/home/oem/TNA_research/checkpoints/{exp}/best.pt'
    if not os.path.exists(ck):
        print(f'[{exp}] 체크포인트 없음: {ck}'); continue
    c = torch.load(ck, map_location=dev, weights_only=False); cfg = c['cfg']; m = cfg['model']
    _, _, test_loader = build_dataloaders(
        csv_files=csvs(cfg), T=cfg['graph']['T'], radius=cfg['graph']['radius'],
        K_max=cfg['graph']['K_max'], K_max2=cfg['graph'].get('K_max2', 0),
        batch_size=4096, train_ratio=cfg['data']['train_ratio'], val_ratio=cfg['data']['val_ratio'],
        num_workers=6, neighbor_mode=cfg['graph'].get('neighbor_mode', 'hybrid'),
        ego_relative=cfg['graph'].get('ego_relative', False), verbose=False)
    model = ETNAGraphSAGE(
        node_dim=m['node_dim'], edge_dim=m['edge_dim'], hidden_dim=m['hidden_dim'], d_e=m['d_e'],
        T=cfg['graph']['T'], encoder_type=m['encoder_type'], use_attention=m.get('use_attention', True),
        use_2hop=m.get('use_2hop', True), num_classes=m['num_classes'], dropout=m['dropout'],
        temporal_target=m.get('temporal_target', 'both')).to(dev)
    model.load_state_dict(c['model_state']); model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for b in test_loader:
            bg = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            preds.append(model(bg).argmax(-1).cpu().numpy()); ys.append(b['y'].numpy())
    P = np.concatenate(preds); Y = np.concatenate(ys)
    print(f'\n{"="*64}\n[{exp}]  hidden_dim={m["hidden_dim"]} T={cfg["graph"]["T"]} target={m.get("temporal_target")}  (test {len(Y):,})')
    print(f'  Overall Acc {(P==Y).mean():.4f} | Macro-F1 {f1_score(Y,P,average="macro"):.4f}')
    print(classification_report(Y, P, target_names=LABELS, digits=4, zero_division=0))
    cm = confusion_matrix(Y, P)
    print('  혼동행렬 (행=정답, 열=예측):')
    print('             ' + '  '.join(f'{l:>10}' for l in LABELS))
    for i, l in enumerate(LABELS):
        row = '  '.join(f'{cm[i,j]:>10,}' for j in range(len(LABELS)))
        rec = cm[i,i]/cm[i].sum() if cm[i].sum() else 0
        print(f'  {l:>10} {row}   recall={rec:.4f}')
