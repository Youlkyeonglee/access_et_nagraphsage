"""
[신규] DRIFT 본 모델 체크포인트에서 LaneChange AUC + macro AUC 사후 산출.
train.py는 정확도만 보고 → 경량 A/B(AUC)와 비교하려면 AUC 필요. 재학습 없음.
both/node/edge best.pt를 각자의 cfg로 복원해 test set 예측 → AUC.
기존 코드 불변 (build_dataloaders/모델만 import).
"""
import sys, numpy as np, torch
from sklearn.metrics import roc_auc_score
sys.path.insert(0, '/home/oem/TNA_research')
from modules.data_manager import build_dataloaders
from models.et_nagraphsage import ETNAGraphSAGE

dev='cuda' if torch.cuda.is_available() else 'cpu'
import glob, os
def csvs(cfg):
    dd=cfg['data']['data_dir']; fs=sorted(glob.glob(os.path.join(dd,'*.csv')))
    return fs

for exp in ['both','node','edge']:
    ck=f'/home/oem/TNA_research/checkpoints/D-drift-2hop-{exp}/best.pt'
    c=torch.load(ck, map_location=dev, weights_only=False); cfg=c['cfg']; m=cfg['model']
    _,_,test_loader=build_dataloaders(
        csv_files=csvs(cfg), T=cfg['graph']['T'], radius=cfg['graph']['radius'],
        K_max=cfg['graph']['K_max'], K_max2=cfg['graph'].get('K_max2',0),
        batch_size=4096, train_ratio=cfg['data']['train_ratio'], val_ratio=cfg['data']['val_ratio'],
        num_workers=6, neighbor_mode=cfg['graph'].get('neighbor_mode','hybrid'),
        ego_relative=cfg['graph'].get('ego_relative',False), verbose=False)
    model=ETNAGraphSAGE(node_dim=m['node_dim'],edge_dim=m['edge_dim'],hidden_dim=m['hidden_dim'],
        d_e=m['d_e'],T=cfg['graph']['T'],encoder_type=m['encoder_type'],
        use_attention=m.get('use_attention',True),use_2hop=m.get('use_2hop',True),
        num_classes=m['num_classes'],dropout=m['dropout'],
        temporal_target=m.get('temporal_target','both')).to(dev)
    model.load_state_dict(c['model_state']); model.eval()
    probs=[]; ys=[]
    with torch.no_grad():
        for b in test_loader:
            bg={k:v.to(dev) if isinstance(v,torch.Tensor) else v for k,v in b.items()}
            p=torch.softmax(model(bg),-1).cpu().numpy(); probs.append(p); ys.append(b['y'].numpy())
    P=np.concatenate(probs); Y=np.concatenate(ys)
    macro=roc_auc_score(Y,P,multi_class='ovr',average='macro')
    lc=roc_auc_score((Y==1).astype(int),P[:,1])
    acc=(P.argmax(1)==Y).mean()
    print(f'[{exp:4s}] State_Acc {acc:.4f} | macro-AUC {macro:.4f} | LaneChange-AUC {lc:.4f}')
