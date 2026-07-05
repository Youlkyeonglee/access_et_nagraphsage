"""fp32 flagship 4-seed 앙상블 test 평가 + 혼동행렬.
개별 모델 softmax 평균 = 앙상블. 추가 학습 없음."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train import get_csv_files, evaluate
from modules.data_manager import build_dataloaders
from models.et_nagraphsage import ETNAGraphSAGE

CKPTS = {
    's42':  'checkpoints/D-h192/best.pt',
    's846': 'checkpoints/D-h192-fp32-s846/best.pt',
    's862': 'checkpoints/D-h192-fp32-s862/best.pt',
    's995': 'checkpoints/D-h192-fp32-s995/best.pt',
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
labels = ['Stop', 'LaneChange', 'Normal']

# test_loader (첫 ckpt cfg 기준; 4개 동일 설정)
c0 = torch.load(list(CKPTS.values())[0], map_location='cpu')
cfg = c0['cfg']
_, _, test_loader = build_dataloaders(
    csv_files=get_csv_files(cfg), T=cfg['graph']['T'], radius=cfg['graph']['radius'],
    K_max=cfg['graph']['K_max'], K_max2=cfg['graph'].get('K_max2', 0),
    batch_size=cfg['train']['batch_size'], train_ratio=cfg['data']['train_ratio'],
    val_ratio=cfg['data']['val_ratio'], num_workers=cfg['train']['num_workers'],
    neighbor_mode=cfg['graph'].get('neighbor_mode', 'hybrid'),
)

def build(cfg):
    m = cfg['model']
    return ETNAGraphSAGE(
        node_dim=m['node_dim'], edge_dim=m['edge_dim'], hidden_dim=m['hidden_dim'],
        d_e=m['d_e'], T=cfg['graph']['T'], encoder_type=m['encoder_type'],
        use_attention=m.get('use_attention', True), use_2hop=m.get('use_2hop', True),
        num_classes=m['num_classes'], dropout=m['dropout'],
        temporal_target=m.get('temporal_target', 'both'),
    ).to(device)

# 모델 로드
models = {}
for name, path in CKPTS.items():
    ck = torch.load(path, map_location=device)
    mdl = build(ck['cfg']); mdl.load_state_dict(ck['model_state']); mdl.eval()
    models[name] = mdl

# test 순회: 개별 + 앙상블 softmax 누적
N = 3
conf_ens = np.zeros((N, N), dtype=np.int64)
per_model_correct = {k: 0 for k in models}
ens_correct = tot = 0
ens_class_correct = np.zeros(N); ens_class_total = np.zeros(N)

with torch.no_grad():
    for batch in test_loader:
        bg = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        y = bg['y']
        prob_sum = None
        for name, mdl in models.items():
            logits = mdl(bg)
            p = F.softmax(logits, dim=-1)
            per_model_correct[name] += (logits.argmax(-1) == y).sum().item()
            prob_sum = p if prob_sum is None else prob_sum + p
        ens_pred = prob_sum.argmax(-1)
        ens_correct += (ens_pred == y).sum().item(); tot += y.size(0)
        for t, pr in zip(y.cpu().numpy(), ens_pred.cpu().numpy()):
            conf_ens[t, pr] += 1
            ens_class_total[t] += 1
            if t == pr: ens_class_correct[t] += 1

print("===== 개별 모델 Test Acc =====")
for k in models: print(f"  {k}: {per_model_correct[k]/tot:.4f}")
print(f"\n===== 앙상블 (softmax 평균) =====")
print(f"  Ensemble Test Acc: {ens_correct/tot:.4f}")
for i, lbl in enumerate(labels):
    print(f"    {lbl:11s}: {ens_class_correct[i]/max(ens_class_total[i],1):.4f}")

print(f"\n===== 혼동행렬 (앙상블, row=정답 / col=예측) =====")
print(f"{'':12s}" + "".join(f"{l:>12s}" for l in labels))
for i, lbl in enumerate(labels):
    row = conf_ens[i]
    print(f"{lbl:12s}" + "".join(f"{v:>12d}" for v in row) + f"   (recall {row[i]/max(row.sum(),1):.3f})")
# LC 오분류 분해
lc = conf_ens[1]
print(f"\nLaneChange 오분류: →Stop {lc[0]} ({lc[0]/lc.sum()*100:.1f}%) | →Normal {lc[2]} ({lc[2]/lc.sum()*100:.1f}%)")
print(f"Normal→LaneChange 오분류: {conf_ens[2,1]} ({conf_ens[2,1]/conf_ens[2].sum()*100:.2f}%)")
