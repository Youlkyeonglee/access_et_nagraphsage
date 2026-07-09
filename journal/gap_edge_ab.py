"""
[신규] 공업탑 gap-엣지 ablation — 새 이웃↔이웃 gap이 기존 나↔이웃 엣지보다 기여하나?
====================================================================================
경량 GRU, 파일별 시간분할(70/30), 5-seed. 두 과제 모두 평가:
  (1) 현재상태 분류(3-class) → LaneChange-AUC
  (2) anticipation (t2n<=H, H=6) → 미래 차선변경 예측 AUC
4-way ablation:
  A: world ego only
  B: + edge_base  (기존 나↔이웃 gap)         ← 공업탑에서 ≈0이었던 것
  C: + edge_gap   (새 이웃↔이웃 hole+closing) ← 사용자 제안
  D: + both
핵심 판정: C가 A/B를 유의미하게 넘으면 gap-acceptance 신호가 산다. C≈B면 중복.
"""
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from scipy import stats

dev='cuda' if torch.cuda.is_available() else 'cpu'
d=np.load('/home/oem/TNA_research/journal/gap_edge_data.npz')
WE=d['world_seq'].astype(np.float32); EB=d['edge_base'].astype(np.float32); EG=d['edge_gap'].astype(np.float32)
YC=d['y_cur'].astype(np.int64); T2N=d['t2n']; FR=d['frame']; FI=d['file_idx']; N=len(YC)
H_ANT=6
def clean(X): return np.clip(np.nan_to_num(X,nan=0.,posinf=0.,neginf=0.),-500,500).astype(np.float32)
WE,EB,EG=map(clean,[WE,EB,EG])
SEEDS=[0,1,2,3,4]

# 파일별 시간분할 (앞70% train)
tr=np.zeros(N,bool)
for f in np.unique(FI):
    m=FI==f
    if m.sum()<20: tr|=m; continue
    thr=np.quantile(FR[m],0.7); tr|=m&(FR<=thr)
te=~tr
def std(X):
    mu=X[tr].reshape(-1,X.shape[-1]).mean(0); sd=X[tr].reshape(-1,X.shape[-1]).std(0)+1e-6
    return ((X-mu)/sd).astype(np.float32)
WE,EB,EG=std(WE),std(EB),std(EG)

def feats(tag):
    parts=[WE]
    if 'B' in tag: parts.append(EB)
    if 'G' in tag: parts.append(EG)
    return np.concatenate(parts,-1)

class G(nn.Module):
    def __init__(s,fin,nc,h=64):
        super().__init__(); s.g=nn.GRU(fin,h,batch_first=True)
        s.f=nn.Sequential(nn.Linear(h,h),nn.ReLU(),nn.Dropout(0.2),nn.Linear(h,nc))
    def forward(s,x): o,_=s.g(x); return s.f(o[:,-1])

def run(X,Y,nc,seed,pos_idx=None):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr=torch.tensor(X[tr]).to(dev); Ytr=torch.tensor(Y[tr]).to(dev); Xte=torch.tensor(X[te]).to(dev)
    cw=torch.tensor([ (Y[tr]!=c).sum()/max(1,(Y[tr]==c).sum()) for c in range(nc)],dtype=torch.float32)
    m=G(X.shape[-1],nc).to(dev); opt=torch.optim.Adam(m.parameters(),1e-3,weight_decay=1e-5)
    lf=nn.CrossEntropyLoss(weight=cw.to(dev)); bs=8192
    for ep in range(25):
        m.train(); idx=torch.randperm(len(Xtr),device=dev)
        for i in range(0,len(Xtr),bs):
            b=idx[i:i+bs]; opt.zero_grad(); lf(m(Xtr[b]),Ytr[b]).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        pr=np.concatenate([torch.softmax(m(Xte[i:i+16384]),-1).cpu().numpy() for i in range(0,len(Xte),16384)])
    yte=Y[te]
    if nc==3:
        return roc_auc_score((yte==1).astype(int),pr[:,1])   # LaneChange AUC
    return roc_auc_score(yte,pr[:,1])                          # anticipation AUC

def eval_task(name,Y,nc):
    print(f'\n=== {name} ===')
    res={}
    for tag in ['A','AB','AG','ABG']:
        X=feats(tag); aucs=[run(X,Y,nc,s) for s in SEEDS]
        res[tag]=np.array(aucs)
        print(f'  {tag:3s} ({X.shape[-1]:2d}ch): {res[tag].mean():.4f} ± {res[tag].std(ddof=1):.4f}')
    def cmp(a,b,lbl):
        t,p=stats.ttest_rel(res[b],res[a])
        dz=(res[b]-res[a]).mean()
        print(f'  {lbl}: Δ {dz:+.4f}  p={p:.4f}  {"유의미↑" if p<0.05 and dz>0 else "무의/음"}')
    cmp('A','AB','기존엣지 B vs A(ego)')
    cmp('A','AG','새 gap엣지 G vs A(ego)')
    cmp('AB','ABG','gap 추가 (AB→ABG)')   # 기존 엣지 위에 gap이 더 주나?
    return res

# 현재상태 3-class → LaneChange AUC
print(f'표본 {N:,}  train {tr.sum():,}  test {te.sum():,}  dev {dev}')
eval_task('현재상태 분류 (LaneChange-AUC)', YC, 3)
# anticipation: 미래 H프레임 내 차선변경
YA=(T2N<=H_ANT).astype(np.int64)
print(f'\nanticipation 양성률 {YA.mean()*100:.1f}%')
eval_task(f'anticipation H={H_ANT} (미래 차선변경 AUC)', YA, 2)
