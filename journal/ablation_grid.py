"""
논문 메인 테이블: 2×2 절제 (표현 × 엣지) × (탐지·예측), 5-seed GRU.
표현: world 좌표 ego vs 극좌표(도로정렬) ego
엣지: 없음 vs +맵 링-gap
→ 표현 이득(world→polar) + 엣지 이득(no→edge)을 유의성과 함께.
"""
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from scipy import stats

dev='cuda' if torch.cuda.is_available() else 'cpu'
d=np.load('/home/oem/TNA_research/journal/anticipation_data.npz')
PO=d['ego_seq'].astype(np.float32); WO=d['world_seq'].astype(np.float32); ED=d['edge_seq'].astype(np.float32)
T2N=d['t2n']; FR=d['frame']; FI=d['file_idx']; YDET=d['y_det'].astype(np.float32); N=len(T2N)
SEEDS=[0,1,2,3,4]; H=6

tr=np.zeros(N,bool)
for f in np.unique(FI):
    m=FI==f; thr=np.quantile(FR[m],0.7); tr|=m&(FR<=thr)
te=~tr
def std(X):
    mu=X[tr].reshape(-1,X.shape[-1]).mean(0); sd=X[tr].reshape(-1,X.shape[-1]).std(0)+1e-6
    return ((X-mu)/sd).astype(np.float32)
PO=std(PO); WO=std(WO); ED=std(ED)

class G(nn.Module):
    def __init__(s,fin,h=64):
        super().__init__(); s.g=nn.GRU(fin,h,batch_first=True)
        s.f=nn.Sequential(nn.Linear(h,h),nn.ReLU(),nn.Dropout(0.2),nn.Linear(h,1))
    def forward(s,x): o,_=s.g(x); return s.f(o[:,-1]).squeeze(-1)

def run(Xnp,Y,seed):
    torch.manual_seed(seed); np.random.seed(seed)
    X=torch.tensor(Xnp)
    Xtr=X[tr].to(dev); Ytr=torch.tensor(Y[tr],device=dev); Xte=X[te].to(dev)
    pw=torch.tensor([(Ytr==0).sum()/max(1,(Ytr==1).sum())],device=dev)
    m=G(X.shape[-1]).to(dev); opt=torch.optim.Adam(m.parameters(),1e-3,weight_decay=1e-5)
    lf=nn.BCEWithLogitsLoss(pos_weight=pw); bs=8192
    for ep in range(20):
        m.train(); idx=torch.randperm(len(Xtr),device=dev)
        for i in range(0,len(Xtr),bs):
            b=idx[i:i+bs]; opt.zero_grad(); lf(m(Xtr[b]),Ytr[b]).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        p=np.concatenate([torch.sigmoid(m(Xte[i:i+16384])).cpu().numpy() for i in range(0,len(Xte),16384)])
    return roc_auc_score(Y[te],p)

def cell(Xnp,Y):
    v=np.array([run(Xnp,Y,s) for s in SEEDS]); return v
def grid(Y,name):
    print(f'\n=== {name} (양성 {Y.mean()*100:.1f}%) ===')
    aw=cell(WO,Y); bw=cell(np.concatenate([WO,ED],-1),Y)
    ap=cell(PO,Y); bp=cell(np.concatenate([PO,ED],-1),Y)
    def s(v): return f'{v.mean():.4f}±{v.std(ddof=1):.4f}'
    print(f'          |   엣지없음        |   +맵엣지')
    print(f'  world   | {s(aw)} | {s(bw)}')
    print(f'  polar   | {s(ap)} | {s(bp)}')
    # 이득 + 유의성
    def pt(a,b):
        t,p=stats.ttest_rel(b,a); return f'{(b-a).mean():+.4f} (p={p:.4f})'
    print(f'  표현 이득(world→polar, 엣지없음): {pt(aw,ap)}')
    print(f'  엣지 이득(polar, no→edge):        {pt(ap,bp)}')
    print(f'  둘 다(world/none → polar/edge):   {pt(aw,bp)}')

print(f'표본 {N:,}  device {dev}  seeds {SEEDS}')
grid(YDET,'탐지 (현재 lane_change)')
grid((T2N<=H).astype(np.float32), f'예측 (미래 {H}프레임)')
