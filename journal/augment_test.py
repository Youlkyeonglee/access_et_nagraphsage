"""
허점 차단: 도로정렬(극좌표)을 world '대체'가 아니라 '추가(augment)'하면 이득이 있나?
 world(6) vs world+도로정렬채널(ρ,Δρ,접선,|Δρ|)(10). 탐지·예측, 5-seed, paired t.
"""
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from scipy import stats

dev='cuda' if torch.cuda.is_available() else 'cpu'
d=np.load('/home/oem/TNA_research/journal/anticipation_data.npz')
PO=d['ego_seq'].astype(np.float32); WO=d['world_seq'].astype(np.float32)
T2N=d['t2n']; FR=d['frame']; FI=d['file_idx']; YDET=d['y_det'].astype(np.float32); N=len(T2N)
SEEDS=[0,1,2,3,4]; H=6
# 도로정렬 전용 채널 = ego_seq의 [ρ,Δρ,접선,|Δρ|] (index 2..5; 0,1=speed,accel은 world와 중복)
POL=PO[:,:,2:6]
AUG=np.concatenate([WO,POL],-1)   # world + 도로정렬 추가

tr=np.zeros(N,bool)
for f in np.unique(FI):
    m=FI==f; thr=np.quantile(FR[m],0.7); tr|=m&(FR<=thr)
te=~tr
def std(X):
    mu=X[tr].reshape(-1,X.shape[-1]).mean(0); sd=X[tr].reshape(-1,X.shape[-1]).std(0)+1e-6
    return ((X-mu)/sd).astype(np.float32)
WO=std(WO); AUG=std(AUG)

class G(nn.Module):
    def __init__(s,fin,h=64):
        super().__init__(); s.g=nn.GRU(fin,h,batch_first=True)
        s.f=nn.Sequential(nn.Linear(h,h),nn.ReLU(),nn.Dropout(0.2),nn.Linear(h,1))
    def forward(s,x): o,_=s.g(x); return s.f(o[:,-1]).squeeze(-1)

def run(Xnp,Y,seed):
    torch.manual_seed(seed); np.random.seed(seed)
    X=torch.tensor(Xnp); Xtr=X[tr].to(dev); Ytr=torch.tensor(Y[tr],device=dev); Xte=X[te].to(dev)
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

def task(Y,name):
    w=np.array([run(WO,Y,s) for s in SEEDS]); a=np.array([run(AUG,Y,s) for s in SEEDS])
    t,p=stats.ttest_rel(a,w)
    print(f'[{name}] world {w.mean():.4f}±{w.std(ddof=1):.4f}  world+도로정렬 {a.mean():.4f}±{a.std(ddof=1):.4f}  '
          f'Δ {(a-w).mean():+.4f}  p={p:.4f}  {"이득 유의미" if p<0.05 and (a-w).mean()>0 else "이득 없음/무의"}')

print(f'표본 {N:,} device {dev}')
task(YDET,'탐지')
task((T2N<=H).astype(np.float32),f'예측 H={H}')
