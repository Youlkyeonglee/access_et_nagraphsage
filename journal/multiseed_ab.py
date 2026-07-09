"""Phase1-①: 다seed A vs B (GRU) — +ΔAUC 유의성 확인."""
import numpy as np, torch, torch.nn as nn, sys
from sklearn.metrics import roc_auc_score
from scipy import stats

dev='cuda' if torch.cuda.is_available() else 'cpu'
d=np.load('/home/oem/TNA_research/journal/anticipation_data.npz')
E=d['ego_seq'].astype(np.float32); ED=d['edge_seq'].astype(np.float32)
T2N=d['t2n']; FR=d['frame']; FI=d['file_idx']; N=len(T2N)
H=int(sys.argv[1]) if len(sys.argv)>1 else 6
SEEDS=[0,1,2,3,4]

tr=np.zeros(N,bool)
for f in np.unique(FI):
    m=FI==f; thr=np.quantile(FR[m],0.7); tr|=m&(FR<=thr)
te=~tr
def std(X):
    mu=X[tr].reshape(-1,X.shape[-1]).mean(0); sd=X[tr].reshape(-1,X.shape[-1]).std(0)+1e-6
    return (X-mu)/sd
E=std(E); ED=std(ED)
Xa=torch.tensor(E); Xb=torch.tensor(np.concatenate([E,ED],-1))
Y=(T2N<=H).astype(np.float32)

class G(nn.Module):
    def __init__(s,fin,h=64):
        super().__init__(); s.g=nn.GRU(fin,h,batch_first=True)
        s.f=nn.Sequential(nn.Linear(h,h),nn.ReLU(),nn.Dropout(0.2),nn.Linear(h,1))
    def forward(s,x): o,_=s.g(x); return s.f(o[:,-1]).squeeze(-1)

def run(X,seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr=X[tr].to(dev); Ytr=torch.tensor(Y[tr],device=dev)
    Xte=X[te].to(dev)
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

print(f'H={H} 양성 {Y.mean()*100:.1f}%  seeds={SEEDS}')
As,Bs=[],[]
for s in SEEDS:
    a=run(Xa,s); b=run(Xb,s); As.append(a); Bs.append(b)
    print(f'  seed{s}: A {a:.4f}  B {b:.4f}  Δ {b-a:+.4f}')
As=np.array(As); Bs=np.array(Bs); D=Bs-As
t,p=stats.ttest_rel(Bs,As)
print(f'\nA  {As.mean():.4f}±{As.std(ddof=1):.4f}')
print(f'B  {Bs.mean():.4f}±{Bs.std(ddof=1):.4f}')
print(f'ΔAUC {D.mean():+.4f}±{D.std(ddof=1):.4f}  paired t={t:.2f} p={p:.4f}  {"유의미" if p<0.05 else "무의"}')
