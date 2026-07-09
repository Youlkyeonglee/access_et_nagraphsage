"""
[3] 시계열 GRU로 정식 A vs B (지평별).
 A = GRU(ego_seq[10,6])  B = GRU([ego_seq|edge_seq][10,14])
 시간분할, 표준화(train fit), pos_weight BCE. test AUC/AP 비교.
"""
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

dev='cuda' if torch.cuda.is_available() else 'cpu'
d=np.load('/home/oem/TNA_research/journal/anticipation_data.npz')
E=d['ego_seq'].astype(np.float32); ED=d['edge_seq'].astype(np.float32)
T2N=d['t2n']; FR=d['frame']; FI=d['file_idx']; N=len(T2N)

# 시간분할
tr=np.zeros(N,bool)
for f in np.unique(FI):
    m=FI==f; thr=np.quantile(FR[m],0.7); tr|=m&(FR<=thr)
te=~tr

def standardize(X, tr):
    mu=X[tr].reshape(-1,X.shape[-1]).mean(0); sd=X[tr].reshape(-1,X.shape[-1]).std(0)+1e-6
    return (X-mu)/sd
E=standardize(E,tr); ED=standardize(ED,tr)
Xa=torch.tensor(E); Xb=torch.tensor(np.concatenate([E,ED],axis=-1))

class GRUClf(nn.Module):
    def __init__(s,fin,h=64):
        super().__init__(); s.g=nn.GRU(fin,h,batch_first=True); s.f=nn.Sequential(nn.Linear(h,h),nn.ReLU(),nn.Dropout(0.2),nn.Linear(h,1))
    def forward(s,x): o,_=s.g(x); return s.f(o[:,-1]).squeeze(-1)

def train_eval(X,Y,tag):
    Xtr=X[tr].to(dev); Ytr=torch.tensor(Y[tr],dtype=torch.float32,device=dev)
    Xte=X[te].to(dev); Yte=Y[te]
    pw=torch.tensor([(Ytr==0).sum()/max(1,(Ytr==1).sum())],device=dev)
    m=GRUClf(X.shape[-1]).to(dev); opt=torch.optim.Adam(m.parameters(),1e-3,weight_decay=1e-5)
    lossf=nn.BCEWithLogitsLoss(pos_weight=pw)
    bs=8192
    for ep in range(20):
        m.train(); idx=torch.randperm(len(Xtr),device=dev)
        for i in range(0,len(Xtr),bs):
            b=idx[i:i+bs]; opt.zero_grad(); l=lossf(m(Xtr[b]),Ytr[b]); l.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        ps=[]
        for i in range(0,len(Xte),16384): ps.append(torch.sigmoid(m(Xte[i:i+16384])).cpu().numpy())
        p=np.concatenate(ps)
    return roc_auc_score(Yte,p), average_precision_score(Yte,p)

print(f'표본 {N:,} train {tr.sum():,} test {te.sum():,}  device {dev}')
print('H  | 양성% |  A(ego GRU)     | B(+edge GRU)    | ΔAUC')
for Hh in [6,15,30]:
    Y=(T2N<=Hh).astype(np.int64)
    aA,pA=train_eval(Xa,Y,'A'); aB,pB=train_eval(Xb,Y,'B')
    print('%2d | %4.1f  | AUC %.4f AP%.3f | AUC %.4f AP%.3f | %+.4f'%(Hh,Y.mean()*100,aA,pA,aB,pB,aB-aA))
