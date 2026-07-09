"""
A 검정: DRIFT 현재상태 분류에서 ego-only(A) vs +edge(B). 5-seed GRU, 시간분할.
3-class(stop/LC/normal) → macro AUC(OvR) + LC(소수) AUC. paired t.
공업탑 탐지에선 edge≈0이었음 → DRIFT서 edge가 사는지 판정.
"""
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from scipy import stats

dev='cuda' if torch.cuda.is_available() else 'cpu'
d=np.load('/home/oem/TNA_research/journal/drift_data.npz')
E=np.nan_to_num(d['ego_seq'].astype(np.float32),nan=0.,posinf=0.,neginf=0.)
ED=np.nan_to_num(d['edge_seq'].astype(np.float32),nan=0.,posinf=0.,neginf=0.)
# 극단 이상치 클리핑(추적 노이즈: accel 등)
E=np.clip(E,-500,500); ED=np.clip(ED,-500,500)
Y=d['y'].astype(np.int64); FR=d['frame']; FI=d['file_idx']; N=len(Y)
SC=d['scene'] if 'scene' in d else np.zeros(N,np.int8)
SEEDS=[0,1,2,3,4]

tr=np.zeros(N,bool)
for f in np.unique(FI):
    m=FI==f;
    if m.sum()<20: tr|=m; continue
    thr=np.quantile(FR[m],0.7); tr|=m&(FR<=thr)
te=~tr
def std(X):
    mu=X[tr].reshape(-1,X.shape[-1]).mean(0); sd=X[tr].reshape(-1,X.shape[-1]).std(0)+1e-6
    return ((X-mu)/sd).astype(np.float32)
E=std(E); ED=std(ED)
Xa=torch.tensor(E); Xb=torch.tensor(np.concatenate([E,ED],-1))
Yt=torch.tensor(Y)

class G(nn.Module):
    def __init__(s,fin,h=64,nc=3):
        super().__init__(); s.g=nn.GRU(fin,h,batch_first=True)
        s.f=nn.Sequential(nn.Linear(h,h),nn.ReLU(),nn.Dropout(0.2),nn.Linear(h,nc))
    def forward(s,x): o,_=s.g(x); return s.f(o[:,-1])

cw=torch.tensor([ (Y[tr]!=c).sum()/max(1,(Y[tr]==c).sum()) for c in range(3)],dtype=torch.float32)
def run(X,seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr=X[tr].to(dev); Ytr=Yt[tr].to(dev); Xte=X[te].to(dev)
    m=G(X.shape[-1]).to(dev); opt=torch.optim.Adam(m.parameters(),1e-3,weight_decay=1e-5)
    lf=nn.CrossEntropyLoss(weight=cw.to(dev)); bs=8192
    for ep in range(20):
        m.train(); idx=torch.randperm(len(Xtr),device=dev)
        for i in range(0,len(Xtr),bs):
            b=idx[i:i+bs]; opt.zero_grad(); lf(m(Xtr[b]),Ytr[b]).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        pr=np.concatenate([torch.softmax(m(Xte[i:i+16384]),-1).cpu().numpy() for i in range(0,len(Xte),16384)])
    yte=Y[te]
    macro=roc_auc_score(yte,pr,multi_class='ovr',average='macro')
    lc=roc_auc_score((yte==1).astype(int),pr[:,1])   # 차선변경(소수) AUC
    # 도로(scene)별 LaneChange AUC
    sc_te=SC[te]; per={}
    for s in np.unique(sc_te):
        msk=sc_te==s
        if (yte[msk]==1).sum()>20 and (yte[msk]!=1).sum()>20:
            per[int(s)]=roc_auc_score((yte[msk]==1).astype(int),pr[msk,1])
    return macro,lc,per

u,c=np.unique(SC,return_counts=True)
print(f'표본 {N:,} train {tr.sum():,} test {te.sum():,} dev {dev}  도로수 {len(u)} 분포 {dict(zip(u.tolist(),c.tolist()))}')
mA=[];lA=[];mB=[];lB=[]; perA={};perB={}
for s in SEEDS:
    a=run(Xa,s); b=run(Xb,s); mA.append(a[0]);lA.append(a[1]);mB.append(b[0]);lB.append(b[1])
    for k,v in a[2].items(): perA.setdefault(k,[]).append(v)
    for k,v in b[2].items(): perB.setdefault(k,[]).append(v)
    print(f'  seed{s}: A(macro {a[0]:.4f}, LC {a[1]:.4f})  B(macro {b[0]:.4f}, LC {b[1]:.4f})')
mA,lA,mB,lB=map(lambda x:np.array(x),[mA,lA,mB,lB])
def rep(a,b,name):
    t,p=stats.ttest_rel(b,a)
    print(f'{name}: A {a.mean():.4f}±{a.std(ddof=1):.4f}  B {b.mean():.4f}±{b.std(ddof=1):.4f}  Δ {(b-a).mean():+.4f}  p={p:.4f}  {"유의미" if p<0.05 and (b-a).mean()>0 else "무의/음"}')
print('=== A(ego) vs B(+edge) [전 도로 합쳐서] ===')
rep(mA,mB,'macro-AUC')
rep(lA,lB,'LaneChange-AUC')
print('=== 도로별 LaneChange-AUC (edge 일반성) ===')
names=['A','B','C','D','E','I']
for k in sorted(perA):
    a=np.mean(perA[k]); b=np.mean(perB.get(k,[a]))
    nm=names[k] if k<len(names) else str(k)
    print(f'  도로 {nm}: A {a:.4f} → B(+edge) {b:.4f}  Δ {b-a:+.4f}  {"↑" if b-a>0.005 else "≈"}')
