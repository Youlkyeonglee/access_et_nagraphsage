"""
전체 데이터 지평 스윕 (GBDT, derisk 단계).
npz 로드 → 시계열을 요약피처로 평탄화(마지막프레임 + 통계) → 지평 H별 A vs B AUC.
시계열 GNN 이전에 "전체 데이터서도 지평추세 유지되나" 확인용.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

d=np.load('/home/oem/TNA_research/journal/anticipation_data.npz')
E=d['ego_seq']; ED=d['edge_seq']; T2N=d['t2n']; FR=d['frame']; FI=d['file_idx']
N=len(T2N); print(f'표본 {N:,}  ego_seq {E.shape}  edge_seq {ED.shape}')

def flat(seq):
    # 마지막 프레임 + 창 평균/표준편차/(마지막-처음)
    last=seq[:,-1,:]; mean=seq.mean(1); std=seq.std(1); delta=seq[:,-1,:]-seq[:,0,:]
    return np.concatenate([last,mean,std,delta],axis=1)

Xe=flat(E); Xn=flat(ED)
# 시간분할: 파일별 프레임 70% 경계
tr=np.zeros(N,bool)
for f in np.unique(FI):
    m=FI==f; thr=np.quantile(FR[m],0.7); tr|=m&(FR<=thr)
te=~tr
print(f'train {tr.sum():,}  test {te.sum():,}')

def auc(X,Y):
    clf=HistGradientBoostingClassifier(max_iter=300,learning_rate=0.08,max_depth=4,
        l2_regularization=1.0,random_state=0).fit(X[tr],Y[tr])
    p=clf.predict_proba(X[te])[:,1]
    return roc_auc_score(Y[te],p), average_precision_score(Y[te],p)

print('\nH  | 양성% |  A(ego)         | B(+edge)        | ΔAUC')
for Hh in [3,6,10,15,20,30]:
    Y=(T2N<=Hh).astype(int)
    if Y[tr].sum()<50 or Y[te].sum()<50: continue
    aA,pA=auc(Xe,Y); aB,pB=auc(np.c_[Xe,Xn],Y)
    print('%2d | %4.1f  | AUC %.4f AP%.3f | AUC %.4f AP%.3f | %+.4f'%(Hh,Y.mean()*100,aA,pA,aB,pB,aB-aA))
