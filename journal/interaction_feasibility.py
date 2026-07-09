"""
상호작용 실현성 진단 — spatial-temporal+edge 방향이 성립하는지 데이터로 확인.
 (1) car-following 결합: ego 가속도가 앞차(same-ring leader) gap·접근속도에 반응하나?
 (2) 근접 이벤트 수: 양보/상충 라벨 재료가 될 근접 쌍이 충분한가?
극좌표(로터리 중심) 기준 링/arc 사용.
"""
import glob, numpy as np, pandas as pd
import sys; sys.path.insert(0,'/home/oem/TNA_research/journal')
from map_features import build_map_world, polar

CSVS=sorted(glob.glob('/home/oem/data/TII_data/Gongeoptap/*.csv'))[:5]
SAME=2.0; ARC_MAX=30.0; CLOSE=5.0
def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi

acc_all=[]; gap_all=[]; rel_all=[]      # leader 있는 샘플: ego accel, leader arc-gap, rel_speed
n_leader=0; n_total=0; n_close_frame=0
pair_close=set()                          # (file,i,j) 근접 경험한 쌍
n_pairs=set()

for fi,csv in enumerate(CSVS):
    df=pd.read_csv(csv); df=df[df['category'].isin(['stop','lane_change','normal_driving'])].copy()
    df=df.sort_values(['frame','object_id']).reset_index(drop=True)
    Hm,C,lanes=build_map_world(df)
    pos=np.column_stack([df.position_x.to_numpy(),df.position_z.to_numpy()])
    rho,th=polar(pos,C); df['rho']=rho; df['th']=th
    CIRC=np.sign(df.sort_values(['object_id','frame']).groupby('object_id')['th']
                 .apply(lambda s: wrap(s.diff()).sum()).sum()) or 1.0
    for fr,g in df.groupby('frame'):
        if len(g)<2: continue
        oid=g['object_id'].to_numpy(); r=g['rho'].to_numpy(); a=g['th'].to_numpy()
        sp=g['speed'].to_numpy(); ac=g['acceleration'].to_numpy()
        p=np.column_stack([g.position_x.to_numpy(),g.position_z.to_numpy()])
        for k in range(len(g)):
            if sp[k]<1.0: continue
            n_total+=1
            dr=r-r[k]; arc=wrap(a-a[k])*r[k]; fwd=CIRC*arc
            # same-ring leader: 같은 링, 앞(fwd>0), arc<한도, 가장 가까운
            same=(np.abs(dr)<SAME)&(fwd>0)&(fwd<ARC_MAX)&(np.arange(len(g))!=k)
            if same.any():
                j=np.where(same)[0][np.argmin(fwd[same])]
                n_leader+=1; acc_all.append(ac[k]); gap_all.append(fwd[j]); rel_all.append(sp[j]-sp[k])
            # 근접(유클리드) 이벤트
            d=np.sqrt(((p-p[k])**2).sum(1)); d[k]=1e9
            jm=np.argmin(d)
            if d[jm]<CLOSE:
                n_close_frame+=1
                pair_close.add((fi,min(oid[k],oid[jm]),max(oid[k],oid[jm])))
            n_pairs.add((fi,min(oid[k],oid[jm]),max(oid[k],oid[jm])))

acc=np.array(acc_all); gap=np.array(gap_all); rel=np.array(rel_all)
print(f'파일 {len(CSVS)}개  이동샘플 {n_total:,}  leader 있는 샘플 {n_leader:,} ({100*n_leader/max(1,n_total):.1f}%)')
print('=== (1) car-following 결합 (leader 존재 시) ===')
print(f'  corr(ego accel, leader gap)      = {np.corrcoef(acc,gap)[0,1]:+.3f}   (양수=gap 클수록 가속 → 추종)')
print(f'  corr(ego accel, rel_speed(앞-나)) = {np.corrcoef(acc,rel)[0,1]:+.3f}   (양수=앞차 빠를수록 가속)')
# 근접(gap<8) 시 감속 경향
near=gap<8; far=gap>=8
print(f'  근접(gap<8m) 평균 accel {acc[near].mean():+.3f} vs 먼(≥8m) {acc[far].mean():+.3f}  (근접서 더 감속?)')
print('=== (2) 근접 이벤트(양보/상충 재료) ===')
print(f'  근접(<{CLOSE}m) 발생 프레임 {n_close_frame:,}')
print(f'  근접 경험한 고유 쌍 {len(pair_close):,} / 전체 최근접 쌍 {len(n_pairs):,} ({100*len(pair_close)/max(1,len(n_pairs)):.1f}%)')
