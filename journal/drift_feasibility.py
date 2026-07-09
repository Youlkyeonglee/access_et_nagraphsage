"""
DRIFT 상호작용 실현성 — heading 기반 car-following 결합 (맵 불필요).
leader = ego 진행방향 앞(longitudinal>0) + 같은 차로(lateral 작음) + 최근접.
측정: ego 가속도(부호있음)가 leader gap·상대속도에 반응하나?
"""
import glob, numpy as np, pandas as pd
CSVS=sorted(glob.glob('/home/oem/data/TII_data/DRIFT_csv/*.csv'))[:8]

acc=[]; gap=[]; rel=[]; latw=None
# 먼저 최근접 leader 거리 분포로 스케일 파악
for csv in CSVS:
    df=pd.read_csv(csv); df=df[df.category.isin(['stop','lane_change','normal_driving'])].copy()
    for fr,g in df.groupby('frame'):
        if len(g)<2: continue
        p=np.column_stack([g.position_x.to_numpy(),g.position_z.to_numpy()])
        dx=g.direction_x.to_numpy(); dz=g.direction_z.to_numpy()
        sp=g.speed.to_numpy(); ac=g.acceleration.to_numpy()
        nrm=np.hypot(dx,dz)
        for k in range(len(g)):
            if sp[k]<1.0 or nrm[k]<1e-3: continue
            hx,hz=dx[k]/nrm[k],dz[k]/nrm[k]
            rel_p=p-p[k]
            lon=rel_p[:,0]*hx+rel_p[:,1]*hz          # 진행방향(앞+)
            lat=np.abs(-rel_p[:,0]*hz+rel_p[:,1]*hx)  # 횡방향 거리
            ahead=(lon>0.5)&(lat<8.0)&(np.arange(len(g))!=k)  # 앞 + 같은차로(lat<8)
            if ahead.any():
                j=np.where(ahead)[0][np.argmin(lon[ahead])]
                acc.append(ac[k]); gap.append(lon[j]); rel.append(sp[j]-sp[k])

acc=np.array(acc); gap=np.array(gap); rel=np.array(rel)
# 이상치 제거(추적 노이즈)
m=(np.abs(acc)<200)&(gap<np.percentile(gap,99))
acc,gap,rel=acc[m],gap[m],rel[m]
print(f'DRIFT: leader 있는 샘플 {len(acc):,}   gap 중앙값 {np.median(gap):.1f}')
print('=== car-following 결합 (가속도 부호있음) ===')
print(f'  corr(ego accel, leader gap)       = {np.corrcoef(acc,gap)[0,1]:+.3f}  (양수=gap클수록 가속=추종)')
print(f'  corr(ego accel, rel_speed(앞-나))  = {np.corrcoef(acc,rel)[0,1]:+.3f}  (양수=앞차빠르면 가속=추종)')
q=np.percentile(gap,30)
near=gap<q
print(f'  근접(gap<{q:.0f}) 평균 accel {acc[near].mean():+.2f}  vs 먼곳 {acc[~near].mean():+.2f}  (근접서 감속?)')
print(f'  [참고] 공업탑 car-following corr는 +0.06~0.14, 근접감속 약함이었음')
