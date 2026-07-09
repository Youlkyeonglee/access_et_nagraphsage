"""
진입-양보 실현성 검정 (로터리 상호작용의 핵심 장면).
 진입 차량 = 반경 ρ가 감소(중심으로 합류) 하는 차량
 순환 충돌차 = 진입차의 합류 각도 부근에서 이미 순환 중(더 안쪽 반경, 접선 이동)인 차량
 측정: 진입차가 순환 충돌차의 gap/접근에 따라 감속(양보)하는가?
 비교군: car-following(같은 링 앞차) 결합과 대비.
"""
import glob, numpy as np, pandas as pd, sys
sys.path.insert(0,'.'); from map_features import build_map_world, polar
def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi
CSVS=sorted(glob.glob('/home/oem/data/TII_data/Gongeoptap/*.csv'))[:5]
W=5; ARC=25.0

y_dsp=[]; y_gap=[]; y_present=[]   # 진입차: 다음Δspeed, 순환차 gap, 순환차 존재여부
n_enter=0; n_enter_conflict=0

for csv in CSVS:
    df=pd.read_csv(csv); df=df[df.category.isin(['stop','lane_change','normal_driving'])].copy()
    df=df.sort_values(['object_id','frame'])
    df['dsp_next']=df.groupby('object_id')['speed'].diff().shift(-1)
    Hm,C,lanes=build_map_world(df)
    pos=np.column_stack([df.position_x.to_numpy(),df.position_z.to_numpy()])
    r,t=polar(pos,C); df['rho']=r; df['th']=t
    # 반경 추세(최근 W프레임): 진입 판정
    df['drho']=df.groupby('object_id')['rho'].diff()
    df['drho_w']=df.groupby('object_id')['drho'].rolling(W,min_periods=2).mean().reset_index(0,drop=True)
    CIRC=np.sign(df.sort_values(['object_id','frame']).groupby('object_id')['th'].apply(lambda s: wrap(s.diff()).sum()).sum()) or 1.0
    df=df.sort_values(['frame','object_id'])
    for fr,g in df.groupby('frame'):
        if len(g)<2: continue
        r=g.rho.to_numpy(); a=g.th.to_numpy(); sp=g.speed.to_numpy()
        dn=g.dsp_next.to_numpy(); dw=g.drho_w.to_numpy()
        for k in range(len(g)):
            if sp[k]<1.0 or not np.isfinite(dn[k]) or not np.isfinite(dw[k]): continue
            if dw[k] > -0.3: continue              # 진입차: 반경 감소(중심으로) 아니면 skip
            n_enter+=1
            # 순환 충돌차: 더 안쪽(dr<0, 이미 순환) + 합류각 부근(arc<한도) + 앞(fwd>0)
            dr=r-r[k]; arc=wrap(a-a[k])*r[k]; fwd=CIRC*arc
            conf=(dr< -1.0)&(dr> -18.0)&(np.abs(arc)<ARC)&(fwd>0)&(np.arange(len(g))!=k)
            if conf.any():
                j=np.where(conf)[0][np.argmin(fwd[conf])]
                y_dsp.append(dn[k]); y_gap.append(fwd[j]); y_present.append(1); n_enter_conflict+=1
            else:
                y_dsp.append(dn[k]); y_gap.append(np.nan); y_present.append(0)

y_dsp=np.array(y_dsp); y_gap=np.array(y_gap); y_present=np.array(y_present)
print(f'진입 샘플 {n_enter:,}  그중 순환충돌차 있음 {n_enter_conflict:,} ({100*n_enter_conflict/max(1,n_enter):.1f}%)')
has=y_present==1
print('=== 진입-양보 결합 ===')
print(f'  순환차 있을때 다음Δspeed {y_dsp[has].mean():+.3f}  vs  없을때 {y_dsp[~has].mean():+.3f}  (있을때 더 감속=양보?)')
if has.sum()>100:
    gg=y_gap[has]; dd=y_dsp[has]
    print(f'  corr(다음Δspeed, 순환차 gap) = {np.corrcoef(dd,gg)[0,1]:+.3f}  (양수=gap클수록 가속=양보안함)')
    close=gg<8
    print(f'  순환차 근접(gap<8m) Δspeed {dd[close].mean():+.3f}  vs  먼곳 {dd[~close].mean():+.3f}')
