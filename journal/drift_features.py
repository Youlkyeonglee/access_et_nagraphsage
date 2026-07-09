"""
DRIFT 피처 파이프라인 (A: 현재상태 분류, ego-only vs +edge).
 - ego 시계열 [T,6]: world 좌표 [pos_x,pos_z,speed,dir_x,dir_z,accel]
 - edge 시계열 [T,8]: heading 기반 이웃 gap (앞/뒤 차로별) — 맵 불필요
 - 라벨 y: 현재 프레임 category (0=stop,1=lane_change,2=normal)  → 분류
누수 방지: lane_id 입력 미사용.
"""
import glob, numpy as np, pandas as pd
import re
# 6개 도로종류(A~E,I)에서 각 5파일씩 균등 샘플링 → 전 도로 커버
_all=sorted(glob.glob('/home/oem/data/TII_data/DRIFT_csv/*.csv'))
_by={}
for f in _all:
    pre=re.search(r'DRIFT_([A-Z]+)_',f).group(1); _by.setdefault(pre,[]).append(f)
CSVS=[]; SCENE={}
for pre in sorted(_by):
    for f in _by[pre][:5]:
        CSVS.append(f); SCENE[f]=pre
print('도로종류별 파일:',{k:min(5,len(v)) for k,v in sorted(_by.items())})
OUT='/home/oem/TNA_research/journal/drift_data.npz'
W=9; LANE=8.0; LON_MAX=60.0; ADJ_LAT=(4.0,14.0)   # 같은차로 lat<4, 인접차로 4~14
LABEL={'stop':0,'lane_change':1,'normal_driving':2}

def edge_at(p, dx, dz, sp, k):
    """프레임 k 시점 ego에 대한 8D edge: [앞gap,앞relS, 뒤gap, 좌앞gap,좌relS, 우앞gap,우relS, 이웃수]"""
    n=len(p); nrm=np.hypot(dx[k],dz[k])
    BIG=LON_MAX
    f=[BIG,0.,BIG, BIG,0., BIG,0., 0.]
    if nrm<1e-3: return f
    hx,hz=dx[k]/nrm,dz[k]/nrm
    rel=p-p[k]; lon=rel[:,0]*hx+rel[:,1]*hz; lat=-rel[:,0]*hz+rel[:,1]*hx
    idx=np.arange(n)!=k
    def nearest(mask):
        m=mask&idx
        if m.any():
            j=np.where(m)[0][np.argmin(np.abs(lon[m]))]; return abs(lon[j]), float(sp[j]-sp[k])
        return BIG,0.
    same_ahead=(np.abs(lat)<ADJ_LAT[0])&(lon>0)&(lon<LON_MAX)
    same_behind=(np.abs(lat)<ADJ_LAT[0])&(lon<0)&(lon>-LON_MAX)
    left_ahead=(lat<=-ADJ_LAT[0])&(lat>-ADJ_LAT[1])&(lon>0)&(lon<LON_MAX)
    right_ahead=(lat>=ADJ_LAT[0])&(lat<ADJ_LAT[1])&(lon>0)&(lon<LON_MAX)
    f[0],f[1]=nearest(same_ahead); f[2],_=nearest(same_behind)
    f[3],f[4]=nearest(left_ahead); f[5],f[6]=nearest(right_ahead)
    f[7]=int(((np.abs(lat)<ADJ_LAT[1])&(np.abs(lon)<LON_MAX)&idx).sum())
    return f

E,ED,Y,FR,FI,SC=[],[],[],[],[],[]
_scene_ids=sorted(set(SCENE.values())); _sid={s:i for i,s in enumerate(_scene_ids)}
for fi,csv in enumerate(CSVS):
    _scene=_sid[SCENE[csv]]
    df=pd.read_csv(csv); df=df[df.category.isin(LABEL)].copy()
    df=df.sort_values(['frame','object_id']).reset_index(drop=True)
    # ── 프레임-차량별 edge 8D를 1회만 계산해 캐싱 (겹치는 window 중복 제거) ──
    edge_cache={}   # (frame, oid) -> edge8
    for fr,g in df.groupby('frame'):
        P=g[['position_x','position_z']].to_numpy()
        DX=g.direction_x.to_numpy(); DZ=g.direction_z.to_numpy()
        SP=g.speed.to_numpy(); OID=g.object_id.to_numpy()
        for kk in range(len(g)):
            edge_cache[(fr,OID[kk])]=np.asarray(edge_at(P,DX,DZ,SP,kk),np.float32)
    for oid,g in df.groupby('object_id'):
        g=g.sort_values('frame'); n=len(g)
        if n<W+1: continue
        fr=g.frame.to_numpy(); cat=g.category.to_numpy()
        px=g.position_x.to_numpy(); pz=g.position_z.to_numpy(); sp=g.speed.to_numpy()
        dx=g.direction_x.to_numpy(); dz=g.direction_z.to_numpy(); ac=g.acceleration.to_numpy()
        for i in range(W,n):
            if sp[i]<1.0: continue
            eseq=np.stack([px[i-W:i+1],pz[i-W:i+1],sp[i-W:i+1],
                           dx[i-W:i+1],dz[i-W:i+1],ac[i-W:i+1]],axis=1).astype(np.float32)
            edseq=np.stack([edge_cache[(fk,oid)] for fk in fr[i-W:i+1]],axis=0)
            E.append(eseq); ED.append(edseq); Y.append(LABEL[cat[i]]); FR.append(fr[i]); FI.append(fi); SC.append(_scene)

E=np.array(E,np.float32); ED=np.array(ED,np.float32); Y=np.array(Y,np.int8); FR=np.array(FR,np.int64); FI=np.array(FI,np.int16); SC=np.array(SC,np.int8)
np.savez_compressed(OUT, ego_seq=E, edge_seq=ED, y=Y, frame=FR, file_idx=FI, scene=SC)
print(f'저장 {OUT}  표본 {len(Y):,}  ego {E.shape} edge {ED.shape}')
u,c=np.unique(Y,return_counts=True)
print('클래스 분포:', {['stop','LC','normal'][k]:int(v) for k,v in zip(u,c)})
