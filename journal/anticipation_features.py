"""
저널(가+바) 피처 파이프라인 — 전체 Gongeoptap 처리 후 npz 저장.
per-sample 저장:
  ego_seq   : [N, W+1, F_ego]  ego 극좌표 시계열 (시계열 모델용)
  edge_seq  : [N, W+1, F_edge] 인접/같은 링 gap 시계열
  t2n       : [N]  다음 lane_id 전이까지 프레임 수 (라벨은 H로 파생)
  frame     : [N]  시간분할용
  file_idx  : [N]
누수 방지: 입력에 lane_id 미사용(전이는 라벨 생성에만).
"""
import glob, json, numpy as np, pandas as pd
from scipy.spatial import cKDTree

IMG_W, IMG_H = 1600., 1200.
ANNO = '/home/oem/TNA_research/road_data/lane_annotations.json'
CSVS = sorted(glob.glob('/home/oem/data/TII_data/Gongeoptap/*.csv'))
OUT  = '/home/oem/TNA_research/journal/anticipation_data.npz'
W = 9                     # 과거창 길이(프레임): 시계열 W+1=10
SAME, ADJ_LO, ADJ_HI = 2.0, 2.0, 7.0
ARC_MAX = 50.0; BIG = ARC_MAX
F_EGO, F_EDGE = 6, 8

def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi

def fit_H(src, dst):
    A=[]
    for (x,y),(u,v) in zip(src,dst):
        A.append([-x,-y,-1,0,0,0,x*u,y*u,u]); A.append([0,0,0,-x,-y,-1,x*v,y*v,v])
    _,_,Vt=np.linalg.svd(np.array(A), full_matrices=False); return Vt[-1].reshape(3,3)

def apply_H(H,p):
    q=np.column_stack([p,np.ones(len(p))])@H.T; return q[:,:2]/q[:,2:3]

def map_center(df):
    s=df[df.speed>0.5]
    src=np.column_stack([s.bbox_cx.to_numpy()/IMG_W, s.bbox_cy.to_numpy()/IMG_H]).astype(float)
    dst=np.column_stack([s.position_x.to_numpy(), s.position_z.to_numpy()]).astype(float)
    H=fit_H(src,dst)
    anno=json.load(open(ANNO))
    allc=np.vstack([apply_H(H,np.array(L['centerline'],float)) for L in anno['lanes']])
    return allc.mean(0)

def ego_feat(spd,acc,rh,an,i):
    drho=np.diff(rh[i-W:i+1]); dth=wrap(np.diff(an[i-W:i+1]))
    # 각 시점 피처: [speed,accel,rho,drho,dth*rho(접선),|drho|]
    seq=[]
    for k in range(i-W, i+1):
        dr = rh[k]-rh[k-1] if k>0 else 0.0
        dt = wrap(an[k]-an[k-1]) if k>0 else 0.0
        seq.append([spd[k],acc[k],rh[k],dr,dt*rh[k],abs(dr)])
    return np.array(seq,dtype=np.float32)   # [W+1,6]

E,WE,ED,T2N,FR,FI,YD=[],[],[],[],[],[],[]   # WE: world 좌표 ego, YD: 탐지 라벨
for fidx,csv in enumerate(CSVS):
    df=pd.read_csv(csv); df=df[df['category'].isin(['stop','lane_change','normal_driving'])].copy()
    df=df.sort_values(['frame','object_id']).reset_index(drop=True)
    C=map_center(df)
    d=np.column_stack([df.position_x.to_numpy(),df.position_z.to_numpy()])-C
    df['rho']=np.hypot(d[:,0],d[:,1]); df['th']=np.arctan2(d[:,1],d[:,0])
    CIRC=np.sign(df.sort_values(['object_id','frame']).groupby('object_id')['th']
                 .apply(lambda s: wrap(s.diff()).sum()).sum()) or 1.0
    frame_grp={f:g for f,g in df.groupby('frame')}
    for oid,g in df.groupby('object_id'):
        g=g.sort_values('frame')
        lane=g['lane_id'].to_numpy().astype(str); spd=g['speed'].to_numpy(); acc=g['acceleration'].to_numpy()
        rh=g['rho'].to_numpy(); an=g['th'].to_numpy(); cat=g['category'].to_numpy()
        fr=g['frame'].to_numpy(); px=g['position_x'].to_numpy(); pz=g['position_z'].to_numpy(); n=len(g)
        dxr=g['direction_x'].to_numpy(); dzr=g['direction_z'].to_numpy()   # world 방향(절제용)
        trans=np.zeros(n,bool)
        for i in range(1,n):
            if lane[i]!='' and lane[i-1]!='' and lane[i]!=lane[i-1]: trans[i]=True
        tidx=np.where(trans)[0]
        for i in range(W, n-1):
            if cat[i]=='stop' or spd[i]<1.0: continue
            nxt=tidx[tidx>i]; t2n=int(nxt[0]-i) if len(nxt) else 9999
            eseq=ego_feat(spd,acc,rh,an,i)
            # edge 시계열: 각 시점 프레임의 인접/같은 링 gap
            edseq=np.zeros((W+1,F_EDGE),dtype=np.float32)
            for wi,k in enumerate(range(i-W,i+1)):
                fk=fr[k]; fg=frame_grp.get(fk)
                f=[BIG,0,BIG, BIG,0,BIG, BIG,0][:F_EDGE]
                f=[BIG,0.,BIG,BIG,0.,BIG,BIG,0.]  # sameLead,relS,sameFoll,outAhead,relS,outBehind,inAhead,inBehind
                if fg is not None and len(fg)>1:
                    r2=fg['rho'].to_numpy(); a2=fg['th'].to_numpy(); s2=fg['speed'].to_numpy()
                    dr=r2-rh[k]; arc=wrap(a2-an[k])*rh[k]; fwd=CIRC*arc; sel=np.abs(arc)<=ARC_MAX
                    same=sel&(np.abs(dr)<SAME); out=sel&(dr>=ADJ_LO)&(dr<=ADJ_HI); inn=sel&(dr<=-ADJ_LO)&(dr>=-ADJ_HI)
                    def nr(mask,ah):
                        mm=mask&((fwd>0) if ah else (fwd<0))
                        if mm.any():
                            j=np.where(mm)[0][np.argmin(np.abs(fwd[mm]))]; return abs(fwd[j]),float(s2[j]-spd[k])
                        return BIG,0.
                    f[0],f[1]=nr(same,1); f[2],_=nr(same,0)
                    f[3],f[4]=nr(out,1);  f[5],_=nr(out,0)
                    f[6],_=nr(inn,1);     f[7],_=nr(inn,0)
                edseq[wi]=f
            # world 좌표 ego 시계열 [W+1,6] = [pos_x,pos_z,speed,dir_x,dir_z,accel] (절제용)
            wseq=np.stack([px[i-W:i+1],pz[i-W:i+1],spd[i-W:i+1],
                           dxr[i-W:i+1],dzr[i-W:i+1],acc[i-W:i+1]],axis=1).astype(np.float32)
            E.append(eseq); WE.append(wseq); ED.append(edseq); T2N.append(t2n); FR.append(fr[i]); FI.append(fidx)
            YD.append(int(cat[i]=='lane_change'))

E=np.array(E,dtype=np.float32); ED=np.array(ED,dtype=np.float32)
T2N=np.array(T2N,dtype=np.int32); FR=np.array(FR,dtype=np.int64); FI=np.array(FI,dtype=np.int16)
YD=np.array(YD,dtype=np.int8); WE=np.array(WE,dtype=np.float32)
np.savez_compressed(OUT, ego_seq=E, world_seq=WE, edge_seq=ED, t2n=T2N, frame=FR, file_idx=FI, y_det=YD)
print(f'저장: {OUT}')
print(f'표본 {len(T2N):,}  ego_seq {E.shape}  edge_seq {ED.shape}')
for Hh in [3,6,10,15,20,30]:
    print(f'  H={Hh:2d} 양성률 {(T2N<=Hh).mean()*100:.1f}%')
