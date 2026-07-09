"""
[신규] 공업탑 gap-acceptance 엣지 피처 — 이웃↔이웃 간격(hole) + 구멍 개폐속도.
====================================================================================
기존 코드 불변. 기존 엣지(나↔이웃 arc 거리)와 대비할 새 엣지를 별도 npz로 저장한다.

핵심 novelty(사용자 제안):
  - 기존 엣지 = 나 ↔ 이웃 거리  → 두 노드 빼기라 (부분) 중복
  - 새   엣지 = 이웃 ↔ 이웃 간격(hole) + 구멍 개폐속도(앞이웃 v − 뒤이웃 v)
      · hole    = 목표 링(안/바깥)에서 앞이웃~뒤이웃 arc 간격 = 내가 낄 구멍 크기
      · closing = 앞이웃 속도 − 뒤이웃 속도 (>0 열림, <0 닫힘) ← 나-이웃 엣지에 없는 신호
      · half_a/half_b = ego에서 앞/뒤 이웃까지 arc (구멍 안 내 위치, 참고)

저장(per-sample):
  world_seq [N,W+1,6]  ego world 시계열 [pos_x,pos_z,speed,dir_x,dir_z,accel] (공정 비교용)
  edge_base [N,W+1,8]  기존 스타일 나↔이웃 gap (재현: anticipation_features.py 정의)
  edge_gap  [N,W+1,6]  새 이웃↔이웃 gap [in_hole,in_close, out_hole,out_close, in_halfA, out_halfA]
  y_cur     [N]        현재상태 3-class (0=stop,1=lane_change,2=normal)
  t2n       [N]        다음 lane_id 전이까지 프레임 수 (anticipation 라벨 파생용)
  frame,file_idx       시간분할용
누수 방지: lane_id는 라벨(전이)에만, 입력 피처엔 미사용.
"""
import glob, json, numpy as np, pandas as pd

IMG_W, IMG_H = 1600., 1200.
ANNO = '/home/oem/TNA_research/road_data/lane_annotations.json'
CSVS = sorted(glob.glob('/home/oem/data/TII_data/Gongeoptap/*.csv'))
OUT  = '/home/oem/TNA_research/journal/gap_edge_data.npz'
W = 9                                   # 과거창: 시계열 W+1=10
SAME, ADJ_LO, ADJ_HI = 2.0, 2.0, 7.0    # 링밴드 폭(기존과 동일)
ARC_MAX = 50.0; BIG = ARC_MAX
LABEL = {'stop':0,'lane_change':1,'normal_driving':2}

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

WE,EB,EG,YC,T2N,FR,FI=[],[],[],[],[],[],[]
for fidx,csv in enumerate(CSVS):
    df=pd.read_csv(csv); df=df[df['category'].isin(LABEL)].copy()
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
        dxr=g['direction_x'].to_numpy(); dzr=g['direction_z'].to_numpy()
        trans=np.zeros(n,bool)
        for i in range(1,n):
            if lane[i]!='' and lane[i-1]!='' and lane[i]!=lane[i-1]: trans[i]=True
        tidx=np.where(trans)[0]
        for i in range(W, n):
            if spd[i]<1.0: continue
            nxt=tidx[tidx>i]; t2n=int(nxt[0]-i) if len(nxt) else 9999
            eb=np.zeros((W+1,8),dtype=np.float32)   # 기존 스타일 나↔이웃
            eg=np.zeros((W+1,6),dtype=np.float32)   # 새 이웃↔이웃 hole
            for wi,k in enumerate(range(i-W,i+1)):
                fg=frame_grp.get(fr[k])
                b=[BIG,0.,BIG,BIG,0.,BIG,BIG,0.]    # sameLead,relS,sameFoll,outAh,relS,outBe,inAh,inBe
                gpair=[BIG,0.,BIG,0.,BIG,BIG]       # in_hole,in_close,out_hole,out_close,in_halfA,out_halfA
                if fg is not None and len(fg)>1:
                    r2=fg['rho'].to_numpy(); a2=fg['th'].to_numpy(); s2=fg['speed'].to_numpy()
                    dr=r2-rh[k]; arc=wrap(a2-an[k])*rh[k]; fwd=CIRC*arc; sel=np.abs(arc)<=ARC_MAX
                    same=sel&(np.abs(dr)<SAME); out=sel&(dr>=ADJ_LO)&(dr<=ADJ_HI); inn=sel&(dr<=-ADJ_LO)&(dr>=-ADJ_HI)
                    def nr(mask,ah):
                        mm=mask&((fwd>0) if ah else (fwd<0))
                        if mm.any():
                            j=np.where(mm)[0][np.argmin(np.abs(fwd[mm]))]
                            return abs(float(fwd[j])), float(s2[j]-spd[k]), float(s2[j])
                        return BIG,0.,np.nan
                    # 기존 나↔이웃 (재현)
                    b[0],b[1],_=nr(same,1); b[2],_,_=nr(same,0)
                    b[3],b[4],_=nr(out,1);  b[5],_,_=nr(out,0)
                    b[6],_,_   =nr(inn,1);  b[7],_,_=nr(inn,0)
                    # 새 이웃↔이웃 hole (목표 링: inner / outer)
                    def hole(mask):
                        aA,_,vA=nr(mask,1)   # 앞이웃 arc, 속도
                        aB,_,vB=nr(mask,0)   # 뒤이웃 arc, 속도
                        if aA>=BIG or aB>=BIG or np.isnan(vA) or np.isnan(vB):
                            return BIG,0.,BIG           # 한쪽이라도 없으면 구멍 미정의
                        h=aA+aB                          # 이웃~이웃 간격(내가 낄 구멍)
                        close=vA-vB                       # 앞이웃v − 뒤이웃v (>0 열림)
                        return float(h),float(close),float(aA)
                    gpair[0],gpair[1],gpair[4]=hole(inn)  # inner: hole, closing, ego→앞이웃 arc
                    gpair[2],gpair[3],gpair[5]=hole(out)  # outer
                eb[wi]=b; eg[wi]=gpair
            wseq=np.stack([px[i-W:i+1],pz[i-W:i+1],spd[i-W:i+1],
                           dxr[i-W:i+1],dzr[i-W:i+1],acc[i-W:i+1]],axis=1).astype(np.float32)
            WE.append(wseq); EB.append(eb); EG.append(eg)
            YC.append(LABEL[cat[i]]); T2N.append(t2n); FR.append(fr[i]); FI.append(fidx)

WE=np.array(WE,np.float32); EB=np.array(EB,np.float32); EG=np.array(EG,np.float32)
YC=np.array(YC,np.int8); T2N=np.array(T2N,np.int32); FR=np.array(FR,np.int64); FI=np.array(FI,np.int16)
np.savez_compressed(OUT, world_seq=WE, edge_base=EB, edge_gap=EG, y_cur=YC, t2n=T2N, frame=FR, file_idx=FI)
print(f'저장 {OUT}')
print(f'표본 {len(YC):,}  world {WE.shape}  edge_base {EB.shape}  edge_gap {EG.shape}')
u,c=np.unique(YC,return_counts=True)
print('현재상태 분포:', {['stop','LC','normal'][k]:int(v) for k,v in zip(u,c)})
# 구멍 정의된 비율(저밀도 진단)
in_def=(EG[:,-1,0]<BIG).mean(); out_def=(EG[:,-1,2]<BIG).mean()
print(f'마지막프레임 구멍 정의율: inner {in_def*100:.1f}%  outer {out_def*100:.1f}%  (저밀도면 낮음)')
for Hh in [3,6,10,15]:
    print(f'  anticipation H={Hh:2d} 양성률 {(T2N<=Hh).mean()*100:.1f}%')
