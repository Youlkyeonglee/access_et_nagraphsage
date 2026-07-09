"""
HD맵 → world 변환 + 로터리 극좌표(반경 ρ, 각도 θ) 유틸.
- 호모그래피(정규화이미지→world)를 CSV bbox↔position에서 fit
- annotation centerline을 world로 변환 → 로터리 중심 C, 링 반경대 확인
"""
import json, numpy as np, pandas as pd

IMG_W, IMG_H = 1600., 1200.
ANNO = '/home/oem/TNA_research/road_data/lane_annotations.json'

def fit_homography(bbox_norm, world):
    A=[]
    for (x,y),(u,v) in zip(bbox_norm, world):
        A.append([-x,-y,-1,0,0,0,x*u,y*u,u]); A.append([0,0,0,-x,-y,-1,x*v,y*v,v])
    _,_,Vt=np.linalg.svd(np.array(A), full_matrices=False); return Vt[-1].reshape(3,3)

def apply_H(H, pts):
    p=np.column_stack([pts, np.ones(len(pts))])@H.T
    return p[:,:2]/p[:,2:3]

def build_map_world(df):
    """df(한 파일)로 호모그래피 fit → annotation을 world로 변환, 중심·링정보 반환."""
    s=df[df.speed>0.5]
    src=np.column_stack([s.bbox_cx.to_numpy()/IMG_W, s.bbox_cy.to_numpy()/IMG_H]).astype(float)
    dst=np.column_stack([s.position_x.to_numpy(), s.position_z.to_numpy()]).astype(float)
    H=fit_homography(src,dst)
    anno=json.load(open(ANNO))
    lanes=[]
    allc=[]
    for L in anno['lanes']:
        cl=np.array(L['centerline'],float)         # 정규화 이미지좌표
        clw=apply_H(H, cl)                          # world
        lanes.append({'label':L['label'],'cw':clw})
        allc.append(clw)
    allc=np.vstack(allc)
    C=allc.mean(0)                                  # 로터리 중심(근사)
    # 각 lane 평균 반경
    for l in lanes:
        l['radius']=np.linalg.norm(l['cw']-C,axis=1).mean()
    return H,C,lanes

def polar(pos, C):
    d=pos-C
    return np.hypot(d[...,0],d[...,1]), np.arctan2(d[...,1],d[...,0])

if __name__=='__main__':
    import glob
    csv=sorted(glob.glob('/home/oem/data/TII_data/Gongeoptap/*.csv'))[0]
    df=pd.read_csv(csv)
    H,C,lanes=build_map_world(df)
    print('로터리 중심 C(world):', np.round(C,1))
    rads=sorted(set(round(l['radius'],1) for l in lanes))
    print(f'lane 개수 {len(lanes)}, 반경대(world m) 범위 [{min(rads):.1f},{max(rads):.1f}]')
    print('반경 예시:', rads[:12])
    # 세그먼트-링 라벨과 반경 정합 확인 (예: 3-1..3-4가 반경 순증하나)
    seg3=sorted([(l['label'],round(l['radius'],1)) for l in lanes if l['label'].startswith('3-')])
    print('세그먼트3 (라벨,반경):', seg3)
    # sanity: 차량 극좌표 + lane_change 프레임의 반경변화가 큰가
    df=df.sort_values(['object_id','frame'])
    pos=np.column_stack([df.position_x,df.position_z])
    rho,th=polar(pos,C)
    df['rho']=rho
    df['drho']=df.groupby('object_id')['rho'].diff().abs()
    m=df['category']=='lane_change'; n=df['category']=='normal_driving'
    lc=df.loc[m,'drho'].mean(); nd=df.loc[n,'drho'].mean()
    print('|Δρ| 평균  lane_change=%.3f  normal=%.3f  (클수록 반경이동=차선변경 신호)'%(lc,nd))
