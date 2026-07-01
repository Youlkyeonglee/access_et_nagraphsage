# ET-NAGraphSAGE

Edge-Temporal Neighbor-Aware GraphSAGE — 차량 움직임 상태 분류 (Stop / Lane Change / Normal Driving)

NAGraphSAGE(per-frame Edge-Aware GNN)의 후속 연구로, **노드·엣지 피처를 모두 시계열로 인코딩**하고
2-hop 공간 집계 + Supervised Contrastive Loss를 결합한다.

---

## 데이터셋

### Gongeoptap (주 데이터셋)

원형교차로(roundabout) 환경 차량 궤적 데이터. 본 실험의 주 결과는 이 데이터 기준.

| 항목 | 값 |
|---|---|
| 경로 | `/home/oem/data/TII_data/Gongeoptap/*.csv` |
| CSV 파일 수 | **11개** (`received_file_20240822_*.csv`) |
| 총 행 수 | **996,122** (헤더 제외) |
| 수집일 | 2024-08-22 (10:15 ~ 11:50, 약 5분 간격 세션) |
| 좌표계 | World 좌표 (position_x/z) |

**파일별 행수**
```
101524: 118,132   102025:  81,561   103025:  99,022   104526:  91,141
105526:  87,641   110026:  14,729   111027:  97,065   112527:  88,704
113528: 102,581   114028: 112,735   115028: 102,822
```

**클래스 분포 (불균형)**
| 클래스 | 라벨 | 개수 | 비율 |
|---|---|---|---|
| Stop | 0 | 401,549 | 40.3% |
| Lane Change | 1 | 205,212 | **20.6%** (희귀 클래스, bottleneck) |
| Normal Driving | 2 | 389,361 | 39.1% |

**CSV 컬럼**
```
frame, object_id, bbox_cx, bbox_cy, bbox_w, bbox_h,
position_x, position_y, position_z, speed, acceleration,
direction_x, direction_y, direction_z, class, category, lane_id
```
- 노드 피처 (6D): `position_x, position_z, speed, direction_x, direction_z, acceleration`
- 엣지 피처 (5D): `rel_speed, rel_accel, rel_dir_x, rel_dir_z, distance` (이웃 쌍에서 계산)
- 라벨: `category` → {stop:0, lane_change:1, normal_driving:2}

### DRIFT (일반화 검증용, 보조)

- 경로: `/home/oem/data/TII_data/Drift/{A,B,C,D,E,I}/`
- 6개 도로 환경. 데이터로더는 지원하나 주 실험은 Gongeoptap 기준.

### 데이터 분할 (Temporal Split)

각 CSV 내 프레임을 **시간 순서로** 분할 (무작위 아님, 미래 누수 방지):
- train 0.70 / val 0.15 / test 0.15
- T프레임 슬라이딩 윈도우, 프레임 간격(gap) 검증으로 결측 구간 배제

샘플 수: T=1 기준 ~699K, T=10 시퀀스 구성 시 ~620K (-11%)

> ⚠️ 데이터는 저장소에 포함되지 않음(`.gitignore`). 각 서버에 동일 경로로 존재한다고 가정.
> 서버별 경로가 다르면 `configs/*.yaml`의 `data.data_dir`를 로컬에서 수정 (커밋 금지).

---

## 구조

```
train.py              # 메인 학습 (GRU/LSTM/Mamba, 2-hop)
train_supcon.py       # + Supervised Contrastive Loss
train_feat.py         # feature 보강 실험용 (node8D/edge7D)
models/
  et_nagraphsage.py         # 2-hop ET-SAGE (C1 gate/C2 msg/C3 β agg)
  et_nagraphsage_supcon.py  # + projection head
  temporal_encoder.py       # GRU/LSTM/Mamba + temporal attention
modules/
  data_manager.py           # 시계열 그래프 데이터로더 (2-hop)
  data_manager_feat.py      # feature 보강판
configs/              # 실험별 yaml
scripts/plot_results.py     # 결과 시각화
docs/ET-NAGraphSAGE.html    # 연구 계획서/실험 기록 (rich)
RUN_QUEUE.md          # 다중 서버 실험 협업 보드
results/              # 서버별 결과 로그
```

## 환경

```bash
conda activate tna_research
```
- PyTorch 2.7.0 (CUDA 12.8), scipy(cKDTree), pandas, matplotlib
- Mamba 실험은 `mamba_ssm` 필요 (컴파일) — **로컬 서버에서만 실행**

## 실행 예시

```bash
# 2-hop + SupCon (현재 최고 계열)
python train_supcon.py --config configs/et_nagraphsage_2hop_supcon_ep500.yaml

# baseline (T=1, temporal 없음)
python train.py --config configs/et_nagraphsage.yaml --T 1
```

## 다중 서버 협업

`RUN_QUEUE.md` 참조. 요약:
- **로컬 서버** (원본): Mamba 계열 + 문서/보드 관리
- **학교 서버** (원격): 비-Mamba 실험 (Mamba 금지)
- 결과는 `results/local.md` / `results/school.md`에 분리 기록 (충돌 방지)

### 용어 (혼동 방지)
- **Ablation 그룹**: A(인코더) · B(시퀀스길이) · C(시계열대상) · D(공간구조) · E(이웃정책) — 글자는 여기에만
- **실험 ID**: `그룹-변형` (예: `C-node`, `E-count`) — 어느 Ablation인지 이름으로 식별
- **서버**: 로컬 / 학교 (A/B 글자 안 씀)

## 현재 최고 성능 (150 epoch 기준, 500ep 검증중)

| 모델 | State_Acc | Stop | LaneChange | Normal |
|---|---|---|---|---|
| NAGraphSAGE (baseline, 미발간) | 94.54% | — | — | — |
| 2-hop K1=10 | **94.33%** | 99.87 | 81.26 | 95.00 |
| 2-hop + SupCon | 94.32% | 99.86 | 81.15 | 95.05 |
