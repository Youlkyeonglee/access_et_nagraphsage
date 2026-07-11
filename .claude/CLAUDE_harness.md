# 가상환경
conda activate tna_research


# 데이터 경로: /home/oem/yklee/data

# userEmail
yklee00815@gmail.com

# currentDate
2026-06-28

---

# 연구 목표 — ET-NAGraphSAGE

## 핵심 목표: State_Acc 향상 (단일 최우선 지표)
- **Baseline**: NAGraphSAGE 94.54% Acc (Gongeoptap, World 좌표, EdgeDim3)
- **목표**: State_Acc 유의미한 향상 (NAGraphSAGE 대비 통계적으로 유의미한 개선)
- **방향**: NAGraphSAGE의 per-frame 한계를 **Edge+Node 시계열 인코딩**으로 극복

## 핵심 Novelty (2가지만)
1. **Edge feature 시계열 인코딩** ← 진짜 novelty. 기존 시공간 GNN 어디에도 없음.
   - 이웃 차량과의 kinematic 관계(상대 속도·가속도·방향·거리)의 T프레임 변화를 학습
   - 기존 STGCN/DCRNN은 edge를 단순 거리 또는 고정 가중치로만 처리
2. **Node feature 시계열 인코딩** + NAGraphSAGE 공간 집계 통합
   - 과거 T프레임 운동학 시퀀스 → temporal encoder → edge-aware 메시지 패싱

## 제거된 것들 (이유)
- ~~Dual-Head (미래 K스텝 상태 예측)~~: State_Acc 목표와 직접 연관 없음. multi-task loss가 분류를 방해할 수 있음. 논문 주장 희석.
- ~~Mamba = 핵심 contribution~~: T=10~15에서 GRU와 실질 차이 불명확. Ablation에서 이기는 인코더를 채택.
- ~~고정 97% 목표~~: 논문은 "유의미한 향상"으로 주장. Gongeoptap은 구조적 난이도가 있음.

## 아키텍처 (3단계)
```
STAGE 1: TEMPORAL ENCODER (신규)
  Node: 차량 i의 T프레임 노드 피처 시퀀스 → temporal encoder → h_i^temp
  Edge: 이웃 j와의 T프레임 엣지 피처 시퀀스 → temporal encoder → e_ij^temp
  인코더 타입: GRU / LSTM / Mamba 비교 → ablation으로 결정

STAGE 2: SPATIAL (NAGraphSAGE, 기존 구조 유지)
  Edge-Aware Message Passing (edge 독립 MLP 인코딩 유지)
  Add Aggregator + Node Update MLP

STAGE 3: OUTPUT (단일 분류 헤드)
  현재 상태 분류만: Stop / Lane Change / Normal Driving
```

## 손실 함수 (단순화)
```
L_total = L_CE + λ_KL · L_KL
L_CE  = CrossEntropy(ŷ_t, y_t)        ← 현재 상태 분류
L_KL  = KL(p_attention ∥ Uniform)     ← attention 균등화 (기존 유지)
λ_KL ∈ [0.1, 0.5]
```

## Ablation 설계 (4그룹)
- **A: temporal encoder 타입** — GRU / LSTM / Mamba / Transformer (T=10 고정)
- **B: 시퀀스 길이 T** — 1(baseline) / 5 / 10 / 15 프레임
- **C: 시계열 대상** — node only / edge only / **node+edge (제안)** ← 핵심 ablation
- **D: NAGraphSAGE 층수 / KL λ** — 기존 논문 범위 유지

## 베이스라인 수치 (논문 Table VI, XI)
| 모델 | Acc (%) | Macro-F1 (%) |
|---|---|---|
| GraphSAGE | 92.74 | 90.69 |
| NAGraphSAGE (World, EdgeDim3 avg) | **94.07±0.28** | **92.56±0.37** |
| NAGraphSAGE (World, best, Table XI) | **94.54±1.03** | — |

## 저장소 경로
- **T-NAGraphSAGE 저장소 (현재)**: `/home/oem/TNA_research/`
- NAGraphSAGE 원본 (수정 금지): `/home/oem/graph_vehicle_v1/`
- M2MambaV2 (별도): `/home/oem/access2/`
- 데이터: `/home/oem/data/` (Gongeoptap, DRIFT)
- 계획서: `/home/oem/TNA_research/docs/TNA_research_plan.html`
- 논문 PDF: `/home/oem/TNA_research/docs/IEEE_TII_25_9364_20260401_rev1_0_NAGraphSAGE.pdf`
- Mamba 기초: `/home/oem/access2/review/20260515_process.html` (sec-basics)

## 코드 참조
- NAGraphSAGE 모델: `/home/oem/graph_vehicle_v1/proposed/models/model_node_final.py`
- NAGraphSAGE Conv: `/home/oem/graph_vehicle_v1/proposed/models/convlayer_final.py`
- 학습 스크립트: `/home/oem/graph_vehicle_v1/proposed/train_proposed_node_final.py`
- Config: `/home/oem/graph_vehicle_v1/proposed/config_proposed_world.yaml`
