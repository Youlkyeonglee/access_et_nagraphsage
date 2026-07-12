# comparison/ — 외부 문헌 baseline 재구현

`docs/TSEM_journal_design.html` §구조 "관련연구 (Related Work)"에서 Table I 공정비교용으로
지정한 코드-공개 모델을 실제로 가져와 TSEM-SAGE와 같은 데이터·평가로 재학습하기 위한 폴더.

## 원칙

1. **원본 코드는 그대로 보존한다.** 각 모델은 `comparison/<Model>/`에 공식 GitHub 저장소를
   그대로 clone해서 라이선스·출처를 보존한다 (수정하지 않음).
2. **어댑터는 별도 폴더에 새로 작성한다.** `comparison/<model>_tsem/`에 우리 데이터 파이프라인
   (`modules/data_manager_tsem.py`)과 평가 코드(`modules/tsem_eval.py`)를 그대로 재사용하도록
   변환하는 코드만 추가한다 — 원본 저장소 파일은 건드리지 않는다.
3. **원본 아키텍처의 핵심 메커니즘은 최대한 보존**하고, 우리 과제(3클래스 미래 state 분류)에
   맞춰 꼭 필요한 부분만 바꾼다 — 회귀 디코더(좌표 예측) → 분류 헤드(Linear→3) 교체가 공통 패턴.
4. **기준선(target)은 10차-2(TSEM-SAGE 최종 채택 모델)다.** 2026-07-10 기준 공식 채택된 제안
   모델은 semantic+position 10D + NAGraphSAGE spatial(`docs/TSEM_journal_design.html` §베이스라인,
   실험명 `10차-2`) — **test acc 81.91% / macro-F1 81.95%**(multi-seed 81.97±0.14%)다. 예전 9차
   (semantic 8D만, 78.05%/77.22%)는 더 이상 최종 제안이 아니므로 비교 문서에 "TSEM-SAGE 78.05%"라고
   쓰지 않는다 — 아래 표 및 각 모델 결과는 전부 10차-2를 기준선으로 삼는다.
5. **평가는 100% 동일한 코드로 한다.** `modules/tsem_eval.py::evaluate_tsem` +
   `modules/data_manager_tsem.py::build_tsem_dataloaders`를 그대로 import해서 쓴다 —
   TSEM-SAGE와 다른 평가 코드를 새로 만들지 않는다 (accuracy/macro-F1 계산 방식이 달라지면
   비교 자체가 무의미해지기 때문).

## 진행 상황

계획을 바꿔 5개 전부 이 서버(GPU 0-3)에서 직접 순차/병행 실행 중이다 — "타 서버 예정" 문구는
더 이상 유효하지 않음.

| 모델 | 상태 | 위치 |
|---|---|---|
| CRAT-Pred (ICRA 2022) | ✅ **300epoch 완료**(best@78) — test acc 75.02% / F1 72.84% | `comparison/crat-pred/`(원본), `comparison/cratpred_tsem/`(어댑터) |
| HiVT (CVPR 2022) | ✅ **300epoch 완료** — test acc 77.48% / F1 76.35% | `comparison/HiVT/`(원본), `comparison/hivt_tsem/`(어댑터) |
| QCNet (CVPR 2023) | ✅ **300epoch 완료** — test acc 77.97% / F1 76.96% | `comparison/QCNet/`(원본), `comparison/qcnet_tsem/`(어댑터) |
| SIMPL (RA-L 2024) | ⏳ 구현+검증 완료, 300epoch 실행 대기 | `comparison/SIMPL/`(원본), `comparison/simpl_tsem/`(어댑터) |
| Forecast-MAE (ICCV 2023) | ⏳ 구현+검증 완료, 300epoch 실행 대기(멀티 GPU 재벤치마크 필요 — 아래 참조) | `comparison/forecast-mae/`(원본), `comparison/forecastmae_tsem/`(어댑터) |
| STGCN (Yu et al., IJCAI 2018) | ⏳ 구현+검증 완료(2026-07-11), 300epoch 실행 대기 | `comparison/STGCN/`(원본, hazdzz/STGCN), `comparison/stgcn_tsem/`(어댑터) |
| DCRNN (Li et al., ICLR 2018) | ⏳ 구현+검증 완료(2026-07-11), 300epoch 실행 대기 | `comparison/DCRNN/`(원본, chnsh/DCRNN_PyTorch), `comparison/dcrnn_tsem/`(어댑터) |
| TGN (Rossi et al., ICML 2020 WS) | ⏳ 구현+검증 완료(2026-07-11), 300epoch 실행 대기 | `comparison/TGN/`(원본, twitter-research/tgn), `comparison/tgn_tsem/`(어댑터) |
| LSTM (원본 repo 없음 — 아키텍처 대조군) | ⏳ 구현+검증 완료(2026-07-11), 300epoch 실행 대기 | `comparison/lstm_tsem/`(어댑터만) |
| Transformer (원본 repo 없음 — 아키텍처 대조군) | ⏳ 구현+검증 완료(2026-07-11), 300epoch 실행 대기 | `comparison/transformer_tsem/`(어댑터만) |

전체 진행상황·설계 배경은 `docs/TSEM_journal_design.html` §"🆚 비교 연구" 탭에도 동일하게 기록돼 있다.

**2026-07-11 추가 6종(STGCN/DCRNN/TGN/LSTM/Transformer)**: 문헌 baseline 5개(HiVT/QCNet/CRAT-Pred/
SIMPL/Forecast-MAE)와 별도로, 그래프-시계열 계열의 대표 아키텍처(STGCN/DCRNN/TGN)와 순수 아키텍처
대조군(LSTM/Transformer, 원본 repo 없음)을 추가했다. 상세 설계·검증은 하단
"§STGCN/DCRNN/TGN/LSTM/Transformer 어댑터" 절 참조. **STGCN/DCRNN/TGN은 위 5개와 같은
`comparison/<Model>/`(원본 clone, 미수정) + `comparison/<model>_tsem/`(어댑터) 2단 구조**를 그대로
따랐고, **LSTM/Transformer는 원본 GitHub repo가 존재하지 않는 일반 아키텍처**라 어댑터 폴더만
만들었다(§원칙 2 예외 — 표준 `nn.LSTM`/`nn.TransformerEncoder`이므로 "보존할 원본 구현"이 없음).

## 5개 어댑터 — 설계 차이 한눈에 비교

| 비교 축 | HiVT | QCNet | CRAT-Pred | SIMPL | Forecast-MAE |
|---|---|---|---|---|---|
| 원본 연도·venue | CVPR 2022 | CVPR 2023 | ICRA 2022 | IEEE RA-L 2024 | ICCV 2023 |
| 핵심 novelty | hierarchical local→global attention | query-centric 상대표현 | Crystal Graph Conv(edge feature 명시) | Symmetric Fusion Transformer | 마스킹-복원 자기지도 사전학습 |
| **시간 인코더** | Transformer(BOS/CLS/padding 토큰) | Attention+Fourier(시간축도 attention) | **LSTM**(단일 레이어) | **1D CNN**(ResNet+FPN) | Conv1d+Transformer(원래 NATTEN) |
| **공간/상호작용 인코더** | AAEncoder+GlobalInteractor(2단) | 시간·공간 attention 매 layer 반복 | CGConv 2-layer+MHA 1-layer | SftLayer(edge+src+tgt 결합 memory) | Transformer blocks(에이전트 토큰) |
| **회전 불변성 방식** | 노드별 **명시적 회전행렬** | **상대각도**(atan2) | scene 1회 정렬만 | **상대각도** cos/sin쌍 | scene 1회 정렬(CRAT-Pred와 동일) |
| **대체 불가피했던 의존성** | `torch_geometric` | `torch_geometric`+`torch_cluster` | `torch_geometric`(CGConv만) | 없음(원래 순수 PyTorch) | `natten`(**대체 불가능**, CUDA 커널)+`timm` |
| 재구현 방식 | scatter 기반 수식 재구현 | scatter 기반 수식 재구현 | scatter 기반(가장 단순) | 내장 attention 위임 | **아키텍처 자체 대체** |
| 재구현 검증 | 4/4 PASS | 6/6 PASS | 3/3 PASS | 5/5 PASS(실버그 발견·수정) | 5/5 PASS |
| 파라미터(스모크) | 512,835 | 532,899 | **68,291**(최경량) | 488,035 | 305,731(+사전학습 411,092) |
| 속도(스모크) | 237초/ep(최느림) | 136초/ep | **57.6초/ep**(최속) | 261초/ep | 74.3초/ep(finetune) |
| 학습 프로토콜 | 단일단계 | 단일단계 | 단일단계 | 단일단계 | **2단계**(사전학습→미세조정) |

## 5개 어댑터 — 입력값(모델 입력 규격) 정리

각 어댑터가 실제로 신경망에 넣는 per-timestep 노드 피처와 edge/관계 피처의 차원·구성이 서로
다르다 — "같은 원시 데이터를 쓰지만 모델별로 뽑아 쓰는 채널이 다르다"는 점을 명시해둔다
(전부 `modules/data_manager_tsem.py`가 만드는 동일한 6D raw 노드 피처
`[pos_x, pos_z, speed, dir_x, dir_z, accel]`에서 출발 — 각 `model.py`의 `_build_inputs`/
`_build_graph`에서 확인).

| 모델 | 노드 입력(차원) | 구성 | edge/관계 입력(차원) | 구성 |
|---|---|---|---|---|
| HiVT | **6D** | Δx,Δz(변위)+speed+dir_x,dir_z+accel — raw 6D 전 채널 사용 | 2D | rel_pos(위치차, 원본 그대로) |
| QCNet | 4D | motion_norm(변위 크기)+motion_angle(변위-heading 각도)+speed+accel | 4D(시간축)+3D(공간축) | r_t: dist,angle,rel_head,Δt / r_a2a: dist,angle,rel_head |
| CRAT-Pred | **3D**(최소) | Δx,Δy(변위)+valid flag만 — speed/accel 등 추가 채널 의도적으로 미사용 | 2D | 상대 center(target−source, 원본 그대로) |
| SIMPL | **3D**(최소) | Δx,Δy(변위)+valid flag만 — CRAT-Pred와 동일 이유로 최소 유지 | 5D | RPE: cos_a1,sin_a1,cos_a2,sin_a2,dist |
| Forecast-MAE | 4D | Δx,Δy(변위)+speed_diff(속도 프레임간 차분)+valid flag — **2026-07-11 수정, 원본과 동일 정의** | 없음 | 명시적 edge 없음 — scene-level attention이 암묵적으로 상호작용 학습 |
| STGCN | **6D** | Δx,Δz(scene-align 변위)+speed+dir_x,dir_z+accel — raw 6D 전 채널(HiVT와 동일 근거) | 없음(노드 edge feature 아님) | N×N 인접행렬(Gaussian distance kernel, σ=10m) — 그래프 conv 자체가 관계 정보 |
| DCRNN | **6D** | 위와 동일(HiVT와 동일 근거) | 없음(노드 edge feature 아님) | N×N 양방향 확산 전이행렬 P_f,P_b(Gaussian distance kernel 기반 인접행렬에서 유도) |
| TGN | **6D** | 위와 동일(HiVT와 동일 근거) | 5D | edge_seqs 그대로: rel_speed,rel_accel,rel_dir_x,rel_dir_z,distance — message function 입력 |
| LSTM(원본 repo 없음) | **6D** | Δx,Δz(scene-align 변위)+speed+dir_x,dir_z+accel — HiVT와 동일 정보량 | 없음(공간 상호작용 모듈 자체가 없음) | — |
| Transformer(원본 repo 없음) | **6D** | 위와 동일 | 없음(공간 상호작용 모듈 자체가 없음) | — |

**해석(신규 6종)**:
- STGCN/DCRNN/TGN 셋 다 6D 전 채널을 사용 — 원 논문들은 각각 도로 센서 하나당 스칼라(교통 속도)
  하나만 입력으로 쓰지만, 여기선 HiVT와 동일하게 "정보량 손실 없이 준다"는 원칙을 적용했다.
- STGCN·DCRNN은 **명시적 edge feature가 없다** — 대신 그래프 자체(인접행렬/확산 전이행렬)가
  관계 정보를 담는 구조라, HiVT/CRAT-Pred류의 "edge feature 벡터"라는 개념 자체가 원 논문에 없다.
  인접행렬은 두 아키텍처 모두 Gaussian distance kernel(`exp(-dist²/σ²)`)로 구성 — 원 논문(도로망
  거리 기반 가중 인접행렬)에 더 충실한 선택이라 판단해 이진(0/1) 대신 채택.
- TGN만 유일하게 `edge_seqs` 5D(다른 5개 baseline과 동일한 원시 edge feature)를 그대로
  message function 입력으로 쓴다 — 원 논문 자체가 "상호작용(edge event)마다 메시지를 만든다"는
  설계라 이 edge feature 개념과 정확히 대응된다.
- LSTM/Transformer는 **공간 상호작용 모듈 자체가 없는 게 설계 의도**라 edge 입력 개념이 없다
  (아래 "LSTM/Transformer 어댑터 — 설계 메모" 참조) — 나머지 9개 baseline 전부가 갖는 학습된
  공간 상호작용 메커니즘(attention/GNN/diffusion conv)을 뺀 순수 시간축-only 대조군.

**해석(기존 5종)**:
- HiVT만 원시 6D를 전부 사용 — TSEM-SAGE와 "같은 정보량"을 주기 위한 의도적 확장(README 위쪽
  HiVT 절 참조).
- QCNet은 원시 채널을 그대로 넣지 않고 **가공된(엔지니어링된) 피처**(변위 크기·각도)로 재표현 —
  원본 논문 설계를 그대로 따른 것.
- CRAT-Pred·SIMPL은 의도적으로 **최소 입력**(변위+valid flag 3D)만 사용 — "최소한의 정보로
  얼마나 되는가"가 두 모델을 고른 이유 중 하나라 채널을 늘리지 않았다(README 각 절 참조).
- **(정정, 2026-07-11) Forecast-MAE도 나머지 4개와 마찬가지로 변위 기반이다.** 이전 버전
  문서에는 "Forecast-MAE만 절대 위치를 쓴다"고 잘못 적혀 있었으나, 원 논문
  (`av2_extractor.py:155-174`)도 프레임간 변위+속도차분을 쓴다는 걸 확인해 adapter를 그에 맞게
  수정했다(아래 "원 논문 실제 입력 vs TSEM-adapted 입력 대조" 표 참조). 즉 5개 baseline 전부
  변위 기반 입력이라는 게 맞는 서술이다.
- Forecast-MAE만 **명시적 edge 입력이 없다** — 나머지 4개는 전부 상대위치/각도를 별도 edge
  피처로 계산해 넣지만, Forecast-MAE는 에이전트 토큰들을 한 시퀀스로 놓고 scene-level
  self-attention에 상호작용 학습을 전적으로 맡긴다(원본 설계 그대로).

## 5개 어댑터 — 원 논문(pristine 코드) 실제 입력 vs TSEM-adapted 입력 대조 (2026-07-11, 원본 코드 직접 조사)

`comparison/HiVT/`, `comparison/QCNet/`, `comparison/crat-pred/`, `comparison/SIMPL/`,
`comparison/forecast-mae/` 5개 pristine clone의 전처리·모델 forward 코드를 직접 읽어 "원 논문이
실제로 뭘 입력으로 쓰는지"를 확인하고, 우리 adapter(`comparison/<model>_tsem/model.py`)가 거기서
무엇을 뺐는지/바꿨는지 전부 대조했다. 목적: 위 "입력값 정리" 표는 **우리 adapter끼리의 비교**였고,
이 표는 **각 adapter가 원 논문 대비 무엇을 잃었는지**를 보여준다.

| 모델 | 원 논문 agent 노드 입력 | 원 논문 맵/lane 입력 | 원 논문 edge/관계 입력 | 원 논문 출력 |
|---|---|---|---|---|
| HiVT | 2D: Δx,Δy(변위, actor별 헤딩 재정렬) | **있음** — lane centerline vector(2D) + intersection/turn_direction/traffic_control 3종 카테고리 embedding, radius 50m | AA/AL 2D(상대위치) | multi-modal(6모드) Laplace 궤적, 30스텝 |
| QCNet | 4D: motion_norm,motion_angle,velocity_norm,velocity_angle + agent type(10종) embedding | **있음(가장 정교)** — polygon+point 2-level lane graph, PRED/SUCC/LEFT/RIGHT 등 5종 edge, marking type 17종 | temporal 4D + spatial 3D, map 쪽 4D+edge type 5종 | multi-modal(6모드) propose-refine 2단계, 60스텝 |
| CRAT-Pred | 3D: Δx,Δy,valid | **없음**(원래부터 map-free) | 2D: 상대 center(target−source) | multi-modal(6모드) 궤적, 30스텝 |
| SIMPL | AV1: 3D(vel dx,dy+pad) / AV2: 14D(disp+heading+vel+type onehot7+pad) | **있음** — LaneNet, 10-point polyline당 10D(AV1)/16D(AV2) | RPE 5D(cos/sin Δheading, cos/sin bearing, dist) — agent·lane 통합 scene 그래프 | multi-modal(6모드) Bezier 궤적, 30/60스텝 |
| Forecast-MAE | 4D: Δx,Δy,velocity_diff,valid(history) + agent category(4종) embedding | **있음** — LaneEmbeddingLayer(PointNet식, polyline 20점×3D) | 명시적 edge 없음(scene self-attention) | pretrain=변위 복원, finetune=multi-modal 궤적 |

| 모델 | TSEM-adapted 입력 | 뺀 것/바꾼 것 | 왜 |
|---|---|---|---|
| HiVT | 6D(Δx,Δy+speed+dir+accel), edge 2D(원본과 동일) | lane 맵 전체(centerline+intersection/turn/traffic 임베딩), agent type, multi-modal 궤적 디코더 | 이 데이터셋엔 원본 Argoverse 형식의 HD맵 lane-graph가 없다(`road_data/`에 lane_id는 있으나 이 adapter들엔 배선 안 됨 — 별도 작업). 이 과제(3클래스 분류)엔 agent type이 전부 "차량" 하나뿐이라 구분할 게 없고, 회귀용 multi-modal 디코더는 분류 헤드로 교체(공통 패턴) |
| QCNet | 4D(motion_norm,angle,speed,accel), temporal 4D+spatial 3D | 2-level lane graph 전체, agent type(10→1종), propose-refine 2단계 구조 | 맵 없음(동일 사유). propose-refine은 회귀 정밀도를 높이는 구조라 분류 태스크엔 불필요 |
| CRAT-Pred | 3D(Δx,Δy,valid), edge 2D | **없음** — 5개 중 유일하게 원 논문과 거의 1:1 재현 (원래 map-free라 뺄 맵 자체가 없음). 회귀 디코더만 분류 헤드로 교체 | 해당 없음 — 원 논문 설계와 가장 가까운 재현 |
| SIMPL | 3D(Δx,Δy,valid, **AV1 최소 config 채택**), RPE 5D(원본과 동일 정의) | LaneNet 전체(lane 맵), AV2의 heading+type onehot 확장(14D→3D로 축소) | 맵 없음(동일 사유). AV2 확장판 대신 AV1 최소 config를 기준으로 삼음 — "최소 입력으로 얼마나 되는가"라는 SIMPL을 고른 이유와도 부합, RPE(agent-agent 상대각도·거리)는 원본 정의 그대로 보존 |
| Forecast-MAE | 4D(Δx,Δy,speed_diff,valid) — **2026-07-11 수정**: 이전엔 변위 대신 절대위치를 잘못 사용하고 있었음(원 논문도 변위 사용임을 이번 조사로 확인, 아래 참조) | lane 맵 전체(LaneEmbeddingLayer), agent category embedding, future 궤적 복원 브랜치, propose-refine | 맵 없음(동일 사유). future 복원은 우리 데이터에 미래 좌표 정답이 없어서(3클래스 라벨만 존재) 애초에 불가능 |

> **⚠️ 2026-07-11 수정 — Forecast-MAE adapter의 실수 정정.** 기존 문서(위 "5개 어댑터 — 입력값
> 정리" 표, 그리고 이 문서 하단 옛 버전)에는 "Forecast-MAE만 원 논문 설계 그대로 절대위치를
> 쓴다"고 적혀 있었다. 이건 **검증 없이 내린 잘못된 추정**이었다 — 이번에
> `comparison/forecast-mae/` 원본 코드(`src/datasets/av2_extractor.py:155-174`)를 직접
> 읽어보니 **원본도 프레임간 변위(Δx,Δy)와 속도의 프레임간 차분(velocity_diff)을 쓰지,
> 절대위치를 쓰지 않는다.** 즉 우리 `forecastmae_tsem/model.py`가 절대위치를 쓴 건 원 논문을
> 따른 게 아니라 **의도치 않은 설계 이탈**이었다. `comparison/forecastmae_tsem/model.py::_build_inputs`를
> 수정해 원본과 동일하게 프레임간 변위+속도차분 기반으로 바꾸고(`test_model.py` 3/3 재통과
> 확인), 아직 300epoch 본 실행 전이라 실제 결과에는 영향 없음. **이 문서 위쪽의 "5개 어댑터 —
> 입력값 정리" 표와 "왜 이 입력값 차이가 있어도 비교연구로서 정당한가" 절의 "Forecast-MAE만
> 절대위치" 서술은 이 표 기준으로 갱신되어야 한다** (아래에서 갱신함).

### 왜 이 입력값 차이가 있어도 비교연구로서 정당한가 (2026-07-12 기록, 2026-07-11 코드 검증으로 보강)

**먼저 진짜로 통제된 것부터 확인.** 5개 baseline과 TSEM-SAGE는 다음을 전부 동일하게 공유한다:
데이터 소스(`node_seq`/`nbr_node_seqs`, 동일 raw 6D 텐서), train/val/test split, W=10/H=10, 라벨
정의(stop ±2, LC window), loss 레시피(focal+class_weight+kl_weight 0.1+uncertainty weight —
5개 config 전부 동일하게 맞춤, 위 §참고), 학습 예산(300 epoch/patience 50), 평가 코드
(`modules/tsem_eval.py` 공용). **모델 자체와 그 모델이 그 raw 6D를 어떻게 가공해서 받는지만
다르다.** 즉 "다른 데이터를 봤다"가 아니라 "같은 데이터를 다르게 가공해서 봤다"는 뜻이고, 이건
비교연구에서 원래부터 통제 대상이 아니다(각 아키텍처가 원 논문에서 정의한 입력 가공 방식은
그 아키텍처의 일부이지, 실험자가 임의로 끼워 맞춘 변수가 아니다).

**핵심 확인(코드 직접 대조): 입력 차이는 자의적이지 않고, 5개 baseline 전부가 "같은 규칙"을
따른다.** `hivt_tsem/model.py`, `qcnet_tsem/model.py`, `cratpred_tsem/model.py`,
`simpl_tsem/model.py`, `forecastmae_tsem/model.py` 5개 파일의 입력 구성 함수를 전부 대조한 결과,
**예외 없이 전부** ego(자기 차량)의 anchor 시점 위치·헤딩을 원점으로 좌표를 이동시키고 회전시켜
정규화한다(`pos_centered = pos - ego_pos_anchor; rotate by ego_heading`). 이건 궤적예측 분야의
표준 관행인 **translation+rotation invariance(장소·방향 불변성)** 이며, 5개 원 논문이 전부 이
방식을 쓰기 때문에 우리도 그대로 재현한 것이다. **세계좌표(world-absolute position)를 그대로
받는 baseline은 5개 중 단 하나도 없다.**

반대로 **TSEM-SAGE 자신의 최종 우승 config(10차-2)만 이 불변성을 깨고 있다** — raw position
채널 2D(`raw_append='position'`, 세계좌표 그대로)와 semantic 채널의 Δρ(로터리 중심 world 고정
상수 C=(72.86,-13.45)까지의 거리 변화)를 쓴다. 즉 입력 차이의 정체는 "baseline이 정보를
못 받아서 불리하다"가 아니라 **"TSEM-SAGE만 장소-특정(site-specific) 신호를 켜고, baseline
5개는 전부 원 논문 그대로 장소-불변 신호만 쓴다"**는 것이다.

**이게 왜 불공정이 아니라 "이미 실측된 트레이드오프"인가.** 이 프로젝트 자체의 이전 실험(CLAUDE.md
"확정된 핵심 발견" §1, 위치 암기 가설)에서 이미 이렇게 확인했다: raw 절대좌표 채널은 +3.74%p를
주지만 이건 **로터리 단일 지점 암기**이고, 교차 장소(DRIFT) 평가에서 이 position 모델은
LC recall이 80%→0.3%로 붕괴한다. 즉 TSEM-SAGE가 이 벤치마크(공업탑 단일 장소)에서 baseline을
이기는 이유 중 일부는 "NAGraphSAGE 구조가 더 낫다"가 아니라 **"이 장소 하나만 맞히도록 세계좌표를
암기했다"**일 수 있고, baseline들은 원 논문 설계상 애초에 이 암기를 할 수 없게(불변성 유지) 만들어져
있다 — 우리가 baseline에게서 이 정보를 빼앗은 게 아니라, baseline의 아키텍처 자체(rotate2/rotate6,
RPE 등 회전-불변 모듈)가 세계좌표를 받도록 설계되어 있지 않다.

**결론 — 논문 Discussion/Limitation에 다음을 명시할 것:**

> "5개 baseline은 원 논문이 채택한 ego-anchor 중심 좌표 정규화(translation+rotation invariance)를
> 예외 없이 그대로 따랐으며, 세계좌표(world-absolute position)나 고정 랜드마크(로터리 중심) 기준
> 거리 정보를 받지 않는다. 반면 TSEM-SAGE의 최종 config는 이 불변성을 의도적으로 깨고 raw
> 절대좌표 채널과 로터리 중심 기준 거리(Δρ)를 포함한다. 본 프로젝트의 별도 교차-장소 평가에서
> 이 절대좌표 채널이 단일 장소 암기로 인해 타 장소에서 일반화되지 않음을 확인했다(LC recall
> 80%→0.3%). 따라서 본 비교연구에서 TSEM-SAGE가 보이는 우위는 (a) NAGraphSAGE 계열 spatial 구조,
> (b) 위 장소-특정 좌표 신호, (c) 나머지 semantic feature 중 하나 이상에서 기인할 수 있으며 본
> 실험 설계로는 세 요인이 완전히 분리되지 않는다. 이는 baseline의 원 설계(불변성)를 보존하기 위한
> 의도적 선택이었으나, 향후 연구에서는 (i) TSEM-SAGE에서 position 채널뿐 아니라 semantic 8D 중
> 로터리 중심 참조 채널(Δρ·접선)까지 제거한 완전 invariant 버전(6D: v,a,j,ω,d_lat,κ)과의 비교,
> (ii) baseline 중 하나에 세계좌표/Δρ를 이식하는 입력-통제 ablation을 통해 세 요인을 분리하는
> 것이 필요하다."

### semantic 8D 채널별 site-specific 여부 재점검 (2026-07-11, 코드 직접 확인 — 중요 정정)

위에서 "semantic-only(9차, 78.05%)는 위치 채널이 없으니 공정한 비교용"이라고 서술했는데,
`models/tsem_semantic_derivation.py`(`SemanticDerivation.forward`)를 채널 단위로 직접 읽어보니
**이 서술이 부정확했다.** semantic 8채널 `[v, a, j, ω, d_lat, κ, Δρ, 접선]`을 계산 방식별로
분해하면:

| 채널 | 계산 방식 | site-specific? |
|---|---|---|
| v, a, j, ω, κ | 자기 궤적의 시간 미분/차분만 사용(speed, heading 변화율 등) | **아니다** — 순수 운동학, 장소·방향 불변 |
| d_lat | `h0,dx,dz = 윈도우 첫 프레임 기준 heading/위치`, 즉 **자기 궤적 시작점 기준 상대 횡변위** | **아니다** — baseline들의 ego-relative 방식과 동일 성격, 외부 참조 불필요 |
| **Δρ (drho)** | `rho = 거리(현재위치, CENTER)` 의 프레임간 변화, `CENTER=(72.86,-13.45)`는 **로터리 중심 world 고정 상수**(공업탑 전용) | **그렇다** — 이 로터리라는 걸 알아야 계산 가능 |
| **접선(tangent)** | `dtheta·rho` — Δρ와 같은 rho/theta(로터리 중심 기준 극좌표)에서 유도 | **그렇다** — Δρ와 동일한 이유 |

즉 semantic 8D 중 **6채널(v,a,j,ω,d_lat,κ)만 진짜로 baseline과 동급으로 불변**이고, **Δρ·접선
2채널은 raw position 채널과 마찬가지로 site-specific**하다. "semantic-only(9차)는 공정한
비교용"이라는 기존 서술은 틀렸다 — 9차도 Δρ·접선을 포함하므로 baseline과 완전히 동급이 아니다.
**진짜 apples-to-apples 버전은 semantic 6D(v,a,j,ω,d_lat,κ)이고, 이 조합은 지금까지 단독으로
학습·측정된 적이 없다** (10차 계열 ablation 표에는 8D/10D/11D/12D만 있음, `structure-tbl-input`
참조).

**spatial(ETSAGELayer)은 별도 조치 불필요**: spatial 레이어는 raw position을 직접 받지 않고
temporal encoder가 만든 노드 임베딩(=위 입력으로 만들어진 것)만 받으며, edge feature도
`[rel_speed, rel_accel, rel_dir, distance]`(두 차량 간 상대량, `modules/data_manager.py`)라
고정 랜드마크 참조가 없다. 즉 temporal 입력 단에서 position+Δρ+접선만 제거하면 spatial 경로도
자동으로 함께 정리된다 — 이중으로 손볼 곳은 없다.

**정규화(normalization) 부재도 별도로 확인**: `models/tsem_sage.py`/`train_tsem.py` 전체에
z-score/BatchNorm 등 명시적 feature 정규화가 없다 — position(수십 미터), speed(0~15 m/s),
ω(라디안, 매우 작은 값) 등이 스케일 그대로 concat된다. 이건 site-암기 문제와는 **별개의 축**이다
(정규화해도 Δρ·position이 "이 로터리"라는 정보를 담고 있다는 사실 자체는 안 바뀐다 — 스케일만
조정될 뿐 정보량은 그대로) — 학습 안정성 관점에서 정석대로 고치는 게 맞지만, fairness 문제의
해법은 아니다.

**✅ 실측 완료 (2026-07-12) — 결과는 기대와 다르게 나왔다.** 진짜 공정한 비교용 TSEM-SAGE 버전
(**semantic 6D: v,a,j,ω,d_lat,κ만 사용, position·Δρ·접선 전부 제외**)을 실제로 학습했다. 구현은
`models/tsem_semantic_derivation.py`의 `variant='invariant'`(6D 반환) 옵션 + `models/tsem_sage.py`의
`semantic_variant` 배선 + 전용 config `configs/tsem_sage_invariant_6d.yaml`(하네스 원칙 §2). HiVT
300epoch 종료를 자동 감지해 대기 중이던 SIMPL/Forecast-MAE보다 먼저 실행되도록 배치했다. 결과
(`checkpoints/tsem/tsem_sage_invariant_6d/results.json`):

| 모델 | test acc | test macro-F1 |
|---|---|---|
| Persist baseline | 63.17% | 50.55% |
| CRAT-Pred | 75.02% | 72.84% |
| **TSEM-SAGE 6D invariant(구조만, zero landmark)** | **76.21%** | **74.78%** |
| HiVT | 77.48% | 76.35% |
| QCNet | 77.97% | 76.96% |
| TSEM-SAGE semantic 8D(9차, Δρ·접선 포함) | 78.05% | 77.22% |
| TSEM-SAGE 10D(10차-2, site-specific 최종) | 81.91~81.97% | 81.95~82.03% |
| SIMPL / Forecast-MAE | 아직 미실행 | — |

**솔직한 해석 — "구조만으로 baseline을 전부 이긴다"는 희망은 부분적으로만 맞았다.** 6D
invariant(76.21%)는 가장 단순한 baseline인 CRAT-Pred(75.02%)는 이기지만, 더 정교한
HiVT(77.48%)·QCNet(77.97%)에는 1~1.8%p 뒤진다. 이건 NAGraphSAGE 구조에 novelty가 없다는 뜻이
아니라, **구조의 우위가 무조건적이지 않고 더 풍부한 정보를 쓰는 baseline 앞에서는 좁혀지거나
역전된다**는 훨씬 정직하고 미묘한 그림이다. 10D(81.91~81.97%)의 압도적 우위 대부분은 여전히
위치 암기 신호(Δρ+raw position)에서 나온다는 게 이 결과로 더 분명해졌다.

논문 서사 반영 방향: "구조는 약한 baseline 대비 경쟁력이 있지만, 강한 baseline 앞에서는
site-specific 신호(10D)가 진짜 우위를 만든다"는 문장을 Results/Discussion에 명시할 것 — 감추지
않고 그대로 반영하는 게 위치 암기 진단 서사의 정직성과도 일치한다. SIMPL·Forecast-MAE 결과가
나오면 이 표를 갱신한다. 5개 중 하나(CRAT-Pred/HiVT)에 world-absolute position/Δρ를 역으로
이식하는 입력-통제 ablation은 여전히 옵션으로 남아있다.

### hypothesis 1(공간 집계 시점) 검증 (2026-07-12)

6D invariant가 HiVT·QCNet에 뒤지는 이유를 "TSEM-SAGE가 spatial 집계를 anchor 프레임에서 딱
1번만 한다"는 구조적 차이로 설명할 수 있는지 검증하는 두 갈래. 도식은 Obsidian
`Drawing 2026-07-12 spatial-temporal 검증.excalidraw.md` 참조.

**① edge-temporal — ✅ 구현 완료, 학습은 보류(사용자 결정).** 구조 순서(시간 먼저 요약 →
anchor에서 spatial 1회)는 그대로 두고, edge 표현만 바꾼다. `models/tsem_sage.py::TSEMSAGE`에
`edge_temporal: bool = False`(opt-in, 기본값 유지로 하위호환) 추가 — 켜지면 기존
`edge_proj`(Linear, anchor 프레임만) 대신 `_encode_edges()`가 `TemporalEncoder`(GRU, et_nagraphsage.py와
동일 모듈 재사용)로 edge_seqs 전체 W프레임을 인코딩한다. 전용 config
`configs/tsem_sage_invariant_6d_edgetemporal.yaml`(6D invariant와 동일, `edge_temporal: true`만
다름). 실제 데이터로더로 forward+backward 정합성 검증 완료(NaN 없음, params 172,073→175,657
— GRU 엣지 인코더만큼 증가, `edge_temporal=False`는 기존과 파라미터 수 완전히 동일해 하위호환
확인됨). **학습은 미실행** — 필요 시
`python train_tsem.py --config configs/tsem_sage_invariant_6d_edgetemporal.yaml --gpus 0`로 실행.

**② spatial-temporal 교차 — 🔄 학습 진행 중.** 파이프라인 순서 자체를 뒤집는다. `models/tsem_sage.py`에
`TSEMSAGEInterleaved` 신설 — 매 프레임 t마다 먼저 `ETSAGELayer`(spatial)로 공간인지 벡터 s_t를
만들고, 시퀀스 s_1..s_W를 GRU(시간 인코더)에 넣어 anchor 표현을 얻는다(HiVT의 AAEncoder→
TemporalEncoder, QCNet의 시공간 attention 교대 반복과 동일 원리). 전용 config
`configs/tsem_sage_invariant_6d_interleaved.yaml`(6D invariant와 라벨/loss/augmentation 전부
동일, 구조만 다름), 1-epoch 스모크 통과(94.5s/epoch, NaN 없음, params 216,581).

**2026-07-12 정정 ①(epoch 수)**: 최초 실행 시 `num_epochs: 500`(TSEM-SAGE 자체 실험 관례, 10차-2와
동일)으로 설정했었으나, 이 실험은 `comparison/` 5개 baseline과 직접 비교하기 위한 것이라 **300epoch/
patience 50** 프로토콜(HiVT/QCNet/CRAT-Pred/SIMPL/Forecast-MAE/STGCN/DCRNN/TGN/LSTM/Transformer
전부 동일)로 통일해야 한다는 지적을 받아 config를 300으로 고치고 처음부터(30 epoch 지점에서)
재시작했다. 같은 이유로 `configs/tsem_sage_invariant_6d_edgetemporal.yaml`(①, 아직 미실행)도
300으로 맞춰뒀다. 6D invariant(76.21%/74.78%, `configs/tsem_sage_invariant_6d.yaml`)는 이미
227epoch에서 early stop돼 300 미만이라 재실행 불필요.

**2026-07-12 정정 ②(GPU) — 실측 결과 4-GPU가 오히려 느림, 단일 GPU로 원복.** epoch 1 시작 전
GPU 4개(DataParallel, batch 16384→GPU당 4096)로 전환해봤으나, 실측 결과 **123.9s/epoch로 단일
GPU(92~96s/epoch)보다 약 30% 느렸다** — 이 모델이 작아서(216,581 params, comparison/의 최경량
그룹과 비슷한 급) DataParallel의 모델 복제·scatter/gather 오버헤드가 배치 병렬화 이득을 상쇄한다는
"5개 어댑터 — 멀티 GPU 배치 스케일링 실측"의 기존 관찰(작은 모델일수록 이득이 작거나 없음)과
정확히 일치하는 결과다. `--gpus 0`(batch 4096)로 원복해 재시작했다.

완료되면 6D invariant와 직접 비교해 이 표에 추가한다.

## 5개 어댑터 — 멀티 GPU(DataParallel) 배치 스케일링 실측 (2026-07-11)

<blockquote><b>⚠️ 2026-07-11 18:55 — 아래 표의 Forecast-MAE 수치는 신뢰할 수 없음(오염된 측정).</b>
사후 확인 결과, Forecast-MAE 벤치마크 3건(사전학습 4GPU 18:44 완료, 미세조정 4GPU 18:47 완료,
미세조정 2GPU 18:51 완료)이 <b>실제 CRAT-Pred 300epoch 학습(18:38 시작)·HiVT 300epoch
학습(18:46 시작)과 같은 GPU 0-3에서 동시에</b> 돌아가는 도중 측정됐다 — "Forecast-MAE는 멀티
GPU에서 이득이 없다"는 결론은 순수 아키텍처 문제가 아니라 <b>다른 두 실제 학습 작업과의 자원
경쟁 때문일 가능성이 크다.</b> CRAT-Pred의 1GPU 기준값(57.6초)도 다른 작업 없이 측정된 반면
4GPU 값들은 그렇지 않아 비교 기준 자체가 아니다. <b>GPU가 비는 대로 격리된 환경에서
재측정 필요</b> — 그 전까지 이 표의 "결론" 칸은 잠정치로만 취급할 것.</blockquote>

`nn.DataParallel`은 배치를 GPU 수만큼 쪼개 각 GPU에 할당한다 — config의 `batch_size`를 그대로
쓰면 GPU당 실제 배치가 1/n_gpus로 줄어, 작은 모델은 모델 복제·scatter/gather 오버헤드가 실제
연산량을 넘어 GPU를 늘릴수록 오히려 느려질 수 있다. 이를 막기 위해 5개 스크립트 전부 총 배치를
`config batch_size × n_gpus`로 자동 스케일하도록 수정했는데(GPU당 배치는 config 값 유지),
결과는 모델마다 갈렸다(단, 아래 Forecast-MAE 행은 위 경고 참조):

| 모델 | 1GPU(config 배치) | 2GPU(스케일 후) | 4GPU(스케일 전) | 4GPU(스케일 후) | 결론 |
|---|---|---|---|---|---|
| CRAT-Pred | 57.6초/ep(격리 측정) | — | 120초/ep(격리 측정) | **69.7초/ep**(격리 측정) | 스케일링으로 42% 개선, 그래도 1GPU보다 느림 — 이 3개 값은 신뢰 가능 |
| Forecast-MAE(finetune) | 74.3초/ep(격리 측정, 신뢰 가능) | 85.4초/ep⚠️오염 | 79.0초/ep⚠️오염 | — | **재측정 필요** — 현재 결론 잠정치 |
| Forecast-MAE(pretrain) | 81.6초/ep(격리 측정, 신뢰 가능) | — | — | 81.9초/ep⚠️오염 | **재측정 필요** — 현재 결론 잠정치 |

**결론(확정)**: CRAT-Pred는 배치 스케일링이 확실히 도움되지만(그래도 1GPU가 근소하게 더
빠름) — 이 값들은 다른 작업 없이 격리된 상태로 측정해 신뢰 가능하다.

**결론(잠정, 재측정 필요)**: Forecast-MAE는 4GPU/2GPU 모두 1GPU보다 느리게 측정됐지만, 위
경고대로 그 측정이 실제 CRAT-Pred·HiVT 300epoch 학습과 GPU를 나눠 쓰던 도중 이뤄져 신뢰할 수
없다 — `train_forecastmae_tsem.py`에는 일단 `n_gpus>1`일 때 `--gpus 0` 권장 경고를 넣어뒀지만
(강제로 막지는 않음), **GPU가 비는 대로 격리 상태에서 재벤치마크해서 이 경고 문구를 유지할지
빼야 할지 다시 판단해야 한다.** HiVT/QCNet/SIMPL은 아직 멀티 GPU 실측 자체를 안 했다 —
파라미터·연산량이 CRAT-Pred·Forecast-MAE보다 훨씬 커(488K~533K vs 68K~411K) 배치 스케일링의
이득이 더 클 것으로 예상되지만, 이것도 실제로 **다른 작업 없는 격리 환경에서** 확인해야 한다
(지금은 이 두 모델 자체가 GPU를 점유 중이라 서로가 서로의 벤치마크 대상이 될 수 없다).

**공통점**:
- 그래프 구조 동일 — ego+1-hop(K=6) 고정 크기(N=7), 2-hop 미사용
- 맵 제거 — HiVT/QCNet/SIMPL/Forecast-MAE 4개가 원래 HD맵·차선을 쓰지만 우리 데이터엔 맵 없어 전부 제거
- 회귀→분류 헤드 교체 — 5개 전부 다중모달 궤적 회귀 → `Linear(hidden→num_classes)` 분류로 교체
- 데이터로더·loss·평가 100% 공유 — `modules/data_manager_tsem.py`·`modules/tsem_eval.py`·
  `train_tsem.py`의 FocalLoss/class weight/OneCycleLR/early stopping을 5개 전부 그대로 import
- 독립 브루트포스 검증 — 벡터화 코드의 헬퍼를 재사용하지 않는 별도 경로로 5개 전부 대조
  (SIMPL에서 이 방식 덕분에 실제 버그를 발견)
- ego는 항상 그래프의 slot 0 — 배치 offset·마스킹과 무관하게 고정, 분류용 임베딩을 항상 같은 위치에서 추출
- 1-epoch 스모크테스트 — 합성 데이터 검증에 더해 실제 캐시(575,686 train 샘플)로 최소 1 epoch를
  돌려 NaN 없이 Persist baseline을 상회하는지 5개 전부 확인

## HiVT 어댑터 — 설계 메모

원본 HiVT는 (1) `torch_geometric`에 강하게 의존하고 (2) Argoverse HD맵(차선 벡터) 융합
(`ALEncoder`)을 포함하며 (3) 다중 미래 좌표(회귀, num_modes개 가설)를 출력한다. 이 프로젝트
환경(`tna_research`)에는 `torch_geometric`이 설치돼 있지 않고, 우리 데이터에는 차선 벡터도
미래 좌표 정답도 없다(3클래스 state 라벨만 있음). 그래서:

- **차선 융합(ALEncoder) 제거** — 맵이 없으므로 당연히 제외. 대신 이웃(1-hop) 상호작용은
  원본과 동일한 `AAEncoder`(agent-agent attention) + `GlobalInteractor`로 처리.
- **`torch_geometric` 없이 동일 수식 재구현** — `MessagePassing`/`softmax`가 하던 일
  (그룹별 softmax + scatter-add 집계)을 `torch.scatter_reduce_`/`scatter_add_`로 직접 구현.
  attention 수식·게이트 업데이트·MLP 구조는 원본과 동일하다(`comparison/HiVT/models/local_encoder.py`,
  `global_interactor.py`와 1:1 대응 — 파일 상단 주석에 대응 관계 명시).
- **node_dim 2→6 확장** — 원본은 [Δx,Δy] 2D 변위만 입력으로 쓰지만, 우리 데이터는 이미
  6D 운동학 벡터(pos_x,pos_z,speed,dir_x,dir_z,accel)를 제공한다. 회전(rotate_mat)은
  위치 변위쌍[Δx,Δz]과 방향쌍[dir_x,dir_z](둘 다 평면 위 벡터)에만 적용하고, speed·accel은
  회전과 무관한 스칼라라 그대로 통과시킨다 — TSEM-SAGE가 쓰는 것과 동일한 원시 피처를
  HiVT에도 동일하게 제공해 "입력 정보량 차이로 인한 불공정"을 없앤다.
- **num_modes(다중 미래 가설) 제거** — 회귀 전용 개념이라 단일 분류 문제엔 적용 불가.
  `GlobalInteractor`의 `multihead_proj`(모드별 fan-out)는 생략하고 레이어 출력을 그대로 사용.
- **2-hop 이웃 미사용** — 원본 HiVT 자체가 1-hop(로컬) + global interactor(전역) 2단 구조라,
  우리 2-hop 이웃까지 넣으면 원본 설계 취지에서 벗어난다. ego + 1-hop(K=6)까지만 사용.
- **edge_dim=2 유지(원본 그대로)** — NAGraphSAGE류 5D 물리 edge feature(rel_speed 등)를
  주입하지 않고, 원본처럼 위치차(rel_pos)만 edge attr로 사용 — "실제 HiVT 코드"의 설계를
  임의로 유리하게 바꾸지 않기 위함.
- **데이터로더·loss·스케줄러·평가는 `train_tsem.py`와 100% 동일** — `train_hivt_tsem.py`는
  `train_tsem.py`의 핵심 학습 루프를 그대로 재사용하고 모델 클래스만 교체한다.

## HiVT 어댑터 — 재구현 검증 (2026-07-11 완료)

`torch_geometric` 없이 직접 짠 그룹별 softmax·scatter 집계(`segment_softmax`/`scatter_sum_nodes`,
`model.py`)에 버그가 숨어 있을 위험이 가장 컸다. `comparison/hivt_tsem/test_model.py`에서
**완전히 독립적인 브루트포스 구현**(파이썬 for문으로 노드별 들어오는 edge만 걸러 PyTorch 내장
`torch.softmax`로 직접 정규화 — 벡터화 코드의 헬퍼 함수를 재사용하지 않음)과 대조해 4개 테스트를
전부 통과시켰다:

| 테스트 | 결과 | 오차 |
|---|---|---|
| `segment_softmax` vs `torch.softmax`(수동 그룹핑) | PASS | 5.6e-17 |
| 그룹별 softmax 합=1 | PASS | ~1e-16 |
| `TSEMAAEncoder` 1스텝 (고립노드 포함) | PASS | 0.0 (완전 일치) |
| `TSEMGlobalInteractorLayer` (고립노드 포함) | PASS | 3.3e-16 |
| 배치 처리 vs 샘플별 단독 처리 (그래프 누수 검사) | PASS | 6.0e-08 (float32) |

전부 float64 반올림 오차 수준 — 그룹별 softmax·attention·gate 업데이트·배치 offset 로직이 의도한
수식과 정확히 일치함을 확인했다. 고립 노드(들어오는 edge 0개)를 의도적으로 포함해 scatter 기반
구현에서 버그가 가장 잘 숨는 엣지케이스도 검증했다. 재실행: `python comparison/hivt_tsem/test_model.py`.

이건 노드 4개짜리 합성 그래프로 한 **수식 정확성** 검증이며, 실제 데이터 파이프라인 자체는
1-epoch 스모크테스트(2026-07-10, GPU0)로 별도 확인했다 — 캐시 재사용(기존 TSEM-SAGE 실험과 동일
W/H/radius/K라 즉시 로드), test acc 70.92% / macro-F1 68.93%로 Persist baseline(63.17%/50.55%)을
1 epoch만에 상회, NaN 없이 정상 학습됨을 확인. 파라미터 수 512,835개, ~237초/epoch(GPU 1개).

## HiVT 어댑터 — ✅ 실제 300-epoch 학습 완료 (2026-07-12, 이 서버에서 직접 실행)

18:46 시작, `checkpoints/comparison/hivt_tsem_w10_h10/results.json` 기준 최종 test 결과:

| 지표 | HiVT(300ep) | Persist baseline | 10차-2(TSEM-SAGE 최종) |
|---|---|---|---|
| Accuracy | **77.48%** | 63.17% | 81.91% |
| Macro-F1 | **76.35%** | 50.55% | 81.95% |
| stop recall/precision | 94.15% / 79.11% | — | — |
| lane_change recall/precision | 72.04% / 71.87% | — | — |
| normal recall/precision | 65.00% / 78.53% | — | — |

Persist 대비 accuracy +14.31%p, macro-F1 +25.80%p. 5개 baseline 중 CRAT-Pred(75.02%)는 이기지만
QCNet(77.97%)에는 근소하게 뒤진다. 10차-2(81.91%)에는 accuracy −4.43%p, macro-F1 −5.60%p 뒤진다 —
raw 6D 전 채널(TSEM-SAGE와 동일 정보량)을 쓰는데도 10차-2보다 낮다는 건, 10차-2의 우위 중 상당
부분이 NAGraphSAGE 구조나 정보량 자체가 아니라 semantic 분해+위치 암기 신호(Δρ·raw position)에서
온다는 근거를 하나 더 보탠다(아래 "semantic 8D 채널별 site-specific 여부 재점검" 참조).

## QCNet 어댑터 — 설계 메모

원본 QCNet(HiVT와 같은 저자, 후속작)은 (1) `torch_geometric`·`torch_cluster`(radius_graph)에
의존, (2) 맵-에이전트(pl2a) attention 포함, (3) DETR류 anchor-free+anchor-refine 2단계 다중모달
회귀 디코더를 쓴다. HiVT와 가장 큰 설계 차이는 **회전 불변성을 얻는 방식**이다 — HiVT는 노드별
회전행렬(`rotate_mat`)을 명시적으로 곱하지만, QCNet은 **상대 각도(`angle_between_2d_vectors`)를
피처로 직접 사용**해 회전행렬 없이도 회전 불변성을 얻는다. 그래서 QCNet-adapted는 HiVT-adapted와
달리 scene 중심 정렬·회전 전처리 자체가 필요 없다(모든 입력이 상대 거리/각도라 절대 좌표계와 무관).
또한 HiVT는 "1-hop 공간 그래프 + temporal transformer"를 분리 처리하지만, QCNet은 **매 layer마다
시간축 self-attention(t_attn) → 공간축 self-attention(a2a_attn)을 번갈아 반복**하고, 공간 그래프도
anchor 시점 1회가 아니라 **매 timestep마다** 새로 구성한다는 점이 다르다.

- **맵-에이전트(pl2a) attention 제거** — 맵 데이터 없음.
- **`torch_geometric`·`torch_cluster` 없이 동일 수식 재구현** — `AttentionLayer`(비-이분 그래프
  버전만 필요, `bipartite=False`)를 `torch.scatter_reduce_`/`scatter_add_`로 직접 구현.
  `FourierEmbedding`은 순수 `nn.Module`이라 PyG 의존성이 없어 그대로 재구현(수식 100% 동일).
- **x_a 입력 4D 중 2D 대체** — 원본은 [motion 크기, motion-heading 각도, velocity 크기,
  velocity-heading 각도]인데, velocity가 Argoverse 센서 고유 피처라 우리 데이터엔 없음(우리
  방향벡터=heading 자체라 velocity-heading 각도가 항상 0으로 퇴화). 대신 [motion 크기,
  motion-heading 각도, speed, accel]로 대체 — 개수(4D)는 유지하되 뒤 2개를 우리 데이터의
  진짜 독립 신호로 교체.
- **agent-type 카테고리 임베딩 제거** — 우리 데이터는 전부 같은 타입(차량)이라 불필요.
- **DETR류 다중모달 회귀 디코더 제거** — 분류 헤드(ego 노드의 anchor 시점 임베딩 → Linear→3)로 교체.
- **2-hop 이웃 미사용, a2a_radius=graph.radius(20m)** — HiVT-adapted와 동일 이유(원 설계 취지 보존).
- **데이터로더·loss·스케줄러·평가는 `train_tsem.py`와 100% 동일**(HiVT-adapted와 동일 패턴).

## QCNet 어댑터 — 재구현 검증 (2026-07-11 완료)

`comparison/qcnet_tsem/test_model.py`에서 HiVT-adapted와 동일한 방법론(완전히 독립적인
브루트포스 구현과 대조)으로 6개 테스트를 전부 통과시켰다:

| 테스트 | 결과 | 오차 |
|---|---|---|
| `segment_softmax` vs `torch.softmax`(수동 그룹핑) | PASS | 0.0 (완전 일치) |
| `angle_between_2d_vectors`/`wrap_angle` 알려진 값 검산 | PASS | ~1e-7 |
| `AttentionLayer` 1스텝 (고립노드 포함) | PASS | 6.7e-16 |
| `FourierEmbedding` 결정적 재현성 | PASS | 0.0 |
| 배치 처리 vs 샘플별 단독 처리 (그래프 누수 검사) | PASS | 4.9e-07 (float32) |

재실행: `python comparison/qcnet_tsem/test_model.py`.

1-epoch 스모크테스트(2026-07-11, GPU0 단독) 결과 — 캐시는 HiVT-adapted와 동일해 즉시 재사용:

| 지표 | QCNet-adapted (1 epoch) | HiVT-adapted (1 epoch) | Persist baseline |
|---|---|---|---|
| Accuracy | 72.11% | 70.92% | 63.17% |
| Macro-F1 | 70.32% | 68.93% | 50.55% |
| 속도 | 136초/epoch | 237초/epoch | — |

1 epoch만에 NaN 없이 Persist baseline을 상회했고, HiVT-adapted보다도 근소하게 높은 acc/F1 —
다만 이건 1 epoch짜리 약한 신호일 뿐 최종 비교 근거는 아니다(두 모델 다 100 epoch 완주해야
의미 있는 비교). 흥미로운 점은 QCNet-adapted가 파라미터 수(532,899)는 HiVT-adapted(512,835)와
비슷한데 **속도는 더 빠르다** — HiVT의 `nn.TransformerEncoder`(전체 토큰 dense self-attention,
4 layer)보다 QCNet의 sparse edge 기반 attention(3 layer)이 우리처럼 작은 그래프(ego+6이웃)에서는
계산량이 더 적기 때문으로 보인다. 파라미터 수: `python -c "..."` 대신 학습 로그의
`모델: QCNet-adapted  params=532,899` 참조. 체크포인트는
`checkpoints/comparison/qcnet_smoketest/`에 저장, 최종 실험에는 사용하지 않고 삭제 예정.

## QCNet 어댑터 — ✅ 실제 300-epoch 학습 완료 (2026-07-12, 이 서버에서 직접 실행)

21:12 시작, HiVT와 겹치는 시간대에 같은 GPU 0-3을 공유하며 실행됐다(자원 경쟁은 있었으나 학습
정합성엔 영향 없음). `checkpoints/comparison/qcnet_tsem_w10_h10/results.json` 기준 최종 test 결과:

| 지표 | QCNet(300ep) | Persist baseline | 10차-2(TSEM-SAGE 최종) |
|---|---|---|---|
| Accuracy | **77.97%** | 63.17% | 81.91% |
| Macro-F1 | **76.96%** | 50.55% | 81.95% |
| stop recall/precision | 93.64% / 78.88% | — | — |
| lane_change recall/precision | 71.37% / 74.63% | — | — |
| normal recall/precision | 66.92% / 78.57% | — | — |

Persist 대비 accuracy +14.80%p, macro-F1 +26.41%p — **5개 문헌 baseline 중 현재까지 최고 성능**
(HiVT 77.48%, CRAT-Pred 75.02%보다 높음). 10차-2(81.91%)에는 여전히 accuracy −3.94%p, macro-F1
−4.99%p 뒤진다. 가공된 4D 입력(motion 크기·각도+speed+accel)만으로 raw 6D를 그대로 쓰는 HiVT보다
나은 성능이 나온 건 QCNet의 시공간 attention 교대 반복 구조가 이 작은 그래프(ego+6이웃)에서도
효율적으로 작동함을 시사한다.

## CRAT-Pred 어댑터 — 설계 메모

원본(Schmidt et al., ICRA 2022)은 HiVT/QCNet보다 구조가 훨씬 단순하다 — `torch_geometric`
의존은 CGConv(Crystal Graph Convolution, 재료과학에서 유래) 레이어 딱 하나뿐이고, 나머지는
순수 PyTorch(`nn.LSTM`, `nn.MultiheadAttention`)다. 파이프라인: **에이전트별 독립 LSTM**(시간
인코딩) → **CGConv 2-layer**(공간, 완전연결 그래프 + anchor 시점 상대 center를 edge_attr로) →
**MultiheadSelfAttention 1-layer**(공간 재조정) → 다중모달 회귀 디코더(제거 대상). HiVT/QCNet과
달리 attention을 반복(iterative)하지 않고 "GNN 먼저, attention은 마지막에 한 번"이라는 얕은
구조라, HiVT/QCNet-adapted 대비 **"복잡한 반복형 attention이 꼭 필요한가"**를 검증하는 대조군
역할을 한다.

- **CGConv를 `torch_geometric` 없이 재구현** — softmax가 없는 gated-sum 집계라
  (`sigmoid(W_f·z)·softplus(W_s·z)`를 edge마다 계산 후 타깃 노드로 `scatter_add`) HiVT/QCNet의
  attention보다 재구현 난이도가 낮다 — 이게 §관련연구(A) 표에서 CRAT-Pred를 HEAT과 함께 재구현
  최우선 후보로 꼽은 이유였다.
- **`EncoderLstm`·`MultiheadSelfAttention`은 원본 그대로 재사용** — 둘 다 순수 PyTorch 내장
  모듈(`nn.LSTM`, `nn.MultiheadAttention`)이라 애초에 PyG 의존성이 없다.
- **LSTM 입력은 원본과 100% 동일하게 3D([Δx,Δy,valid flag])로 유지** — HiVT/QCNet과 달리
  speed·dir·accel 등 우리 데이터의 추가 채널을 넣지 않았다. CRAT-Pred라는 모델의 존재 의의가
  "최소한의 입력(변위+유효플래그)만으로 얼마나 되는가"이므로, 채널을 늘리면 이 모델을 고른
  비교 포인트 자체가 흐려진다는 판단 — HiVT/QCNet과 다른 선택이며 의도적이다.
- **DETR류 다중모달 회귀 디코더(`DecoderResidual`, mod_steps 앙상블) 제거** — 분류 헤드로 교체.
- **가변 에이전트 수 → 고정 크기(1+K)+`valid_agent` 마스크** — HiVT/QCNet과 동일한 배치 벡터화
  패턴. `nbr_mask=0`인 슬롯은 그래프에서 완전히 제외(개별 프레임만 마스킹하는 HiVT/QCNet과 달리,
  원본이 애초에 "관측된 에이전트만" 다루는 가변 그래프 설계라 이쪽이 더 원본에 충실).
- **scene 정렬은 원본과 동일하게 1회만** — HiVT의 노드별 회전행렬이나 QCNet의 상대각도 인코딩
  같은 추가 회전-불변 장치가 원본에 아예 없다(ego 헤딩 기준 scene 전체를 한 번 정렬하는 게 전부).
  이 어댑터도 그 설계를 그대로 따른다.

## CRAT-Pred 어댑터 — 재구현 검증 (2026-07-11 완료)

`comparison/cratpred_tsem/test_model.py`에서 HiVT/QCNet-adapted와 동일한 방법론으로 3개
테스트를 통과시켰다(CGConv는 softmax가 없어 검증할 표면적 자체가 더 작음):

| 테스트 | 결과 | 오차 |
|---|---|---|
| `scatter_sum_nodes` vs 노드별 for-loop 합산 | PASS | 0.0 (완전 일치) |
| `TSEMCGConv` 1스텝 (고립노드 포함, BatchNorm running-stats 고정) | PASS | 8.9e-16 |
| 배치 처리 vs 샘플별 단독 처리 (그래프 누수 검사) | PASS | 1.5e-08 (float32) |

재실행: `python comparison/cratpred_tsem/test_model.py`.

1-epoch 스모크테스트(2026-07-11, GPU0 단독) 결과 — 캐시는 앞선 두 모델과 동일해 즉시 재사용:

| 지표 | CRAT-Pred-adapted (1ep) | QCNet-adapted (1ep) | HiVT-adapted (1ep) | Persist |
|---|---|---|---|---|
| Accuracy | 69.61% | 72.11% | 70.92% | 63.17% |
| Macro-F1 | 67.41% | 70.32% | 68.93% | 50.55% |
| 파라미터 수 | **68,291** | 532,899 | 512,835 | — |
| 속도 | **57.6초/epoch** | 136초/epoch | 237초/epoch | — |

세 모델 중 정확도는 가장 낮지만(1 epoch 기준, 아직 결론 아님), **파라미터 수는 1/7~1/8, 속도는
2~4배 빠르다** — "단순한 구조가 이 정도 데이터 규모에서는 오히려 유리할 수 있다"는 가설과 부합.
흥미롭게도 LC recall은 세 모델 중 가장 높다(val 74.81% vs QCNet 71.25% vs HiVT 69.56%) — 다만
1 epoch만의 관측이라 노이즈일 가능성을 배제할 수 없다. 체크포인트는
`checkpoints/comparison/cratpred_smoketest/`에 저장, 최종 실험에는 사용하지 않고 삭제 예정.

**다음 단계**: 나머지 두 모델과 동일하게 전체 300-epoch(patience 50) 학습은 타 서버에서 진행 예정 —
`comparison/cratpred_tsem/config.yaml` + `train_cratpred_tsem.py`를 그대로 복사해 실행.

## CRAT-Pred 어댑터 — ✅ 실제 300-epoch 학습 완료 (2026-07-11, 이 서버에서 직접 실행)

당초 "타 서버에서 진행 예정"이었으나 실제로는 이 서버 GPU 0-3에서 바로 실행해 완료됐다
(`kl_weight: 0.1`/`use_uncertainty_weight: true` 수정 반영 + 배치 스케일링 적용 후 버전).
**best epoch 78**(early stop patience 50, `checkpoints/comparison/cratpred_tsem_w10_h10/`)
기준 최종 test 결과:

| 지표 | CRAT-Pred(300ep, best@78) | Persist baseline | 10차-2(TSEM-SAGE 최종) |
|---|---|---|---|
| Accuracy | **75.02%** | 63.17% | 81.91% |
| Macro-F1 | **72.84%** | 50.55% | 81.95% |
| stop recall/precision | 95.81% / 76.94% | — | — |
| lane_change recall/precision | 59.30% / 71.08% | — | — |
| normal recall/precision | 63.59% / 74.29% | — | — |

Persist baseline 대비 accuracy **+11.85%p**, macro-F1 **+22.29%p**로 뚜렷하게 상회 — 학습이
정상적으로 수렴했다는 확실한 근거다. 다만 10차-2(TSEM-SAGE 최종 채택 모델) 대비로는 accuracy
**−6.89%p**, macro-F1 **−9.11%p** 낮다 — 파라미터 수(68,291, TSEM-SAGE의 약 1/2.5)가 훨씬
적은 경량 구조라는 점을 감안하면 예상 범위 내 결과다.

**1-epoch 스모크테스트 대비 변화**: accuracy 69.61%→75.02%(+5.41%p), macro-F1 67.41%→72.84%
(+5.43%p)로 학습이 진행될수록 꾸준히 개선됐다. 다만 <b>LC recall은 오히려 하락</b>했다(1-epoch
70.45%(val) → 300ep 완주 후 59.30%(test)) — 학습 초반엔 LC를 과다예측하다가, 학습이 진행되며
normal/stop 쪽 정확도가 함께 올라가면서 LC recall이 상대적으로 낮아진 것으로 보인다(LC
precision은 오히려 71.08%로 높아짐 — recall↔precision 트레이드오프 방향). 단일 run이라 확정적
결론은 아니고, 다른 4개 모델도 완주 후 같은 패턴을 보이는지 비교가 필요하다.

## SIMPL 어댑터 — 설계 메모

원본(Zhang et al., IEEE RA-L 2024)은 앞선 셋과 또 다른 설계를 쓴다 — 시간 인코더가 **1D
CNN(ResNet+FPN, 다중 스케일)**로, 셋 중 유일하게 transformer도 LSTM도 아니다. 공간 융합은
"Symmetric Fusion Transformer(SFT)"라는 독특한 패턴 — **각 타깃 토큰마다 (상대위치 edge + source
노드 + target 노드)를 결합한 전용 memory 텐서**를 만들고, 그 memory를 key/value 삼아 PyTorch
내장 `nn.MultiheadAttention`을 호출한다. 회전 불변성은 QCNet과 유사하게 상대각도를 특징으로
직접 쓰지만, atan2 각도값 대신 **cos/sin 쌍을 그대로 특징 벡터에 넣어** wrap-around 불연속을
피한다.

- **attention 자체는 재구현하지 않음** — SftLayer의 핵심은 "memory 텐서를 올바르게 만드는
  브로드캐스트 로직"이지 attention 수식 자체가 아니다. `nn.MultiheadAttention`(PyTorch 내장,
  신뢰 가능)에 위임하고, 우리가 재구현·검증해야 할 건 memory 구성 브로드캐스트뿐 — 그래서
  HiVT/QCNet과 달리 `segment_softmax`류 함수가 아예 없다.
- **LaneNet(맵 인코더)·MLPDecoder(다중모달 베지어/모노미얼 회귀) 제거** — 맵 없음, 분류 헤드로 교체.
  RPE(상대위치 인코딩)도 에이전트끼리만 계산.
- **ActorNet 입력은 원본과 동일 3D([Δx,Δy,valid flag])** — CRAT-Pred와 동일한 이유(원본 설계를
  그대로 시험)로 채널을 늘리지 않음.
- **n_fpn_scale 4→3** — 원본은 T=20(historical_steps) 기준 설계인데 우리는 T=10이라 다운샘플링
  스케일을 하나 줄임. FPN 업샘플도 원본의 `scale_factor=2`(정확히 절반씩 줄어든다고 가정) 대신
  `size=`를 명시적으로 지정해 임의 T에서도 길이 불일치 없이 안전하게 동작하도록 강건화 —
  수식·구조는 동일, 견고성만 개선.
- **2-hop 이웃 미사용** — 원본도 관측된 모든 에이전트(1-hop)만 사용.

## SIMPL 어댑터 — 재구현 검증 (2026-07-11 완료, 실제 버그 1건 발견·수정)

`comparison/simpl_tsem/test_model.py`에서 (1) RPE 수식, (2) SftLayer의 memory 브로드캐스트
로직을 독립 브루트포스와 대조했다. **이 과정에서 실제 인덱싱 버그를 발견했다** — 값이 그럴듯해
보여도 조용히 틀렸을 뻔한 사례라 자세히 기록한다.

**발견된 버그**: `TSEMSftLayer`에서 memory 텐서 `memory[b,i,j]`(i=target, j=source 후보)를
만드는 것 자체는 맞았지만, 이를 `nn.MultiheadAttention`에 넣기 위해 (배치, 시퀀스) 축으로
재배열하는 과정에서 **배치=i(타깃), 시퀀스=j(소스)로 잘못 배정**했다 — 원본 SIMPL의 수식을
끝까지 추적하면 실제로는 배치=j(타깃), 시퀀스=i(소스)여야 한다(원본 `_build_memory`의
`src_x`/`tar_x` 브로드캐스트 규칙을 대수적으로 역추적해서 확인). `permute(2,0,1,3)`을
`permute(1,0,2,3)`으로 수정해 해결 — 수정 후 브루트포스와 완전히 일치(오차 4.4e-16).

흥미롭게도 처음 작성한 브루트포스 테스트 코드 자체에도 **별도의, 우연히 겹친 유사한 버그**
(src_x/tar_x 변수명을 반대로 할당)가 있었다 — 두 버그가 우연히 부분적으로 상쇄되지 않고 정직하게
드러난 덕분에(첫 실행에서 오차 0.54, 수정 후에도 오차 0.82로 계속 실패) 두 곳 모두 찾아 고칠 수
있었다. 이게 바로 "완전히 독립적인 두 경로를 대조"하는 검증 방식의 요점이다 — 벡터화 코드가
가져다 쓰는 헬퍼를 브루트포스가 재사용했다면 같은 버그가 양쪽에 똑같이 있어 통과했을 것이다.

| 테스트 | 결과 | 오차 |
|---|---|---|
| `build_rpe_batched` vs `build_rpe`(단일 그래프) | PASS | 0.0 (완전 일치) |
| RPE 알려진 값 검산(거리 0.2, 수직 헤딩 cos=0) | PASS | 0.0 |
| `TSEMSftLayer` node 출력 (버그 수정 후) | PASS | 4.4e-16 |
| `TSEMSftLayer` edge 업데이트 | PASS | 0.0 (완전 일치) |
| 배치 처리 vs 샘플별 단독 처리 (그래프 누수 검사) | PASS | 1.5e-08 (float32) |

재실행: `python comparison/simpl_tsem/test_model.py`.

1-epoch 스모크테스트(2026-07-11, GPU0 단독, **버그 수정 후 모델로 실행**) 결과:

| 지표 | SIMPL-adapted (1ep) | CRAT-Pred (1ep) | QCNet (1ep) | HiVT (1ep) | Persist |
|---|---|---|---|---|---|
| Accuracy | 70.62% | 69.61% | 72.11% | 70.92% | 63.17% |
| Macro-F1 | 68.43% | 67.41% | 70.32% | 68.93% | 50.55% |
| LC recall(test) | **70.61%** | — | — | — | — |
| 파라미터 수 | 488,035 | 68,291 | 532,899 | 512,835 | — |
| 속도 | 261초/epoch(최느림) | 57.6초/epoch | 136초/epoch | 237초/epoch | — |

네 모델 다 Persist를 여유 있게 상회, 1 epoch 기준으로는 QCNet이 accuracy/F1 최고, CRAT-Pred가
가장 가볍고 빠름, SIMPL은 가장 느리지만(4-layer SFT + CNN) LC recall 계열에서 강세를 보이는
경향이 계속 관찰된다(3·4번째 모델 모두 CRAT-Pred/SIMPL이 QCNet/HiVT보다 val LC recall이 높았음)
— 다만 전부 1 epoch 신호일 뿐 100 epoch 완주 전까지는 확정적 결론이 아니다.

**다음 단계**: 네 모델 전부 전체 300-epoch(patience 50) 학습은 타 서버에서 진행 예정 —
`comparison/simpl_tsem/config.yaml` + `train_simpl_tsem.py`를 그대로 복사해 실행.

## Forecast-MAE 어댑터 — 설계 메모

원본(Cheng, Mei & Liu, ICCV 2023)의 핵심 아이디어는 아키텍처가 아니라 **학습 프로토콜** —
(1) 레이블 없이 마스킹된 에이전트의 과거 궤적을 복원하도록 인코더를 자기지도 사전학습하고,
(2) 그 인코더 가중치를 로드해 분류/회귀 헤드를 얹어 지도학습으로 미세조정한다. 이건 TSEM-SAGE의
augmentation 실험(9차, `configs/tsem_sage_9th.yaml`)과 비교 가치가 큰 유일한 후보다 — "레이블
부족·과적합을 데이터 증강으로 보완"(우리 접근) vs "레이블 없는 사전학습으로 보완"(Forecast-MAE
접근) 중 어느 쪽이 더 낫거나 상호보완적인지 직접 비교할 수 있다.

- **NATTEN(Neighborhood Attention, CUDA 커널) 대체 — 유일하게 "얇은 wrapper 제거"가 아니라
  "대체 커널이 없어 재현 불가능"한 경우.** 원본의 `AgentEmbeddingLayer`는 지역(windowed)
  attention을 위해 `natten` 패키지에 강하게 의존하는데, 이 환경엔 설치돼 있지 않고 CUDA 커널
  없이는 동작 자체가 안 된다. `torch_geometric`(HiVT/QCNet)처럼 "MessagePassing/softmax만
  scatter로 재구현하면 되는" 의존성과는 성격이 다르다 — 그래서 **Conv1d 토크나이저 + 표준
  global self-attention(TSEMBlock)**으로 대체했다. 우리 W=10(원 논문 historical_steps=50)처럼
  짧은 시퀀스에서는 지역·전역 attention의 실질 차이가 작다는 점이 이 대체를 뒷받침한다.
- **`transformer_blocks.py::Block`은 원본 그대로 재구현** — 순수 `nn.MultiheadAttention` 기반이라
  `natten` 의존이 없고, `timm.DropPath`만 직접 재구현(5줄 내외의 표준 stochastic depth)하면
  그대로 재현 가능.
- **차선(LaneEmbeddingLayer)·차선 마스킹·차선 복원loss 제거** — 맵 데이터 없음.
- **미래 궤적 복원(future_embed/future_pred) 제거** — 우리 데이터엔 미래 좌표 정답이 없다(3클래스
  state 라벨만). 사전학습 복원 대상은 **과거 궤적(history)만** — 원 손실 3항(future+hist+lane)
  중 hist 항만 남긴 셈.
- **원조 MAE의 비대칭 encoder/decoder(마스킹 토큰을 인코더 입력에서 아예 제거해 계산량을 줄이는
  트릭) 생략** — 우리 그래프는 에이전트 7개뿐이라 그 최적화의 이득이 미미하다. 대신 BERT류
  방식(마스킹 위치를 mask_token으로 치환해 전부 함께 인코딩)으로 단순화 — 마스킹+복원 학습
  신호 자체는 동일.
- **MultimodalDecoder(다중모달 회귀) 제거** — 미세조정 시 분류 헤드로 교체.
- **2단계 프로토콜은 원본 그대로 유지** — `train_forecastmae_tsem.py --mode pretrain`(레이블
  미사용) → `--mode finetune --pretrained_ckpt ...`(사전학습 인코더 로드 후 지도학습, TSEM-SAGE와
  동일 FocalLoss/class weight/OneCycleLR/early stopping).

## Forecast-MAE 어댑터 — 재구현 검증 (2026-07-11 완료)

`comparison/forecastmae_tsem/test_model.py`에서 (1) 마스킹 샘플링(유효 에이전트 중 정확히
mask_ratio 비율을, 최소 1개는 항상 남기고 선택하는지 — n_valid=1~7 전 케이스 통계적으로 확인),
(2) 복원 loss의 정렬(마스킹된+실제관측된 위치에만 걸리는지), (3) 배치 처리와 그래프 격리를
검증했다. `TSEMBlock`(=`nn.MultiheadAttention` 감싼 표준 transformer layer)은 attention 자체를
재구현하지 않아(PyTorch 내장에 위임) 별도 브루트포스 대조가 필요 없었다.

| 테스트 | 결과 |
|---|---|
| 마스킹은 항상 valid_agent의 부분집합 | PASS |
| n_valid=1이면 마스킹 0개 (복원할 context 없어짐 방지) | PASS (7가지 n_valid 케이스 × 다수 샘플) |
| n_valid=2~7일 때 마스킹 개수 ≈ round(n_valid×0.5) | PASS |
| reg_mask(마스킹∧관측) 벡터화 vs 브루트포스 | PASS (완전 일치) |
| finetune 배치 처리 vs 샘플별 단독 처리 (그래프 누수 검사) | PASS (1.2e-07) |

재실행: `python comparison/forecastmae_tsem/test_model.py`.

1-epoch 스모크테스트(2026-07-11, GPU0 단독, 사전학습 1epoch → 그 인코더로 미세조정 1epoch) 결과:

| 단계 | 지표 | 값 |
|---|---|---|
| 사전학습 | train/val 복원 loss(L1) | 5.49 / 5.55 |
| 사전학습 | 속도 | 81.6초/epoch |
| 미세조정 | 인코더 로드 | `missing=[]`, `unexpected=[]` (키 완전 일치) |
| 미세조정 | Accuracy | 69.37% |
| 미세조정 | Macro-F1 | 67.31% |
| 미세조정 | LC recall(test) | **74.06%** (5개 모델 중 최고) |
| 미세조정 | 속도 | 74.3초/epoch |

사전학습 인코더가 미세조정 모델에 오류 없이 로드됐고(빠진 키·안 쓰인 키 0개), 1 epoch만의
미세조정으로 Persist(63.17%/50.55%)를 상회했다 — 파이프라인이 의도한 2단계 프로토콜대로
정상 작동함을 확인. LC recall이 5개 모델 중 가장 높다는 점은 흥미롭다(SIMPL 70.61%, CRAT-Pred는
val 기준 74.81%로 근접) — "마스킹 복원 사전학습이 희귀 클래스(LC) 표현을 더 잘 학습하게
해주는가"는 100 epoch 완주 후 봐야 할 핵심 질문 중 하나. 체크포인트는
`checkpoints/comparison/forecastmae_smoketest/`(`pretrain_encoder.pt` + `best.pt`)에 저장,
최종 실험에는 사용하지 않고 삭제 예정.

**다음 단계**: 다섯 모델 전부 전체 학습(사전학습+미세조정 300-epoch, patience 50)은 타
서버에서 진행 예정 — `comparison/forecastmae_tsem/config.yaml` +
`train_forecastmae_tsem.py`를 그대로 복사해 실행(2단계 커맨드는 파일 상단 docstring 참조).

## 다섯 모델 종합 (1-epoch 스모크테스트 기준, 2026-07-11)

| 모델 | Accuracy | Macro-F1 | LC recall | 파라미터 | 속도 |
|---|---|---|---|---|---|
| QCNet | **72.11%** | **70.32%** | — | 532,899 | 136초/ep |
| HiVT | 70.92% | 68.93% | — | 512,835 | 237초/ep(최느림) |
| SIMPL | 70.62% | 68.43% | 70.61% | 488,035 | 261초/ep |
| CRAT-Pred | 69.61% | 67.41% | — | **68,291**(최경량) | **57.6초/ep**(최속) |
| Forecast-MAE | 69.37% | 67.31% | **74.06%**(최고) | 305,731(+사전학습 411,092) | 74.3초/ep(finetune) |
| 10차-2(TSEM-SAGE, 100ep 완주) | **81.91%** | **81.95%** | — | — | — |
| Persist baseline | 63.17% | 50.55% | — | — | — |

**주의**: 위 5개 비교 모델 수치는 전부 **1 epoch**만 돈 스모크테스트로, 정식 비교가 아니라
"파이프라인이 정상 동작하는가"를 확인한 것뿐이다. 10차-2는 100 epoch 완주한 최종 수치라 직접
비교하면 안 된다 — 5개 모델도 100 epoch 완주해야 진짜 비교가 성립한다(타 서버에서 예정).

## STGCN/DCRNN/TGN/LSTM/Transformer 어댑터 (2026-07-11 추가)

위 5개 문헌 baseline과 별도로, 그래프-시계열 예측 계열의 대표 아키텍처 3종(STGCN/DCRNN/TGN)과
"학습되는 공간 상호작용 모듈이 아예 없을 때 얼마나 되는가"를 보는 순수 아키텍처 대조군 2종
(LSTM/Transformer)을 추가했다. 5개 문헌 baseline과 동일하게 `modules/data_manager_tsem.py`+
`modules/tsem_eval.py`+`train_tsem.py`의 loss/scheduler/early-stopping을 100% 그대로 재사용하고,
data/graph/train/loss/augment config 섹션도 `cratpred_tsem/config.yaml`과 동일하게 맞췄다(모델
섹션만 고유 하이퍼파라미터). 전부 재구현 검증(`test_model.py`, 독립 브루트포스 대조) PASS,
실데이터 캐시(W10/H10/r20/K6-4, 기존 실험과 공유)로 50-step 짧은 forward+backward에서 NaN 없이
loss가 감소함을 확인했다(GPU 4개가 이미 다른 300-epoch job으로 21~22GB/24GB씩 차 있어, 정식
1-epoch 스모크테스트 대신 여유 있는 GPU에서 소규모 스텝만 확인 — 아래 각 절 참조).

### STGCN 어댑터 — 설계 메모

원본(Yu, Yin & Zhu, IJCAI 2018)은 `hazdzz/STGCN`(순수 PyTorch, `torch_geometric` 의존 없음)을
`comparison/STGCN/`에 clone. 핵심은 **ST-Conv 블록**(temporal gated conv → graph conv → temporal
gated conv 반복) — 원래는 고정된 도로 센서망(그래프 구조가 매 샘플 동일) 위의 교통량 회귀용이라,
우리처럼 샘플마다 다른 ego+이웃(K=6, N=7) 그래프를 쓰려면 배치별로 인접행렬을 새로 구성해야 한다.

- **Temporal Gated Conv**: 원본과 동일하게 1D causal convolution + GLU(Gated Linear Unit) 그대로
  재현, 커널 크기(Kt=3)만 W=10 시퀀스 길이에 맞게 축소.
- **Graph Conv**: 원본의 1차 근사(GCN 스타일) 정규화 인접행렬 Â=D^-1/2(A+I)D^-1/2를 N=7 dense
  텐서 행렬곱으로 구현(`torch_geometric`/sparse 불필요, 원본도 실제로는 dense 지원). 인접행렬은
  이진 대신 **Gaussian distance kernel**(σ=10m)로 구성 — 원 논문(도로망 거리 기반 가중 인접행렬)
  에 더 충실하다는 판단.
- 다중스텝 회귀 출력(원본은 미래 여러 스텝 교통량 예측) 제거 → ego 노드 최종 표현을 분류 헤드로.
- 노드 입력 6D(HiVT와 동일 근거), scene 1회 정렬(ego anchor 위치·헤딩 기준 회전)은 다른 baseline과
  동일 적용.

**재구현 검증(`comparison/stgcn_tsem/test_model.py`, 8개 항목 전부 PASS)**: 정규화 인접행렬 vs
손계산(3노드), 무효 노드 고립 처리, graph conv einsum vs 노드별 for-loop, causal conv의 인과성
(미래 프레임이 과거 출력에 영향 없음), 배치 vs 샘플별 단독 처리(그래프 누수 없음) 등. **검증 중
실제 버그 1건 발견·수정**: `LayerNorm`이 vertex+channel 축을 함께 정규화하면서, graph conv가
무효 이웃의 메시지 기여는 정확히 0으로 처리했음에도 무효 이웃의 raw 피처가 LayerNorm 통계량에는
그대로 섞여 들어가고 있었다 — ST-Conv 스택 진입 전에 무효 노드 피처를 명시적으로 0으로 만드는
방식으로 수정 후 재검증, 전부 PASS. 파라미터 7,827개(6개 신규 baseline 중 최경량).

### DCRNN 어댑터 — 설계 메모

원본(Li, Yi, Shahabi & Liu, ICLR 2018)은 저자 공식 구현이 TensorFlow뿐이라, PyTorch 포트인
`chnsh/DCRNN_PyTorch`를 `comparison/DCRNN/`에 clone. 핵심은 **DCGRU** — 표준 GRU의 게이트 내부
선형변환(Wx+Uh)을 그래프 위의 **diffusion convolution**(K-hop 양방향 확산, forward
P_f=D_O⁻¹A·backward P_b=D_I⁻¹Aᵀ)으로 치환한 구조. 원본은 seq2seq 인코더-디코더로 멀티스텝 회귀를
하지만, 분류 문제인 우리는 **인코더만** 사용(디코더 제거 — 다른 baseline과 동일한 "회귀→분류"
공통 패턴).

- diffusion conv `Σ_{k=0}^{K}(θ_{k,f}P_f^k x + θ_{k,b}P_b^k x)`를 N=7 dense 텐서의 반복 행렬곱
  (`bmm`)으로 구현 — 원본의 sparse/Chebyshev 재귀 대신 논문 원 수식을 그대로 직접 계산(작은
  그래프라 dense가 더 단순하고 안전).
  인접행렬은 STGCN과 동일하게 Gaussian distance kernel로 구성해 P_f/P_b를 유도.
- DCGRU 셀은 표준 GRU 게이트 수식(update/reset/candidate)에서 Wx+Uh만 diffusion conv로 치환 —
  원본과 동일 재현. W=10 프레임 순차 적용 후 ego(slot 0) 최종 hidden state를 분류 헤드로.
- 노드 입력 6D(HiVT와 동일 근거), 2-hop 이웃 미사용(K-hop diffusion conv 자체가 확산 범위를
  커버하므로 원본 2-hop 데이터는 불필요 — 다른 baseline과 동일 근거).

**재구현 검증(`comparison/dcrnn_tsem/test_model.py`, 4개 항목 전부 PASS)**: `build_diffusion_supports`
(P_f,P_b 구성, 무효 노드 마스킹 포함) vs 손계산(4노드), 벡터화 diffusion conv vs 노드별 for-loop
(K=3), **K=0일 때 DCGRU가 표준 fc-GRU와 수식적으로 완전히 동치가 되는지**(diffusion 항이 결과에
영향 없음을 직접 확인 — sanity check), 배치 vs 샘플별 단독 처리(그래프 누수 없음). 파라미터
190,659개(hidden_dim=64, K=2, num_layers=2 — 6개 신규 baseline 중 최대).

### TGN 어댑터 — 설계 메모 (가장 큰 설계 변경)

원본(Rossi et al., ICML 2020 Workshop, Twitter Research, Apache 2.0)을 `comparison/TGN/`에 clone.
**원본 TGN은 비동기 연속시간 이벤트 스트림 위에서, 전체 데이터셋에 걸쳐 지속되는 전역 memory
테이블**로 동작하지만, 우리 과제는 W=10 고정 길이 윈도우 단위의 지도학습 분류이고 그래프도
샘플마다(KNN) 다르다 — 전역 영속 memory를 그대로 쓰면 train/val/test 분할을 가로질러 정보가
새는 구조적 모순이 생긴다. 그래서 **memory를 샘플 단위로 리셋**한다: 각 샘플의 에이전트 슬롯
(1+K=N개)마다 윈도우 시작 시 zero-initialized memory를 두고, W개 프레임을 순서대로 "이벤트
배치"로 처리하며 원본과 동일한 message→aggregate→GRUCell 갱신 규칙을 매 프레임 적용한다. 이
이탈은 "연속 스트림 데이터가 없다"는 데이터 특성상 불가피한 최소 변경이라 판단했다(다른 4개도
"핵심 메커니즘 보존, 필요한 만큼만 변경" 원칙을 따름).

- **Message function**: `MLP([memory_src, memory_dst, edge_feat, Δt인코딩])` — `edge_seqs`
  5D(rel_speed,rel_accel,rel_dir_x,rel_dir_z,distance)를 그대로 edge feature로 사용(원본
  `MLPMessageFunction`과 구조 동일 — TGN이 유일하게 다른 5개 문헌 baseline과 같은 edge feature
  개념을 쓰는 이유).
- **Message aggregator**: 원본의 mean aggregator를 masked mean(N≤7 dense)으로 재현.
- **Memory updater**: `nn.GRUCell(message_dim, memory_dim)`, 원본과 동일.
- **Embedding module**: `TemporalAttentionLayer`를 masked `nn.MultiheadAttention` 기반으로 재현,
  원본처럼 매 프레임이 아니라 **anchor 시점(t=W-1)에만** 적용(원본도 실제 쿼리 시점에만 임베딩을
  계산하므로 동일 원칙).
- 노드 입력 6D, scene 1회 정렬은 다른 baseline과 동일 적용, 1-hop만 사용.

**재구현 검증(`comparison/tgn_tsem/test_model.py`, 5개 항목 전부 PASS)**: masked mean vs
for-loop, GRUCell 순차 memory 갱신 손계산 대조, **전체 forward 파이프라인을 완전히 독립적으로
손계산한 memory 갱신+attention과 K=1 케이스로 end-to-end 대조**(오차 0), temporal attention의
masked softmax vs 수동 QK^T softmax(고립 노드 케이스 포함). **검증 중 실제 버그 1건 발견·수정**:
브루트포스 참조 구현 쪽에서 head 결합 후 `out_proj`를 빠뜨렸고, 고립 노드(이웃 0개)일 때의
zero-fill 정책도 원본과 다르게 처리하고 있었다 — 수정 후 재검증 전부 PASS. 배치 vs 샘플별 단독
처리(memory가 샘플 단위로 리셋되므로 누수가 없어야 함)도 확인. 파라미터 81,695개
(memory_dim=64, message_dim=64, time_dim=16).

### LSTM / Transformer 어댑터 — 설계 메모 (원본 repo 없음, 순수 아키텍처 대조군)

이 둘은 **문헌에서 가져온 baseline이 아니라, "학습되는 공간 상호작용 모듈이 아예 없을 때 이
데이터셋에서 얼마나 되는가"를 보는 대조군**이다 — 나머지 9개(TSEM-SAGE 포함 8개 문헌/그래프
baseline)는 전부 attention/GNN/diffusion conv 중 하나로 명시적인 공간 상호작용을 학습하지만, 이
둘은 그 모듈을 의도적으로 제거했다. 그래서 원본 GitHub repo를 clone하지 않고(§원칙 2의 전제인
"보존할 원본 구현" 자체가 없음 — 표준 PyTorch 내장 모듈 `nn.LSTM`/`nn.TransformerEncoder`를 그대로
사용), 두 어댑터는 **공간 처리 부분을 서로 동일하게 맞춰**(masked mean-pooling, 그래프/attention
전혀 없음) "시간 인코더를 LSTM→Transformer로 바꾸면(공간 모듈 없이) 뭐가 달라지는가"만 순수하게
비교할 수 있게 설계했다.

- **공통 설계**: 노드 입력 6D(scene 1회 정렬된 변위+speed+dir+accel, HiVT와 동일 정보량). ego와
  각 이웃(K=6)을 **가중치 공유 시간 인코더**로 독립 인코딩한 뒤, 유효한 이웃 임베딩을 학습 파라미터
  없는 **masked mean-pooling**으로 합친다(attention도 GNN도 아님 — 이게 이 두 baseline의 핵심
  포인트). `concat([ego_embed, pooled_nbr_embed])` → MLP → 3클래스.
- **LSTM**(`comparison/lstm_tsem/`): 단일 `nn.LSTM`(CRAT-Pred의 `TSEMEncoderLstm`과 같은 방식이나
  입력 차원만 6으로 확장), 마지막 timestep hidden state를 임베딩으로 사용. 파라미터 26,883개.
- **Transformer**(`comparison/transformer_tsem/`): 표준 `nn.TransformerEncoder`(2 layer, d_model=64,
  nhead=4, sinusoidal positional encoding, **causal mask**로 미래 프레임을 보지 못하게 함 — LSTM의
  "마지막 hidden state만 앞의 정보를 담는다"는 성질과 동등하게 맞추기 위함), anchor 시점(t=W-1)
  출력을 임베딩으로 사용. 파라미터 75,843개.

**재구현 검증**: 둘 다 커스텀 scatter/attention 알고리즘이 없어(masked mean-pool은 단순 연산,
Transformer는 `nn.MultiheadAttention` 내장에 위임) 검증 표면적이 작지만, 공통으로 (1) masked
mean-pooling 벡터화 vs 브루트포스 for-loop 대조, (2) 배치 처리 vs 샘플별 단독 처리(그래프/배치
누수 검사), (3) 이웃이 전부 무효인 고립 ego 케이스에서 NaN/Inf 없음을 확인했다(전부 PASS,
`comparison/lstm_tsem/test_model.py` / `comparison/transformer_tsem/test_model.py`). Transformer는
추가로 positional encoding이 시간축마다 서로 다른 값을 갖는지도 확인했다.

### 신규 5종 — 파라미터/검증 요약

| 모델 | 파라미터 | 원본 repo | 재구현 검증 | 비고 |
|---|---|---|---|---|
| STGCN | 7,827(**6개 중 최경량**) | `hazdzz/STGCN` | 8/8 PASS(실버그 1건 수정) | Gaussian kernel 인접행렬 |
| DCRNN | 190,659(**6개 중 최대**) | `chnsh/DCRNN_PyTorch` | 4/4 PASS | K=0→표준GRU 동치 sanity check |
| TGN | 81,695 | `twitter-research/tgn` | 5/5 PASS(실버그 1건 수정) | memory 샘플단위 리셋(설계 이탈) |
| LSTM | 26,883 | 없음(대조군) | 3/3 PASS | 공간모듈 없음, masked mean-pool |
| Transformer | 75,843 | 없음(대조군) | 4/4 PASS | 공간모듈 없음, LSTM과 대조쌍 |

**주의**: 위 5종은 아직 1-epoch 스모크테스트(정식 accuracy/F1 측정)가 아니라 50-step 미니 학습만
확인했다(GPU 여유 부족 — 상단 참조) — 위 "다섯 모델 종합" 표의 5개 문헌 baseline과 달리 아직
accuracy/F1/LC recall 수치가 없다. **다음 단계**: GPU가 비는 대로 각 `comparison/<model>_tsem/
train_<model>_tsem.py --config comparison/<model>_tsem/config.yaml`로 1-epoch 스모크 → 이상 없으면
300-epoch(patience 50) 본 실행. 결과가 나오는 대로 이 절과 위 "다섯 모델 종합" 표를 함께 갱신할 것.
