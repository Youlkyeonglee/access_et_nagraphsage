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

| 모델 | 상태 | 위치 |
|---|---|---|
| HiVT (CVPR 2022) | ✅ 구현+검증 완료, **학습은 타 서버에서 진행 예정** | `comparison/HiVT/`(원본), `comparison/hivt_tsem/`(어댑터) |
| QCNet (CVPR 2023) | ✅ 구현+검증 완료, **학습은 타 서버에서 진행 예정** | `comparison/QCNet/`(원본), `comparison/qcnet_tsem/`(어댑터) |
| CRAT-Pred (ICRA 2022) | ✅ 구현+검증 완료, **학습은 타 서버에서 진행 예정** | `comparison/crat-pred/`(원본), `comparison/cratpred_tsem/`(어댑터) |
| SIMPL (RA-L 2024) | ✅ 구현+검증 완료, **학습은 타 서버에서 진행 예정** | `comparison/SIMPL/`(원본), `comparison/simpl_tsem/`(어댑터) |
| Forecast-MAE (ICCV 2023) | ✅ 구현+검증 완료, **학습은 타 서버에서 진행 예정** | `comparison/forecast-mae/`(원본), `comparison/forecastmae_tsem/`(어댑터) |

전체 진행상황·설계 배경은 `docs/TSEM_journal_design.html` §"🆚 비교 연구" 탭에도 동일하게 기록돼 있다.

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

**다음 단계**: 전체 300-epoch(early stop patience 50) 학습은 이 서버가 아닌 다른 서버에서 진행 예정 —
`comparison/hivt_tsem/config.yaml` + `train_hivt_tsem.py`를 그대로 복사해 실행하면 된다
(데이터 경로 `/home/oem/data/TII_data/`가 타 서버에도 동일해야 캐시 재사용 가능, 없으면 최초 1회
캐시 빌드 필요). 기준선은 10차-2(acc 81.91% / F1 81.95%).

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

**다음 단계**: HiVT와 동일하게 전체 300-epoch(patience 50) 학습은 타 서버에서 진행 예정 —
`comparison/qcnet_tsem/config.yaml` + `train_qcnet_tsem.py`를 그대로 복사해 실행.

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
