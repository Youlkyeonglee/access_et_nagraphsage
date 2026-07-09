# ET-NAGraphSAGE-MR — 다중관계 엣지 수정 기록 (Step 1)

> 목적: GraphSAGE 약점 "모든 엣지를 같은 타입으로 취급"을 R-GCN식 **관계타입별 메시지**로 보완.
> 규칙: **기존 코드 불변.** 새 파일에만 작성.
> 작성일: 2026-07-05

## 신규 파일 (기존 파일 0개 수정)
| 파일 | 역할 |
|---|---|
| `models/et_nagraphsage_mr.py` | ETNAGraphSAGEMR + ETSAGELayerMR + relation_type() |
| `train_mr.py` | 학습 스크립트 (train.py 헬퍼 재사용, 모델만 교체) |
| `docs/mr_edge_modification.md` | 본 기록 |

## 무엇을 바꿨나 — C2(메시지 함수) 단 하나

기존 `ETSAGELayer`와 **유일한 구조 차이는 C2**. C1(게이트)·C3(β어텐션)·업데이트·temporal encoder·2-hop 파이프라인은 100% 동일.

| 단계 | 기존 (et_nagraphsage.py) | 신규 (et_nagraphsage_mr.py) |
|---|---|---|
| C1 게이트 | `α=σ(W_α·e)` | 동일 |
| **C2 메시지** | `m=ReLU(Linear(cat(h_nbr·α, e)))` — **단일 weight** | `m=ReLU(Linear_r(cat(h_nbr·α, e)))` — **관계타입 r별 weight** |
| C3 집계 | `β=softmax(w_β·e)`, `h_N=Σβ·m` | 동일 |
| 업데이트 | `ReLU(Linear(cat(h_ego,h_N)))` | 동일 |

구현: `lin_msg`를 `Linear(in+d_e, R*out)`로 만들어 R개 메시지를 한 번에 계산 후,
엣지의 관계타입으로 `gather`하여 선택. (R=관계 수)

## 관계타입 r 정의 (relation_type)

마지막 프레임 t의 **ego heading 기준 상대위치**로 각 엣지를 분류:
```
h    = ego_dir / |ego_dir|            (heading 단위벡터)
perp = (-h_z, h_x)                    (좌수직)
rel  = nbr_pos - ego_pos
lon  = rel·h    (전방+ / 후방-)
lat  = rel·perp (좌- / 우+)
```
| R | 타입 |
|---|---|
| 4 (기본) | 0 ahead(\|lat\|≤w, lon≥0) · 1 behind(\|lat\|≤w, lon<0) · 2 left(lat<-w) · 3 right(lat≥w) |
| 2 | 0 ahead(lon≥0) · 1 behind(lon<0) |
| 6 | 앞/뒤 각각 좌·우 세분 (확장 여지) |

- `lat_w` = 좌우 판정 임계(기본 2.5m ≈ 차선폭). 1-hop·2-hop 모두 동일 로직 적용.
- **주의**: raw 좌표 기하 의존 → **공업탑 base(비표준화)에서 정확**. DRIFT(ego_relative+z-score)는 좌표 왜곡되어 in-model 타이핑 부적합 → 별도 처리 필요(향후).

## 파라미터 영향
| R | 파라미터 | vs 기존(172,549) |
|---|---|---|
| 2 | 213,765 | +24% |
| 4 | 296,197 | +72% |
| 6 | 378,629 | +119% |
- 증가분은 C2의 관계별 메시지 weight(R배). d_e·hidden은 동일.

## 검증 실험 (진행 중)
- **MR-gg-both-R4** (GPU2): 공업탑 base(h128), R=4, temporal_target=both, seed42
- **Ctrl-gg-both** (GPU3): 동일 조건 **plain 모델**(비-MR) — 공정 비교 대조군 (+ school.md의 'both 미완' 채움)
- config `et_nagraphsage_2hop_base_ep500.yaml`, data `/home/oem/data/TII_data/`, 500ep/cosine/patience150.
- 판정: **MR-both vs Ctrl-both** State_Acc·LaneChange 비교 → 관계타입 분리가 기여하는가.
  - 특히 **LC↔Normal 경계**(school 혼동행렬 병목) 개선 여부 주목.

## 단계 로드맵
- **Step 1 (현재)**: C2 관계타입별 메시지 (기하 4타입).
- Step 2: 관계별 β어텐션(C3도 타입별) or 학습형 soft 타이핑.
- Step 3: motion decomposition 피처와 결합 (분해된 node/edge를 관계별 메시지에).
- Step 4: DRIFT용 — raw좌표 관계타입을 데이터 파이프라인에서 산출(표준화 무관).
