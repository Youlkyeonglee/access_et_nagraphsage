# 가상환경
conda activate tna_research

# 데이터 경로
- 연구실(로컬) 서버: `/home/oem/data/TII_data/` (Gongeoptap, DRIFT)
- openDD (로터리 7곳, rounD 대체 외부검증 후보): `/home/oem/data/TII_data/opendd_dataset/` — 데이터 분석 결과는 `docs/TSEM_journal_design.html` "데이터 설계 > openDD" 참조, 학습 미착수
- 학교 서버: `/home/oem/yklee/data/`
- ⚠️ `configs/*.yaml`의 `data_dir`는 서버마다 다름 — **커밋/pull 시 이 필드 충돌 주의** (b9be26d에서 실제 발생)
- 172.30.1.84 서버 접속 정보는 비공개 로컬 파일로 관리 — 저장소(GitHub 동기화 대상)에는 기록하지 않음.

# userEmail
yklee00815@gmail.com

# currentDate
2026-07-11

---

# 연구 목표 — TSEM-SAGE (저널 트랙, IEEE Access)

> 기존 ET-NAGraphSAGE(현재 상태 **탐지**)는 컨퍼런스(IWIS2026) 트랙으로 분리됨.
> 저널 트랙은 **미래 상태 예측(anticipation)**: 입력 과거 W프레임 → 정답 `state(t+H)`.

## 핵심 과제 정의
```
입력 X(t): kinematics[t-W+1 … t] + 이웃 (KNN @ t, lane_id는 입력에 없음)
정답 y(t): instant_state(t+H)   — speed·lane_id로 오프라인 계산
클래스:   0=stop, 1=lane_change, 2=normal
Baseline: Persist (y_persist = state(t))
기준설정: W=10, H=10 (1초 뒤 예측)
```

## 라벨 정의 (확정, 변경 금지 — 비교성 유지)
- **lane_change**: window 정의 — "(t, t+H] 내 발생 여부" (Jain et al. ICCV 2015 표준). point 정의는 폐기됨.
- **stop**: ±2 다수결 persistence (B안, `_TsemFileData._stop_persist_state()`). δ=2 유지, δ3/4는 "라벨 강건성" ablation으로만 게재.
- 구현: `modules/tsem_instant_label.py`, `modules/data_manager_tsem.py`

## 최종 모델 (확정)
- **TSEMSAGE 10D** = semantic 8채널(v,a,j,ω,d_lat,κ,Δρ,접선) + raw position 2채널 (`raw_append='position'`)
- Config: `configs/tsem_sage_10th_2.yaml`
- **최종 수치: accuracy 81.97±0.14 / macro-F1 82.03±0.14 (4-seed 41-44)** — std 0.14가 ablation 유의성 기준
- 인코더: **GRU 채택 확정** (성능·안정성·수렴속도 3축. LSTM 동급 −0.15%p, Mamba는 fp16 발산 5회·fp32 필수·−0.5%p)
- **핵심 novelty 용어: "구조 참조 위치(structure-referenced position)"** — polar 11D([ρ,sinθ,cosθ] augment) 81.82가
  raw 절대좌표(81.91)를 무손실 대체, 이식 가능. 논문 서사 = **2-트랙**: site-specific(10D) / transferable(semantic 8D 또는 polar 11D)

## 아키텍처 (4 Stage)
```
STAGE A: SemanticDerivation — raw 6D → semantic 8D (Δρ·접선은 forward 시점 계산, 캐시는 raw만)
         로터리 중심 C=(72.86,-13.45) world 고정상수. + raw_append(position/polar) 채널
STAGE B: Temporal encoder — DecompTemporalEncoder(GRU), W=10 프레임
STAGE C: Spatial — ETSAGELayer (edge-aware message passing, NAGraphSAGE 계승. 기여 +1.25%p)
STAGE D: 분류 헤드 + 보조헤드(classifier_temporal/classifier_spatial, opt-in)
```

## 손실 함수
```
L = FocalLoss(logits,y)·unc_weight + λ_T·L_aux_temporal + λ_S·L_aux_spatial + λ_KL·KL(softmax‖Uniform)
λ_T=λ_S=0.3, λ_KL=0.1, class_weight_power=0.7, uncertainty weight=Beta-Binomial (§12.12)
```

## 학습 필수 설정 (경험적 확정)
- **Augmentation 1+2+3** (noise_std=1.0, neighbor/frame dropout 0.1) — 없으면 ep90 이후 과적합 붕괴. 회전(4)은 로터리 중심 기준 회전으로 수정됨, rotate_deg=0 미사용.
- **early_stop_patience: 50**, 종료 후 `best.pt` 재로드해 test 평가 (마지막 epoch 평가 버그 수정됨)
- Mamba 실험 시 **fp32 필수** (`--no_amp`) — AMP fp16에서 selective scan forward overflow
- `nohup` 실행 시 `python -u` 필수 (stdout 버퍼링)

## 확정된 핵심 발견 (재실험 불필요)
1. **위치 암기 가설 확증(2026-07-12 정량 검증 추가)**: raw 절대좌표 +3.74%p는 로터리 단일 지점 암기. 교차 장소(DRIFT) 평가에서 position 모델 붕괴(LC recall 80%→0.3%). semantic은 상대적 장소 불변(단 Δρ·접선 2채널은 site-specific, §아래). 정량 검증 2건 완료: ①정적 좌표 격자 룩업(재학습 없음)은 60~64%대(Persist 수준)에 그쳐 신경망의 81.91%와 17~20%p 격차 — 단순 암기표로는 설명 안 됨. ④10D 체크포인트 추론 시 raw position 채널 고정 시 -17.36%p 하락(순수 운동학 채널은 거의 0) — semantic 속 `tangent`(site-specific) 단독 고정만으로도 -16.69%p로 거의 동급, 위치 정보가 raw+semantic 여러 채널에 분산 저장됨을 확인. 종합: 신경망이 얻는 초과성능(81.91%−64%=+18%p)이 정확히 위치 의존 몫과 일치 — "이 로터리에만 유효한 정교한 위치 활용법"이라는 결론. ②position-only 신경망(순수 좌표만으로 신경망이 어디까지 가는지) 학습 진행 중, 상세는 `docs/TSEM_journal_design.html` §데이터 설계 "위치 암기 검증①·④ 종합" 참조.
2. **W×H 3×3 완전 단조**: W10>W20>W30(관측창 포화), H5>H10>H15(0.5초당 −2.8%p) → "W=10 충분, 성능은 H가 지배"
3. **edge-temporal은 공업탑에서 구조적 무용** (정보중복·라벨 이웃무관·ego지배) — DRIFT에서만 성립(+3.6%p, p=0.0002). 장기지평/DRIFT에서만 edge 부활.
4. **CSV acceleration은 부호 없는 |a|** — 감속 방향 정보 없음. a:=Δv 재정의는 미실행 ablation 후보.
5. **남은 난제 = "1초 뒤 출발" anticipation** (출발임박 normal recall 19.5%). speed-reg 보조헤드는 방향 적중이나 효과 소폭(19.5→22%) → Discussion 소재. 남은 후보: soft label, 이웃 출발 전파.

## 서버 분담
- **로컬(이 컴퓨터)**: Mamba 계열 전부 + 문서/보드 관리
- **학교 서버**: 비-Mamba (mamba_ssm 미설치). 분담 보드 = `RUN_QUEUE.md` (single source of truth)
- 진행 중(학교): radius 10/30, aug 단독, noaug

---

# 하네스 엔지니어링 원칙 (에이전트 작업 규약)

이 리포지터리는 에이전트-중심으로 운영한다. 핵심: **리포지터리 밖 지식은 존재하지 않는 것과 같다.**

## 1. 기록 시스템 (System of Record)
모든 의사결정·실험 결과·설계 변경은 버전 관리되는 파일에 기록 후 작업 종료:
- **설계·의사결정 로그**: `docs/TSEM_journal_design.html` — §12.x에 실험별 상세 기록, "실험 현황" 탭(page-experiments)이 전체 실험 축별 상태 마스터 표. **코드 변경 시 반드시 함께 갱신** (Doc-Gardening).
- **구현 레지스트리**: `docs/TSEM_implementation_registry.md` — 신규/재사용/기각 파일 목록. 새 파일 추가 시 갱신.
- **실험 분담 보드**: `RUN_QUEUE.md` — 두 서버 협업의 단일 기준. pull → 실행 → 결과 push.
- **결과 수치**: `results/` (예: `20260709school.md`) + 체크포인트별 `results.json`.
- 이 CLAUDE.md는 **나침반**이다 — 상세 지침을 담지 않고 위 문서로 안내한다. 목표·아키텍처가 바뀌면 이 파일부터 갱신.

## 2. 실험 격리 (재현성)
- **실험당 전용 config 분리** (`configs/tsem_sage_N차.yaml`) — 공유 config를 플래그로 껐다 켜지 않음.
- 캐시 키 분리: 라벨 정의 변경 시 캐시 키에 반영 (예: `_sd{δ}`). 캐시는 raw만 저장, 파생 채널은 forward 계산 → 캐시 재생성 최소화.
- 경로 분리: 캐시 `cache/tsem/`, 체크포인트 `checkpoints/tsem/{실험명}/`, 로그 `logs/tsem/`.
- 새 기능은 **opt-in 기본값**(기존 실험 동작 불변) + CLI 배선 (`--W --H --seed --radius --stop_delta --raw_append --speed_reg --no_amp --grad_clip --lr`).

## 3. 검증 루프 (자율 품질 관리)
- 새 라벨/피처/증강 구현 시: 스모크 스크립트(`scripts/tsem_verify_dataloader.py`)로 분포·shape 검증 → 짧은 학습 → 본 실험.
- 결과 보고는 항상 **각자 라벨 기준 Persist baseline과 비교** (라벨이 다르면 accuracy 직접비교 금지).
- 유의성 판단 기준: 4-seed std 0.14 (이보다 작은 차이는 노이즈로 판정).
- 성능 주장 전 과적합 확인: val 곡선이 peak 후 하락하는지, best.pt 기준 test인지.

## 4. 수정 금지 영역
- NAGraphSAGE 원본 (`/home/oem/graph_vehicle_v1/`) — 읽기 전용 참조.
- 확정 라벨 정의(LC window, stop ±2) — 변경은 비교성 파괴, ablation으로만.
- 기존 ET-NAGraphSAGE 계열 코드(`train.py`, `modules/data_manager.py`, `models/et_nagraphsage.py` 호출부) — TSEM은 병행 파일로 구현, 기본값으로 하위호환 유지.
  - **적용 패턴(실례, 2026-07-13)**: edge feature에 bearing/Δheading을 추가할 때 `modules/data_manager.py::_compute_edge_feat`/`_recompute_edges`/`EDGE_DIM`(컨퍼런스 트랙과 공유)은 그대로 두고, `modules/data_manager_tsem.py`에 `_recompute_edges_bearing`(신규 함수, `EDGE_DIM_BEARING=7`)을 병행 구현 + `edge_feat_variant='legacy'|'bearing'` opt-in 플래그로 선택하게 함(기본값 `'legacy'`가 기존 동작과 100% 동일). 앞으로 공용 파일의 핵심 계산을 바꾸고 싶을 때는 이 방식(새 함수를 TSEM 전용 파일에 추가 + opt-in 플래그, 기본값은 항상 기존 동작 유지)을 기본으로 따를 것 — 공용 파일을 직접 고치면 이미 학습된 체크포인트·재현 결과가 깨질 수 있다.

## 5. 커밋 → 완료 처리
- 커밋 메시지는 Conventional Commits 형식: `feat(scope): 설명`

---

## 저장소 경로
- **TSEM/T-NAGraphSAGE 저장소 (현재)**: `/home/oem/TNA_research/`
- NAGraphSAGE 원본 (수정 금지): `/home/oem/graph_vehicle_v1/`
- M2MambaV2 (별도): `/home/oem/access2/`
- 원격 학교 서버: 172.30.1.84 (GitHub `access_et_nagraphsage.git` 경유 협업)

## 핵심 코드 (TSEM)
- 모델: `models/tsem_sage.py` (TSEMSAGE / TSEMSemanticOnly / TSEMNAGraphSAGEAdapted)
- Semantic 파생: `models/tsem_semantic_derivation.py`
- 라벨: `modules/tsem_instant_label.py` / 데이터: `modules/data_manager_tsem.py` / 증강: `modules/tsem_augment.py`
- 학습: `train_tsem.py` / 평가: `modules/tsem_eval.py`
- 분석: `journal/` (cross_location_eval, stop_error_analysis, speedreg_subgroup_eval, qualitative_cases 등)
- 논문 그림: `journal/paper_figs/`

## 베이스라인 수치 (참고, 컨퍼런스 트랙)
| 트랙 | 모델 | 지표 |
|---|---|---|
| 저널(예측) | **TSEM-SAGE 10D (최종)** | **81.97±0.14 acc / 82.03±0.14 F1** |
| 저널(예측) | Persist baseline (window+B안 라벨) | 각 실험 results.json 참조 |
| 컨퍼런스(탐지) | NAGraphSAGE (World, best) | 94.54±1.03 acc |
| 컨퍼런스(탐지) | Flagship D-h192 fp32 4-seed | 95.37±0.45 acc (p=0.0042) |
