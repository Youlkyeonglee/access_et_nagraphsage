# TSEM 구현 파일 레지스트리

**작성:** 2026-07-07  
**목적:** ET-NAGraphSAGE 선행 코드(`train.py`, `data_manager.py`, `et_nagraphsage.py` 등)와 **파일명·경로로 명확히 구분**된 TSEM 저널 연구 구현 목록.

설계 문서: [TSEM_journal_design.html](./TSEM_journal_design.html)

---

## 진행 상황

### 1단계 — dataloader 스모크 테스트 ✅ (2026-07-07)

**명령:** `scripts/tsem_verify_dataloader.py --W 10 --H 10`

| 항목 | 결과 |
|------|------|
| CSV | 11개 (Gongeoptap) |
| `instant_state` 규칙 | OK |
| 순간 라벨 분포 (파일 1개, train 샘플 기준) | stop **35.8%**, lane_change **2.2%**, normal **61.9%** |
| 샘플 수 (W=10, H=10) | train **575,686** / val **119,901** / test **120,435** |
| batch shape | `node_seq` [B,10,6], `nbr_node_seqs` [B,6,10,6], `y`·`y_persist` [B] — 정상 |

**해석:** §1.2 순간 LC 정의로 LC 비율이 ±6 smear(~20%)보다 **훨씬 희귀**함을 데이터에서 확인. 학습 시 class weight / macro-F1 병기 필요.

### 2단계 — (다음) 짧은 학습 스모크 / Persist 하한선

- [ ] `train_tsem.py` 1~5 epoch (또는 소배치) — loss·val acc 수렴 확인
- [ ] test에서 **Persist** (`y_persist`) vs TSEM 비교

---

## 1. 신규 파일 (TSEM 전용)

| 경로 | 역할 | 기존 코드와 관계 |
|------|------|------------------|
| `modules/tsem_instant_label.py` | 순간 `state(f)` 규칙 (stop/LC/normal). ±6·CSV `category` 미사용 | `road_data/20250318json_class_lanechage.py` **대체 안 함** (기각 참고만) |
| `modules/data_manager_tsem.py` | `y=state(t+H)`, 과거 W, 이웃 KNN @ t. 캐시 `cache/tsem/` | `modules/data_manager.py` **병행** (기존 `y=category(t)` 유지) |
| `modules/tsem_eval.py` | Future State Acc, per-class recall, macro-F1, Persist 평가 | `train.py`의 `evaluate` **대체 안 함** |
| `models/tsem_semantic_derivation.py` | Stage A: raw 6D → semantic 6D | `journal/anticipation_features.py`와 별도 (world §5.3) |
| `models/tsem_sage.py` | TSEM-SAGE / TSEMSemanticOnly | `models/et_nagraphsage.py` **import만** (`ETSAGELayer`) |
| `train_tsem.py` | TSEM 학습 엔트리 | `train.py` **병행** |
| `configs/tsem_sage.yaml` | W, H, 모델·학습 설정 | `configs/et_nagraphsage*.yaml` **병행** |
| `scripts/tsem_verify_dataloader.py` | 데이터·라벨 스모크 테스트 | `scripts/verify_dataloader.py` **병행** |

---

## 2. 재사용 (수정 없음)

| 경로 | TSEM에서 쓰는 부분 |
|------|-------------------|
| `models/decomp_encoder.py` | Stage B `DecompTemporalEncoder` |
| `models/et_nagraphsage.py` | `ETSAGELayer` (spatial @ t) |
| `models/temporal_encoder.py` | `DecompTemporalEncoder` 내부 |
| `modules/data_manager.py` | `_compute_edge_feat`, `_recompute_edges` 등 유틸 import |

---

## 3. 캐시·출력 경로 (기존과 분리)

| 항목 | TSEM | ET-NAGraphSAGE (기존) |
|------|------|------------------------|
| 샘플 캐시 | `cache/tsem/` | `cache/` |
| 체크포인트 | `checkpoints/tsem/` | `checkpoints/` |
| 로그 | `logs/tsem/` | `logs/` |

---

## 4. 실행 예시

```bash
# 1) dataloader·instant 라벨 확인
python scripts/tsem_verify_dataloader.py --W 10 --H 10

# 2) 학습
python train_tsem.py --config configs/tsem_sage.yaml

# 3) semantic-only ablation
python train_tsem.py --config configs/tsem_sage.yaml --model tsem_semantic_only
```

---

## 5. 데이터 요구사항

- CSV에 **`lane_id` 열 필수** (LC 순간 라벨용).
- **`category` 열은 정답에 사용하지 않음** (kinematics 행 필터에도 미사용).
- 기본 데이터 경로: `configs/tsem_sage.yaml` → `/home/oem/data/TII_data/Gongeoptap/*.csv`

---

## 6. 정답·평가 정의

```
입력 X(t): kinematics[t-W+1 … t]  (+ 이웃, lane_id 입력 없음)
정답 y(t): instant_state(t+H)     — speed·lane_id로 오프라인 계산
Persist:   y_persist = instant_state(t)
```

클래스: `0=stop`, `1=lane_change`, `2=normal`

---

## 7. 미구현 (다음 단계)

- [ ] `NAGraphSAGE-adapted` @ `state(t+H)` 베이스라인 (`train_tsem.py`에 모델 스위치 추가)
- [ ] Raw-Temporal+GNN 베이스라인
- [ ] W ∈ {10,20,30}, H ∈ {5,10,15} 스윕 스크립트
- [ ] 로터리용 Δρ semantic 채널 (`tsem_semantic_derivation.py` 확장)

---

## 8. 파일 트리 (TSEM만)

```
TNA_research/
├── configs/tsem_sage.yaml
├── train_tsem.py
├── modules/
│   ├── tsem_instant_label.py
│   ├── data_manager_tsem.py
│   └── tsem_eval.py
├── models/
│   ├── tsem_semantic_derivation.py
│   └── tsem_sage.py
├── scripts/tsem_verify_dataloader.py
├── docs/TSEM_implementation_registry.md   ← 이 문서
└── cache/tsem/                            ← 실행 시 생성
```
