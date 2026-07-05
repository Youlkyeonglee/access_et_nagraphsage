# 학교 서버 (SCHOOL) 실험 결과 로그

> 학교 서버(비-Mamba) 결과를 아래에 append. Mamba 실험은 하지 말 것.
> 실험 ID는 RUN_QUEUE.md 기준 (C-node, C-edge, E-count, E-radius, B-T5).

## 진행중 — 시계열 ablation (flagship h192 fp32, seed 42, 2026-07-05 실행)

> 목적: Table II(시퀀스 길이) + Ablation A(인코더)를 **flagship 조건(h192, K6/4, fp32, 500ep)으로 일관 완성**.
> 기존 T=1/T=10, encoder 비교는 옛 h128이라 조건 불일치 → h192로 재실행. IWIS2026 + 저널 공용.
> 비교 기준: flagship GRU/T=10/h192 = **95.37±0.45**(seed42=95.33). 단일 시드 확인 후 4-seed 확장 예정.

| 실험 | 축 / 변형 | 캐시 | 명령(핵심 인자) | 상태 |
|---|---|---|---|---|
| A-lstm-h192 | 인코더=LSTM (T=10) | fp32 T10 재사용 | `--encoder_type lstm --hidden_dim 192 --batch_size 2048 --seed 42` | 🟡 학습중 |
| B-T1-h192 | 시퀀스 T=1 (no-temporal) | fp32 T1 신규빌드 | `--T 1 --hidden_dim 192 --batch_size 2048 --seed 42` | 🟡 학습중 |
| B-T5-h192 | 시퀀스 T=5 | fp32 T5 신규빌드 | `--T 5 --hidden_dim 192 --batch_size 2048 --seed 42` | 🟡 학습중 |

- 공통 config `et_nagraphsage_2hop_base_ep500.yaml`, fp32 data_manager(f19d50e)로 실행 후 HEAD 복원.
- **Mamba(Ablation A)**: 이 서버 `mamba_ssm` 미설치 + "학교 서버 Mamba 금지" 규칙 → **미실행(별도 트랙 처리 대기)**.
- 결과 대기중 → 완주 시 Test State_Acc로 갱신.

---

## Tier 1-2 성능 향상 실험 (2026-07-05 완료)

### ① 4-seed 앙상블 (Tier 1, 무료) ✅
fp32 4-seed(42/846/862/995) softmax 평균 앙상블 (`ensemble_eval.py`):
- **Ensemble Test Acc = 95.74%** (Stop 99.83 / LC 86.18 / Normal 96.19)
- 4-seed avg(95.37) 대비 **+0.37%p**, LC·Normal 동반 상승. best-seed(95.81)엔 근소하게 못 미침(약한 s846=94.77이 평균 저하).
- → **추가 학습 없이 95.74% 확보. 현재 최고 단일 성능은 best-seed 95.81.**

### ② K1=10 × h192 결합 (Tier 1, 미검증 조합) ❌ 하락
채널폭(h192)과 receptive field(K1=10)를 결합. fp32 K10/6 캐시(27G) + `--K_max 10 --K_max2 6 --hidden_dim 192 --batch_size 1024`. `checkpoints/D-k10h192-fp32-s{42,846,862,995}/`.

| 시드 | Test | LC |
|---|---|---|
| s42/846/862/995 | 0.9451/0.9445/0.9453/0.9445 | ~0.820 |
| **avg±std** | **94.49 ± 0.04 %** | 82.00 ± 0.20 % |

- **결합 vs flagship h192-K6/4(95.37): −0.88%p | vs K10-h128(94.73): −0.24%p** — **두 축 단독보다 모두 낮음.**
- std 0.04로 매우 일관 → 노이즈 아닌 구조적 결과. **채널폭×receptive field는 additive가 아니라 간섭.** K10의 많은/먼 이웃이 h192 용량과 겹쳐 LC(85.7→82.0)를 특히 악화. → **sweet spot은 h192-K6/4 확정, "더 많이"가 아님.**

### 진단 — 혼동행렬 (Tier 2, 앙상블 기준)
```
정답\예측    Stop  LaneChange  Normal   recall
LaneChange   387    22888      3283     0.862   (→Normal 12.4%, →Stop 1.5%)
Normal        41     1982     51049     0.962   (→LC 3.8%)
```
- 병목은 **LC↔Normal 양방향 경계**. 단순 class weight/Focal은 tradeoff 한계(LC↑ ⇒ Normal→LC↑). 구조적 feature 분리 필요 → 저널 트랙 과제.

### 📌 현재 최고 성능 정리
| 구성 | State_Acc | 비고 |
|---|---|---|
| **앙상블 (h192-K6/4 4-seed)** | **95.74** | 최고, 무료 |
| best-seed (h192-K6/4 s862) | 95.81 | 단일 최고 |
| **flagship h192-K6/4 4-seed** | **95.37±0.45** | 공식·유의미 p=0.0042 |
| K10-h128 4-seed | 94.73±0.31 | |
| K10×h192 4-seed | 94.49±0.04 | 결합 하락(간섭) |

---

## ★★ Flagship D-h192 fp32 4-seed — 공식 수치 (2026-07-04 완료)

> config `et_nagraphsage_2hop_base_ep500.yaml` + `--hidden_dim 192 --batch_size 2048`, 500ep.
> **fp32 캐시**(`cache/…_323b38d0ca`) 사용 — f19d50e 코드로 실행(seed42 원본과 동일 조건).
> 체크포인트: `checkpoints/D-h192`(s42=원본) + `checkpoints/D-h192-fp32-s{846,862,995}/`.

| 시드 | Val | **Test(State_Acc)** | LaneChange |
|---|---|---|---|
| s42 (원본 D-h192) | 0.9554 | 0.9533 | 0.860 |
| s846 | 0.9515 | 0.9477 | 0.833 |
| s862 | 0.9590 | **0.9581** | 0.871 |
| s995 | 0.9589 | 0.9557 | 0.866 |
| **avg±std** | — | **95.37 ± 0.45 %** | **85.75 ± 1.79 %** |

- **🎯 유의성 확정**: NAGraphSAGE avg(94.07±0.28) 대비 **+1.30%p, Welch t=4.94, p=0.0042 → 유의미(p<0.05)**.
  NAGraphSAGE **best(94.54) 대비 +0.83%p**. best-seed **95.81%**. → **flagship headline = 95.37±0.45%**.
- 앞선 K1=10 500ep 4-seed(94.73±0.31) 대비도 +0.64%p 우위. **채널폭 h192가 최고 단일 지렛대.**

### ⚠️ 캐시 정밀도(fp32 vs f16ne) — 별도 발견
동일 4-seed를 f16ne(float16) 캐시로도 학습: **f16ne 94.71±0.10 vs fp32 95.37±0.45 = −0.66%p**.
동일 seed42: fp32 95.33 vs f16ne 94.84 (−0.49%p). `cudnn.deterministic=True`라 순수 float16 양자화 효과.
→ **공식 수치는 fp32**. f16ne는 속도/용량 이점이나 정확도 손실이 있어 최종 보고엔 미사용. (f16ne 체크포인트 `D-h192-s{42,846,862,995}`는 참고 보존.)

### 채널폭 sweet-spot 표 (base K6/4, 500ep, fp32)
| hidden_dim | Test(State_Acc) | 비고 |
|---|---|---|
| 128 (base) | 94.15 | 기존 |
| **192** | **95.37 ± 0.45 (4-seed)** 🥇 | sweet spot, best-seed 95.81 |
| 256 | 94.67 | 과용량 하락 시작 |
| 384 | 94.44 | 과용량 추가 하락 |
→ **h192가 sweet spot 확정**(192 최고, 256·384 하락). 용량 7배 격차 해소가 유효 축.

---

## 완료 (192.168.1.11 서버, 2026-07-02)

> 공통: config `configs/et_nagraphsage_2hop_base_ep500.yaml` (data_dir 로컬 `/home/oem/yklee/data/`),
> 조립 텐서 **디스크 캐시** 적용(`cache/…_323b38d0ca`) → CSV/그래프 계산 스킵, 4실험 캐시 공유(184s→32s/epoch).
> GPU 4장 동시 실행. 학습 종료 후 test 평가는 `eval_ckpt.py`로 사후 재계산(프로세스가 test 전 종료됨).
> **baseline NAGraphSAGE = 94.54%**

| 실험 | best ep | Val | **Test(State_Acc)** | Stop | LaneChange | Normal |
|---|---|---|---|---|---|---|
| **D-h192** 🥇 | 492 | 0.9554 | **0.9533** | 0.998 | 0.860 | 0.952 |
| D-h256 | 491 | 0.9504 | 0.9467 | 0.999 | 0.824 | 0.953 |
| C-node | 438 | 0.9482 | 0.9440 | 0.999 | 0.829 | 0.943 |
| C-edge | 355 | 0.9253 | 0.9252 | 0.998 | 0.742 | 0.940 |

**해석**
- **D-h192(95.33%)가 baseline 94.54% +0.79%p 상회** — 채널폭 192가 최적. h256은 과용량으로 하락.
- **C: node-only(94.40%) ≫ edge-only(92.52%)** — 노드 시계열이 주 신호. LaneChange 격차 큼(82.9 vs 74.2).
- ⚠️ **제안 both(node+edge) 결과 미완** — C ablation novelty 입증 위해 `--temporal_target both` 필수.

### 재현 명령 (배치 크기: C=4096, D-h192=2048, D-h256=1024 — D는 OOM 회피용 축소)
```
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/et_nagraphsage_2hop_base_ep500.yaml --temporal_target node --experiment C-node
CUDA_VISIBLE_DEVICES=1 python train.py --config configs/et_nagraphsage_2hop_base_ep500.yaml --temporal_target edge --experiment C-edge
CUDA_VISIBLE_DEVICES=2 python train.py --config configs/et_nagraphsage_2hop_base_ep500.yaml --hidden_dim 192 --batch_size 2048 --experiment D-h192
CUDA_VISIBLE_DEVICES=3 python train.py --config configs/et_nagraphsage_2hop_base_ep500.yaml --hidden_dim 256 --batch_size 1024 --experiment D-h256
```
- 사후 test 평가: `python eval_ckpt.py [실험명...]` → `checkpoints/<exp>/results.json` + `results/summary.csv`
