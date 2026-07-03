# 학교 서버 (SCHOOL) 실험 결과 로그

> 학교 서버(비-Mamba) 결과를 아래에 append. Mamba 실험은 하지 말 것.
> 실험 ID는 RUN_QUEUE.md 기준 (C-node, C-edge, E-count, E-radius, B-T5).

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
