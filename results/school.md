# 학교 서버 (SCHOOL) 실험 결과 로그

> 학교 서버(비-Mamba) 결과를 아래에 append. Mamba 실험은 하지 말 것.
> 실험 ID는 RUN_QUEUE.md 기준 (C-node, C-edge, E-count, E-radius, B-T5).

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
