# 학교 서버 (SCHOOL) 실험 결과 로그

> 학교 서버(비-Mamba) 결과를 아래에 append. Mamba 실험은 하지 말 것.
> 실험 ID는 RUN_QUEUE.md 기준 (C-node, C-edge, E-count, E-radius, B-T5).

## 진행중 (192.168.1.11 서버, 2026-07-02 01:29 KST 기준)

> 공통: config `configs/et_nagraphsage_2hop_base_ep500.yaml` (data_dir 로컬 `/home/oem/yklee/data/`),
> 조립 텐서 **디스크 캐시** 적용(`cache/…_323b38d0ca`) → CSV/그래프 계산 스킵, 4실험 캐시 공유.
> GPU 4장 동시 실행(C-node/C-edge/D-h192/D-h256).

### C-node — 학습중 🟡 (2026-07-02 01:28 KST 재시작)
- Ablation: C 시계열대상 / 변형: node-only (엣지 시계열 제거)
- 서버: 192.168.1.11, GPU 0 | 파라미터 172,549
- 명령: `CUDA_VISIBLE_DEVICES=0 python train.py --config configs/et_nagraphsage_2hop_base_ep500.yaml --temporal_target node --experiment C-node`
- 결과 대기중

### C-edge — 학습중 🟡 (2026-07-02 01:28 KST 재시작)
- Ablation: C 시계열대상 / 변형: edge-only (노드 시계열 제거)
- 서버: 192.168.1.11, GPU 1 | 파라미터 172,549
- 명령: `CUDA_VISIBLE_DEVICES=1 python train.py --config configs/et_nagraphsage_2hop_base_ep500.yaml --temporal_target edge --experiment C-edge`
- 결과 대기중

### D-h192 — 학습중 🟡 (2026-07-02 01:29 KST 시작)
- Ablation: D 채널폭 / 변형: hidden_dim=192 (용량 확대)
- 서버: 192.168.1.11, GPU 2 | 파라미터 367,493
- 명령: `CUDA_VISIBLE_DEVICES=2 python train.py --config configs/et_nagraphsage_2hop_base_ep500.yaml --hidden_dim 192 --experiment D-h192`
- 결과 대기중

### D-h256 — 학습중 🟡 (2026-07-02 01:29 KST 시작)
- Ablation: D 채널폭 / 변형: hidden_dim=256 (용량 확대)
- 서버: 192.168.1.11, GPU 3 | 파라미터 636,165
- 명령: `CUDA_VISIBLE_DEVICES=3 python train.py --config configs/et_nagraphsage_2hop_base_ep500.yaml --hidden_dim 256 --experiment D-h256`
- 결과 대기중
