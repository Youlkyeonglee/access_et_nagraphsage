# ET-NAGraphSAGE 다중 서버 실험 협업 보드

> **이 파일이 두 서버 간 실험 분담의 단일 기준(single source of truth)이다.**
> GitHub 웹에서 바로 읽고, 각 서버는 pull → 배정 실험 실행 → 결과 파일 push 순으로 협업한다.

- 저장소: `https://github.com/Youlkyeonglee/access_et_nagraphsage.git`
- 데이터: 두 서버에 동일 존재 (경로만 서버별 `configs/*.yaml`의 `data_dir` 확인)
- 최신 갱신: 2026-07-01 (Server A)

---

## 0. 서버 역할

| 서버 | 별칭 | 담당 | 제약 |
|---|---|---|---|
| 원본 (현재) | **Server A** | Mamba 계열 전부 + 보드/문서 관리 | — |
| 물리 다른 서버 | **Server B** | 비-Mamba 실험 | ⚠️ **Mamba 실험 금지** (mamba_ssm 컴파일 필요, Server A 전용) |

> **Mamba 학습은 Server A에서만 한다.** Server B는 GRU/LSTM/Transformer 및 baseline 재현만.

---

## 1. Server B 최초 셋업 (1회만)

```bash
# 1) 클론
git clone https://github.com/Youlkyeonglee/access_et_nagraphsage.git
cd access_et_nagraphsage

# 2) conda 환경 (Server A와 동일 이름 권장)
conda activate tna_research   # 없으면 requirements 설치

# 3) 데이터 경로 확인 — Server B의 실제 경로로 맞출 것
#    configs/*.yaml 의 data.data_dir 가 Server B 경로와 같은지 확인
grep data_dir configs/et_nagraphsage.yaml
#    다르면 각 config의 data_dir 수정 (커밋하지 말 것 — 로컬 경로는 서버마다 다름)

# 4) 데이터로더 동작 검증
python scripts/verify_dataloader.py
```

주의: `checkpoints/`, `logs/`, `data/`, `mamba/`는 `.gitignore`로 **동기화 안 됨**(각 서버 로컬). 코드·config·문서만 공유된다.

---

## 2. 협업 프로토콜 (충돌 방지)

```bash
# 실험 시작 전 — 항상 먼저 pull
git pull origin main

# 결과는 "서버별 개별 파일"에만 기록 → 절대 충돌 안 남
#   Server A → results/server_A.md
#   Server B → results/server_B.md
# (RUN_QUEUE.md 의 '상태' 칸은 Server A만 갱신)

# 결과 push
git add results/server_B.md
git commit -m "result: <실험명> by Server B"
git push origin main
```

**규칙**
1. **코드/config 충돌 방지**: Server B는 새 실험이 필요하면 `configs/`에 **새 파일**을 만든다(기존 파일 수정 금지).
2. **결과 기록**: 각자 `results/server_X.md`에만 append (상대 파일 건드리지 않음).
3. **상태 칸 갱신**: 이 `RUN_QUEUE.md`의 상태는 Server A가 종합 갱신. Server B는 결과 파일로 보고.
4. push 전 반드시 `git pull --rebase origin main`.

---

## 3. 실험 대기 큐

우선순위: ★★★ = 논문 필수, ★★ = 중요, ★ = 보강

### Server B 배정 (비-Mamba)

| # | 우선 | 실험 | 상태 |
|---|---|---|---|
| B1 | ★★★ | node-only temporal (Ablation C) — 3.1 | ⬜ 대기 |
| B2 | ★★★ | edge-only temporal (Ablation C) — 3.2 | ⬜ 대기 |
| B3 | ★★ | 이웃 정책 count (Ablation E) — 3.3 | ⬜ 대기 |
| B4 | ★★ | 이웃 정책 radius (Ablation E) — 3.4 | ⬜ 대기 |
| B5 | ★ | Transformer 인코더 (Ablation A) — 3.5 | ⬜ 대기 |
| B6 | ★ | T=5 (Ablation B) — 3.6 | ⬜ 대기 |

> B1~B4는 코드 구현 완료(2026-07-01, Server A). `--temporal_target`, `--neighbor_mode` CLI로 제어.
> **기준 비교값**: 2-hop base(both/hybrid) = 94.15%(150ep). C/E는 이 값과 비교한다.

### Server A 배정 (Mamba + 진행중)

| # | 우선 | 실험 | 상태 |
|---|---|---|---|
| A1 | — | 2-hop+SupCon 500ep | 🟡 진행중 |
| A2 | — | K1=10 500ep (대조군) | 🟡 진행중 |
| A3 | — | K1=10+SupCon 500ep (본命) | 🟡 진행중 |
| A4 | ★ | Mamba + 2-hop (Ablation A, Mamba 전용) | ⬜ 대기 |
| A5 | ★★ | 최종모델 × seed 1~4 (유의성) | ⬜ 대기 |

---

## 3.1 ~ 3.6 상세 명령 (Server B)

모두 `configs/et_nagraphsage_2hop_base_ep500.yaml` (2-hop, 500ep, patience150) 기준.
GPU 번호(`CUDA_VISIBLE_DEVICES`)는 Server B의 빈 GPU로 조정할 것.
**실행 전 `git pull origin main` 필수.** 데이터 경로가 다르면 config의 `data_dir` 로컬 수정.

**B1 — node-only temporal (엣지 시계열 제거)** ★★★
```bash
git pull origin main
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/et_nagraphsage_2hop_base_ep500.yaml \
  --temporal_target node \
  --experiment et_nag_C_nodeonly_ep500 \
  > /tmp/C_nodeonly.log 2>&1 &
```
목적: 엣지 시계열의 기여 격리. **node+edge(94.15%)보다 낮아야** "엣지 시계열이 필요하다"가 증명됨.

**B2 — edge-only temporal (노드 시계열 제거)** ★★★
```bash
git pull origin main
CUDA_VISIBLE_DEVICES=1 python train.py \
  --config configs/et_nagraphsage_2hop_base_ep500.yaml \
  --temporal_target edge \
  --experiment et_nag_C_edgeonly_ep500 \
  > /tmp/C_edgeonly.log 2>&1 &
```
목적: 노드 시계열의 기여 격리.

**B3 — 이웃 정책 count (반경 무관 최근접 K대)** ★★
```bash
git pull origin main
CUDA_VISIBLE_DEVICES=2 python train.py \
  --config configs/et_nagraphsage_2hop_base_ep500.yaml \
  --neighbor_mode count \
  --experiment et_nag_E_count_ep500 \
  > /tmp/E_count.log 2>&1 &
```
목적: NAGraphSAGE 최고기록이 count 방식이었음 → 공정 비교 confound 제거.

**B4 — 이웃 정책 radius (반경 내 전부)** ★★
```bash
git pull origin main
CUDA_VISIBLE_DEVICES=3 python train.py \
  --config configs/et_nagraphsage_2hop_base_ep500.yaml \
  --neighbor_mode radius \
  --experiment et_nag_E_radius_ep500 \
  > /tmp/E_radius.log 2>&1 &
```

**B5 — Transformer 인코더** ★ ⛔ **아직 실행 불가**
> `temporal_encoder.py`는 현재 gru/lstm/mamba만 지원. transformer 미구현.
> Server A가 구현·커밋 후 아래 명령 사용 가능. (그때까지 B1~B4, B6 먼저)
```bash
# (구현 후) git pull origin main
# CUDA_VISIBLE_DEVICES=0 python train.py \
#   --config configs/et_nagraphsage_2hop_base_ep500.yaml \
#   --encoder_type transformer \
#   --experiment et_nag_A_transformer_ep500 > /tmp/A_transformer.log 2>&1 &
```

**B6 — T=5** ★
```bash
git pull origin main
CUDA_VISIBLE_DEVICES=1 python train.py \
  --config configs/et_nagraphsage_2hop_base_ep500.yaml \
  --T 5 \
  --experiment et_nag_B_T5_ep500 \
  > /tmp/B_T5.log 2>&1 &
```

> 결과 확인: 각 로그 끝의 `Test Acc (State_Acc)` 및 클래스별 정확도를 `results/server_B.md`에 기록.

---

## 4. 결과 보고 양식 (results/server_B.md 에 append)

```markdown
### <실험명> (2026-07-XX)
- config: configs/xxx.yaml
- Best Val: 0.XXXX (epoch NN)
- Test Acc: 0.XXXX
- Stop / LaneChange / Normal: 0.XX / 0.XX / 0.XX
- 비고:
```
