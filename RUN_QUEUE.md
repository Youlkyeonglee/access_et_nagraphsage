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

| # | 우선 | 실험 | 명령 | 상태 |
|---|---|---|---|---|
| B1 | ★★★ | node-only temporal (Ablation C) | `아래 3.1 참조` | ⬜ 대기 |
| B2 | ★★★ | edge-only temporal (Ablation C) | `아래 3.2 참조` | ⬜ 대기 |
| B3 | ★★ | 이웃 정책 count (Ablation E) | `아래 3.3 참조` | ⬜ 대기 |
| B4 | ★★ | 이웃 정책 radius (Ablation E) | `아래 3.4 참조` | ⬜ 대기 |
| B5 | ★ | Transformer 인코더 (Ablation A) | `--encoder_type transformer` | ⬜ 대기 |
| B6 | ★ | T=5 (Ablation B) | `--T 5` | ⬜ 대기 |

### Server A 배정 (Mamba + 진행중)

| # | 우선 | 실험 | 상태 |
|---|---|---|---|
| A1 | — | 2-hop+SupCon 500ep | 🟡 진행중 |
| A2 | — | K1=10 500ep (대조군) | 🟡 진행중 |
| A3 | — | K1=10+SupCon 500ep (본命) | 🟡 진행중 |
| A4 | ★ | Mamba + 2-hop (Ablation A, Mamba 전용) | ⬜ 대기 |
| A5 | ★★ | 최종모델 × seed 1~4 (유의성) | ⬜ 대기 |

---

## 3.1 ~ 3.4 상세 명령 (Server B)

> ⚠️ 아래 실험들은 **Ablation C/E 코드가 아직 미구현**이다. Server A가 구현·커밋 후
> 이 섹션에 정확한 명령을 채운다. 그 전까지 B5(Transformer), B6(T=5)부터 진행 가능.

**B5 — Transformer 인코더 (지금 실행 가능)**
```bash
git pull origin main
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/et_nagraphsage_2hop_supcon_ep500.yaml \
  --encoder_type transformer \
  --experiment et_nagraphsage_2hop_transformer_ep500 \
  > /tmp/transformer_ep500.log 2>&1 &
```
※ `temporal_encoder.py`가 transformer 타입을 지원하는지 먼저 확인 (미지원 시 Server A가 추가).

**B6 — T=5 (지금 실행 가능)**
```bash
git pull origin main
CUDA_VISIBLE_DEVICES=1 python train_supcon.py \
  --config configs/et_nagraphsage_2hop_supcon_ep500.yaml \
  --T 5 \
  --experiment et_nagraphsage_2hop_supcon_T5_ep500 \
  > /tmp/supcon_T5.log 2>&1 &
```

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
