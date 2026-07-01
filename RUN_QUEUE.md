# ET-NAGraphSAGE 다중 서버 실험 협업 보드

> **이 파일이 두 서버 간 실험 분담의 단일 기준(single source of truth)이다.**
> GitHub 웹에서 바로 읽고, 각 서버는 pull → 배정 실험 실행 → 결과 파일 push 순으로 협업한다.

- 저장소: `https://github.com/Youlkyeonglee/access_et_nagraphsage.git`
- 데이터: 두 서버에 동일 존재 (경로만 서버별 `configs/*.yaml`의 `data_dir` 확인)
- 최신 갱신: 2026-07-01

---

## 용어 정리 (혼동 방지)

| 개념 | 표기 | 설명 |
|---|---|---|
| **Ablation 그룹** | A · B · C · D · E | 연구 축. **글자는 오직 여기에만 사용** |
| **실험 ID** | `그룹-변형` | 예: `C-node`, `E-count`. 이름만 봐도 어느 Ablation인지 앎 |
| **서버** | 로컬 / 학교 | 글자(A/B) 안 씀 → Ablation과 헷갈리지 않음 |

> ⚠️ 과거 표기 `Server A/B`, `B1~B4`는 **폐기**. Ablation B(시퀀스 길이)와 혼동되기 때문.

**Ablation 그룹 정의**
- **A** = Temporal Encoder 타입 (GRU/LSTM/Mamba/Transformer)
- **B** = 시퀀스 길이 T (1/5/10/15)
- **C** = 시계열 대상 (node/edge/node+edge) ★핵심
- **D** = 공간 구조 (hop 수·이웃 수·d_e)
- **E** = 이웃 선택 정책 (count/radius/hybrid) ★

---

## 0. 서버 역할

| 서버 | 위치 | 담당 | 제약 |
|---|---|---|---|
| **로컬 서버** (LOCAL) | 이 컴퓨터 (원본) | Mamba 계열 전부 + 문서/보드 관리 | — |
| **학교 서버** (SCHOOL) | 원격 | 비-Mamba 실험 | ⚠️ **Mamba 금지** (mamba_ssm 컴파일 필요, 로컬 전용) |

> **Mamba 학습은 로컬 서버에서만.** 학교 서버는 GRU/LSTM 및 baseline 재현만.

---

## 1. 학교 서버 최초 셋업 (1회만)

```bash
# 1) 클론
git clone https://github.com/Youlkyeonglee/access_et_nagraphsage.git
cd access_et_nagraphsage

# 2) conda 환경 (없으면 새로 생성)
conda create -n tna_research python=3.10 -y
conda activate tna_research
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128   # CUDA 버전 맞춰서
pip install -r requirements.txt
#   ※ 학교 서버는 Mamba 안 하므로 mamba-ssm 설치 불필요

# 3) 데이터 경로 확인 — 학교 서버의 실제 경로로 맞출 것
grep data_dir configs/et_nagraphsage.yaml
#    다르면 각 config의 data_dir 수정 (커밋하지 말 것 — 로컬 경로는 서버마다 다름)

# 4) 데이터로더 동작 검증
python scripts/verify_dataloader.py
```

주의: `checkpoints/`, `logs/`, `data/`, `mamba/`는 `.gitignore`로 **동기화 안 됨**(각 서버 로컬). 코드·config·문서만 공유.

---

## 2. 협업 프로토콜 (충돌 방지)

```bash
# 실험 시작 전 — 항상 먼저 pull
git pull origin main

# 결과는 "서버별 개별 파일"에만 기록 → 절대 충돌 안 남
#   로컬 서버 → results/local.md
#   학교 서버 → results/school.md

# 결과 push
git add results/school.md
git commit -m "result: <실험ID> (학교 서버)"
git push origin main
```

**규칙**
1. **config 충돌 방지**: 학교 서버가 새 실험이 필요하면 `configs/`에 **새 파일**을 만든다(기존 수정 금지).
2. **결과 기록**: 각자 `results/local.md` / `results/school.md`에만 append.
3. **상태 칸 갱신**: 이 보드의 상태는 로컬 서버가 종합 갱신. 학교 서버는 결과 파일로 보고.
4. push 전 반드시 `git pull --rebase origin main`.

---

## 3. 실험 대기 큐

우선순위: ★★★ = 논문 필수, ★★ = 중요, ★ = 보강

### 학교 서버 배정 (비-Mamba)

| 실험 ID | Ablation | 변형 | 우선 | 상태 | 명령 |
|---|---|---|---|---|---|
| `C-node`   | C 시계열대상 | node-only (엣지 시계열 제거) | ★★★ | ⬜ 대기 | §3.1 |
| `C-edge`   | C 시계열대상 | edge-only (노드 시계열 제거) | ★★★ | ⬜ 대기 | §3.2 |
| `D-h192` / `D-h256` / `D-h384` | D 채널폭 | hidden_dim 확대 (용량 7배 격차 해소) | ★★★ | ⬜ 대기 | §3.7 |
| `E-count-K6` / `E-radius-r20` | E 이웃정책 | 정책 비교 (핵심) | ★★ | ⬜ 대기 | §3.3 |
| `E-hybrid-r10/r30`, `E-count-K5/K10` | E 이웃정책 | 반경·K 민감도 | ★ | ⬜ 대기 | §3.3 |
| `B-T5`     | B 시퀀스길이 | T=5 | ★ | ⬜ 대기 | §3.5 |
| `A-transformer` | A 인코더 | Transformer | ★ | ⛔ 미구현 | §3.6 |

> Ablation E 전체 조합(9개, mode×r×K)은 문서 「Ablation 상세 → E」 페이지 표 참조.

> `C-*`, `E-*`는 코드 구현 완료(2026-07-01). `--temporal_target`, `--neighbor_mode` CLI로 제어.
> **기준 비교값**: 2-hop 기준(node+edge / hybrid) = 94.15%(150ep). C·E는 이 값과 비교한다.

### 로컬 서버 배정 (Mamba + 진행중)

| 실험 ID | Ablation | 변형 | 상태 |
|---|---|---|---|
| `D-supcon-ep500`   | D+SupCon | 2-hop+SupCon 500ep | 🟡 진행중 |
| `D-k10-ep500`      | D | K1=10 500ep (대조군) | 🟡 진행중 |
| `D-k10-supcon-ep500` | D+SupCon | K1=10+SupCon 500ep (본命) | 🟡 진행중 |
| `A-mamba`          | A 인코더 | Mamba + 2-hop (Mamba 전용) | ⬜ 대기 |
| `seed-final`       | 유의성 | 최종모델 × seed 1~4 | ⬜ 대기 |

---

## 3. 상세 명령 (학교 서버)

모두 `configs/et_nagraphsage_2hop_base_ep500.yaml` (2-hop, 500ep, patience150) 기준.
`CUDA_VISIBLE_DEVICES`는 학교 서버의 빈 GPU로 조정. **실행 전 `git pull origin main` 필수.**
데이터 경로가 다르면 config의 `data_dir` 로컬 수정.

### 3.1 `C-node` — node-only (엣지 시계열 제거) ★★★
```bash
git pull origin main
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/et_nagraphsage_2hop_base_ep500.yaml \
  --temporal_target node \
  --experiment C-node \
  > /tmp/C-node.log 2>&1 &
```
목적: 엣지 시계열의 기여 격리. **node+edge(94.15%)보다 낮아야** "엣지 시계열이 필요하다"가 증명됨.

### 3.2 `C-edge` — edge-only (노드 시계열 제거) ★★★
```bash
git pull origin main
CUDA_VISIBLE_DEVICES=1 python train.py \
  --config configs/et_nagraphsage_2hop_base_ep500.yaml \
  --temporal_target edge \
  --experiment C-edge \
  > /tmp/C-edge.log 2>&1 &
```
목적: 노드 시계열의 기여 격리.

### 3.3~3.4 `E-*` — 이웃 선택 정책 전체 조합 (mode × r × K) ★★
CLI: `--neighbor_mode {count|radius|hybrid}`, `--radius`, `--K_max`(K1), `--K_max2`(K2).
기준 `E-base` = hybrid/r20/K6·4 = 현재 94.15%. 우선순위: ①정책비교(★★) > ②③반경·K민감도(★) > ④(☆).

```bash
git pull origin main
CFG=configs/et_nagraphsage_2hop_base_ep500.yaml

# ① 정책 비교 (r=20, K=6·4 고정) — 핵심
CUDA_VISIBLE_DEVICES=0 python train.py --config $CFG --neighbor_mode count  --experiment E-count-K6   > /tmp/E-count-K6.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python train.py --config $CFG --neighbor_mode radius --experiment E-radius-r20 > /tmp/E-radius-r20.log 2>&1 &

# ② 반경 민감도 (hybrid, r 변화)
CUDA_VISIBLE_DEVICES=2 python train.py --config $CFG --neighbor_mode hybrid --radius 10 --experiment E-hybrid-r10 > /tmp/E-hybrid-r10.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python train.py --config $CFG --neighbor_mode hybrid --radius 30 --experiment E-hybrid-r30 > /tmp/E-hybrid-r30.log 2>&1 &

# ③ count K 민감도
python train.py --config $CFG --neighbor_mode count --K_max 5  --K_max2 4 --experiment E-count-K5  > /tmp/E-count-K5.log  2>&1 &
python train.py --config $CFG --neighbor_mode count --K_max 10 --K_max2 6 --experiment E-count-K10 > /tmp/E-count-K10.log 2>&1 &

# ④ radius 반경 민감도 (여유 시)
# python train.py --config $CFG --neighbor_mode radius --radius 10 --experiment E-radius-r10 &
# python train.py --config $CFG --neighbor_mode radius --radius 30 --experiment E-radius-r30 &
```
목적: ①번(count vs hybrid)이 핵심 — NAGraphSAGE가 count 방식이었으므로 공정비교 confound 제거.
②③④는 r=20·K=6 선택의 타당성 근거. 전체 조합·의미는 문서 <b>Ablation E 페이지</b> 표 참조.

### 3.5 `B-T5` — 시퀀스 길이 T=5 ★
```bash
git pull origin main
CUDA_VISIBLE_DEVICES=1 python train.py \
  --config configs/et_nagraphsage_2hop_base_ep500.yaml \
  --T 5 \
  --experiment B-T5 \
  > /tmp/B-T5.log 2>&1 &
```

### 3.7 `D-h*` — 채널 폭 hidden_dim 확대 ★★★
> NAGraphSAGE 최고(94.64%)는 파라미터 **1.25M**, 우리 최고는 **172K** (7배 작음).
> 채널 폭을 키워 용량을 맞춘다. hidden_dim=384면 1.39M로 NAGraphSAGE급.
> 파라미터: h128=172K / h192=367K / h256=636K / h384=1.39M.
```bash
git pull origin main
CFG=configs/et_nagraphsage_2hop_base_ep500.yaml
CUDA_VISIBLE_DEVICES=0 python train.py --config $CFG --hidden_dim 192 --experiment D-h192 > /tmp/D-h192.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python train.py --config $CFG --hidden_dim 256 --experiment D-h256 > /tmp/D-h256.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python train.py --config $CFG --hidden_dim 384 --experiment D-h384 > /tmp/D-h384.log 2>&1 &
```
목적: 용량 부족이 성능 한계인지 검증. 과적합 시 dropout↑. 효과 있으면 최고 구조(K1=10+SupCon)에 결합.

### 3.6 `A-transformer` — Transformer 인코더 ★ ⛔ 아직 실행 불가
> `temporal_encoder.py`는 현재 gru/lstm/mamba만 지원. transformer 미구현.
> 로컬 서버가 구현·커밋 후 사용 가능.
```bash
# (구현 후)
# python train.py --config configs/et_nagraphsage_2hop_base_ep500.yaml \
#   --encoder_type transformer --experiment A-transformer > /tmp/A-transformer.log 2>&1 &
```

> 결과 확인: 각 로그 끝의 `Test Acc (State_Acc)` 및 클래스별 정확도를 `results/school.md`에 기록.

---

## 4. 결과 보고 양식 (results/school.md 에 append)

```markdown
### <실험ID> (2026-07-XX)
- Ablation: <그룹> / 변형: <변형>
- config: configs/xxx.yaml
- Best Val: 0.XXXX (epoch NN)
- Test Acc: 0.XXXX
- Stop / LaneChange / Normal: 0.XX / 0.XX / 0.XX
- 비고:
```
