#!/bin/bash
# 2026-07-12 예약 실행: 현재 GPU0/1/2에서 돌고 있는 STGCN/TGN/LSTM이 끝나는 대로
# 그 자리에 DCRNN / Forecast-MAE(pretrain->finetune) / SIMPL을 순서대로 실행한다.
# GPU3(Transformer)는 뒤에 예약된 작업이 없어 끝나면 그대로 비워둔다.
cd /home/oem/TNA_research
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tna_research
set -u
mkdir -p logs/tsem

STGCN_PID=1997065   # GPU0
TGN_PID=1997066     # GPU1
LSTM_PID=1997067    # GPU2

wait_for_pid() {
  local pid="$1"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
  done
}

# --- GPU0: STGCN 종료 대기 -> DCRNN ---
(
  wait_for_pid "$STGCN_PID"
  echo "[queue] $(date '+%F %T') STGCN(GPU0, pid $STGCN_PID) 종료 감지, DCRNN 실행 시작"
  python -u comparison/dcrnn_tsem/train_dcrnn_tsem.py \
      --config comparison/dcrnn_tsem/config.yaml \
      --experiment dcrnn_tsem_w10_h10 --gpus 0 \
      > logs/tsem/comparison_dcrnn_tsem_w10_h10.log 2>&1
  echo "[queue] $(date '+%F %T') DCRNN(GPU0) 학습 종료"
) &

# --- GPU1: TGN 종료 대기 -> Forecast-MAE (pretrain -> finetune) ---
(
  wait_for_pid "$TGN_PID"
  echo "[queue] $(date '+%F %T') TGN(GPU1, pid $TGN_PID) 종료 감지, Forecast-MAE 사전학습 시작"
  python -u comparison/forecastmae_tsem/train_forecastmae_tsem.py \
      --config comparison/forecastmae_tsem/config.yaml --mode pretrain \
      --experiment forecastmae_tsem_w10_h10 --gpus 1 \
      > logs/tsem/comparison_forecastmae_tsem_pretrain.log 2>&1
  echo "[queue] $(date '+%F %T') Forecast-MAE 사전학습 종료, 미세조정 시작"
  python -u comparison/forecastmae_tsem/train_forecastmae_tsem.py \
      --config comparison/forecastmae_tsem/config.yaml --mode finetune \
      --experiment forecastmae_tsem_w10_h10 --gpus 1 \
      > logs/tsem/comparison_forecastmae_tsem_finetune.log 2>&1
  echo "[queue] $(date '+%F %T') Forecast-MAE(GPU1) 학습 종료"
) &

# --- GPU2: LSTM 종료 대기 -> SIMPL ---
(
  wait_for_pid "$LSTM_PID"
  echo "[queue] $(date '+%F %T') LSTM(GPU2, pid $LSTM_PID) 종료 감지, SIMPL 실행 시작"
  python -u comparison/simpl_tsem/train_simpl_tsem.py \
      --config comparison/simpl_tsem/config.yaml \
      --experiment simpl_tsem_w10_h10 --gpus 2 \
      > logs/tsem/comparison_simpl_tsem_w10_h10.log 2>&1
  echo "[queue] $(date '+%F %T') SIMPL(GPU2) 학습 종료"
) &

wait
echo "[queue] $(date '+%F %T') 예약된 3개 작업(DCRNN/Forecast-MAE/SIMPL) 전부 종료"
