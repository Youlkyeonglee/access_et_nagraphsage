#!/bin/bash
# 2026-07-12 예약 실행 (재작성본 — 최초 큐 스크립트가 launch 직후 원인불명으로 죽어서 재실행) :
# GPU0(STGCN)/GPU1(TGN)이 끝나는 대로 DCRNN / Forecast-MAE(pretrain->finetune)를 실행한다.
# SIMPL은 LSTM 종료를 이미 감지해 GPU2에서 별도로 즉시 시작했다(이 스크립트 관할 아님).
set -u
cd /home/oem/TNA_research
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tna_research
mkdir -p logs/tsem

STGCN_PID=1997065   # GPU0
TGN_PID=1997066     # GPU1

wait_for_pid() {
  local pid="$1"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
  done
}

# --- GPU0: STGCN 종료 대기 -> DCRNN ---
(
  wait_for_pid "$STGCN_PID"
  echo "[queue2] $(date '+%F %T') STGCN(GPU0, pid $STGCN_PID) 종료 감지, DCRNN 실행 시작"
  python -u comparison/dcrnn_tsem/train_dcrnn_tsem.py \
      --config comparison/dcrnn_tsem/config.yaml \
      --experiment dcrnn_tsem_w10_h10 --gpus 0 \
      > logs/tsem/comparison_dcrnn_tsem_w10_h10.log 2>&1
  echo "[queue2] $(date '+%F %T') DCRNN(GPU0) 학습 종료"
) &

# --- GPU1: TGN 종료 대기 -> Forecast-MAE (pretrain -> finetune) ---
(
  wait_for_pid "$TGN_PID"
  echo "[queue2] $(date '+%F %T') TGN(GPU1, pid $TGN_PID) 종료 감지, Forecast-MAE 사전학습 시작"
  python -u comparison/forecastmae_tsem/train_forecastmae_tsem.py \
      --config comparison/forecastmae_tsem/config.yaml --mode pretrain \
      --experiment forecastmae_tsem_w10_h10 --gpus 1 \
      > logs/tsem/comparison_forecastmae_tsem_pretrain.log 2>&1
  echo "[queue2] $(date '+%F %T') Forecast-MAE 사전학습 종료, 미세조정 시작"
  python -u comparison/forecastmae_tsem/train_forecastmae_tsem.py \
      --config comparison/forecastmae_tsem/config.yaml --mode finetune \
      --experiment forecastmae_tsem_w10_h10 --gpus 1 \
      > logs/tsem/comparison_forecastmae_tsem_finetune.log 2>&1
  echo "[queue2] $(date '+%F %T') Forecast-MAE(GPU1) 학습 종료"
) &

wait
echo "[queue2] $(date '+%F %T') 예약된 2개 작업(DCRNN/Forecast-MAE) 전부 종료"
