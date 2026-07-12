#!/bin/bash
# 2026-07-12 예약 실행: GPU2에서 돌고 있는 edge-temporal 재검증 실험
# (configs/tsem_sage_invariant_6d_edgetemporal.yaml)이 끝나면 그 자리에 SIMPL을 실행한다.
cd /home/oem/TNA_research
PS1=''
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tna_research
set -u
mkdir -p logs/tsem

EDGETEMPORAL_PID=2166426   # GPU2

while kill -0 "$EDGETEMPORAL_PID" 2>/dev/null; do
  sleep 30
done

echo "[queue3] $(date '+%F %T') edge-temporal 재검증(GPU2, pid $EDGETEMPORAL_PID) 종료 감지, SIMPL 실행 시작"
python -u comparison/simpl_tsem/train_simpl_tsem.py \
    --config comparison/simpl_tsem/config.yaml \
    --experiment simpl_tsem_w10_h10 --gpus 2 \
    > logs/tsem/comparison_simpl_tsem_w10_h10.log 2>&1
echo "[queue3] $(date '+%F %T') SIMPL(GPU2) 학습 종료"
