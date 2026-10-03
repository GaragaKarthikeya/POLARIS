#!/usr/bin/env bash
# One static setting on one CPU-trace rotation, with COSM's configuration (LPDDR5-6400, 2 ch x 2 ranks).
# usage: run_fixed.sh <exp-name> <model: bloom_sd|deepseek_sd|qwen2_sd> <cpu-trace> <n_PTL> <pred_th> <bus_th> <io_send_interval> <idle_threshold> <result-dir>
set -euo pipefail
: "${COSM_ROOT:?set COSM_ROOT}"
cd "$COSM_ROOT"
python3 simulations/simulator/Simulator.py \
  --experiment-name "$1" --model "$2" --N 1024 --pipeline-block-size 4194304 --pim-type 4 \
  --cpu-trace "$3" --num-expected-insts 34699 --trace-type commandreg \
  --pim-task-length "$4" --pim-task-length-min 8 --pim-bitwidth 16 \
  --pim-arbiter PIMArbiter_CPUFirstO3Predict --cpu-scheduler PIMScheduler_Cluster \
  --block-interval 300 --block-cycle 299 --block-bank-num 1 --additional-read-latency 0 \
  --idle-threshold "$8" --additional-cpu-clk-ratio 8 --script-path "$0" --result-dir "$9" \
  --nbl64 16 --pred-th "$5" --bus-th "$6" --io-send-interval "$7" > /dev/null 2>&1 || true
# only simulation.log is needed afterwards
D="simulations/$9/$1"
[ -d "$D" ] && find "$D" -mindepth 1 -maxdepth 1 ! -name simulation.log -exec rm -rf {} +
exit 0
