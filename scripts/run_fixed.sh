#!/usr/bin/env bash
# One static setting on one CPU-trace rotation, with COSM's configuration (BLOOM-1B1, LPDDR5-6400, 2 ch x 2 ranks).
# usage: run_fixed.sh <exp-name> <cpu-trace> <n_PTL> <pred_th> <bus_th> <io_send_interval> <idle_threshold> <result-dir>
set -euo pipefail
: "${COSM_ROOT:?set COSM_ROOT}"
cd "$COSM_ROOT"
python3 simulations/simulator/Simulator.py \
  --experiment-name "$1" --model bloom_sd --N 1024 --pipeline-block-size 4194304 --pim-type 4 \
  --cpu-trace "$2" --num-expected-insts 34699 --trace-type commandreg \
  --pim-task-length "$3" --pim-task-length-min 8 --pim-bitwidth 16 \
  --pim-arbiter PIMArbiter_CPUFirstO3Predict --cpu-scheduler PIMScheduler_Cluster \
  --block-interval 300 --block-cycle 299 --block-bank-num 1 --additional-read-latency 0 \
  --idle-threshold "$7" --additional-cpu-clk-ratio 8 --script-path "$0" --result-dir "$8" \
  --nbl64 16 --pred-th "$4" --bus-th "$5" --io-send-interval "$6" > /dev/null 2>&1
# only the logs are needed afterwards
rm -f "simulations/$8/$1"/*/pim.trace "simulations/$8/$1"/*/pim_ideal.trace
