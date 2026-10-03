#!/usr/bin/env bash
# All replay experiments of the letter: convergence for every (model, application), OS switching per model,
# and the ablations. Results go to data/emulate/.  usage: bash scripts/emulate_all.sh [parallel-jobs]
set -euo pipefail
cd "$(dirname "$0")"
J=${1:-12}
export OMP_NUM_THREADS=1
APPS=$(python3 -c "from workloads import WORKLOADS; print(' '.join(WORKLOADS))")
MODELS=$(python3 -c "from workloads import MODELS; print(' '.join(MODELS))")
mkdir -p ../data/emulate
{
  for m in $MODELS; do for t in $APPS; do
    echo "python3 emulate.py --model $m --app $t --decodes 200 --seeds 10 --budget 0.02 --out ../data/emulate/${m}_converge_${t}.json"
  done; done
  for m in $MODELS; do
    echo "python3 emulate.py --model $m --schedule $(echo $APPS | tr ' ' ',') --slice 15 --rounds 4 --seeds 10 --budget 0.02 --out ../data/emulate/${m}_switch.json"
  done
  for mode in factored random; do
    echo "python3 emulate.py --model bloom --app 10 --mode $mode --decodes 200 --seeds 10 --budget 0.02 --out ../data/emulate/bloom_ablation_${mode}_10.json"
  done
  echo "python3 emulate.py --model bloom --app 10 --agent-kw pairwise=1 --decodes 200 --seeds 10 --budget 0.02 --out ../data/emulate/bloom_ablation_pairwise_10.json"
} | xargs -P "$J" -I{} bash -c '{} > /dev/null'
echo "done: $(ls ../data/emulate | wc -l) result files"
