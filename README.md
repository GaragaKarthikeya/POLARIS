# POLARIS: Per-Kernel Online Tuning of Concurrent PIM/CPU Memory Controllers

This repository contains POLARIS and the scripts that reproduce every result in the letter
*"POLARIS: Per-Kernel Online Tuning of Concurrent PIM/CPU Memory Controllers on Mobile Devices"*.

POLARIS is a runtime agent that sets the parameters of COSM's concurrent PIM/CPU memory controller
(PIM command length `n_PTL`, `pred_th`, `bus_th`, `io_send_interval`, `idle_threshold`) at every PIM kernel
launch. It learns from the kernel's own latency and the CPU slowdown the kernel caused, keeps one model per
CPU application, and enforces a CPU-slowdown budget.

POLARIS does **not** modify COSM's scheduling logic. It is installed on top of COSM's public artifact
(Ramulator 2.0 based, MIT license) as one Python module and a small patch to COSM's simulation driver.

## Contents

| Path | What it is |
|---|---|
| `polaris/polaris_agent.py` | The agent: per-application Bayesian models with pairwise parameter interactions, Thompson sampling, Lagrangian CPU budget. Also contains the ablation policies (`fixed`, `random`, `tab`, `factored`). |
| `harness/simulator_polaris.patch` | Patch that turns COSM's `Simulator.py` into `Simulator_polaris.py`: one decision per PIM kernel, PIM traces generated at the selected command length, CPU-only reference run per kernel. |
| `scripts/setup.sh` | Installs POLARIS into a COSM checkout, writes rotated CPU traces, builds the simulator. |
| `scripts/workloads.py` | Workloads, trace rotations, and the 405-setting parameter grid. |
| `scripts/sweep.py`, `scripts/run_fixed.sh` | Simulate all 405 static settings x 4 CPU workloads x 5 trace rotations (8100 runs). |
| `scripts/parse_sweep.py` | Collects the sweep into `data/sweep.json`. |
| `scripts/static_gap.py` | Cost of a static setting under a CPU budget (per-application best vs. best single setting vs. COSM default), optionally chosen and scored on disjoint trace rotations. |
| `scripts/observations.py` | Replay variability, parameter interactions (one-at-a-time tuning), per-kernel reward noise. |
| `scripts/emulate.py` | Replays POLARIS's decisions against the sweep results (one application, or the OS switching between applications). |
| `scripts/run_polaris.py` | Live POLARIS sessions in the simulator. |
| `scripts/compare_live.py` | Compares live sessions with the replay. |
| `figures/` | Plotting scripts (`fig_static.py`, `fig_converge.py`, `fig_switch.py`) and the TikZ source of the overview figure. |
| `data/sweep.json` | All 8100 sweep runs (per-kernel latency and CPU-alone latency). |
| `data/emulate/` | Replay results behind the convergence, switching, and ablation numbers. |
| `data/live/` | Decision logs of six live sessions on the unmodified artifact and their comparison with the replay. |

## Requirements

* Linux, x86-64. GCC 12+ and CMake 3.30+ (as required by COSM's artifact).
* Python 3.10+ with `numpy`, `matplotlib`, `pyyaml` (`pip install -r requirements.txt`).
* COSM's artifact: <https://doi.org/10.5281/zenodo.19660293>.
* The full sweep takes about 2.5 hours on 24 cores and about 30 GB of disk while running.

## Setup

```bash
# unpack COSM's artifact somewhere, then
export COSM_ROOT=/path/to/cosm-artifact
bash scripts/setup.sh 16          # 16 = make jobs
```

## Reproducing the results

All commands run from `scripts/`.

**Without simulation (uses `data/sweep.json`):**

```bash
python3 static_gap.py --budgets 1.5,2,3,5,100                  # Section II: cost of a static setting
python3 static_gap.py --select abc --evaluate de              # same, settings chosen and scored on disjoint rotations
python3 emulate.py --app 10 --decodes 200 --seeds 10          # convergence with Tencent Meeting on the CPU
python3 emulate.py --schedule 10,22,51,sp70 --slice 40 --rounds 3 --seeds 10   # OS switches applications
python3 emulate.py --app 10 --mode factored --seeds 10        # ablations: factored, random, --agent-kw pairwise=1
python3 observations.py                                       # replay variability, interactions, reward noise
python3 ../figures/fig_static.py; python3 ../figures/fig_converge.py; python3 ../figures/fig_switch.py   # -> figures/out/
```

**Regenerating the sweep:**

```bash
COSM_ROOT=... python3 sweep.py --jobs 24      # resumable; skips finished runs
COSM_ROOT=... python3 parse_sweep.py          # -> data/sweep.json
```

**Live POLARIS sessions in the simulator:**

```bash
COSM_ROOT=... python3 run_polaris.py --tag conv --seed 1 --app 10 --decodes 150
COSM_ROOT=... python3 run_polaris.py --tag switch --seed 1 --schedule 10,22,51,sp70 --slice 40 --rounds 3
```

Each decode step is one simulator invocation; the agent's state (`state.json`) and a per-kernel decision log
(`decisions.log`) are kept in `simulations/results_polaris/session_<tag>-s<seed>/`. Compare them with the replay:

```bash
COSM_ROOT=... python3 compare_live.py --tag conv --app 10 --emulated ../data/emulate/converge_10.json
```

## Notes on variability

* COSM's simulator replays a CPU trace from the same point for every PIM kernel. We therefore evaluate every
  setting on five rotations of each CPU trace (`scripts/make_rotations.py`) and report means.
* COSM's PIM trace generator places PIM operands in randomly drawn rows without a fixed seed, so repeated runs of
  the same setting differ slightly. `Simulator_polaris.py` accepts `--trace-seed` to fix this for POLARIS runs.
* COSM's simulator occasionally crashes (segmentation fault) for `n_PTL = 16` with a long idle threshold; the
  analyses average each setting over the rotations it has.
* Results reproduce statistically, not bit-for-bit.

## License

MIT (see `LICENSE`). COSM and Ramulator 2.0 are distributed under their own MIT license.
