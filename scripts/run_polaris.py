"""Live POLARIS sessions in the simulator: one simulator invocation per decode step, the agent's state persists
between steps. The OS schedule decides which CPU application runs during each decode step.
usage:
  COSM_ROOT=... python3 scripts/run_polaris.py --tag conv --seed 1 --app 10 --decodes 150
  COSM_ROOT=... python3 scripts/run_polaris.py --tag switch --seed 1 --schedule 10,22,51,sp70 --slice 40 --rounds 3"""
import argparse, os, subprocess
from workloads import WORKLOADS, ROTATIONS

ap = argparse.ArgumentParser()
ap.add_argument('--tag', required=True)
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--app', default='')
ap.add_argument('--decodes', type=int, default=150)
ap.add_argument('--schedule', default='')
ap.add_argument('--slice', type=int, default=40)
ap.add_argument('--rounds', type=int, default=3)
ap.add_argument('--budget', type=float, default=0.02)
ap.add_argument('--mode', default='ts')
ap.add_argument('--fixed', default='', help='run a static setting instead of the agent (reference)')
ap.add_argument('--result-dir', default='results_polaris')
a = ap.parse_args()
root = os.environ['COSM_ROOT']
sched = ([t for _ in range(a.rounds) for t in a.schedule.split(',') for _ in range(a.slice)] if a.schedule else [a.app] * a.decodes)
sess = os.path.join(root, 'simulations', a.result_dir, f'session_{a.tag}-s{a.seed}')
os.makedirs(sess, exist_ok=True)
for i, app in enumerate(sched):
    name = f'{a.tag}-s{a.seed}-d{i + 1:03d}'
    log = os.path.join(root, 'simulations', a.result_dir, name, 'simulation.log')
    if os.path.exists(log) and open(log).read().rstrip().endswith('EOF'):
        continue
    src, prefix, _ = WORKLOADS[app]
    variants = ','.join(f'{prefix}_r{r}.txt' for r in ROTATIONS)
    cmd = ['python3', 'simulations/simulator/Simulator_polaris.py',
           '--experiment-name', name, '--model', 'bloom_sd', '--N', '1024', '--pipeline-block-size', '4194304', '--pim-type', '1',
           '--cpu-trace', f'{prefix}_ra.txt', '--num-expected-insts', '34699', '--trace-type', 'commandreg',
           '--pim-task-length', '128', '--pim-task-length-min', '8', '--pim-bitwidth', '16',
           '--pim-arbiter', 'PIMArbiter_CPUFirstO3Predict', '--cpu-scheduler', 'PIMScheduler_Cluster',
           '--block-interval', '300', '--block-cycle', '299', '--block-bank-num', '1', '--additional-read-latency', '0',
           '--idle-threshold', '0', '--additional-cpu-clk-ratio', '8', '--script-path', os.path.abspath(__file__),
           '--result-dir', a.result_dir, '--nbl64', '16', '--pred-th', '16', '--bus-th', '12', '--io-send-interval', '480',
           '--agent-state', os.path.join(sess, 'state.json'), '--agent-log', os.path.join(sess, 'decisions.log'),
           '--agent-seed', str(a.seed), '--agent-budget', str(a.budget), '--agent-mode', a.mode, '--agent-app', app,
           '--cpu-trace-variants', variants]
    if a.fixed:
        cmd += ['--fixed-config', a.fixed]
    subprocess.run(cmd, cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    print(f'decode {i + 1}/{len(sched)} ({app}) done', flush=True)
