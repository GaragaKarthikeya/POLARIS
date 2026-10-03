"""Simulate every static setting (405) x model (3) x CPU workload (12) x trace rotation (5) = 72,900 runs.
usage: COSM_ROOT=... python3 scripts/sweep.py [--jobs 24] [--models bloom,deepseek,qwen2] [--apps 10,22,...]
                                              [--rotations abcde] [--dry-run]
Runs are issued rotation by rotation, so a complete smaller dataset exists early; finished runs are skipped."""
import argparse, os, subprocess
from workloads import MODELS, WORKLOADS, configs, result_dir

ap = argparse.ArgumentParser()
ap.add_argument('--jobs', type=int, default=os.cpu_count())
ap.add_argument('--models', default=','.join(MODELS))
ap.add_argument('--apps', default=','.join(WORKLOADS))
ap.add_argument('--rotations', default='abcde')
ap.add_argument('--dry-run', action='store_true')
a = ap.parse_args()
root = os.environ['COSM_ROOT']
here = os.path.dirname(os.path.abspath(__file__))
jobs = []
for r in a.rotations:
    for m in a.models.split(','):
        for tag in a.apps.split(','):
            prefix = WORKLOADS[tag][1]
            rd = result_dir(m, tag)
            for c in configs():
                t, p, b, io, i = c.split(':')
                name = f'sd-{r}-{t}-{p}-{b}-{io}-{i}'
                log = os.path.join(root, 'simulations', rd, name, 'simulation.log')
                if os.path.exists(log) and open(log).read().rstrip().endswith('EOF'):
                    continue
                jobs.append(f'{name} {MODELS[m][0]} {prefix}_r{r}.txt {t} {p} {b} {io} {i} {rd}')
print(f'{len(jobs)} runs to go')
if not a.dry_run and jobs:
    subprocess.run(['xargs', '-P', str(a.jobs), '-L', '1', 'bash', os.path.join(here, 'run_fixed.sh')],
                   input='\n'.join(jobs) + '\n', text=True, check=False)
