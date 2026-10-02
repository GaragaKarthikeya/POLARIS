"""Simulate every static setting (405) x CPU workload (4) x trace rotation (5) = 8100 runs.
usage: COSM_ROOT=... python3 scripts/sweep.py [--jobs 24] [--apps 10,22,51,sp70] [--rotations abcde] [--dry-run]
Runs that already finished are skipped, so the sweep can be resumed."""
import argparse, os, subprocess
from workloads import WORKLOADS, configs

ap = argparse.ArgumentParser()
ap.add_argument('--jobs', type=int, default=os.cpu_count())
ap.add_argument('--apps', default=','.join(WORKLOADS))
ap.add_argument('--rotations', default='abcde')
ap.add_argument('--dry-run', action='store_true')
a = ap.parse_args()
root = os.environ['COSM_ROOT']
here = os.path.dirname(os.path.abspath(__file__))
jobs = []
for r in a.rotations:                      # rotation-major order: a complete pass over all apps finishes first
    for tag in a.apps.split(','):
        prefix = WORKLOADS[tag][1]
        for c in configs():
            t, p, b, io, i = c.split(':')
            name, rd = f'sd-{r}-{t}-{p}-{b}-{io}-{i}', f'results_sd_{tag}'
            log = os.path.join(root, 'simulations', rd, name, 'simulation.log')
            if os.path.exists(log) and open(log).read().rstrip().endswith('EOF'):
                continue
            jobs.append(f'{name} {prefix}_r{r}.txt {t} {p} {b} {io} {i} {rd}')
print(f'{len(jobs)} runs to go')
if not a.dry_run and jobs:
    subprocess.run(['xargs', '-P', str(a.jobs), '-L', '1', 'bash', os.path.join(here, 'run_fixed.sh')],
                   input='\n'.join(jobs) + '\n', text=True, check=False)
