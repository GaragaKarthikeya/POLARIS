"""Write rotated copies of COSM's CPU traces into $COSM_ROOT/simulations/traces.
The rotated names keep the prefix COSM uses to pick each trace's warm-up length."""
import os
from workloads import WORKLOADS

root = os.environ['COSM_ROOT']
tdir = os.path.join(root, 'simulations', 'traces')
for tag, (src, prefix, step) in WORKLOADS.items():
    lines = open(os.path.join(tdir, src)).read().splitlines()
    n = len(lines)
    for i, ch in enumerate('abcde'):
        k = i * step if step else i * n // 7
        with open(os.path.join(tdir, f'{prefix}_r{ch}.txt'), 'w') as f:
            f.write('\n'.join(lines[k:] + lines[:k]) + '\n')
    print(f'{src}: {n} lines -> {prefix}_r[a-g].txt')
