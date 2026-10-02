"""Collect the sweep into one JSON: {app: {setting: {rotation: {thp, slow, L: {kernel: [lat_ms, cpu_alone_lat_ms]}}}}}.
usage: COSM_ROOT=... python3 scripts/parse_sweep.py [out.json]"""
import glob, json, os, re, sys
from workloads import WORKLOADS

root = os.environ['COSM_ROOT']
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'sweep.json')
D = {}
for tag in WORKLOADS:
    for d in glob.glob(os.path.join(root, 'simulations', f'results_sd_{tag}', 'sd-*')):
        _, r, t, p, b, io, i = os.path.basename(d).split('-')
        s = open(os.path.join(d, 'simulation.log')).read()
        if '\nEOF' not in s:
            continue
        lat = {}
        for blk in re.split(r'running layer \[ ', s)[1:]:
            lat[blk.split(' ]')[0]] = float(re.search(r'cur latency : ([\d.e-]+)', blk).group(1))
        L = {k: (v, lat['cpu_ideal_' + k]) for k, v in lat.items() if not k.startswith(('pim_ideal_', 'cpu_ideal_'))}
        thp = float(re.search(r'pim throughput: ([\d.]+)', s).group(1))
        cd = float(re.search(r'cpu degradation:\s+([\d.]+)', s).group(1))
        D.setdefault(tag, {}).setdefault(f'{t}:{p}:{b}:{io}:{i}', {})[r] = dict(thp=thp, slow=100 * (1 - cd), L=L)
json.dump(D, open(out, 'w'))
print(out, {t: len(v) for t, v in D.items()})
