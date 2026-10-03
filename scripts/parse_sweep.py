"""Collect the sweep into one JSON:
  {"layers": {model: transformer layers},
   "data":   {model: {app: {setting: {rotation: {thp, slow, L: {kernel: [lat_ms, cpu_alone_lat_ms]}}}}}}}
usage: COSM_ROOT=... python3 scripts/parse_sweep.py [out.json]"""
import glob, json, os, re, sys
from workloads import MODELS, WORKLOADS, result_dir

root = os.environ['COSM_ROOT']
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'sweep.json')
D, layers = {}, {}
for m in MODELS:
    for tag in WORKLOADS:
        for d in glob.glob(os.path.join(root, 'simulations', result_dir(m, tag), 'sd-*')):
            _, r, t, p, b, io, i = os.path.basename(d).split('-')
            f = os.path.join(d, 'simulation.log')
            if not os.path.exists(f):
                continue
            s = open(f).read()
            if '\nEOF' not in s:
                continue
            lat = {}
            for blk in re.split(r'running layer \[ ', s)[1:]:
                lat[blk.split(' ]')[0]] = float(re.search(r'cur latency : ([\d.e-]+)', blk).group(1))
            L = {k: (v, lat['cpu_ideal_' + k]) for k, v in lat.items() if not k.startswith(('pim_ideal_', 'cpu_ideal_'))}
            layers[m] = int(re.search(r'layer_latency: [\d.e-]+ \* (\d+)=', s).group(1))
            thp = float(re.search(r'pim throughput: ([\d.]+)', s).group(1))
            cd = float(re.search(r'cpu degradation:\s+([\d.]+)', s).group(1))
            D.setdefault(m, {}).setdefault(tag, {}).setdefault(f'{t}:{p}:{b}:{io}:{i}', {})[r] = dict(thp=thp, slow=100 * (1 - cd), L=L)
json.dump(dict(layers=layers, data=D), open(out, 'w'))
print(out, {m: {t: len(v) for t, v in D[m].items()} for m in D})
