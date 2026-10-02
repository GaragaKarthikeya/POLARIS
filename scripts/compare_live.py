"""Compare live POLARIS sessions in the simulator (run_polaris.py) with the replay-based evaluation (emulate.py).
For each live decode step we compute (i) the measured decode throughput and CPU slowdown, and (ii) the expected
throughput of the settings the agent chose, from the sweep (the metric plotted by the emulator).
usage: COSM_ROOT=... python3 scripts/compare_live.py --tag val10 --app 10 [--emulated data/emulate/converge_10.json]"""
import argparse, glob, json, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--tag', required=True)
ap.add_argument('--app', required=True)
ap.add_argument('--budget', type=float, default=2.0, help='percent')
ap.add_argument('--data', default=os.path.join(HERE, '..', 'data', 'sweep.json'))
ap.add_argument('--emulated', default='')
ap.add_argument('--result-dir', default='results_polaris')
a = ap.parse_args()
root = os.environ['COSM_ROOT']
D = json.load(open(a.data))[a.app]
W = lambda k: 1 if k.endswith('output') else 16
mean_lat = {c: {k: (np.mean([v[r]['L'][k][0] for r in v]), np.mean([v[r]['L'][k][1] for r in v]))
                for k in next(iter(v.values()))['L']} for c, v in D.items()}
static = {c: 1000 / sum(W(k) * x[0] for k, x in m.items()) for c, m in mean_lat.items()}
slow = {c: 100 * (1 - sum(W(k) * x[1] for k, x in m.items()) / sum(W(k) * x[0] for k, x in m.items())) for c, m in mean_lat.items()}
best = max((c for c in static if slow[c] <= a.budget), key=lambda c: static[c])

def expected(choice):
    return 1000 / sum(W(k) * mean_lat[c][k][0] for k, c in choice.items()) / static[best]

quality, measured, cpu = [], [], []
for sess in sorted(glob.glob(os.path.join(root, 'simulations', a.result_dir, f'session_{a.tag}-s*'))):
    seed = sess.rsplit('-s', 1)[1]
    steps = {}
    for line in open(os.path.join(sess, 'decisions.log')):
        t = line.split()
        steps.setdefault(int(t[0]), {})[t[1].split('|', 1)[1]] = t[2]
    q, m, s = [], [], []
    for step in sorted(steps):
        log = os.path.join(root, 'simulations', a.result_dir, f'{a.tag}-s{seed}-d{step + 1:03d}', 'simulation.log')
        txt = open(log).read()
        q.append(expected(steps[step]))
        m.append(float(re.search(r'pim throughput: ([\d.]+)', txt).group(1)) / static[best])
        s.append(100 * (1 - float(re.search(r'cpu degradation:\s+([\d.]+)', txt).group(1))))
    quality.append(q); measured.append(m); cpu.append(s)
n = min(map(len, quality))
Q, M, S = (np.array([x[:n] for x in v]) for v in (quality, measured, cpu))
print(f'{len(Q)} live sessions x {n} decode steps; best static setting {best} ({static[best]:.1f} tok/s)')
E = None
if a.emulated:
    e = json.load(open(a.emulated))
    E = np.array(e['thp'])[:, :n] / static[best]
    ES = np.array(e['slow'])[:, :n]
for lo, hi in ((0, 10), (10, 30), (30, n)):
    line = f'steps {lo + 1:3d}-{hi:3d}: live chosen-setting quality {Q[:, lo:hi].mean():.3f}, live measured {M[:, lo:hi].mean():.3f}, live CPU {S[:, lo:hi].mean():.2f}%'
    if E is not None:
        line += f' | replay quality {E[:, lo:hi].mean():.3f}, replay CPU {ES[:, lo:hi].mean():.2f}%'
    print(line)
