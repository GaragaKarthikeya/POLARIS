"""Trace-driven emulation of POLARIS on the sweep results.
Every kernel invocation draws one CPU-trace rotation and receives the latency and CPU-alone latency that the
simulator measured for that (application, setting, rotation, kernel). Because each kernel is simulated as an
independent run, this reproduces what the simulator returns for the same decisions (up to operand placement).
usage:
  python3 scripts/emulate.py --app 10 --decodes 200 --seeds 10                 # one application
  python3 scripts/emulate.py --schedule 10,22,51,sp70 --slice 40 --rounds 3    # OS switches applications
Output: per-decode expected throughput (CPU-phase averaged) of the settings POLARIS chose, written as JSON."""
import argparse, json, os, random, sys
import multiprocessing as mp
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'polaris'))
from polaris_agent import Agent, CONFIGS

ap = argparse.ArgumentParser()
ap.add_argument('--data', default=os.path.join(HERE, '..', 'data', 'sweep.json'))
ap.add_argument('--model', default='bloom')
ap.add_argument('--app', default='')
ap.add_argument('--schedule', default='')
ap.add_argument('--slice', type=int, default=40, help='decode steps per OS time slice')
ap.add_argument('--rounds', type=int, default=3)
ap.add_argument('--decodes', type=int, default=200)
ap.add_argument('--seeds', type=int, default=10)
ap.add_argument('--budget', type=float, default=0.02)
ap.add_argument('--mode', default='ts')
ap.add_argument('--agent-kw', default='', help="extra Agent kwargs 'k=v,...'")
ap.add_argument('--out', default='')
a = ap.parse_args()
from sweepdata import Sweep
SW = Sweep(a.data)
MODEL = a.model
D = SW.D[MODEL]
KERNELS = sorted(next(iter(next(iter(D[next(iter(D))].values())).values()))['L'])
W = {k: SW.weight(MODEL, k) for k in KERNELS}
ROT = {t: {c: sorted(v) for c, v in D[t].items()} for t in D}     # rotations each setting has (a few runs crash in COSM)

def expected(app, choice):
    lat = sum(W[k] * np.mean([D[app][choice[k]][r]['L'][k][0] for r in ROT[app][choice[k]]]) for k in KERNELS)
    cl = sum(W[k] * np.mean([D[app][choice[k]][r]['L'][k][1] for r in ROT[app][choice[k]]]) for k in KERNELS)
    return 1000 / lat, 100 * (1 - cl / lat)

def run(args):
    schedule, seed = args
    kw = {}
    for kv in filter(None, a.agent_kw.split(',')):
        k, v = kv.split('=')
        kw[k] = float(v) if v.replace('.', '', 1).replace('-', '', 1).isdigit() else v
    agent = Agent(seed=seed, mode=a.mode, budget=a.budget, **kw)
    env = random.Random(7919 * seed + 17)
    out = []
    for app in schedule:
        choice = {}
        for k in KERNELS:
            ctx = f'{app}|{k}'
            c = agent.choose(ctx)
            r = env.choice(ROT[app][c])
            lat, cpu = D[app][c][r]['L'][k]
            agent.update(ctx, c, lat, cpu, W[k])
            choice[k] = c
        agent.end_decode()
        out.append(expected(app, choice) + (agent.best(f'{app}|{KERNELS[0]}'),))
    return out

if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    if a.schedule:
        apps = a.schedule.split(',')
        schedule = [t for _ in range(a.rounds) for t in apps for _ in range(a.slice)]
    else:
        schedule = [a.app] * a.decodes
    with mp.get_context('fork').Pool(min(a.seeds, os.cpu_count())) as p:
        R = p.map(run, [(schedule, s) for s in range(1, a.seeds + 1)])
    static = {t: {c: expected(t, {k: c for k in KERNELS}) for c in CONFIGS if c in D[t]} for t in set(schedule)}
    res = dict(model=a.model, schedule=schedule, budget=a.budget, mode=a.mode,
               thp=[[x[0] for x in r] for r in R], slow=[[x[1] for x in r] for r in R], believed=[[x[2] for x in r] for r in R],
               static={t: {c: v for c, v in s.items()} for t, s in static.items()})
    out = a.out or os.path.join(HERE, '..', 'data', 'emulate', f'{a.model}_{a.mode}_{a.app or "switch"}_{a.budget:g}.json')
    json.dump(res, open(out, 'w'))
    q = np.array(res['thp'])
    for t in sorted(set(schedule)):
        ok = [c for c in static[t] if static[t][c][1] <= max(100 * a.budget, min(v[1] for v in static[t].values()))]
        best = max(ok, key=lambda c: static[t][c][0])
        idx = [i for i, x in enumerate(schedule) if x == t]
        print(f'{t:5s} best static {best} {static[t][best][0]:.1f} | POLARIS first 10 {q[:, idx[:10]].mean():.1f}, last 20 {q[:, idx[-20:]].mean():.1f} '
              f'({q[:, idx[-20:]].mean() / static[t][best][0]:.1%}) | CPU last 20 {np.array(res["slow"])[:, idx[-20:]].mean():.2f}%')
    print('wrote', out)
