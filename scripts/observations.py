"""Numbers behind the observations in Section II of the letter.
usage: python3 scripts/observations.py [data/sweep.json] [--budget 2]
  1. replay variability: how much one setting's throughput changes with the CPU-trace rotation, and how the
     setting that wins on a single rotation ranks on the mean over all rotations;
  2. parameter interactions: variance of log decode latency explained by an additive model vs. a model with
     pairwise terms, and the throughput reached by tuning one parameter at a time (coordinate descent);
  3. per-kernel noise: coefficient of variation of one kernel's latency across rotations."""
import argparse, json, os, statistics as st
import numpy as np
from workloads import WORKLOADS, NAMES, LEVELS, COSM_DEFAULT

ap = argparse.ArgumentParser()
ap.add_argument('data', nargs='?', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'sweep.json'))
ap.add_argument('--budget', type=float, default=2.0)
a = ap.parse_args()
D = json.load(open(a.data))
apps = [t for t in WORKLOADS if t in D]
W = lambda k: 1 if k.endswith('output') else 16


def mean_table(t):
    return {c: (st.mean(v[r]['thp'] for r in v), st.mean(v[r]['slow'] for r in v)) for c, v in D[t].items()}


print('== 1. replay variability ==')
for t in apps:
    M = mean_table(t)
    dev = []
    for c, v in D[t].items():
        x = [v[r]['thp'] for r in v]
        if len(x) > 1:
            dev.append(max(abs(y / st.mean(x) - 1) for y in x))
    rots = sorted(set().union(*[set(v) for v in D[t].values()]))
    feas = [c for c in M if M[c][1] <= a.budget]
    order = sorted(feas, key=lambda c: -M[c][0])
    winners, ranks = [], []
    for r in rots:
        ok = [c for c in D[t] if r in D[t][c] and D[t][c][r]['slow'] <= a.budget]
        w = max(ok, key=lambda c: D[t][c][r]['thp'])
        winners.append(w)
        ranks.append(order.index(w) + 1 if w in order else None)
    print(f'  {NAMES[t]:16s} max deviation from the mean over rotations: median {100 * np.median(dev):.1f}%, 90th pct {100 * np.percentile(dev, 90):.1f}%, max {100 * max(dev):.1f}%')
    print(f'  {"":16s} single-rotation winners (budget {a.budget:g}%): {len(set(winners))} distinct of {len(rots)}; their ranks on the mean: {ranks} '
          f'(infeasible on the mean: {sum(r is None for r in ranks)})')

print('\n== 2. parameter interactions ==')
def onehot(c, pairwise):
    v = [LEVELS[j].index(int(x)) for j, x in enumerate(c.split(':'))]
    f = [1.0]
    for j in range(5):
        f += [1.0 if v[j] == k else 0.0 for k in range(len(LEVELS[j]))]
    if pairwise:
        for i in range(5):
            for j in range(i + 1, 5):
                f += [1.0 if (v[i] == p and v[j] == q) else 0.0 for p in range(len(LEVELS[i])) for q in range(len(LEVELS[j]))]
    return f

for t in apps:
    M = mean_table(t)
    C = list(M)
    y = np.log([1000 / M[c][0] for c in C])          # log decode latency
    out = []
    for pw in (False, True):
        X = np.array([onehot(c, pw) for c in C])
        w, *_ = np.linalg.lstsq(X, y, rcond=None)
        out.append(1 - np.sum((y - X @ w) ** 2) / np.sum((y - y.mean()) ** 2))
    # tuning one parameter at a time from the default setting, under the budget
    cur = COSM_DEFAULT if M[COSM_DEFAULT][1] <= a.budget else max((c for c in C if M[c][1] <= a.budget), key=lambda c: -M[c][1])
    for _ in range(10):
        prev = cur
        for j in range(5):
            cand = []
            for lv in LEVELS[j]:
                v = cur.split(':'); v[j] = str(lv); c = ':'.join(v)
                if c in M and M[c][1] <= a.budget:
                    cand.append(c)
            if cand:
                cur = max(cand, key=lambda c: M[c][0])
        if cur == prev:
            break
    best = max((c for c in C if M[c][1] <= a.budget), key=lambda c: M[c][0])
    print(f'  {NAMES[t]:16s} R^2 additive {out[0]:.3f}, with pairwise terms {out[1]:.3f} | one-at-a-time tuning reaches {cur} = {M[cur][0] / M[best][0]:.1%} of the best {best}')

print('\n== 3. per-kernel noise (coefficient of variation across rotations) ==')
for t in apps:
    cv = []
    for c, v in D[t].items():
        if len(v) < 2:
            continue
        for k in next(iter(v.values()))['L']:
            x = [v[r]['L'][k][0] for r in v]
            cv.append(np.std(x) / np.mean(x))
    print(f'  {NAMES[t]:16s} median {100 * np.median(cv):.1f}%, mean {100 * np.mean(cv):.1f}%')
