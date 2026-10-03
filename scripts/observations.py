"""Numbers behind the observations in Section II of the letter, for every model.
usage: python3 scripts/observations.py [data/sweep.json] [--models bloom,...] [--budget 2]"""
import argparse, os, statistics as st
import numpy as np
from sweepdata import Sweep
from workloads import NAMES, MODEL_NAMES, LEVELS, COSM_DEFAULT

ap = argparse.ArgumentParser()
ap.add_argument('data', nargs='?', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'sweep.json'))
ap.add_argument('--models', default='')
ap.add_argument('--apps', default='', help='restrict to these applications')
ap.add_argument('--budget', type=float, default=2.0)
a = ap.parse_args()
S = Sweep(a.data)


def onehot(c):
    v = [LEVELS[j].index(int(x)) for j, x in enumerate(c.split(':'))]
    f = [1.0]
    for j in range(5):
        f += [1.0 if v[j] == k else 0.0 for k in range(len(LEVELS[j]))]
    return f


for m in (a.models.split(',') if a.models else S.models()):
    print(f'===== {MODEL_NAMES[m]} =====')
    print(f'{"application":16s} {"dev med":>8s} {"dev p90":>8s} {"winners":>8s} {"infeas":>7s} {"R2 add":>7s} {"1-at-a-time":>12s} {"kernel cv":>9s}')
    agg = {k: [] for k in ('dev', 'win', 'inf', 'r2', 'cd', 'cv')}
    for t in (a.apps.split(',') if a.apps else S.apps(m)):
        V = S.runs(m, t)
        M = S.table(m, t, min_rots=1)
        rots = sorted(set().union(*[set(v) for v in V.values()]))
        dev = [max(abs(V[c][r]['thp'] / M[c][0] - 1) for r in V[c]) for c in V if len(V[c]) > 1]
        feas = sorted((c for c in M if M[c][1] <= a.budget), key=lambda c: -M[c][0])
        winners = []
        for r in rots:
            ok = [c for c in V if r in V[c] and V[c][r]['slow'] <= a.budget]
            if ok:
                winners.append(max(ok, key=lambda c: V[c][r]['thp']))
        infeas = sum(w not in feas for w in winners)
        C = list(M)
        y = np.log([1000 / M[c][0] for c in C])
        X = np.array([onehot(c) for c in C])
        w, *_ = np.linalg.lstsq(X, y, rcond=None)
        r2 = 1 - np.sum((y - X @ w) ** 2) / np.sum((y - y.mean()) ** 2)
        start = COSM_DEFAULT if M[COSM_DEFAULT][1] <= a.budget else min(C, key=lambda c: M[c][1])
        cur = start
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
        cd = M[cur][0] / M[feas[0]][0] if feas else float('nan')
        cv = [np.std([V[c][r]['L'][k][0] for r in V[c]]) / np.mean([V[c][r]['L'][k][0] for r in V[c]])
              for c in V if len(V[c]) > 1 for k in next(iter(V[c].values()))['L']]
        print(f'{NAMES[t]:16s} {100*np.median(dev):7.1f}% {100*np.percentile(dev, 90):7.1f}% {len(set(winners)):4d}/{len(winners):<3d} '
              f'{infeas:7d} {r2:7.3f} {cd:11.1%} {100*np.median(cv):8.1f}%')
        for k, v in zip(agg, (np.median(dev), len(set(winners)) / max(len(winners), 1), infeas, r2, cd, np.median(cv))):
            agg[k].append(v)
    print(f'{"ALL":16s} median dev {100*np.median(agg["dev"]):.1f}% (range {100*min(agg["dev"]):.1f}-{100*max(agg["dev"]):.1f}%), '
          f'R2 {min(agg["r2"]):.3f}-{max(agg["r2"]):.3f}, one-at-a-time {min(agg["cd"]):.1%}-{max(agg["cd"]):.1%} '
          f'(< 95% for {sum(x < 0.95 for x in agg["cd"])}/{len(agg["cd"])} apps), kernel cv {100*min(agg["cv"]):.1f}-{100*max(agg["cv"]):.1f}%')
