"""Cost of a static setting (Sec. II of the letter): per-application best setting under a CPU-slowdown budget vs.
the best single static setting vs. COSM's default.
usage: python3 scripts/static_gap.py [data/sweep.json] [--budgets 2,3,5] [--select abc --evaluate de]
With --select/--evaluate, settings are chosen on one set of trace rotations and scored on the others
(removes the optimistic bias of picking the maximum of noisy estimates)."""
import argparse, json, os, statistics as st
from workloads import WORKLOADS, NAMES, COSM_DEFAULT

ap = argparse.ArgumentParser()
ap.add_argument('data', nargs='?', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'sweep.json'))
ap.add_argument('--budgets', default='1.5,2,3,5,100')
ap.add_argument('--select', default='')
ap.add_argument('--evaluate', default='')
a = ap.parse_args()
D = json.load(open(a.data))
apps = [t for t in WORKLOADS if t in D]

def table(rots):
    """mean over the requested rotations a setting has; settings missing more than one are dropped
    (COSM's simulator occasionally crashes for n_PTL=16 with a long idle threshold)."""
    out = {}
    for t in apps:
        out[t] = {}
        for c in D[t]:
            have = [r for r in rots if r in D[t][c]]
            if len(have) >= max(1, len(rots) - 1):
                out[t][c] = (st.mean(D[t][c][r]['thp'] for r in have), st.mean(D[t][c][r]['slow'] for r in have))
    return out

common = sorted(set().union(*[set(v) for t in apps for v in D[t].values()]))
S = table(a.select or common)          # used to choose settings
E = table(a.evaluate or a.select or common)   # used to score them
print(f'rotations: select on {a.select or "".join(common)}, evaluate on {a.evaluate or a.select or "".join(common)}')
for bud in map(float, a.budgets.split(',')):
    best = {t: max((c for c in S[t] if S[t][c][1] <= bud), key=lambda c: S[t][c][0]) for t in apps}
    cand = [c for c in S[apps[0]] if all(c in S[t] and S[t][c][1] <= bud for t in apps)]
    single = max(cand, key=lambda c: st.geometric_mean([S[t][c][0] / S[t][best[t]][0] for t in apps]))
    oracle = {t: max((c for c in E[t] if E[t][c][1] <= bud), key=lambda c: E[t][c][0]) for t in apps}
    print(f'\nCPU budget {bud:g}%: best single static {single}')
    for t in apps:
        ref = E[t][oracle[t]][0]
        print(f'  {NAMES[t]:16s} per-app best {best[t]:18s} {E[t][best[t]][0] / ref:6.1%} (cpu {E[t][best[t]][1]:.2f}%) | '
              f'single {E[t][single][0] / ref:6.1%} (cpu {E[t][single][1]:.2f}%) | '
              f'COSM {E[t][COSM_DEFAULT][0] / ref:6.1%} (cpu {E[t][COSM_DEFAULT][1]:.2f}%{", over budget" if E[t][COSM_DEFAULT][1] > bud else ""})')
    g = lambda c: st.geometric_mean([E[t][c][0] / E[t][oracle[t]][0] for t in apps if c in E[t]])
    print(f'  GMEAN: per-app best {st.geometric_mean([E[t][best[t]][0] / E[t][oracle[t]][0] for t in apps]):.1%}, '
          f'single static {g(single):.1%}, COSM default {g(COSM_DEFAULT):.1%}')
