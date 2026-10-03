"""Fig. (budget sweep only): cost of a static setting. Top: per model, geomean over its applications of the default setting, the best
single setting, and the best setting per application under a 2% CPU-slowdown budget (dots: individual applications).
Bottom: best single setting vs. budget, one line per model.  usage: python3 figures/fig_static.py [data/sweep.json]"""
import os, sys, statistics as st
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'scripts'))
import paper_style as ps
ps.apply()
import matplotlib.pyplot as plt
from sweepdata import Sweep, eff_budget, pick_single
from workloads import MODEL_NAMES, COSM_DEFAULT

S = Sweep(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'data', 'sweep.json'))
B = 2.0
models = [m for m in ('bloom', 'deepseek', 'qwen2') if m in S.models()]
T = {m: {t: S.table(m, t) for t in S.apps(m)} for m in models}

def analyse(m, bud):
    apps = list(T[m])
    EB = {t: eff_budget(T[m][t], bud) for t in apps}
    best = {t: max((c for c in T[m][t] if T[m][t][c][1] <= EB[t]), key=lambda c: T[m][t][c][0]) for t in apps}
    single, single_over = pick_single({t: T[m][t] for t in apps}, EB, best)
    rel_single = {t: T[m][t][single][0] / T[m][t][best[t]][0] for t in apps}
    rel_def = {t: T[m][t][COSM_DEFAULT][0] / T[m][t][best[t]][0] for t in apps}
    over = {t: T[m][t][COSM_DEFAULT][1] > EB[t] for t in apps}
    return single, rel_single, rel_def, over, single_over

fig, b = plt.subplots(figsize=(ps.COL, 1.35))
buds = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
sty = {'bloom': (ps.NAVY, 's', '-'), 'deepseek': (ps.RED, 'o', '--'), 'qwen2': (ps.GREEN, '^', '-.')}
for m in models:
    g, viol = [], []
    for bud in buds:
        r = analyse(m, bud)
        g.append(st.geometric_mean(r[1].values())); viol.append(len(r[4]) > 0)
    c, mk, ls = sty[m]
    g, viol = np.array(g), np.array(viol)
    b.plot(buds, np.where(viol, np.nan, g), color=c, marker=mk, ls=ls, mec='black', mew=0.4, label=MODEL_NAMES[m])
    b.scatter(np.array(buds)[viol], g[viol], marker=mk, facecolors='white', edgecolors=c, s=14, lw=0.8, zorder=3)
    print(f'  {MODEL_NAMES[m]} budget sweep: ' + ' '.join(f'{x:.3f}{"*" if v else ""}' for x, v in zip(g, viol)))
b.set_xlabel('CPU-slowdown budget (%)')
b.set_ylabel('Best single setting\n(geomean, norm.)', fontsize=6.5)
b.set_ylim(0.55, 1.12)
b.legend(loc='lower right', ncol=1, fontsize=6)
fig.tight_layout()
os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(HERE, 'out', f'fig_budget.{ext}'), dpi=300)
