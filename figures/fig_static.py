"""Fig. 2: cost of a static setting. Top: per model, geomean over its applications of the default setting, the best
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

fig, (a, b) = plt.subplots(2, 1, figsize=(ps.COL, 2.6), gridspec_kw={'height_ratios': [1.35, 1]})
w = 0.26
for i, m in enumerate(models):
    single, rs, rd, over, _ = analyse(m, B)
    apps = list(rs)
    gd, gs = st.geometric_mean(rd.values()), st.geometric_mean(rs.values())
    for k, (v, g, col, hat, lab) in enumerate(((rd, gd, 'white', '////', 'Default setting'), (rs, gs, ps.BLUE, '', 'Best single setting'),
                                              ({t: 1.0 for t in apps}, 1.0, ps.GREEN, 'xxxx', 'Best per application'))):
        x = i + (k - 1) * w
        a.bar(x, g, w, color=col, hatch=hat, edgecolor='black', zorder=2, label=lab if i == 0 else None)
        if k < 2:
            xs = x + np.linspace(-w * 0.28, w * 0.28, len(apps))
            ys = np.array([v[t] for t in apps])
            red = np.array([over[t] for t in apps]) if k == 0 else np.zeros(len(apps), bool)
            a.scatter(xs[~red], ys[~red], s=5, color='black', zorder=3, lw=0)
            a.scatter(xs[red], ys[red], s=9, color=ps.RED, marker='x', zorder=3, lw=0.8)
    print(f'{MODEL_NAMES[m]}: default geomean {gd:.3f} (over budget for {sum(over.values())}/{len(apps)}), single {single} geomean {gs:.3f}, worst {min(rs.values()):.3f}')
a.set_xticks(range(len(models)))
a.set_xticklabels([MODEL_NAMES[m] for m in models])
a.set_ylim(0.4, 1.32)
a.set_ylabel('PIM throughput\n(norm. to per-app best)')
ps.top_legend(a, 3)
a.text(0.99, 0.03, '× = app over the 2% budget', transform=a.transAxes, fontsize=6, color=ps.RED, ha='right', va='bottom')
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
b.set_ylabel('Best single setting\n(geomean, norm.)')
b.set_ylim(0.55, 1.12)
b.legend(loc='lower right', ncol=1, fontsize=6)
fig.tight_layout(h_pad=0.6)
fig.align_ylabels([a, b])
os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(HERE, 'out', f'fig_static.{ext}'), dpi=300)
