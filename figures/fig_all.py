"""Results at every evaluation point (3 models x 12 CPU applications, COSM's evaluation set) under a 2% CPU-slowdown
budget. Top: PIM throughput of the default setting, the best single setting for the model, and POLARIS after
convergence, normalized to the best setting for each application. Bottom: CPU slowdown of the default setting and of
POLARIS.  usage: python3 figures/fig_all.py [data/sweep.json] [data/emulate]"""
import json, os, statistics as st, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'scripts'))
import paper_style as ps
ps.apply()
import matplotlib.pyplot as plt
from sweepdata import Sweep, eff_budget, pick_single
from workloads import MODEL_NAMES, NAMES, WORKLOADS, COSM_DEFAULT

S = Sweep(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'data', 'sweep.json'))
EMU = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, '..', 'data', 'emulate')
B = 2.0
LAST = 20                     # POLARIS after convergence: the last 20 of 200 decode steps
SHORT = {'10': 'TM', '22': 'Browser', '30': 'X', '40': 'Note', '51': 'Video', '60': 'Music', 'pa0': 'Ludcmp',
         'pa1': 'Covar.', 'pa2': 'Floyd', 'sp70': 'lbm', 'sp71': 'omnetpp', 'sp72': 'povray'}
models = [m for m in ('bloom', 'deepseek', 'qwen2') if m in S.models()]

rows = []                     # (model, app, default, single, polaris, cpu_default, cpu_polaris, default_over)
for m in models:
    apps = [t for t in WORKLOADS if t in S.D[m]]
    T = {t: S.table(m, t) for t in apps}
    EB = {t: eff_budget(T[t], B) for t in apps}
    best = {t: max((c for c in T[t] if T[t][c][1] <= EB[t]), key=lambda c: T[t][c][0]) for t in apps}
    single, _ = pick_single({t: T[t] for t in apps}, EB, best)
    for t in apps:
        f = os.path.join(EMU, f'{m}_converge_{t}.json')
        if os.path.exists(f):
            d = json.load(open(f))
            pol = float(np.mean(np.array(d['thp'])[:, -LAST:])) / T[t][best[t]][0]
            pcpu = float(np.mean(np.array(d['slow'])[:, -LAST:]))
        else:
            pol, pcpu = np.nan, np.nan
        rows.append((m, t, T[t][COSM_DEFAULT][0] / T[t][best[t]][0], T[t][single][0] / T[t][best[t]][0], pol,
                     T[t][COSM_DEFAULT][1], pcpu, T[t][COSM_DEFAULT][1] > EB[t]))

# x positions: 12 applications + GMEAN per model, a gap between models
xs, labels, groups = [], [], []
x = 0.0
for m in models:
    mr = [r for r in rows if r[0] == m]
    for r in mr:
        xs.append(x); labels.append(SHORT[r[1]]); x += 1
    xs.append(x); labels.append('GMEAN'); groups.append((m, xs[-len(mr) - 1], x)); x += 1.6
gm = lambda v: float(np.exp(np.nanmean(np.log(v))))
vals = {k: [] for k in ('d', 's', 'p', 'cd', 'cp', 'over')}
for m in models:
    mr = [r for r in rows if r[0] == m]
    for r in mr:
        for k, v in zip(('d', 's', 'p', 'cd', 'cp', 'over'), r[2:]):
            vals[k].append(v)
    for k, i in zip(('d', 's', 'p'), (2, 3, 4)):
        vals[k].append(gm([r[i] for r in mr]))
    vals['cd'].append(float(np.mean([r[5] for r in mr]))); vals['cp'].append(float(np.nanmean([r[6] for r in mr])))
    vals['over'].append(False)
xs = np.array(xs)

fig, (a, b) = plt.subplots(2, 1, figsize=(ps.DCOL, 2.3), sharex=True, gridspec_kw={'height_ratios': [1.55, 1]})
w = 0.27
import matplotlib.transforms as mtrans
from matplotlib.lines import Line2D
for m, x0, x1 in groups:
    b.text((x0 + x1) / 2, -0.95, MODEL_NAMES[m], ha='center', va='top', fontsize=7, fontweight='bold',
           transform=mtrans.blended_transform_factory(b.transData, b.transAxes))
for k, (key, col, hat, lab) in enumerate((('d', 'white', '////', 'Default setting'), ('s', ps.BLUE, '', 'Best single setting'),
                                          ('p', ps.GREEN, 'xxxx', 'POLARIS'))):
    a.bar(xs + (k - 1) * w, vals[key], w, color=col, hatch=hat, edgecolor='black', lw=0.4, label=lab, zorder=2)
over = np.array(vals['over'])
a.scatter(xs[over] - w, np.array(vals['d'])[over] + 0.05, marker='x', s=8, color=ps.RED, lw=0.8, zorder=3)
for m, x0, x1 in groups:
    for ax in (a, b):
        ax.add_patch(plt.Rectangle((x1 - 0.5, -10), 1.0, 20, facecolor=ps.SHADE, edgecolor='#BFBFBF', hatch='////', lw=0, zorder=0))
a.axhline(1.0, color='black', lw=0.6, zorder=1)
a.set_ylim(0.4, 1.38)
a.set_ylabel('PIM throughput\n(norm. to best)', labelpad=4)
h, l = a.get_legend_handles_labels()
h.append(Line2D([], [], marker='x', ls='', color=ps.RED, markersize=4, mew=0.8)); l.append('Default over the 2% budget')
a.legend(h, l, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=4)
b.bar(xs - w / 2, vals['cd'], w, color='white', hatch='////', edgecolor='black', lw=0.4, zorder=2)
b.bar(xs + w / 2, vals['cp'], w, color=ps.GREEN, hatch='xxxx', edgecolor='black', lw=0.4, zorder=2)
b.axhline(B, color=ps.RED, lw=0.8, ls='--', zorder=3)
b.set_ylim(0, 4.6)
b.set_ylabel('CPU slow-\ndown (%)', labelpad=4)
b.set_xticks(xs)
b.set_xticklabels(labels, rotation=90, fontsize=5.5)
b.set_xlim(xs[0] - 0.7, xs[-1] + 0.7)
fig.tight_layout(h_pad=0.2)
fig.align_ylabels([a, b])
os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(HERE, 'out', f'fig_all.{ext}'), dpi=300)
for m in models:
    mr = [r for r in rows if r[0] == m]
    print(f'{MODEL_NAMES[m]:17s} default {gm([r[2] for r in mr]):.3f} (over budget {sum(r[7] for r in mr)}/{len(mr)}, max CPU {max(r[5] for r in mr):.2f}%) | '
          f'single {gm([r[3] for r in mr]):.3f} (worst {min(r[3] for r in mr):.3f}) | POLARIS {gm([r[4] for r in mr]):.3f} '
          f'(worst {np.nanmin([r[4] for r in mr]):.3f}), CPU {np.nanmean([r[6] for r in mr]):.2f}% (max {np.nanmax([r[6] for r in mr]):.2f}%)')
