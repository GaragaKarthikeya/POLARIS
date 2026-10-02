"""Fig. 1 (draft): cost of a static setting vs. CPU-slowdown budget, from the faithful sweep (sd.json)."""
import json, os, statistics as st, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as ps
ps.apply()
import matplotlib.pyplot as plt

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'sweep.json')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
os.makedirs(OUT, exist_ok=True)
D = json.load(open(SRC))
APPS = [('10', 'Tencent\nMeeting'), ('22', 'Browser'), ('51', 'Video'), ('sp70', '519.lbm')]
COSM = '128:16:12:480:0'
M = {}
for t, _ in APPS:
    cf = D[t]
    rots = sorted(set.intersection(*[set(v) for v in cf.values()]))
    M[t] = {c: (st.mean(cf[c][r]['thp'] for r in rots), st.mean(cf[c][r]['slow'] for r in rots)) for c in cf}

def analyse(bud):
    best = {t: max((c for c in M[t] if M[t][c][1] <= bud), key=lambda c: M[t][c][0]) for t, _ in APPS}
    cand = [c for c in M['10'] if all(M[t][c][1] <= bud for t, _ in APPS)]
    single = max(cand, key=lambda c: st.geometric_mean([M[t][c][0] / M[t][best[t]][0] for t, _ in APPS]))
    return best, single

fig, (a, b) = plt.subplots(1, 2, figsize=(ps.DCOL * 0.62, 1.7), gridspec_kw={'width_ratios': [2.3, 1]})
best, single = analyse(2.0)
n, w = len(APPS) + 1, 0.26
rows = {'COSM default': [M[t][COSM][0] / M[t][best[t]][0] for t, _ in APPS],
        'Best single static': [M[t][single][0] / M[t][best[t]][0] for t, _ in APPS],
        'Per-application best': [1.0 for _ in APPS]}
style = {'COSM default': ('white', '////'), 'Best single static': (ps.BLUE, ''), 'Per-application best': (ps.GREEN, 'xxxx')}
a.set_ylim(0.6, 1.15)
ps.shade_group(a, n - 1 - 0.45, n - 1 + 0.45)
for k, (lab, v) in enumerate(rows.items()):
    vals = v + [st.geometric_mean(v)]
    bars = a.bar(np.arange(n) + (k - 1) * w, vals, w, color=style[lab][0], hatch=style[lab][1], edgecolor='black', label=lab, zorder=2)
    if lab == 'COSM default':
        for j, (t, _) in enumerate(APPS):
            if M[t][COSM][1] > 2.0:
                a.text(j - w, vals[j] + 0.01, '×', ha='center', va='bottom', fontsize=7, color=ps.RED)
a.set_xticks(range(n)); a.set_xticklabels([x for _, x in APPS] + ['GMEAN'])
a.set_xlim(-0.55, n - 0.45)
a.set_ylabel('PIM Throughput\n(norm. to per-app best)')
ps.top_legend(a, 3)
a.text(0.60, 0.92, '× = exceeds 2% CPU budget', transform=a.transAxes, fontsize=6, color=ps.RED)

buds = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
g, wst = [], []
for bud in buds:
    try:
        bst, sg = analyse(bud)
    except ValueError:
        g.append(np.nan); wst.append(np.nan); continue
    r = [M[t][sg][0] / M[t][bst[t]][0] for t, _ in APPS]
    g.append(st.geometric_mean(r)); wst.append(min(r))
b.plot(buds, g, color=ps.NAVY, marker='s', mec='black', mew=0.4, label='GMEAN')
b.plot(buds, wst, color=ps.PURPLE, marker='^', ls='--', mec='black', mew=0.4, label='Worst app')
b.set_xlabel('CPU-Slowdown Budget (%)')
b.set_ylabel('Best Single Static\n(norm. to per-app best)')
b.set_ylim(0.6, 1.05)
b.legend(loc='lower right')
fig.tight_layout(w_pad=1.0)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(OUT, f'fig_static.{ext}'), dpi=300)
print('single @2%:', single, ' per-app best:', best)
print('budget sweep gmean:', [f'{x:.3f}' for x in g], ' worst:', [f'{x:.3f}' for x in wst])
