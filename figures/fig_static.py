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
for t, _ in APPS:   # mean over the rotations each setting has (a few runs crash in COSM's simulator)
    M[t] = {c: (st.mean(v[r]['thp'] for r in v), st.mean(v[r]['slow'] for r in v)) for c, v in D[t].items() if len(v) >= 4}

def analyse(bud):
    best = {t: max((c for c in M[t] if M[t][c][1] <= bud), key=lambda c: M[t][c][0]) for t, _ in APPS}
    cand = [c for c in M['10'] if all(c in M[t] and M[t][c][1] <= bud for t, _ in APPS)]
    single = max(cand, key=lambda c: st.geometric_mean([M[t][c][0] / M[t][best[t]][0] for t, _ in APPS]))
    return best, single

fig, (a, b) = plt.subplots(2, 1, figsize=(ps.COL, 2.45), gridspec_kw={'height_ratios': [1.45, 1]})
best, single = analyse(2.0)
n, w = len(APPS) + 1, 0.26
rows = {'Default setting': [M[t][COSM][0] / M[t][best[t]][0] for t, _ in APPS],
        'Best single setting': [M[t][single][0] / M[t][best[t]][0] for t, _ in APPS],
        'Best per application': [1.0 for _ in APPS]}
style = {'Default setting': ('white', '////'), 'Best single setting': (ps.BLUE, ''), 'Best per application': (ps.GREEN, 'xxxx')}
a.set_ylim(0.6, 1.2)
ps.shade_group(a, n - 1 - 0.45, n - 1 + 0.45)
for k, (lab, v) in enumerate(rows.items()):
    vals = v + [st.geometric_mean(v)]
    bars = a.bar(np.arange(n) + (k - 1) * w, vals, w, color=style[lab][0], hatch=style[lab][1], edgecolor='black', label=lab, zorder=2)
    if lab == 'Default setting':
        for j, (t, _) in enumerate(APPS):
            if M[t][COSM][1] > 2.0:
                a.text(j - w, vals[j] + 0.01, '×', ha='center', va='bottom', fontsize=7, color=ps.RED)
a.set_xticks(range(n)); a.set_xticklabels([x for _, x in APPS] + ['GMEAN'])
a.set_xlim(-0.55, n - 0.45)
a.set_ylabel('PIM throughput\n(norm. to per-app best)')
ps.top_legend(a, 3)
a.text(0.99, 0.95, '× = exceeds the 2% CPU budget', transform=a.transAxes, fontsize=6, color=ps.RED, ha='right', va='top')

buds = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
g, wst = [], []
for bud in buds:
    try:
        bst, sg = analyse(bud)
    except ValueError:
        g.append(np.nan); wst.append(np.nan); continue
    r = [M[t][sg][0] / M[t][bst[t]][0] for t, _ in APPS]
    g.append(st.geometric_mean(r)); wst.append(min(r))
b.plot(buds, g, color=ps.NAVY, marker='s', mec='black', mew=0.4, label='Geomean')
b.plot(buds, wst, color=ps.PURPLE, marker='^', ls='--', mec='black', mew=0.4, label='Worst application')
b.set_xlabel('CPU-slowdown budget (%)')
b.set_ylabel('Best single setting\n(norm. to per-app best)')
b.set_ylim(0.55, 1.08)
b.legend(loc='lower right', ncol=2)
fig.tight_layout(h_pad=0.6)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(OUT, f'fig_static.{ext}'), dpi=300)
print('single @2%:', single, ' per-app best:', best)
print('budget sweep gmean:', [f'{x:.3f}' for x in g], ' worst:', [f'{x:.3f}' for x in wst])
