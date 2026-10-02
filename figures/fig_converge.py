"""Fig. 3: POLARIS learning from an empty model while one application runs (2% CPU-slowdown budget).
usage: python3 figures/fig_converge.py [data/emulate/converge_<app>.json ...]"""
import glob, json, os, sys, statistics as st
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as ps
ps.apply()
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
files = sys.argv[1:] or sorted(glob.glob(os.path.join(HERE, '..', 'data', 'emulate', 'converge_*.json')))
NAMES = {'10': 'Tencent Meeting', '22': 'Browser', '51': 'Video', 'sp70': '519.lbm'}
STY = {'10': (ps.NAVY, 's', '-'), '22': (ps.RED, 'o', '--'), '51': (ps.GREEN, '^', '-.'), 'sp70': (ps.PURPLE, 'D', ':')}
B = 2.0

def smooth(x, k=5):
    return np.array([x[max(0, i - k + 1): i + 1].mean() for i in range(len(x))])

runs = {}
for f in files:
    d = json.load(open(f))
    app = d['schedule'][0]
    S = d['static'][app]
    best = max((c for c in S if S[c][1] <= B), key=lambda c: S[c][0])
    runs[app] = (np.array(d['thp']) / S[best][0], np.array(d['slow']))
# best single setting across the applications (same definition as static_gap.py)
statics = {app: json.load(open(f))['static'][json.load(open(f))['schedule'][0]] for f in files for app in [json.load(open(f))['schedule'][0]]}
bestapp = {a: max((c for c in s if s[c][1] <= B), key=lambda c: s[c][0]) for a, s in statics.items()}
cand = [c for c in statics[next(iter(statics))] if all(c in s and s[c][1] <= B for s in statics.values())]
single = max(cand, key=lambda c: st.geometric_mean([statics[a][c][0] / statics[a][bestapp[a]][0] for a in statics]))
single_g = st.geometric_mean([statics[a][single][0] / statics[a][bestapp[a]][0] for a in statics])

fig, (a, b) = plt.subplots(2, 1, figsize=(ps.COL, 2.05), sharex=True, gridspec_kw={'height_ratios': [1.7, 1]})
n = next(iter(runs.values()))[0].shape[1]
x = np.arange(1, n + 1)
a.axhline(1.0, color='black', lw=0.8, ls='-', zorder=1)
a.axhline(single_g, color=ps.DGRAY, lw=1.0, ls='--', zorder=1)
a.text(n, single_g - 0.012, f'best single setting (geomean {single_g:.2f})', ha='right', va='top', fontsize=6, color=ps.DGRAY)
a.text(n, 1.075, 'best setting per application = 1.0', ha='right', va='bottom', fontsize=6)
for app in ('10', '22', '51', 'sp70'):
    if app not in runs:
        continue
    c, m, ls = STY[app]
    q, s = runs[app]
    a.plot(x, smooth(q.mean(0)), color=c, ls=ls, marker=m, markevery=25, mec='black', mew=0.4, label=NAMES[app])
    b.plot(x, smooth(s.mean(0)), color=c, ls=ls, marker=m, markevery=25, mec='black', mew=0.4)
a.set_ylabel('PIM throughput\n(norm. to best)')
a.set_ylim(0.55, 1.15)
ps.top_legend(a, 4, handlelength=1.8)
b.axhline(B, color='black', lw=0.8)
b.text(n, B - 0.25, '2% budget', ha='right', va='top', fontsize=6)
b.set_ylabel('CPU\nslowdown (%)')
b.set_xlabel('Decode step')
b.set_xlim(1, n)
b.set_ylim(0, 4.5)
fig.tight_layout(h_pad=0.3)
os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(HERE, 'out', f'fig_converge.{ext}'), dpi=300)
for app, (q, s) in runs.items():
    m = q.mean(0)
    r95 = next((i + 1 for i in range(len(m) - 9) if m[i:i + 10].mean() >= 0.95), None)
    print(f'{NAMES[app]:16s} first 10: {q[:, :10].mean():.3f}  last 20: {q[:, -20:].mean():.3f}  steps to 95%: {r95}  '
          f'CPU first 25: {s[:, :25].mean():.2f}%  last 50: {s[:, -50:].mean():.2f}%')
print(f'best single setting {single}: geomean {single_g:.3f}')
