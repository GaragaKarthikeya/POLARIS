"""Fig. 3: POLARIS learning from an empty model with one application on the CPU (2% budget). One line per model:
geomean over its applications of the throughput of the chosen settings, normalized to each application's best
setting; band: range over applications.  usage: python3 figures/fig_converge.py [data/emulate]"""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'scripts'))
import paper_style as ps
ps.apply()
import matplotlib.pyplot as plt
from workloads import MODEL_NAMES

DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'data', 'emulate')
B = 2.0
sty = {'bloom': (ps.NAVY, 's', '-', ps.LBLUE), 'deepseek': (ps.RED, 'o', '--', '#F2DCDB'), 'qwen2': (ps.GREEN, '^', '-.', ps.LGREEN)}

def smooth(x, k=5):
    return np.array([x[max(0, i - k + 1): i + 1].mean() for i in range(len(x))])

fig, (a, b) = plt.subplots(2, 1, figsize=(ps.COL, 2.05), sharex=True, gridspec_kw={'height_ratios': [1.7, 1]})
for m in ('bloom', 'deepseek', 'qwen2'):
    files = sorted(glob.glob(os.path.join(DIR, f'{m}_converge_*.json')))
    if not files:
        continue
    Q, Sl = [], []
    for f in files:
        d = json.load(open(f))
        app = d['schedule'][0]
        st_ = d['static'][app]
        best = max((c for c in st_ if st_[c][1] <= max(B, min(v[1] for v in st_.values()))), key=lambda c: st_[c][0])
        Q.append(np.array(d['thp']).mean(0) / st_[best][0])
        Sl.append(np.array(d['slow']).mean(0))
    Q, Sl = np.array(Q), np.array(Sl)
    g = np.exp(np.log(Q).mean(0))
    x = np.arange(1, Q.shape[1] + 1)
    c, mk, ls, light = sty[m]
    a.fill_between(x, smooth(Q.min(0)), smooth(Q.max(0)), color=light, alpha=0.5, lw=0)
    a.plot(x, smooth(g), color=c, ls=ls, marker=mk, markevery=25, mec='black', mew=0.4, label=MODEL_NAMES[m])
    b.plot(x, smooth(Sl.mean(0)), color=c, ls=ls, marker=mk, markevery=25, mec='black', mew=0.4)
    r95 = next((i + 1 for i in range(len(g) - 9) if g[i:i + 10].mean() >= 0.95), None)
    print(f'{MODEL_NAMES[m]:17s} {len(files)} apps: first 10 {g[:10].mean():.3f}, last 20 {g[-20:].mean():.3f} (apps {Q[:, -20:].mean(1).min():.3f}-{Q[:, -20:].mean(1).max():.3f}), '
          f'steps to 95% {r95}, CPU first 25 {Sl[:, :25].mean():.2f}% last 50 {Sl[:, -50:].mean():.2f}% (max app {Sl[:, -50:].mean(1).max():.2f}%)')
a.axhline(1.0, color='black', lw=0.7)
a.set_ylabel('PIM throughput\n(norm. to best)', labelpad=4)
a.set_ylim(0.6, 1.15)
ps.top_legend(a, 3, handlelength=1.8)
b.axhline(B, color='black', lw=0.8)
b.text(x[-1], B + 0.3, '2% budget', ha='right', va='bottom', fontsize=6)
b.set_ylabel('CPU slow-\ndown (%)', labelpad=4)
b.set_xlabel('Decode step')
b.set_xlim(1, x[-1])
b.set_ylim(0, 4.5)
fig.tight_layout(h_pad=0.3)
fig.align_ylabels([a, b])
os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(HERE, 'out', f'fig_converge.{ext}'), dpi=300)
