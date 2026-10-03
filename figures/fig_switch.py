"""Fig. 4: the operating system round-robins the CPU among all twelve applications (15 decode steps per slice,
four rounds) under a 2% CPU-slowdown budget. POLARIS keeps one model per application. One curve per LLM.
usage: python3 figures/fig_switch.py [data/emulate]"""
import json, os, statistics as st, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'scripts'))
import paper_style as ps
ps.apply()
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
from workloads import MODEL_NAMES

DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'data', 'emulate')
B = 2.0
DEFAULT = '128:16:12:480:0'
SHORT = {'10': 'TM', '22': 'Brw', '30': 'X', '40': 'Note', '51': 'Vid', '60': 'Mus', 'pa0': 'Lud', 'pa1': 'Cov',
         'pa2': 'Flo', 'sp70': 'lbm', 'sp71': 'omn', 'sp72': 'pov'}
SHADES = ['#DCE6F2', '#F2DCDB', '#EBF1DE', '#E4DFEC', '#DAEEF3', '#FDE9D9', '#F2F2F2', '#DDD9C4', '#C6D9F1', '#E6B9B8', '#D7E4BD', '#CCC1DA']
sty = {'bloom': (ps.NAVY, '-'), 'deepseek': (ps.RED, '--'), 'qwen2': (ps.GREEN, '-.')}

def smooth(v, k=3):
    return np.array([v[max(0, i - k + 1): i + 1].mean() for i in range(len(v))])

fig, (a, b) = plt.subplots(2, 1, figsize=(ps.DCOL, 1.9), sharex=True, gridspec_kw={'height_ratios': [1.4, 1.1]})
sched = None
for m in ('bloom', 'deepseek', 'qwen2'):
    f = os.path.join(DIR, f'{m}_switch.json')
    if not os.path.exists(f):
        continue
    d = json.load(open(f))
    sched, S = d['schedule'], d['static']
    apps = list(dict.fromkeys(sched))
    best = {t: max((c for c in S[t] if S[t][c][1] <= max(B, min(v[1] for v in S[t].values()))), key=lambda c: S[t][c][0]) for t in apps}
    EB = {t: max(B, min(v[1] for v in S[t].values())) for t in apps}
    common = set.intersection(*[set(S[t]) for t in apps])
    single = max(common, key=lambda c: (-sum(S[t][c][1] > EB[t] for t in apps), st.geometric_mean([S[t][c][0] / S[t][best[t]][0] for t in apps])))
    q = np.array(d['thp']) / np.array([S[t][best[t]][0] for t in sched])
    s = np.array(d['slow'])
    c, ls = sty[m]
    x = np.arange(1, len(sched) + 1)
    a.plot(x, smooth(q.mean(0)), color=c, ls=ls, lw=1.0, label=MODEL_NAMES[m])
    b.plot(x, smooth(s.mean(0)), color=c, ls=ls, lw=1.0)
    n1 = len(apps) * (len(sched) // len(apps) // (len(sched) // len(apps) // 1)) if False else None
    first = [i for i in range(len(sched)) if sched[i] != (sched[i - 1] if i else None)]
    rounds = len(first) // len(apps)
    per_round = [np.mean([q[:, i0:i0 + 15].mean() for i0 in first[r * len(apps):(r + 1) * len(apps)]]) for r in range(rounds)]
    single_g = float(np.mean([S[t][single][0] / S[t][best[t]][0] for t in sched]))
    default_g = float(np.mean([S[t][DEFAULT][0] / S[t][best[t]][0] for t in sched]))
    print(f'{MODEL_NAMES[m]:17s} POLARIS {q.mean():.3f} (per round ' + ' '.join(f'{v:.3f}' for v in per_round) +
          f'), first 3 steps of later visits {np.mean([q[:, i0:i0 + 3].mean() for i0 in first[len(apps):]]):.3f} | '
          f'single {single_g:.3f} | default {default_g:.3f} | CPU POLARIS {s.mean():.2f}% (later rounds {s[:, first[len(apps)]:].mean():.2f}%), '
          f'default {np.mean([S[t][DEFAULT][1] for t in sched]):.2f}%')
starts = [i for i in range(len(sched)) if i == 0 or sched[i] != sched[i - 1]] + [len(sched)]
apps = list(dict.fromkeys(sched))
for k, (i0, i1) in enumerate(zip(starts[:-1], starts[1:])):
    col = SHADES[apps.index(sched[i0]) % len(SHADES)]
    for ax in (a, b):
        ax.axvspan(i0 + 0.5, i1 + 0.5, color=col, lw=0, zorder=0)
    a.text((i0 + i1) / 2 + 0.5, 1.02 if k % 2 == 0 else 1.11, SHORT[sched[i0]], ha='center', va='bottom', fontsize=5,
           transform=mtrans.blended_transform_factory(a.transData, a.transAxes))
for r in range(1, len(starts[:-1]) // len(apps)):
    x0 = starts[r * len(apps)] + 0.5
    for ax in (a, b):
        ax.axvline(x0, color='black', lw=0.8)
    a.text(x0 + 3, 0.52, f'round {r + 1}', ha='left', va='center', fontsize=6, fontweight='bold')
a.text(4, 0.52, 'round 1', ha='left', va='center', fontsize=6, fontweight='bold')
a.axhline(1.0, color='black', lw=0.6)
a.set_ylim(0.45, 1.2)
a.set_ylabel('PIM throughput\n(norm. to best)', labelpad=4)
h_, l_ = a.get_legend_handles_labels()
b.legend(h_, l_, loc='lower right', ncol=3, fontsize=6)
b.axhline(B, color='black', lw=0.8)
b.set_ylim(0, 4.6)
b.set_ylabel('CPU slow-\ndown (%)', labelpad=4)
b.set_xlabel('Decode step (the operating system switches applications every 15 steps)')
b.set_xlim(0.5, len(sched) + 0.5)
fig.tight_layout(h_pad=0.2)
fig.align_ylabels([a, b])
os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(HERE, 'out', f'fig_switch.{ext}'), dpi=300)
