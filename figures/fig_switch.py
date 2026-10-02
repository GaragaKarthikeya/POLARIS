"""Fig. 4: the operating system switches the CPU among four applications; POLARIS keeps one model per application.
usage: python3 figures/fig_switch.py [data/emulate/switch.json]"""
import json, os, sys, statistics as st
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as ps
ps.apply()
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
f = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'data', 'emulate', 'switch.json')
d = json.load(open(f))
B = 2.0
DEFAULT = '128:16:12:480:0'
NAMES = {'10': 'Tencent', '22': 'Browser', '51': 'Video', 'sp70': 'lbm'}
SHADE = {'10': '#DCE6F2', '22': '#F2DCDB', '51': '#EBF1DE', 'sp70': '#E4DFEC'}
sched, S = d['schedule'], d['static']
apps = list(dict.fromkeys(sched))
best = {a: max((c for c in S[a] if S[a][c][1] <= B), key=lambda c: S[a][c][0]) for a in apps}
cand = [c for c in S[apps[0]] if all(c in S[a] and S[a][c][1] <= B for a in apps)]
single = max(cand, key=lambda c: st.geometric_mean([S[a][c][0] / S[a][best[a]][0] for a in apps]))
q = np.array(d['thp']) / np.array([S[a][best[a]][0] for a in sched])
s = np.array(d['slow'])
x = np.arange(1, len(sched) + 1)
ref_single = np.array([S[a][single][0] / S[a][best[a]][0] for a in sched])
ref_def = np.array([S[a][DEFAULT][0] / S[a][best[a]][0] for a in sched])
cpu_single = np.array([S[a][single][1] for a in sched])
cpu_def = np.array([S[a][DEFAULT][1] for a in sched])

def smooth(v, k=3):
    return np.array([v[max(0, i - k + 1): i + 1].mean() for i in range(len(v))])

fig, (a, b) = plt.subplots(2, 1, figsize=(ps.DCOL, 1.95), sharex=True, gridspec_kw={'height_ratios': [1.6, 1]})
starts = [0] + [i for i in range(1, len(sched)) if sched[i] != sched[i - 1]] + [len(sched)]
for i0, i1 in zip(starts[:-1], starts[1:]):
    for ax in (a, b):
        ax.axvspan(i0 + 0.5, i1 + 0.5, color=SHADE[sched[i0]], lw=0, zorder=0)
    a.text((i0 + i1) / 2 + 0.5, 1.21, NAMES[sched[i0]], ha='center', va='center', fontsize=6)
a.step(x, ref_def, where='mid', color=ps.DGRAY, lw=1.0, ls=':', label='Default setting')
a.step(x, ref_single, where='mid', color=ps.BLUE, lw=1.1, ls='--', label='Best single setting')
a.plot(x, smooth(q.mean(0)), color=ps.NAVY, lw=1.2, label='POLARIS (mean of 10 seeds)')
a.axhline(1.0, color='black', lw=0.6)
a.set_ylim(0.45, 1.27)
a.set_ylabel('PIM throughput\n(norm.)', labelpad=2)
a.set_yticks([0.6, 0.8, 1.0, 1.2])
a.legend(loc='lower center', bbox_to_anchor=(0.5, 1.07), ncol=3)
b.step(x, cpu_def, where='mid', color=ps.DGRAY, lw=1.0, ls=':')
b.step(x, cpu_single, where='mid', color=ps.BLUE, lw=1.1, ls='--')
b.plot(x, smooth(s.mean(0)), color=ps.NAVY, lw=1.2)
b.axhline(B, color='black', lw=0.8)
b.set_ylabel('CPU slow-\ndown (%)', labelpad=8)
b.set_yticks([0, 2, 4])
b.set_xlabel('Decode step (the operating system switches applications every 40 steps)')
b.set_xlim(0.5, len(sched) + 0.5)
b.set_ylim(0, 4.5)
fig.tight_layout(h_pad=0.3)
os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(HERE, 'out', f'fig_switch.{ext}'), dpi=300)
for k, (i0, i1) in enumerate(zip(starts[:-1], starts[1:])):
    app = sched[i0]
    print(f'slice {k + 1:2d} {NAMES[app]:8s} first 5: {q[:, i0:i0 + 5].mean():.3f} last 10: {q[:, i1 - 10:i1].mean():.3f} '
          f'| single {ref_single[i0]:.3f} default {ref_def[i0]:.3f} | CPU {s[:, i0:i1].mean():.2f}% (default {cpu_def[i0]:.2f}%)')
print(f'whole schedule: POLARIS {q.mean():.3f}, single {ref_single.mean():.3f} ({single}), default {ref_def.mean():.3f}; '
      f'CPU POLARIS {s.mean():.2f}%, default {cpu_def.mean():.2f}%')
