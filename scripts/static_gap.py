"""Cost of a static setting (Sec. II of the letter): per-application best setting under a CPU-slowdown budget vs.
the best single setting for all applications of a model vs. COSM's default.
usage: python3 scripts/static_gap.py [data/sweep.json] [--models bloom,...] [--budgets 2,3,5] [--select abc --evaluate de]
With --select/--evaluate, settings are chosen on one set of trace rotations and scored on the others."""
import argparse, os, statistics as st
from sweepdata import Sweep, eff_budget, pick_single
from workloads import NAMES, MODEL_NAMES, COSM_DEFAULT

ap = argparse.ArgumentParser()
ap.add_argument('data', nargs='?', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'sweep.json'))
ap.add_argument('--models', default='')
ap.add_argument('--apps', default='', help='restrict to these applications')
ap.add_argument('--budgets', default='1.5,2,3,5,100')
ap.add_argument('--select', default='')
ap.add_argument('--evaluate', default='')
ap.add_argument('--quiet', action='store_true', help='only the per-model summary lines')
a = ap.parse_args()
S = Sweep(a.data)
summary = []
for m in (a.models.split(',') if a.models else S.models()):
    apps = a.apps.split(',') if a.apps else S.apps(m)
    T_sel = {t: S.table(m, t, list(a.select) or None) for t in apps}
    T_eval = {t: S.table(m, t, list(a.evaluate or a.select) or None) for t in apps}
    for bud0 in map(float, a.budgets.split(',')):
        EB = {t: eff_budget(T_sel[t], bud0) for t in apps}
        bud = bud0
        best = {t: max((c for c in T_sel[t] if T_sel[t][c][1] <= EB[t]), key=lambda c: T_sel[t][c][0]) for t in apps}
        single, single_over = pick_single(T_sel, EB, best)
        oracle = {t: max((c for c in T_eval[t] if T_eval[t][c][1] <= max(EB[t], eff_budget(T_eval[t], bud0))), key=lambda c: T_eval[t][c][0]) for t in apps}
        rel = lambda t, c: T_eval[t][c][0] / T_eval[t][oracle[t]][0]
        g_single = st.geometric_mean([rel(t, single) for t in apps])
        worst = min(apps, key=lambda t: rel(t, single))
        over = [t for t in apps if T_eval[t][COSM_DEFAULT][1] > EB[t]]
        line = (f'{MODEL_NAMES[m]:17s} budget {bud:5g}%: best single {single:18s} [over budget for {len(single_over)}] geomean {g_single:.1%}, worst {rel(worst, single):.1%} '
                f'({NAMES[worst]}) | default over budget for {len(over)}/{len(apps)} apps'
                + (f', max {max(T_eval[t][COSM_DEFAULT][1] for t in over):.2f}%' if over else ''))
        print(line)
        summary.append((m, bud, g_single))
        if not a.quiet:
            for t in apps:
                print(f'    {NAMES[t]:16s} per-app best {best[t]:18s} {rel(t, best[t]):6.1%} (cpu {T_eval[t][best[t]][1]:.2f}%) | '
                      f'single {rel(t, single):6.1%} (cpu {T_eval[t][single][1]:.2f}%) | default {rel(t, COSM_DEFAULT):6.1%} '
                      f'(cpu {T_eval[t][COSM_DEFAULT][1]:.2f}%{", over" if t in over else ""})')
