"""Shared loader for data/sweep.json."""
import json, statistics as st
from workloads import kernel_weight


class Sweep:
    def __init__(self, path):
        d = json.load(open(path))
        self.layers, self.D = d['layers'], d['data']

    def models(self):
        return list(self.D)

    def apps(self, model):
        return list(self.D[model])

    def runs(self, model, app):
        """{setting: {rotation: {thp, slow, L}}}"""
        return self.D[model][app]

    def table(self, model, app, rots=None, min_rots=None):
        """{setting: (mean PIM throughput, mean CPU slowdown %)} over the requested rotations a setting has.
        Settings with fewer than min_rots of them are dropped (COSM's simulator occasionally crashes)."""
        out = {}
        for c, v in self.D[model][app].items():
            have = [r for r in (rots or v) if r in v]
            need = min_rots if min_rots is not None else max(1, len(rots or v) - 1)
            if len(have) >= need:
                out[c] = (st.mean(v[r]['thp'] for r in have), st.mean(v[r]['slow'] for r in have))
        return out

    def weight(self, model, kernel):
        return kernel_weight(kernel, self.layers[model])


def eff_budget(table, budget):
    """budget actually attainable at an evaluation point: if no setting meets it, the lowest slowdown any setting reaches"""
    return max(budget, min(v[1] for v in table.values()))

def pick_single(tables, eb, best):
    """best single setting for a set of applications: fewest applications over their (effective) budget, then the
    highest geomean throughput relative to each application's best setting. Returns (setting, apps over budget)."""
    import statistics as _st
    apps = list(tables)
    common = set.intersection(*[set(tables[t]) for t in apps])
    def key(c):
        over = sum(tables[t][c][1] > eb[t] for t in apps)
        return (-over, _st.geometric_mean([tables[t][c][0] / tables[t][best[t]][0] for t in apps]))
    c = max(common, key=key)
    return c, [t for t in apps if tables[t][c][1] > eb[t]]
