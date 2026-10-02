"""POLARIS: kernel-level runtime tuning agent for the COSM memory controller.

Decision unit : one PIM kernel invocation (one transformer-layer operator).
Context       : "<app>|<kernel>": the CPU application on the core (one model per application) and the PIM kernel
                (a per-kernel offset inside that application's model). Contexts without '|' belong to app "".
Action        : a full controller config  task_len:pred:bus:io:idle  (405 combinations; task length = native
                PIM command length nPTL, emitted by the PIM runtime for this kernel).
Reward        : the kernel's own latency (decode latency is the sum of kernel latencies, so this is the
                objective itself) with a Lagrangian penalty on the CPU slowdown: cost = lat * (1 + mu * (slow - b)).
Learner       : contextual bandit (gamma = 0: one kernel's config does not change the next kernel's latency).
                Per application, two Bayesian linear models over knob features (per-level one-hots + per-kernel
                offset; pairwise level combinations are optional and did not help) predict log-latency and CPU slowdown. Thompson sampling
                picks the config. mu (per application) follows dual ascent on the measured decode-level slowdown.
"""
import json
import random

import numpy as np

LV = [[16, 32, 64, 128, 256], [8, 16, 32], [2, 6, 12], [240, 480, 960], [0, 50, 150]]
CONFIGS = [f'{a}:{b}:{c}:{d}:{e}' for a in LV[0] for b in LV[1] for c in LV[2] for d in LV[3] for e in LV[4]]
IDX = {c: i for i, c in enumerate(CONFIGS)}
MAX_KERNELS = 16


def _cfg_features(c, pairwise=True):
    v = [LV[j].index(int(x)) for j, x in enumerate(c.split(':'))]
    f = []
    for j in range(5):
        f += [1.0 if v[j] == k else 0.0 for k in range(len(LV[j]))]
    if pairwise:
        for a in range(5):
            for b in range(a + 1, 5):
                f += [1.0 if (v[a] == i and v[b] == k) else 0.0 for i in range(len(LV[a])) for k in range(len(LV[b]))]
    return f


def split_ctx(ctx):
    return tuple(ctx.split('|', 1)) if '|' in ctx else ('', ctx)


class Agent:
    def __init__(self, budget=0.05, seed=1, mode='ts', start='128:16:12:480:0', pairwise=0, prior_sd=0.3,
                 lat_noise=0.10, slow_noise=0.02, eta=1.0, mu_max=100.0, eps=0.1, budget_in_pct=0,
                 constraint='lagrange', margin_eta=0.25, ts_scale=0.3, mu0=5.0):
        self.budget = float(budget) / (100.0 if int(budget_in_pct) else 1.0)
        self.mode, self.start = mode, start
        self.prior_sd, self.lat_noise, self.slow_noise = float(prior_sd), float(lat_noise), float(slow_noise)
        self.eta, self.mu_max, self.eps = float(eta), float(mu_max), float(eps)
        # 'lagrange': minimise lat*(1+mu*(s-b)), mu by dual ascent (budget met on average)
        # 'ctsm'    : constrained Thompson sampling, feasible iff sampled s <= b - margin; the margin is adapted from
        #             the measured slowdown so that the long-run slowdown tracks b
        self.constraint, self.margin_eta = constraint, float(margin_eta)
        self.ts_scale, self.mu0 = float(ts_scale), float(mu0)   # posterior-draw temperature; initial budget multiplier
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.X = np.array([_cfg_features(c, bool(int(pairwise))) for c in CONFIGS])      # 405 x Fc
        self.Fc = self.X.shape[1]
        self.F = self.Fc + MAX_KERNELS
        self.apps = {}                      # app -> model dict
        self.tab = {}                       # (ctx, cfg) -> [n, sum lat, sum cpu_lat]   (ablation 'tab')
        self.fac = {}                       # (ctx, knob, value) -> [n, sum rel cost]   (ablation 'factored')

    # ---------------------------------------------------------------- per-application model
    def _new_model(self):
        prior = np.full(self.F, 1.0 / self.prior_sd ** 2)
        prior[self.Fc:] = 1e-4                                  # per-kernel offsets: nearly flat prior
        return dict(A=[np.diag(prior), np.diag(prior)], b=[np.zeros(self.F), np.zeros(self.F)],
                    kern={}, mu=self.mu0, margin=0.0, dec=[0.0, 0.0], cache=None)

    def _model(self, app):
        if app not in self.apps:
            self.apps[app] = self._new_model()
        return self.apps[app]

    def mu(self, ctx):
        return self._model(split_ctx(ctx)[0])['mu']

    # ---------------------------------------------------------------- persistence
    def save(self, path):
        apps = {a: dict(A=[x.tolist() for x in m['A']], b=[x.tolist() for x in m['b']], kern=m['kern'], mu=m['mu'], margin=m['margin'])
                for a, m in self.apps.items()}
        d = dict(apps=apps, tab={'\t'.join(k): v for k, v in self.tab.items()},
                 fac={'\t'.join(map(str, k)): v for k, v in self.fac.items()},
                 rng=self.rng.getstate(), np_rng=self.np_rng.bit_generator.state)
        json.dump(d, open(path, 'w'))

    def load(self, path):
        d = json.load(open(path))
        self.apps = {}
        for a, m in d['apps'].items():
            self.apps[a] = dict(A=[np.array(x) for x in m['A']], b=[np.array(x) for x in m['b']], kern=m['kern'],
                                mu=m['mu'], margin=m.get('margin', 0.0), dec=[0.0, 0.0], cache=None)
        self.tab = {tuple(k.split('\t')): v for k, v in d['tab'].items()}
        self.fac = {}
        for k, v in d['fac'].items():
            c, j, x = k.split('\t')
            self.fac[(c, int(j), x)] = v
        st = d['rng']
        self.rng.setstate((st[0], tuple(st[1]), st[2]))
        self.np_rng.bit_generator.state = d['np_rng']

    # ---------------------------------------------------------------- model helpers
    def _kernel_index(self, m, kernel):
        if kernel not in m['kern']:
            m['kern'][kernel] = len(m['kern'])
        return m['kern'][kernel]

    def _posterior(self, m):
        if m['cache'] is None:
            out = []
            for k in range(2):
                cov = np.linalg.inv(m['A'][k])
                cov = (cov + cov.T) / 2
                out.append((cov @ m['b'][k], np.linalg.cholesky(cov + 1e-12 * np.eye(self.F))))
            m['cache'] = out
        return m['cache']

    def predict(self, ctx, sample=False):
        """predicted latency and slowdown of every config for this kernel (posterior mean or a Thompson draw)."""
        app, kernel = split_ctx(ctx)
        m = self._model(app)
        k = self._kernel_index(m, kernel)
        ws = []
        for mean, L in self._posterior(m):
            ws.append(mean + self.ts_scale * (L @ self.np_rng.standard_normal(self.F)) if sample else mean)
        lat = np.exp(self.X @ ws[0][:self.Fc] + ws[0][self.Fc + k])
        slow = self.X @ ws[1][:self.Fc] + ws[1][self.Fc + k]
        return lat, slow

    def cost(self, lat, slow, mu):
        return lat * (1.0 + mu * (slow - self.budget))

    # ---------------------------------------------------------------- policy
    def choose(self, ctx):
        mu = self.mu(ctx)
        if self.mode == 'fixed':
            return self.start
        if self.mode == 'random':
            return self.rng.choice(CONFIGS)
        if self.mode == 'tab':                                  # per-config table, no generalisation, optimistic
            unv = [c for c in CONFIGS if (ctx, c) not in self.tab]
            if unv:
                return self.rng.choice(unv)
            return min(CONFIGS, key=lambda c: self.cost(self.tab[(ctx, c)][1] / self.tab[(ctx, c)][0],
                                                        1 - self.tab[(ctx, c)][2] / self.tab[(ctx, c)][1], mu))
        if self.mode == 'factored':                             # one independent eps-greedy bandit per knob
            if self.rng.random() < self.eps:
                return self.rng.choice(CONFIGS)
            pick = []
            for j in range(5):
                def score(x):
                    s = self.fac.get((ctx, j, str(x)))
                    return (s[1] / s[0] if s and s[0] else -1e9) + 1e-9 * self.rng.random()
                pick.append(str(min(LV[j], key=score)))
            return ':'.join(pick)
        lat, slow = self.predict(ctx, sample=True)              # Thompson sampling on the application's model
        if self.constraint in ('ctsm', 'ctsm_mean'):
            m = self._model(split_ctx(ctx)[0])
            if self.constraint == 'ctsm_mean':              # explore on latency, constrain on the posterior mean
                slow = self.predict(ctx, sample=False)[1]
            ok = slow <= self.budget - m['margin']
            if ok.any():
                return CONFIGS[int(np.argmin(np.where(ok, lat, np.inf)))]
            return CONFIGS[int(np.argmin(slow))]
        return CONFIGS[int(np.argmin(self.cost(lat, slow, mu)))]

    def best(self, ctx):
        lat, slow = self.predict(ctx, sample=False)
        if self.constraint in ('ctsm', 'ctsm_mean'):
            ok = slow <= self.budget - self._model(split_ctx(ctx)[0])['margin']
            return CONFIGS[int(np.argmin(np.where(ok, lat, np.inf)))] if ok.any() else CONFIGS[int(np.argmin(slow))]
        return CONFIGS[int(np.argmin(self.cost(lat, slow, self.mu(ctx))))]

    def update(self, ctx, cfg, lat, cpu_lat, weight):
        app, kernel = split_ctx(ctx)
        m = self._model(app)
        slow = 1.0 - cpu_lat / lat
        phi = np.zeros(self.F)
        phi[:self.Fc] = self.X[IDX[cfg]]
        phi[self.Fc + self._kernel_index(m, kernel)] = 1.0
        for k, (y, sd) in enumerate(((np.log(lat), self.lat_noise), (slow, self.slow_noise))):
            m['A'][k] += np.outer(phi, phi) / sd ** 2
            m['b'][k] += phi * y / sd ** 2
        t = self.tab.setdefault((ctx, cfg), [0, 0.0, 0.0])
        t[0] += 1; t[1] += lat; t[2] += cpu_lat
        ref = self.tab.setdefault((ctx, '*'), [0, 0.0, 0.0])
        ref[0] += 1; ref[1] += lat; ref[2] += cpu_lat
        rel = self.cost(lat, slow, m['mu']) / (ref[1] / ref[0])
        for j, x in enumerate(cfg.split(':')):
            f = self.fac.setdefault((ctx, j, x), [0, 0.0])
            f[0] += 1; f[1] += rel
        m['dec'][0] += weight * lat
        m['dec'][1] += weight * cpu_lat

    def end_decode(self):
        """once per decode step: refresh posteriors, and dual ascent on each application's CPU budget."""
        for m in self.apps.values():
            m['cache'] = None
            if m['dec'][0] > 0:
                slow = 1 - m['dec'][1] / m['dec'][0]
                m['mu'] = min(self.mu_max, max(0.0, m['mu'] + self.eta * (slow - self.budget) / self.budget))
                m['margin'] = min(self.budget, max(-0.5 * self.budget, m['margin'] + self.margin_eta * (slow - self.budget)))
            m['dec'] = [0.0, 0.0]
