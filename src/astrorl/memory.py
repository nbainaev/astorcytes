"""Calcium familiarity memory, neuronal EMA trace and inhibitory gating."""

import numpy as np
from numba import njit


@njit(cache=True)
def seed_rng(seed):
    np.random.seed(seed)


@njit(cache=True)
def ca_constant(ca, velocity, dt, tau, target):
    relative = ca - target
    b = velocity + relative / tau
    ec = np.exp(-dt / tau)
    return target + (relative + b * dt) * ec, (velocity - b * dt / tau) * ec


@njit(cache=True)
def ca_gap(g, ca, velocity, nsteps, dt, tau, amplitude, gthr):
    high = 0
    if g > gthr:
        high = min(nsteps, max(0, int(np.ceil(np.log(g / gthr) / dt)) - 1))
    if high > 0:
        ca, velocity = ca_constant(ca, velocity, high * dt, tau, amplitude)
    if nsteps > high:
        ca, velocity = ca_constant(ca, velocity, (nsteps - high) * dt, tau, 0.0)
    return g * np.exp(-dt * nsteps), ca, velocity


@njit(cache=True)
def calcium_step_sparse(g, ca, velocity, signal, duration, tau, amplitude, gthr):
    """Exact distribution of 1ms Poisson bins, skipping constant-forcing gaps.

    Poisson splitting: total events ~ Poisson(rate*T), bins iid uniform.
    This preserves the discrete 1ms model, without 1000 mostly empty updates.
    """
    nsteps = max(1, int(round(duration / 0.001)))
    dt = duration / nsteps
    eg = np.exp(-dt)
    for i in range(g.size):
        count = np.random.poisson(15.0 * signal[i] * duration)
        bins = np.sort(np.random.randint(0, nsteps, count))
        last = -1
        k = 0
        while k < count:
            current = bins[k]
            gg, cc, vv = ca_gap(
                g[i], ca[i], velocity[i], current - last - 1, dt, tau, amplitude, gthr
            )
            same = 1
            while k + same < count and bins[k + same] == current:
                same += 1
            gg = gg * eg + 0.2 * same
            cc, vv = ca_constant(cc, vv, dt, tau, amplitude if gg > gthr else 0.0)
            g[i] = gg
            ca[i] = cc
            velocity[i] = vv
            last = current
            k += same
        gg, cc, vv = ca_gap(g[i], ca[i], velocity[i], nsteps - last - 1, dt, tau, amplitude, gthr)
        g[i] = gg
        ca[i] = max(0.0, cc)
        velocity[i] = vv


class NeuronalTraceBank:
    def __init__(self, n=16, tau_s=3.0, gain=1.0, second_tau_s=None):
        if tau_s <= 0 or (second_tau_s is not None and second_tau_s <= 0):
            raise ValueError("tau must be positive")
        self.n = n
        self.tau = tau_s
        self.gain = gain
        self.second_tau = second_tau_s
        self.reset()

    def reset(self):
        self.trace = np.zeros(self.n)
        self.slow = np.zeros(self.n)

    def advance(self, real_feature, duration_s):
        x = np.asarray(real_feature)
        if x.shape != (self.n,) or duration_s < 0 or not np.isfinite(x).all() or (x < 0).any():
            raise ValueError("Invalid trace stimulus")
        decay = np.exp(-duration_s / self.tau)
        self.trace = decay * self.trace + self.gain * (1 - decay) * x
        if self.second_tau is not None:
            decay = np.exp(-duration_s / self.second_tau)
            self.slow = decay * self.slow + self.gain * (1 - decay) * x

    def active_memory(self):
        return np.clip(
            self.trace if self.second_tau is None else 0.5 * (self.trace + self.slow), 0, 1
        )

    def familiarity(self, candidates):
        x = np.asarray(candidates)
        return (x @ self.active_memory()) / np.maximum(x.sum(-1), 1e-8)

    def snapshot(self):
        return self.trace.copy(), self.slow.copy()


class FamiliarityMemory:
    def __init__(self, domains, kind, params):
        self.kind = kind
        self.p = params
        self.g = np.zeros(domains)
        self.ca = np.zeros(domains)
        self.velocity = np.zeros(domains)
        if kind == "trace":
            cfg = params[kind]
            self.trace = NeuronalTraceBank(domains, cfg["tau"], cfg["gain"], cfg.get("second_tau"))

    def write(self, u, seed):
        if self.kind == "trace":
            self.trace.advance(u, self.p["duration"])
        else:
            seed_rng(seed)
            cfg = self.p["astro"]
            calcium_step_sparse(
                self.g,
                self.ca,
                self.velocity,
                u,
                self.p["duration"],
                cfg["tau"],
                cfg["amplitude"],
                cfg["gthr"],
            )

    def active(self):
        if self.kind == "trace":
            return self.trace.active_memory()
        cfg = self.p["astro"]
        return np.clip(
            (self.ca - cfg["threshold"]) / max(cfg["amplitude"] - cfg["threshold"], 1e-8), 0, 1
        )

    def familiarity(self, u):
        return u @ self.active() / np.maximum(u.sum(-1), 1e-8)


def gating(f, params):
    mode = params["gate"]["mode"]
    contrast = params["gate"]["contrast"]
    if mode == "absolute":
        return np.clip(f, 0, params["gate"].get("max_gate", 1.0))
    relative = f - f.min()
    if mode == "thresholded_relative":
        relative = np.maximum(0, relative - params["gate"].get("minimum", 0.03))
    return np.clip(relative / max(contrast, 1e-8), 0, params["gate"].get("max_gate", 1.0))
