"""Local eligibility and fixed direct feedback for feedforward MLP controls."""

import copy
import hashlib
import numpy as np


def fingerprint(arrays):
    h = hashlib.sha256()
    for value in arrays:
        a = np.ascontiguousarray(value)
        h.update(str((a.shape, a.dtype)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def softmax(x):
    p = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return p / p.sum(axis=-1, keepdims=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def credit(records, baseline, entropy):
    ret = np.zeros(len(records))
    carry = 0.0
    for t in range(len(records) - 1, -1, -1):
        carry = records[t]["reward"] + 0.99 * carry
        ret[t] = carry
    delta = []
    for t, r in enumerate(records):
        p = r["probability"]
        lp = np.log(np.maximum(p, 1e-30))
        ent = -np.sum(p * lp)
        score = -p.copy()
        score[r["action"]] += 1.0
        delta.append(0.99**t / 50 * ((ret[t] - baseline[t]) * score - entropy * p * (lp + ent)))
    return np.asarray(delta), ret


class LocalMLP:
    def __init__(self, seed, inputs, outputs, zero_output=False):
        r = np.random.default_rng(seed)
        self.p = dict(
            w1=r.normal(0, 1 / np.sqrt(inputs), (inputs, 64)),
            b1=np.zeros(64),
            w2=r.normal(0, 1 / 8, (64, 64)),
            b2=np.zeros(64),
            w3=r.normal(0, 0.01, (64, outputs)),
            b3=np.zeros(outputs),
        )
        if zero_output:
            self.p["w3"][:] = 0.0
        self.B1 = r.normal(0, 0.01, (64, outputs))
        self.B2 = r.normal(0, 0.01, (64, outputs))
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = copy.deepcopy(self.m)
        self.t = {k: 0 for k in self.p}

    def forward(self, x):
        p = self.p
        x = np.asarray(x)
        a1 = x @ p["w1"] + p["b1"]
        h1 = np.maximum(a1, 0)
        a2 = h1 @ p["w2"] + p["b2"]
        h2 = np.maximum(a2, 0)
        return h2 @ p["w3"] + p["b3"], (x, h1, h2, a1, a2)

    def gradients(self, cache, delta, hidden=True):
        x, h1, h2, a1, a2 = cache
        if x.ndim == 1:
            x = x[None]
            h1 = h1[None]
            h2 = h2[None]
            a1 = a1[None]
            a2 = a2[None]
            delta = delta[None]
        g = dict(w3=h2.T @ delta, b3=delta.sum(0))
        if hidden:
            d1 = (delta @ self.B1.T) * (a1 > 0)
            d2 = (delta @ self.B2.T) * (a2 > 0)
            g.update(w1=x.T @ d1, b1=d1.sum(0), w2=h1.T @ d2, b2=d2.sum(0))
        return g

    def step(self, g, lr, hidden_lr=None, hidden_cap=None):
        norm = float(np.sqrt(sum(np.square(v).sum() for v in g.values())))
        hidden_before = {k: v.copy() for k, v in self.p.items() if k in g and k not in ("w3", "b3")}
        for k, value in g.items():
            value = value / max(1.0, norm / 5)
            self.t[k] += 1
            t = self.t[k]
            self.m[k] = 0.9 * self.m[k] + 0.1 * value
            self.v[k] = 0.999 * self.v[k] + 0.001 * value * value
            rate = lr if k in ("w3", "b3") or hidden_lr is None else hidden_lr
            self.p[k] += (
                rate * (self.m[k] / (1 - 0.9**t)) / (np.sqrt(self.v[k] / (1 - 0.999**t)) + 1e-8)
            )
        if hidden_cap and hidden_before:
            change = np.sqrt(sum(np.square(self.p[k] - v).sum() for k, v in hidden_before.items()))
            factor = min(1.0, hidden_cap / max(change, 1e-30))
            for k, v in hidden_before.items():
                self.p[k][:] = v + factor * (self.p[k] - v)
        assert all(np.isfinite(v).all() for v in self.p.values())
        return norm
