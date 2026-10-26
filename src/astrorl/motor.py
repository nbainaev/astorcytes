"""Four direct Motor LIF neurons with local detached-reset sensitivities."""

from dataclasses import dataclass
import copy
import numpy as np
from numba import njit


@njit(cache=True)
def lif_window(raw, gate, threshold, phase, bias, amplitude, steps, width, influence):
    """Output count and dcount/draw for the specified surrogate graph.

    Hard resets and refractory state are detached. Warmup spikes reset the
    sensitivities too.
    """
    n = raw.size
    alpha = np.exp(-0.003 / 0.02)
    vm = phase.copy()
    vi = np.zeros(n)
    cond = np.zeros(n)
    mr = np.zeros(n)
    ir = np.zeros(n)
    q = np.zeros(n)
    counts = np.zeros(n)
    derivative = np.zeros(n)
    inh = np.zeros(n)
    drive = amplitude * (1 - np.tanh(raw) ** 2)
    excit = bias + amplitude * np.tanh(raw)
    warm = int(round(steps * 0.2))
    for t in range(steps):
        for j in range(n):
            vi[j] = alpha * vi[j] + (1 - alpha) * 1.12
            si = 1.0 if t >= warm and vi[j] >= 1.25 / (1 + gate[j]) and ir[j] <= 0 else 0.0
            vi[j] *= 1 - si
            ir[j] = 6 if si > 0 else max(0.0, ir[j] - 1)
            cond[j] = 0.78 * cond[j] + si
            inh[j] += si
            vm[j] = alpha * vm[j] + (1 - alpha) * (excit[j] - influence * cond[j])
            q[j] = alpha * q[j] + (1 - alpha) * drive[j]
            active = 1.0 if mr[j] <= 0 else 0.0
            spike = 1.0 if vm[j] >= threshold[j] and active else 0.0
            if t >= warm:
                counts[j] += spike
                derivative[j] += active * q[j] / (1 + width * abs(vm[j] - threshold[j])) ** 2
            vm[j] *= 1 - spike
            q[j] *= 1 - spike
            mr[j] = 3 if spike > 0 else max(0.0, mr[j] - 1)
    return counts, derivative, inh


@dataclass
class MotorConfig:
    name: str = "motor"
    bias: float = 1.23
    amplitude: float = 0.75
    inhibition_scale: float = 1.0
    temperature: float = 0.03
    temperature_end: float = 0.003
    steps: int = 60
    lr: float = 0.01
    lr_end: float = 0.001
    entropy: float = 0.02
    entropy_end: float = 0.001
    anneal_fraction: float = 1.0
    bias_trainable: bool = True
    surrogate_width: float = 6.0


class SpikingMotorHead:
    def __init__(self, inputs, config):
        self.config = config
        self.temperature = config.temperature
        self.learning_rate = config.lr
        self.params = dict(wo=np.zeros((4, inputs), dtype=np.float64), bo=np.zeros(4))
        self.threshold = np.ones(4)
        self.phase = np.zeros(4)
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = copy.deepcopy(self.m)
        self.iteration = 0

    def forward(self, feature, gate, gain, influence):
        c = self.config
        length = c.steps - round(0.2 * c.steps)
        raw = self.params["wo"] @ feature + self.params["bo"]
        counts, derivative, inhibitory = lif_window(
            raw,
            gate * gain,
            self.threshold,
            self.phase,
            c.bias,
            c.amplitude,
            c.steps,
            c.surrogate_width,
            influence * c.inhibition_scale,
        )
        scale = length * self.temperature
        logits = counts / scale
        probability = np.exp(logits - logits.max())
        probability /= probability.sum()
        cache = (feature, derivative / scale)
        diagnostics = dict(
            motor_counts=float(counts.mean()),
            inhibitory_counts=float(inhibitory.mean()),
            motor_silent_fraction=float(np.mean(counts == 0)),
            motor_saturation=float(np.mean(np.abs(np.tanh(raw)) > 0.99)),
            motor_derivative_mean=float(derivative.mean() / scale),
        )
        return probability, cache, diagnostics

    def gradients(self, records, baseline, entropy, gamma=0.99, horizon=50):
        returns = np.zeros(len(records))
        carry = 0.0
        for t in range(len(records) - 1, -1, -1):
            carry = records[t]["reward"] + gamma * carry
            returns[t] = carry
        gradients = {k: np.zeros_like(v) for k, v in self.params.items()}
        for t, record in enumerate(records):
            p = record["probability"]
            logp = np.log(np.maximum(p, 1e-30))
            ent = -np.sum(p * logp)
            score = -p.copy()
            score[record["action"]] += 1.0
            delta = (
                gamma**t
                / horizon
                * ((returns[t] - baseline[t]) * score - entropy * p * (logp + ent))
            )
            feature, derivative = record["head_cache"]
            signal = delta * derivative
            gradients["wo"] += np.outer(signal, feature)
            if self.config.bias_trainable:
                gradients["bo"] += signal
        return gradients, returns

    def step(self, gradients):
        norm = float(np.sqrt(sum(np.sum(g * g) for g in gradients.values())))
        self.iteration += 1
        for key, gradient in gradients.items():
            gradient = gradient / max(norm, 1.0)
            self.m[key] = 0.9 * self.m[key] + 0.1 * gradient
            self.v[key] = 0.999 * self.v[key] + 0.001 * gradient * gradient
            self.params[key] += (
                self.learning_rate
                * (self.m[key] / (1 - 0.9**self.iteration))
                / (np.sqrt(self.v[key] / (1 - 0.999**self.iteration)) + 1e-8)
            )
            assert np.isfinite(self.params[key]).all()
        return norm
