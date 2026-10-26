"""Adaptive LIF dynamics and symmetric local e-prop eligibility."""

import math
import numpy as np
import torch
from torch import nn
from numba import njit


@njit(cache=True)
def advance(
    inputs,
    wi,
    wr,
    mask,
    v,
    b,
    spike,
    refractory,
    rate,
    phase,
    ev,
    ea,
    previous_e,
    filtered_e,
    alpha,
    rho,
    kappa,
    beta,
    threshold,
    damp,
    n_ref,
    steps,
    mean_readout,
    eligibility,
    spike_counts,
):
    n = v.size
    channels = wi.shape[0]
    mean = np.zeros(n)
    mean_e = np.zeros_like(filtered_e)
    for t in range(steps):
        phase += inputs
        input_spikes = (phase >= 1).astype(np.float64)
        phase -= input_spikes
        pre = np.ones(channels)
        pre[: inputs.size] = input_spikes
        last = spike.copy()
        b[:] = rho * b + last
        v[:] = alpha * v + pre @ wi + last @ wr - threshold * last
        relative = (v - threshold - beta * b) / threshold
        active = refractory <= 0
        spike[:] = ((relative > 0) & active).astype(np.float64)
        spike_counts += spike
        refractory[:] = np.minimum(n_ref, np.maximum(0, refractory + n_ref * spike - 1))
        if eligibility:
            psi = damp / threshold * np.maximum(0, 1 - np.abs(relative)) * active
            # ea(t)=rho ea(t-1)+e(t-1) retains LOCAL adaptive feedback.
            # ev(t)=alpha ev(t-1)+presynaptic spike(t).
            for i in range(channels + n):
                for j in range(n):
                    ea[i, j] = rho * ea[i, j] + previous_e[i, j]
                    drive = pre[i] if i < channels else last[i - channels]
                    ev[i, j] = alpha * ev[i, j] + drive
                    e = psi[j] * (ev[i, j] - beta[j] * ea[i, j])
                    if i >= channels:
                        e *= mask[i - channels, j]
                    previous_e[i, j] = e
                    filtered_e[i, j] = kappa * filtered_e[i, j] + (1 - kappa) * e
        rate[:] = kappa * rate + (1 - kappa) * spike
        if mean_readout:
            mean += rate / steps
            if eligibility:
                mean_e += filtered_e / steps
    if mean_readout:
        return mean, mean_e[:channels].copy(), mean_e[channels:].copy()
    return rate.copy(), filtered_e[:channels].copy(), filtered_e[channels:].copy()


class BellecALIF(nn.Module):
    """Tutorial ALIF with half regular LIF / half adaptive neurons by default.

    Weights are signed and dense, as in the reference tutorial; no autapses.
    dt=1ms; z is a binary spike per tick. Positive beta applies to unnormalized b.
    """

    def __init__(
        self,
        input_dim=24,
        neurons=64,
        steps=40,
        seed=0,
        tau_m_ms=20.0,
        tau_a_ms=2000.0,
        tau_out_ms=20.0,
        threshold=0.6,
        damp=0.3,
        n_ref=5,
        adaptive_fraction=0.5,
        input_scale=1.0,
        recurrent_scale=1.0,
        input_identity=False,
        input_bias=False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.neurons = neurons
        self.steps = steps
        self.alpha = math.exp(-1 / tau_m_ms)
        self.rho = math.exp(-1 / tau_a_ms)
        self.kappa = math.exp(-1 / tau_out_ms)
        self.threshold = threshold
        self.damp = damp
        self.n_ref = n_ref
        beta = np.zeros(neurons)
        beta[neurons - round(neurons * adaptive_fraction) :] = (
            1.7 * (1 - self.rho) / (1 - self.alpha) if adaptive_fraction > 0 else 0
        )
        self.register_buffer("beta", torch.tensor(beta, dtype=torch.float64))
        self.register_buffer("rec_mask", 1 - torch.eye(neurons, dtype=torch.float64))
        self.register_buffer("signs", torch.ones(neurons, dtype=torch.float64))
        rng = np.random.default_rng(seed)
        wi = rng.normal(0, input_scale / np.sqrt(input_dim), (input_dim, neurons))
        if input_identity:
            if input_dim < neurons:
                raise ValueError(
                    "Topographic input initialization needs at least one channel per neuron"
                )
            wi *= 0.01
            wi[:neurons] += np.eye(neurons) * input_scale
        if input_bias:
            # Internal trainable current bias, represented as a constant-1
            # synapse. External encoder and its phase/spike stream are unchanged.
            wi = np.vstack((wi, np.zeros((1, neurons))))
        self.win = nn.Parameter(torch.tensor(wi, dtype=torch.float64))
        self.wrec = nn.Parameter(
            torch.tensor(
                rng.normal(0, recurrent_scale / np.sqrt(neurons), (neurons, neurons)),
                dtype=torch.float64,
            )
            * self.rec_mask
        )
        self.reset()

    def reset(self):
        self._state = None

    def observe(self, input_rate, generator=None, spike_mode="regular", readout="mean"):
        if spike_mode != "regular":
            raise ValueError(
                "Use regular encoder; stochastic inputs can be supplied as explicit pre-spike sequences in checks"
            )
        if self._state is None:
            self._state = FastBellecCore(self)
        result = self._state.observe(
            input_rate.detach().numpy() if torch.is_tensor(input_rate) else input_rate,
            eligibility=True,
            mean_readout=readout == "mean",
        )
        return tuple(torch.from_numpy(v).float() for v in result)

    def eprop_update(self, local_in_gradient, local_rec_gradient, lr=1e-3, norm_clip=1.0):
        gi = local_in_gradient.double()
        gr = local_rec_gradient.double() * self.rec_mask
        norm = torch.sqrt(gi.square().sum() + gr.square().sum())
        scale = min(1.0, norm_clip / max(float(norm), 1e-12))
        with torch.no_grad():
            self.win -= lr * scale * gi
            self.wrec -= lr * scale * gr
            self.wrec *= self.rec_mask
        self._state = None


class FastBellecCore:
    def __init__(self, network):
        self.net = network
        self.wi = network.win.detach().numpy().copy()
        self.wr = (network.wrec * network.rec_mask).detach().numpy().copy()
        self.mask = network.rec_mask.numpy()
        self.beta = network.beta.numpy()
        self.reset()

    def reset(self):
        n = self.net.neurons
        c = self.wi.shape[0]
        self.v = np.zeros(n)
        self.b = np.zeros(n)
        self.spike = np.zeros(n)
        self.refractory = np.zeros(n)
        self.rate = np.zeros(n)
        self.phase = np.zeros(self.net.input_dim)
        self.spike_counts = np.zeros(n)
        self.ev = np.zeros((c + n, n))
        self.ea = self.ev.copy()
        self.previous_e = self.ev.copy()
        self.filtered_e = self.ev.copy()

    def observe(self, inputs, eligibility=True, mean_readout=True):
        n = self.net
        return advance(
            np.asarray(inputs, dtype=np.float64),
            self.wi,
            self.wr,
            self.mask,
            self.v,
            self.b,
            self.spike,
            self.refractory,
            self.rate,
            self.phase,
            self.ev,
            self.ea,
            self.previous_e,
            self.filtered_e,
            n.alpha,
            n.rho,
            n.kappa,
            self.beta,
            n.threshold,
            n.damp,
            n.n_ref,
            n.steps,
            mean_readout,
            eligibility,
            self.spike_counts,
        )
