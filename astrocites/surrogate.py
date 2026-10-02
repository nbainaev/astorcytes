import torch
import torch.nn.functional as F


def impulse_integral(impulse_amplitude, impulse_length, impulse_shape_factor, invert=True):
    """Closed-form total ``a_pre`` area delivered by a single pre-synaptic spike.

    This is the sum of the cumulative ``a_pre`` buffer over the full impulse
    waveform window (``impulse_length - 1`` timesteps during which the impulse
    state is non-zero). For ``invert=True`` (the biphasic waveform used by
    ``conn_XY``), the derivation yields:

        impulse_integral = A * (L * (2k - 1) - 1) / 2

    where ``A`` is ``impulse_amplitude``, ``L`` is ``impulse_length`` and ``k``
    is ``impulse_shape_factor``.
    """
    A = float(impulse_amplitude)
    L = float(impulse_length)
    k = float(impulse_shape_factor)
    if invert:
        return A * (L * (2.0 * k - 1.0) - 1.0) / 2.0
    else:
        raise NotImplementedError("impulse_integral for invert=False is not used by conn_XY")


def compute_scale_c(impulse_amplitude, impulse_length, impulse_shape_factor, invert,
                    intensity, time_steps):
    """Scale constant mapping a weight ``w[s, a]`` to the effective per-timestep
    input current ``I_eff(a) = c * w[s, a]``.

    ``c = impulse_integral * intensity / time_steps``: the per-spike charge area
    times the expected number of input spikes (``intensity``) spread over the
    simulation window (``time_steps``).
    """
    ii = impulse_integral(impulse_amplitude, impulse_length, impulse_shape_factor, invert)
    return ii * float(intensity) / float(time_steps)


def lif_rate(current, thresh, decay, tc_decay, refrac, eps=1e-8):
    """Analytical steady-state firing rate of a LIF neuron under constant current.

    Neuron model (matches ``LIFNodes`` with ``rest = reset = 0``):

        v[t+1] = decay * v[t] + I ;  spike when v >= thresh ;  reset to 0

    The minimum current needed to ever reach threshold at steady state is
    ``I_theta = thresh * (1 - decay)``. For ``I > I_theta`` the inter-spike
    interval is ``refrac + tc_decay * ln(I / (I - I_theta))``, giving rate
    ``r(I) = 1 / interval``. For ``I <= I_theta`` the rate is zero. The function
    is continuous (``r -> 0`` as ``I -> I_theta+``).

    Parameters
    ----------
    current : torch.Tensor
        Effective input currents (one per candidate action).
    """
    I_theta = thresh * (1.0 - decay)
    suprathreshold = current > I_theta
    safe = torch.where(suprathreshold, current, torch.full_like(current, I_theta + eps))
    denom = (safe - I_theta).clamp(min=eps)
    ratio = safe / denom
    interval = refrac + tc_decay * torch.log(ratio)
    rate = 1.0 / interval
    return torch.where(suprathreshold, rate, torch.zeros_like(rate))


def lif_rate_deriv(current, thresh, decay, tc_decay, refrac, eps=1e-8):
    """Closed-form ``dr/dI`` for :func:`lif_rate`.

        f(I)  = refrac + tc_decay * ln(I / (I - I_theta))
        r(I)  = 1 / f(I)
        r'(I) = tc_decay * I_theta / (I * (I - I_theta) * f(I)^2)

    Zero for subthreshold inputs.
    """
    I_theta = thresh * (1.0 - decay)
    suprathreshold = current > I_theta
    safe = torch.where(suprathreshold, current, torch.full_like(current, I_theta + eps))
    diff = (safe - I_theta).clamp(min=eps)
    interval = refrac + tc_decay * torch.log(safe / diff)
    rprime = tc_decay * I_theta / (safe * diff * interval * interval)
    return torch.where(suprathreshold, rprime, torch.zeros_like(rprime))


def softplus_rate(current, I_theta, scale):
    """Smooth proxy surrogate rate: ``r(I) = softplus(scale * (I - I_theta))``."""
    return F.softplus(scale * (current - I_theta))


def softplus_rate_deriv(current, I_theta, scale):
    """``dr/dI = scale * sigmoid(scale * (I - I_theta))``."""
    return scale * torch.sigmoid(scale * (current - I_theta))


def nmda_calcium(current, I_half, k, n_hill=1.0):
    """NMDA-mediated calcium level (Mg2+ unblock sigmoid).

        Ca(I) = sigmoid(n_hill * (I - I_half) / k)

    Monotonically increasing from 0 to 1. ``I_half`` is the half-activation
    current (default: the LIF threshold current ``I_theta``), ``k`` is the slope
    factor, and ``n_hill`` is the Hill cooperativity coefficient. ``n_hill = 1``
    gives the simple NMDA Mg2+ unblock sigmoid; ``n_hill ~ 3-4`` approximates the
    cooperativity of Ca2+/calmodulin binding to CaMKII (Hill, 1985; Chin &
    Means, 2000), sharpening the calcium-to-plasticity threshold.
    """
    return torch.sigmoid(n_hill * (current - I_half) / k)


def nmda_rate(current, I_half, k, ca_baseline=0.5, n_hill=1.0):
    """Rate/logit for the NMDA surrogate.

    Defined as the integral of calcium-above-baseline::

        r(I) = (k / n_hill) * softplus(n_hill * (I - I_half) / k) - ca_baseline * I

    so that ``r'(I) = Ca(I) - ca_baseline`` (see :func:`nmda_rate_deriv`).

    With ``ca_baseline = 0.5``, ``I_half = I_theta``, ``n_hill = 1`` this
    simplifies to ``r(I) = k * ln(cosh((I - I_theta) / (2k)))`` — a smooth
    threshold function that is approximately zero below threshold and grows
    linearly above.
    """
    scaled = n_hill * (current - I_half) / k
    return (k / n_hill) * F.softplus(scaled) - ca_baseline * current


def nmda_rate_deriv(current, I_half, k, ca_baseline=0.5, n_hill=1.0):
    """Derivative of :func:`nmda_rate`.

        r'(I) = Ca(I) - ca_baseline

    This is **monotonically increasing** (calcium level minus baseline), with a
    sign change at the current where ``Ca(I) = ca_baseline``. For the default
    ``ca_baseline = 0.5`` and ``I_half = I_theta``, the sign change occurs
    exactly at the LIF threshold: subthreshold synapses get negative
    eligibility (LTD direction), suprathreshold get positive (LTP direction) —
    a BCM-like calcium threshold mechanism (Bienenstock, Cooper & Munro, 1982).

    The ``n_hill`` parameter controls the steepness of the calcium transition
    around threshold, modeling the cooperativity of Ca2+/calmodulin binding
    (n_hill ~ 4 in biology).
    """
    return nmda_calcium(current, I_half, k, n_hill) - ca_baseline


def rate_and_deriv(current, kind, params):
    """Dispatch to the requested surrogate kind, returning ``(r, r')``.

    Parameters
    ----------
    current : torch.Tensor
        Effective input currents (one per candidate action).
    kind : str
        ``"lif"``, ``"softplus"``, or ``"nmda"``.
    params : dict
        Keyword arguments forwarded to the chosen surrogate functions.
        For ``"lif"``: ``thresh``, ``decay``, ``tc_decay``, ``refrac``.
        For ``"softplus"``: ``I_theta``, ``scale``.
        For ``"nmda"``: ``I_half``, ``k``, ``ca_baseline``, ``n_hill``.
    """
    if kind == "lif":
        return lif_rate(current, **params), lif_rate_deriv(current, **params)
    elif kind == "softplus":
        return softplus_rate(current, **params), softplus_rate_deriv(current, **params)
    elif kind == "nmda":
        return nmda_rate(current, **params), nmda_rate_deriv(current, **params)
    else:
        raise ValueError(f"Unknown surrogate kind: {kind!r}")


def eligibility_gradient(adj_positions, chosen_idx, I_eff, c, temperature, kind, params):
    """Compute the closed-form REINFORCE eligibility gradient ``d log pi / d w[s, a]``
    for each candidate action ``a``, where the policy is
    ``pi(a) = softmax(r(I_eff(a)) / T)``.

    Returns a 1-D tensor of length ``len(adj_positions)`` aligned with
    ``adj_positions``: the gradient entries to scatter into ``eligibility[s, adj]``.

        d log pi(a*) / d w[s, a] = (1/T) * (1[a=a*] - pi(a)) * r'(I_eff(a)) * c
    """
    rate, rate_deriv = rate_and_deriv(I_eff, kind, params)
    probs = torch.softmax(rate / temperature, dim=0)
    onehot = torch.zeros_like(probs)
    onehot[chosen_idx] = 1.0
    return (1.0 / temperature) * (onehot - probs) * rate_deriv * c


def spike_eligibility(probs, spike_counts, chosen_idx, time_steps, temperature):
    """Return the score-like spike eligibility for one sampled action."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if time_steps <= 0:
        raise ValueError("time_steps must be positive")

    spike_rates = spike_counts / time_steps
    onehot = torch.zeros_like(probs)
    onehot[chosen_idx] = 1.0
    return ((onehot - probs) * spike_rates) / temperature


def build_lif_params(thresh, dt, tc_decay, refrac):
    """Convenience: pack the neuron constants into the dict expected by the
    ``"lif"`` surrogate, computing ``decay`` from ``dt`` and ``tc_decay``."""
    decay = float(torch.exp(torch.tensor(-float(dt) / float(tc_decay))).item())
    return {"thresh": float(thresh), "decay": decay, "tc_decay": float(tc_decay), "refrac": float(refrac)}


def build_nmda_params(thresh, dt, tc_decay, k_frac=0.5, ca_baseline=0.5, n_hill=1.0):
    """Pack neuron constants into the dict expected by the ``"nmda"`` surrogate.

    ``I_half`` defaults to the LIF threshold current ``I_theta = thresh * (1 -
    decay)``, so calcium half-activation coincides with the spike threshold.
    ``k`` defaults to ``k_frac * I_theta`` (slope factor). ``ca_baseline``
    defaults to 0.5 (sign change at threshold). ``n_hill`` is the Hill
    cooperativity coefficient (1 = simple sigmoid; ~4 = Ca2+/CaM cooperativity).
    """
    decay = float(torch.exp(torch.tensor(-float(dt) / float(tc_decay))).item())
    I_theta = float(thresh) * (1.0 - decay)
    return {
        "I_half": I_theta,
        "k": k_frac * I_theta,
        "ca_baseline": ca_baseline,
        "n_hill": n_hill,
    }
