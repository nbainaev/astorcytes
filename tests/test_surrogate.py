import torch
import pytest

from astrocites.connection import Connection
from astrocites.learning import NoOp
from astrocites.surrogate import (
    impulse_integral,
    compute_scale_c,
    lif_rate,
    lif_rate_deriv,
    softplus_rate,
    softplus_rate_deriv,
    nmda_calcium,
    nmda_rate,
    nmda_rate_deriv,
    eligibility_gradient,
    spike_eligibility,
    build_lif_params,
    build_nmda_params,
)


class _MockLayer:
    def __init__(self, n):
        self.n = n
        self.shape = (n,)


# ---------------------------------------------------------------------------
# 1. Impulse integral: closed form vs. numerical simulation of the waveform
# ---------------------------------------------------------------------------

def _simulate_impulse_a_pre(impulse_amplitude, impulse_length, impulse_shape_factor):
    """Drive a real ``Connection.compute`` with a single spike and sum the
    ``a_pre`` buffer over the entire impulse window."""
    source = _MockLayer(1)
    target = _MockLayer(1)
    conn = Connection(source, target,
                      impulse_amplitude=impulse_amplitude,
                      impulse_length=impulse_length,
                      impulse_shape_factor=impulse_shape_factor,
                      invert=True, update_rule=NoOp, w=torch.ones(1, 1))
    total = 0.0
    for step in range(impulse_length):
        s = torch.tensor([True]) if step == 0 else torch.tensor([False])
        conn.compute(s)
        total += float(conn.a_pre.sum())
    return total


@pytest.mark.parametrize("A, L, k", [
    (0.5, 40, 0.9),
    (0.5, 40, 0.5),
    (1.0, 40, 0.9),
    (0.3, 50, 0.8),
    (0.5, 20, 0.7),
])
def test_impulse_integral_matches_numerical(A, L, k):
    numerical = _simulate_impulse_a_pre(A, L, k)
    analytical = impulse_integral(A, L, k, invert=True)
    assert abs(numerical - analytical) < 1e-4, (
        f"impulse_integral mismatch for A={A}, L={L}, k={k}: "
        f"numerical={numerical:.6f}, analytical={analytical:.6f}"
    )


def test_impulse_integral_default_params():
    assert abs(impulse_integral(0.5, 40, 0.9, invert=True) - 7.75) < 1e-10


# ---------------------------------------------------------------------------
# 2. LIF rate derivative: closed form vs. autograd
# ---------------------------------------------------------------------------

LIF_PARAMS = dict(thresh=7.0, decay=0.9934, tc_decay=150.0, refrac=40.0)


def test_lif_rate_is_zero_subthreshold():
    I_theta = LIF_PARAMS["thresh"] * (1.0 - LIF_PARAMS["decay"])
    I_sub = torch.tensor([I_theta * 0.5, I_theta * 0.99, 0.0])
    r = lif_rate(I_sub, **LIF_PARAMS)
    assert torch.all(r == 0.0)


def test_lif_rate_deriv_autograd():
    I_theta = LIF_PARAMS["thresh"] * (1.0 - LIF_PARAMS["decay"])
    current = torch.linspace(I_theta * 1.1, I_theta * 5.0, 50, requires_grad=True)
    r = lif_rate(current, **LIF_PARAMS)
    autograd_grad = torch.autograd.grad(r.sum(), current, create_graph=False)[0]
    manual_grad = lif_rate_deriv(current.detach(), **LIF_PARAMS)
    assert torch.allclose(autograd_grad, manual_grad, rtol=1e-4, atol=1e-6), (
        f"LIF rate derivative mismatch:\n autograd={autograd_grad}\n manual  ={manual_grad}"
    )


def test_lif_rate_deriv_zero_subthreshold():
    I_theta = LIF_PARAMS["thresh"] * (1.0 - LIF_PARAMS["decay"])
    I_sub = torch.tensor([I_theta * 0.5, 0.0])
    rprime = lif_rate_deriv(I_sub, **LIF_PARAMS)
    assert torch.all(rprime == 0.0)


# ---------------------------------------------------------------------------
# 3. Softplus rate derivative: closed form vs. autograd
# ---------------------------------------------------------------------------

def test_softplus_rate_deriv_autograd():
    current = torch.linspace(-2.0, 5.0, 80, requires_grad=True)
    params = dict(I_theta=0.05, scale=2.0)
    r = softplus_rate(current, **params)
    autograd_grad = torch.autograd.grad(r.sum(), current, create_graph=False)[0]
    manual_grad = softplus_rate_deriv(current.detach(), **params)
    assert torch.allclose(autograd_grad, manual_grad, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. Full eligibility gradient: closed form vs. autograd on log softmax
# ---------------------------------------------------------------------------

NMDA_PARAMS = dict(I_half=0.0465, k=0.0233, ca_baseline=0.5, n_hill=1.0)
NMDA_HILL4_PARAMS = dict(I_half=0.0465, k=0.0233, ca_baseline=0.5, n_hill=4.0)


@pytest.mark.parametrize("kind, params", [
    ("lif", LIF_PARAMS),
    ("softplus", dict(I_theta=0.05, scale=2.0)),
    ("nmda", NMDA_PARAMS),
    ("nmda", NMDA_HILL4_PARAMS),
])
def test_eligibility_gradient_autograd(kind, params):
    torch.manual_seed(0)
    n_actions = 5
    c = 0.12
    temperature = 1.5
    chosen_idx = 2

    w_slice = torch.linspace(0.4, 0.9, n_actions, requires_grad=True)
    I_eff = c * w_slice
    adj_positions = list(range(n_actions))

    # Closed-form eligibility gradient
    manual = eligibility_gradient(adj_positions, chosen_idx, I_eff.detach(),
                                  c, temperature, kind, params)

    # Autograd reference: d/dw log softmax(r(c*w)/T)[chosen_idx]
    if kind == "lif":
        r = lif_rate(I_eff, **params)
    elif kind == "softplus":
        r = softplus_rate(I_eff, **params)
    else:
        r = nmda_rate(I_eff, **params)
    log_probs = torch.log_softmax(r / temperature, dim=0)
    autograd = torch.autograd.grad(log_probs[chosen_idx], w_slice, create_graph=False)[0]

    assert torch.allclose(manual, autograd, rtol=1e-4, atol=1e-6), (
        f"Eligibility gradient mismatch ({kind}):\n manual  ={manual}\n autograd={autograd}"
    )


@pytest.mark.parametrize("chosen_idx", [0, 1, 2, 3, 4])
def test_eligibility_gradient_all_actions(chosen_idx):
    n_actions = 5
    c = 0.12
    temperature = 1.0
    w_slice = torch.linspace(0.4, 0.9, n_actions, requires_grad=True)
    I_eff = c * w_slice
    adj_positions = list(range(n_actions))

    manual = eligibility_gradient(adj_positions, chosen_idx, I_eff.detach(),
                                  c, temperature, "lif", LIF_PARAMS)
    r = lif_rate(I_eff, **LIF_PARAMS)
    log_probs = torch.log_softmax(r / temperature, dim=0)
    autograd = torch.autograd.grad(log_probs[chosen_idx], w_slice, create_graph=False)[0]
    assert torch.allclose(manual, autograd, rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# 5. Integration: scale constant and build_lif_params helpers
# ---------------------------------------------------------------------------

def test_compute_scale_c():
    c = compute_scale_c(0.5, 40, 0.9, True, intensity=15.0, time_steps=1000)
    expected = 7.75 * 15.0 / 1000.0
    assert abs(c - expected) < 1e-10


def test_build_lif_params():
    p = build_lif_params(thresh=7.0, dt=1.0, tc_decay=150.0, refrac=40)
    assert p["thresh"] == 7.0
    assert p["refrac"] == 40.0
    assert p["tc_decay"] == 150.0
    expected_decay = float(torch.exp(torch.tensor(-1.0 / 150.0)).item())
    assert abs(p["decay"] - expected_decay) < 1e-12


# ---------------------------------------------------------------------------
# 6. End-to-end: the rates produce plausible spike counts for default params
# ---------------------------------------------------------------------------

def test_lif_rate_plausible_counts():
    """For default params and weights in [0.5, 0.8], the surrogate rate should
    predict a handful of spikes over 1000 timesteps (matching what the SNN
    would produce)."""
    p = build_lif_params(thresh=7.0, dt=1.0, tc_decay=150.0, refrac=40)
    c = compute_scale_c(0.5, 40, 0.9, True, intensity=15.0, time_steps=1000)
    time_steps = 1000
    for w_val in [0.5, 0.65, 0.8]:
        I_eff = torch.tensor([c * w_val])
        rate = lif_rate(I_eff, **p)
        expected_spikes = (rate * time_steps).item()
        assert expected_spikes > 0, f"Expected spikes for w={w_val} but got 0"
        assert expected_spikes < 100, f"Too many spikes ({expected_spikes}) for w={w_val}"


# ---------------------------------------------------------------------------
# 7. NMDA calcium surrogate: derivative, monotonicity, sign change
# ---------------------------------------------------------------------------

def test_nmda_rate_deriv_autograd():
    current = torch.linspace(0.0, 0.15, 80, requires_grad=True)
    r = nmda_rate(current, **NMDA_PARAMS)
    autograd_grad = torch.autograd.grad(r.sum(), current, create_graph=False)[0]
    manual_grad = nmda_rate_deriv(current.detach(), **NMDA_PARAMS)
    assert torch.allclose(autograd_grad, manual_grad, rtol=1e-5, atol=1e-6)


def test_nmda_calcium_is_sigmoid():
    I_half = NMDA_PARAMS["I_half"]
    k = NMDA_PARAMS["k"]
    ca_at_half = nmda_calcium(torch.tensor([I_half]), I_half, k)
    assert abs(ca_at_half.item() - 0.5) < 1e-6


def test_nmda_rate_deriv_monotonically_increasing():
    current = torch.linspace(0.0, 0.15, 200)
    rprime = nmda_rate_deriv(current, **NMDA_PARAMS)
    diffs = rprime[1:] - rprime[:-1]
    assert (diffs >= -1e-6).all(), "nmda_rate_deriv should be monotonically non-decreasing"


def test_nmda_rate_deriv_sign_change_at_threshold():
    I_half = NMDA_PARAMS["I_half"]
    below = nmda_rate_deriv(torch.tensor([I_half * 0.5]), **NMDA_PARAMS)
    at = nmda_rate_deriv(torch.tensor([I_half]), **NMDA_PARAMS)
    above = nmda_rate_deriv(torch.tensor([I_half * 2.0]), **NMDA_PARAMS)
    assert below.item() < 0, "Should be negative below threshold (LTD)"
    assert abs(at.item()) < 1e-6, "Should be ~zero at threshold (Ca=ca_baseline)"
    assert above.item() > 0, "Should be positive above threshold (LTP)"


def test_nmda_rate_smooth_threshold():
    """Rate should be ~0 below threshold and growing above it."""
    I_half = NMDA_PARAMS["I_half"]
    r_below = nmda_rate(torch.tensor([I_half * 0.1]), **NMDA_PARAMS).item()
    r_at = nmda_rate(torch.tensor([I_half]), **NMDA_PARAMS).item()
    r_above = nmda_rate(torch.tensor([I_half * 2.0]), **NMDA_PARAMS).item()
    assert abs(r_below) < 0.01, "Rate should be ~0 well below threshold"
    assert r_at < r_above, "Rate should increase through threshold"


def test_build_nmda_params():
    p = build_nmda_params(thresh=7.0, dt=1.0, tc_decay=150.0)
    decay = float(torch.exp(torch.tensor(-1.0 / 150.0)).item())
    I_theta = 7.0 * (1.0 - decay)
    assert abs(p["I_half"] - I_theta) < 1e-10
    assert abs(p["k"] - 0.5 * I_theta) < 1e-10
    assert p["ca_baseline"] == 0.5
    assert p["n_hill"] == 1.0


def test_build_nmda_params_custom():
    p = build_nmda_params(thresh=7.0, dt=1.0, tc_decay=150.0, k_frac=0.3, ca_baseline=0.3, n_hill=4.0)
    decay = float(torch.exp(torch.tensor(-1.0 / 150.0)).item())
    I_theta = 7.0 * (1.0 - decay)
    assert abs(p["k"] - 0.3 * I_theta) < 1e-10
    assert p["ca_baseline"] == 0.3
    assert p["n_hill"] == 4.0


# ---------------------------------------------------------------------------
# 8. NMDA Hill cooperativity (n_hill > 1)
# ---------------------------------------------------------------------------

def test_nmda_hill4_deriv_autograd():
    """Closed-form r' matches autograd for n_hill=4."""
    current = torch.linspace(0.0, 0.15, 80, requires_grad=True)
    r = nmda_rate(current, **NMDA_HILL4_PARAMS)
    autograd_grad = torch.autograd.grad(r.sum(), current, create_graph=False)[0]
    manual_grad = nmda_rate_deriv(current.detach(), **NMDA_HILL4_PARAMS)
    assert torch.allclose(autograd_grad, manual_grad, rtol=1e-5, atol=1e-6)


def test_nmda_hill_sharpens_threshold():
    """Higher n_hill should produce steeper transition around I_half."""
    I_half = NMDA_PARAMS["I_half"]
    k = NMDA_PARAMS["k"]
    delta = k * 0.5
    pts = torch.tensor([I_half - delta, I_half, I_half + delta])
    rprime_n1 = nmda_rate_deriv(pts, I_half, k, ca_baseline=0.5, n_hill=1.0)
    rprime_n4 = nmda_rate_deriv(pts, I_half, k, ca_baseline=0.5, n_hill=4.0)
    slope_n1 = (rprime_n1[2] - rprime_n1[0]).item()
    slope_n4 = (rprime_n4[2] - rprime_n4[0]).item()
    assert slope_n4 > slope_n1 * 3, "n_hill=4 should be significantly steeper than n_hill=1"


def test_nmda_hill_sign_change_preserved():
    """Sign change at threshold should hold for any n_hill."""
    I_half = NMDA_HILL4_PARAMS["I_half"]
    below = nmda_rate_deriv(torch.tensor([I_half * 0.5]), **NMDA_HILL4_PARAMS)
    at = nmda_rate_deriv(torch.tensor([I_half]), **NMDA_HILL4_PARAMS)
    above = nmda_rate_deriv(torch.tensor([I_half * 2.0]), **NMDA_HILL4_PARAMS)
    assert below.item() < 0
    assert abs(at.item()) < 1e-6
    assert above.item() > 0


def test_nmda_hill_monotonically_increasing():
    """Monotonicity should hold for n_hill=4."""
    current = torch.linspace(0.0, 0.15, 200)
    rprime = nmda_rate_deriv(current, **NMDA_HILL4_PARAMS)
    diffs = rprime[1:] - rprime[:-1]
    assert (diffs >= -1e-6).all()


def test_nmda_hill_backward_compat():
    """n_hill=1 should match the old sigmoid-based formula exactly."""
    current = torch.linspace(0.0, 0.15, 50)
    I_half, k = 0.0465, 0.0233
    ca_new = nmda_calcium(current, I_half, k, n_hill=1.0)
    ca_old = torch.sigmoid((current - I_half) / k)
    assert torch.allclose(ca_new, ca_old)


# ---------------------------------------------------------------------------
# 9. Spike-based eligibility (reward-modulated Hebbian mode)
# ---------------------------------------------------------------------------

def test_spike_eligibility_sign_and_magnitude():
    """The spike-based eligibility grad_slice should have correct sign and
    be proportional to spike counts normalized by time_steps."""
    time_steps = 1000
    spike_counts = torch.tensor([3.0, 0.0, 7.0, 5.0, 1.0])
    probs = torch.tensor([0.15, 0.05, 0.40, 0.30, 0.10])
    chosen_idx = 2

    grad_slice = spike_eligibility(probs, spike_counts, chosen_idx, time_steps, temperature=1.0)

    assert grad_slice[chosen_idx] > 0, "Chosen action with spikes should have positive eligibility"
    for i in range(len(probs)):
        if i != chosen_idx and spike_counts[i] > 0:
            assert grad_slice[i] < 0, "Non-chosen action with spikes should have negative eligibility"
        if spike_counts[i] == 0:
            assert grad_slice[i] == 0.0, "Zero-spike action should have zero eligibility"

    expected_chosen = (1.0 - probs[chosen_idx]) * (spike_counts[chosen_idx] / time_steps)
    assert abs(grad_slice[chosen_idx].item() - expected_chosen) < 1e-8


def test_spike_eligibility_scales_inversely_with_temperature():
    probs = torch.tensor([0.2, 0.5, 0.3])
    spike_counts = torch.tensor([2.0, 5.0, 3.0])
    chosen_idx = 1

    e_T1 = spike_eligibility(probs, spike_counts, chosen_idx, time_steps=100, temperature=1.0)
    e_T05 = spike_eligibility(probs, spike_counts, chosen_idx, time_steps=100, temperature=0.5)

    assert torch.allclose(e_T05, 2 * e_T1)
