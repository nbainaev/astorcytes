import pytest
import numpy as np
import torch

from astrocites.experiment import _sample_input_spike_train
from astrocites.network import Network, NetworkMonitor
from astrocites.nodes import Input
from astrocites.utils import bernoulli_loader
from astrocites.utils import make_rng_streams


def test_bernoulli_loader_mean():
    torch.manual_seed(0)
    probs = torch.tensor([[0.0, 0.2, 0.0]])
    x = torch.stack(list(bernoulli_loader(probs, time=10_000, dt=1.0)))

    assert x.shape == (10_000, 3)
    assert x[:, 0].sum() == 0
    assert x[:, 2].sum() == 0
    assert abs(x[:, 1].float().mean().item() - 0.2) < 0.02


def test_bernoulli_loader_deterministic_probabilities_and_dt():
    probs = torch.tensor([[0.0, 1.0]])
    samples = torch.stack(list(bernoulli_loader(probs, time=1000, dt=2)))

    assert samples.shape == (500, 2)
    assert torch.equal(samples[:, 0], torch.zeros(500))
    assert torch.equal(samples[:, 1], torch.ones(500))


@pytest.mark.parametrize(
    "probs,time,dt,message",
    [
        ([[0.1, 0.2], [0.3, 0.4]], 3, 1, "timesteps"),
        ([[0.1, 1.1]], 1, 1, "probabilities"),
        ([[0.1], [float("nan")]], 2, 1, "probabilities"),
        ([[[0.1]]], 1, 1, "Expected"),
    ],
)
def test_bernoulli_loader_rejects_invalid_input(probs, time, dt, message):
    with pytest.raises(ValueError, match=message):
        list(bernoulli_loader(probs, time=time, dt=dt))


def _count_input_layer_spikes(intensity, seed):
    torch.manual_seed(seed)
    sample = _sample_input_spike_train(0, 1, intensity, time_steps=1000, dt=1)
    network = Network(dt=1)
    layer = Input(n=1, traces=True, thresh=7, rest=0, reset=0, refrac=40)
    network.add_layer(layer, "X")
    network.add_monitor(NetworkMonitor(network, state_vars=("s",)), "Network")
    network.run(inpts={"X": sample}, time=1000)
    return int(network.monitors["Network"].get()["X"]["s"].sum().item())


def test_higher_intensity_increases_refractory_filtered_input_spikes():
    # Fixed seeds make this a directional smoke test, not a population estimate.
    seeds = [3, 42]
    low_count = sum(_count_input_layer_spikes(5, seed) for seed in seeds)
    high_count = sum(_count_input_layer_spikes(25, seed) for seed in seeds)

    assert high_count > low_count


def test_rng_streams_keep_weights_and_policy_independent_of_sensory_draws():
    first = make_rng_streams(42)
    second = make_rng_streams(42)

    weights_a = first["weights"].normal(0.65, 0.1, size=16)
    _ = torch.bernoulli(torch.full((10_000,), 0.15), generator=second["sensory"])
    weights_b = second["weights"].normal(0.65, 0.1, size=16)
    assert np.array_equal(weights_a, weights_b)

    policy_a = torch.rand(16, generator=first["action_torch"])
    policy_b = torch.rand(16, generator=second["action_torch"])
    assert torch.equal(policy_a, policy_b)


def test_legacy_input_mode_reproduces_static_positive_drive():
    sample = _sample_input_spike_train(
        2, 4, intensity=15, time_steps=1000, dt=1,
        input_mode="legacy_static_positive",
    )

    assert sample.shape == (1, 4)
    assert torch.equal(sample, torch.tensor([[0.0, 0.0, 0.015, 0.0]]))


def test_paper_bernoulli_input_mode_samples_a_timestep_dependent_train():
    generator = torch.Generator().manual_seed(4)
    sample = _sample_input_spike_train(
        current_position=2, n_neurons=4, intensity=150,
        time_steps=1000, dt=1, probability=1.0,
        generator=generator, input_mode="paper_bernoulli",
    )
    assert sample.shape == (1000, 1, 4)
    assert torch.equal(sample[:, 0, 2], torch.ones(1000))
    assert torch.equal(sample[:, 0, [0, 1, 3]], torch.zeros(1000, 3))
