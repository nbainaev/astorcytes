import numpy as np
import torch

from astrocites.experiment import setup_and_run_simulation
from astrocites.utils import create_adjacency_matrix, make_rng_streams


def _run_one_episode(optimized_connections):
    grid_size = 3
    n_states = grid_size * grid_size
    mask = create_adjacency_matrix(grid_size)
    weights = torch.as_tensor(mask, dtype=torch.float32) * 0.65
    diagnostics = {}
    rngs = make_rng_streams(123)
    result = setup_and_run_simulation(
        NA=n_states,
        weights_mask_XY=mask,
        weights_init_XY=weights,
        weights_init_XI=torch.eye(n_states),
        n_steps=3,
        current_position=4,
        goal=8,
        learning_rate=1,
        wmin=0.001,
        wmax=1.0,
        weight_decay=0,
        post_spike_weight_decay=0.005,
        reset=0,
        refrac=40,
        thresh=7,
        intensity=150,
        time_steps=1000,
        dt=1,
        enable_astrocyte=True,
        alpha=0.001,
        k=0.2,
        enable_stdp=True,
        diagnostics=diagnostics,
        input_probability=1.0,
        input_mode="paper_bernoulli",
        sensory_generator=rngs["sensory"],
        policy_rng=rngs["action_numpy"],
        optimized_connections=optimized_connections,
    )
    return result, diagnostics


def test_optimized_backend_matches_dense_for_complete_fixed_seed_episode():
    (route_dense, weights_dense, success_dense, _time_dense), diag_dense = _run_one_episode(False)
    (route_fast, weights_fast, success_fast, _time_fast), diag_fast = _run_one_episode(True)

    assert route_fast == route_dense
    assert success_fast == success_dense
    assert np.allclose(weights_fast, weights_dense, atol=1e-7, rtol=0)
    for key in (
        "X_spikes", "Y_spikes", "I_spikes", "tie_decisions",
        "delta_w_stdp_norm", "stdp_step_norm_sum", "weight_drift_norm",
    ):
        assert np.allclose(diag_fast[key], diag_dense[key], atol=1e-7, rtol=0), key
