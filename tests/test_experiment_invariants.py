from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from astrocites.experiment import (
    run_experiment,
    setup_and_run_simulation,
    setup_and_run_simulation_reinforce,
)


def _tiny_simulation_inputs():
    mask = np.zeros((4, 4), dtype=int)
    mask[0, 1] = mask[1, 0] = 1
    weights = torch.as_tensor(mask, dtype=torch.float32) * 0.6
    return mask, weights, torch.eye(4)


def _base_kwargs(mask, weights, weights_xi, n_steps=1):
    return dict(
        NA=4,
        weights_mask_XY=mask,
        weights_init_XY=weights.clone(),
        weights_init_XI=weights_xi,
        n_steps=n_steps,
        current_position=0,
        goal=1,
        learning_rate=1,
        wmin=0.001,
        wmax=1,
        weight_decay=0,
        post_spike_weight_decay=0.005,
        reset=0,
        refrac=40,
        thresh=7,
        intensity=15,
        time_steps=1000,
        dt=1,
        enable_astrocyte=False,
        alpha=0.001,
        k=0.2,
    )


def test_frozen_simulation_preserves_weights_and_counts_final_action():
    np.random.seed(0)
    torch.manual_seed(0)
    mask, weights, weights_xi = _tiny_simulation_inputs()
    weights_before = weights.clone()

    positions, weights_after, reached_goal, _ = setup_and_run_simulation(
        **_base_kwargs(mask, weights, weights_xi),
        enable_stdp=False,
    )

    assert np.allclose(weights_after, weights_before.numpy())
    assert len(positions) - 1 == 1
    assert positions[-1] == 1
    assert reached_goal == 1


def test_zero_action_simulation_returns_initialized_weights():
    mask, weights, weights_xi = _tiny_simulation_inputs()
    kwargs = _base_kwargs(mask, weights, weights_xi, n_steps=0)

    positions, weights_after, reached_goal, _ = setup_and_run_simulation(
        **kwargs,
        enable_stdp=False,
    )

    assert positions == [0]
    assert np.array_equal(weights_after, weights.numpy())
    assert reached_goal == 0


def test_reinforce_trajectory_counts_final_action():
    np.random.seed(0)
    torch.manual_seed(0)
    mask, weights, weights_xi = _tiny_simulation_inputs()
    kwargs = _base_kwargs(mask, weights, weights_xi)

    positions, _weights_after, reached_goal, _, _ = setup_and_run_simulation_reinforce(
        **kwargs,
        reinforce_lr=0.01,
        temperature=1.0,
        reward_goal=10.0,
        step_penalty=-0.1,
        gamma=0.99,
        baseline_decay=0.01,
        policy_mix_beta=0.0,
        trace_decay=0.95,
        enable_stdp_during_training=False,
        surrogate_kind="spike",
    )

    assert len(positions) - 1 == 1
    assert positions[-1] == 1
    assert reached_goal == 1


def test_run_experiment_rejects_too_few_configured_seeds():
    config = {
        "grid_size": 2,
        "start_position": 0,
        "goal_position": 3,
        "n_steps": 0,
        "experiment": {"num_experiments": 2, "seeds": [42]},
    }

    with pytest.raises(ValueError, match="provide one seed per requested experiment"):
        run_experiment(config, logger=object())


def test_default_config_has_a_fixed_seed_for_every_experiment():
    config_path = Path(__file__).parents[1] / "configs" / "default.yaml"
    config = yaml.safe_load(config_path.read_text())

    assert len(config["experiment"]["seeds"]) == config["experiment"]["num_experiments"]
    assert config["experiment"]["seeds"] == [42, 223, 1337]
    assert config["learning"]["enable_stdp"] is True
    assert "enable_stdp_during_training" not in config["reinforce"]


def test_stdp_can_be_disabled_independently_of_reinforce(monkeypatch):
    from astrocites.learning import WeightDependentPostPre
    mask, weights, weights_xi = _tiny_simulation_inputs()
    kwargs = _base_kwargs(mask, weights, weights_xi)
    calls = []
    def deterministic_update(self, current_position=None, adjacent_positions=None, **_kwargs):
        calls.append((current_position, tuple(adjacent_positions)))
        self.connection.w.data[current_position, adjacent_positions[0]] += 1e-4
    monkeypatch.setattr(WeightDependentPostPre, "update", deterministic_update)
    frozen = {}
    setup_and_run_simulation(**kwargs, enable_stdp=False, diagnostics=frozen,
                             input_mode="legacy_static_positive")
    assert calls == []
    assert frozen["delta_w_stdp_norm"] == 0.0
    plastic = {}
    setup_and_run_simulation(**kwargs, enable_stdp=True, diagnostics=plastic,
                             input_mode="legacy_static_positive")
    assert calls
    assert plastic["delta_w_stdp_norm"] > 0


def test_frozen_reinforce_evaluation_preserves_weights_baseline_and_softmax(monkeypatch):
    from astrocites.connection import Connection
    from astrocites.utils import create_adjacency_matrix
    mask = create_adjacency_matrix(3)
    weights = torch.as_tensor(mask, dtype=torch.float32) * 0.5
    weights[1, 0] = 0.8
    weights[1, 2] = 0.5
    weights_before = weights.clone()
    xi = torch.eye(9)
    baseline = 2.75
    captured = []
    def forbid_update(*_args, **_kwargs):
        raise AssertionError("frozen evaluation attempted to update RL state")
    def capture_policy(probs, num_samples, generator=None):
        captured.append(probs.detach().clone())
        return torch.tensor([0])
    monkeypatch.setattr(Connection, "accumulate_trace", forbid_update)
    monkeypatch.setattr(Connection, "compute_and_apply_reinforce_update", forbid_update)
    monkeypatch.setattr(torch, "multinomial", capture_policy)
    diagnostics = {}
    positions, weights_after, _success, _elapsed, baseline_after = setup_and_run_simulation_reinforce(
        NA=9, weights_mask_XY=mask, weights_init_XY=weights, weights_init_XI=xi,
        n_steps=1, current_position=1, goal=0, learning_rate=1, wmin=0.001, wmax=1,
        weight_decay=0, post_spike_weight_decay=0.005, reset=0, refrac=40, thresh=7,
        intensity=15, time_steps=1000, dt=1, enable_astrocyte=False,
        alpha=0.001, k=0.2, reinforce_lr=0.0, temperature=1.0, reward_goal=10,
        step_penalty=-0.1, gamma=0.99, baseline_decay=0.01,
        policy_mix_beta=1.0, trace_decay=0.95, enable_stdp_during_training=False,
        surrogate_kind="spike", running_baseline=baseline,
        input_probability=1.0, apply_reinforce_update=False,
        diagnostics=diagnostics,
    )
    assert positions == [1, 0]
    assert len(captured) == 1
    expected = torch.softmax(torch.tensor([0.8, 0.5, 0.5]), dim=0)
    assert torch.allclose(captured[0], expected)
    assert torch.equal(torch.as_tensor(weights_after), weights_before)
    assert baseline_after == baseline
    assert diagnostics["reward_return"] == pytest.approx(9.9)


def test_main_reference_simulation_does_not_mutate_checkpoint_tensor():
    mask, weights, weights_xi = _tiny_simulation_inputs()
    before = weights.clone()
    kwargs = _base_kwargs(mask, weights, weights_xi)
    setup_and_run_simulation(**kwargs, baseline_mode="main_reference",
                             enable_stdp=True, input_probability=0.0)
    assert torch.equal(weights, before)


def test_main_reference_input_and_structural_mask_timing():
    mask, weights, weights_xi = _tiny_simulation_inputs()
    diagnostics = {}
    kwargs = _base_kwargs(mask, weights, weights_xi, n_steps=0)
    _, source_weights, _, _ = setup_and_run_simulation(
        **kwargs, baseline_mode="main_reference", diagnostics=diagnostics,
    )
    _, plausible_weights, _, _ = setup_and_run_simulation(
        **kwargs, baseline_mode="plausible", diagnostics={},
    )
    forbidden = mask == 0
    assert np.allclose(source_weights[forbidden], 0.001)
    assert np.allclose(plausible_weights[forbidden], 0.0)

    kwargs["n_steps"] = 1
    source_diagnostics = {}
    setup_and_run_simulation(
        **kwargs, baseline_mode="main_reference", enable_stdp=False,
        diagnostics=source_diagnostics,
    )
    assert source_diagnostics["X_spikes"] == 25


def test_condition_labels_cover_all_independent_learning_combinations():
    from astrocites.experiment import _condition_label

    assert _condition_label(False, False) == "none"
    assert _condition_label(True, False) == "stdp"
    assert _condition_label(False, True) == "reinforce"
    assert _condition_label(True, True) == "stdp+reinforce"


def test_run_experiment_wires_stdp_switch_when_reinforce_is_off(tmp_path):
    import csv
    from astrocites.logs import FileLogger

    config = {
        "grid_size": 2,
        "start_position": 0,
        "goal_position": 1,
        "n_steps": 1,
        "learning": {"enable_stdp": False},
        "reinforce": {"enable": False, "enable_stdp_during_training": False},
        "simulation": {
            "intensity": 15,
            "input_probability": 1.0,
            "input_refractory": 40,
            "time_steps": 1000,
            "dt": 1,
        },
        "experiment": {"num_experiments": 1, "seeds": [21], "num_cycles": 1},
        "protocol": {"verification_mode": "none"},
        "output_dir": str(tmp_path / "logs"),
        "run_name": "independent-stdp-switch",
        "diagnostics": {
            "enable": True,
            "output_dir": str(tmp_path / "diagnostics"),
            "evaluation_rollouts": 0,
        },
    }
    run_experiment(config, logger=FileLogger(output_dir=config["output_dir"]))
    path = tmp_path / "diagnostics" / "independent-stdp-switch" / "raw_per_seed.csv"
    with path.open(newline="") as stream:
        training = next(row for row in csv.DictReader(stream) if row["phase"] == "training")

    assert training["condition"] == "none"
    assert training["enable_stdp"] == "False"
    assert training["enable_reinforce"] == "False"


def test_stdp_switch_rejects_conflicting_canonical_and_legacy_values():
    from astrocites.experiment import _resolve_enable_stdp

    with pytest.raises(ValueError, match="Conflicting STDP settings"):
        _resolve_enable_stdp(
            {"enable_stdp": True},
            {"enable_stdp_during_training": False},
        )

    assert _resolve_enable_stdp(
        {"enable_stdp": False}, {"enable_stdp_during_training": False}
    ) is False
    assert _resolve_enable_stdp({}, {"enable_stdp_during_training": False}) is False


def test_torch_thread_limit_is_configurable_independent_of_diagnostics(monkeypatch):
    from astrocites.experiment import _configure_runtime

    calls = []
    monkeypatch.setattr(torch, "set_num_threads", calls.append)
    _configure_runtime({
        "runtime": {"torch_num_threads": 3},
        "diagnostics": {"enable": False},
    })
    _configure_runtime({
        "runtime": {"torch_num_threads": 2},
        "diagnostics": {"enable": True},
    })
    assert calls == [3, 2]
    with pytest.raises(ValueError, match="positive integer"):
        _configure_runtime({"runtime": {"torch_num_threads": 0}})


def test_run_experiment_precomputes_adjacency_once_per_grid(monkeypatch, tmp_path):
    import astrocites.experiment as experiment

    actual_build = experiment._build_adjacency_data
    builds = []
    simulation_calls = []

    def counted_build(mask):
        result = actual_build(mask)
        builds.append(result)
        return result

    def no_op_simulation(**kwargs):
        simulation_calls.append(kwargs)
        return (
            [kwargs["current_position"]],
            kwargs["weights_init_XY"].detach().cpu().numpy(),
            0,
            0.0,
        )

    class MemoryLogger:
        def __init__(self):
            self.active_exp_dir = tmp_path

        def start_experiment(self, name):
            self.active_exp_dir = tmp_path / name
            self.active_exp_dir.mkdir()

        def log_params(self, *_args, **_kwargs):
            pass

        def log_metrics(self, *_args, **_kwargs):
            pass

        def finish(self):
            pass

    monkeypatch.setattr(experiment, "_build_adjacency_data", counted_build)
    monkeypatch.setattr(experiment, "setup_and_run_simulation", no_op_simulation)
    run_experiment({
        "grid_size": 2,
        "start_position": 0,
        "goal_position": 3,
        "n_steps": 0,
        "learning": {"enable_stdp": False},
        "reinforce": {"enable": False},
        "experiment": {"num_experiments": 1, "seeds": [7], "num_cycles": 0},
        "protocol": {"verification_mode": "none"},
        "output_dir": str(tmp_path),
    }, logger=MemoryLogger())

    assert len(builds) == 1
    assert len(simulation_calls) == 1
    assert simulation_calls[0]["adjacency_data"] is builds[0]
    assert tuple(simulation_calls[0]["mask_tensor"].shape) == (4, 4)
