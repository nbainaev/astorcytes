import csv
import math

from astrocites.experiment import run_experiment
from astrocites.logs import FileLogger


def test_short_training_run_writes_real_training_and_frozen_diagnostics(tmp_path):
    config = {
        "grid_size": 3,
        "start_position": 1,
        "goal_position": 4,
        "n_steps": 1,
        "neuron": {
            "learning_rate": 1,
            "wmin": 0.001,
            "wmax": 1,
            "weight_decay": 0,
            "post_spike_weight_decay": 0.005,
            "reset": 0,
            "refrac": 40,
            "thresh": 7,
        },
        "astrocyte": {"enable": True, "alpha": 0.001, "k": 0.2},
        "reinforce": {
            "enable": True,
            "learning_rate": 0.01,
            "temperature": 1.0,
            "reward_goal": 10.0,
            "step_penalty": -0.1,
            "gamma": 0.99,
            "baseline_decay": 0.01,
            "policy_mix_beta": 0.0,
            "enable_stdp_during_training": True,
            "trace_decay": 0.95,
            "surrogate": {"type": "lif"},
        },
        "simulation": {
            "intensity": 15.0,
            "input_probability": 1.0,
            "input_refractory": 40,
            "time_steps": 1000,
            "dt": 1,
        },
        "experiment": {"num_experiments": 1, "seeds": [42], "num_cycles": 1},
        "output_dir": str(tmp_path / "logs"),
        "run_name": "diagnostics-wiring-test",
        "diagnostics": {
            "enable": True,
            "output_dir": str(tmp_path / "diagnostics"),
            "evaluation_rollouts": 1,
            "evaluation_checkpoints": [1],
            "evaluation_astrocyte_modes": [False, True],
        },
    }
    run_experiment(config, logger=FileLogger(output_dir=config["output_dir"]))

    diagnostic_dir = tmp_path / "diagnostics" / "diagnostics-wiring-test"
    with (diagnostic_dir / "raw_per_seed.csv").open(newline="") as stream:
        episode_rows = list(csv.DictReader(stream))
    train = next(row for row in episode_rows if row["phase"] == "training")
    assert float(train["raw_input_events"]) == 1000.0
    assert int(train["X_spikes"]) > 0
    assert int(train["Y_spikes"]) > 0
    assert int(train["I_spikes"]) > 0
    assert float(train["current_l1_X_Y"]) > 0
    assert float(train["stdp_raw_update_l1"]) > 0
    assert float(train["stdp_step_norm_sum"]) > 0
    assert int(train["decision_count"]) == 1
    assert 0.0 <= float(train["tie_frequency"]) <= 1.0
    assert 0.0 <= float(train["astro_active_fraction"]) <= 1.0
    assert float(train["fraction_weights_outside_bounds"]) == 0.0
    assert math.isfinite(float(train["delta_w_stdp_norm"]))
    assert float(train["delta_w_stdp_norm"]) > 0
    assert float(train["delta_w_rl_norm"]) > 0

    with (diagnostic_dir / "policy_diagnostics.csv").open(newline="") as stream:
        policy_rows = list(csv.DictReader(stream))
    assert policy_rows
    assert all(math.isfinite(float(row["kl_behavior_to_surrogate"])) for row in policy_rows)
    assert all(float(row["eligibility_norm"]) >= 0 for row in policy_rows)
    assert all(float(row["delta_w_stdp_norm_step"]) >= 0 for row in policy_rows)

    with (diagnostic_dir / "evaluation_rollouts.csv").open(newline="") as stream:
        evaluation_rows = list(csv.DictReader(stream))
    assert len(evaluation_rows) == 2
    assert {row["astrocyte_enabled"] for row in evaluation_rows} == {"False", "True"}
    assert all(float(row["weight_drift_norm"]) == 0 for row in evaluation_rows)
    assert all(float(row["fraction_weights_outside_bounds"]) == 0 for row in evaluation_rows)
    assert all(int(row["X_spikes"]) > 0 for row in evaluation_rows)
