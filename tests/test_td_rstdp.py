import numpy as np
import pytest
import torch

from astrocites.connection import Connection
from astrocites.critic import TabularTDCritic
from astrocites.experiment import (
    select_rstdp_modulator,
    setup_and_run_simulation_td_rstdp,
)
from astrocites.learning import EligibilityPostPre
from astrocites.nodes import Nodes
from astrocites.utils import create_adjacency_matrix


def _eligibility_connection(weights=None, *, wmin=0.1, wmax=0.8,
                            post_spike_weight_decay=0.0):
    source = Nodes(n=2, traces=True)
    target = Nodes(n=2, traces=True)
    if weights is None:
        weights = torch.full((2, 2), 0.5)
    connection = Connection(
        source, target, update_rule=EligibilityPostPre,
        w=weights.clone(), nu=[10, 10], wmin=wmin, wmax=wmax,
        post_spike_weight_decay=post_spike_weight_decay,
        enable_rstdp_state=True, clamp_initial_weights=False,
    )
    return connection


def test_eligibility_stdp_records_local_tag_without_direct_weight_update():
    connection = _eligibility_connection(post_spike_weight_decay=10.0)
    connection.update_rule.STDP_base = torch.zeros_like(connection.update_rule.STDP_base)
    connection.update_rule.STDP_base[5, 60] = 0.02
    connection.source.s[0, 0] = True
    connection.target.x_neg[0, 1] = 1.0
    before = connection.w.detach().clone()

    connection.update_rule.update(current_position=0, adjacent_positions=[1])

    assert torch.equal(connection.w, before)
    assert connection.rstdp_drive[0, 1] > 0
    assert torch.count_nonzero(connection.rstdp_drive) == 1
    assert torch.count_nonzero(connection.eligibility) == 0


def test_eligibility_stdp_excludes_post_spike_weight_decay():
    connection = _eligibility_connection(post_spike_weight_decay=10.0)
    connection.update_rule.STDP_base = torch.zeros_like(connection.update_rule.STDP_base)
    connection.target.s[0, 1] = True
    before = connection.w.detach().clone()

    connection.update_rule.update(current_position=0, adjacent_positions=[1])

    assert torch.equal(connection.w, before)
    assert connection.rstdp_drive[0, 1] == 0


def test_action_level_eligibility_is_decay_times_old_plus_window_tag():
    connection = _eligibility_connection()
    connection.rstdp_eligibility[0, 1] = 2.0
    connection.rstdp_drive[0, 1] = 3.0

    connection.update_rstdp_eligibility(0.25)

    assert connection.rstdp_eligibility[0, 1].item() == pytest.approx(3.5)
    assert torch.count_nonzero(connection.rstdp_drive) == 0


@pytest.mark.parametrize(
    "td_error,expected",
    [(0.0, 0.5), (1.0, 0.7), (-1.0, 0.3)],
)
def test_td_error_sign_controls_actor_weight_update(td_error, expected):
    connection = _eligibility_connection(torch.tensor([[0.0, 0.5], [0.0, 0.0]]))
    connection.rstdp_eligibility[0, 1] = 2.0
    mask = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    before = connection.w.detach().clone()

    connection.apply_rstdp_update(td_error, learning_rate=0.1, mask=mask)

    assert connection.w[0, 1].item() == pytest.approx(expected)
    assert connection.w[1, 0] == 0
    if td_error == 0.0:
        assert torch.equal(connection.w, before)
    assert torch.count_nonzero(connection.eligibility) == 0


def test_td_rstdp_clamps_weights_and_keeps_structural_zeros():
    weights = torch.tensor([[0.0, 0.79], [0.0, 0.0]])
    connection = _eligibility_connection(weights, wmin=0.1, wmax=0.8)
    connection.rstdp_eligibility[0, 1] = 10.0
    mask = torch.tensor([[0.0, 1.0], [0.0, 0.0]])

    connection.apply_rstdp_update(1.0, learning_rate=1.0, mask=mask)

    assert connection.w[0, 1].item() == pytest.approx(0.8)
    assert torch.count_nonzero(connection.w[mask == 0]) == 0
    assert connection.w[0, 1] <= connection.wmax


def test_tabular_td0_terminal_nonterminal_and_update():
    critic = TabularTDCritic(n_states=3, learning_rate=0.1, gamma=0.9)
    critic.values[:] = torch.tensor([3.0, 0.0, 10.0])

    terminal_delta = critic.compute_td_error(0, reward=2.0, next_state=2, done=True)
    nonterminal_delta = critic.compute_td_error(0, reward=2.0, next_state=2, done=False)
    assert terminal_delta.item() == pytest.approx(-1.0)
    assert nonterminal_delta.item() == pytest.approx(8.0)

    critic.update(0, nonterminal_delta)
    assert critic.values[0].item() == pytest.approx(3.8)


def test_forced_reward_path_propagates_value_backwards():
    critic = TabularTDCritic(n_states=5, learning_rate=0.2, gamma=0.9)
    transitions = [
        (0, 1, 0.0, False),
        (1, 2, 0.0, False),
        (2, 3, 0.0, False),
        (3, 4, 1.0, True),
    ]
    for _ in range(200):
        for state, next_state, reward, done in transitions:
            delta = critic.compute_td_error(state, reward, next_state, done)
            critic.update(state, delta)

    assert critic.values[3] > critic.values[2] > critic.values[1] > critic.values[0]


def test_rstdp_modulator_selects_td_error_or_immediate_reward():
    assert select_rstdp_modulator("td_error", -0.7, 1.25) == 1.25
    assert select_rstdp_modulator("reward", -0.7, 1.25) == -0.7
    with pytest.raises(ValueError, match="Unknown R-STDP actor modulator"):
        select_rstdp_modulator("advantage", 0.0, 0.0)


def test_reward_rstdp_training_path_runs_without_critic():
    mask = create_adjacency_matrix(2)
    weights = torch.as_tensor(mask, dtype=torch.float32) * 0.6
    diagnostics = {}
    steps = []

    positions, _weights, _success, _elapsed, episode_return = (
        setup_and_run_simulation_td_rstdp(
            NA=4, weights_mask_XY=mask, weights_init_XY=weights,
            weights_init_XI=torch.eye(4), n_steps=1,
            current_position=0, goal=3, learning_rate=1.0,
            wmin=0.001, wmax=1.0, weight_decay=0.0,
            post_spike_weight_decay=0.005, reset=0, refrac=40, thresh=7,
            intensity=150.0, time_steps=1000, dt=1,
            enable_astrocyte=False, alpha=0.001, k=0.2,
            critic=None, actor_learning_rate=0.001,
            eligibility_tau_ms=1000.0, temperature=1.0,
            reward_goal=10.0, step_penalty=-0.1,
            input_probability=1.0, input_mode="paper_bernoulli",
            sensory_generator=torch.Generator().manual_seed(20),
            policy_generator=torch.Generator().manual_seed(21),
            apply_actor_update=True, apply_critic_update=False,
            actor_modulator="reward", gamma=0.99,
            diagnostics=diagnostics, actor_step_diagnostics=steps,
        )
    )

    assert len(positions) == 2
    assert episode_return == pytest.approx(steps[0]["reward"])
    assert steps[0]["actor_modulator"] == "reward"
    assert steps[0]["actor_signal"] == pytest.approx(steps[0]["reward"])
    assert np.isnan(steps[0]["td_error"])
    assert np.isnan(diagnostics["critic_value_start"])
    assert np.isnan(diagnostics["mean_abs_td_error"])
    assert np.isfinite(diagnostics["mean_actor_update_norm"])


def test_one_training_action_logs_td_and_updates_critic():
    mask = create_adjacency_matrix(2)
    weights = torch.as_tensor(mask, dtype=torch.float32) * 0.6
    critic = TabularTDCritic(n_states=4, learning_rate=0.1, gamma=0.99)
    diagnostics = {}
    steps = []

    setup_and_run_simulation_td_rstdp(
        NA=4, weights_mask_XY=mask, weights_init_XY=weights,
        weights_init_XI=torch.eye(4), n_steps=1,
        current_position=0, goal=3, learning_rate=1.0,
        wmin=0.001, wmax=1.0, weight_decay=0.0,
        post_spike_weight_decay=0.005, reset=0, refrac=40, thresh=7,
        intensity=150.0, time_steps=1000, dt=1,
        enable_astrocyte=False, alpha=0.001, k=0.2,
        critic=critic, actor_learning_rate=0.001,
        eligibility_tau_ms=1000.0, temperature=1.0,
        reward_goal=10.0, step_penalty=-0.1,
        input_probability=1.0, input_mode="paper_bernoulli",
        sensory_generator=torch.Generator().manual_seed(20),
        policy_generator=torch.Generator().manual_seed(21),
        apply_actor_update=True, apply_critic_update=True,
        diagnostics=diagnostics, actor_step_diagnostics=steps,
    )

    assert steps and steps[0]["done"] == 1
    assert steps[0]["td_target"] == pytest.approx(steps[0]["reward"])
    assert steps[0]["td_error"] != 0
    assert critic.values[0] != 0
    assert np.isfinite(diagnostics["mean_actor_update_norm"])
    assert diagnostics["delta_w_stdp_norm"] == 0.0
    assert diagnostics["fraction_weights_outside_bounds"] == 0.0


def test_tabular_critic_save_writes_values_and_grid_heatmap(tmp_path):
    critic = TabularTDCritic(n_states=9, learning_rate=0.1, gamma=0.9)
    critic.values[:] = torch.arange(9, dtype=torch.float32)

    saved_path = critic.save(tmp_path, grid_size=3, make_heatmap=True)

    assert np.array_equal(np.load(saved_path), np.arange(9, dtype=np.float32))
    assert (tmp_path / "critic_values.png").is_file()


@pytest.mark.parametrize("policy_mode", ["spike_softmax", "common_spike_argmax"])
def test_frozen_td_rstdp_rollout_preserves_actor_and_critic(policy_mode):
    from astrocites.utils import create_adjacency_matrix

    mask = create_adjacency_matrix(3)
    weights = torch.as_tensor(mask, dtype=torch.float32) * 0.6
    weights_before = weights.clone()
    critic = TabularTDCritic(n_states=9, learning_rate=0.1, gamma=0.99)
    critic.values[:] = torch.linspace(0.0, 0.8, 9)
    critic_before = critic.values.clone()
    diagnostics = {}
    step_rows = []
    result = setup_and_run_simulation_td_rstdp(
        NA=9, weights_mask_XY=mask, weights_init_XY=weights,
        weights_init_XI=torch.eye(9), n_steps=1,
        current_position=1, goal=8, learning_rate=1.0,
        wmin=0.001, wmax=1.0, weight_decay=0.0,
        post_spike_weight_decay=0.005, reset=0, refrac=40, thresh=7,
        intensity=150.0, time_steps=1000, dt=1,
        enable_astrocyte=False, alpha=0.001, k=0.2,
        critic=critic, actor_learning_rate=0.001,
        eligibility_tau_ms=1000.0, temperature=1.0,
        reward_goal=10.0, step_penalty=-0.1,
        input_probability=1.0, input_mode="paper_bernoulli",
        sensory_generator=torch.Generator().manual_seed(10),
        policy_generator=torch.Generator().manual_seed(11),
        policy_rng=np.random.RandomState(11),
        apply_actor_update=False, apply_critic_update=False,
        policy_mode=policy_mode,
        diagnostics=diagnostics, actor_step_diagnostics=step_rows,
        decision_metadata={"phase": "frozen_evaluation"},
    )

    assert len(result[0]) == 2
    assert result[0][1] in {0, 2, 4}
    assert np.array_equal(result[1], weights_before.numpy())
    assert torch.equal(critic.values, critic_before)
    assert diagnostics["mean_actor_update_norm"] == 0.0
    assert diagnostics["mean_eligibility_norm"] == 0.0
    assert step_rows and step_rows[0]["actor_update_norm"] == 0.0


def test_critic_is_shared_across_cycles_and_reinitialized_for_each_seed(monkeypatch, tmp_path):
    import astrocites.experiment as experiment
    from astrocites.logs import FileLogger

    seen = []

    def fake_td_episode(**kwargs):
        critic = kwargs["critic"]
        seen.append((id(critic), float(critic.values[0].item())))
        critic.update(0, torch.tensor(1.0))
        weights = kwargs["weights_init_XY"].detach().cpu().numpy()
        return [0, 1], weights, 1, 0.0, 1.0

    monkeypatch.setattr(experiment, "setup_and_run_simulation_td_rstdp", fake_td_episode)
    config = {
        "grid_size": 2,
        "start_position": 0,
        "goal_position": 1,
        "n_steps": 0,
        "learning": {"enable_stdp": False},
        "reinforce": {"enable": False, "policy_mix_beta": 0.0},
        "actor_critic": {
            "enable": True, "actor_rule": "td_rstdp", "critic": "tabular_td0",
            "actor_learning_rate": 0.001, "critic_learning_rate": 0.1,
            "gamma": 0.99, "eligibility_tau_ms": 1000,
            "temperature": 1.0, "policy": "spike_softmax",
        },
        "experiment": {"num_experiments": 2, "seeds": [10, 11], "num_cycles": 2},
        "protocol": {"verification_mode": "none"},
        "output_dir": str(tmp_path / "logs"),
    }

    experiment.run_experiment(config, logger=FileLogger(output_dir=config["output_dir"]))

    assert [value for _critic_id, value in seen] == pytest.approx([0.0, 0.1, 0.0, 0.1])
    assert seen[0][0] == seen[1][0]
    assert seen[2][0] == seen[3][0]
    assert seen[0][0] != seen[2][0]


@pytest.mark.parametrize(
    "learning,reinforce,message",
    [
        ({"enable_stdp": True}, {"enable": False}, "learning.enable_stdp=false"),
        ({"enable_stdp": False}, {"enable": True}, "cannot be enabled together"),
    ],
)
def test_td_rstdp_rejects_mixed_plasticity_paths(learning, reinforce, message):
    from astrocites.experiment import run_experiment

    config = {
        "grid_size": 2,
        "start_position": 0,
        "goal_position": 1,
        "n_steps": 0,
        "learning": learning,
        "reinforce": reinforce,
        "actor_critic": {"enable": True},
        "experiment": {"num_experiments": 1, "seeds": [1], "num_cycles": 0},
    }
    with pytest.raises(ValueError, match=message):
        run_experiment(config, logger=object())
