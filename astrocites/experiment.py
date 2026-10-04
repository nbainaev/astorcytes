import os
import random
import json
import csv
import math
from pathlib import Path
import numpy as np
import torch
import scipy.io as sio
from time import time as t

from astrocites.nodes import Input, LIFNodes
from astrocites.connection import Connection
from astrocites.learning import WeightDependentPostPre, EligibilityPostPre, NoOp
from astrocites.critic import TabularTDCritic
from astrocites.network import Network, SpikeCountMonitor
from astrocites.utils import (
    bernoulli_loader, create_adjacency_matrix, make_rng_streams, derive_seed,
)
from astrocites.surrogate import (
    compute_scale_c,
    eligibility_gradient,
    spike_eligibility,
    build_lif_params,
    rate_and_deriv,
)
from astrocites.logs import ExperimentLogger, FileLogger


def _sample_input_spike_train(current_position, n_neurons, intensity, time_steps, dt,
                              probability=None, generator=None, input_mode="bernoulli"):
    """Sample sensory events; ``intensity`` is expected raw events per window.

    Bernoulli candidates are subsequently filtered by the Input layer's
    source-faithful refractory gate.
    """
    input_probs = torch.zeros(1, n_neurons)
    if input_mode == "legacy_static_positive":
        input_probs[0, current_position] = (intensity / time_steps) * dt
        return input_probs
    if input_mode not in {"bernoulli", "paper_bernoulli"}:
        raise ValueError(f"Unknown input mode: {input_mode!r}")
    input_probs[0, current_position] = (
        (intensity / time_steps) * dt if probability is None else probability
    )
    return torch.stack(
        list(bernoulli_loader(input_probs, time=time_steps, dt=dt, generator=generator)),
        dim=0,
    ).unsqueeze(1)


def _fill_simulation_diagnostics(
    diagnostics, positions, reached_goal, elapsed, connection, weight_mask,
    initial_weights, stdp_delta, input_spikes, output_spikes,
    inhibitory_spikes=0, raw_input_events=np.nan, tie_decisions=0,
    decision_count=0, behavior_entropy_sum=0.0, current_l1=None,
    stdp_step_norm_sum=0.0, stdp_raw_update_l1=0.0,
    astro_active_neuron_steps=0, astro_neuron_steps=0,
    astro_threshold_drop_sum=0.0, reinforce_delta=None, reinforce_preclip=None,
    reward_return=0.0, baseline_before=None, baseline_after=None,
    update_interaction=None, actor_critic_metrics=None,
):
    if diagnostics is None:
        return
    allowed_weights = connection.w.detach()[weight_mask.bool()]
    outside_bounds = (
        (allowed_weights < connection.wmin) | (allowed_weights > connection.wmax)
    )
    diagnostics.update({
        "success": int(reached_goal),
        "route_length": len(positions) - 1,
        "raw_input_events": float(raw_input_events),
        "X_spikes": int(input_spikes),
        "Y_spikes": int(output_spikes),
        "I_spikes": int(inhibitory_spikes),
        "tie_decisions": int(tie_decisions),
        "decision_count": int(decision_count),
        "tie_frequency": float(tie_decisions / decision_count) if decision_count else 0.0,
        "behavior_entropy_nats_per_decision": (
            float(behavior_entropy_sum / decision_count) if decision_count else 0.0
        ),
        "stdp_step_norm_sum": float(stdp_step_norm_sum),
        "stdp_raw_update_l1": float(stdp_raw_update_l1),
        "astro_active_fraction": (
            float(astro_active_neuron_steps / astro_neuron_steps)
            if astro_neuron_steps else 0.0
        ),
        "astro_mean_threshold_drop": (
            float(astro_threshold_drop_sum / astro_neuron_steps)
            if astro_neuron_steps else 0.0
        ),
        "net_weight_displacement_norm": float(
            torch.linalg.vector_norm(connection.w.detach() - initial_weights).item()
        ),
        "w_min": float(allowed_weights.min().item()) if allowed_weights.numel() else 0.0,
        "w_max": float(allowed_weights.max().item()) if allowed_weights.numel() else 0.0,
        "fraction_weights_outside_bounds": (
            float(outside_bounds.float().mean().item()) if outside_bounds.numel() else 0.0
        ),
        "delta_w_stdp_norm": float(torch.linalg.vector_norm(stdp_delta).item()),
        "delta_w_rl_preclip_norm": float(
            torch.linalg.vector_norm(reinforce_preclip).item()
        ) if reinforce_preclip is not None else 0.0,
        "delta_w_rl_norm": float(
            torch.linalg.vector_norm(reinforce_delta).item()
        ) if reinforce_delta is not None else 0.0,
        "reward_return": float(reward_return),
        "baseline_before": baseline_before,
        "baseline_after": baseline_after,
        "weight_drift_norm": float(
            torch.linalg.vector_norm(connection.w.detach() - initial_weights).item()
        ),
        "elapsed_seconds": float(elapsed),
    })
    if current_l1:
        for connection_name, value in current_l1.items():
            diagnostics[f"current_l1_{connection_name}"] = float(value)
    if update_interaction:
        diagnostics.update(update_interaction)
    if actor_critic_metrics:
        diagnostics.update(actor_critic_metrics)


def _apply_selected_action_stdp_gain(connection, weights_before, current_position,
                                      selected_position, gain):
    if gain == 1.0:
        return
    raw_delta = (
        connection.w.detach()[current_position, selected_position]
        - (weights_before[current_position, selected_position] if weights_before.ndim == 2 else weights_before)
    )
    connection.w.data[current_position, selected_position] = (
        (weights_before[current_position, selected_position] if weights_before.ndim == 2 else weights_before) + gain * raw_delta
    )


def _categorical_entropy(probs):
    return float((-(probs * probs.clamp_min(1e-12).log()).sum()).item())


def _select_action_from_counts(spike_counts, adjacency_row, current_position, rng=None):
    """Source-notebook argmax with uniform tie breaking over valid neighbors."""
    summed = np.squeeze(np.asarray(spike_counts)) + np.asarray(adjacency_row)
    max_val = np.max(summed)
    candidates = np.where(
        (summed == max_val) & (np.asarray(adjacency_row) == 1)
    )[0]
    candidates = candidates[candidates != current_position]
    selector = np.random if rng is None else rng
    return int(selector.choice(candidates)), int(len(candidates))


def _build_adjacency_data(weights_mask_XY):
    mask = np.asarray(weights_mask_XY)
    adjacency_lists = []
    adjacency_tensors = []
    for source, row in enumerate(mask):
        adjacent = [int(index) for index in np.flatnonzero(row) if index != source]
        adjacency_lists.append(adjacent)
        adjacency_tensors.append(torch.as_tensor(adjacent, dtype=torch.long))
    return adjacency_lists, adjacency_tensors


def _resolve_enable_stdp(learning_cfg, reinforce_cfg):
    """Resolve the canonical STDP flag and validate its legacy alias."""
    canonical = learning_cfg.get("enable_stdp")
    legacy = reinforce_cfg.get("enable_stdp_during_training")
    if canonical is not None and legacy is not None:
        if bool(canonical) != bool(legacy):
            raise ValueError(
                "Conflicting STDP settings: use learning.enable_stdp; "
                "reinforce.enable_stdp_during_training is a legacy alias"
            )
        return bool(canonical)
    if canonical is not None:
        return bool(canonical)
    if legacy is not None:
        return bool(legacy)
    return True


def _configure_runtime(config):
    runtime_cfg = config.get("runtime", {})
    num_threads = runtime_cfg.get("torch_num_threads")
    if num_threads is None:
        return
    num_threads = int(num_threads)
    if num_threads < 1:
        raise ValueError("runtime.torch_num_threads must be a positive integer")
    torch.set_num_threads(num_threads)


def _condition_label(enable_stdp, enable_reinforce):
    if enable_stdp and enable_reinforce:
        return "stdp+reinforce"
    if enable_stdp:
        return "stdp"
    if enable_reinforce:
        return "reinforce"
    return "none"


def select_rstdp_modulator(actor_modulator, immediate_reward, td_error):
    """Select the global third factor for a reward/TD-modulated STDP actor."""
    if actor_modulator == "td_error":
        return float(td_error)
    if actor_modulator == "reward":
        return float(immediate_reward)
    raise ValueError(f"Unknown R-STDP actor modulator: {actor_modulator!r}")

def _categorical_divergences(p, q):
    p = p.clamp_min(1e-12)
    q = q.clamp_min(1e-12)
    p = p / p.sum()
    q = q / q.sum()
    midpoint = 0.5 * (p + q)
    kl = torch.sum(p * torch.log(p / q))
    js = 0.5 * torch.sum(p * torch.log(p / midpoint)) + 0.5 * torch.sum(q * torch.log(q / midpoint))
    return float(kl.item()), float(js.item())


def _compute_policy_diagnostics(probs, spike_prefs, chosen_idx, diagnostic_I,
                                temperature, surrogate_params, thresh, dt,
                                tc_decay, refrac, grad_slice, time_steps):
    params = surrogate_params or build_lif_params(thresh, dt, tc_decay, refrac)
    diagnostic_rate, _ = rate_and_deriv(diagnostic_I, "lif", params)
    probs_surrogate = torch.softmax(diagnostic_rate / temperature, dim=0)
    kl, js = _categorical_divergences(probs, probs_surrogate)
    eligibility_spike = spike_eligibility(probs, spike_prefs, chosen_idx, time_steps, temperature)
    grad_norm = float(torch.linalg.vector_norm(grad_slice).item())
    spike_grad_norm = float(torch.linalg.vector_norm(eligibility_spike).item())
    cosine = float("nan")
    if grad_norm > 1e-12 and spike_grad_norm > 1e-12:
        cosine = float(torch.nn.functional.cosine_similarity(
            grad_slice.reshape(1, -1), eligibility_spike.reshape(1, -1), dim=1,
        ).item())
    return probs_surrogate, kl, js, grad_norm, spike_grad_norm, cosine


def _write_rows(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _stats_record(label, metric, values, n_seeds, rng):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"group": label, "metric": metric, "N_seeds": n_seeds}
    boot = values[rng.integers(0, len(values), size=(5000, len(values)))].mean(axis=1)
    return {
        "group": label, "metric": metric, "N_seeds": int(n_seeds),
        "mean": float(values.mean()), "median": float(np.median(values)),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "IQR": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
        "ci95_mean_low": float(np.quantile(boot, 0.025)),
        "ci95_mean_high": float(np.quantile(boot, 0.975)),
    }


def _build_run_summaries(episode_rows, evaluation_rows, seed):
    records = []
    rng = np.random.default_rng(seed)
    episode_groups = {}
    for row in episode_rows:
        if row.get("phase") == "training":
            key = (row.get("condition"), row.get("cycle"))
            episode_groups.setdefault(key, {}).setdefault(row["seed"], []).append(row)
    for (condition, cycle), per_seed in episode_groups.items():
        label = f"training:{condition}:cycle={cycle}"
        for metric in ("route_length", "success", "X_spikes", "Y_spikes", "I_spikes",
                       "raw_input_events", "tie_frequency",
                       "delta_w_stdp_norm", "stdp_step_norm_sum", "stdp_raw_update_l1",
                       "delta_w_rl_preclip_norm", "delta_w_rl_norm",
                       "stdp_rl_update_cosine", "stdp_rl_opposite_sign_fraction",
                       "astro_active_fraction", "astro_mean_threshold_drop",
                       "net_weight_displacement_norm", "reward_return"):
            seed_values = []
            for rows in per_seed.values():
                values = [row.get(metric) for row in rows if row.get(metric) is not None]
                if values:
                    seed_values.append(float(np.mean(values)))
            records.append(_stats_record(label, metric, seed_values, len(seed_values), rng))

    eval_groups = {}
    for row in evaluation_rows:
        key = (
            row["condition"], row["astrocyte_enabled"], row["checkpoint_cycle"],
            row.get("evaluation_policy", "default"),
        )
        eval_groups.setdefault(key, {}).setdefault(row["seed"], []).append(row)
    for (algorithm, astro, cycle, evaluation_policy), per_seed in eval_groups.items():
        seed_metric_values = {metric: [] for metric in (
            "success_rate", "failure_rate", "mean_route_length",
            "mean_success_route_length", "median_route_length",
            "mean_X_spikes", "mean_Y_spikes", "mean_I_spikes",
            "mean_tie_frequency", "mean_astro_active_fraction",
            "mean_astro_mean_threshold_drop", "mean_delta_w_stdp_norm",
            "mean_delta_w_rl_norm", "weight_drift_norm",
        )}
        for seed_rows in per_seed.values():
            route = [float(row["route_length"]) for row in seed_rows]
            success = [int(row["success"]) for row in seed_rows]
            successful_routes = [float(row["route_length"]) for row in seed_rows if row["success"]]
            seed_metric_values["success_rate"].append(float(np.mean(success)))
            seed_metric_values["failure_rate"].append(1.0 - float(np.mean(success)))
            seed_metric_values["mean_route_length"].append(float(np.mean(route)))
            seed_metric_values["mean_success_route_length"].append(
                float(np.mean(successful_routes)) if successful_routes else np.nan
            )
            seed_metric_values["median_route_length"].append(float(np.median(route)))
            for metric in (
                "X_spikes", "Y_spikes", "I_spikes", "tie_frequency",
                "astro_active_fraction", "astro_mean_threshold_drop",
                "delta_w_stdp_norm", "delta_w_rl_norm", "weight_drift_norm",
            ):
                vals = [float(row[metric]) for row in seed_rows if row.get(metric) is not None]
                metric_key = "weight_drift_norm" if metric == "weight_drift_norm" else f"mean_{metric}"
                seed_metric_values[metric_key].append(float(np.mean(vals)) if vals else np.nan)
        label = (
            f"frozen_eval:{algorithm}:astro={astro}:cycle={cycle}"
            f":policy={evaluation_policy}"
        )
        for metric, values in seed_metric_values.items():
            valid = [value for value in values if np.isfinite(value)]
            records.append(_stats_record(label, metric, valid, len(valid), rng))
    return records


def setup_and_run_simulation(
    NA, weights_mask_XY, weights_init_XY, weights_init_XI, n_steps,
    current_position, goal, learning_rate, wmin, wmax, weight_decay,
    post_spike_weight_decay, reset, refrac, thresh, intensity, time_steps, dt,
    enable_astrocyte, alpha, k,
    enable_stdp=True,
    diagnostics=None,
    input_refractory=None,
    input_probability=None,
    input_mode="bernoulli",
    sensory_generator=None,
    policy_rng=None,
    baseline_mode="plausible",
    stdp_selected_action_gain=None,
    apply_action_mask=True,
    optimized_connections=True,
    mask_tensor=None,
    adjacency_data=None,
):
    if baseline_mode not in {"plausible", "main_reference"}:
        raise ValueError(f"Unknown baseline mode: {baseline_mode!r}")
    main_reference_stdp = baseline_mode == "main_reference"
    if main_reference_stdp:
        input_mode = "legacy_static_positive"
        input_probability = None
        input_refractory = refrac
        stdp_selected_action_gain = 1.5
    elif stdp_selected_action_gain is None:
        stdp_selected_action_gain = 1.0
    network = Network(dt=dt)
    input_layer = Input(n=NA, traces=True, thresh=thresh, rest=reset, reset=reset,
                        refrac=refrac if input_refractory is None else input_refractory)
    output_layer = LIFNodes(n=NA, traces=True, thresh=thresh * torch.ones(NA), rest=reset, reset=reset, refrac=refrac)
    inhibitor_layer = LIFNodes(n=NA, traces=True, thresh=thresh * torch.ones(NA), rest=reset, reset=reset, refrac=refrac, dt=dt,
                               enable_astrocyte=enable_astrocyte, alpha=alpha, k=k)
    if mask_tensor is None:
        mask_tensor = torch.as_tensor(weights_mask_XY, dtype=torch.float32)
    conn_XY = Connection(input_layer, output_layer, impulse_amplitude=0.5, impulse_amplitude_2=0.5,
                         impulse_length=40, impulse_shape_factor=0.9, invert=True,
                         update_rule=WeightDependentPostPre, w=weights_init_XY.clone(), nu=[10, 10],
                         wmin=wmin, wmax=wmax, weight_decay=weight_decay,
                         post_spike_weight_decay=post_spike_weight_decay,
                         clamp_initial_weights=apply_action_mask,
                         enforce_post_stdp_bounds=not main_reference_stdp,
                         apply_structural_mask_during_stdp=not main_reference_stdp)
    conn_XY.main_reference_stdp = main_reference_stdp
    if not main_reference_stdp:
        conn_XY.structural_mask = mask_tensor.bool()
        if apply_action_mask:
            conn_XY.w.data.mul_(mask_tensor)
    conn_XI = Connection(input_layer, inhibitor_layer, impulse_amplitude=0.5, impulse_amplitude_2=0.5,
                         impulse_length=40, impulse_shape_factor=0.9, invert=True, update_rule=NoOp,
                         diagonal_connection=True,
                         w=weights_init_XI, nu=[learning_rate, learning_rate], wmin=-100, wmax=wmax,
                         weight_decay=0, post_spike_weight_decay=post_spike_weight_decay)
    conn_IY = Connection(inhibitor_layer, output_layer, impulse_amplitude=0.5, impulse_amplitude_2=0.5,
                          impulse_length=40, impulse_shape_factor=0.9, invert=True, update_rule=NoOp,
                          diagonal_connection=True,
                          w=-weights_init_XI, nu=[learning_rate, learning_rate], wmin=-100, wmax=wmax,
                          weight_decay=0, post_spike_weight_decay=post_spike_weight_decay)
    network.add_layer(input_layer, 'X')
    network.add_layer(output_layer, 'Y')
    network.add_layer(inhibitor_layer, 'I')
    network.add_connection(conn_XY, 'X', 'Y')
    network.add_connection(conn_XI, 'X', 'I')
    network.add_connection(conn_IY, 'I', 'Y')
    global_monitor = SpikeCountMonitor(network, layer_names=("X", "Y", "I") if diagnostics is not None else ("Y",))
    network.add_monitor(global_monitor, 'Network')
    if adjacency_data is None:
        adjacency_data = _build_adjacency_data(weights_mask_XY)
    adjacency_lists, adjacency_tensors = adjacency_data
    start = t()
    initial_weights = conn_XY.w.detach().clone() if diagnostics is not None else None
    stdp_delta = torch.zeros_like(conn_XY.w) if diagnostics is not None else None
    input_spikes = 0
    output_spikes = 0
    inhibitory_spikes = 0
    raw_input_events = 0
    raw_input_events_available = True
    tie_decisions = 0
    decision_count = 0
    behavior_entropy_sum = 0.0
    stdp_step_norm_sum = 0.0
    current_l1 = {}
    stdp_raw_update_l1 = 0.0
    astro_active_neuron_steps = 0
    astro_neuron_steps = 0
    astro_threshold_drop_sum = 0.0
    positions = [int(current_position)]
    weights_2d = None
    for _ in range(n_steps):
        if current_position == goal:
            print("success")
            break
        adjacent_positions = adjacency_lists[current_position]
        adj_tensor = adjacency_tensors[current_position]
        sample = _sample_input_spike_train(
            current_position, NA, intensity, time_steps, dt,
            probability=input_probability, generator=sensory_generator,
            input_mode=input_mode,
        )
        inpts = {'X': sample}
        if sample.ndim == 3:
            raw_input_events += int(sample.sum().item())
        else:
            raw_input_events_available = False
        injects_v = {'I': torch.full((NA,), 0.02)}
        weights_before_stdp = conn_XY.w[current_position, adj_tensor].detach().clone()
        network_diagnostics = {} if diagnostics is not None else None
        network.run(inpts=inpts, time=time_steps, injects_v=injects_v,
                    current_position=current_position, adjacent_positions=adjacent_positions,
                    conn_XY=conn_XY, enable_stdp=enable_stdp,
                    diagnostics=network_diagnostics,
                    optimized_connections=optimized_connections)
        if network_diagnostics is not None:
            for conn_name, amount in network_diagnostics.get("synaptic_current_l1", {}).items():
                current_l1[conn_name] = current_l1.get(conn_name, 0.0) + amount
            for layer_name, amount in network_diagnostics.get("net_current_l1", {}).items():
                current_l1[f"net_{layer_name}"] = current_l1.get(f"net_{layer_name}", 0.0) + amount
            stdp_raw_update_l1 += network_diagnostics.get("stdp_raw_update_l1", 0.0)
            astro_active_neuron_steps += network_diagnostics.get("astro_active_neuron_steps", 0)
            astro_neuron_steps += network_diagnostics.get("astro_neuron_steps", 0)
            astro_threshold_drop_sum += network_diagnostics.get("astro_threshold_drop_sum", 0.0)
        recordings = network.monitors['Network'].get()
        if diagnostics is not None:
            input_spikes += int(recordings['X']['s'].sum().item())
            output_spikes += int(recordings['Y']['s'].sum().item())
            inhibitory_spikes += int(recordings['I']['s'].sum().item())
        spikes = np.asarray(recordings['Y']['s'])
        summed = np.squeeze(np.sum(spikes, axis=0))
        new_position, n_tied_candidates = _select_action_from_counts(
            summed, weights_mask_XY[current_position, :], current_position,
            rng=policy_rng,
        )
        decision_count += 1
        tie_decisions += int(n_tied_candidates > 1)
        behavior_entropy_sum += float(np.log(max(n_tied_candidates, 1)))
        selected_idx = adjacent_positions.index(new_position)
        _apply_selected_action_stdp_gain(conn_XY, weights_before_stdp[selected_idx], current_position, new_position, stdp_selected_action_gain)
        step_delta = conn_XY.w[current_position, adj_tensor].detach() - weights_before_stdp
        stdp_step_norm_sum += float(torch.linalg.vector_norm(step_delta).item())
        if stdp_delta is not None:
            stdp_delta[current_position, adj_tensor] += step_delta
        if apply_action_mask:
            conn_XY.w.data.mul_(mask_tensor)
        weights_2d = None
        current_position = int(new_position)
        positions.append(current_position)
        network.reset_()
    weights_2d = conn_XY.w.detach().cpu().numpy()
    elapsed = t() - start
    reached_goal = int(current_position == goal)
    _fill_simulation_diagnostics(
        diagnostics, positions, reached_goal, elapsed, conn_XY, mask_tensor,
        initial_weights, stdp_delta, input_spikes, output_spikes,
        inhibitory_spikes=inhibitory_spikes,
        raw_input_events=raw_input_events if raw_input_events_available else np.nan,
        tie_decisions=tie_decisions, decision_count=decision_count,
        behavior_entropy_sum=behavior_entropy_sum, current_l1=current_l1,
        stdp_step_norm_sum=stdp_step_norm_sum,
        stdp_raw_update_l1=stdp_raw_update_l1,
        astro_active_neuron_steps=astro_active_neuron_steps,
        astro_neuron_steps=astro_neuron_steps,
        astro_threshold_drop_sum=astro_threshold_drop_sum,
    )
    return positions, weights_2d, reached_goal, elapsed


def setup_and_run_simulation_td_rstdp(
    NA, weights_mask_XY, weights_init_XY, weights_init_XI, n_steps,
    current_position, goal, learning_rate, wmin, wmax, weight_decay,
    post_spike_weight_decay, reset, refrac, thresh, intensity, time_steps, dt,
    enable_astrocyte, alpha, k, critic, actor_learning_rate,
    eligibility_tau_ms, temperature, reward_goal, step_penalty,
    input_refractory=None, input_probability=None, input_mode="paper_bernoulli",
    sensory_generator=None, policy_generator=None, policy_rng=None,
    actor_weight_decay=0.0, apply_actor_update=True, apply_critic_update=True,
    policy_mode="spike_softmax", diagnostics=None, actor_step_diagnostics=None,
    decision_metadata=None, optimized_connections=True, mask_tensor=None,
    adjacency_data=None, actor_modulator="td_error", gamma=0.99,
):
    """Run the independent SNN-actor + reward/TD-modulated STDP path.

    ``EligibilityPostPre`` records only local STDP pair tags during a decision
    window. Direct ordinary-STDP weight changes and the REINFORCE rule are not
    used in this execution path. ``actor_modulator="reward"`` is an
    immediate-reward control and does not update or require a critic.
    """
    if policy_mode not in {"spike_softmax", "common_spike_argmax"}:
        raise ValueError(f"Unknown TD-RSTDP policy mode: {policy_mode!r}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if eligibility_tau_ms <= 0:
        raise ValueError("eligibility_tau_ms must be positive")
    if not 0.0 <= actor_weight_decay < 1.0:
        raise ValueError("actor_weight_decay must be in [0, 1)")
    if actor_modulator not in {"td_error", "reward"}:
        raise ValueError("actor_modulator must be 'td_error' or 'reward'")
    if actor_modulator == "td_error" and critic is None:
        raise ValueError("TD-error-modulated R-STDP requires a critic")
    if actor_modulator == "reward" and apply_critic_update:
        raise ValueError("Immediate-reward R-STDP must not update a critic")
    if critic is not None and critic.values.numel() != NA:
        raise ValueError(f"Critic has {critic.values.numel()} states; expected {NA}")

    network = Network(dt=dt)
    input_layer = Input(
        n=NA, traces=True, thresh=thresh, rest=reset, reset=reset,
        refrac=refrac if input_refractory is None else input_refractory,
    )
    output_layer = LIFNodes(
        n=NA, traces=True, thresh=thresh * torch.ones(NA),
        rest=reset, reset=reset, refrac=refrac,
    )
    inhibitor_layer = LIFNodes(
        n=NA, traces=True, thresh=thresh * torch.ones(NA),
        rest=reset, reset=reset, refrac=refrac, dt=dt,
        enable_astrocyte=enable_astrocyte, alpha=alpha, k=k,
    )
    if mask_tensor is None:
        mask_tensor = torch.as_tensor(weights_mask_XY, dtype=torch.float32)

    conn_XY = Connection(
        input_layer, output_layer, impulse_amplitude=0.5,
        impulse_amplitude_2=0.5, impulse_length=40,
        impulse_shape_factor=0.9, invert=True,
        update_rule=EligibilityPostPre, w=weights_init_XY.clone(), nu=[10, 10],
        wmin=wmin, wmax=wmax, weight_decay=0.0,
        post_spike_weight_decay=0.0, enable_rstdp_state=True,
    )
    conn_XY.structural_mask = mask_tensor.bool()
    conn_XY.w.data.mul_(mask_tensor)
    conn_XI = Connection(
        input_layer, inhibitor_layer, impulse_amplitude=0.5,
        impulse_amplitude_2=0.5, impulse_length=40,
        impulse_shape_factor=0.9, invert=True, update_rule=NoOp,
        diagonal_connection=True, w=weights_init_XI, nu=[learning_rate, learning_rate],
        wmin=-100, wmax=wmax, weight_decay=0.0,
        post_spike_weight_decay=post_spike_weight_decay,
    )
    conn_IY = Connection(
        inhibitor_layer, output_layer, impulse_amplitude=0.5,
        impulse_amplitude_2=0.5, impulse_length=40,
        impulse_shape_factor=0.9, invert=True, update_rule=NoOp,
        diagonal_connection=True, w=-weights_init_XI,
        nu=[learning_rate, learning_rate], wmin=-100, wmax=wmax,
        weight_decay=0.0, post_spike_weight_decay=post_spike_weight_decay,
    )
    network.add_layer(input_layer, "X")
    network.add_layer(output_layer, "Y")
    network.add_layer(inhibitor_layer, "I")
    network.add_connection(conn_XY, "X", "Y")
    network.add_connection(conn_XI, "X", "I")
    network.add_connection(conn_IY, "I", "Y")
    network.add_monitor(
        SpikeCountMonitor(network, layer_names=("X", "Y", "I")), "Network",
    )
    if adjacency_data is None:
        adjacency_data = _build_adjacency_data(weights_mask_XY)
    adjacency_lists, adjacency_tensors = adjacency_data

    action_window_ms = float(time_steps) * float(dt)
    eligibility_decay = math.exp(-action_window_ms / float(eligibility_tau_ms))
    initial_weights = conn_XY.w.detach().clone()
    episode_return = 0.0
    reward_discount = 1.0
    positions = [int(current_position)]
    x_spikes = y_spikes = i_spikes = raw_input_events = 0
    raw_input_events_available = True
    current_l1 = {}
    astro_active_neuron_steps = astro_neuron_steps = 0
    astro_threshold_drop_sum = 0.0
    decision_count = tie_decisions = 0
    entropy_sum = 0.0
    td_errors = []
    eligibility_drive_norms = []
    eligibility_norms = []
    actor_update_norms = []
    start_value = (
        float(critic.values[int(current_position)].item())
        if critic is not None else float("nan")
    )
    start = t()

    for action_idx in range(n_steps):
        if current_position == goal:
            break
        adjacent_positions = adjacency_lists[current_position]
        adj_tensor = adjacency_tensors[current_position]
        sample = _sample_input_spike_train(
            current_position, NA, intensity, time_steps, dt,
            probability=input_probability, generator=sensory_generator,
            input_mode=input_mode,
        )
        if sample.ndim == 3:
            raw_input_events += int(sample.sum().item())
        else:
            raw_input_events_available = False
        network_diagnostics = {} if diagnostics is not None else None
        network.run(
            inpts={"X": sample}, time=time_steps,
            injects_v={"I": torch.full((NA,), 0.02)},
            current_position=current_position,
            adjacent_positions=adjacent_positions, conn_XY=conn_XY,
            # This enables the local tag rule only. EligibilityPostPre never
            # writes W; actor weights change below, after the TD transition.
            enable_stdp=apply_actor_update, diagnostics=network_diagnostics,
            optimized_connections=optimized_connections,
        )
        recordings = network.monitors["Network"].get()
        x_count = int(recordings["X"]["s"].sum().item())
        y_count = int(recordings["Y"]["s"].sum().item())
        i_count = int(recordings["I"]["s"].sum().item())
        x_spikes += x_count
        y_spikes += y_count
        i_spikes += i_count
        if network_diagnostics is not None:
            for name, value in network_diagnostics.get("synaptic_current_l1", {}).items():
                current_l1[name] = current_l1.get(name, 0.0) + value
            for name, value in network_diagnostics.get("net_current_l1", {}).items():
                key = f"net_{name}"
                current_l1[key] = current_l1.get(key, 0.0) + value
            astro_active_neuron_steps += network_diagnostics.get("astro_active_neuron_steps", 0)
            astro_neuron_steps += network_diagnostics.get("astro_neuron_steps", 0)
            astro_threshold_drop_sum += network_diagnostics.get("astro_threshold_drop_sum", 0.0)

        all_y_counts = recordings["Y"]["s"].sum(dim=0).reshape(-1)
        spike_prefs = all_y_counts[adj_tensor].to(dtype=torch.float32)
        tied_count = int(torch.count_nonzero(spike_prefs == spike_prefs.max()).item())
        if policy_mode == "spike_softmax":
            probs = torch.softmax(spike_prefs / temperature, dim=0)
            if policy_generator is None:
                selected_idx = torch.multinomial(probs, 1).item()
            else:
                selected_idx = torch.multinomial(probs, 1, generator=policy_generator).item()
            new_position = adjacent_positions[selected_idx]
            policy_entropy = _categorical_entropy(probs)
        else:
            new_position, tied_count = _select_action_from_counts(
                all_y_counts, weights_mask_XY[current_position], current_position,
                rng=policy_rng,
            )
            selected_idx = adjacent_positions.index(new_position)
            policy_entropy = math.log(max(tied_count, 1))
        decision_count += 1
        tie_decisions += int(tied_count > 1)
        entropy_sum += policy_entropy

        reward = float(step_penalty) + (float(reward_goal) if new_position == goal else 0.0)
        done = bool(new_position == goal or action_idx == n_steps - 1)
        if actor_modulator == "td_error":
            value_state = float(critic.values[current_position].item())
            value_next = float(critic.values[new_position].item())
            td_error = critic.compute_td_error(
                current_position, reward, new_position, done,
            )
            td_error_value = float(td_error.item())
            td_target = value_state + td_error_value
            critic_update_signal = td_error
        else:
            value_state = value_next = td_target = td_error_value = float("nan")
            critic_update_signal = None
        actor_signal = select_rstdp_modulator(
            actor_modulator, reward, td_error_value,
        )

        drive = conn_XY.rstdp_drive.detach().clone()
        drive_norm = float(torch.linalg.vector_norm(drive).item())
        if apply_actor_update:
            conn_XY.update_rstdp_eligibility(eligibility_decay)
        eligibility_norm = float(torch.linalg.vector_norm(conn_XY.rstdp_eligibility).item())
        if apply_critic_update:
            critic.update(current_position, critic_update_signal)
        if apply_actor_update:
            actor_delta, actor_requested_delta = conn_XY.apply_rstdp_update(
                actor_signal, actor_learning_rate, mask_tensor,
                actor_weight_decay=actor_weight_decay,
            )
            actor_update_norm = float(torch.linalg.vector_norm(actor_delta).item())
            actor_update_norms.append(actor_update_norm)
        else:
            actor_delta = torch.zeros_like(conn_XY.w)
            actor_requested_delta = torch.zeros_like(conn_XY.w)
            actor_update_norm = 0.0
        if math.isfinite(td_error_value):
            td_errors.append(td_error_value)
        eligibility_drive_norms.append(drive_norm)
        eligibility_norms.append(eligibility_norm)
        episode_return += reward_discount * reward
        reward_discount *= float(critic.gamma if critic is not None else gamma)

        allowed_weights = conn_XY.w.detach()[mask_tensor.bool()]
        at_bounds = (
            (allowed_weights <= float(wmin) + 1e-8)
            | (allowed_weights >= float(wmax) - 1e-8)
        )
        saturation_fraction = float(at_bounds.float().mean().item()) if at_bounds.numel() else 0.0
        fraction_at_wmin = float(
            (allowed_weights <= float(wmin) + 1e-8).float().mean().item()
        ) if allowed_weights.numel() else 0.0
        fraction_at_wmax = float(
            (allowed_weights >= float(wmax) - 1e-8).float().mean().item()
        ) if allowed_weights.numel() else 0.0
        if actor_step_diagnostics is not None:
            actor_step_diagnostics.append({
                **(decision_metadata or {}),
                "decision_index": len(positions) - 1,
                "state": int(current_position),
                "next_state": int(new_position),
                "chosen_action": int(new_position),
                "reward": reward,
                "done": int(done),
                "actor_modulator": actor_modulator,
                "actor_signal": actor_signal,
                "value_state": value_state,
                "value_next": value_next,
                "td_target": td_target,
                "td_error": td_error_value,
                "abs_td_error": abs(td_error_value),
                "eligibility_drive_norm": drive_norm,
                "eligibility_norm": eligibility_norm,
                "actor_update_norm": actor_update_norm,
                "actor_requested_update_norm": float(
                    torch.linalg.vector_norm(actor_requested_delta).item()
                ),
                "w_min": float(allowed_weights.min().item()) if allowed_weights.numel() else 0.0,
                "w_max": float(allowed_weights.max().item()) if allowed_weights.numel() else 0.0,
                "weight_saturation_fraction": saturation_fraction,
                "fraction_at_wmin": fraction_at_wmin,
                "fraction_at_wmax": fraction_at_wmax,
                "policy_mode": policy_mode,
                "apply_actor_update": bool(apply_actor_update),
                "apply_critic_update": bool(apply_critic_update),
                "policy_entropy": policy_entropy,
                "candidate_positions": json.dumps(adjacent_positions),
                "candidate_y_spike_counts": json.dumps(spike_prefs.tolist()),
                "X_spikes": x_count,
                "Y_spikes": y_count,
                "I_spikes": i_count,
                "tie_count": tied_count,
                "astro_active_fraction": network_diagnostics.get("astro_active_fraction", 0.0)
                if network_diagnostics is not None else 0.0,
            })

        current_position = int(new_position)
        positions.append(current_position)
        network.reset_()

    elapsed = t() - start
    reached_goal = int(current_position == goal)
    weights_2d = conn_XY.w.detach().cpu().numpy()
    final_values = critic.values.detach() if critic is not None else None
    seen_states = {positions[0]}
    revisit_count = 0
    for visited_state in positions[1:]:
        revisit_count += int(visited_state in seen_states)
        seen_states.add(visited_state)
    backtrack_count = sum(
        positions[index] == positions[index - 2]
        for index in range(2, len(positions))
    )
    actor_critic_metrics = {
        "episode_return": float(episode_return),
        "unique_states_visited": len(seen_states),
        "revisit_count": int(revisit_count),
        "revisit_rate": float(revisit_count / decision_count) if decision_count else 0.0,
        "immediate_backtrack_rate": (
            float(backtrack_count / (len(positions) - 2)) if len(positions) > 2 else 0.0
        ),
        "mean_abs_td_error": (
            float(np.mean(np.abs(td_errors))) if td_errors
            else (float("nan") if actor_modulator == "reward" else 0.0)
        ),
        "rms_td_error": (
            float(np.sqrt(np.mean(np.square(td_errors)))) if td_errors
            else (float("nan") if actor_modulator == "reward" else 0.0)
        ),
        "actor_modulator": actor_modulator,
        "mean_eligibility_drive_norm": float(np.mean(eligibility_drive_norms)) if eligibility_drive_norms else 0.0,
        "mean_eligibility_norm": float(np.mean(eligibility_norms)) if eligibility_norms else 0.0,
        "mean_actor_update_norm": float(np.mean(actor_update_norms)) if actor_update_norms else 0.0,
        "actor_update_norm_sum": float(np.sum(actor_update_norms)),
        "critic_value_start": start_value,
        "critic_value_min": float(final_values.min().item()) if final_values is not None else float("nan"),
        "critic_value_max": float(final_values.max().item()) if final_values is not None else float("nan"),
        "weight_saturation_fraction": float((
            (conn_XY.w.detach()[mask_tensor.bool()] <= float(wmin) + 1e-8)
            | (conn_XY.w.detach()[mask_tensor.bool()] >= float(wmax) - 1e-8)
        ).float().mean().item()) if bool(mask_tensor.bool().any()) else 0.0,
        "fraction_at_wmin": float((
            conn_XY.w.detach()[mask_tensor.bool()] <= float(wmin) + 1e-8
        ).float().mean().item()) if bool(mask_tensor.bool().any()) else 0.0,
        "fraction_at_wmax": float((
            conn_XY.w.detach()[mask_tensor.bool()] >= float(wmax) - 1e-8
        ).float().mean().item()) if bool(mask_tensor.bool().any()) else 0.0,
        "policy_mode": policy_mode,
        "eligibility_decay_per_action": eligibility_decay,
    }
    _fill_simulation_diagnostics(
        diagnostics, positions, reached_goal, elapsed, conn_XY, mask_tensor,
        initial_weights, torch.zeros_like(conn_XY.w), x_spikes, y_spikes,
        inhibitory_spikes=i_spikes,
        raw_input_events=raw_input_events if raw_input_events_available else np.nan,
        tie_decisions=tie_decisions, decision_count=decision_count,
        behavior_entropy_sum=entropy_sum, current_l1=current_l1,
        astro_active_neuron_steps=astro_active_neuron_steps,
        astro_neuron_steps=astro_neuron_steps,
        astro_threshold_drop_sum=astro_threshold_drop_sum,
        reward_return=episode_return,
        actor_critic_metrics=actor_critic_metrics,
    )
    return positions, weights_2d, reached_goal, elapsed, episode_return


def setup_and_run_simulation_reinforce(
    NA, weights_mask_XY, weights_init_XY, weights_init_XI, n_steps,
    current_position, goal, learning_rate, wmin, wmax, weight_decay,
    post_spike_weight_decay, reset, refrac, thresh, intensity, time_steps, dt,
    enable_astrocyte, alpha, k,
    reinforce_lr, temperature, reward_goal, step_penalty,
    gamma, baseline_decay, policy_mix_beta, trace_decay,
    enable_stdp_during_training,
    surrogate_kind="lif", surrogate_params=None,
    running_baseline=0.0,
    diagnostics=None,
    input_refractory=None,
    input_probability=None,
    input_mode="bernoulli",
    sensory_generator=None,
    policy_generator=None,
    decision_diagnostics=None,
    decision_metadata=None,
    baseline_mode="plausible",
    stdp_selected_action_gain=None,
    apply_reinforce_update=True,
    optimized_connections=True,
    mask_tensor=None,
    adjacency_data=None,
):
    if baseline_mode not in {"plausible", "main_reference"}:
        raise ValueError(f"Unknown baseline mode: {baseline_mode!r}")
    main_reference_stdp = baseline_mode == "main_reference"
    if main_reference_stdp:
        input_mode = "legacy_static_positive"
        input_probability = None
        input_refractory = refrac
        stdp_selected_action_gain = 1.5
    elif stdp_selected_action_gain is None:
        stdp_selected_action_gain = 1.0
    network = Network(dt=dt)
    input_layer = Input(n=NA, traces=True, thresh=thresh, rest=reset, reset=reset,
                        refrac=refrac if input_refractory is None else input_refractory)
    output_layer = LIFNodes(n=NA, traces=True, thresh=thresh * torch.ones(NA), rest=reset, reset=reset, refrac=refrac)
    inhibitor_layer = LIFNodes(n=NA, traces=True, thresh=thresh * torch.ones(NA), rest=reset, reset=reset, refrac=refrac, dt=dt,
                               enable_astrocyte=enable_astrocyte, alpha=alpha, k=k)
    if mask_tensor is None:
        mask_tensor = torch.as_tensor(weights_mask_XY, dtype=torch.float32)
    conn_XY = Connection(input_layer, output_layer, impulse_amplitude=0.5, impulse_amplitude_2=0.5,
                         impulse_length=40, impulse_shape_factor=0.9, invert=True,
                         update_rule=WeightDependentPostPre, w=weights_init_XY.clone(), nu=[10, 10],
                         wmin=wmin, wmax=wmax, weight_decay=weight_decay, post_spike_weight_decay=post_spike_weight_decay,
                         baseline_decay=baseline_decay, policy_mix_beta=policy_mix_beta,
                         gamma=gamma, trace_decay=trace_decay, temperature=temperature,
                         clamp_initial_weights=apply_reinforce_update,
                         enforce_post_stdp_bounds=not main_reference_stdp,
                         apply_structural_mask_during_stdp=not main_reference_stdp)
    conn_XY.main_reference_stdp = main_reference_stdp
    if not main_reference_stdp:
        conn_XY.structural_mask = mask_tensor.bool()
        if apply_reinforce_update:
            conn_XY.w.data.mul_(mask_tensor)
    conn_XI = Connection(input_layer, inhibitor_layer, impulse_amplitude=0.5, impulse_amplitude_2=0.5,
                         impulse_length=40, impulse_shape_factor=0.9, invert=True, update_rule=NoOp,
                         w=weights_init_XI, nu=[learning_rate, learning_rate], wmin=-100, wmax=wmax,
                         diagonal_connection=True,
                         weight_decay=0, post_spike_weight_decay=post_spike_weight_decay)
    conn_IY = Connection(inhibitor_layer, output_layer, impulse_amplitude=0.5, impulse_amplitude_2=0.5,
                          impulse_length=40, impulse_shape_factor=0.9, invert=True, update_rule=NoOp,
                          w=-weights_init_XI, nu=[learning_rate, learning_rate], wmin=-100, wmax=wmax,
                          diagonal_connection=True,
                          weight_decay=0, post_spike_weight_decay=post_spike_weight_decay)
    network.add_layer(input_layer, 'X')
    network.add_layer(output_layer, 'Y')
    network.add_layer(inhibitor_layer, 'I')
    network.add_connection(conn_XY, 'X', 'Y')
    network.add_connection(conn_XI, 'X', 'I')
    network.add_connection(conn_IY, 'I', 'Y')
    global_monitor = SpikeCountMonitor(network, layer_names=("X", "Y", "I") if diagnostics is not None or decision_diagnostics is not None else ("Y",))
    network.add_monitor(global_monitor, 'Network')
    if adjacency_data is None:
        adjacency_data = _build_adjacency_data(weights_mask_XY)
    adjacency_lists, adjacency_tensors = adjacency_data

    conn_XY.running_baseline = running_baseline
    use_surrogate = surrogate_kind != "spike"
    effective_intensity = (
        input_probability * int(time_steps / dt)
        if input_probability is not None else intensity
    )
    diagnostic_c = None
    surrogate_c = None
    if use_surrogate and apply_reinforce_update:
        surrogate_c = compute_scale_c(
            conn_XY.impulse_amplitude, conn_XY.impulse_length,
            conn_XY.impulse_shape_factor, conn_XY.invert,
            effective_intensity, time_steps,
        )
        diagnostic_c = surrogate_c
        if surrogate_params is None:
            surrogate_params = build_lif_params(thresh, dt, float(output_layer.tc_decay), refrac)
    elif decision_diagnostics is not None:
        diagnostic_c = compute_scale_c(
            conn_XY.impulse_amplitude, conn_XY.impulse_length,
            conn_XY.impulse_shape_factor, conn_XY.invert,
            effective_intensity, time_steps,
        )
        if use_surrogate and surrogate_params is None:
            surrogate_params = build_lif_params(thresh, dt, float(output_layer.tc_decay), refrac)

    start = t()
    initial_weights = conn_XY.w.detach().clone() if diagnostics is not None else None
    stdp_delta = torch.zeros_like(conn_XY.w) if diagnostics is not None else None
    stdp_delta_edges = (
        {} if diagnostics is None and decision_diagnostics is not None else None
    )
    input_spikes = 0
    output_spikes = 0
    inhibitory_spikes = 0
    raw_input_events = 0
    raw_input_events_available = True
    tie_decisions = 0
    decision_count = 0
    behavior_entropy_sum = 0.0
    stdp_step_norm_sum = 0.0
    current_l1 = {}
    stdp_raw_update_l1 = 0.0
    astro_active_neuron_steps = 0
    astro_neuron_steps = 0
    astro_threshold_drop_sum = 0.0
    positions = [int(current_position)]
    first_decision_row = len(decision_diagnostics) if decision_diagnostics is not None else 0
    for _ in range(n_steps):
        if current_position == goal:
            print("success")
            break
        adjacent_positions = adjacency_lists[current_position]
        adj_tensor = adjacency_tensors[current_position]
        sample = _sample_input_spike_train(
            current_position, NA, intensity, time_steps, dt,
            probability=input_probability, generator=sensory_generator,
            input_mode=input_mode,
        )
        inpts = {'X': sample}
        if sample.ndim == 3:
            raw_input_events += int(sample.sum().item())
        else:
            raw_input_events_available = False
        injects_v = {'I': torch.full((NA,), 0.02)}
        weights_before_stdp = conn_XY.w[current_position, adj_tensor].detach().clone()
        network_diagnostics = {} if diagnostics is not None else None
        network.run(inpts=inpts, time=time_steps, injects_v=injects_v,
                    current_position=current_position, adjacent_positions=adjacent_positions,
                    conn_XY=conn_XY, enable_stdp=enable_stdp_during_training,
                    diagnostics=network_diagnostics,
                    optimized_connections=optimized_connections)
        if network_diagnostics is not None:
            for conn_name, amount in network_diagnostics.get("synaptic_current_l1", {}).items():
                current_l1[conn_name] = current_l1.get(conn_name, 0.0) + amount
            for layer_name, amount in network_diagnostics.get("net_current_l1", {}).items():
                current_l1[f"net_{layer_name}"] = current_l1.get(f"net_{layer_name}", 0.0) + amount
            stdp_raw_update_l1 += network_diagnostics.get("stdp_raw_update_l1", 0.0)
            astro_active_neuron_steps += network_diagnostics.get("astro_active_neuron_steps", 0)
            astro_neuron_steps += network_diagnostics.get("astro_neuron_steps", 0)
            astro_threshold_drop_sum += network_diagnostics.get("astro_threshold_drop_sum", 0.0)
        recordings = network.monitors['Network'].get()
        if diagnostics is not None:
            input_spikes += int(recordings['X']['s'].sum().item())
            output_spikes += int(recordings['Y']['s'].sum().item())
            inhibitory_spikes += int(recordings['I']['s'].sum().item())
        spike_counts_by_target = recordings['Y']['s'].sum(dim=0).squeeze(0)
        summed = spike_counts_by_target
        w = network.connections['X_Y'].w.detach()
        weight_prefs = w[current_position, adj_tensor]
        spike_prefs = summed[adj_tensor].to(dtype=torch.float32)
        logits = policy_mix_beta * weight_prefs + (1 - policy_mix_beta) * spike_prefs
        probs = torch.softmax(logits / temperature, dim=0)
        probs = probs / probs.sum()
        if policy_generator is None:
            dist = torch.distributions.Categorical(probs)
            chosen_idx = dist.sample().item()
        else:
            chosen_idx = torch.multinomial(probs, 1, generator=policy_generator).item()
        new_position = adjacent_positions[chosen_idx]
        decision_count += 1
        behavior_entropy_sum += _categorical_entropy(probs)
        spike_ties = int(torch.count_nonzero(spike_prefs == spike_prefs.max()).item() > 1)
        tie_decisions += spike_ties

        step_reward = step_penalty
        if new_position == goal:
            step_reward += reward_goal
        conn_XY.accumulate_return(step_reward)

        if apply_reinforce_update or decision_diagnostics is not None:
            if use_surrogate:
                w_slice = network.connections['X_Y'].w.data[current_position, adj_tensor]
                c_for_gradient = surrogate_c if surrogate_c is not None else diagnostic_c
                I_eff = c_for_gradient * w_slice
                grad_slice = eligibility_gradient(adjacent_positions, chosen_idx, I_eff, c_for_gradient,
                                                  temperature, surrogate_kind, surrogate_params)
            else:
                grad_slice = spike_eligibility(
                    probs, spike_prefs, chosen_idx, time_steps, temperature
                )
        else:
            grad_slice = None
        _apply_selected_action_stdp_gain(conn_XY, weights_before_stdp[chosen_idx], current_position, new_position, stdp_selected_action_gain)
        step_stdp_delta = conn_XY.w[current_position, adj_tensor].detach() - weights_before_stdp
        stdp_step_norm_sum += float(torch.linalg.vector_norm(step_stdp_delta).item())
        if stdp_delta is not None:
            stdp_delta[current_position, adj_tensor] += step_stdp_delta
        elif stdp_delta_edges is not None:
            for edge_idx, target_position in enumerate(adjacent_positions):
                edge = (int(current_position), int(target_position))
                stdp_delta_edges[edge] = (
                    stdp_delta_edges.get(edge, 0.0) + float(step_stdp_delta[edge_idx].item())
                )
        if decision_diagnostics is not None:
            diagnostic_w = conn_XY.w.detach()[current_position, adj_tensor]
            diagnostic_I = diagnostic_c * diagnostic_w
            (probs_surrogate, kl, js, grad_norm, spike_grad_norm, cosine) = _compute_policy_diagnostics(probs, spike_prefs, chosen_idx, diagnostic_I, temperature, surrogate_params, thresh, dt, float(output_layer.tc_decay), refrac, grad_slice, time_steps)
        if decision_diagnostics is not None:
            decision_diagnostics.append({
                **(decision_metadata or {}),
                "decision_index": len(positions) - 1,
                "position": int(current_position),
                "candidate_positions": json.dumps(adjacent_positions),
                "chosen_position": int(new_position),
                "candidate_spike_counts": json.dumps(spike_prefs.tolist()),
                "pi_behavior": json.dumps(probs.tolist()),
                "pi_surrogate": json.dumps(probs_surrogate.tolist()),
                "entropy_behavior_nats": _categorical_entropy(probs),
                "entropy_surrogate_nats": _categorical_entropy(probs_surrogate),
                "kl_behavior_to_surrogate": kl,
                "js_behavior_surrogate": js,
                "eligibility_norm": grad_norm,
                "spike_eligibility_norm": spike_grad_norm,
                "eligibility_cosine_surrogate_vs_spike": cosine,
                "delta_w_stdp_norm_step": float(torch.linalg.vector_norm(step_stdp_delta).item()),
                "delta_w_rl_norm_episode": np.nan,
                "input_spikes_episode_so_far": input_spikes,
                "output_spikes_episode_so_far": output_spikes,
            })
        if apply_reinforce_update:
            conn_XY.accumulate_trace(current_position, adjacent_positions, grad_slice)
            conn_XY.w.data.mul_(mask_tensor)
        current_position = int(new_position)
        positions.append(current_position)
        network.reset_()

    elapsed = t() - start
    reached_goal = (current_position == goal)

    reward_return = conn_XY.reward_accumulator
    baseline_before = conn_XY.running_baseline
    record_update_diagnostics = diagnostics is not None or decision_diagnostics is not None
    weights_before_reinforce = conn_XY.w.detach().clone() if record_update_diagnostics else None
    reinforce_preclip = None
    update_interaction = None
    if apply_reinforce_update:
        if diagnostics is not None:
            eligibility_snapshot = conn_XY.eligibility.detach()
            reinforce_preclip = reinforce_lr * (reward_return - baseline_before) * eligibility_snapshot
            stdp_flat = stdp_delta.reshape(-1)
            rl_flat = reinforce_preclip.reshape(-1)
            overlap = (stdp_flat != 0) & (rl_flat != 0)
            if bool(overlap.any()):
                opposite = (torch.sign(stdp_flat[overlap]) != torch.sign(rl_flat[overlap])).float().mean()
                cosine = torch.nn.functional.cosine_similarity(stdp_flat.reshape(1, -1), rl_flat.reshape(1, -1), dim=1)[0]
                update_interaction = {"stdp_rl_update_cosine": float(cosine.item()),
                                      "stdp_rl_opposite_sign_fraction": float(opposite.item())}
            else:
                update_interaction = {"stdp_rl_update_cosine": np.nan,
                                      "stdp_rl_opposite_sign_fraction": np.nan}
        conn_XY.compute_and_apply_reinforce_update(lr=reinforce_lr, mask=mask_tensor)
        reinforce_delta = conn_XY.w.detach() - weights_before_reinforce if record_update_diagnostics else None
    else:
        reinforce_delta = torch.zeros_like(conn_XY.w) if record_update_diagnostics else None
    if decision_diagnostics is not None:
        rl_delta_norm = float(torch.linalg.vector_norm(reinforce_delta).item())
        if stdp_delta is not None:
            stdp_delta_norm = float(torch.linalg.vector_norm(stdp_delta).item())
        else:
            stdp_delta_norm = float(np.sqrt(sum(value * value for value in stdp_delta_edges.values())))
        for row in decision_diagnostics[first_decision_row:]:
            row["delta_w_rl_norm_episode"] = rl_delta_norm
            row["delta_w_stdp_norm_episode"] = stdp_delta_norm

    weights_2d = network.connections['X_Y'].w.detach().cpu().numpy()
    new_baseline = conn_XY.running_baseline
    _fill_simulation_diagnostics(
        diagnostics, positions, int(reached_goal), elapsed, conn_XY, mask_tensor,
        initial_weights, stdp_delta, input_spikes, output_spikes,
        inhibitory_spikes=inhibitory_spikes,
        raw_input_events=raw_input_events if raw_input_events_available else np.nan,
        tie_decisions=tie_decisions, decision_count=decision_count,
        behavior_entropy_sum=behavior_entropy_sum, current_l1=current_l1,
        stdp_step_norm_sum=stdp_step_norm_sum,
        stdp_raw_update_l1=stdp_raw_update_l1,
        astro_active_neuron_steps=astro_active_neuron_steps,
        astro_neuron_steps=astro_neuron_steps,
        astro_threshold_drop_sum=astro_threshold_drop_sum,
        reinforce_delta=reinforce_delta, reinforce_preclip=reinforce_preclip,
        reward_return=reward_return, baseline_before=baseline_before,
        baseline_after=new_baseline, update_interaction=update_interaction,
    )
    return positions, weights_2d, int(reached_goal), elapsed, new_baseline


def run_experiment(config: dict, logger: ExperimentLogger = None):
    _configure_runtime(config)
    if logger is None:
        logger = FileLogger(output_dir=config.get("output_dir", "results"))

    N = config["grid_size"]
    NA = N * N
    weights_mask_XY = create_adjacency_matrix(N)
    mask_tensor = torch.as_tensor(weights_mask_XY, dtype=torch.float32)
    adjacency_data = _build_adjacency_data(weights_mask_XY)
    simulation_precompute_args = {
        "mask_tensor": mask_tensor,
        "adjacency_data": adjacency_data,
    }
    n_steps = config["n_steps"]
    current_position = config["start_position"]
    goal = config["goal_position"]

    neuron_cfg = config.get("neuron", {})
    learning_rate = neuron_cfg.get("learning_rate", config.get("learning_rate", 1))
    wmin = neuron_cfg.get("wmin", config.get("wmin", 0.001))
    wmax = neuron_cfg.get("wmax", config.get("wmax", 1))
    weight_decay = neuron_cfg.get("weight_decay", config.get("weight_decay", 0))
    post_spike_weight_decay = neuron_cfg.get("post_spike_weight_decay", config.get("post_spike_weight_decay", 0.005))
    reset = neuron_cfg.get("reset", config.get("reset", 0))
    refrac = neuron_cfg.get("refrac", config.get("refrac", 40))
    thresh = neuron_cfg.get("thresh", config.get("thresh", 7))

    astro_cfg = config.get("astrocyte", {})
    enable_astrocyte = astro_cfg.get("enable", config.get("enable_astrocyte", True))
    alpha = astro_cfg.get("alpha", config.get("alpha", 0.001))
    k = astro_cfg.get("k", config.get("k", 0.2))

    sim_cfg = config.get("simulation", {})
    intensity = sim_cfg.get("intensity", config.get("intensity", 15.0))
    time_steps = sim_cfg.get("time_steps", config.get("time_steps", 1000))
    dt = sim_cfg.get("dt", config.get("dt", 1))
    input_probability = sim_cfg.get("input_probability")
    input_refractory = sim_cfg.get("input_refractory")
    input_mode = sim_cfg.get("input_mode", "bernoulli")

    reinf_cfg = config.get("reinforce", {})
    reinf_enable = reinf_cfg.get("enable", False)
    reinf_lr = reinf_cfg.get("learning_rate", 0.01)
    temperature = reinf_cfg.get("temperature", 1.0)
    reward_goal = reinf_cfg.get("reward_goal", 10.0)
    step_penalty = reinf_cfg.get("step_penalty", -0.1)
    gamma = reinf_cfg.get("gamma", 0.99)
    baseline_decay = reinf_cfg.get("baseline_decay", 0.01)
    policy_mix_beta = reinf_cfg.get("policy_mix_beta", 0.0)
    trace_decay = reinf_cfg.get("trace_decay", 0.95)

    actor_critic_cfg = config.get("actor_critic", {})
    actor_critic_enable = bool(actor_critic_cfg.get("enable", False))
    actor_learning_rate = float(actor_critic_cfg.get("actor_learning_rate", 0.001))
    critic_learning_rate = float(actor_critic_cfg.get("critic_learning_rate", 0.1))
    actor_critic_gamma = float(actor_critic_cfg.get("gamma", 0.99))
    eligibility_tau_ms = float(actor_critic_cfg.get("eligibility_tau_ms", 1000.0))
    actor_temperature = float(actor_critic_cfg.get("temperature", 1.0))
    actor_weight_decay = float(actor_critic_cfg.get("actor_weight_decay", 0.0))
    actor_policy = actor_critic_cfg.get("policy", "spike_softmax")
    actor_rule = actor_critic_cfg.get("actor_rule", "td_rstdp")
    critic_kind = actor_critic_cfg.get("critic", "tabular_td0")

    learning_cfg = config.get("learning", {})
    baseline_mode = config.get("baseline_mode", learning_cfg.get("baseline_mode", "plausible"))
    stdp_selected_action_gain = float(learning_cfg.get(
        "main_selected_action_stdp_gain",
        1.5 if baseline_mode == "main_reference" else 1.0,
    ))
    if baseline_mode == "main_reference":
        input_mode, input_probability, input_refractory = "legacy_static_positive", None, refrac
        stdp_selected_action_gain = 1.5
    enable_stdp_during_training = _resolve_enable_stdp(learning_cfg, reinf_cfg)
    if actor_critic_enable:
        if reinf_enable:
            raise ValueError("TD-RSTDP and REINFORCE cannot be enabled together")
        if enable_stdp_during_training:
            raise ValueError(
                "TD-RSTDP requires learning.enable_stdp=false; its local "
                "EligibilityPostPre rule is not ordinary direct-weight STDP"
            )
        if baseline_mode != "plausible":
            raise ValueError("TD-RSTDP currently supports baseline_mode='plausible' only")
        if actor_rule == "td_rstdp" and critic_kind != "tabular_td0":
            raise ValueError("actor_rule=td_rstdp requires critic=tabular_td0")
        if actor_rule == "reward_rstdp" and critic_kind not in {"none", None}:
            raise ValueError("actor_rule=reward_rstdp requires critic=none")
        if actor_rule not in {"td_rstdp", "reward_rstdp"}:
            raise ValueError("actor_rule must be td_rstdp or reward_rstdp")
        if actor_policy != "spike_softmax":
            raise ValueError("TD-RSTDP training currently requires policy=spike_softmax")
        if policy_mix_beta != 0.0:
            raise ValueError("TD-RSTDP uses Y spike counts only; policy_mix_beta must be 0")
        if actor_learning_rate < 0 or critic_learning_rate < 0:
            raise ValueError("TD-RSTDP learning rates must be non-negative")
        if eligibility_tau_ms <= 0 or actor_temperature <= 0:
            raise ValueError("eligibility_tau_ms and actor temperature must be positive")
        if not 0.0 <= actor_critic_gamma <= 1.0:
            raise ValueError("actor_critic.gamma must be in [0, 1]")
        if not 0.0 <= actor_weight_decay < 1.0:
            raise ValueError("actor_weight_decay must be in [0, 1)")
    protocol_cfg = config.get("protocol", {})
    initial_baseline_stdp = bool(protocol_cfg.get(
        "initial_baseline_stdp", baseline_mode == "main_reference",
    ))
    legacy_verification_stdp = bool(protocol_cfg.get("legacy_verification_stdp", False))
    run_legacy_verification = bool(protocol_cfg.get("run_legacy_verification", True))
    verification_mode = protocol_cfg.get("verification_mode")
    if verification_mode is None:
        if not run_legacy_verification:
            verification_mode = "none"
        elif baseline_mode == "main_reference" or legacy_verification_stdp:
            verification_mode = "source_verification"
        else:
            verification_mode = "frozen_evaluation"
    if baseline_mode not in {"plausible", "main_reference"}:
        raise ValueError(f"Unknown baseline mode: {baseline_mode!r}")

    if verification_mode not in {"source_verification", "frozen_evaluation", "none"}:
        raise ValueError(f"Unknown verification mode: {verification_mode!r}")
    surr_cfg = reinf_cfg.get("surrogate", {})
    surrogate_kind = surr_cfg.get("type", "lif")
    if surrogate_kind == "lif":
        decay = float(np.exp(-dt / surr_cfg.get("tc_decay", 150.0)))
        surrogate_params = {
            "thresh": thresh,
            "decay": surr_cfg.get("decay", decay),
            "tc_decay": surr_cfg.get("tc_decay", 150.0),
            "refrac": refrac,
        }
    elif surrogate_kind == "softplus":
        I_theta_default = thresh * (1.0 - float(np.exp(-dt / 150.0)))
        surrogate_params = {
            "I_theta": surr_cfg.get("I_theta", I_theta_default),
            "scale": surr_cfg.get("scale", 1.0),
        }
    elif surrogate_kind == "nmda":
        I_theta_default = thresh * (1.0 - float(np.exp(-dt / 150.0)))
        surrogate_params = {
            "I_half": surr_cfg.get("I_half", I_theta_default),
            "k": surr_cfg.get("k", 0.5 * I_theta_default),
            "ca_baseline": surr_cfg.get("ca_baseline", 0.5),
            "n_hill": surr_cfg.get("n_hill", 1.0),
        }
    elif surrogate_kind == "spike":
        surrogate_params = None
    else:
        raise ValueError(f"Unknown surrogate type: {surrogate_kind!r}")

    exp_cfg = config.get("experiment", {})
    num_cycles = exp_cfg.get("num_cycles", config.get("num_cycles", 20))
    num_experiments = exp_cfg.get("num_experiments", config.get("num_experiments", 5))
    seeds = exp_cfg.get("seeds", [])
    if seeds and len(seeds) < num_experiments:
        raise ValueError(
            f"Configured {len(seeds)} seeds for {num_experiments} experiments; "
            "provide one seed per requested experiment"
        )

    all_experiment_results = []
    diagnostic_cfg = config.get("diagnostics", {})
    diagnostics_enabled = bool(diagnostic_cfg.get("enable", False))
    all_episode_diagnostics = []
    all_evaluation_rollouts = []
    all_policy_diagnostics = []
    all_actor_critic_steps = []
    evaluation_rollouts = int(diagnostic_cfg.get("evaluation_rollouts", 20))
    evaluation_checkpoints = set(diagnostic_cfg.get("evaluation_checkpoints", [num_cycles]))
    evaluation_astrocyte_modes = diagnostic_cfg.get("evaluation_astrocyte_modes", [False, True])

    for exp_idx in range(num_experiments):
        experiment_num = exp_idx + 1
        logger.start_experiment(f"experiment_{experiment_num}")
        print(f"\nEXPERIMENT {experiment_num}/{num_experiments}")
        print("-" * 20)

        if seeds and exp_idx < len(seeds):
            seed = seeds[exp_idx]
        else:
            seed = random.randint(0, 2**31 - 1)

        config['seed'] = seed
        logger.log_params({
            **config,
            "baseline_mode": baseline_mode,
            "input_mode": input_mode,
            "enable_stdp": enable_stdp_during_training,
            "enable_reinforce": reinf_enable,
            "apply_reinforce_update": reinf_enable,
            "enforce_post_stdp_bounds": baseline_mode != "main_reference",
            "chosen_stdp_gain": stdp_selected_action_gain,
            "verification_mode": verification_mode,
            "enable_actor_critic": actor_critic_enable,
            "actor_rule": actor_rule if actor_critic_enable else None,
            "critic": critic_kind if actor_critic_enable else None,
            "actor_learning_rate": actor_learning_rate if actor_critic_enable else None,
            "critic_learning_rate": critic_learning_rate if actor_critic_enable else None,
            "actor_critic_gamma": actor_critic_gamma if actor_critic_enable else None,
            "eligibility_tau_ms": eligibility_tau_ms if actor_critic_enable else None,
            "actor_policy": actor_policy if actor_critic_enable else None,
        })
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        rngs = make_rng_streams(seed)
        td_critic = (
            TabularTDCritic(
                n_states=NA, learning_rate=critic_learning_rate,
                gamma=actor_critic_gamma,
                init_value=float(actor_critic_cfg.get("init_value", 0.0)),
            )
            if actor_critic_enable and actor_rule == "td_rstdp" else None
        )
        training_condition = (
            actor_rule if actor_critic_enable
            else _condition_label(enable_stdp_during_training, reinf_enable)
        )

        weights_rand_dist_XY = rngs["weights"].normal(0.65, 0.1, size=(NA, NA))
        weights_rand_dist_XY[weights_rand_dist_XY > 0.8] = 0.8
        weights_rand_dist_XY[weights_rand_dist_XY < 0.5] = 0.5
        weights_init_XY = torch.Tensor(weights_mask_XY * weights_rand_dist_XY).float()
        weights_init_XI = torch.eye(NA)

        all_routes = []
        all_weight_matrices = []
        route_lengths = []

        QAZ_initial = weights_rand_dist_XY * weights_mask_XY
        all_weight_matrices.append(QAZ_initial.copy())

        print(f"\n{'=' * 60}")
        print("INITIAL RUN (BASELINE)")
        print(f"{'=' * 60}")

        baseline_diag = {} if diagnostics_enabled else None
        positions, weights_2d, goal_reached, elapsed_time = setup_and_run_simulation(
            NA=NA, weights_mask_XY=weights_mask_XY, weights_init_XY=weights_init_XY,
            weights_init_XI=weights_init_XI, n_steps=n_steps, current_position=current_position,
            goal=goal, learning_rate=learning_rate, wmin=wmin, wmax=wmax,
            weight_decay=weight_decay, post_spike_weight_decay=post_spike_weight_decay,
            reset=reset, refrac=refrac, thresh=thresh, intensity=intensity,
            time_steps=time_steps, dt=dt, enable_astrocyte=False, alpha=alpha, k=k,
            enable_stdp=initial_baseline_stdp,
            diagnostics=baseline_diag, input_refractory=input_refractory,
            input_probability=input_probability, input_mode=input_mode,
            sensory_generator=rngs["eval_sensory"], policy_rng=rngs["eval_action_numpy"],
            baseline_mode=baseline_mode,
            stdp_selected_action_gain=stdp_selected_action_gain,
            **simulation_precompute_args,
        )
        if diagnostics_enabled:
            all_episode_diagnostics.append({
                "seed": seed, "experiment_num": experiment_num,
                "phase": "initial_baseline", "cycle": 0,
                "input_probability": input_probability, "input_mode": input_mode,
                "input_refractory": input_refractory if input_refractory is not None else refrac,
                **baseline_diag,
            })
        all_routes.append(f"Route (initial baseline): {positions}")
        route_lengths.append(len(positions) - 1)

        logger.log_metrics({
            "initial/route_length": len(positions) - 1,
            "initial/goal_reached": goal_reached,
            "initial/elapsed_time": elapsed_time
        }, step=0)

        running_baseline = 0.0

        for cycle_idx in range(num_cycles):
            cycle_num = cycle_idx + 1

            print(f"\n{'=' * 60}")
            print(f"TRAINING CYCLE {cycle_num}/{num_cycles}")
            print(f"{'=' * 60}")

            train_diag = {} if diagnostics_enabled else None
            training_episode_return = np.nan
            if actor_critic_enable:
                (positions_train, weights_2d_train, goal_reached_train,
                 elapsed_time_train, training_episode_return) = setup_and_run_simulation_td_rstdp(
                    NA=NA, weights_mask_XY=weights_mask_XY,
                    weights_init_XY=weights_init_XY,
                    weights_init_XI=weights_init_XI, n_steps=n_steps,
                    current_position=current_position, goal=goal,
                    learning_rate=learning_rate, wmin=wmin, wmax=wmax,
                    weight_decay=weight_decay,
                    post_spike_weight_decay=post_spike_weight_decay,
                    reset=reset, refrac=refrac, thresh=thresh,
                    intensity=intensity, time_steps=time_steps, dt=dt,
                    enable_astrocyte=enable_astrocyte, alpha=alpha, k=k,
                    critic=td_critic, actor_learning_rate=actor_learning_rate,
                    eligibility_tau_ms=eligibility_tau_ms,
                    temperature=actor_temperature, reward_goal=reward_goal,
                    step_penalty=step_penalty,
                    input_refractory=input_refractory,
                    input_probability=input_probability, input_mode=input_mode,
                    sensory_generator=rngs["sensory"],
                    policy_generator=rngs["action_torch"],
                    actor_weight_decay=actor_weight_decay,
                    actor_modulator=(
                        "reward" if actor_rule == "reward_rstdp" else "td_error"
                    ),
                    gamma=actor_critic_gamma,
                    apply_critic_update=(actor_rule == "td_rstdp"),
                    policy_mode="spike_softmax", diagnostics=train_diag,
                    actor_step_diagnostics=(
                        all_actor_critic_steps if diagnostics_enabled else None
                    ),
                    decision_metadata={
                        "seed": seed, "experiment_num": experiment_num,
                        "phase": "training", "cycle": cycle_num,
                        "condition": training_condition,
                        "astrocyte_enabled": enable_astrocyte,
                    },
                    **simulation_precompute_args,
                )
            elif reinf_enable:
                positions_train, weights_2d_train, goal_reached_train, elapsed_time_train, running_baseline = setup_and_run_simulation_reinforce(
                    NA=NA, weights_mask_XY=weights_mask_XY, weights_init_XY=weights_init_XY,
                    weights_init_XI=weights_init_XI, n_steps=n_steps, current_position=current_position,
                    goal=goal, learning_rate=learning_rate, wmin=wmin, wmax=wmax,
                    weight_decay=weight_decay, post_spike_weight_decay=post_spike_weight_decay,
                    reset=reset, refrac=refrac, thresh=thresh, intensity=intensity,
                    time_steps=time_steps, dt=dt, enable_astrocyte=enable_astrocyte, alpha=alpha, k=k,
                    reinforce_lr=reinf_lr, temperature=temperature,
                    reward_goal=reward_goal, step_penalty=step_penalty,
                    gamma=gamma, baseline_decay=baseline_decay,
                    policy_mix_beta=policy_mix_beta, trace_decay=trace_decay,
                    enable_stdp_during_training=enable_stdp_during_training,
                    surrogate_kind=surrogate_kind, surrogate_params=surrogate_params,
                    running_baseline=running_baseline,
                    diagnostics=train_diag,
                    input_refractory=input_refractory,
                    input_probability=input_probability, input_mode=input_mode,
                    sensory_generator=rngs["sensory"],
                    policy_generator=rngs["action_torch"],
                    decision_diagnostics=all_policy_diagnostics if diagnostics_enabled else None,
                    decision_metadata={"seed": seed, "experiment_num": experiment_num,
                                       "phase": "training", "cycle": cycle_num,
                                       "input_probability": input_probability,
                                       "input_mode": input_mode,
                                       "input_refractory": input_refractory if input_refractory is not None else refrac,
                                       "surrogate_kind": surrogate_kind},
                    baseline_mode=baseline_mode,
                    stdp_selected_action_gain=stdp_selected_action_gain,
                    **simulation_precompute_args,
                )
            else:
                positions_train, weights_2d_train, goal_reached_train, elapsed_time_train = setup_and_run_simulation(
                    NA=NA, weights_mask_XY=weights_mask_XY, weights_init_XY=weights_init_XY,
                    weights_init_XI=weights_init_XI, n_steps=n_steps, current_position=current_position,
                    goal=goal, learning_rate=learning_rate, wmin=wmin, wmax=wmax,
                    weight_decay=weight_decay, post_spike_weight_decay=post_spike_weight_decay,
                    reset=reset, refrac=refrac, thresh=thresh, intensity=intensity,
                    time_steps=time_steps, dt=dt, enable_astrocyte=enable_astrocyte, alpha=alpha, k=k,
                    diagnostics=train_diag, input_refractory=input_refractory,
                    input_probability=input_probability, input_mode=input_mode,
                    sensory_generator=rngs["sensory"], policy_rng=rngs["action_numpy"],
                    enable_stdp=enable_stdp_during_training,
                    baseline_mode=baseline_mode,
                    stdp_selected_action_gain=stdp_selected_action_gain,
                    **simulation_precompute_args,
                )
            if diagnostics_enabled:
                all_episode_diagnostics.append({
                    "seed": seed, "experiment_num": experiment_num,
                    "phase": "training", "cycle": cycle_num,
                    "algorithm": training_condition,
                    "condition": training_condition,
                    "baseline_mode": baseline_mode,
                    "input_mode": input_mode,
                    "enable_stdp": enable_stdp_during_training,
                    "enable_reinforce": reinf_enable,
                    "apply_reinforce_update": reinf_enable,
                    "enforce_post_stdp_bounds": baseline_mode != "main_reference",
                    "chosen_stdp_gain": stdp_selected_action_gain,
                    "verification_mode": verification_mode,
                    "training_stdp_enabled": enable_stdp_during_training,
                    "training_reinforce_enabled": reinf_enable,
                    "enable_actor_critic": actor_critic_enable,
                    "training_actor_critic_enabled": actor_critic_enable,
                    "astrocyte_enabled": enable_astrocyte,
                    "surrogate_kind": surrogate_kind if reinf_enable else "none",
                    "input_probability": input_probability,
                    "input_mode": input_mode,
                    "input_refractory": input_refractory if input_refractory is not None else refrac,
                    **train_diag,
                })

            QAZ_after_train = weights_2d_train * weights_mask_XY
            all_weight_matrices.append(QAZ_after_train.copy())
            weights_init_XY = torch.Tensor(weights_2d_train).float()
            all_routes.append(f"Route (training, cycle {cycle_num}): {positions_train}")
            route_lengths.append(len(positions_train) - 1)

            if verification_mode != "none":
                print(f"\n{'=' * 60}")
                print(f"{verification_mode.upper()} CYCLE {cycle_num}/{num_cycles}")
                print(f"{'=' * 60}")
                verify_diag = {} if diagnostics_enabled else None
                weights_before_verify = weights_init_XY.detach().clone()

                if verification_mode == "source_verification":
                    # Notebook semantics: online STDP affects later actions in this
                    # temporary rollout; the returned matrix is intentionally discarded.
                    positions_verify, weights_2d_verify, goal_reached_verify, elapsed_time_verify = setup_and_run_simulation(
                        NA=NA, weights_mask_XY=weights_mask_XY, weights_init_XY=weights_init_XY,
                        weights_init_XI=weights_init_XI, n_steps=n_steps,
                        current_position=current_position, goal=goal,
                        learning_rate=learning_rate, wmin=wmin, wmax=wmax,
                        weight_decay=weight_decay, post_spike_weight_decay=post_spike_weight_decay,
                        reset=reset, refrac=refrac, thresh=thresh, intensity=intensity,
                        time_steps=time_steps, dt=dt, enable_astrocyte=False, alpha=alpha, k=k,
                        enable_stdp=True, diagnostics=verify_diag,
                        input_refractory=refrac, input_probability=None,
                        input_mode="legacy_static_positive",
                        sensory_generator=rngs["eval_sensory"],
                        policy_rng=rngs["eval_action_numpy"],
                        baseline_mode="main_reference", stdp_selected_action_gain=1.5,
                        **simulation_precompute_args,
                    )
                elif reinf_enable:
                    (positions_verify, weights_2d_verify, goal_reached_verify,
                     elapsed_time_verify, verify_baseline) = setup_and_run_simulation_reinforce(
                        NA=NA, weights_mask_XY=weights_mask_XY, weights_init_XY=weights_init_XY,
                        weights_init_XI=weights_init_XI, n_steps=n_steps,
                        current_position=current_position, goal=goal,
                        learning_rate=learning_rate, wmin=wmin, wmax=wmax,
                        weight_decay=weight_decay, post_spike_weight_decay=post_spike_weight_decay,
                        reset=reset, refrac=refrac, thresh=thresh, intensity=intensity,
                        time_steps=time_steps, dt=dt, enable_astrocyte=False, alpha=alpha, k=k,
                        reinforce_lr=reinf_lr, temperature=temperature, reward_goal=reward_goal,
                        step_penalty=step_penalty, gamma=gamma, baseline_decay=baseline_decay,
                        policy_mix_beta=policy_mix_beta, trace_decay=trace_decay,
                        enable_stdp_during_training=False, surrogate_kind=surrogate_kind,
                        surrogate_params=surrogate_params, running_baseline=running_baseline,
                        diagnostics=verify_diag, input_refractory=input_refractory,
                        input_probability=input_probability, input_mode=input_mode,
                        sensory_generator=rngs["eval_sensory"],
                        policy_generator=rngs["eval_action_torch"],
                        decision_diagnostics=all_policy_diagnostics if diagnostics_enabled else None,
                        decision_metadata={"seed": seed, "experiment_num": experiment_num,
                                           "phase": "frozen_evaluation", "cycle": cycle_num,
                                           "protocol_role": "verification",
                                           "astrocyte_enabled": False},
                        baseline_mode=baseline_mode,
                        stdp_selected_action_gain=stdp_selected_action_gain,
                        apply_reinforce_update=False,
                        **simulation_precompute_args,
                    )
                    if verify_baseline != running_baseline:
                        raise RuntimeError("Frozen verification changed the REINFORCE baseline")
                else:
                    positions_verify, weights_2d_verify, goal_reached_verify, elapsed_time_verify = setup_and_run_simulation(
                        NA=NA, weights_mask_XY=weights_mask_XY, weights_init_XY=weights_init_XY,
                        weights_init_XI=weights_init_XI, n_steps=n_steps,
                        current_position=current_position, goal=goal,
                        learning_rate=learning_rate, wmin=wmin, wmax=wmax,
                        weight_decay=weight_decay, post_spike_weight_decay=post_spike_weight_decay,
                        reset=reset, refrac=refrac, thresh=thresh, intensity=intensity,
                        time_steps=time_steps, dt=dt, enable_astrocyte=False, alpha=alpha, k=k,
                        enable_stdp=False, diagnostics=verify_diag,
                        input_refractory=input_refractory, input_probability=input_probability,
                        input_mode=input_mode, sensory_generator=rngs["eval_sensory"],
                        policy_rng=rngs["eval_action_numpy"], baseline_mode=baseline_mode,
                        stdp_selected_action_gain=stdp_selected_action_gain,
                        apply_action_mask=False,
                        **simulation_precompute_args,
                    )

                if verification_mode == "frozen_evaluation" and not np.array_equal(
                    weights_2d_verify, weights_before_verify.detach().cpu().numpy(),
                ):
                    raise RuntimeError("Frozen verification changed learned weights")
                if diagnostics_enabled:
                    all_episode_diagnostics.append({
                        "seed": seed, "experiment_num": experiment_num,
                        "phase": verification_mode, "cycle": cycle_num,
                        "condition": training_condition,
                        "baseline_mode": baseline_mode, "input_mode": input_mode,
                        "enable_stdp": verification_mode == "source_verification",
                        "enable_reinforce": reinf_enable and verification_mode == "frozen_evaluation",
                        "apply_reinforce_update": False,
                        "enforce_post_stdp_bounds": baseline_mode != "main_reference",
                        "chosen_stdp_gain": stdp_selected_action_gain,
                        "astrocyte_enabled": False,
                        "input_probability": input_probability,
                        "input_refractory": input_refractory if input_refractory is not None else refrac,
                        **verify_diag,
                    })
                all_routes.append(f"Route ({verification_mode}, cycle {cycle_num}): {positions_verify}")
                route_lengths.append(len(positions_verify) - 1)
                verify_route_length = len(positions_verify) - 1
                verify_success = goal_reached_verify
            else:
                positions_verify = []
                goal_reached_verify = np.nan
                elapsed_time_verify = 0.0
                verify_route_length = np.nan
                verify_success = np.nan

            metrics = {
                "train/route_length": len(positions_train) - 1,
                "train/goal_reached": goal_reached_train,
                "train/elapsed_time": elapsed_time_train,
                "verify/route_length": verify_route_length,
                "verify/goal_reached": verify_success,
                "verify/elapsed_time": elapsed_time_verify,
            }
            if actor_critic_enable:
                metrics.update({
                    "train/episode_return": training_episode_return,
                    "train/mean_abs_td_error": train_diag["mean_abs_td_error"]
                    if train_diag is not None else np.nan,
                    "train/mean_actor_update_norm": train_diag["mean_actor_update_norm"]
                    if train_diag is not None else np.nan,
                    "train/critic_value_start": (
                        train_diag["critic_value_start"] if train_diag is not None
                        else float(td_critic.values[current_position].item())
                        if td_critic is not None else np.nan
                    ),
                })
            logger.log_metrics(metrics, step=cycle_num)

            if diagnostics_enabled and cycle_num in evaluation_checkpoints and evaluation_rollouts > 0:
                for astrocyte_mode in evaluation_astrocyte_modes:
                    policy_modes = (
                        [("native_spike_softmax", "spike_softmax"),
                         ("common_spike_argmax", "common_spike_argmax")]
                        if actor_critic_enable else (
                            [("native_reinforce_softmax", None),
                             ("common_spike_argmax", None)]
                            if reinf_enable else [("common_spike_argmax", None)]
                        )
                    )
                    for evaluation_policy, td_policy_mode in policy_modes:
                        for rollout_idx in range(evaluation_rollouts):
                            eval_seed = derive_seed(
                                seed, 100_000 + cycle_num * 1000 + rollout_idx,
                            )
                            eval_rngs = make_rng_streams(eval_seed)
                            eval_diag = {}
                            if actor_critic_enable:
                                frozen_weights = weights_init_XY.detach().clone()
                                frozen_critic = (
                                    td_critic.values.detach().clone()
                                    if td_critic is not None else None
                                )
                                (eval_positions, eval_weights, eval_success,
                                 eval_elapsed, eval_return) = setup_and_run_simulation_td_rstdp(
                                    NA=NA, weights_mask_XY=weights_mask_XY,
                                    weights_init_XY=weights_init_XY,
                                    weights_init_XI=weights_init_XI, n_steps=n_steps,
                                    current_position=current_position, goal=goal,
                                    learning_rate=learning_rate, wmin=wmin, wmax=wmax,
                                    weight_decay=weight_decay,
                                    post_spike_weight_decay=post_spike_weight_decay,
                                    reset=reset, refrac=refrac, thresh=thresh,
                                    intensity=intensity, time_steps=time_steps, dt=dt,
                                    enable_astrocyte=bool(astrocyte_mode), alpha=alpha, k=k,
                                    critic=td_critic,
                                    actor_learning_rate=actor_learning_rate,
                                    eligibility_tau_ms=eligibility_tau_ms,
                                    temperature=actor_temperature,
                                    reward_goal=reward_goal, step_penalty=step_penalty,
                                    input_refractory=input_refractory,
                                    input_probability=input_probability,
                                    input_mode=input_mode,
                                    sensory_generator=eval_rngs["sensory"],
                                    policy_generator=eval_rngs["eval_action_torch"],
                                    policy_rng=eval_rngs["eval_action_numpy"],
                                    actor_weight_decay=actor_weight_decay,
                                    actor_modulator=(
                                        "reward" if actor_rule == "reward_rstdp"
                                        else "td_error"
                                    ),
                                    gamma=actor_critic_gamma,
                                    apply_actor_update=False,
                                    apply_critic_update=False,
                                    policy_mode=td_policy_mode,
                                    diagnostics=eval_diag,
                                    actor_step_diagnostics=all_actor_critic_steps,
                                    decision_metadata={
                                        "seed": seed, "experiment_num": experiment_num,
                                        "phase": "frozen_evaluation", "cycle": cycle_num,
                                        "rollout": rollout_idx,
                                        "condition": training_condition,
                                        "astrocyte_enabled": bool(astrocyte_mode),
                                        "evaluation_policy": evaluation_policy,
                                    },
                                    **simulation_precompute_args,
                                )
                                if not np.array_equal(
                                    eval_weights, frozen_weights.cpu().numpy(),
                                ):
                                    raise RuntimeError("Frozen TD-RSTDP evaluation changed actor weights")
                                if td_critic is not None and not torch.equal(
                                    td_critic.values, frozen_critic,
                                ):
                                    raise RuntimeError("Frozen TD-RSTDP evaluation changed critic values")
                            elif reinf_enable and evaluation_policy == "native_reinforce_softmax":
                                (eval_positions, eval_weights, eval_success,
                                 eval_elapsed, _eval_baseline) = setup_and_run_simulation_reinforce(
                                    NA=NA, weights_mask_XY=weights_mask_XY,
                                    weights_init_XY=weights_init_XY,
                                    weights_init_XI=weights_init_XI, n_steps=n_steps,
                                    current_position=current_position, goal=goal,
                                    learning_rate=learning_rate, wmin=wmin, wmax=wmax,
                                    weight_decay=weight_decay,
                                    post_spike_weight_decay=post_spike_weight_decay,
                                    reset=reset, refrac=refrac, thresh=thresh,
                                    intensity=intensity, time_steps=time_steps, dt=dt,
                                    enable_astrocyte=bool(astrocyte_mode), alpha=alpha, k=k,
                                    reinforce_lr=0.0, temperature=temperature,
                                    reward_goal=reward_goal, step_penalty=step_penalty,
                                    gamma=gamma, baseline_decay=baseline_decay,
                                    policy_mix_beta=policy_mix_beta, trace_decay=trace_decay,
                                    enable_stdp_during_training=False,
                                    surrogate_kind=surrogate_kind,
                                    surrogate_params=surrogate_params,
                                    running_baseline=running_baseline, diagnostics=eval_diag,
                                    input_refractory=input_refractory,
                                    input_probability=input_probability, input_mode=input_mode,
                                    sensory_generator=eval_rngs["sensory"],
                                    policy_generator=eval_rngs["eval_action_torch"],
                                    decision_diagnostics=all_policy_diagnostics,
                                    baseline_mode=baseline_mode,
                                    stdp_selected_action_gain=stdp_selected_action_gain,
                                    apply_reinforce_update=False,
                                    **simulation_precompute_args,
                                    decision_metadata={
                                        "seed": seed, "experiment_num": experiment_num,
                                        "phase": "frozen_evaluation", "cycle": cycle_num,
                                        "rollout": rollout_idx,
                                        "astrocyte_enabled": bool(astrocyte_mode),
                                    },
                                )
                            else:
                                (eval_positions, eval_weights, eval_success,
                                 eval_elapsed) = setup_and_run_simulation(
                                    NA=NA, weights_mask_XY=weights_mask_XY,
                                    weights_init_XY=weights_init_XY,
                                    weights_init_XI=weights_init_XI, n_steps=n_steps,
                                    current_position=current_position, goal=goal,
                                    learning_rate=learning_rate, wmin=wmin, wmax=wmax,
                                    weight_decay=weight_decay,
                                    post_spike_weight_decay=post_spike_weight_decay,
                                    reset=reset, refrac=refrac, thresh=thresh,
                                    intensity=intensity, time_steps=time_steps, dt=dt,
                                    enable_astrocyte=bool(astrocyte_mode), alpha=alpha, k=k,
                                    enable_stdp=False, diagnostics=eval_diag,
                                    input_refractory=input_refractory,
                                    input_probability=input_probability, input_mode=input_mode,
                                    sensory_generator=eval_rngs["sensory"],
                                    policy_rng=eval_rngs["action_numpy"],
                                    baseline_mode=baseline_mode,
                                    stdp_selected_action_gain=stdp_selected_action_gain,
                                    apply_action_mask=False,
                                    **simulation_precompute_args,
                                )
                            all_evaluation_rollouts.append({
                                "seed": seed, "experiment_num": experiment_num,
                                "checkpoint_cycle": cycle_num,
                                "rollout": rollout_idx,
                                "condition": training_condition,
                                "evaluation_policy": evaluation_policy,
                                "baseline_mode": baseline_mode,
                                "input_mode": input_mode,
                                "enable_stdp": False,
                                "enable_reinforce": reinf_enable,
                                "enable_actor_critic": actor_critic_enable,
                                "apply_reinforce_update": False,
                                "apply_actor_update": False,
                                "apply_critic_update": False,
                                "enforce_post_stdp_bounds": baseline_mode != "main_reference",
                                "chosen_stdp_gain": stdp_selected_action_gain,
                                "verification_mode": verification_mode,
                                "training_stdp_enabled": enable_stdp_during_training,
                                "training_reinforce_enabled": reinf_enable,
                                "training_actor_critic_enabled": actor_critic_enable,
                                "training_astrocyte_enabled": enable_astrocyte,
                                "astrocyte_enabled": bool(astrocyte_mode),
                                "success": int(eval_success),
                                "route_length": len(eval_positions) - 1,
                                "failure": int(not eval_success),
                                "elapsed_seconds": eval_elapsed,
                                "reward_return": eval_return if actor_critic_enable else eval_diag.get("reward_return", np.nan),
                                "weight_drift_norm": eval_diag.get("weight_drift_norm", np.nan),
                                "fraction_weights_outside_bounds": eval_diag.get("fraction_weights_outside_bounds", np.nan),
                                "X_spikes": eval_diag.get("X_spikes", np.nan),
                                "Y_spikes": eval_diag.get("Y_spikes", np.nan),
                                "I_spikes": eval_diag.get("I_spikes", np.nan),
                                "raw_input_events": eval_diag.get("raw_input_events", np.nan),
                                "tie_frequency": eval_diag.get("tie_frequency", np.nan),
                                "astro_active_fraction": eval_diag.get("astro_active_fraction", np.nan),
                                "astro_mean_threshold_drop": eval_diag.get("astro_mean_threshold_drop", np.nan),
                                "delta_w_stdp_norm": eval_diag.get("delta_w_stdp_norm", np.nan),
                                "stdp_step_norm_sum": eval_diag.get("stdp_step_norm_sum", np.nan),
                                "stdp_raw_update_l1": eval_diag.get("stdp_raw_update_l1", np.nan),
                                "delta_w_rl_preclip_norm": eval_diag.get("delta_w_rl_preclip_norm", np.nan),
                                "delta_w_rl_norm": eval_diag.get("delta_w_rl_norm", np.nan),
                                "mean_abs_td_error": eval_diag.get("mean_abs_td_error", np.nan),
                                "mean_actor_update_norm": eval_diag.get("mean_actor_update_norm", np.nan),
                                "input_probability": input_probability,
                                "input_refractory": input_refractory if input_refractory is not None else refrac,
                            })

        results_dir = str(logger.active_exp_dir)
        if (
            td_critic is not None
            and bool(actor_critic_cfg.get("save_critic_values", False))
        ):
            td_critic.save(
                results_dir, grid_size=N,
                make_heatmap=bool(actor_critic_cfg.get("save_critic_heatmap", True)),
            )

        weights_filename = os.path.join(results_dir, "weight_matrices.txt")
        with open(weights_filename, 'w') as f:
            f.write(f"WEIGHT MATRICES: EXPERIMENT RESULTS #{experiment_num}\n")
            f.write("=" * 50 + "\n\n")
            for i, weight_matrix in enumerate(all_weight_matrices):
                if i == 0:
                    f.write("INITIAL WEIGHT MATRIX:\n")
                else:
                    f.write(f"WEIGHT MATRIX AFTER TRAINING CYCLE {i}:\n")
                np.savetxt(f, weight_matrix, fmt='%.4f')
                f.write("\n" + "-" * 30 + "\n\n")

        routes_filename = os.path.join(results_dir, "navigation_routes.txt")
        with open(routes_filename, 'w') as f:
            f.write(f"NAVIGATION ROUTES: EXPERIMENT RESULTS #{experiment_num}\n")
            f.write("=" * 50 + "\n\n")
            f.write("ALL ROUTES:\n")
            f.write("=" * 30 + "\n")
            for i, route in enumerate(all_routes):
                f.write(f"{i + 1:2d}. {route}\n")
            f.write("\n")
            f.write("ROUTE LENGTHS:\n")
            f.write("=" * 30 + "\n")
            for i, length in enumerate(route_lengths):
                f.write(f"{i + 1:2d}. Route length: {length}\n")

        mat_filename = os.path.join(results_dir, "weight_matrices.mat")
        sio.savemat(mat_filename, {'weight_matrices': np.array(all_weight_matrices)})
        lengths_mat_filename = os.path.join(results_dir, "route_lengths.mat")
        sio.savemat(lengths_mat_filename, {'route_lengths': np.array(route_lengths)})

        all_experiment_results.append({
            "experiment_num": experiment_num,
            "routes": all_routes,
            "route_lengths": route_lengths,
            "weight_matrices": all_weight_matrices,
        })

    print("\n" + "=" * 50)
    print(f"ALL {num_experiments} EXPERIMENTS COMPLETED!")

    if diagnostics_enabled:
        run_name = config.get("run_name", "run")
        diag_dir = Path(diagnostic_cfg.get(
            "output_dir", Path(config.get("output_dir", "results")) / "diagnostics" / run_name,
        ))
        if diag_dir.name != run_name and diagnostic_cfg.get("output_dir"):
            diag_dir = diag_dir / run_name
        _write_rows(diag_dir / "raw_per_seed.csv", all_episode_diagnostics)
        _write_rows(diag_dir / "evaluation_rollouts.csv", all_evaluation_rollouts)
        _write_rows(diag_dir / "policy_diagnostics.csv", all_policy_diagnostics)
        _write_rows(diag_dir / "actor_critic_steps.csv", all_actor_critic_steps)
        _write_rows(
            diag_dir / "summary_statistics.csv",
            _build_run_summaries(all_episode_diagnostics, all_evaluation_rollouts,
                                 seed=int(config.get("seed", 0)) + 983),
        )
        (diag_dir / "metadata.json").write_text(json.dumps({
            "run_name": run_name,
            "config": config,
            "rng_streams": {
                "weights": "NumPy RandomState(seed), independent; sequence matches pre-stream initializer",
                "sensory": "Torch Generator from derive_seed(seed, 1)",
                "policy": "independent NumPy/Torch stream from derive_seed(seed, 2)",
                "evaluation_sensory": "Torch Generator from derive_seed(seed, 3)",
                "evaluation_policy": "independent NumPy/Torch stream from derive_seed(seed, 4)",
            },
            "evaluation": {
                "rollouts_per_checkpoint_mode": evaluation_rollouts,
                "checkpoints": sorted(evaluation_checkpoints),
                "astrocyte_modes": evaluation_astrocyte_modes,
                "policy_modes": (
                    ["native_spike_softmax", "common_spike_argmax"]
                    if actor_critic_enable else (
                        ["native_reinforce_softmax", "common_spike_argmax"]
                        if reinf_enable else ["common_spike_argmax"]
                    )
                ),
                "stdp": False, "reinforce_lr": 0.0,
            },
        }, indent=2, default=str))

    logger.finish()
    return all_experiment_results
