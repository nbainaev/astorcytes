"""Episode budgets, training trials, paired frozen evaluation and scalar metrics."""

import gzip, json, time
import numpy as np
import torch
from astrorl.memory import FamiliarityMemory as Memory, gating
from astrorl.environment import geometry
from astrorl.learner import Learner
from astrorl.mlp import fingerprint
from astrorl.logging import read_json as read, write_json as write
from .prediction_metrics import input_codebook, nearest_cell
from .configuration import CONFIGS, OUT
from .configuration import GROUPS as GROUPS, KINDS as KINDS


def make_learner(kind, rl, seed, hp):
    return Learner(
        kind,
        rl,
        seed,
        hp,
        model_config=read(CONFIGS / "model.json"),
        memory_config=read(CONFIGS / "memory.json"),
    )


def schedule(spec, seed, count, split):
    free = np.asarray(
        [p for p in sorted(map(tuple, spec["free"])) if p != max(map(tuple, spec["free"]))]
    )
    rng = np.random.default_rng(np.random.SeedSequence([20261023, seed, split, 917]))
    return free[rng.integers(len(free), size=count)]


def action_seed(seed, episode, split):
    return int(np.random.SeedSequence([20261023, seed, split, episode, 913]).generate_state(1)[0])


def summarize(rows):
    keys = [
        k
        for k, v in rows[0].items()
        if isinstance(v, (int, float, np.number)) and k not in ("episode",)
    ]
    out = {k: float(np.mean([r[k] for r in rows if r.get(k) is not None])) for k in keys}
    # A missing move/collision stratum is retained as missing, never replaced by 0.
    for key in ("prediction_move_accuracy", "prediction_collision_accuracy"):
        vals = [r[key] for r in rows if r[key] is not None]
        out[key] = float(np.mean(vals)) if vals else None
    return out


def trajectory(learner, spec, start, seed, training=False, memory_on=True):
    free, index, goal, dist, tr = geometry(spec)
    size = spec["size"]
    state = index[tuple(start)]
    assert 0 < dist[tuple(start)] <= 50
    learner.reset()
    codebook = input_codebook(learner, spec, free)
    rng = np.random.default_rng(seed)
    previous = None
    use_memory = memory_on and learner.kind in ("astro", "trace")
    memory = (
        Memory(
            learner.input_channels,
            "trace" if learner.kind == "trace" else "astro",
            learner.memory_params,
        )
        if use_memory
        else None
    )
    records = []
    visited = {tuple(start)}
    path = [tuple(start)]
    pairs = set()
    cover = []
    occupancy = np.zeros((size, size), np.int64)
    arrivals = occupancy.copy()
    pred_count = occupancy.copy()
    pred_error = occupancy.copy()
    back = collisions = revisits = correct = move_correct = collision_correct = moves = 0
    errors = []
    entropies = []
    gates = []
    silent = []
    discrimination = []
    for tick in range(50):
        pos = free[state]
        obs = np.asarray(pos, dtype=np.float32) / (size - 1)
        rep = learner.representation(obs, previous, training)
        predicted_input = learner.prediction(obs, rep)
        if use_memory:
            memory.write(rep["input_code"], int((seed + tick * 7919) % 2**32))
            gate = gating(memory.familiarity(predicted_input), learner.memory_params)
        else:
            gate = np.zeros(4)
        p, cache, diag = learner.policy(obs, rep, gate, use_memory)
        assert np.isfinite(p).all() and abs(p.sum() - 1) < 1e-6
        action = min(3, int(np.searchsorted(np.cumsum(p), rng.random(), side="right")))
        nxt_state = int(tr[state, action])
        nxt = free[nxt_state]
        hit = nxt_state == goal
        collision = nxt_state == state
        moved = not collision
        target_obs = np.asarray(nxt, dtype=np.float32) / (size - 1)
        accurate = nearest_cell(predicted_input[action], action, codebook) == nxt_state
        correct += accurate
        move_correct += accurate and moved
        collision_correct += accurate and collision
        moves += moved
        pred_count[pos] += 1
        pred_error[pos] += not accurate
        next_input = learner.input_code(target_obs, action)
        errors.append(float(np.square(predicted_input[action] - next_input).mean()))
        discrimination.append(float(np.var(predicted_input, axis=0).mean()))
        occupancy[pos] += 1
        arrivals[nxt] += 1
        pairs.add((pos, action))
        back += moved and len(path) > 1 and nxt == path[-2]
        revisits += nxt in visited
        collisions += collision
        visited.add(nxt)
        path.append(nxt)
        cover.append(len(visited) / len(free))
        entropies.append(float(-np.sum(p * np.log(np.maximum(p, 1e-30)))))
        gates.append(float(diag["mean_gate"]))
        silent.append(float(diag["motor_silent_fraction"]))
        if training:
            records.append(
                dict(
                    **rep,
                    obs=obs,
                    action=action,
                    next_obs=target_obs,
                    next_input=next_input,
                    reward=float(hit) - 0.01,
                    terminal=hit or tick == 49,
                    probability=p,
                    head_cache=cache,
                )
            )
        previous = action
        state = nxt_state
        if hit:
            break
    T = tick + 1
    success = int(state == goal)
    distance = dist[tuple(start)]
    reward = success - 0.01 * T
    q = occupancy[occupancy > 0] / T
    persisted = T - moves
    row = dict(
        steps=T,
        success=success,
        reward=reward,
        reward01=(reward + 0.5) / (1.5 - 0.01 * distance),
        optimal_steps=distance,
        optimal_over_steps=distance / T,
        path_efficiency=success * distance / T,
        coverage=len(visited) / len(free),
        coverage_auc=float(np.mean(cover)),
        coverage_auc_50=float(np.mean(cover + [cover[-1]] * (50 - T))),
        unique_states=len(visited),
        novelty_rate=(len(visited) - 1) / T,
        exploration_efficiency=(len(visited) - 1) / min(T, len(free) - 1),
        backtracking_rate=back / max(moves, 1),
        backtracking_per_action=back / T,
        collision_rate=collisions / T,
        revisit_rate=revisits / T,
        state_action_coverage=len(pairs) / (4 * len(free)),
        spatial_entropy=float(-np.sum(q * np.log(q)) / np.log(len(free))),
        loop2_rate=float(np.mean([path[k] == path[k - 2] for k in range(2, len(path))]))
        if T >= 2
        else 0.0,
        loop4_rate=float(np.mean([path[k] == path[k - 4] for k in range(4, len(path))]))
        if T >= 4
        else 0.0,
        entropy=float(np.mean(entropies)),
        mean_gate=float(np.mean(gates)),
        motor_silent_fraction=float(np.mean(silent)),
        prediction_accuracy=correct / T,
        prediction_move_accuracy=move_correct / moves if moves else None,
        prediction_collision_accuracy=collision_correct / persisted if persisted else None,
        prediction_balanced_accuracy=0.5 * (move_correct / moves + collision_correct / persisted)
        if moves and persisted
        else correct / T,
        prediction_mse=float(np.mean(errors)),
        prediction_rmse=float(np.sqrt(np.mean(errors))),
        persistence_accuracy=persisted / T,
        action_discrimination=float(np.mean(discrimination)),
        prediction_move_count=moves,
        prediction_collision_count=persisted,
    )
    assert 0 <= row["reward01"] <= 1 + 1e-9 and 0 < row["optimal_over_steps"] <= 1 + 1e-9
    return (
        row,
        records,
        dict(occupancy=occupancy, arrivals=arrivals, pred_count=pred_count, pred_error=pred_error),
        path,
    )


def prediction_probe(learner, spec, seed, contexts=128, length=4):
    """Common forced-random contexts, not additional training/evaluation episodes.

    Balanced source contexts include the goal, with no terminal rule; this offline
    transition diagnostic measures prediction on identical inputs for all models.
    Only the evaluator consults transitions. No map/optimal actions reach agents.
    """
    free, index, goal, dist, tr = geometry(spec)
    size = spec["size"]
    codebook = input_codebook(learner, spec, free)
    rng = np.random.default_rng(seed)
    correct = []
    moving = []
    errors = []
    persistence = []
    cells = []
    actions = []
    pred_count = np.zeros((size, size), np.int64)
    pred_error = pred_count.copy()
    for _ in range(contexts):
        state = int(rng.integers(len(free)))
        learner.reset()
        previous = None
        for _ in range(length):
            pos = free[state]
            obs = np.asarray(pos, dtype=np.float32) / (size - 1)
            rep = learner.representation(obs, previous, False)
            predicted_input = learner.prediction(obs, rep)
            action = int(rng.integers(4))
            next_state = int(tr[state, action])
            nxt = free[next_state]
            target_obs = np.asarray(nxt, dtype=np.float32) / (size - 1)
            accurate = nearest_cell(predicted_input[action], action, codebook) == next_state
            correct.append(accurate)
            moving.append(next_state != state)
            target_input = learner.input_code(target_obs, action)
            errors.append(float(np.square(predicted_input[action] - target_input).mean()))
            persistence.append(next_state == state)
            pred_count[pos] += 1
            pred_error[pos] += not accurate
            cells.append(pos)
            actions.append(action)
            previous = action
            state = next_state
    c = np.asarray(correct)
    move = np.asarray(moving)
    acc_move = float(c[move].mean()) if move.any() else None
    acc_col = float(c[~move].mean()) if (~move).any() else None
    result = dict(
        prediction_accuracy=float(c.mean()),
        prediction_move_accuracy=acc_move,
        prediction_collision_accuracy=acc_col,
        prediction_balanced_accuracy=0.5 * (acc_move + acc_col),
        prediction_mse=float(np.mean(errors)),
        prediction_rmse=float(np.sqrt(np.mean(errors))),
        persistence_accuracy=float(np.mean(persistence)),
        prediction_examples=len(c),
        state_action_coverage=len(set(zip(cells, actions))) / (4 * len(free)),
        source_hash=fingerprint([np.asarray(cells), np.asarray(actions)]),
    )
    return result, dict(pred_count=pred_count, pred_error=pred_error)


def evaluate(learner, spec, seed, count=100, split=8):
    before = learner.weights_hash()
    base = learner.baseline.copy()
    modes = ("on", "off") if learner.kind in ("astro", "trace") else ("off",)
    starts = schedule(spec, seed, count, split)
    results = {}
    arrays = {}
    episode_rows = {}
    for mode in modes:
        rows = []
        totals = None
        for i, start in enumerate(starts):
            row, _, stats, _ = trajectory(
                learner, spec, start, action_seed(seed, i, split), memory_on=mode == "on"
            )
            row.update(episode=i + 1)
            rows.append(row)
            if totals is None:
                totals = {k: v.copy() for k, v in stats.items()}
            else:
                for k, v in stats.items():
                    totals[k] += v
        results[mode] = summarize(rows)
        arrays[mode] = totals
        episode_rows[mode] = rows
    assert before == learner.weights_hash() and np.array_equal(base, learner.baseline), (
        "Evaluation mutated learned state"
    )
    probe, probe_arrays = prediction_probe(learner, spec, action_seed(seed, 0, split + 20))
    assert before == learner.weights_hash()
    return (
        dict(
            evaluation=results,
            probe=probe,
            count=count,
            start_hash=fingerprint(starts),
            weights_frozen=True,
        ),
        arrays,
        episode_rows,
        probe_arrays,
    )


def objective(result):
    group = result["task"]["group"]
    kind = result["task"]["kind"]
    final = result["final"]
    ev = final["evaluation"]["on" if kind in ("astro", "trace") else "off"]
    pred = final["probe"]["prediction_balanced_accuracy"]
    if group == "no_rl" or kind == "random":
        return 0.4 * ev["coverage_auc_50"] + 0.2 * ev["novelty_rate"] + 0.4 * pred
    return 0.5 * ev["reward01"] + 0.3 * ev["optimal_over_steps"] + 0.2 * pred


def checkpoint(path, learner, task, episode, history, development, occupancy, starts):
    learner.core = None
    tmp = path.with_suffix(".tmp")
    torch.save(
        dict(
            learner=learner,
            task=task,
            episode=episode,
            history=history,
            development=development,
            occupancy=occupancy,
            start_hash=fingerprint(starts),
        ),
        tmp,
    )
    tmp.replace(path)


def case(task):
    torch.set_num_threads(1)
    kind = task["kind"]
    group = task["group"]
    spec = task["map"]
    seed = task["seed"]
    hp = task["hp"]
    episodes = task["episodes"]
    stage = task["stage"]
    began = time.time()
    folder = (
        OUT
        / ("runs" if stage == "final" else "tuning_raw")
        / ("main" if stage == "final" else stage)
        / group
        / kind
        / spec["name"]
        / hp["name"]
        / f"s{seed}"
    )
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "summary.json").exists():
        cached = read(folder / "summary.json")
        assert cached["task"] == task
        return cached
    resume = folder / "resume.pt"
    starts = schedule(spec, seed, episodes, 1)
    if resume.exists():
        state = torch.load(resume, weights_only=False)
        assert state["task"] == task
        learner = state["learner"]
        history = state["history"]
        development = state["development"]
        occupancy = state["occupancy"]
        first = state["episode"]
        assert state["start_hash"] == fingerprint(starts)
    else:
        learner = make_learner(kind, group == "rl", seed, hp)
        history = []
        development = []
        first = 0
        occupancy = np.zeros((6, spec["size"], spec["size"]), np.int64)
        initial, _ = prediction_probe(
            learner, spec, action_seed(seed, 0, 5), contexts=32 if stage != "final" else 128
        )
        development.append(dict(episode=0, training_steps=0, **initial))
    for ep in range(first + 1, episodes + 1):
        learner.schedule(ep, episodes)
        row, records, stats, _ = trajectory(
            learner, spec, starts[ep - 1], action_seed(seed, ep, 1), True
        )
        row.update(learner.update(records))
        row.update(episode=ep)
        row["actor_learning_rate"] = learner.actor_lr
        row["entropy_coefficient"] = learner.entropy
        row["training_steps"] = sum(r["steps"] for r in history) + row["steps"]
        history.append(row)
        occupancy[min(5, (ep - 1) * 6 // episodes)] += stats["occupancy"]
        if ep % 100 == 0 or ep == episodes:
            if stage == "final" or ep == episodes:
                probe, _ = prediction_probe(
                    learner, spec, action_seed(seed, 0, 5), contexts=32 if stage != "final" else 128
                )
                development.append(dict(episode=ep, training_steps=row["training_steps"], **probe))
            checkpoint(resume, learner, task, ep, history, development, occupancy, starts)
        if ep % 25 == 0:
            write(
                folder / "progress.json",
                dict(
                    episode=ep,
                    episodes=episodes,
                    training_steps=row["training_steps"],
                    seconds=time.time() - began,
                ),
            )
    final, eval_arrays, eval_rows, probe_arrays = evaluate(
        learner, spec, seed, count=task.get("eval_count", 100), split=8 if stage == "final" else 7
    )
    result = dict(
        task=task,
        final=final,
        last100=summarize(history[-min(100, len(history)) :]),
        training_steps=sum(r["steps"] for r in history),
        episodes=episodes,
        start_hash=fingerprint(starts),
        seconds=time.time() - began,
        path=str(folder),
        metadata=learner.metadata(),
        learning_audit=dict(
            nonzero_prediction_updates=sum(r["prediction_core_gradient_norm"] > 0 for r in history),
            max_policy_core_gradient=max(r["policy_core_gradient_norm"] for r in history),
            actor_updates=sum(r["actor_gradient_norm"] > 0 for r in history),
            predictor_changed=learner.predictor_hash() != learner.initial_predictor_hash,
        ),
    )
    result["objective"] = objective(result)
    assert result["learning_audit"]["predictor_changed"]
    if group == "no_rl" or kind == "random":
        assert result["metadata"]["initial_policy_hash"] == result["metadata"]["final_policy_hash"]
    if stage == "final":
        checkpoint(
            folder / "checkpoint_0600.pt",
            learner,
            task,
            episodes,
            history,
            development,
            occupancy,
            starts,
        )
        with gzip.open(folder / "metrics.json.gz", "wt") as f:
            json.dump(
                dict(training=history, development=development, evaluation=eval_rows),
                f,
                allow_nan=False,
            )
        data = dict(
            training_occupancy=occupancy,
            probe_count=probe_arrays["pred_count"],
            probe_error=probe_arrays["pred_error"],
        )
        for mode, arrays in eval_arrays.items():
            for key, value in arrays.items():
                data[f"eval_{mode}_{key}"] = value
        np.savez_compressed(folder / "spatial.npz", **data)
    write(folder / "summary.json", result)
    if stage == "final" and resume.exists():
        resume.unlink()
    return result
