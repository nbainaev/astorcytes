"""Independent local-rule derivatives and scientific experiment invariants."""

from pathlib import Path

from concurrent.futures import ProcessPoolExecutor, as_completed
import copy, shutil
import numpy as np
import torch
from . import engine as e
from astrorl.mlp import LocalMLP, sigmoid
from .tuning import candidates


def local_derivatives():
    net = LocalMLP(70013, 6, 20)
    rng = np.random.default_rng(441)
    x = rng.normal(size=(9, 6))
    delta = rng.normal(size=(9, 20))
    q, cache = net.forward(x)
    analytic = net.gradients(cache, delta)
    p = {k: torch.tensor(v, requires_grad=True) for k, v in net.p.items()}
    xx, h1, h2, a1, a2 = cache
    # Independent sum of local surrogate objectives, with detached pre-activity.
    loss = (torch.relu(torch.tensor(xx) @ p["w1"] + p["b1"]) * torch.tensor(delta @ net.B1.T)).sum()
    loss += (
        torch.relu(torch.tensor(h1) @ p["w2"] + p["b2"]) * torch.tensor(delta @ net.B2.T)
    ).sum()
    loss += ((torch.tensor(h2) @ p["w3"] + p["b3"]) * torch.tensor(delta)).sum()
    loss.backward()
    errors = {k: float(np.max(np.abs(v - p[k].grad.numpy()))) for k, v in analytic.items()}
    assert max(errors.values()) < 1e-11, errors
    other = copy.deepcopy(net)
    other.p["w3"] *= 19
    other.p["w2"] *= -7
    alt = other.gradients(cache, delta)
    assert all(np.array_equal(analytic[k], alt[k]) for k in ("w1", "b1", "w2", "b2"))
    # A single mean-squared prediction objective, with the channel mean explicit.
    raw = torch.tensor(q, requires_grad=True)
    target = rng.uniform(size=q.shape)
    loss = torch.square(torch.sigmoid(raw) - torch.tensor(target)).mean()
    loss.backward()
    prediction = sigmoid(q)
    dq = 2 * (prediction - target) * prediction * (1 - prediction) / prediction.size
    output_error = float(np.max(np.abs(raw.grad.numpy() - dq)))
    assert output_error < 1e-12
    return dict(
        local_surrogate_errors=errors,
        prediction_output_error=output_error,
        hidden_signal_independent_of_downstream_weights=True,
    )


def matched_initial_behavior(spec):
    hp = candidates("noastro", "no_rl", spec)[0]
    start = e.schedule(spec, 70900, 1, 1)[0]
    paths = []
    hashes = []
    for kind in ("random", "noastro", "mlp"):
        learner = e.make_learner(kind, False, 70900, hp)
        row, records, stats, path = e.trajectory(learner, spec, start, 88117, False)
        assert row["steps"] <= 50 and 0 <= row["reward01"] <= 1
        paths.append(path)
    assert paths[0] == paths[1] == paths[2]
    for kind in ("noastro", "astro", "trace"):
        learner = e.make_learner(kind, False, 70901, hp)
        hashes.append(
            (
                learner.initial_backbone_hash,
                learner.initial_predictor_hash,
                learner.initial_policy_hash,
            )
        )
    assert len(set(hashes)) == 1
    return dict(uniform_no_rl_policies_paired=True, identical_snn_initialization=True)


def main():
    torch.set_num_threads(1)
    e.OUT.mkdir(parents=True, exist_ok=True)
    specs = e.read(e.CONFIGS / "maps.json")["maps"]
    derivative = local_derivatives()
    paired = matched_initial_behavior(specs[0])
    tasks = []
    # 40 short runs exercise all four maps, five models and both learning groups
    # with exactly 30 workers, plus on/off frozen evaluation where applicable.
    for group in e.GROUPS:
        for kind in e.KINDS:
            for spec in specs:
                hp = candidates(kind, group, spec)[0]
                tasks.append(
                    dict(
                        kind=kind,
                        group=group,
                        map=spec,
                        hp=hp,
                        seed=70910,
                        stage="check",
                        episodes=2,
                        eval_count=2,
                    )
                )
    summaries = []
    with ProcessPoolExecutor(max_workers=30) as pool:
        for future in as_completed([pool.submit(e.case, t) for t in tasks]):
            row = future.result()
            summaries.append(row)
            assert row["training_steps"] <= 100
            assert row["learning_audit"]["max_policy_core_gradient"] == 0
            assert row["final"]["weights_frozen"]
            assert row["learning_audit"]["nonzero_prediction_updates"] > 0
            if row["task"]["kind"] != "mlp":
                assert (
                    row["metadata"]["initial_backbone_hash"]
                    != row["metadata"]["final_backbone_hash"]
                )
            expected = {"on", "off"} if row["task"]["kind"] in ("astro", "trace") else {"off"}
            assert set(row["final"]["evaluation"]) == expected
            print(
                "CHECK",
                len(summaries),
                40,
                row["task"]["group"],
                row["task"]["kind"],
                row["task"]["map"]["name"],
                flush=True,
            )
    assert len(summaries) == 40
    # Reload a real checkpoint and independently repeat frozen evaluation.
    one = summaries[0]
    state = torch.load(Path(one["path"]) / "resume.pt", weights_only=False)
    again, *_ = e.evaluate(state["learner"], one["task"]["map"], 70910, count=2, split=7)
    assert again == one["final"], "Checkpoint changed frozen predictions or rollout metrics"
    e.write(e.OUT / "check_runs.json", summaries)
    e.write(
        e.OUT / "checks.json",
        dict(
            passed=True,
            derivatives=derivative,
            paired_initialization=paired,
            smoke_runs=40,
            workers=30,
            frozen_evaluation_repeats_exactly_after_reload=True,
        ),
    )
    for row in summaries:
        path = Path(row["path"]).resolve()
        assert "tuning_raw" in path.parts and e.OUT in path.parents
        shutil.rmtree(path)
    print("ALL CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
