"""Stratified parameter search and independent validation seed stages."""

import copy, hashlib
import numpy as np
from . import engine as e


def candidates(kind, group, spec, n=18):
    tuning = e.read(e.CONFIGS / "tuning.json")
    m = tuning["motor_by_map"].get(spec["name"], tuning["motor"])
    prediction = tuning["prediction"]
    base = dict(
        name="c00",
        actor_lr=m["lr"],
        actor_lr_end=m.get("lr_end", m["lr"] / 10),
        entropy=m["entropy"],
        entropy_end=m["entropy_end"],
        anneal_fraction=m["anneal_fraction"],
        motor_bias=m["bias"],
        motor_amplitude=m["amplitude"],
        motor_temperature=m["temperature"],
        motor_temperature_end=m["temperature_end"],
        inhibition_scale=m["inhibition_scale"],
        pred_lr=prediction["prediction_lr"],
        core_lr=prediction["core_lr"],
        core_cap=prediction["core_step_cap"],
        pred_epochs=prediction["prediction_epochs"],
        prediction_scale=prediction["prediction_scale"],
        memory_tau=8.0 if kind == "trace" else 0.6,
        memory_gain=1.0,
        memory_threshold=0.03,
        gate_gain=2.0 if kind == "trace" else 1.0,
        memory_contrast=0.025,
        memory_glutamate_threshold=1.0,
        mlp_pred_hidden_lr=0.001,
        mlp_pred_hidden_cap=0.1,
    )
    if kind == "mlp":
        base.update(
            actor_lr=0.003,
            actor_lr_end=0.0003,
            entropy=0.02,
            entropy_end=0.001,
            pred_lr=0.003,
            pred_epochs=8,
        )
    if group == "no_rl":
        base.update(motor_temperature=0.03, motor_temperature_end=0.03)
    configs = [base]
    seed = int(hashlib.sha256(f"{kind}:{group}:{spec['name']}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    choices = dict(
        actor_lr=[0.0005, 0.0015, 0.003, 0.006, 0.01],
        entropy=[0.005, 0.015, 0.03, 0.05],
        entropy_end=[0.0001, 0.0005, 0.002],
        anneal_fraction=[0.5, 2 / 3, 1.0],
        motor_bias=[1.23, 1.45, 1.7],
        motor_amplitude=[0.75, 1.0],
        motor_temperature=[0.02, 0.03, 0.05],
        motor_temperature_end=[0.002, 0.003, 0.0075, 0.015],
        inhibition_scale=[0.1, 0.25, 0.5, 1.0],
        pred_lr=[0.0003, 0.001, 0.003, 0.01],
        core_lr=[0.0001, 0.0003, 0.0006],
        core_cap=[0.000025, 0.00005, 0.0001],
        pred_epochs=[4, 8, 16],
        prediction_scale=[25.0, 50.0, 75.0],
        memory_tau=[2.0, 4.0, 8.0, 16.0, 32.0] if kind == "trace" else [0.3, 0.6, 1.2, 2.4],
        memory_gain=[0.1, 0.25, 0.5, 1.0, 2.0],
        memory_threshold=[0.03, 0.1, 0.2, 0.4],
        memory_glutamate_threshold=[0.3, 0.6, 1.0, 1.5, 2.0],
        gate_gain=[0.5, 1.0, 2.0, 3.0],
        memory_contrast=[0.005, 0.01, 0.025, 0.05, 0.1],
        mlp_pred_hidden_lr=[0.0003, 0.001, 0.003],
        mlp_pred_hidden_cap=[0.025, 0.1, 0.25],
    )
    # Stratified permutations cover every prespecified level before repetition.
    columns = {
        k: np.asarray((values * ((n + len(values) - 2) // len(values)))[: n - 1])[
            rng.permutation(n - 1)
        ]
        for k, values in choices.items()
    }
    for i in range(n - 1):
        c = copy.deepcopy(base)
        c.update({k: (v[i].item() if hasattr(v[i], "item") else v[i]) for k, v in columns.items()})
        c["name"] = f"c{i + 1:02d}"
        c["actor_lr_end"] = c["actor_lr"] / 10
        if kind == "mlp" and c["actor_lr"] > 0.006:
            c["actor_lr"] = 0.006
            c["actor_lr_end"] = 0.0006
        configs.append(c)
    return configs


def rank(rows, keep):
    groups = {}
    for row in rows:
        t = row["task"]
        key = (t["group"], t["kind"], t["map"]["name"])
        groups.setdefault(key, {}).setdefault(t["hp"]["name"], []).append(row)
    result = {}
    ranking = []
    for key, configs in groups.items():
        choices = []
        for name, rr in configs.items():
            values = np.array([r["objective"] for r in rr])
            sd = float(values.std(ddof=1)) if len(rr) > 1 else 0.0
            choices.append(
                dict(
                    name=name,
                    hp=rr[0]["task"]["hp"],
                    mean=float(values.mean()),
                    sd=sd,
                    score=float(values.mean() - 0.25 * sd),
                    seeds=[r["task"]["seed"] for r in rr],
                )
            )
        choices.sort(key=lambda x: (-x["score"], x["name"]))
        result[key] = [c["hp"] for c in choices[:keep]]
        ranking.append(dict(group=key[0], kind=key[1], map=key[2], candidates=choices))
    return result, ranking
