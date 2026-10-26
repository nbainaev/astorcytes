"""Run the article study with configured conditions or a parameter search."""

from . import engine as e
from .configuration import study_configuration
from .jobs import make_tasks, run_pool


def _configuration(mode):
    configuration = study_configuration()
    configuration["execution"] = mode
    configuration["architecture"] = e.read(e.CONFIGS / "model.json")
    configuration["memory"] = e.read(e.CONFIGS / "memory.json")
    if mode == "tune":
        configuration["search_settings"] = e.read(e.CONFIGS / "tuning.json")
    if (e.OUT / "protocol.json").exists():
        assert e.read(e.OUT / "protocol.json") == configuration, (
            "Output directory belongs to another study configuration."
        )
    e.write(e.OUT / "protocol.json", configuration)
    return configuration


def _final(configuration, selected, workers):
    mapping = {(r["group"], r["kind"], r["map"]): [r["hp"]] for r in selected}
    expected = {(g, k, s["name"]) for g in e.GROUPS for k in e.KINDS for s in configuration["maps"]}
    assert len(selected) == 40 and set(mapping) == expected
    path = e.OUT / "locked_parameters.json"
    if path.exists():
        assert e.read(path) == selected, (
            "Condition parameters changed; use a different output directory."
        )
    e.write(path, selected)
    tasks = make_tasks(
        configuration["maps"],
        mapping,
        "final",
        configuration["training_episodes"],
        configuration["final_seeds"],
    )
    rows = run_pool(tasks, "final", workers)
    assert len(rows) == 400
    e.write(
        e.OUT / "status.json", dict(stage="plotting", completed=400, total=400, workers=workers)
    )
    from .analysis import report

    report()
    e.write(
        e.OUT / "status.json",
        dict(stage="complete", completed=400, total=400, workers=workers, report="REPORT_RU.md"),
    )


def run(workers=30):
    configuration = _configuration("run")
    _final(configuration, e.read(e.CONFIGS / "conditions.json"), workers)


def tune(workers=30):
    from .tuning import candidates, rank

    configuration = _configuration("tune")
    specs = configuration["maps"]
    settings = e.read(e.CONFIGS / "tuning.json")
    mapping = {
        (g, k, s["name"]): candidates(k, g, s) for g in e.GROUPS for k in e.KINDS for s in specs
    }
    e.write(
        e.OUT / "candidates.json",
        [dict(group=g, kind=k, map=m, configs=c) for (g, k, m), c in mapping.items()],
    )
    for stage in settings["stages"]:
        rows = run_pool(
            make_tasks(specs, mapping, stage["name"], stage["episodes"], stage["seeds"]),
            stage["name"],
            workers,
        )
        mapping, ranking = rank(rows, stage["keep"])
        e.write(e.OUT / f"{stage['name']}_ranking.json", ranking)
    selected = [dict(group=g, kind=k, map=m, hp=values[0]) for (g, k, m), values in mapping.items()]
    _final(configuration, selected, workers)
