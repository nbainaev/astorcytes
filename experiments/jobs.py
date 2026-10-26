"""Deterministic trial construction, caching and parallel job execution."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import hashlib, json, shutil, traceback
from . import engine as e


def task_id(task):
    return hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()[:24]


def run_pool(tasks, stage, workers=30):
    cache = e.OUT / "index" / stage
    cache.mkdir(parents=True, exist_ok=True)
    done = []
    pending = []
    for task in tasks:
        path = cache / f"{task_id(task)}.json"
        if path.exists():
            row = e.read(path)
            assert row["task"] == task
            done.append(row)
        else:
            pending.append(task)
    e.write(
        e.OUT / "status.json",
        dict(stage=stage, completed=len(done), total=len(tasks), workers=workers),
    )
    failures = []
    if pending:
        pending.sort(key=task_id)  # interleave model/map/group strata deterministically
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(e.case, t): t for t in pending}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    row = future.result()
                    e.write(cache / f"{task_id(task)}.json", row)
                    done.append(row)
                    if task["stage"] != "final":
                        path = Path(row["path"]).resolve()
                        assert e.OUT.resolve() in path.parents and "tuning_raw" in path.parts
                        shutil.rmtree(path)  # authorized raw cleanup after compact metrics commit
                    print(
                        stage,
                        len(done),
                        len(tasks),
                        task["group"],
                        task["kind"],
                        task["map"]["name"],
                        task["hp"]["name"],
                        task["seed"],
                        round(row["objective"], 4),
                        flush=True,
                    )
                except Exception:
                    error = dict(task=task, traceback=traceback.format_exc())
                    failures.append(error)
                    e.write(e.OUT / "errors" / f"{stage}_{task_id(task)}.json", error)
                    print(error["traceback"], flush=True)
                e.write(
                    e.OUT / "status.json",
                    dict(
                        stage=stage,
                        completed=len(done),
                        total=len(tasks),
                        failed=len(failures),
                        workers=workers,
                    ),
                )
    if failures:
        raise RuntimeError(
            f"{len(failures)} failed {stage} runs; see errors/. Successful runs are cached."
        )
    e.write(e.OUT / f"{stage}_index.json", done)
    return done


def make_tasks(specs, mapping, stage, episodes, seeds):
    return [
        dict(
            map=spec,
            kind=kind,
            group=group,
            hp=hp,
            stage=stage,
            episodes=episodes,
            seed=seed,
            eval_count=100,
        )
        for group in e.GROUPS
        for kind in e.KINDS
        for spec in specs
        for hp in mapping[(group, kind, spec["name"])]
        for seed in seeds
    ]
