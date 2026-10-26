"""Configuration consistency, task budgets and dependency boundaries."""

import ast
import csv
import json
import importlib
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def test_experiment_module_interfaces(self):
        engine = importlib.import_module("experiments.engine")
        for name in ("engine", "jobs", "runner", "tuning", "analysis", "checks"):
            module = importlib.import_module(f"experiments.{name}")
            for node in ast.walk(ast.parse(Path(module.__file__).read_text())):
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "e"
                ):
                    self.assertTrue(hasattr(engine, node.attr), f"{name}: engine.{node.attr}")

    def test_condition_table(self):
        selected = json.loads((ROOT / "configs/conditions.json").read_text())
        maps = json.loads((ROOT / "configs/maps.json").read_text())["maps"]
        expected = {
            (g, k, s["name"])
            for g in ("no_rl", "rl")
            for k in ("mlp", "random", "noastro", "astro", "trace")
            for s in maps
        }
        keyed = {(r["group"], r["kind"], r["map"]): r["hp"] for r in selected}
        self.assertEqual(len(selected), 40)
        self.assertEqual(set(keyed), expected)
        with (ROOT / "configs/conditions.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 40)
        for row in rows:
            hp = keyed[(row["group"], row["kind"], row["map"])]
            self.assertEqual(set(row), set(hp) | {"group", "kind", "map"})
            for key, value in hp.items():
                parsed = float(row[key]) if isinstance(value, (int, float)) else row[key]
                self.assertEqual(parsed, value)

    def test_task_budgets(self):
        from experiments.configuration import study_configuration
        from experiments.jobs import make_tasks

        config = study_configuration()
        conditions = json.loads((ROOT / "configs/conditions.json").read_text())
        mapping = {(r["group"], r["kind"], r["map"]): [r["hp"]] for r in conditions}
        tasks = make_tasks(
            config["maps"], mapping, "final", config["training_episodes"], config["final_seeds"]
        )
        self.assertEqual(len(tasks), 400)
        self.assertEqual(
            len({(t["group"], t["kind"], t["map"]["name"], t["seed"]) for t in tasks}), 400
        )
        self.assertTrue(all(t["episodes"] == 600 and t["eval_count"] == 100 for t in tasks))

    def test_model_package_is_independent(self):
        for file in (ROOT / "src/astrorl").glob("*.py"):
            tree = ast.parse(file.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertFalse(
                        any(a.name.split(".")[0] == "experiments" for a in node.names), file
                    )
                if isinstance(node, ast.ImportFrom):
                    self.assertNotEqual((node.module or "").split(".")[0], "experiments", file)


if __name__ == "__main__":
    unittest.main()
