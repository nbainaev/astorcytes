"""Study configuration and output locations."""

from pathlib import Path
import os
from astrorl.logging import read_json

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (
    ROOT / "configs" if (ROOT / "configs").exists() else Path(__file__).resolve().parent / "configs"
)
OUT = Path(os.environ.get("ASTRORL_OUTPUT_DIR", "results")).resolve()
KINDS = ("mlp", "random", "noastro", "astro", "trace")
GROUPS = ("no_rl", "rl")


def study_configuration():
    study = read_json(CONFIGS / "study.json")
    study["maps"] = read_json(CONFIGS / "maps.json")["maps"]
    assert (
        study["training_episodes"],
        study["max_steps"],
        study["evaluation_episodes_per_mode"],
    ) == (600, 50, 100)
    assert study["final_seeds"] == list(range(62000, 62010))
    return study
