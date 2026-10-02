import sys

from astrocites import run


def test_cli_experiment_count_and_cycle_overrides_update_nested_config(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "astrocites",
            "-c", "configs/default.yaml",
            "--num-experiments", "2",
            "--num-cycles", "5",
        ],
    )
    monkeypatch.setattr(run.names_generator, "generate_name", lambda: "test-run")
    monkeypatch.setattr(run, "get_logger", lambda **_kwargs: object())
    monkeypatch.setattr(
        run,
        "run_experiment",
        lambda config, logger: captured.update(config=config, logger=logger),
    )

    run.main()

    assert captured["config"]["experiment"]["num_experiments"] == 2
    assert captured["config"]["experiment"]["num_cycles"] == 5
