import json

from astrocites.logs import FileLogger


def test_file_logger_separates_experiment_parameters_and_metrics(tmp_path):
    logger = FileLogger(output_dir=tmp_path, experiment_name="run")
    logger.start_experiment("experiment_1")
    logger.log_params({"seed": 1})
    logger.log_metrics({"success": 1}, step=1)
    logger.start_experiment("experiment_2")
    logger.log_params({"seed": 2})
    logger.log_metrics({"success": 0}, step=1)
    logger.finish()

    exp1 = tmp_path / "run" / "experiment_1"
    exp2 = tmp_path / "run" / "experiment_2"
    assert json.loads((exp1 / "params.json").read_text()) == {"seed": 1}
    assert json.loads((exp2 / "params.json").read_text()) == {"seed": 2}
    assert '"success": 1' in (exp1 / "metrics.jsonl").read_text()
    assert '"success": 0' in (exp2 / "metrics.jsonl").read_text()
