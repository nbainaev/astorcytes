import os
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ExperimentLogger(ABC):
    @property
    def active_exp_dir(self) -> Path:
        return getattr(self, 'exp_dir', None)

    @abstractmethod
    def log_params(self, params: dict):
        ...

    @abstractmethod
    def log_metrics(self, metrics: dict, step: int = None):
        ...

    @abstractmethod
    def log_artifact(self, filepath: str):
        ...

    @abstractmethod
    def finish(self):
        ...

    def start_experiment(self, name: str):
        pass


class FileLogger(ExperimentLogger):
    def __init__(self, output_dir: str, experiment_name: str = None, **kwargs):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name or f"experiment_{timestamp}"
        self.exp_dir = self.output_dir / self.experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self._active_exp_dir = self.exp_dir
        self.params_path = self.exp_dir / "params.json"
        self.metrics_path = self.exp_dir / "metrics.jsonl"
        self._metrics_file = open(self.metrics_path, 'a')
        self._params = {}
        self._metrics = []
        logger.info(f"FileLogger: logging to {self.exp_dir}")

    def log_params(self, params: dict):
        self._params.update(params)
        with open(self.params_path, 'w') as f:
            json.dump(self._params, f, indent=2, default=str)

    def log_metrics(self, metrics: dict, step: int = None):
        entry = {}
        if step is not None:
            entry["step"] = step
        entry.update(metrics)
        entry["timestamp"] = datetime.now().isoformat()
        self._metrics_file.write(json.dumps(entry, default=str) + "\n")
        self._metrics_file.flush()

    def log_artifact(self, filepath: str):
        import shutil
        src = Path(filepath)
        dst = self._active_exp_dir / src.name
        if src.exists():
            shutil.copy2(src, dst)

    def start_experiment(self, name: str):
        if not self._metrics_file.closed:
            self._metrics_file.close()
        self._active_exp_dir = self.exp_dir / name
        self._active_exp_dir.mkdir(parents=True, exist_ok=True)
        self.params_path = self._active_exp_dir / "params.json"
        self.metrics_path = self._active_exp_dir / "metrics.jsonl"
        self._metrics_file = open(self.metrics_path, "a")
        self._params = {}

    @property
    def active_exp_dir(self) -> Path:
        return self._active_exp_dir

    def finish(self):
        self._metrics_file.close()


class CometLogger(ExperimentLogger):
    def __init__(self, output_dir: str = None, experiment_name: str = None,
                 workspace: str = None, project_name: str = None, **kwargs):
        try:
            import comet_ml
        except ImportError:
            raise ImportError("comet-ml is required for CometLogger. Install with: pip install comet-ml")
        self._comet_ml = comet_ml
        self._workspace = workspace
        self._project_name = project_name
        self._output_dir = output_dir
        if output_dir is not None:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = experiment_name or f"experiment_{timestamp}"
            self.exp_dir = self.output_dir / self.experiment_name
            self.exp_dir.mkdir(parents=True, exist_ok=True)
            self._active_exp_dir = self.exp_dir
        else:
            self.exp_dir = None
            self._active_exp_dir = None
        self._experiments = []
        self._active_experiment = self._comet_ml.Experiment(
            workspace=workspace,
            project_name=project_name,
            experiment_name=experiment_name,
            **kwargs,
        )
        self._experiments.append(self._active_experiment)

    def log_params(self, params: dict):
        self._active_experiment.log_parameters(params)

    def log_metrics(self, metrics: dict, step: int = None):
        self._active_experiment.log_metrics(metrics, step=step)

    def log_artifact(self, filepath: str):
        self._active_experiment.log_asset(filepath)

    @property
    def active_exp_dir(self) -> Path:
        return self._active_exp_dir

    def start_experiment(self, name: str):
        if self._active_experiment is not None:
            self._active_experiment.end()
        if self.exp_dir is not None:
            self._active_exp_dir = self.exp_dir / name
            self._active_exp_dir.mkdir(parents=True, exist_ok=True)
        self._active_experiment = self._comet_ml.Experiment(
            workspace=self._workspace,
            project_name=self._project_name,
            experiment_name=name,
        )
        self._experiments.append(self._active_experiment)

    def finish(self):
        self._active_experiment.end()


class AimLogger(ExperimentLogger):
    def __init__(self, output_dir: str = None, experiment_name: str = None,
                 repo: str = None, **kwargs):
        try:
            from aim import Run
        except ImportError:
            raise ImportError("aim is required for AimLogger. Install with: pip install aim")
        self._aim = Run
        self._repo = repo
        self._output_dir = output_dir
        if output_dir is not None:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = experiment_name or f"experiment_{timestamp}"
            self.exp_dir = self.output_dir / self.experiment_name
            self.exp_dir.mkdir(parents=True, exist_ok=True)
            self._active_exp_dir = self.exp_dir
        else:
            self.exp_dir = None
            self._active_exp_dir = None
        self._runs = []
        self._active_run = self._aim(repo=repo, experiment=experiment_name, **kwargs)
        self._runs.append(self._active_run)

    def log_params(self, params: dict):
        for key, value in params.items():
            self._active_run[("params", key)] = value

    def log_metrics(self, metrics: dict, step: int = None):
        for key, value in metrics.items():
            self._active_run.track(value, name=key, step=step)

    def log_artifact(self, filepath: str):
        self._active_run.track_artifact(filepath)

    @property
    def active_exp_dir(self) -> Path:
        return self._active_exp_dir

    def start_experiment(self, name: str):
        if self._active_run is not None:
            self._active_run.close()
        if self.exp_dir is not None:
            self._active_exp_dir = self.exp_dir / name
            self._active_exp_dir.mkdir(parents=True, exist_ok=True)
        self._active_run = self._aim(repo=self._repo, experiment=name)
        self._runs.append(self._active_run)

    def finish(self):
        self._active_run.close()


def get_logger(logger_type: str = "file", **kwargs) -> ExperimentLogger:
    loggers = {
        "file": FileLogger,
        "comet": CometLogger,
        "aim": AimLogger,
    }
    logger_cls = loggers.get(logger_type.lower())
    if logger_cls is None:
        raise ValueError(f"Unknown logger type: {logger_type}. Choose from: {list(loggers.keys())}")
    return logger_cls(**kwargs)