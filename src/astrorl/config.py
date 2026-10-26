"""Architecture and optimizer configuration for the navigation models."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    seed: int = 42
    obs_dim: int = 2
    n_actions: int = 4
    neurons: int = 100
    input_channels: int = 24
    rsnn_steps: int = 128
    input_action_gain: float = 0.0
    input_probability_scale: float = 1.0
    predictor_hidden: int = 128
    predictor_scale: float = 1.0
    bellec_options: dict = field(
        default_factory=lambda: dict(
            input_scale=0.05, recurrent_scale=0.1, tau_out_ms=40.0, input_bias=True
        )
    )


@dataclass
class PredictionSettings:
    core_lr: float
    prediction_lr: float
    core_step_cap: float
    prediction_epochs: int
    prediction_scale: float
