"""Predictive ALIF representation with calibrated currents and policy features."""

import numpy as np
import torch
from torch import nn
from .alif import BellecALIF, FastBellecCore
from .encoders import InputSpikeCoder
from .prediction import ObservationPredictor


class PredictionBackbone(nn.Module):
    def __init__(self, config, settings):
        super().__init__()
        self.c = config
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.rnn = BellecALIF(
                config.input_channels,
                config.neurons,
                steps=config.rsnn_steps,
                seed=config.seed,
                **config.bellec_options,
            )
            self.predictor = ObservationPredictor(
                config.neurons,
                config.input_channels,
                actions=config.n_actions,
                hidden=config.predictor_hidden,
                feature_scale=config.predictor_scale,
            )
        self.input_coder = InputSpikeCoder(config)
        self._calibrate_current()
        core = FastBellecCore(self.rnn)
        rng = np.random.default_rng(8100000 + config.seed)
        values = []
        for obs in rng.uniform(0, 1, (128, 2)).astype(np.float32):
            values.append(core.observe(self.input_coder.encode(obs, None).numpy(), False, False)[0])
        values = np.asarray(values)
        self.center = torch.tensor(values.mean(0), dtype=torch.float32)
        self.scale = torch.tensor(1 / np.maximum(values.std(0), 0.002), dtype=torch.float32)
        self.core_optimizer = torch.optim.Adam(self.rnn.parameters(), lr=settings.core_lr)
        self.prediction_optimizer = torch.optim.Adam(
            self.predictor.parameters(), lr=settings.prediction_lr
        )

    def _calibrate_current(self):
        net = self.rnn
        rng = np.random.default_rng(811113)
        samples = rng.uniform(0, 1, (128, 2)).astype(np.float32)
        probabilities = np.asarray([self.input_coder.encode(obs, None).numpy() for obs in samples])
        interval = 1000.0 / 10.0
        desired = net.threshold * (1 - net.alpha) / (1 - net.alpha**interval)
        bias = desired - probabilities.mean(0) @ net.win[:-1].detach().numpy()
        with torch.no_grad():
            net.win[-1].copy_(torch.tensor(bias, dtype=net.win.dtype))

    def features(self, rates):
        values = (rates - self.center) * self.scale
        return values / torch.sqrt(values.square().sum(-1, keepdim=True) + 1e-8)
