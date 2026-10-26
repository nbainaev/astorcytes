"""Fixed observation/action projection shared by RSNN input and memory."""

import numpy as np
import torch


class InputSpikeCoder:
    def __init__(self, config):
        rng = np.random.default_rng(config.seed + 10)
        self.w = torch.tensor(
            rng.normal(0, 2.5, (config.obs_dim + config.n_actions, config.input_channels)),
            dtype=torch.float32,
        )
        self.bias = torch.tensor(rng.normal(0, 0.65, config.input_channels), dtype=torch.float32)
        self.n_actions = config.n_actions
        self.action_gain = config.input_action_gain
        self.probability_scale = config.input_probability_scale

    def encode(self, obs, prev_action=None):
        obs = torch.as_tensor(obs, dtype=torch.float32).reshape(-1)
        action = torch.zeros(self.n_actions)
        if prev_action is not None:
            action[prev_action] = 1.0
        data = torch.cat((obs - 0.5, self.action_gain * action))
        return (
            (0.05 + 0.82 * torch.sigmoid(data @ self.w + self.bias)) * self.probability_scale
        ).clamp(0, 1)
