"""A single action-conditioned prediction of the next RSNN input observation."""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ObservationPredictor(nn.Module):
    def __init__(self, neurons, input_channels, actions=4, hidden=128, feature_scale=1.0):
        super().__init__()
        self.actions = actions
        self.feature_scale = feature_scale
        self.trunk = nn.Sequential(
            nn.Linear(neurons + actions, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.next_input = nn.Linear(hidden, input_channels)

    def forward(self, state, action):
        action = F.one_hot(torch.as_tensor(action, dtype=torch.long), self.actions).float()
        features = torch.cat((self.feature_scale * state, action), -1)
        return torch.sigmoid(self.next_input(self.trunk(features)))


def prediction_loss(model, states, actions, next_input):
    return F.mse_loss(model.predictor(states, actions), next_input)


def prediction_batch_loss(model, states, records, settings):
    actions = torch.tensor(np.asarray([record["action"] for record in records]))
    target = torch.tensor(
        np.asarray([record["next_input"] for record in records]), dtype=torch.float32
    )
    return prediction_loss(model, states * settings.prediction_scale, actions, target)
