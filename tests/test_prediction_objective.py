"""Single next-input MSE and identity of the code received by the ALIF core."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import unittest
import numpy as np
import torch
from astrorl.learner import Learner
from astrorl.prediction import ObservationPredictor, prediction_loss, prediction_batch_loss
from astrorl.mlp import sigmoid

ROOT = Path(__file__).resolve().parents[1]


class PredictionObjectiveTests(unittest.TestCase):
    def test_only_one_output_and_mse(self):
        net = ObservationPredictor(100, 24)
        model = SimpleNamespace(predictor=net)
        states = torch.randn(5, 100, requires_grad=True)
        actions = torch.tensor([0, 1, 2, 3, 0])
        target = torch.rand(5, 24)
        predicted = net(states, actions)
        self.assertEqual(tuple(predicted.shape), (5, 24))
        self.assertEqual(set(dict(net.named_children())), {"trunk", "next_input"})
        loss = prediction_loss(model, states, actions, target)
        torch.testing.assert_close(loss, ((predicted - target) ** 2).mean(), rtol=0, atol=0)
        loss.backward()
        self.assertGreater(float(states.grad.norm()), 0.0)

    def test_batch_has_no_coordinate_or_auxiliary_targets(self):
        model = SimpleNamespace(predictor=ObservationPredictor(100, 24))
        states = torch.randn(2, 100)
        records = [
            dict(action=0, next_input=np.full(24, 0.2, np.float32)),
            dict(action=2, next_input=np.full(24, 0.7, np.float32)),
        ]
        loss = prediction_batch_loss(model, states, records, SimpleNamespace(prediction_scale=2.0))
        expected = prediction_loss(
            model,
            states * 2,
            torch.tensor([0, 2]),
            torch.tensor(np.stack([r["next_input"] for r in records])),
        )
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)

    def test_mlp_output_derivative_averages_all_channels(self):
        rng = np.random.default_rng(13)
        raw = rng.normal(size=(5, 24))
        target = rng.uniform(size=raw.shape)
        value = torch.tensor(raw, requires_grad=True)
        ((torch.sigmoid(value) - torch.tensor(target)) ** 2).mean().backward()
        p = sigmoid(raw)
        expected = 2 * (p - target) * p * (1 - p) / p.size
        np.testing.assert_allclose(value.grad.numpy(), expected, rtol=0, atol=1e-15)

    def test_projection_is_fixed_and_core_receives_the_same_code(self):
        torch.set_num_threads(1)
        read = lambda name: json.loads((ROOT / "configs" / name).read_text())
        hp = read("conditions.json")[0]["hp"]
        learner = Learner("noastro", False, 62000, hp, read("model.json"), read("memory.json"))
        self.assertFalse(learner.input_coder.w.requires_grad)
        learner.reset()
        with patch.object(
            learner.core, "observe", return_value=(np.zeros(100), None, None)
        ) as observe:
            representation = learner.representation(np.array([0.2, 0.7], np.float32), 1, False)
        np.testing.assert_array_equal(observe.call_args.args[0], representation["input_code"])
        self.assertEqual(representation["input_code"].shape, (24,))
        self.assertIs(learner.input_coder, learner.model.input_coder)
