"""Model unit checks on fixed inputs, without environment rollouts."""

import copy
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from astrorl.learner import Learner
from astrorl.memory import FamiliarityMemory, NeuronalTraceBank
from astrorl.mlp import LocalMLP, softmax
from astrorl.motor import MotorConfig, SpikingMotorHead

ROOT = Path(__file__).resolve().parents[1]


class LocalLearningTests(unittest.TestCase):
    def test_direct_feedback_gradient(self):
        net = LocalMLP(41, 6, 20)
        rng = np.random.default_rng(17)
        x = rng.normal(size=(9, 6))
        delta = rng.normal(size=(9, 20))
        _, cache = net.forward(x)
        analytic = net.gradients(cache, delta)
        weights = {key: torch.tensor(value, requires_grad=True) for key, value in net.p.items()}
        xx, h1, h2, _, _ = cache
        loss = (
            torch.relu(torch.tensor(xx) @ weights["w1"] + weights["b1"])
            * torch.tensor(delta @ net.B1.T)
        ).sum()
        loss += (
            torch.relu(torch.tensor(h1) @ weights["w2"] + weights["b2"])
            * torch.tensor(delta @ net.B2.T)
        ).sum()
        loss += ((torch.tensor(h2) @ weights["w3"] + weights["b3"]) * torch.tensor(delta)).sum()
        loss.backward()
        for key, gradient in analytic.items():
            np.testing.assert_allclose(gradient, weights[key].grad.numpy(), atol=1e-12, rtol=0)
        altered = copy.deepcopy(net)
        altered.p["w3"] *= 19
        altered.p["w2"] *= -7
        alternate = altered.gradients(cache, delta)
        for key in ("w1", "b1", "w2", "b2"):
            np.testing.assert_array_equal(analytic[key], alternate[key])

    def test_uniform_policy_and_motor_update(self):
        mlp = LocalMLP(3, 2, 4, zero_output=True)
        logits, _ = mlp.forward(np.array([0.1, -0.2]))
        np.testing.assert_array_equal(softmax(logits), np.full(4, 0.25))
        head = SpikingMotorHead(6, MotorConfig())
        p, cache, _ = head.forward(np.ones(6), np.zeros(4), 1.0, 0.0)
        np.testing.assert_array_equal(p, np.full(4, 0.25))
        record = dict(probability=p, head_cache=cache, action=0, reward=1.0)
        gradients, returns = head.gradients([record], np.zeros(50), 0.01)
        self.assertEqual(returns[0], 1.0)
        self.assertGreater(head.step(gradients), 0.0)

    def test_trace_write_and_read(self):
        bank = NeuronalTraceBank(2, tau_s=2.0, gain=0.5)
        bank.advance(np.array([1.0, 0.0]), 1.0)
        expected = np.array([0.5 * (1 - np.exp(-0.5)), 0.0])
        np.testing.assert_allclose(bank.active_memory(), expected)
        before = bank.snapshot()
        bank.familiarity(np.eye(2))
        for a, b in zip(before, bank.snapshot()):
            np.testing.assert_array_equal(a, b)
        bank.reset()
        np.testing.assert_array_equal(bank.active_memory(), np.zeros(2))

    def test_memory_queries_preserve_state(self):
        config = json.loads((ROOT / "configs/memory.json").read_text())
        for kind in ("astro", "trace"):
            memory = FamiliarityMemory(4, kind, config[kind])
            memory.write(np.array([1.0, 0.2, 0.0, 0.0]), 711)
            before = memory.active().copy()
            memory.familiarity(np.eye(4))
            np.testing.assert_array_equal(memory.active(), before)


class BackboneTests(unittest.TestCase):
    def test_snn_conditions_share_architecture_and_initial_weights(self):
        torch.set_num_threads(1)
        read = lambda name: json.loads((ROOT / "configs" / name).read_text())
        hp = next(
            c["hp"]
            for c in read("conditions.json")
            if c["kind"] == "noastro" and c["group"] == "rl"
        )
        models = [
            Learner(kind, True, 62000, hp, read("model.json"), read("memory.json"))
            for kind in ("noastro", "astro", "trace")
        ]
        self.assertEqual(len({m.initial_backbone_hash for m in models}), 1)
        self.assertEqual(len({m.initial_predictor_hash for m in models}), 1)
        self.assertEqual(len({m.initial_policy_hash for m in models}), 1)
        for model in models:
            self.assertEqual(model.head.params["wo"].shape, (4, 100))
            before = model.backbone_hash()
            model.head.step({key: np.ones_like(value) for key, value in model.head.params.items()})
            self.assertEqual(model.backbone_hash(), before)


if __name__ == "__main__":
    unittest.main()
