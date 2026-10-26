"""Prediction learning and detached motor reinforcement for all model conditions."""

import copy
import numpy as np
import torch
from .config import ModelConfig, PredictionSettings
from .encoders import InputSpikeCoder
from .alif import FastBellecCore
from .backbone import PredictionBackbone
from .motor import MotorConfig, SpikingMotorHead
from .mlp import LocalMLP, fingerprint, softmax, sigmoid, credit
from .plasticity import apply_core, eprop_vector
from .prediction import prediction_batch_loss


class Learner:
    def __init__(self, kind, rl, seed, hp, model_config, memory_config):
        self.kind = kind
        self.rl = rl
        self.seed = seed
        self.hp = hp
        self.baseline = np.zeros(50)
        self.replay = []
        self.replay_rng = np.random.default_rng(seed + 711113)
        self.episode = 0
        self.entropy = hp["entropy"]
        self.actor_lr = hp["actor_lr"]
        self.is_mlp = kind == "mlp"
        self.c = ModelConfig(seed=seed, **model_config)
        self.input_coder = InputSpikeCoder(self.c)
        self.input_channels = self.c.input_channels
        if self.is_mlp:
            self.actor = LocalMLP(seed, 2, 4, zero_output=True)
            self.predictor = LocalMLP(seed + 23917, 6, self.input_channels)
        else:
            self.settings = PredictionSettings(
                core_lr=hp["core_lr"],
                prediction_lr=hp["pred_lr"],
                core_step_cap=hp["core_cap"],
                prediction_epochs=hp["pred_epochs"],
                prediction_scale=hp["prediction_scale"],
            )
            self.model = PredictionBackbone(self.c, self.settings)
            self.input_coder = self.model.input_coder
            motor = MotorConfig(
                name=hp["name"],
                bias=hp["motor_bias"],
                amplitude=hp["motor_amplitude"],
                inhibition_scale=hp["inhibition_scale"],
                temperature=hp["motor_temperature"],
                temperature_end=hp["motor_temperature_end"],
                lr=hp["actor_lr"],
                lr_end=hp["actor_lr_end"],
                entropy=hp["entropy"],
                entropy_end=hp["entropy_end"],
                anneal_fraction=hp["anneal_fraction"],
                steps=60,
                bias_trainable=True,
            )
            self.head = SpikingMotorHead(self.c.neurons, motor)
            self.memory_params = copy.deepcopy(
                memory_config["trace" if kind == "trace" else "astro"]
            )
            self.memory_params["astro"].update(
                tau=hp["memory_tau"],
                threshold=hp["memory_threshold"],
                gthr=hp["memory_glutamate_threshold"],
            )
            self.memory_params["trace"].update(tau=hp["memory_tau"], gain=hp["memory_gain"])
            self.memory_params["gate"].update(gain=hp["gate_gain"], contrast=hp["memory_contrast"])
        self.initial_policy_hash = self.policy_hash()
        self.initial_predictor_hash = self.predictor_hash()
        self.initial_backbone_hash = self.backbone_hash()

    def parameters(self):
        if self.is_mlp:
            return [*self.actor.p.values(), *self.predictor.p.values()]
        return [
            *[p.detach().numpy() for p in self.model.rnn.parameters()],
            *[p.detach().numpy() for p in self.model.predictor.parameters()],
            *self.head.params.values(),
        ]

    def weights_hash(self):
        return fingerprint(self.parameters())

    def policy_hash(self):
        return fingerprint(self.actor.p.values() if self.is_mlp else self.head.params.values())

    def predictor_hash(self):
        return fingerprint(
            self.predictor.p.values()
            if self.is_mlp
            else [p.detach().numpy() for p in self.model.predictor.parameters()]
        )

    def backbone_hash(self):
        return (
            None
            if self.is_mlp
            else fingerprint([p.detach().numpy() for p in self.model.rnn.parameters()])
        )

    def reset(self):
        self.core = None if self.is_mlp else FastBellecCore(self.model.rnn)

    def schedule(self, ep, episodes):
        hp = self.hp
        f = (
            min(1.0, (ep - 1) / max(1.0, (episodes - 1) * hp["anneal_fraction"]))
            if self.rl
            else 0.0
        )
        self.entropy = hp["entropy"] + (hp["entropy_end"] - hp["entropy"]) * f
        self.actor_lr = hp["actor_lr"] + (hp["actor_lr_end"] - hp["actor_lr"]) * f
        if not self.is_mlp:
            self.head.temperature = (
                hp["motor_temperature"]
                + (hp["motor_temperature_end"] - hp["motor_temperature"]) * f
            )
            self.head.learning_rate = self.actor_lr

    def input_code(self, obs, previous=None):
        return self.input_coder.encode(obs, previous).numpy()

    def representation(self, obs, previous, training):
        current_input = self.input_code(obs, previous)
        if self.is_mlp:
            return dict(input_code=current_input)
        z, ei, er = self.core.observe(current_input, training, False)
        return dict(
            input_code=current_input, z=z, ei=ei if training else None, er=er if training else None
        )

    def prediction(self, obs, rep):
        if self.is_mlp:
            x = np.c_[np.repeat((2 * np.asarray(obs) - 1)[None], 4, axis=0), np.eye(4)]
            output, _ = self.predictor.forward(x)
            return sigmoid(output)
        with torch.no_grad():
            z = torch.tensor(rep["z"], dtype=torch.float32) * self.settings.prediction_scale
            return self.model.predictor(z.expand(4, -1), torch.arange(4)).numpy()

    def policy(self, obs, rep, gate, memory_on):
        if self.kind == "random":
            return np.full(4, 0.25), None, dict(motor_silent_fraction=0.0, mean_gate=0.0)
        if self.is_mlp:
            output, cache = self.actor.forward(2 * np.asarray(obs) - 1)
            return softmax(output), cache, dict(motor_silent_fraction=0.0, mean_gate=0.0)
        feature = (
            self.model.features(torch.tensor(rep["z"], dtype=torch.float32))
            .numpy()
            .astype(np.float64)
        )
        p, cache, diag = self.head.forward(
            feature, gate, self.memory_params["gate"]["gain"], float(memory_on)
        )
        diag["mean_gate"] = float(np.mean(gate))
        return p, cache, diag

    def update(self, records):
        result = dict(actor_gradient_norm=0.0, policy_core_gradient_norm=0.0)
        if self.rl and self.kind != "random":
            if self.is_mlp:
                delta, ret = credit(records, self.baseline, self.entropy)
                x = np.asarray([2 * r["obs"] - 1 for r in records])
                _, cache = self.actor.forward(x)
                result["actor_gradient_norm"] = self.actor.step(
                    self.actor.gradients(cache, delta), self.actor_lr
                )
            else:
                grads, ret = self.head.gradients(records, self.baseline, self.entropy, horizon=50)
                result["actor_gradient_norm"] = self.head.step(grads)
            self.baseline[: len(records)] = 0.95 * self.baseline[: len(records)] + 0.05 * ret
        if self.is_mlp:
            result.update(self._update_mlp_prediction(records))
        else:
            result.update(self._update_snn_prediction(records))
        self.episode += 1
        if not self.rl or self.kind == "random":
            assert self.policy_hash() == self.initial_policy_hash
        return result

    def _update_snn_prediction(self, records):
        m = self.model
        a = m
        s = self.settings
        z = torch.tensor(
            np.asarray([r["z"] for r in records]), dtype=torch.float32, requires_grad=True
        )
        m.prediction_optimizer.zero_grad(set_to_none=True)
        loss = prediction_batch_loss(m, z, records, s)
        loss.backward()
        signal = z.grad.detach().numpy()
        gradient = eprop_vector(
            signal,
            np.asarray([r["ei"] for r in records]),
            np.asarray([r["er"] for r in records]),
            a.rnn.rec_mask.numpy(),
        )
        before = np.concatenate([p.detach().numpy().ravel().copy() for p in a.rnn.parameters()])
        apply_core(a, m.core_optimizer, gradient, s.core_step_cap)
        after = np.concatenate([p.detach().numpy().ravel() for p in a.rnn.parameters()])
        torch.nn.utils.clip_grad_norm_(a.predictor.parameters(), 5.0)
        m.prediction_optimizer.step()
        self.replay.extend({k: r[k] for k in ("z", "action", "next_input", "obs")} for r in records)
        self.replay = self.replay[-6000:]
        for _ in range(s.prediction_epochs - 1):
            ids = self.replay_rng.integers(len(self.replay), size=128)
            batch = [self.replay[i] for i in ids]
            zz = torch.tensor(np.asarray([r["z"] for r in batch]), dtype=torch.float32)
            m.prediction_optimizer.zero_grad(set_to_none=True)
            extra = prediction_batch_loss(m, zz, batch, s)
            extra.backward()
            torch.nn.utils.clip_grad_norm_(a.predictor.parameters(), 5.0)
            m.prediction_optimizer.step()
        return dict(
            prediction_loss=float(loss.detach()),
            next_input_mse=float(loss.detach()),
            core_step_norm=float(np.linalg.norm(after - before)),
            prediction_core_gradient_norm=float(np.linalg.norm(gradient)),
        )

    def _update_mlp_prediction(self, records):
        self.replay.extend({k: r[k] for k in ("action", "next_input", "obs")} for r in records)
        self.replay = self.replay[-6000:]
        hp = self.hp
        first_loss = None
        norms = []
        for epoch in range(hp["pred_epochs"]):
            batch = (
                records
                if epoch == 0
                else [self.replay[i] for i in self.replay_rng.integers(len(self.replay), size=128)]
            )
            obs = np.asarray([r["obs"] for r in batch])
            actions = np.asarray([r["action"] for r in batch])
            x = np.c_[2 * obs - 1, np.eye(4)[actions]]
            raw, cache = self.predictor.forward(x)
            predicted = sigmoid(raw)
            target = np.asarray([r["next_input"] for r in batch])
            mse = np.square(predicted - target).mean()
            if epoch == 0:
                first_loss = float(mse)
            delta = -2 * (predicted - target) * predicted * (1 - predicted) / predicted.size
            gradients = self.predictor.gradients(cache, delta, hidden=epoch == 0)
            norms.append(
                self.predictor.step(
                    gradients, hp["pred_lr"], hp["mlp_pred_hidden_lr"], hp["mlp_pred_hidden_cap"]
                )
            )
        return dict(
            prediction_loss=first_loss,
            next_input_mse=first_loss,
            core_step_norm=0.0,
            prediction_core_gradient_norm=float(norms[0]),
        )

    def metadata(self):
        policy_count = sum(
            p.size for p in (self.actor.p.values() if self.is_mlp else self.head.params.values())
        )
        total = sum(p.size for p in self.parameters())
        if self.kind == "random":
            total -= policy_count
            policy_count = 0
        return dict(
            prediction_target="24-channel fixed RSNN input code before spike generation",
            prediction_objective="mean squared error only",
            memory_input="exact current RSNN input code",
            parameter_count=total,
            trainable_parameter_count=total - (policy_count if not self.rl else 0),
            policy_parameter_count=policy_count,
            policy_learning="uniform random; no policy learning"
            if self.kind == "random"
            else "feedforward local eligibility and fixed direct feedback"
            if self.is_mlp
            else "detached-reset LIF surrogate; no RL gradient to RSNN",
            prediction_learning="feedforward local eligibility and fixed direct feedback"
            if self.is_mlp
            else "Bellec ALIF e-prop plus supervised prediction heads",
            initial_policy_hash=self.initial_policy_hash,
            initial_predictor_hash=self.initial_predictor_hash,
            initial_backbone_hash=self.initial_backbone_hash,
            final_policy_hash=self.policy_hash(),
            final_predictor_hash=self.predictor_hash(),
            final_backbone_hash=self.backbone_hash(),
        )
