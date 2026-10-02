import torch
import numpy as np
from astrocites.learning import NoOp, WeightDependentPostPre


class Connection(torch.nn.Module):
    def __init__(self, source, target, impulse_amplitude=0.5, impulse_amplitude_2=0.5,
                 impulse_length=40, impulse_shape_factor=0.9, invert=False,
                 update_rule=NoOp, w=None, nu=None, wmin=0, wmax=1,
                 weight_decay=0, post_spike_weight_decay=0,
                 diagonal_connection=False, enforce_post_stdp_bounds=True,
                 apply_structural_mask_during_stdp=True,
                 clamp_initial_weights=True,
                 baseline_decay=0.01, policy_mix_beta=0.5,
                 gamma=0.99, trace_decay=0.95, temperature=1.0, **kwargs):
        super().__init__()
        self.source = source
        self.target = target
        self.wmin = wmin
        self.wmax = wmax
        self.diagonal_connection = bool(diagonal_connection)
        self.enforce_post_stdp_bounds = bool(enforce_post_stdp_bounds)
        self.apply_structural_mask_during_stdp = bool(apply_structural_mask_during_stdp)
        self.clamp_initial_weights = bool(clamp_initial_weights)
        if w is None:
            if self.wmin == -np.inf or self.wmax == np.inf:
                w = torch.clamp(torch.rand(source.n, target.n), self.wmin, self.wmax)
            else:
                w = self.wmin + torch.rand(source.n, target.n) * (self.wmax - self.wmin)
        else:
            if self.clamp_initial_weights and (
                self.wmin != -np.inf or self.wmax != np.inf
            ):
                w = torch.clamp(w, self.wmin, self.wmax)
        self.w = torch.nn.Parameter(w, False)
        self.update_rule = update_rule(self, nu=nu, weight_decay=weight_decay,
                                        post_spike_weight_decay=post_spike_weight_decay)
        self.impulse_amplitude = impulse_amplitude
        self.impulse_amplitude_2 = impulse_amplitude_2
        self.impulse_length = impulse_length
        self.impulse_shape_factor = impulse_shape_factor
        self.invert = invert
        self.register_buffer("a_pre", torch.zeros(source.n))
        self.register_buffer("impulse_state", torch.zeros(source.n))
        self.register_buffer("eligibility", torch.zeros(source.n, target.n))
        # impulse_curve is a pure function of the integer impulse state (0..impulse_length),
        # so evaluate it once for every possible state and gather at runtime
        states = torch.arange(int(impulse_length) + 1, dtype=torch.float)
        saved_state = self._buffers["impulse_state"]
        self._buffers["impulse_state"] = states
        try:
            lut = self.impulse_curve().clone()
        finally:
            self._buffers["impulse_state"] = saved_state
        self.register_buffer("impulse_lut", lut)
        self.baseline_decay = baseline_decay
        self.policy_mix_beta = policy_mix_beta
        self.gamma = gamma
        self.trace_decay = trace_decay
        self.temperature = temperature
        self.running_baseline = 0.0
        self.reward_accumulator = 0.0
        self.discount = 1.0

    def impulse_curve(self):
        k = self.impulse_shape_factor
        if self.invert:
            impulse_value_2 = self.impulse_amplitude / (self.impulse_length * k - 1)
            impulse_value_1 = self.impulse_amplitude / (self.impulse_length * (1 - k))
            impulse_bias = 2 * self.impulse_amplitude * (self.impulse_state > (self.impulse_length * (1 - k) + 0.5)).float() * (self.impulse_state <= (self.impulse_length * (1 - k) + 1.5)).float()
            impulse = (-impulse_value_1) * (self.impulse_state > 0).float() * (self.impulse_state <= (self.impulse_length * (1 - k) + 0.5)).float() + (-impulse_value_2) * (self.impulse_state > (self.impulse_length * (1 - k) + 1.5)).float() + impulse_bias
            return impulse
        else:
            impulse_value_2 = self.impulse_amplitude / (self.impulse_length * k - 1)
            impulse_value_1 = self.impulse_amplitude_2 / (self.impulse_length * (1 - k))
            impulse_bias = (self.impulse_amplitude + self.impulse_amplitude_2) * (self.impulse_state >= (self.impulse_length * k)).float() * (self.impulse_state < (self.impulse_length * k + 1)).float()
            impulse = (impulse_value_1) * (self.impulse_state > (self.impulse_length * k)).float() + (impulse_value_2) * (self.impulse_state > 0).float() * (self.impulse_state < (self.impulse_length * k)).float() - impulse_bias
            return impulse

    def update_impulse_state(self, s):
        st = self.impulse_state
        active = st > 0
        st += active.float()
        if s.dim() == 1:
            s = s.unsqueeze(0)
        # new impulses only start where none is in progress; s is never mutated here
        st += (active == 0).float() * s.float().view(-1)
        impulse = self.impulse_lut[st.long()]
        st *= (st < self.impulse_length).float()
        return impulse

    def compute(self, s: torch.Tensor, active_source=None, optimized=True) -> torch.Tensor:
        impulse = self.update_impulse_state(s)
        self.a_pre += impulse
        self.a_pre *= (self.impulse_state > 0).float()
        if optimized and self.diagonal_connection:
            a_post = self.a_pre * torch.diagonal(self.w)
        elif optimized and active_source is not None:
            # Navigation input is restricted to the current state. Keeping the
            # dense W tensor while multiplying its active row is exactly the
            # dense product under that input invariant.
            a_post = self.a_pre[int(active_source)] * self.w[int(active_source), :]
        else:
            a_post = self.a_pre @ self.w
        return a_post.view(1, *self.target.shape)

    def update(self, **kwargs):
        self.update_rule.update(**kwargs)

    def accumulate_trace(self, s_idx, adj_positions, grad_slice, step_reward):
        """Accumulate one step of the REINFORCE eligibility trace.

        Parameters
        ----------
        s_idx : int
            Index of the current (source) position.
        adj_positions : list[int]
            Candidate target positions, aligned with ``grad_slice``.
        grad_slice : torch.Tensor
            Precomputed closed-form eligibility gradient ``d log pi / d w[s, a]``
            for each candidate action ``a`` (length ``len(adj_positions)``),
            produced by :func:`astrocites.surrogate.eligibility_gradient`.
        step_reward : float
            Reward received for the action taken this step.
        """
        self.eligibility *= self.trace_decay
        idx_tensor = torch.as_tensor(adj_positions, dtype=torch.long)
        self.eligibility[s_idx, idx_tensor] += grad_slice
        self.reward_accumulator += self.discount * step_reward
        self.discount *= self.gamma

    def compute_and_apply_reinforce_update(self, lr, mask=None):
        advantage = self.reward_accumulator - self.running_baseline
        self.running_baseline = (1 - self.baseline_decay) * self.running_baseline + self.baseline_decay * self.reward_accumulator
        self.w.data += lr * advantage * self.eligibility
        self.w.data.clamp_(self.wmin, self.wmax)
        if mask is not None:
            self.w.data *= mask
        self.eligibility.zero_()
        self.reward_accumulator = 0.0
        self.discount = 1.0

    def reset_episode(self):
        self.eligibility.zero_()
        self.reward_accumulator = 0.0
        self.discount = 1.0

    def reset_(self):
        self.a_pre.zero_()
        self.impulse_state.zero_()