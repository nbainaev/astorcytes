import torch
import os
import numpy as np
from functools import lru_cache


@lru_cache(maxsize=1)
def _load_stdp_table(stdp_path):
    try:
        table = torch.as_tensor(np.loadtxt(stdp_path), dtype=torch.float32)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Required STDP lookup table not found: {stdp_path}"
        ) from exc
    if table.ndim != 2:
        raise ValueError(f"Expected a 2-D STDP lookup table, got {tuple(table.shape)}")
    return table


class LearningRule:
    def __init__(self, connection, nu=None, reduction=None, weight_decay=0.0, **kwargs):
        self.connection = connection
        self.nu = nu if nu is not None else [0.0, 0.0]
        self.reduction = reduction if reduction is not None else torch.mean
        self.weight_decay = weight_decay

    def update(self, **kwargs):
        if self.weight_decay != 0:
            self.connection.w.data *= (1 - self.weight_decay)


class WeightDependentPostPre(LearningRule):
    def __init__(self, connection, nu=None, reduction=None, weight_decay=0.0,
                 post_spike_weight_decay=0.0, tc_trace=20, tc_trace_neg=20, **kwargs):
        super().__init__(connection, nu, reduction, weight_decay, **kwargs)
        self.post_spike_weight_decay = post_spike_weight_decay
        self.tc_trace = tc_trace
        self.tc_trace_neg = tc_trace_neg
        self.interval = 100
        stdp_path = os.path.join(os.path.dirname(__file__), "..", "STDP.txt")
        try:
            self.STDP_base = _load_stdp_table(stdp_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Required STDP lookup table not found: {stdp_path}"
            ) from exc
        if self.STDP_base.ndim != 2:
            raise ValueError(f"Expected a 2-D STDP lookup table, got {tuple(self.STDP_base.shape)}")
        self.wmin = getattr(connection, 'wmin', 0.001)
        self.wmax = getattr(connection, 'wmax', 1.0)

    def delta_w_custom_single(self, weight, delta_val):
        first_index = int(round(float(weight / self.nu[0] * 100)))
        if not torch.isfinite(torch.as_tensor(delta_val)):
            delta_val = torch.tensor(0.0)
        second_index = int(float(delta_val) + 60)
        n_cols = self.STDP_base.shape[1]
        if second_index < 0 or second_index >= n_cols:
            second_index = 0
        if first_index < 0:
            first_index = -first_index
        if first_index > 100:
            first_index = 100
        return self.STDP_base[first_index][second_index]

    def _stdp_lookup(self, weight, delta):
        """Vectorized table lookup equivalent to per-element delta_w_custom_single."""
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
        first = torch.round(weight / self.nu[0] * 100).long().abs().clamp(max=100)
        second = (delta.double() + 60).long()
        n_cols = self.STDP_base.shape[1]
        second = torch.where((second < 0) | (second >= n_cols), torch.zeros_like(second), second)
        return self.STDP_base[first, second]

    def update(self, current_position=None, adjacent_positions=None, **kwargs):
        if current_position is not None and adjacent_positions is not None:
            conn = self.connection
            adj = torch.as_tensor(adjacent_positions, dtype=torch.long)
            w_adj = conn.w[current_position, adj]
            source_s = conn.source.s[:, current_position].float().unsqueeze(1).unsqueeze(2)
            source_x = conn.source.x[:, current_position].unsqueeze(1).unsqueeze(2)
            target_s = conn.target.s[:, adj].float().unsqueeze(1)
            target_x = conn.target.x_neg[:, adj].unsqueeze(1)

            outer_product_pre = self.reduction(torch.bmm(source_s, target_x), dim=0).view(-1)
            outer_product_pre = torch.clamp(outer_product_pre, min=1e-10)
            delta_pre = self.tc_trace_neg * torch.log(outer_product_pre)
            update = self.nu[0] * self._stdp_lookup(w_adj, delta_pre)

            outer_product_post = self.reduction(torch.bmm(source_x, target_s), dim=0).view(-1)
            outer_product_post = torch.clamp(outer_product_post, min=1e-10)
            delta_post = -self.tc_trace * torch.log(outer_product_post)
            update = update + self.nu[1] * self._stdp_lookup(w_adj, delta_post)

            decay_factor = self.reduction(torch.bmm(torch.ones_like(source_x), target_s), dim=0).view(-1)
            update = update + (-self.post_spike_weight_decay) * w_adj * decay_factor
            conn.last_stdp_update_raw = update.detach()
            conn.w.data[current_position, adj] += update
        else:
            batch_size = self.connection.source.batch_size
            source_s = self.connection.source.s.view(batch_size, -1).unsqueeze(2).float()
            source_x = self.connection.source.x.view(batch_size, -1).unsqueeze(2)
            target_s = self.connection.target.s.view(batch_size, -1).unsqueeze(1).float()
            target_x = self.connection.target.x_neg.view(batch_size, -1).unsqueeze(1)
            update = 0
            outer_product = self.reduction(torch.bmm(source_s, target_x), dim=0)
            outer_product = torch.clamp(outer_product, min=1e-10)
            update += self.nu[0] * self.delta_w_custom(self.tc_trace_neg * torch.log(outer_product))
            outer_product = self.reduction(torch.bmm(source_x, target_s), dim=0)
            outer_product = torch.clamp(outer_product, min=1e-10)
            update += self.nu[1] * self.delta_w_custom(-self.tc_trace * torch.log(outer_product))
            update += (-self.post_spike_weight_decay) * self.connection.w * self.reduction(
                torch.bmm(torch.ones(source_x.shape), target_s), dim=0)
            self.connection.last_stdp_update_raw = update.detach()
            self.connection.w.data += update
        super().update()
        main_reference_stdp = getattr(self.connection, "main_reference_stdp", False)
        enforce_bounds = getattr(
            self.connection, "enforce_post_stdp_bounds", not main_reference_stdp,
        )
        if enforce_bounds:
            self.connection.w.data.clamp_(self.wmin, self.wmax)
        structural_mask = getattr(self.connection, "structural_mask", None)
        apply_mask = getattr(
            self.connection, "apply_structural_mask_during_stdp", not main_reference_stdp,
        )
        if structural_mask is not None and apply_mask:
            self.connection.w.data.masked_fill_(
                ~structural_mask.to(device=self.connection.w.device, dtype=torch.bool), 0
            )

    def delta_w_custom(self, delta):
        return self._stdp_lookup(self.connection.w.detach(), delta)


class NoOp(LearningRule):
    def update(self, **kwargs):
        super().update()
