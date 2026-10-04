"""Small tabular TD(0) critic used by the TD-RSTDP actor path."""

from pathlib import Path

import numpy as np
import torch


class TabularTDCritic:
    """State-value critic with one-step TD(0) updates.

    The object is intentionally independent of the SNN and should be created
    once per random seed, then retained across that seed's training episodes.
    """

    def __init__(self, n_states, learning_rate, gamma, init_value=0.0):
        if int(n_states) < 1:
            raise ValueError("n_states must be positive")
        if float(learning_rate) < 0:
            raise ValueError("critic learning_rate must be non-negative")
        if not 0.0 <= float(gamma) <= 1.0:
            raise ValueError("critic gamma must be in [0, 1]")
        self.values = torch.full(
            (int(n_states),), float(init_value), dtype=torch.float32,
        )
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)

    def compute_td_error(self, state, reward, next_state, done):
        state = self._validate_state(state)
        next_state = self._validate_state(next_state)
        target = torch.as_tensor(float(reward), dtype=self.values.dtype)
        if not done:
            target = target + self.gamma * self.values[next_state]
        return target - self.values[state]

    def update(self, state, delta):
        state = self._validate_state(state)
        delta = torch.as_tensor(delta, dtype=self.values.dtype)
        self.values[state] += self.learning_rate * delta

    def _validate_state(self, state):
        state = int(state)
        if state < 0 or state >= self.values.numel():
            raise IndexError(f"state {state} outside critic table [0, {self.values.numel()})")
        return state

    def save(self, output_dir, grid_size=None, make_heatmap=True):
        """Save raw values and, for square grids, an optional PNG heatmap."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        values = self.values.detach().cpu().numpy().copy()
        np.save(output_dir / "critic_values.npy", values)
        if make_heatmap and grid_size is not None and grid_size * grid_size == len(values):
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5, 4))
            image = ax.imshow(values.reshape(grid_size, grid_size), cmap="viridis")
            ax.set_title("Tabular TD(0) critic values")
            ax.set_xlabel("Column")
            ax.set_ylabel("Row")
            fig.colorbar(image, ax=ax, label="V(s)")
            fig.tight_layout()
            fig.savefig(output_dir / "critic_values.png", dpi=160)
            plt.close(fig)
        return output_dir / "critic_values.npy"
