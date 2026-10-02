import numpy as np
import torch


def derive_seed(seed: int, stream_id: int) -> int:
    """Derive a stable 32-bit seed for an independent stochastic stream."""
    return int(np.random.SeedSequence([int(seed), int(stream_id)]).generate_state(1)[0])


def make_rng_streams(seed: int) -> dict:
    """Create independent weight, sensory, and action RNG streams.

    ``RandomState(seed)`` intentionally preserves the exact initialization
    sequence used by the pre-stream implementation, while no sampling after
    initialization can advance it.
    """
    action_seed = derive_seed(seed, 2)
    return {
        "weights": np.random.RandomState(int(seed)),
        "sensory": torch.Generator(device="cpu").manual_seed(derive_seed(seed, 1)),
        "action_numpy": np.random.RandomState(action_seed),
        "action_torch": torch.Generator(device="cpu").manual_seed(action_seed),
        "eval_sensory": torch.Generator(device="cpu").manual_seed(derive_seed(seed, 3)),
        "eval_action_numpy": np.random.RandomState(derive_seed(seed, 4)),
        "eval_action_torch": torch.Generator(device="cpu").manual_seed(derive_seed(seed, 4)),
    }


def bernoulli_loader(data, time=None, dt=1.0, **kwargs):
    """Yield one Bernoulli spike vector per simulator timestep.

    ``data`` contains per-timestep spike probabilities. ``time`` is the
    simulation duration in the same units as ``dt``. A single probability
    vector is repeated for the full duration.
    """
    probs = torch.as_tensor(data, dtype=torch.float32)

    if probs.ndim == 1:
        probs = probs.unsqueeze(0)
    if probs.ndim != 2:
        raise ValueError(f"Expected [T, N] or [1, N], got {tuple(probs.shape)}")
    if dt <= 0:
        raise ValueError("dt must be positive")

    n_steps = probs.shape[0] if time is None else int(time / dt)
    if n_steps < 0:
        raise ValueError("time must be non-negative")

    if probs.shape[0] == 1:
        probs = probs.expand(n_steps, -1)
    elif probs.shape[0] != n_steps:
        raise ValueError(
            f"Input has {probs.shape[0]} timesteps, expected {n_steps}"
        )

    if not torch.isfinite(probs).all() or torch.any((probs < 0) | (probs > 1)):
        raise ValueError("Bernoulli probabilities must be in [0, 1]")

    generator = kwargs.get("generator")
    samples = torch.bernoulli(probs, generator=generator)
    for t in range(n_steps):
        yield samples[t]


def create_adjacency_matrix(N):
    total_cells = N * N
    adjacency_matrix = np.zeros((total_cells, total_cells), dtype=int)
    for i in range(total_cells):
        row = i // N
        col = i % N
        if col > 0:
            j = i - 1
            adjacency_matrix[i][j] = 1
            adjacency_matrix[j][i] = 1
        if col < N - 1:
            j = i + 1
            adjacency_matrix[i][j] = 1
            adjacency_matrix[j][i] = 1
        if row > 0:
            j = i - N
            adjacency_matrix[i][j] = 1
            adjacency_matrix[j][i] = 1
        if row < N - 1:
            j = i + N
            adjacency_matrix[i][j] = 1
            adjacency_matrix[j][i] = 1
    return adjacency_matrix
