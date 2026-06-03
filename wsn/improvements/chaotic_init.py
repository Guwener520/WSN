"""Chaotic initialization using Logistic map."""

import numpy as np


def logistic_map(x: np.ndarray, mu: float = 4.0) -> np.ndarray:
    """One iteration of the logistic map: x_{n+1} = mu * x_n * (1 - x_n)."""
    return mu * x * (1.0 - x)


def init_chaotic_sequence(
    n_values: int, rng: np.random.Generator, mu: float = 4.0
) -> np.ndarray:
    """Generate a chaotic sequence of length n_values in (0, 1).

    Uses the logistic map to produce a more diverse initial population
    than uniform random sampling.
    """
    # Start from a random seed in (0, 1), avoiding fixed points 0, 0.5, 0.75
    x = rng.uniform(0.01, 0.49)
    seq = np.empty(n_values)
    for i in range(n_values):
        x = logistic_map(x, mu)
        seq[i] = x
    return seq
