"""Mutation operators: Gaussian mutation and Levy flight."""

import numpy as np


def gaussian_mutation(
    positions: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    iteration: int,
    max_iter: int,
    mutation_rate: float = 0.1,
    scale: float = 0.05,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Apply Gaussian mutation to a fraction of the population.

    Only activates in the second half of iterations to perturb
    solutions that may be stuck in local optima.

    Args:
        positions: (N, D) array of agent positions.
        lb, ub: lower/upper bounds per dimension.
        iteration, max_iter: current and total iterations.
        mutation_rate: fraction of agents to mutate per call.
        scale: standard deviation of Gaussian noise, relative to bound range.
        rng: numpy random generator.
    """
    if rng is None:
        rng = np.random.default_rng()
    if iteration < max_iter * 0.5:
        return positions

    n, d = positions.shape
    n_mutate = max(1, int(n * mutation_rate))
    indices = rng.choice(n, size=n_mutate, replace=False)

    noise = rng.normal(0, scale, size=(n_mutate, d)) * (ub - lb)
    positions[indices] = np.clip(positions[indices] + noise, lb, ub)
    return positions


def levy_flight(
    positions: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    iteration: int,
    max_iter: int,
    mutation_rate: float = 0.05,
    beta: float = 1.5,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Apply Levy flight perturbation to selected agents.

    Levy flights produce occasional large jumps, which helps escape
    deep local optima.

    The step size decays with iteration count.
    """
    if rng is None:
        rng = np.random.default_rng()

    n, d = positions.shape
    n_mutate = max(1, int(n * mutation_rate))
    indices = rng.choice(n, size=n_mutate, replace=False)

    # Mantegna's algorithm for Levy stable distribution
    sigma_u = (
        np.exp(np.log(np.math.gamma(1 + beta)) / beta * np.sin(np.pi * beta / 2))
        / np.math.gamma((1 + beta) / 2) / beta / 2 ** ((beta - 1) / 2)
    ) ** (1 / beta)
    u = rng.normal(0, sigma_u, size=(n_mutate, d))
    v = rng.normal(0, 1, size=(n_mutate, d))
    step = u / (np.abs(v) ** (1 / beta))

    step_size = (1.0 - iteration / max_iter) * 0.01 * (ub - lb)
    positions[indices] = np.clip(positions[indices] + step_size * step, lb, ub)
    return positions
