"""Adaptive parameter strategies for swarm intelligence algorithms."""

import numpy as np


def adaptive_inertia_weight(
    iteration: int, max_iter: int, w_start: float = 0.9, w_end: float = 0.4
) -> float:
    """Linearly decaying inertia weight for PSO.

    Higher w early promotes exploration; lower w later promotes exploitation.
    """
    return w_start - (w_start - w_end) * (iteration / max_iter)


def adaptive_gwo_a(iteration: int, max_iter: int) -> float:
    """Nonlinear (exponential) decay of exploration parameter 'a' for GWO.

    Standard GWO uses linear decay 2→0. This nonlinear variant spends
    more time in exploration phase, then transitions faster to exploitation.
    """
    return 2.0 * (1.0 - (iteration / max_iter) ** 2)


def adaptive_woa_a(iteration: int, max_iter: int) -> float:
    """Nonlinear decay of parameter 'a' for WOA.

    Similar rationale to GWO: flatter early, steeper late.
    """
    return 2.0 * np.exp(-iteration / max_iter * 3) + 0.1
