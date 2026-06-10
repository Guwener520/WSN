"""Abstract base for swarm intelligence optimizers."""

from abc import ABC, abstractmethod
from collections.abc import Callable
import numpy as np

SnapshotCallback = Callable[[int, np.ndarray, float, dict], None]


class SwarmOptimizer(ABC):
    """Interface that all SI optimizers must implement."""

    def _notify_callback(
        self,
        callback: SnapshotCallback | None,
        iteration: int,
        positions: np.ndarray,
        fitness: float,
        extra: dict | None = None,
    ) -> None:
        if callback is None:
            return
        callback(iteration, positions.copy(), float(fitness), extra or {})

    @abstractmethod
    def optimize(
        self,
        verbose: bool = False,
        callback: SnapshotCallback | None = None,
    ) -> tuple[np.ndarray, float]:
        """Run optimization, return (best_positions, best_fitness)."""
        ...
