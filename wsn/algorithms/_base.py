"""Abstract base for swarm intelligence optimizers."""

from abc import ABC, abstractmethod
import numpy as np


class SwarmOptimizer(ABC):
    """Interface that all SI optimizers must implement."""

    @abstractmethod
    def optimize(self, verbose: bool = False) -> tuple[np.ndarray, float]:
        """Run optimization, return (best_positions, best_fitness)."""
        ...
