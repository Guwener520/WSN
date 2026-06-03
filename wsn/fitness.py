"""Multi-objective fitness function for WSN coverage optimization."""

import numpy as np
from dataclasses import dataclass


@dataclass
class FitnessMetrics:
    coverage_rate: float      # fraction of grid points covered
    redundancy_rate: float    # average excess coverage per covered point
    connectivity_rate: float  # fraction of reachable node pairs
    fitness: float            # combined weighted score


class FitnessEvaluator:
    """Computes the weighted multi-objective fitness:

        Fitness = w1 * CoverageRate + w2 * (1 - RedundancyRate)
                  + w3 * ConnectivityRate

    All three components are in [0, 1]. Fitness is in [0, 1] when weights sum to 1.
    """

    def __init__(
        self,
        w_coverage: float = 0.5,
        w_redundancy: float = 0.3,
        w_connectivity: float = 0.2,
    ):
        total = w_coverage + w_redundancy + w_connectivity
        self.w1 = w_coverage / total
        self.w2 = w_redundancy / total
        self.w3 = w_connectivity / total

    def evaluate(
        self,
        coverage_matrix: np.ndarray,
        covered_mask: np.ndarray,
        connectivity_rate: float,
    ) -> FitnessMetrics:
        """Evaluate fitness from raw metrics.

        Args:
            coverage_matrix: (n_nodes, n_grid_points) bool — per-node coverage.
            covered_mask:   (n_grid_points,) bool — which grid points are covered.
            connectivity_rate: float in [0, 1].
        """
        n_nodes, n_grid = coverage_matrix.shape

        # Coverage rate
        coverage_rate = float(np.mean(covered_mask))

        # Redundancy rate: avg. (extra nodes covering a covered point) / (n_nodes - 1)
        if coverage_rate > 0 and n_nodes > 1:
            covered_counts = np.sum(coverage_matrix[:, covered_mask], axis=0)
            redundancy_rate = float(np.mean((covered_counts - 1) / (n_nodes - 1)))
        else:
            redundancy_rate = 0.0

        # Combined fitness (higher is better)
        fitness = (
            self.w1 * coverage_rate
            + self.w2 * (1.0 - redundancy_rate)
            + self.w3 * connectivity_rate
        )

        return FitnessMetrics(
            coverage_rate=coverage_rate,
            redundancy_rate=redundancy_rate,
            connectivity_rate=connectivity_rate,
            fitness=fitness,
        )

    def evaluate_from_environment(self, env) -> FitnessMetrics:
        """Convenience: compute fitness directly from a WSNEnvironment instance."""
        cov_matrix = env.coverage_matrix()
        covered = env.covered_grid_mask()
        connectivity = env.connectivity_rate()
        return self.evaluate(cov_matrix, covered, connectivity)
