"""Baseline deployment strategies for comparison."""

import numpy as np

from wsn.fitness import FitnessEvaluator, FitnessMetrics


class RandomBaseline:
    """Random uniform deployment — repeated trials for statistics."""

    def __init__(
        self,
        env,
        fitness_evaluator: FitnessEvaluator,
        n_trials: int = 30,
        seed: int | None = None,
    ):
        self.env = env
        self.evaluator = fitness_evaluator
        self.n_trials = n_trials
        self.rng = np.random.default_rng(seed)
        self.results: list[FitnessMetrics] = []

    def run(self) -> dict:
        """Run multiple random deployments and return aggregated statistics."""
        self.results = []
        coverages = []
        for _ in range(self.n_trials):
            positions = self.env.random_deploy()
            env_copy = type(self.env)(
                self.env.width, self.env.height,
                self.env.grid_resolution, self.env.n_nodes,
                self.env.sensing_radius, self.env.communication_radius,
                obstacles=self.env.obstacles,
            )
            env_copy.set_positions(positions)
            metrics = self.evaluator.evaluate_from_environment(env_copy)
            self.results.append(metrics)
            coverages.append(metrics.coverage_rate)

        coverages = np.array(coverages)
        return {
            "best_coverage": float(np.max(coverages)),
            "mean_coverage": float(np.mean(coverages)),
            "std_coverage": float(np.std(coverages)),
            "best_fitness": float(np.max([r.fitness for r in self.results])),
            "mean_fitness": float(np.mean([r.fitness for r in self.results])),
            "std_fitness": float(np.std([r.fitness for r in self.results])),
        }


class GridBaseline:
    """Regular grid deployment — a single deterministic layout."""

    def __init__(self, env, fitness_evaluator: FitnessEvaluator):
        self.env = env
        self.evaluator = fitness_evaluator

    def run(self) -> dict:
        positions = self.env.grid_deploy()
        env_copy = type(self.env)(
            self.env.width, self.env.height,
            self.env.grid_resolution, self.env.n_nodes,
            self.env.sensing_radius, self.env.communication_radius,
            obstacles=self.env.obstacles,
        )
        env_copy.set_positions(positions)
        metrics = self.evaluator.evaluate_from_environment(env_copy)
        return {
            "coverage": metrics.coverage_rate,
            "fitness": metrics.fitness,
            "redundancy_rate": metrics.redundancy_rate,
            "connectivity_rate": metrics.connectivity_rate,
        }
