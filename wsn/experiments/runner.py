"""Experiment runner for comparing algorithms on a given WSN scenario."""

import time
import numpy as np

from wsn.environment import WSNEnvironment
from wsn.fitness import FitnessEvaluator
from wsn.algorithms._base import SwarmOptimizer
from wsn.experiments.baselines import RandomBaseline, GridBaseline


class ExperimentRunner:
    """Run multiple algorithms (including baselines) and collect results.

    Usage:
        runner = ExperimentRunner(env_config, fitness_weights)
        runner.add_baseline("Random", RandomBaseline(env, evaluator))
        runner.add_algorithm("PSO", PSO(env, evaluator))
        results = runner.run_all()
        runner.print_summary()
    """

    def __init__(
        self,
        env_config: dict,
        fitness_weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
    ):
        self.env_config = env_config
        self.evaluator = FitnessEvaluator(*fitness_weights)
        self.algorithms: dict[str, SwarmOptimizer | RandomBaseline | GridBaseline] = {}
        self.results: dict = {}

    def add_algorithm(self, name: str, algo) -> None:
        """Register an algorithm or baseline.

        Args:
            name: display name (e.g. 'PSO', 'ImprovedGWO', 'Random').
            algo: SwarmOptimizer, RandomBaseline, or GridBaseline instance.
        """
        self.algorithms[name] = algo

    def add_baseline(self, name: str, baseline) -> None:
        """Add a baseline. Convenience alias for add_algorithm."""
        self.add_algorithm(name, baseline)

    def run_all(self, verbose: bool = False) -> dict:
        """Run all registered methods and return a summary dict."""
        for name, algo in self.algorithms.items():
            print(f"\n{'='*50}")
            print(f"  Running: {name}")
            print(f"{'='*50}")

            t0 = time.perf_counter()

            if isinstance(algo, RandomBaseline):
                result = algo.run()
                elapsed = time.perf_counter() - t0
                result["time"] = elapsed
                result["type"] = "baseline_random"
            elif isinstance(algo, GridBaseline):
                result = algo.run()
                elapsed = time.perf_counter() - t0
                result["time"] = elapsed
                result["type"] = "baseline_grid"
            else:
                # SwarmOptimizer
                best_pos, best_fit = algo.optimize(verbose=verbose)
                elapsed = time.perf_counter() - t0

                # Compute final metrics
                env = algo.env
                env.set_positions(best_pos)
                metrics = self.evaluator.evaluate_from_environment(env)

                result = {
                    "type": "swarm",
                    "best_fitness": best_fit,
                    "coverage_rate": metrics.coverage_rate,
                    "redundancy_rate": metrics.redundancy_rate,
                    "connectivity_rate": metrics.connectivity_rate,
                    "time": elapsed,
                    "history": getattr(algo, "history", []),
                    "best_positions": best_pos,
                }

            self.results[name] = result
            print(f"    Done in {result['time']:.2f}s")

        return self.results

    def print_summary(self) -> None:
        """Print a formatted comparison table."""
        if not self.results:
            print("No results yet. Call run_all() first.")
            return

        print(f"\n{'='*70}")
        print(f"{'Algorithm':<20} {'Type':<12} {'Coverage':>10} {'Fitness':>10}")
        print(f"{'-'*70}")

        for name, r in self.results.items():
            algo_type = r["type"]
            if algo_type.startswith("baseline"):
                if "best_coverage" in r:
                    cov = f"{r['best_coverage']:.4f}"
                    fit = f"{r['best_fitness']:.4f}"
                else:
                    cov = f"{r['coverage']:.4f}"
                    fit = f"{r['fitness']:.4f}"
            else:
                cov = f"{r['coverage_rate']:.4f}"
                fit = f"{r['best_fitness']:.4f}"

            print(f"{name:<20} {algo_type:<12} {cov:>10} {fit:>10}")

        print(f"{'='*70}")
