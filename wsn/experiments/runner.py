"""Experiment runner for comparing algorithms on a given WSN scenario."""

import csv
import json
import os
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

    def summary_rows(self) -> list[dict]:
        """Return normalized rows suitable for tables, CSV, and bar charts."""
        if not self.results:
            return []

        rows = []
        for name, result in self.results.items():
            row = {
                "algorithm": name,
                "type": result["type"],
                "coverage": result.get(
                    "mean_coverage",
                    result.get("coverage", result.get("coverage_rate")),
                ),
                "fitness": result.get(
                    "mean_fitness",
                    result.get("fitness", result.get("best_fitness")),
                ),
                "best_coverage": result.get(
                    "best_coverage",
                    result.get("coverage", result.get("coverage_rate")),
                ),
                "best_fitness": result.get(
                    "best_fitness",
                    result.get("fitness"),
                ),
                "redundancy_rate": result.get("redundancy_rate"),
                "connectivity_rate": result.get("connectivity_rate"),
                "time": result.get("time"),
            }
            rows.append(row)
        return rows

    def save_summary_csv(self, filename: str) -> str:
        """Save normalized comparison metrics as CSV."""
        rows = self.summary_rows()
        if not rows:
            raise RuntimeError("No results to save. Call run_all() first.")

        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        fieldnames = list(rows[0].keys())
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return filename

    def save_results_json(self, filename: str) -> str:
        """Save full experiment results, including histories and best positions."""
        if not self.results:
            raise RuntimeError("No results to save. Call run_all() first.")

        def convert(obj):
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            if isinstance(obj, tuple):
                return [convert(v) for v in obj]
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.generic):
                return obj.item()
            return obj

        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(convert(self.results), f, indent=2, ensure_ascii=False)
        return filename

    def plot_metric_bar(
        self,
        metric: str = "coverage",
        filename: str | None = None,
        ax=None,
        title: str | None = None,
    ):
        """Plot a bar chart for a normalized comparison metric.

        Supported metrics: coverage, fitness, best_coverage, best_fitness,
        redundancy_rate, connectivity_rate, time.
        """
        rows = self.summary_rows()
        if not rows:
            raise RuntimeError("No results to plot. Call run_all() first.")
        if metric not in rows[0]:
            raise ValueError(f"Unsupported metric: {metric}")

        labels = [r["algorithm"] for r in rows]
        values = [r[metric] for r in rows]
        if any(v is None for v in values):
            missing = [labels[i] for i, v in enumerate(values) if v is None]
            raise ValueError(f"Metric '{metric}' missing for: {', '.join(missing)}")

        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(9, 5))

        ax.bar(labels, values, color="#4C78A8")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(title or f"Algorithm Comparison: {metric.replace('_', ' ').title()}")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.25)

        if filename is not None:
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            ax.figure.tight_layout()
            ax.figure.savefig(filename, dpi=150)
        return ax
