"""Standard GWO and Improved GWO for WSN coverage optimization."""

import numpy as np

from wsn.algorithms._base import SwarmOptimizer
from wsn.improvements.chaotic_init import init_chaotic_sequence
from wsn.improvements.adaptive_params import adaptive_gwo_a
from wsn.improvements.mutation import gaussian_mutation


class GWO(SwarmOptimizer):
    """Grey Wolf Optimizer for WSN node placement.

    Encodes positions of all N nodes as a vector of length 2 * n_nodes.
    """

    def __init__(
        self,
        env,
        fitness_evaluator,
        n_wolves: int = 30,
        max_iter: int = 200,
        seed: int | None = None,
    ):
        self.env = env
        self.evaluator = fitness_evaluator
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.rng = np.random.default_rng(seed)
        self.dim = 2 * env.n_nodes

        self.lb = np.tile([0.0, 0.0], env.n_nodes)
        self.ub = np.tile([env.width, env.height], env.n_nodes)

        self.alpha_pos: np.ndarray | None = None
        self.alpha_fitness: float = -np.inf
        self.beta_pos: np.ndarray | None = None
        self.beta_fitness: float = -np.inf
        self.delta_pos: np.ndarray | None = None
        self.delta_fitness: float = -np.inf
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    def _decode(self, wolf: np.ndarray) -> np.ndarray:
        return wolf.reshape(-1, 2)

    def _clamp(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.lb, self.ub)

    def _fitness(self, positions: np.ndarray) -> float:
        self.env.set_positions(positions)
        return self.evaluator.evaluate_from_environment(self.env).fitness

    # ------------------------------------------------------------------
    def optimize(self, verbose: bool = False) -> tuple[np.ndarray, float]:
        n = self.n_wolves
        d = self.dim

        positions = self.rng.uniform(low=self.lb, high=self.ub, size=(n, d))
        fitnesses = np.full(n, -np.inf)

        # Evaluate initial population
        for i in range(n):
            fitnesses[i] = self._fitness(self._decode(positions[i]))

        self._update_hierarchy(positions, fitnesses)

        for it in range(self.max_iter):
            a = 2.0 * (1.0 - it / self.max_iter)  # linearly decreasing from 2 to 0

            for i in range(n):
                # Update position using alpha, beta, delta guidance
                for role, pos in [
                    ("alpha", self.alpha_pos),
                    ("beta", self.beta_pos),
                    ("delta", self.delta_pos),
                ]:
                    r1 = self.rng.random(d)
                    r2 = self.rng.random(d)
                    A = 2.0 * a * r1 - a
                    C = 2.0 * r2
                    diff = np.abs(C * pos - positions[i])
                    if role == "alpha":
                        X1 = pos - A * diff
                    elif role == "beta":
                        X2 = pos - A * diff
                    else:
                        X3 = pos - A * diff

                positions[i] = self._clamp((X1 + X2 + X3) / 3.0)

            for i in range(n):
                fitnesses[i] = self._fitness(self._decode(positions[i]))

            self._update_hierarchy(positions, fitnesses)

            best_for_it = {
                "iteration": it,
                "best_fitness": self.alpha_fitness,
                "coverage": self.evaluator.evaluate_from_environment(self.env).coverage_rate,
            }
            self.history.append(best_for_it)
            if verbose and (it + 1) % 20 == 0:
                print(f"  GWO iter {it + 1:3d}: fitness={self.alpha_fitness:.4f}")

        return self._decode(self.alpha_pos), self.alpha_fitness

    def _update_hierarchy(self, positions, fitnesses):
        # Sort by fitness descending; top 3 are alpha, beta, delta
        idx = np.argsort(-fitnesses)
        self.alpha_pos, self.alpha_fitness = positions[idx[0]].copy(), fitnesses[idx[0]]
        self.beta_pos, self.beta_fitness = positions[idx[1]].copy(), fitnesses[idx[1]]
        self.delta_pos, self.delta_fitness = positions[idx[2]].copy(), fitnesses[idx[2]]


class ImprovedGWO(GWO):
    """GWO with chaotic initialization, nonlinear adaptive a, and Gaussian mutation."""

    def __init__(
        self,
        env,
        fitness_evaluator,
        n_wolves: int = 30,
        max_iter: int = 200,
        mutation_rate: float = 0.1,
        mutation_scale: float = 0.05,
        seed: int | None = None,
    ):
        super().__init__(env, fitness_evaluator, n_wolves, max_iter, seed)
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale

    def optimize(self, verbose: bool = False) -> tuple[np.ndarray, float]:
        n = self.n_wolves
        d = self.dim

        # Chaotic initialization
        chaotic_seq = init_chaotic_sequence(n * d, self.rng)
        positions = self.lb + chaotic_seq.reshape(n, d) * (self.ub - self.lb)
        fitnesses = np.full(n, -np.inf)

        for i in range(n):
            fitnesses[i] = self._fitness(self._decode(positions[i]))

        self._update_hierarchy(positions, fitnesses)

        for it in range(self.max_iter):
            a = adaptive_gwo_a(it, self.max_iter)  # nonlinear decay

            for i in range(n):
                for role, pos in [
                    ("alpha", self.alpha_pos),
                    ("beta", self.beta_pos),
                    ("delta", self.delta_pos),
                ]:
                    r1 = self.rng.random(d)
                    r2 = self.rng.random(d)
                    A = 2.0 * a * r1 - a
                    C = 2.0 * r2
                    diff = np.abs(C * pos - positions[i])
                    if role == "alpha":
                        X1 = pos - A * diff
                    elif role == "beta":
                        X2 = pos - A * diff
                    else:
                        X3 = pos - A * diff

                positions[i] = self._clamp((X1 + X2 + X3) / 3.0)

            for i in range(n):
                fitnesses[i] = self._fitness(self._decode(positions[i]))

            self._update_hierarchy(positions, fitnesses)

            # Gaussian mutation in later iterations
            positions = gaussian_mutation(
                positions, self.lb, self.ub,
                iteration=it, max_iter=self.max_iter,
                mutation_rate=self.mutation_rate,
                scale=self.mutation_scale,
                rng=self.rng,
            )

            best_for_it = {
                "iteration": it,
                "best_fitness": self.alpha_fitness,
                "coverage": self.evaluator.evaluate_from_environment(self.env).coverage_rate,
            }
            self.history.append(best_for_it)
            if verbose and (it + 1) % 20 == 0:
                print(f"  ImprovedGWO iter {it + 1:3d}: fitness={self.alpha_fitness:.4f}")

        return self._decode(self.alpha_pos), self.alpha_fitness
