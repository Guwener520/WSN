"""Standard WOA and Improved WOA for WSN coverage optimization."""

import numpy as np

from wsn.algorithms._base import SwarmOptimizer
from wsn.improvements.chaotic_init import init_chaotic_sequence
from wsn.improvements.adaptive_params import adaptive_woa_a
from wsn.improvements.mutation import gaussian_mutation


class WOA(SwarmOptimizer):
    """Whale Optimization Algorithm for WSN node placement.

    Encodes positions of all N nodes as a flattened vector (2 * n_nodes).
    """

    def __init__(
        self,
        env,
        fitness_evaluator,
        n_whales: int = 30,
        max_iter: int = 200,
        b: float = 1.0,            # spiral constant
        seed: int | None = None,
    ):
        self.env = env
        self.evaluator = fitness_evaluator
        self.n_whales = n_whales
        self.max_iter = max_iter
        self.b = b
        self.rng = np.random.default_rng(seed)
        self.dim = 2 * env.n_nodes

        self.lb = np.tile([0.0, 0.0], env.n_nodes)
        self.ub = np.tile([env.width, env.height], env.n_nodes)

        self.best_pos: np.ndarray | None = None
        self.best_fitness: float = -np.inf
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    def _decode(self, whale: np.ndarray) -> np.ndarray:
        return whale.reshape(-1, 2)

    def _clamp(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.lb, self.ub)

    def _fitness(self, positions: np.ndarray) -> float:
        self.env.set_positions(positions)
        return self.evaluator.evaluate_from_environment(self.env).fitness

    # ------------------------------------------------------------------
    def optimize(self, verbose: bool = False) -> tuple[np.ndarray, float]:
        n = self.n_whales
        d = self.dim

        positions = self.rng.uniform(low=self.lb, high=self.ub, size=(n, d))
        fitnesses = np.full(n, -np.inf)

        for i in range(n):
            fitnesses[i] = self._fitness(self._decode(positions[i]))
            if fitnesses[i] > self.best_fitness:
                self.best_fitness = fitnesses[i]
                self.best_pos = positions[i].copy()

        for it in range(self.max_iter):
            a = 2.0 * (1.0 - it / self.max_iter)
            a2 = -1.0 + it * ((-1.0) / self.max_iter)  # -1 to -2

            for i in range(n):
                r = self.rng.random()
                A = 2.0 * a * self.rng.random(d) - a
                C = 2.0 * self.rng.random(d)
                l = self.rng.uniform(-1, 1)
                p = self.rng.random()

                if p < 0.5:
                    if np.linalg.norm(A) < 1:
                        # Encircling prey
                        D = np.abs(C * self.best_pos - positions[i])
                        positions[i] = self.best_pos - A * D
                    else:
                        # Search for prey (random agent)
                        rand_idx = self.rng.integers(n)
                        X_rand = positions[rand_idx]
                        D = np.abs(C * X_rand - positions[i])
                        positions[i] = X_rand - A * D
                else:
                    # Bubble-net attacking (spiral)
                    D_prime = np.abs(self.best_pos - positions[i])
                    positions[i] = (
                        D_prime * np.exp(self.b * l) * np.cos(2.0 * np.pi * l)
                        + self.best_pos
                    )

                positions[i] = self._clamp(positions[i])

            for i in range(n):
                fit = self._fitness(self._decode(positions[i]))
                if fit > self.best_fitness:
                    self.best_fitness = fit
                    self.best_pos = positions[i].copy()

            best_for_it = {
                "iteration": it,
                "best_fitness": self.best_fitness,
                "coverage": self.evaluator.evaluate_from_environment(self.env).coverage_rate,
            }
            self.history.append(best_for_it)
            if verbose and (it + 1) % 20 == 0:
                print(f"  WOA iter {it + 1:3d}: fitness={self.best_fitness:.4f}")

        return self._decode(self.best_pos), self.best_fitness


class ImprovedWOA(WOA):
    """WOA with chaotic initialization, nonlinear adaptive a, and Gaussian mutation."""

    def __init__(
        self,
        env,
        fitness_evaluator,
        n_whales: int = 30,
        max_iter: int = 200,
        b: float = 1.0,
        mutation_rate: float = 0.1,
        mutation_scale: float = 0.05,
        seed: int | None = None,
    ):
        super().__init__(env, fitness_evaluator, n_whales, max_iter, b, seed)
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale

    def optimize(self, verbose: bool = False) -> tuple[np.ndarray, float]:
        n = self.n_whales
        d = self.dim

        # Chaotic initialization
        chaotic_seq = init_chaotic_sequence(n * d, self.rng)
        positions = self.lb + chaotic_seq.reshape(n, d) * (self.ub - self.lb)
        fitnesses = np.full(n, -np.inf)

        for i in range(n):
            fitnesses[i] = self._fitness(self._decode(positions[i]))
            if fitnesses[i] > self.best_fitness:
                self.best_fitness = fitnesses[i]
                self.best_pos = positions[i].copy()

        for it in range(self.max_iter):
            a = adaptive_woa_a(it, self.max_iter)  # nonlinear decay

            for i in range(n):
                A = 2.0 * a * self.rng.random(d) - a
                C = 2.0 * self.rng.random(d)
                l = self.rng.uniform(-1, 1)
                p = self.rng.random()

                if p < 0.5:
                    if np.linalg.norm(A) < 1:
                        D = np.abs(C * self.best_pos - positions[i])
                        positions[i] = self.best_pos - A * D
                    else:
                        rand_idx = self.rng.integers(n)
                        X_rand = positions[rand_idx]
                        D = np.abs(C * X_rand - positions[i])
                        positions[i] = X_rand - A * D
                else:
                    D_prime = np.abs(self.best_pos - positions[i])
                    positions[i] = (
                        D_prime * np.exp(self.b * l) * np.cos(2.0 * np.pi * l)
                        + self.best_pos
                    )

                positions[i] = self._clamp(positions[i])

            for i in range(n):
                fit = self._fitness(self._decode(positions[i]))
                if fit > self.best_fitness:
                    self.best_fitness = fit
                    self.best_pos = positions[i].copy()

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
                "best_fitness": self.best_fitness,
                "coverage": self.evaluator.evaluate_from_environment(self.env).coverage_rate,
            }
            self.history.append(best_for_it)
            if verbose and (it + 1) % 20 == 0:
                print(f"  ImprovedWOA iter {it + 1:3d}: fitness={self.best_fitness:.4f}")

        return self._decode(self.best_pos), self.best_fitness
