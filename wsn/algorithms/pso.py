"""Standard PSO and Improved PSO for WSN coverage optimization."""

import numpy as np

from wsn.algorithms._base import SwarmOptimizer
from wsn.improvements.chaotic_init import init_chaotic_sequence
from wsn.improvements.adaptive_params import adaptive_inertia_weight
from wsn.improvements.mutation import gaussian_mutation


class PSO(SwarmOptimizer):
    """Particle Swarm Optimization for WSN node placement.

    Each particle encodes the 2D positions of all N nodes as a
    flattened vector of length 2 * n_nodes, bounded by [0, area_dim].
    """

    def __init__(
        self,
        env,
        fitness_evaluator,
        n_particles: int = 30,
        w: float = 0.7,          # inertia weight
        c1: float = 1.5,          # cognitive coefficient
        c2: float = 1.5,          # social coefficient
        max_iter: int = 200,
        seed: int | None = None,
    ):
        self.env = env
        self.evaluator = fitness_evaluator
        self.n_particles = n_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        self.rng = np.random.default_rng(seed)
        self.dim = 2 * env.n_nodes

        # per-node bounds
        self.lb = np.tile([0.0, 0.0], env.n_nodes)
        self.ub = np.tile([env.width, env.height], env.n_nodes)

        self.best_pos: np.ndarray | None = None
        self.best_fitness: float = -np.inf
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    def _decode(self, particle: np.ndarray) -> np.ndarray:
        return particle.reshape(-1, 2)

    def _clamp(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.lb, self.ub)

    def _fitness(self, positions: np.ndarray) -> float:
        self.env.set_positions(positions)
        return self.evaluator.evaluate_from_environment(self.env).fitness

    # ------------------------------------------------------------------
    def optimize(self, verbose: bool = False) -> tuple[np.ndarray, float]:
        n = self.n_particles
        d = self.dim

        # Initialize positions & velocities
        positions = self.rng.uniform(low=self.lb, high=self.ub, size=(n, d))
        velocities = np.zeros((n, d))

        pbest_pos = positions.copy()
        pbest_fit = np.full(n, -np.inf)

        for i in range(n):
            fit = self._fitness(self._decode(positions[i]))
            pbest_fit[i] = fit
            if fit > self.best_fitness:
                self.best_fitness = fit
                self.best_pos = positions[i].copy()

        gbest_pos = self.best_pos.copy()

        for it in range(self.max_iter):
            w = self.w  # constant; may be overridden by adaptive variant
            r1 = self.rng.random((n, d))
            r2 = self.rng.random((n, d))

            velocities = (
                w * velocities
                + self.c1 * r1 * (pbest_pos - positions)
                + self.c2 * r2 * (gbest_pos - positions)
            )
            positions = self._clamp(positions + velocities)

            for i in range(n):
                fit = self._fitness(self._decode(positions[i]))
                if fit > pbest_fit[i]:
                    pbest_fit[i] = fit
                    pbest_pos[i] = positions[i].copy()
                if fit > self.best_fitness:
                    self.best_fitness = fit
                    self.best_pos = positions[i].copy()
                    gbest_pos = self.best_pos.copy()

            best_for_it = {
                "iteration": it,
                "best_fitness": self.best_fitness,
                "coverage": self.evaluator.evaluate_from_environment(self.env).coverage_rate,
            }
            self.history.append(best_for_it)
            if verbose and (it + 1) % 20 == 0:
                print(f"  PSO iter {it + 1:3d}: fitness={self.best_fitness:.4f}")

        return self._decode(self.best_pos), self.best_fitness


class ImprovedPSO(PSO):
    """PSO variant with selectable improvement strategies.

    Switches for ablation studies:
      use_chaotic_init:  Logistic map initialization (default True)
      use_adaptive_w:    Linearly decaying inertia weight (default True)
      use_mutation:      Gaussian mutation in later iterations (default True)
    """

    def __init__(
        self,
        env,
        fitness_evaluator,
        n_particles: int = 30,
        w_start: float = 0.9,
        w_end: float = 0.4,
        c1: float = 1.5,
        c2: float = 1.5,
        max_iter: int = 200,
        mutation_rate: float = 0.1,
        mutation_scale: float = 0.05,
        seed: int | None = None,
        use_chaotic_init: bool = True,
        use_adaptive_w: bool = True,
        use_mutation: bool = True,
    ):
        super().__init__(
            env, fitness_evaluator, n_particles, w_start, c1, c2, max_iter, seed
        )
        self.w_start = w_start
        self.w_end = w_end
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.use_chaotic_init = use_chaotic_init
        self.use_adaptive_w = use_adaptive_w
        self.use_mutation = use_mutation

    def optimize(self, verbose: bool = False) -> tuple[np.ndarray, float]:
        n = self.n_particles
        d = self.dim

        # Initialization
        if self.use_chaotic_init:
            chaotic_seq = init_chaotic_sequence(n * d, self.rng)
            positions = self.lb + chaotic_seq.reshape(n, d) * (self.ub - self.lb)
        else:
            positions = self.rng.uniform(low=self.lb, high=self.ub, size=(n, d))
        velocities = np.zeros((n, d))

        pbest_pos = positions.copy()
        pbest_fit = np.full(n, -np.inf)

        for i in range(n):
            fit = self._fitness(self._decode(positions[i]))
            pbest_fit[i] = fit
            if fit > self.best_fitness:
                self.best_fitness = fit
                self.best_pos = positions[i].copy()

        gbest_pos = self.best_pos.copy()

        for it in range(self.max_iter):
            if self.use_adaptive_w:
                w = adaptive_inertia_weight(it, self.max_iter, self.w_start, self.w_end)
            else:
                w = self.w

            r1 = self.rng.random((n, d))
            r2 = self.rng.random((n, d))

            velocities = (
                w * velocities
                + self.c1 * r1 * (pbest_pos - positions)
                + self.c2 * r2 * (gbest_pos - positions)
            )
            positions = self._clamp(positions + velocities)

            for i in range(n):
                fit = self._fitness(self._decode(positions[i]))
                if fit > pbest_fit[i]:
                    pbest_fit[i] = fit
                    pbest_pos[i] = positions[i].copy()
                if fit > self.best_fitness:
                    self.best_fitness = fit
                    self.best_pos = positions[i].copy()
                    gbest_pos = self.best_pos.copy()

            if self.use_mutation:
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
                label = "ImpPSO" if (self.use_chaotic_init and self.use_adaptive_w and self.use_mutation) else "AblPSO"
                print(f"  {label} iter {it + 1:3d}: fitness={self.best_fitness:.4f}")

        return self._decode(self.best_pos), self.best_fitness
