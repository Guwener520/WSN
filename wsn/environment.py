"""WSN environment: grid-based monitoring area and binary sensing model.

Supports optional GPU acceleration via CuPy for distance computations.
Set use_gpu=True to use GPU, falls back to CPU if CuPy is unavailable.

Obstacles: pass obstacles=[{"type":"rect","x":x0,"y":y0,"w":w,"h":h}, ...]
or obstacles=[{"type":"circle","cx":x,"cy":y,"r":r}, ...] to mark regions
as invalid (no grid points, no coverage contribution). This models walls,
pillars, lakes, etc. in realistic deployment scenarios.
"""

import numpy as np

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    cp = None
    _HAS_CUPY = False


class WSNEnvironment:
    """Grid-based monitoring area with N sensor nodes."""

    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        grid_resolution: float = 1.0,
        n_nodes: int = 50,
        sensing_radius: float = 12.0,
        communication_radius: float = 24.0,
        seed: int | None = None,
        use_gpu: bool = False,
        obstacles: list[dict] | None = None,
    ):
        self.width = width
        self.height = height
        self.grid_resolution = grid_resolution
        self.n_nodes = n_nodes
        self.sensing_radius = sensing_radius
        self.communication_radius = communication_radius
        self.use_gpu = use_gpu and _HAS_CUPY
        self.obstacles = obstacles or []

        self.rng = np.random.default_rng(seed)

        # Build grid point coordinates
        x = np.arange(0, width + grid_resolution, grid_resolution)
        y = np.arange(0, height + grid_resolution, grid_resolution)
        self.grid_x, self.grid_y = np.meshgrid(x, y)
        all_points = np.column_stack((self.grid_x.ravel(), self.grid_y.ravel()))

        # Compute valid mask (points NOT inside any obstacle)
        self._valid_mask = self._compute_valid_mask(all_points)
        self.grid_points = all_points[self._valid_mask]    # only valid grid points
        self.n_grid_points = len(self.grid_points)
        self.n_total_grid_points = len(all_points)

        # GPU cache
        self._grid_points_gpu: "cp.ndarray | None" = None  # type: ignore[name-defined]

        self.node_positions: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Obstacle handling
    # ------------------------------------------------------------------
    def _compute_valid_mask(self, points: np.ndarray) -> np.ndarray:
        """Return boolean mask: True for points NOT inside any obstacle."""
        if not self.obstacles:
            return np.ones(len(points), dtype=bool)

        inside_any = np.zeros(len(points), dtype=bool)
        for obs in self.obstacles:
            if obs["type"] == "rect":
                inside = (
                    (points[:, 0] >= obs["x"])
                    & (points[:, 0] <= obs["x"] + obs["w"])
                    & (points[:, 1] >= obs["y"])
                    & (points[:, 1] <= obs["y"] + obs["h"])
                )
            elif obs["type"] == "circle":
                d2 = (points[:, 0] - obs["cx"]) ** 2 + (points[:, 1] - obs["cy"]) ** 2
                inside = d2 <= obs["r"] ** 2
            else:
                raise ValueError(f"Unknown obstacle type: {obs['type']}")
            inside_any |= inside

        return ~inside_any

    @property
    def valid_area_fraction(self) -> float:
        """Fraction of grid points that are valid (not blocked by obstacles)."""
        return self.n_grid_points / self.n_total_grid_points

    # ------------------------------------------------------------------
    # Node deployment
    # ------------------------------------------------------------------
    def _is_valid_position(self, pos: np.ndarray) -> np.ndarray:
        """Check if positions are outside obstacles. pos shape (K, 2)."""
        if not self.obstacles:
            return np.ones(len(pos), dtype=bool)
        valid = np.ones(len(pos), dtype=bool)
        for obs in self.obstacles:
            if obs["type"] == "rect":
                inside = (
                    (pos[:, 0] > obs["x"])
                    & (pos[:, 0] < obs["x"] + obs["w"])
                    & (pos[:, 1] > obs["y"])
                    & (pos[:, 1] < obs["y"] + obs["h"])
                )
            elif obs["type"] == "circle":
                d2 = (pos[:, 0] - obs["cx"]) ** 2 + (pos[:, 1] - obs["cy"]) ** 2
                inside = d2 < obs["r"] ** 2
            valid &= ~inside
        return valid

    def random_deploy(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if not self.obstacles:
            self.node_positions = self.rng.uniform(
                low=[0, 0], high=[self.width, self.height], size=(self.n_nodes, 2)
            )
            return self.node_positions

        # Rejection sampling for obstacle-avoidant placement
        positions = np.empty((self.n_nodes, 2))
        placed = 0
        while placed < self.n_nodes:
            candidates = self.rng.uniform(
                low=[0, 0], high=[self.width, self.height],
                size=(self.n_nodes - placed, 2),
            )
            valid = self._is_valid_position(candidates)
            to_place = min(np.sum(valid), self.n_nodes - placed)
            positions[placed : placed + to_place] = candidates[valid][:to_place]
            placed += to_place

        self.node_positions = positions
        return self.node_positions

    def grid_deploy(self) -> np.ndarray:
        cols = int(np.ceil(np.sqrt(self.n_nodes)))
        rows = int(np.ceil(self.n_nodes / cols))
        x_positions = np.linspace(
            self.width / (2 * cols), self.width * (1 - 1 / (2 * cols)), cols
        )
        y_positions = np.linspace(
            self.height / (2 * rows), self.height * (1 - 1 / (2 * rows)), rows
        )
        xx, yy = np.meshgrid(x_positions, y_positions)
        positions = np.column_stack((xx.ravel(), yy.ravel()))
        # Filter out positions inside obstacles
        valid = self._is_valid_position(positions)
        positions = positions[valid]
        # Pad with random valid positions if we lost too many
        if len(positions) < self.n_nodes:
            extra = self.rng.uniform(
                low=[0, 0], high=[self.width, self.height],
                size=(self.n_nodes * 5, 2),  # oversample then filter
            )
            extra_valid = self._is_valid_position(extra)
            extra = extra[extra_valid]
            positions = np.vstack([positions, extra])
        positions = positions[: self.n_nodes]
        self.node_positions = positions
        return self.node_positions

    def set_positions(self, positions: np.ndarray) -> None:
        if positions.shape != (self.n_nodes, 2):
            raise ValueError(
                f"Expected shape ({self.n_nodes}, 2), got {positions.shape}"
            )
        self.node_positions = positions.astype(float).copy()

    # ------------------------------------------------------------------
    # Binary sensing model (with GPU path)
    # ------------------------------------------------------------------
    def _get_grid_gpu(self):
        """Lazily upload grid points to GPU."""
        if self._grid_points_gpu is None and cp is not None:
            self._grid_points_gpu = cp.asarray(self.grid_points)
        return self._grid_points_gpu

    def coverage_matrix(self) -> np.ndarray:
        """Return (n_nodes, n_grid_points) bool matrix."""
        if self.node_positions is None:
            raise RuntimeError("Deploy nodes first.")

        if self.use_gpu and cp is not None:
            nodes_gpu = cp.asarray(self.node_positions)           # (N, 2)
            grid_gpu = self._get_grid_gpu()                        # (M, 2)
            # (N, M, 2) = nodes[:,None] - grid[None,:]
            diffs = nodes_gpu[:, None, :] - grid_gpu[None, :, :]
            dists = cp.linalg.norm(diffs, axis=2)                  # (N, M)
            result = cp.asnumpy(dists <= self.sensing_radius)
        else:
            diffs = self.grid_points[np.newaxis, :, :] - self.node_positions[:, np.newaxis, :]
            dists = np.linalg.norm(diffs, axis=2)
            result = dists <= self.sensing_radius

        return result

    def covered_grid_mask(self) -> np.ndarray:
        cov = self.coverage_matrix()
        return np.any(cov, axis=0)

    # ------------------------------------------------------------------
    # Communication graph (with GPU path)
    # ------------------------------------------------------------------
    def communication_adjacency(self) -> np.ndarray:
        if self.node_positions is None:
            raise RuntimeError("Deploy nodes first.")

        if self.use_gpu and cp is not None:
            nodes_gpu = cp.asarray(self.node_positions)
            diffs = nodes_gpu[:, None, :] - nodes_gpu[None, :, :]
            dists = cp.linalg.norm(diffs, axis=2)
            cp.fill_diagonal(dists, float("inf"))
            result = cp.asnumpy(dists <= self.communication_radius)
        else:
            diffs = self.node_positions[np.newaxis, :, :] - self.node_positions[:, np.newaxis, :]
            dists = np.linalg.norm(diffs, axis=2)
            np.fill_diagonal(dists, np.inf)
            result = dists <= self.communication_radius

        return result

    def connectivity_rate(self) -> float:
        adj = self.communication_adjacency()
        n = self.n_nodes
        return float(np.sum(adj) / (n * (n - 1)))
