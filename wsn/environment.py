"""WSN environment: grid-based monitoring area and binary sensing model.

Supports optional GPU acceleration via CuPy for distance computations.
Set use_gpu=True to use GPU, falls back to CPU if CuPy is unavailable.
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
    ):
        self.width = width
        self.height = height
        self.grid_resolution = grid_resolution
        self.n_nodes = n_nodes
        self.sensing_radius = sensing_radius
        self.communication_radius = communication_radius
        self.use_gpu = use_gpu and _HAS_CUPY

        self.rng = np.random.default_rng(seed)

        # Build grid point coordinates
        x = np.arange(0, width + grid_resolution, grid_resolution)
        y = np.arange(0, height + grid_resolution, grid_resolution)
        self.grid_x, self.grid_y = np.meshgrid(x, y)
        self.grid_points = np.column_stack(
            (self.grid_x.ravel(), self.grid_y.ravel())
        )
        self.n_grid_points = len(self.grid_points)

        # GPU cache
        self._grid_points_gpu: "cp.ndarray | None" = None  # type: ignore[name-defined]

        self.node_positions: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Node deployment
    # ------------------------------------------------------------------
    def random_deploy(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.node_positions = self.rng.uniform(
            low=[0, 0], high=[self.width, self.height], size=(self.n_nodes, 2)
        )
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
        return positions[: self.n_nodes]

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
