"""WSN environment: grid-based monitoring area and binary sensing model."""

import numpy as np


class WSNEnvironment:
    """Grid-based monitoring area with N sensor nodes.

    The area is discretized into a grid. Each node senses within a
    circular disk of radius Rs (binary sensing model).
    """

    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        grid_resolution: float = 1.0,
        n_nodes: int = 50,
        sensing_radius: float = 12.0,
        communication_radius: float = 24.0,
        seed: int | None = None,
    ):
        self.width = width
        self.height = height
        self.grid_resolution = grid_resolution
        self.n_nodes = n_nodes
        self.sensing_radius = sensing_radius
        self.communication_radius = communication_radius

        self.rng = np.random.default_rng(seed)

        # Build grid point coordinates
        x = np.arange(0, width + grid_resolution, grid_resolution)
        y = np.arange(0, height + grid_resolution, grid_resolution)
        self.grid_x, self.grid_y = np.meshgrid(x, y)
        self.grid_points = np.column_stack(
            (self.grid_x.ravel(), self.grid_y.ravel())
        )
        self.n_grid_points = len(self.grid_points)

        self.node_positions: np.ndarray | None = None  # shape (n_nodes, 2)

    # ------------------------------------------------------------------
    # Node deployment
    # ------------------------------------------------------------------
    def random_deploy(self, seed: int | None = None) -> np.ndarray:
        """Deploy nodes uniformly at random within the area."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.node_positions = self.rng.uniform(
            low=[0, 0], high=[self.width, self.height], size=(self.n_nodes, 2)
        )
        return self.node_positions

    def grid_deploy(self) -> np.ndarray:
        """Deploy nodes on an approximate regular grid."""
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
        """Manually set node positions, shape (n_nodes, 2)."""
        if positions.shape != (self.n_nodes, 2):
            raise ValueError(
                f"Expected shape ({self.n_nodes}, 2), got {positions.shape}"
            )
        self.node_positions = positions.astype(float).copy()

    # ------------------------------------------------------------------
    # Binary sensing model
    # ------------------------------------------------------------------
    def coverage_matrix(self) -> np.ndarray:
        """Return (n_nodes, n_grid_points) bool matrix: True if node covers grid point."""
        if self.node_positions is None:
            raise RuntimeError("Deploy nodes first.")
        diffs = self.grid_points[np.newaxis, :, :] - self.node_positions[:, np.newaxis, :]
        dists = np.linalg.norm(diffs, axis=2)  # (n_nodes, n_grid_points)
        return dists <= self.sensing_radius

    def covered_grid_mask(self) -> np.ndarray:
        """Return boolean array of length n_grid_points: True if covered by >=1 node."""
        cov = self.coverage_matrix()
        return np.any(cov, axis=0)

    # ------------------------------------------------------------------
    # Communication graph
    # ------------------------------------------------------------------
    def communication_adjacency(self) -> np.ndarray:
        """Return (n_nodes, n_nodes) bool adjacency matrix based on communication radius."""
        if self.node_positions is None:
            raise RuntimeError("Deploy nodes first.")
        diffs = self.node_positions[np.newaxis, :, :] - self.node_positions[:, np.newaxis, :]
        dists = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(dists, np.inf)
        return dists <= self.communication_radius

    def connectivity_rate(self) -> float:
        """Fraction of node pairs that are connected (within communication radius)."""
        adj = self.communication_adjacency()
        n = self.n_nodes
        return float(np.sum(adj) / (n * (n - 1)))
