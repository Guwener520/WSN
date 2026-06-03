"""Static plots: node deployment scatter, coverage heatmap, convergence curve."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def plot_deployment(
    env,
    node_positions: np.ndarray | None = None,
    show_sensing: bool = True,
    ax=None,
    title: str = "WSN Node Deployment",
):
    """Plot sensor node positions with optional sensing radius circles.

    Args:
        env: WSNEnvironment instance.
        node_positions: (N, 2) array. Uses env.node_positions if None.
        show_sensing: draw sensing radius circles if True.
        ax: optional matplotlib Axes.
        title: plot title.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    if node_positions is None:
        node_positions = env.node_positions
    if node_positions is None:
        raise ValueError("No node positions available.")

    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    ax.scatter(
        node_positions[:, 0], node_positions[:, 1],
        c="red", s=30, zorder=3, label="Nodes",
    )

    if show_sensing:
        for (x, y) in node_positions:
            circ = Circle(
                (x, y), env.sensing_radius,
                fill=True, alpha=0.08, edgecolor="blue", linewidth=0.3,
            )
            ax.add_patch(circ)

    ax.legend(loc="upper right")
    return ax


def plot_coverage_heatmap(
    env,
    node_positions: np.ndarray | None = None,
    ax=None,
    title: str = "Coverage Heatmap",
):
    """Plot a heatmap showing how many nodes cover each grid point.

    Args:
        env: WSNEnvironment instance.
        node_positions: (N, 2) array. Uses env.node_positions if None.
        ax: optional matplotlib Axes.
        title: plot title.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    if node_positions is not None:
        env.set_positions(node_positions)

    cov = env.coverage_matrix()
    coverage_counts = np.sum(cov, axis=0).reshape(env.grid_y.shape)

    im = ax.pcolormesh(
        env.grid_x, env.grid_y, coverage_counts,
        cmap="YlOrRd", shading="auto",
        vmin=0,
    )
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.colorbar(im, ax=ax, label="Coverage count")
    return ax


def plot_convergence(
    histories: list[dict],
    labels: list[str] | None = None,
    ax=None,
    title: str = "Convergence Curve",
):
    """Plot fitness vs iteration for one or more algorithm histories.

    Each history dict should have keys "iteration" and "best_fitness".
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    if labels is None:
        labels = [f"Algorithm {i+1}" for i in range(len(histories))]

    for hist, label in zip(histories, labels):
        its = [h["iteration"] for h in hist]
        fits = [h["best_fitness"] for h in hist]
        ax.plot(its, fits, label=label, linewidth=1.5)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Fitness")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax
