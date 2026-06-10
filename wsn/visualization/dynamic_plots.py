"""Dynamic visualization: create GIF of the optimization process."""

import os
import numpy as np
import matplotlib.pyplot as plt

from wsn.visualization.static_plots import plot_deployment


def _snapshot_schedule(max_iter: int, snapshot_iters: list[int] | None) -> set[int]:
    if snapshot_iters is None:
        step = max(1, max_iter // 30)
        values = list(range(0, max_iter + 1, step))
    else:
        values = [int(i) for i in snapshot_iters]

    values.extend([0, max_iter])
    return {min(max(i, 0), max_iter) for i in values}


def _deployment_frame(env, positions: np.ndarray, title: str, dpi: int) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    plot_deployment(env, node_positions=positions, ax=ax, title=title)
    fig.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return frame


def make_optimization_gif(
    env,
    algorithm,
    snapshot_iters: list[int] | None = None,
    filename: str = "optimization.gif",
    fps: int = 5,
    dpi: int = 80,
) -> str:
    """Run optimization and capture snapshots, then assemble a GIF.

    Args:
        env: WSNEnvironment instance.
        algorithm: a SwarmOptimizer instance (must have .history).
        snapshot_iters: which iterations to snapshot (default: every 10).
        filename: output GIF filename.
        fps: frames per second.
        dpi: resolution.

    Returns:
        Path to the saved GIF file.

    Note: Calling optimize() will mutate the algorithm's state and env.
    If you need the final result separately, save it before calling this.
    """
    snapshot_set = _snapshot_schedule(algorithm.max_iter, snapshot_iters)
    positions_over_time: list[np.ndarray] = []
    titles: list[str] = []

    def capture(iteration: int, positions: np.ndarray, fitness: float, _: dict) -> None:
        if iteration not in snapshot_set:
            return
        positions_over_time.append(positions)
        titles.append(f"Iteration {iteration} | best fitness={fitness:.4f}")

    best_positions, best_fitness = algorithm.optimize(verbose=False, callback=capture)
    if algorithm.max_iter not in snapshot_set or not positions_over_time:
        positions_over_time.append(best_positions)
        titles.append(f"Iteration {algorithm.max_iter} | best fitness={best_fitness:.4f}")

    env.set_positions(best_positions)
    return make_optimization_gif_from_positions(
        env,
        positions_over_time,
        filename=filename,
        fps=fps,
        dpi=dpi,
        titles=titles,
    )


def make_optimization_gif_from_positions(
    env,
    positions_over_time: list[np.ndarray],
    filename: str = "optimization.gif",
    fps: int = 5,
    dpi: int = 80,
    titles: list[str] | None = None,
) -> str:
    """Build a GIF from a pre-recorded list of node position snapshots.

    Args:
        env: WSNEnvironment instance.
        positions_over_time: list of (N, 2) arrays at each frame.
        filename: output path.
        fps: frames per second.
        dpi: resolution.
        titles: optional frame titles; length must match positions_over_time.

    Returns:
        Path to the saved GIF.
    """
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio

    if not positions_over_time:
        raise ValueError("positions_over_time must contain at least one frame.")
    if fps <= 0:
        raise ValueError("fps must be greater than zero.")
    if titles is not None and len(titles) != len(positions_over_time):
        raise ValueError("titles length must match positions_over_time length.")

    frames = []
    for idx, positions in enumerate(positions_over_time):
        title = titles[idx] if titles is not None else f"Iteration {idx}"
        frames.append(_deployment_frame(env, positions, title, dpi))

    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    imageio.mimsave(filename, frames, duration=max(1, int(1000 / fps)), loop=0)
    return filename
