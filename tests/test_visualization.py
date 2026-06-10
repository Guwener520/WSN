"""Smoke tests for visualization helpers."""

import matplotlib

matplotlib.use("Agg")

from wsn.algorithms import PSO
from wsn.environment import WSNEnvironment
from wsn.fitness import FitnessEvaluator
from wsn.visualization.dynamic_plots import (
    make_optimization_gif,
    make_optimization_gif_from_positions,
)
from wsn.visualization.static_plots import (
    plot_convergence,
    plot_coverage_heatmap,
    plot_deployment,
)


def test_static_plots_and_gif_from_positions(tmp_path):
    env = WSNEnvironment(
        width=30, height=30, grid_resolution=10,
        n_nodes=4, sensing_radius=8, communication_radius=15,
        seed=42,
    )
    first = env.random_deploy(seed=42)
    second = env.grid_deploy()
    env.set_positions(first)

    assert plot_deployment(env) is not None
    assert plot_coverage_heatmap(env) is not None
    assert plot_convergence(
        [[{"iteration": 0, "best_fitness": 0.2}, {"iteration": 1, "best_fitness": 0.4}]],
        labels=["PSO"],
    ) is not None

    output = tmp_path / "from_positions.gif"
    result = make_optimization_gif_from_positions(
        env, [first, second], filename=str(output), fps=2, dpi=40
    )
    assert result == str(output)
    assert output.exists()
    assert output.stat().st_size > 0


def test_make_optimization_gif_runs_optimizer(tmp_path):
    env = WSNEnvironment(
        width=30, height=30, grid_resolution=10,
        n_nodes=4, sensing_radius=8, communication_radius=15,
        seed=42,
    )
    evaluator = FitnessEvaluator()
    pso = PSO(env, evaluator, n_particles=4, max_iter=3, seed=42)

    output = tmp_path / "optimization.gif"
    result = make_optimization_gif(
        env, pso, snapshot_iters=[0, 1, 3], filename=str(output), fps=2, dpi=40
    )

    assert result == str(output)
    assert output.exists()
    assert output.stat().st_size > 0
    assert len(pso.history) == 3
