"""Smoke tests for experiment comparison outputs."""

import matplotlib

matplotlib.use("Agg")

from wsn.algorithms import PSO
from wsn.environment import WSNEnvironment
from wsn.experiments import ExperimentRunner, GridBaseline, RandomBaseline


def test_experiment_runner_exports_and_plots(tmp_path):
    env_config = {
        "width": 30,
        "height": 30,
        "grid_resolution": 10,
        "n_nodes": 4,
        "sensing_radius": 8,
        "communication_radius": 15,
    }
    runner = ExperimentRunner(env_config)

    runner.add_baseline(
        "Random",
        RandomBaseline(WSNEnvironment(**env_config, seed=1), runner.evaluator, n_trials=2),
    )
    runner.add_baseline(
        "Grid",
        GridBaseline(WSNEnvironment(**env_config, seed=1), runner.evaluator),
    )
    runner.add_algorithm(
        "PSO",
        PSO(
            WSNEnvironment(**env_config, seed=1),
            runner.evaluator,
            n_particles=4,
            max_iter=2,
            seed=1,
        ),
    )

    results = runner.run_all(verbose=False)
    assert set(results) == {"Random", "Grid", "PSO"}

    rows = runner.summary_rows()
    assert len(rows) == 3
    assert all(row["coverage"] is not None for row in rows)
    assert all(row["fitness"] is not None for row in rows)

    csv_path = tmp_path / "summary.csv"
    json_path = tmp_path / "results.json"
    png_path = tmp_path / "coverage.png"

    assert runner.save_summary_csv(str(csv_path)) == str(csv_path)
    assert runner.save_results_json(str(json_path)) == str(json_path)
    assert runner.plot_metric_bar("coverage", filename=str(png_path)) is not None

    assert csv_path.exists() and csv_path.stat().st_size > 0
    assert json_path.exists() and json_path.stat().st_size > 0
    assert png_path.exists() and png_path.stat().st_size > 0
