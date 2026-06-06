"""Run full comparison experiment with properly calibrated scenarios."""
import sys, os, json
import numpy as np
from wsn.environment import WSNEnvironment
from wsn.fitness import FitnessEvaluator
from wsn.algorithms import PSO, ImprovedPSO, GWO, ImprovedGWO, WOA, ImprovedWOA
from wsn.experiments import RandomBaseline, GridBaseline

os.makedirs("results", exist_ok=True)

# ---------------------------------------------------------------------------
# Calibrated scenarios: sensing radius small enough that random coverage is low,
# giving SI algorithms meaningful room to show improvement.
#
#   Hard:  N=25, Rs=8   -> random expected ~40%, SI can demonstrate clear lift
#   Medium:N=40, Rs=10  -> random expected ~72%, still substantial upside
# ---------------------------------------------------------------------------
SCENARIOS = {
    "Hard":   dict(n_nodes=25, sensing_radius=8.0, label="N=25 Rs=8"),
    "Medium": dict(n_nodes=40, sensing_radius=10.0, label="N=40 Rs=10"),
}
# Finer grid for more accurate coverage evaluation
COMMON = dict(width=100, height=100, grid_resolution=1.0, communication_radius=24.0)

# 3 independent runs per algorithm for mean +/- std
N_RUNS = 3
SEEDS = [42, 123, 789]
ALGO_MAX_ITER = 200
POP_SIZE = 30

print("=" * 70)
print("  WSN Coverage Optimization — Calibrated Experiment")
print("  Grid: 1.0  |  Runs per algorithm: {}  |  Max iter: {}".format(N_RUNS, ALGO_MAX_ITER))
print("=" * 70)

all_results = {}

for scenario_name, sc in SCENARIOS.items():
    print(f"\n{'='*70}")
    print(f"  Scenario {scenario_name}: {sc['label']}")
    print(f"{'='*70}")

    evaluator = FitnessEvaluator(w_coverage=0.5, w_redundancy=0.3, w_connectivity=0.2)

    def make_env(s=None):
        env_kwargs = {k: v for k, v in sc.items() if k != "label"}
        return WSNEnvironment(**COMMON, **env_kwargs, seed=s)

    # --- Baselines ---
    print("\n[Baselines]")
    env_r = make_env(42)
    rand = RandomBaseline(env_r, evaluator, n_trials=100, seed=42)
    r_result = rand.run()

    env_g = make_env(42)
    grid = GridBaseline(env_g, evaluator)
    g_result = grid.run()

    print(f"  Random  mean_cov={r_result['mean_coverage']:.4f} +/- {r_result['std_coverage']:.4f}")
    print(f"          best_cov={r_result['best_coverage']:.4f}  best_fit={r_result['best_fitness']:.4f}")
    print(f"  Grid    cov={g_result['coverage']:.4f}  fit={g_result['fitness']:.4f}")

    # --- Swarm Algorithms (multi-run) ---
    algo_configs = {
        "PSO":         lambda e, s: PSO(make_env(s), e, n_particles=POP_SIZE, max_iter=ALGO_MAX_ITER, seed=s),
        "ImprovedPSO": lambda e, s: ImprovedPSO(make_env(s), e, n_particles=POP_SIZE, max_iter=ALGO_MAX_ITER, seed=s),
        "GWO":         lambda e, s: GWO(make_env(s), e, n_wolves=POP_SIZE, max_iter=ALGO_MAX_ITER, seed=s),
        "ImprovedGWO": lambda e, s: ImprovedGWO(make_env(s), e, n_wolves=POP_SIZE, max_iter=ALGO_MAX_ITER, seed=s),
        "WOA":         lambda e, s: WOA(make_env(s), e, n_whales=POP_SIZE, max_iter=ALGO_MAX_ITER, seed=s),
        "ImprovedWOA": lambda e, s: ImprovedWOA(make_env(s), e, n_whales=POP_SIZE, max_iter=ALGO_MAX_ITER, seed=s),
    }

    algo_multi_results = {}
    all_histories = {}

    print("\n[Swarm Algorithms]  ({} runs each)".format(N_RUNS))
    for name, factory in algo_configs.items():
        run_fits = []
        run_covs = []
        run_reds = []
        run_conns = []
        best_pos_overall = None
        best_fit_overall = -np.inf
        best_hist = None

        for seed in SEEDS:
            algo = factory(evaluator, seed)
            best_pos, best_fit = algo.optimize(verbose=False)
            algo.env.set_positions(best_pos)
            metrics = evaluator.evaluate_from_environment(algo.env)

            run_fits.append(best_fit)
            run_covs.append(metrics.coverage_rate)
            run_reds.append(metrics.redundancy_rate)
            run_conns.append(metrics.connectivity_rate)

            if best_fit > best_fit_overall:
                best_fit_overall = best_fit
                best_pos_overall = best_pos
                best_hist = algo.history

        algo_multi_results[name] = {
            "mean_fitness": np.mean(run_fits),
            "std_fitness": np.std(run_fits),
            "mean_coverage": np.mean(run_covs),
            "std_coverage": np.std(run_covs),
            "mean_redundancy": np.mean(run_reds),
            "mean_connectivity": np.mean(run_conns),
            "best_fitness": best_fit_overall,
            "best_positions": best_pos_overall.tolist(),
        }
        all_histories[name] = [(h["iteration"], h["best_fitness"]) for h in best_hist]

        print(f"  {name:<14}  fit={np.mean(run_fits):.4f}+/-{np.std(run_fits):.4f}  "
              f"cov={np.mean(run_covs):.4f}+/-{np.std(run_covs):.4f}")

    # --- Summary table ---
    print(f"\n{'─'*75}")
    print(f"{'Method':<16} {'Fitness':>14} {'Coverage':>14} {'Redundancy':>12} {'Connectivity':>14}")
    print(f"{'─'*75}")
    print(f"{'Random (mean)':<16} {r_result['mean_fitness']:>14.4f} {r_result['mean_coverage']:>14.4f} {'--':>12} {'--':>14}")
    print(f"{'Random (best)':<16} {r_result['best_fitness']:>14.4f} {r_result['best_coverage']:>14.4f} {'--':>12} {'--':>14}")
    print(f"{'Grid':<16} {g_result['fitness']:>14.4f} {g_result['coverage']:>14.4f} {'--':>12} {'--':>14}")
    for name, ar in algo_multi_results.items():
        print(f"{name:<16} {ar['mean_fitness']:>10.4f}+/-{ar['std_fitness']:.4f} "
              f"{ar['mean_coverage']:>10.4f}+/-{ar['std_coverage']:.4f} "
              f"{ar['mean_redundancy']:>12.4f} {ar['mean_connectivity']:>14.4f}")
    print(f"{'─'*75}")

    # Improvement over Random (mean)
    best_algo = max(algo_multi_results, key=lambda k: algo_multi_results[k]["mean_fitness"])
    best_cov_improve = (algo_multi_results[best_algo]["mean_coverage"] - r_result["mean_coverage"]) * 100
    print(f"\n  >>> Best: {best_algo}  "
          f"(coverage improvement over random mean: +{best_cov_improve:.1f} pp)")

    all_results[scenario_name] = {
        "random": r_result,
        "grid": g_result,
        "algorithms": algo_multi_results,
        "histories": all_histories,
        "best_name": best_algo,
    }

# --- Save raw results ---
def convert(obj):
    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj

with open("results/experiment_results.json", "w") as f:
    json.dump(convert(all_results), f, indent=2, ensure_ascii=False)
print("\n  Raw data saved: results/experiment_results.json")

# --- Plotly visualizations ---
print("\n[Generating Plotly visualizations...]")
import plotly.graph_objects as go
import plotly.subplots as sp

for scenario_name, data in all_results.items():
    sc = SCENARIOS[scenario_name]

    # 1. Convergence curves
    fig = go.Figure()
    colors = {"PSO": "#1f77b4", "ImprovedPSO": "#ff7f0e", "GWO": "#2ca02c",
              "ImprovedGWO": "#d62728", "WOA": "#9467bd", "ImprovedWOA": "#8c564b"}
    for name, hist in data["histories"].items():
        its = [h[0] for h in hist]
        fits = [h[1] for h in hist]
        fig.add_trace(go.Scatter(x=its, y=fits, mode="lines", name=name,
                                 line=dict(color=colors.get(name), width=2)))

    fig.add_hline(y=data["random"]["mean_fitness"], line_dash="dash",
                  line_color="gray", annotation_text="Random (mean)")
    fig.add_hline(y=data["random"]["best_fitness"], line_dash="dot",
                  line_color="silver", annotation_text="Random (best of 100)")
    fig.add_hline(y=data["grid"]["fitness"], line_dash="dot",
                  line_color="black", annotation_text="Grid")

    fig.update_layout(
        title=f"Convergence Curves — Scenario {scenario_name} ({sc['label']})",
        xaxis_title="Iteration", yaxis_title="Fitness",
        width=900, height=550, template="plotly_white",
    )
    fig.write_html(f"results/convergence_{scenario_name}.html")
    print(f"  Saved: results/convergence_{scenario_name}.html")

    # 2. Bar chart: coverage comparison
    fig2 = go.Figure()
    algo_names = list(data["algorithms"].keys())
    mean_covs = [data["algorithms"][n]["mean_coverage"] for n in algo_names]
    std_covs = [data["algorithms"][n]["std_coverage"] for n in algo_names]

    fig2.add_trace(go.Bar(name="Coverage Rate", x=algo_names, y=mean_covs,
                          error_y=dict(type="data", array=std_covs, visible=True),
                          marker_color="#3498db",
                          text=[f"{c:.4f}" for c in mean_covs], textposition="outside"))

    fig2.add_hline(y=data["random"]["mean_coverage"], line_dash="dash",
                   line_color="gray", annotation_text="Random (mean)")
    fig2.add_hline(y=data["grid"]["coverage"], line_dash="dot",
                   line_color="black", annotation_text="Grid")

    fig2.update_layout(
        title=f"Coverage Comparison — Scenario {scenario_name} ({sc['label']})",
        width=900, height=500, template="plotly_white",
    )
    fig2.write_html(f"results/comparison_{scenario_name}.html")
    print(f"  Saved: results/comparison_{scenario_name}.html")

    # 3. Best deployment heatmap
    best_name = data["best_name"]
    best_pos = np.array(data["algorithms"][best_name]["best_positions"])
    viz_kwargs = {k: v for k, v in sc.items() if k != "label"}
    env_viz = WSNEnvironment(**COMMON, **viz_kwargs, seed=42)
    env_viz.set_positions(best_pos)
    cov = env_viz.coverage_matrix()
    cov_counts = np.sum(cov, axis=0).reshape(env_viz.grid_y.shape)

    fig3 = sp.make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"{best_name} Node Deployment", f"{best_name} Coverage Heatmap"),
        horizontal_spacing=0.15,
    )
    fig3.add_trace(
        go.Scatter(x=best_pos[:, 0], y=best_pos[:, 1], mode="markers",
                   marker=dict(color="red", size=8), name="Nodes"),
        row=1, col=1,
    )
    fig3.add_trace(
        go.Heatmap(x=env_viz.grid_x[0, :], y=env_viz.grid_y[:, 0],
                   z=cov_counts, colorscale="YlOrRd", zmin=0,
                   colorbar=dict(title="Coverage"), name="Coverage"),
        row=1, col=2,
    )
    fig3.update_xaxes(title_text="X", range=[0, 100], row=1, col=1)
    fig3.update_yaxes(title_text="Y", range=[0, 100], row=1, col=1)
    fig3.update_xaxes(title_text="X", range=[0, 100], row=1, col=2)
    fig3.update_yaxes(title_text="Y", range=[0, 100], row=1, col=2)
    fig3.update_layout(
        title=f"Best Deployment: {best_name} — Scenario {scenario_name} ({sc['label']})",
        width=1200, height=500, template="plotly_white",
    )
    fig3.write_html(f"results/heatmap_{scenario_name}.html")
    print(f"  Saved: results/heatmap_{scenario_name}.html")

print(f"\n{'='*70}")
print("  Experiment complete!")
print(f"  Output: {os.path.abspath('results')}/")
print(f"{'='*70}")
