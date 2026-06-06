"""Quick experiment: ~2-3 minutes on CPU, <1 minute on GPU."""
import sys, os, json, time
import numpy as np
os.makedirs("results", exist_ok=True)

from wsn.environment import WSNEnvironment
from wsn.fitness import FitnessEvaluator
from wsn.algorithms import PSO, ImprovedPSO, GWO, ImprovedGWO, WOA, ImprovedWOA
from wsn.experiments import RandomBaseline, GridBaseline

# Check GPU
try:
    import cupy as cp
    HAS_GPU = True
    print(f"GPU: {cp.cuda.runtime.getDeviceCount()} device(s) — CuPy {cp.__version__}")
except ImportError:
    HAS_GPU = False
    print("GPU: not available, using CPU")

# ---- Lightweight config ----
SCENARIOS = {
    "Hard":   dict(n_nodes=25, sensing_radius=8.0, label="N=25 Rs=8"),
    "Medium": dict(n_nodes=40, sensing_radius=10.0, label="N=40 Rs=10"),
}
COMMON = dict(width=100, height=100, grid_resolution=2.0,
              communication_radius=24.0, use_gpu=HAS_GPU)
MAX_ITER = 80
N_PARTICLES = 20

print(f"Config: grid=2.0, max_iter={MAX_ITER}, particles={N_PARTICLES}")
print("=" * 60)

all_results = {}

for sc_name, sc in SCENARIOS.items():
    print(f"\n--- Scenario {sc_name}: {sc['label']} ---")

    evaluator = FitnessEvaluator(w_coverage=0.5, w_redundancy=0.3, w_connectivity=0.2)

    def make_env(s=None):
        kwargs = {k: v for k, v in sc.items() if k != "label"}
        return WSNEnvironment(**COMMON, **kwargs, seed=s)

    # Baselines
    env_r = make_env(42)
    rand = RandomBaseline(env_r, evaluator, n_trials=50, seed=42)
    r = rand.run()
    env_g = make_env(42)
    grid = GridBaseline(env_g, evaluator)
    g = grid.run()

    print(f"  Random mean_cov={r['mean_coverage']:.4f} +/- {r['std_coverage']:.4f}")
    print(f"  Grid   cov={g['coverage']:.4f}  fit={g['fitness']:.4f}")

    # Algorithms (1 seed each for speed)
    algos = {
        "PSO":         PSO(make_env(42), evaluator, n_particles=N_PARTICLES, max_iter=MAX_ITER, seed=42),
        "ImprovedPSO": ImprovedPSO(make_env(42), evaluator, n_particles=N_PARTICLES, max_iter=MAX_ITER, seed=42),
        "GWO":         GWO(make_env(42), evaluator, n_wolves=N_PARTICLES, max_iter=MAX_ITER, seed=42),
        "ImprovedGWO": ImprovedGWO(make_env(42), evaluator, n_wolves=N_PARTICLES, max_iter=MAX_ITER, seed=42),
        "WOA":         WOA(make_env(42), evaluator, n_whales=N_PARTICLES, max_iter=MAX_ITER, seed=42),
        "ImprovedWOA": ImprovedWOA(make_env(42), evaluator, n_whales=N_PARTICLES, max_iter=MAX_ITER, seed=42),
    }

    algo_res = {}
    histories = {}
    best_positions = {}

    t0_total = time.perf_counter()
    for name, algo in algos.items():
        t0 = time.perf_counter()
        best_pos, best_fit = algo.optimize(verbose=False)
        dt = time.perf_counter() - t0

        algo.env.set_positions(best_pos)
        metrics = evaluator.evaluate_from_environment(algo.env)

        algo_res[name] = {
            "fitness": round(best_fit, 4),
            "coverage": round(metrics.coverage_rate, 4),
            "redundancy": round(metrics.redundancy_rate, 4),
            "connectivity": round(metrics.connectivity_rate, 4),
            "time_s": round(dt, 1),
        }
        histories[name] = [(h["iteration"], h["best_fitness"]) for h in algo.history]
        best_positions[name] = best_pos.tolist()
        print(f"  {name:<14} fit={best_fit:.4f}  cov={metrics.coverage_rate:.4f}  "
              f"time={dt:.1f}s")

    dt_total = time.perf_counter() - t0_total
    print(f"  Total time: {dt_total:.1f}s")

    # Summary
    print(f"\n  {'Method':<16} {'Fitness':>8} {'Coverage':>10}")
    print(f"  {'─'*36}")
    print(f"  {'Random (mean)':<16} {r['mean_fitness']:>8.4f} {r['mean_coverage']:>10.4f}")
    print(f"  {'Grid':<16} {g['fitness']:>8.4f} {g['coverage']:>10.4f}")
    best_name = max(algo_res, key=lambda k: algo_res[k]["fitness"])
    for n, ar in algo_res.items():
        marker = " <<<" if n == best_name else ""
        print(f"  {n:<16} {ar['fitness']:>8.4f} {ar['coverage']:>10.4f}{marker}")

    # Improvement
    best_cov = algo_res[best_name]["coverage"]
    rand_cov = r["mean_coverage"]
    print(f"\n  >>> {best_name} improves coverage by +{(best_cov-rand_cov)*100:.1f} pp "
          f"over random ({rand_cov*100:.1f}% -> {best_cov*100:.1f}%)")

    all_results[sc_name] = {
        "random": r, "grid": g, "algorithms": algo_res,
        "histories": histories, "best_positions": best_positions, "best_name": best_name,
    }

# ---- Plots (Plotly) ----
print("\n[Generating plots...]")
import plotly.graph_objects as go
import plotly.subplots as sp

for sc_name, data in all_results.items():
    sc = SCENARIOS[sc_name]

    # Convergence
    fig = go.Figure()
    colors = {"PSO": "#1f77b4", "ImprovedPSO": "#ff7f0e", "GWO": "#2ca02c",
              "ImprovedGWO": "#d62728", "WOA": "#9467bd", "ImprovedWOA": "#8c564b"}
    for name, hist in data["histories"].items():
        fig.add_trace(go.Scatter(x=[h[0] for h in hist], y=[h[1] for h in hist],
                                 mode="lines", name=name,
                                 line=dict(color=colors.get(name), width=2)))
    fig.add_hline(y=data["random"]["mean_fitness"], line_dash="dash",
                  line_color="gray", annotation_text="Random (mean)")
    fig.add_hline(y=data["grid"]["fitness"], line_dash="dot",
                  line_color="black", annotation_text="Grid")
    fig.update_layout(title=f"Convergence — {sc_name} ({sc['label']})",
                      xaxis_title="Iteration", yaxis_title="Fitness",
                      width=800, height=500, template="plotly_white")
    fig.write_html(f"results/quick_convergence_{sc_name}.html")

    # Heatmap for best
    best_name = data["best_name"]
    best_pos = np.array(data["best_positions"][best_name])
    viz_kwargs = {k: v for k, v in sc.items() if k != "label"}
    env_v = WSNEnvironment(**COMMON, **viz_kwargs, seed=42)
    env_v.set_positions(best_pos)
    cov = env_v.coverage_matrix()
    cov_counts = np.sum(cov, axis=0).reshape(env_v.grid_y.shape)

    fig2 = sp.make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"{best_name} Deployment", f"{best_name} Coverage Heatmap"),
    )
    fig2.add_trace(go.Scatter(x=best_pos[:,0], y=best_pos[:,1],
                   mode="markers", marker=dict(color="red", size=8), name="Nodes"),
                   row=1, col=1)
    fig2.add_trace(go.Heatmap(x=env_v.grid_x[0], y=env_v.grid_y[:,0],
                   z=cov_counts, colorscale="YlOrRd", zmin=0,
                   colorbar=dict(title="Coverage")), row=1, col=2)
    fig2.update_xaxes(title="X", range=[0, 100], row=1, col=1)
    fig2.update_yaxes(title="Y", range=[0, 100], row=1, col=1)
    fig2.update_xaxes(title="X", range=[0, 100], row=1, col=2)
    fig2.update_yaxes(title="Y", range=[0, 100], row=1, col=2)
    fig2.update_layout(title=f"Best: {best_name} — {sc_name}",
                       width=1100, height=450, template="plotly_white")
    fig2.write_html(f"results/quick_heatmap_{sc_name}.html")

# Save JSON
def _c(obj):
    if isinstance(obj, dict): return {k: _c(v) for k,v in obj.items()}
    if isinstance(obj, list): return [_c(v) for v in obj]
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.integer,)): return int(obj)
    return obj

with open("results/quick_results.json", "w") as f:
    json.dump(_c(all_results), f, indent=2, ensure_ascii=False)

print(f"\nDone! Output: {os.path.abspath('results')}/")
print(f"  quick_convergence_Hard.html, quick_convergence_Medium.html")
print(f"  quick_heatmap_Hard.html, quick_heatmap_Medium.html")
print(f"  quick_results.json")
