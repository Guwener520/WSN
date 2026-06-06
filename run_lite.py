"""Ultra-light experiment: ~1 minute, 5 seeds, 50 iters. Generates all key charts."""
import os, json, time
import numpy as np
os.makedirs("results", exist_ok=True)

from wsn.environment import WSNEnvironment
from wsn.fitness import FitnessEvaluator
from wsn.algorithms import PSO, ImprovedPSO, GWO, ImprovedGWO, WOA, ImprovedWOA
from wsn.experiments import RandomBaseline, GridBaseline
import plotly.graph_objects as go
import plotly.subplots as sp

# ── Ultra-light config ──────────────────────────────────────────────
GRID_RES = 2.0
MAX_ITER = 50
POP = 20
N_SEEDS = 5
SEEDS = [42, 123, 456, 789, 1024]

OBSTACLES = [
    {"type": "rect", "x": 25, "y": 45, "w": 50, "h": 10},
    {"type": "circle", "cx": 20, "cy": 20, "r": 12},
    {"type": "circle", "cx": 75, "cy": 75, "r": 10},
    {"type": "rect", "x": 0, "y": 85, "w": 30, "h": 15},
]
SCENARIOS = {
    "Hard":      dict(n_nodes=25, sensing_radius=8.0),
    "Medium":    dict(n_nodes=40, sensing_radius=10.0),
    "Irregular": dict(n_nodes=40, sensing_radius=10.0, obstacles=OBSTACLES),
}
COMMON = dict(width=100, height=100, grid_resolution=GRID_RES, communication_radius=24.0)

print(f"WSN Lite Experiment | iter={MAX_ITER} pop={POP} seeds={N_SEEDS} | ~1 min")
print("=" * 60)

evaluator = FitnessEvaluator(0.5, 0.3, 0.2)
all_data = {}
t0 = time.perf_counter()

for sc_name, sc_params in SCENARIOS.items():
    obs = sc_params.get("obstacles")
    print(f"\n-- {sc_name}: N={sc_params['n_nodes']} Rs={sc_params['sensing_radius']} "
          f"{f'obs={len(obs)}' if obs else ''}")

    def make_env(s):
        return WSNEnvironment(**COMMON, **sc_params, seed=s)

    # Baselines
    rand = RandomBaseline(make_env(42), evaluator, n_trials=100, seed=42).run()
    grid = GridBaseline(make_env(42), evaluator).run()

    algo_builders = {
        "PSO": lambda s: PSO(make_env(s), evaluator, n_particles=POP, max_iter=MAX_ITER, seed=s),
        "ImprovedPSO": lambda s: ImprovedPSO(make_env(s), evaluator, n_particles=POP, max_iter=MAX_ITER, seed=s),
        "GWO": lambda s: GWO(make_env(s), evaluator, n_wolves=POP, max_iter=MAX_ITER, seed=s),
        "ImprovedGWO": lambda s: ImprovedGWO(make_env(s), evaluator, n_wolves=POP, max_iter=MAX_ITER, seed=s),
        "WOA": lambda s: WOA(make_env(s), evaluator, n_whales=POP, max_iter=MAX_ITER, seed=s),
        "ImprovedWOA": lambda s: ImprovedWOA(make_env(s), evaluator, n_whales=POP, max_iter=MAX_ITER, seed=s),
    }

    sc_results = {}
    for aname, builder in algo_builders.items():
        fits, covs, best_pos_list = [], [], []
        for s in SEEDS:
            algo = builder(s)
            bp, bf = algo.optimize(verbose=False)
            algo.env.set_positions(bp)
            m = evaluator.evaluate_from_environment(algo.env)
            fits.append(bf)
            covs.append(m.coverage_rate)
        best_idx = int(np.argmax(fits))
        sc_results[aname] = {
            "mean_fitness": float(np.mean(fits)), "std_fitness": float(np.std(fits)),
            "mean_coverage": float(np.mean(covs)), "std_coverage": float(np.std(covs)),
            "best_positions": bp.tolist() if isinstance(bp, np.ndarray) else bp,
        }
        print(f"  {aname:<14} fit={np.mean(fits):.4f}+/-{np.std(fits):.4f}  cov={np.mean(covs):.4f}+/-{np.std(covs):.4f}")

    all_data[sc_name] = {"random": rand, "grid": grid, "algorithms": sc_results}

dt = time.perf_counter() - t0
print(f"\nDone in {dt:.1f}s.")

# Save data before charts
def _c(obj):
    if isinstance(obj, dict): return {k: _c(v) for k,v in obj.items()}
    if isinstance(obj, list): return [_c(v) for v in obj]
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.integer,)): return int(obj)
    return obj
with open("results/lite_data.json", "w") as f:
    json.dump(_c(all_data), f, indent=2, ensure_ascii=False)
print("Data saved. Generating charts...")

# ── Charts ──────────────────────────────────────────────────────────
COLORS = {"PSO": "#1f77b4", "ImprovedPSO": "#ff7f0e", "GWO": "#2ca02c",
          "ImprovedGWO": "#d62728", "WOA": "#9467bd", "ImprovedWOA": "#8c564b"}
algo_names = list(list(all_data.values())[0]["algorithms"].keys())
chart_count = 0

for sc_name, d in all_data.items():
    sc = SCENARIOS[sc_name]

    # Coverage bars
    fig1 = go.Figure()
    sorted_names = sorted(algo_names, key=lambda n: d["algorithms"][n]["mean_coverage"], reverse=True)
    for name in sorted_names:
        ad = d["algorithms"][name]
        fig1.add_trace(go.Bar(name=name, x=[name], y=[ad["mean_coverage"]],
                       error_y=dict(type="data", array=[ad["std_coverage"]], visible=True),
                       marker_color=COLORS.get(name),
                       text=f'{ad["mean_coverage"]:.3f}', textposition="outside"))
    rc, gc = d["random"]["mean_coverage"], d["grid"]["coverage"]
    fig1.add_hline(y=rc, line_dash="dash", line_color="gray", annotation_text=f"Random ({rc:.3f})")
    fig1.add_hline(y=gc, line_dash="dot", line_color="black", annotation_text=f"Grid ({gc:.3f})")
    fig1.update_layout(title=f"Coverage — {sc_name}", yaxis_title="Coverage Rate",
                       width=750, height=420, template="plotly_white", showlegend=False)
    fig1.write_html(f"results/lite_coverage_{sc_name}.html")
    chart_count += 1

    # Heatmap for best algo
    best_name = max(algo_names, key=lambda n: d["algorithms"][n]["mean_fitness"])
    best_pos = np.array(d["algorithms"][best_name]["best_positions"])
    env_hm = WSNEnvironment(**COMMON, **{k:v for k,v in sc.items() if k!='obstacles'},
                           obstacles=sc.get("obstacles"), seed=42)
    env_hm.set_positions(best_pos)
    cov = env_hm.coverage_matrix()
    # Reconstruct full-grid coverage (NaN for obstacle cells)
    full = -np.ones(env_hm.n_total_grid_points)
    valid_mask = env_hm._valid_mask
    full[np.where(valid_mask)[0]] = np.sum(cov, axis=0)
    cov_cnt = np.where(full.reshape(env_hm.grid_y.shape) < 0, np.nan,
                       full.reshape(env_hm.grid_y.shape))

    fig2 = sp.make_subplots(rows=1, cols=2, subplot_titles=(f"{best_name} Deployment", "Coverage Heatmap"))
    fig2.add_trace(go.Scatter(x=best_pos[:,0], y=best_pos[:,1],
                   mode="markers", marker=dict(color="red", size=8), name="Nodes"), row=1, col=1)
    obs = sc.get("obstacles")
    if obs:
        for o in obs:
            if o["type"] == "rect":
                fig2.add_shape(type="rect", x0=o["x"], y0=o["y"], x1=o["x"]+o["w"], y1=o["y"]+o["h"],
                              fillcolor="gray", opacity=0.4, line_width=0, row=1, col=1)
    fig2.add_trace(go.Heatmap(x=env_hm.grid_x[0], y=env_hm.grid_y[:,0], z=cov_cnt,
                   colorscale="YlOrRd", zmin=0, colorbar=dict(title="Coverage")), row=1, col=2)
    fig2.update_xaxes(title="X", range=[0,100], row=1, col=1)
    fig2.update_yaxes(title="Y", range=[0,100], row=1, col=1)
    fig2.update_xaxes(title="X", range=[0,100], row=1, col=2)
    fig2.update_yaxes(title="Y", range=[0,100], row=1, col=2)
    fig2.update_layout(title=f"Best: {best_name} — {sc_name}", width=1100, height=450,
                       template="plotly_white")
    fig2.write_html(f"results/lite_heatmap_{sc_name}.html")
    chart_count += 1

# Cross-scenario improvement summary
fig3 = go.Figure()
for sc_name, d in all_data.items():
    for name in algo_names:
        imp = (d["algorithms"][name]["mean_coverage"] - d["random"]["mean_coverage"]) * 100
        fig3.add_trace(go.Bar(x=[sc_name], y=[imp], name=f"{name} [{sc_name}]",
                       marker_color=COLORS.get(name),
                       text=f'+{imp:.1f}pp', textposition="outside",
                       legendgroup=name, showlegend=(sc_name == "Hard")))
fig3.update_layout(title="Coverage Improvement over Random Baseline",
                   yaxis_title="Improvement (pp)", width=850, height=450,
                   template="plotly_white", barmode="group")
fig3.write_html("results/lite_summary.html")
chart_count += 1

print(f"  {chart_count} charts saved to results/")
print(f"  Total time: {time.perf_counter()-t0:.1f}s")
print("=" * 60)
