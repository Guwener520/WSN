"""Comprehensive experiment: multi-seed statistics + parameter sensitivity + charts.

Output: results/full_*.html, results/full_results.json
Runtime: ~10-12 minutes on CPU.
"""
import sys, os, json, time, itertools
import numpy as np
os.makedirs("results", exist_ok=True)

from wsn.environment import WSNEnvironment
from wsn.fitness import FitnessEvaluator
from wsn.algorithms import PSO, ImprovedPSO, GWO, ImprovedGWO, WOA, ImprovedWOA
from wsn.experiments import RandomBaseline, GridBaseline

# ── Config ──────────────────────────────────────────────────────────
GRID_RES = 2.0
MAX_ITER = 100
POP_SIZE = 25
N_RUNS = 15
EVAL_SEEDS = list(range(N_RUNS))
BASE_SEED = 42

OBSTACLES = [
    {"type": "rect", "x": 25, "y": 45, "w": 50, "h": 10},
    {"type": "circle", "cx": 20, "cy": 20, "r": 12},
    {"type": "circle", "cx": 75, "cy": 75, "r": 10},
    {"type": "rect", "x": 0, "y": 85, "w": 30, "h": 15},
]

SCENARIOS = {
    "Hard":    dict(n_nodes=25, sensing_radius=8.0),
    "Medium":  dict(n_nodes=40, sensing_radius=10.0),
    "Irregular": dict(n_nodes=40, sensing_radius=10.0, obstacles=OBSTACLES),
}
COMMON = dict(width=100, height=100, grid_resolution=GRID_RES, communication_radius=24.0)

print("=" * 65)
print("  WSN Full Experiment")
print(f"  Scenarios: {list(SCENARIOS)} | Grid: {GRID_RES}")
print(f"  max_iter={MAX_ITER} | population={POP_SIZE} | seeds={N_RUNS}")
print("=" * 65)

# ── Part 1: Multi-Seed Comparison ───────────────────────────────────
print("\n" + "=" * 65)
print("  PART 1: Multi-Seed Comparison (~10 min)")
print("=" * 65)

all_data = {}
t_start = time.perf_counter()

for sc_name, sc_params in SCENARIOS.items():
    print(f"\n{'─'*50}")
    obstacles = sc_params.get("obstacles")
    print(f"  {sc_name}: N={sc_params['n_nodes']}, Rs={sc_params['sensing_radius']}")
    if obstacles:
        test_env = WSNEnvironment(**COMMON,
            **{k:v for k,v in sc_params.items() if k != 'obstacles'},
            obstacles=obstacles, seed=0)
        print(f"  Obstacles: {len(obstacles)} regions, valid_area={test_env.valid_area_fraction:.1%}")
    print(f"{'─'*50}")

    evaluator = FitnessEvaluator(w_coverage=0.5, w_redundancy=0.3, w_connectivity=0.2)

    def make_env(seed):
        kwargs = {k: v for k, v in sc_params.items()}
        return WSNEnvironment(**COMMON, **kwargs, seed=seed)

    # Baselines
    env_rand = make_env(42)
    rand = RandomBaseline(env_rand, evaluator, n_trials=200, seed=42)
    rand_res = rand.run()
    env_grid = make_env(42)
    grid = GridBaseline(env_grid, evaluator)
    grid_res = grid.run()

    print(f"  Random (mean of 200) cov={rand_res['mean_coverage']:.4f}+/-{rand_res['std_coverage']:.4f}")
    print(f"  Grid   cov={grid_res['coverage']:.4f} fit={grid_res['fitness']:.4f}")

    algo_builders = {
        "PSO":         lambda s: PSO(make_env(s), evaluator, n_particles=POP_SIZE, max_iter=MAX_ITER, seed=s),
        "ImprovedPSO": lambda s: ImprovedPSO(make_env(s), evaluator, n_particles=POP_SIZE, max_iter=MAX_ITER, seed=s),
        "GWO":         lambda s: GWO(make_env(s), evaluator, n_wolves=POP_SIZE, max_iter=MAX_ITER, seed=s),
        "ImprovedGWO": lambda s: ImprovedGWO(make_env(s), evaluator, n_wolves=POP_SIZE, max_iter=MAX_ITER, seed=s),
        "WOA":         lambda s: WOA(make_env(s), evaluator, n_whales=POP_SIZE, max_iter=MAX_ITER, seed=s),
        "ImprovedWOA": lambda s: ImprovedWOA(make_env(s), evaluator, n_whales=POP_SIZE, max_iter=MAX_ITER, seed=s),
    }

    sc_data = {}
    for algo_name, builder in algo_builders.items():
        seeds_data = []
        for seed in EVAL_SEEDS:
            algo = builder(seed)
            best_pos, best_fit = algo.optimize(verbose=False)
            algo.env.set_positions(best_pos)
            metrics = evaluator.evaluate_from_environment(algo.env)

            target = best_fit * 0.95
            conv_iter = MAX_ITER
            for h in algo.history:
                if h["best_fitness"] >= target:
                    conv_iter = h["iteration"]
                    break

            seeds_data.append({
                "seed": seed,
                "fitness": float(best_fit),
                "coverage": float(metrics.coverage_rate),
                "redundancy": float(metrics.redundancy_rate),
                "connectivity": float(metrics.connectivity_rate),
                "convergence_iter": conv_iter,
                "best_positions": best_pos.tolist(),
            })

        fits = [s["fitness"] for s in seeds_data]
        covs = [s["coverage"] for s in seeds_data]
        convs = [s["convergence_iter"] for s in seeds_data]

        sc_data[algo_name] = {
            "seeds": seeds_data,
            "mean_fitness": float(np.mean(fits)), "std_fitness": float(np.std(fits)),
            "mean_coverage": float(np.mean(covs)), "std_coverage": float(np.std(covs)),
            "mean_convergence": float(np.mean(convs)), "std_convergence": float(np.std(convs)),
            "best_seed_idx": int(np.argmax(fits)),
        }
        print(f"  {algo_name:<14} fit={np.mean(fits):.4f}+/-{np.std(fits):.4f}  "
              f"cov={np.mean(covs):.4f}+/-{np.std(covs):.4f}  "
              f"conv={np.mean(convs):.1f}+/-{np.std(convs):.1f}")

    all_data[sc_name] = {
        "params": {k: v for k, v in sc_params.items() if k != "obstacles"},
        "obstacles": obstacles,
        "random_baseline": rand_res,
        "grid_baseline": grid_res,
        "algorithms": sc_data,
    }

    ranked = sorted(sc_data.items(), key=lambda x: x[1]["mean_fitness"], reverse=True)
    print(f"\n  Rank | {'Algorithm':<14} | {'Fitness':^16} | {'Coverage':^16} | {'Convergence':^12}")
    for i, (name, d) in enumerate(ranked):
        print(f"  {i+1:>4} | {name:<14} | {d['mean_fitness']:.4f}+/-{d['std_fitness']:.4f} | "
              f"{d['mean_coverage']:.4f}+/-{d['std_coverage']:.4f} | "
              f"{d['mean_convergence']:.0f}+/-{d['std_convergence']:.0f}")

dt_total = time.perf_counter() - t_start
print(f"\n  Part 1 done in {dt_total:.1f}s ({dt_total/60:.1f} min)")

# ── Part 2: Parameter Sensitivity ───────────────────────────────────
print("\n" + "=" * 65)
print("  PART 2: Parameter Sensitivity — ImprovedPSO (~2 min)")
print("=" * 65)

env_kwargs_m = {k: v for k, v in SCENARIOS["Medium"].items()}
sens_eval = FitnessEvaluator(0.5, 0.3, 0.2)

mutation_rates = [0.05, 0.10, 0.15, 0.20, 0.30]
mutation_scales = [0.02, 0.05, 0.10, 0.15]
SENS_SEEDS = [42, 123, 456]

sens_results = {}
for mr in mutation_rates:
    for ms in mutation_scales:
        fits = []
        for s in SENS_SEEDS:
            env = WSNEnvironment(**COMMON, **env_kwargs_m, seed=s)
            ipso = ImprovedPSO(env, sens_eval, n_particles=POP_SIZE,
                              max_iter=MAX_ITER, seed=s,
                              mutation_rate=mr, mutation_scale=ms)
            _, best_f = ipso.optimize(verbose=False)
            fits.append(best_f)
        sens_results[f"mr={mr:.2f}_ms={ms:.2f}"] = {
            "mutation_rate": mr, "mutation_scale": ms,
            "mean_fitness": float(np.mean(fits)), "std_fitness": float(np.std(fits)),
        }

top5 = sorted(sens_results.items(), key=lambda x: x[1]["mean_fitness"], reverse=True)[:5]
print(f"\n  Top 5 ImprovedPSO configs:")
for i, (key, d) in enumerate(top5):
    print(f"  {i+1}. {key}  fit={d['mean_fitness']:.4f}+/-{d['std_fitness']:.4f}")
print(f"  Part 2 done.")

# ── Save all data ───────────────────────────────────────────────────
def _convert(obj):
    if isinstance(obj, dict): return {k: _convert(v) for k,v in obj.items()}
    if isinstance(obj, list): return [_convert(v) for v in obj]
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.integer,)): return int(obj)
    return obj

output = {"config": {"grid_res": GRID_RES, "max_iter": MAX_ITER,
         "population": POP_SIZE, "n_runs": N_RUNS},
         "scenarios": _convert(all_data), "sensitivity": _convert(sens_results)}
with open("results/full_results.json", "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print("\n  Full data saved: results/full_results.json")

# ── Part 3: Charts ──────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PART 3: Generating Charts")
print("=" * 65)

import plotly.graph_objects as go
import plotly.subplots as sp

COLORS = {"PSO": "#1f77b4", "ImprovedPSO": "#ff7f0e", "GWO": "#2ca02c",
          "ImprovedGWO": "#d62728", "WOA": "#9467bd", "ImprovedWOA": "#8c564b"}

for sc_name, sc_data in all_data.items():
    algo_names = list(sc_data["algorithms"].keys())

    # Box plot
    fig1 = go.Figure()
    for name in algo_names:
        fits = [s["fitness"] for s in sc_data["algorithms"][name]["seeds"]]
        fig1.add_trace(go.Box(y=fits, name=name, marker_color=COLORS.get(name), boxmean="sd"))
    fig1.add_hline(y=sc_data["random_baseline"]["mean_fitness"],
                   line_dash="dash", line_color="gray", annotation_text="Random (mean)")
    fig1.update_layout(title=f"Fitness Distribution (15 seeds) — {sc_name}",
                       yaxis_title="Fitness", width=800, height=500,
                       template="plotly_white", showlegend=False)
    fig1.write_html(f"results/full_boxplot_{sc_name}.html")

    # Coverage bar chart
    fig2 = go.Figure()
    names_sorted = sorted(algo_names, key=lambda n: sc_data["algorithms"][n]["mean_coverage"], reverse=True)
    for name in names_sorted:
        d = sc_data["algorithms"][name]
        fig2.add_trace(go.Bar(name=name, x=[name], y=[d["mean_coverage"]],
                       error_y=dict(type="data", array=[d["std_coverage"]], visible=True),
                       marker_color=COLORS.get(name),
                       text=f'{d["mean_coverage"]:.3f}', textposition="outside"))
    rc = sc_data["random_baseline"]["mean_coverage"]
    gc = sc_data["grid_baseline"]["coverage"]
    fig2.add_hline(y=rc, line_dash="dash", line_color="gray", annotation_text=f"Random ({rc:.3f})")
    fig2.add_hline(y=gc, line_dash="dot", line_color="black", annotation_text=f"Grid ({gc:.3f})")
    fig2.update_layout(title=f"Coverage Rate (mean+/-std, {N_RUNS} seeds) — {sc_name}",
                       yaxis_title="Coverage Rate", width=800, height=450,
                       template="plotly_white", showlegend=False)
    fig2.write_html(f"results/full_coverage_{sc_name}.html")

    # Convergence speed
    fig3 = go.Figure()
    names_conv = sorted(algo_names, key=lambda n: sc_data["algorithms"][n]["mean_convergence"])
    for name in names_conv:
        d = sc_data["algorithms"][name]
        fig3.add_trace(go.Bar(name=name, x=[name], y=[d["mean_convergence"]],
                       error_y=dict(type="data", array=[d["std_convergence"]], visible=True),
                       marker_color=COLORS.get(name),
                       text=f'{d["mean_convergence"]:.0f}', textposition="outside"))
    fig3.update_layout(title=f"Convergence Speed (iters to 95% fitness, {N_RUNS} seeds) — {sc_name}",
                       yaxis_title="Iterations to Converge", width=800, height=450,
                       template="plotly_white", showlegend=False)
    fig3.write_html(f"results/full_convergence_{sc_name}.html")

    # Heatmap (best algorithm)
    ranked = sorted(sc_data["algorithms"].items(), key=lambda x: x[1]["mean_fitness"], reverse=True)
    best_name = ranked[0][0]
    best_pos = np.array(ranked[0][1]["seeds"][ranked[0][1]["best_seed_idx"]]["best_positions"])

    env_kwargs_viz = {k: v for k, v in sc_data["params"].items()}
    obs = sc_data.get("obstacles")
    if obs: env_kwargs_viz["obstacles"] = obs
    env_viz = WSNEnvironment(**COMMON, **env_kwargs_viz, seed=42)
    env_viz.set_positions(best_pos)
    cov = env_viz.coverage_matrix()
    cov_counts = np.sum(cov, axis=0).reshape(env_viz.grid_y.shape)

    fig4 = sp.make_subplots(rows=1, cols=2,
        subplot_titles=(f"{best_name} Deployment", f"{best_name} Coverage Heatmap"),
        horizontal_spacing=0.15)
    fig4.add_trace(go.Scatter(x=best_pos[:,0], y=best_pos[:,1],
                   mode="markers", marker=dict(color="red", size=10), name="Nodes"), row=1, col=1)
    if obs:
        for o in obs:
            if o["type"] == "rect":
                fig4.add_shape(type="rect", x0=o["x"], y0=o["y"], x1=o["x"]+o["w"], y1=o["y"]+o["h"],
                              fillcolor="gray", opacity=0.4, line_width=0, row=1, col=1)
            elif o["type"] == "circle":
                th = np.linspace(0, 2*np.pi, 60)
                fig4.add_trace(go.Scatter(x=o["cx"]+o["r"]*np.cos(th), y=o["cy"]+o["r"]*np.sin(th),
                               fill="toself", fillcolor="gray", opacity=0.4, mode="none",
                               showlegend=False), row=1, col=1)
    fig4.add_trace(go.Heatmap(x=env_viz.grid_x[0], y=env_viz.grid_y[:,0],
                   z=cov_counts, colorscale="YlOrRd", zmin=0,
                   colorbar=dict(title="Coverage"), name="Coverage"), row=1, col=2)
    fig4.update_xaxes(title="X", range=[0, 100], row=1, col=1)
    fig4.update_yaxes(title="Y", range=[0, 100], row=1, col=1)
    fig4.update_xaxes(title="X", range=[0, 100], row=1, col=2)
    fig4.update_yaxes(title="Y", range=[0, 100], row=1, col=2)
    fig4.update_layout(title=f"Best Deployment: {best_name} — {sc_name}",
                       width=1200, height=500, template="plotly_white")
    fig4.write_html(f"results/full_heatmap_{sc_name}.html")

    print(f"  {sc_name}: box + coverage + convergence + heatmap saved")

# Sensitivity heatmap
mr_vals = sorted(set(d["mutation_rate"] for d in sens_results.values()))
ms_vals = sorted(set(d["mutation_scale"] for d in sens_results.values()))
heatmap_z = np.zeros((len(ms_vals), len(mr_vals)))
for key, d in sens_results.items():
    heatmap_z[ms_vals.index(d["mutation_scale"]), mr_vals.index(d["mutation_rate"])] = d["mean_fitness"]

fig5 = go.Figure(go.Heatmap(
    x=mr_vals, y=ms_vals, z=heatmap_z, colorscale="Viridis",
    text=[[f"{z:.4f}" for z in row] for row in heatmap_z],
    texttemplate="%{text}", colorbar=dict(title="Mean Fitness"),
))
fig5.update_layout(title="ImprovedPSO Sensitivity (mutation_rate x mutation_scale)",
                   xaxis_title="mutation_rate", yaxis_title="mutation_scale",
                   width=700, height=500, template="plotly_white")
fig5.write_html("results/full_sensitivity.html")
print("  Sensitivity heatmap saved")

# Cross-scenario summary
fig6 = go.Figure()
algo_names = list(list(all_data.values())[0]["algorithms"].keys())
for sc_name, sc_data in all_data.items():
    for name in algo_names:
        d = sc_data["algorithms"][name]
        rand_cov = sc_data["random_baseline"]["mean_coverage"]
        improvement = (d["mean_coverage"] - rand_cov) * 100
        fig6.add_trace(go.Bar(
            x=[sc_name], y=[improvement], name=f"{name} [{sc_name}]",
            marker_color=COLORS.get(name),
            text=f'+{improvement:.1f}pp', textposition="outside",
            legendgroup=name,
            showlegend=(sc_name == list(all_data.keys())[0]),
        ))
fig6.update_layout(title="Coverage Improvement over Random Baseline",
                   yaxis_title="Improvement (percentage points)",
                   width=900, height=500, template="plotly_white", barmode="group")
fig6.write_html("results/full_summary.html")
print("  Summary saved")

print(f"\n{'='*65}")
print(f"  Complete! Total: {time.perf_counter()-t_start:.1f}s")
print(f"  Output: {os.path.abspath('results')}/")
print(f"{'='*65}")
