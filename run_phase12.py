"""Phase 1+2: Increased iterations + Ablation study. ~20 minutes."""
import os, json, time
import numpy as np
os.makedirs("results", exist_ok=True)

from wsn.environment import WSNEnvironment
from wsn.fitness import FitnessEvaluator
from wsn.algorithms import PSO, ImprovedPSO, GWO, ImprovedGWO, WOA, ImprovedWOA
from wsn.experiments import RandomBaseline, GridBaseline
import plotly.graph_objects as go

# ── Config ──────────────────────────────────────────────────────────
MAX_ITER = 150
POP = 20
N_SEEDS = 3
SEEDS = [42, 123, 456]

ABLATION_VARIANTS = {
    "PSO (baseline)":    dict(use_chaotic_init=False, use_adaptive_w=False, use_mutation=False),
    "+ Chaotic":          dict(use_chaotic_init=True,  use_adaptive_w=False, use_mutation=False),
    "+ Adaptive":         dict(use_chaotic_init=False, use_adaptive_w=True,  use_mutation=False),
    "+ Mutation":         dict(use_chaotic_init=False, use_adaptive_w=False, use_mutation=True),
    "+ Ch+Ad":            dict(use_chaotic_init=True,  use_adaptive_w=True,  use_mutation=False),
    "ImprovedPSO (all)":  dict(use_chaotic_init=True,  use_adaptive_w=True,  use_mutation=True),
}

SCENARIOS = {
    "Hard":   dict(n_nodes=25, sensing_radius=8.0),
    "Medium": dict(n_nodes=40, sensing_radius=10.0),
}
COMMON = dict(width=100, height=100, grid_resolution=2.0, communication_radius=24.0)

print("=" * 60)
print(f"  Phase 1+2: MAX_ITER={MAX_ITER} POP={POP} seeds={N_SEEDS}")
print("=" * 60)

t_start = time.perf_counter()
evaluator = FitnessEvaluator(0.5, 0.3, 0.2)

# ── Phase 1: Main algorithms comparison ─────────────────────────────
print("\n" + "─" * 50)
print("  PHASE 1: Main Algorithms (higher iterations)")
print("─" * 50)

phase1_data = {}
for sc_name, sc_params in SCENARIOS.items():
    print(f"\n  [{sc_name}] N={sc_params['n_nodes']} Rs={sc_params['sensing_radius']}")

    def make_env(s):
        return WSNEnvironment(**COMMON, **sc_params, seed=s)

    rand = RandomBaseline(make_env(42), evaluator, n_trials=100, seed=42).run()
    grid = GridBaseline(make_env(42), evaluator).run()

    main_algos = {
        "PSO":         lambda s: PSO(make_env(s), evaluator, n_particles=POP, max_iter=MAX_ITER, seed=s),
        "ImprovedPSO": lambda s: ImprovedPSO(make_env(s), evaluator, n_particles=POP, max_iter=MAX_ITER, seed=s),
        "GWO":         lambda s: GWO(make_env(s), evaluator, n_wolves=POP, max_iter=MAX_ITER, seed=s),
        "ImprovedGWO": lambda s: ImprovedGWO(make_env(s), evaluator, n_wolves=POP, max_iter=MAX_ITER, seed=s),
        "WOA":         lambda s: WOA(make_env(s), evaluator, n_whales=POP, max_iter=MAX_ITER, seed=s),
        "ImprovedWOA": lambda s: ImprovedWOA(make_env(s), evaluator, n_whales=POP, max_iter=MAX_ITER, seed=s),
    }

    sc_data = {}
    for name, builder in main_algos.items():
        fits, covs, hists = [], [], []
        best_f, best_h = -np.inf, None
        for s in SEEDS:
            algo = builder(s)
            bp, bf = algo.optimize(verbose=False)
            algo.env.set_positions(bp)
            m = evaluator.evaluate_from_environment(algo.env)
            fits.append(bf)
            covs.append(m.coverage_rate)
            if bf > best_f:
                best_f, best_h = bf, algo.history
        sc_data[name] = {
            "mean_fitness": float(np.mean(fits)), "std_fitness": float(np.std(fits)),
            "mean_coverage": float(np.mean(covs)), "std_coverage": float(np.std(covs)),
            "history": [(h["iteration"], h["best_fitness"]) for h in best_h],
        }
        vs_rand = (np.mean(covs) - rand["mean_coverage"]) * 100
        print(f"    {name:<14} fit={np.mean(fits):.4f}+/-{np.std(fits):.4f}  "
              f"cov={np.mean(covs):.4f}  vs_random=+{vs_rand:.1f}pp")

    phase1_data[sc_name] = {"random": rand, "grid": grid, "algorithms": sc_data}

# ── Phase 2: Ablation study (Medium scenario only) ──────────────────
print("\n" + "─" * 50)
print("  PHASE 2: Ablation Study — ImprovedPSO (Medium scenario)")
print("─" * 50)

sc_m = SCENARIOS["Medium"]
ablation_data = {}

for vname, switches in ABLATION_VARIANTS.items():
    fits, covs, hists = [], [], []
    best_f, best_h = -np.inf, None
    for s in SEEDS:
        env = WSNEnvironment(**COMMON, **sc_m, seed=s)
        ipso = ImprovedPSO(env, evaluator, n_particles=POP, max_iter=MAX_ITER,
                          seed=s, **switches)
        bp, bf = ipso.optimize(verbose=False)
        ipso.env.set_positions(bp)
        m = evaluator.evaluate_from_environment(ipso.env)
        fits.append(bf)
        covs.append(m.coverage_rate)
        if bf > best_f:
            best_f, best_h = bf, ipso.history
    ablation_data[vname] = {
        "mean_fitness": float(np.mean(fits)), "std_fitness": float(np.std(fits)),
        "mean_coverage": float(np.mean(covs)), "std_coverage": float(np.std(covs)),
        "history": [(h["iteration"], h["best_fitness"]) for h in best_h],
    }
    vs_base = (np.mean(covs) - ablation_data["PSO (baseline)"]["mean_coverage"]) * 100
    print(f"    {vname:<22} fit={np.mean(fits):.4f}+/-{np.std(fits):.4f}  "
          f"cov={np.mean(covs):.4f}  vs_baseline={'+' if vs_base>=0 else ''}{vs_base:.1f}pp")

print(f"\n  Elapsed: {time.perf_counter()-t_start:.0f}s. Generating charts...")

# ── Charts ──────────────────────────────────────────────────────────
COLORS = {"PSO": "#1f77b4", "ImprovedPSO": "#ff7f0e", "GWO": "#2ca02c",
          "ImprovedGWO": "#d62728", "WOA": "#9467bd", "ImprovedWOA": "#8c564b"}

chart = 0
for sc_name, d in phase1_data.items():
    # Convergence curves
    fig = go.Figure()
    for name, ad in d["algorithms"].items():
        hist = ad["history"]
        fig.add_trace(go.Scatter(x=[h[0] for h in hist], y=[h[1] for h in hist],
                      mode="lines", name=name,
                      line=dict(color=COLORS.get(name), width=2)))
    fig.add_hline(y=d["random"]["mean_fitness"], line_dash="dash", line_color="gray",
                  annotation_text="Random")
    fig.add_hline(y=d["grid"]["fitness"], line_dash="dot", line_color="black",
                  annotation_text="Grid")
    fig.update_layout(title=f"Convergence ({MAX_ITER} iter) — {sc_name}",
                      xaxis_title="Iteration", yaxis_title="Fitness",
                      width=850, height=500, template="plotly_white")
    fig.write_html(f"results/phase1_convergence_{sc_name}.html")
    chart += 1

    # Bar chart
    fig2 = go.Figure()
    sn = sorted(d["algorithms"].keys(), key=lambda n: d["algorithms"][n]["mean_fitness"], reverse=True)
    for name in sn:
        ad = d["algorithms"][name]
        is_improved = "Improved" in name
        fig2.add_trace(go.Bar(name=name, x=[name], y=[ad["mean_fitness"]],
                       error_y=dict(type="data", array=[ad["std_fitness"]], visible=True),
                       marker_color=COLORS.get(name),
                       marker_pattern_shape="/" if is_improved else "",
                       text=f'{ad["mean_fitness"]:.4f}', textposition="outside"))
    fig2.add_hline(y=d["random"]["mean_fitness"], line_dash="dash", line_color="gray",
                   annotation_text="Random")
    fig2.update_layout(title=f"Fitness Comparison ({MAX_ITER} iter) — {sc_name}",
                       yaxis_title="Fitness", width=800, height=420,
                       template="plotly_white", showlegend=False)
    fig2.write_html(f"results/phase1_bars_{sc_name}.html")
    chart += 1

# Ablation chart
fig3 = go.Figure()
ab_names = list(ABLATION_VARIANTS.keys())
for name in ab_names:
    ad = ablation_data[name]
    is_full = name == "ImprovedPSO (all)"
    fig3.add_trace(go.Bar(name=name, x=[name], y=[ad["mean_fitness"]],
                   error_y=dict(type="data", array=[ad["std_fitness"]], visible=True),
                   marker_color="#ff7f0e" if is_full else "#1f77b4",
                   text=f'{ad["mean_fitness"]:.4f}', textposition="outside"))

baseline_fit = ablation_data["PSO (baseline)"]["mean_fitness"]
fig3.add_hline(y=baseline_fit, line_dash="dash", line_color="gray",
               annotation_text=f"PSO baseline ({baseline_fit:.4f})")
fig3.update_layout(title="Ablation Study — ImprovedPSO Strategy Contributions (Medium)",
                   yaxis_title="Fitness", width=950, height=450,
                   template="plotly_white", showlegend=False)
fig3.write_html("results/phase2_ablation.html")
chart += 1

# Ablation convergence
fig4 = go.Figure()
for name in ab_names:
    hist = ablation_data[name]["history"]
    fig4.add_trace(go.Scatter(x=[h[0] for h in hist], y=[h[1] for h in hist],
                   mode="lines", name=name, line=dict(width=2)))
fig4.update_layout(title="Ablation Convergence — ImprovedPSO Strategy Contributions",
                   xaxis_title="Iteration", yaxis_title="Fitness",
                   width=850, height=500, template="plotly_white")
fig4.write_html("results/phase2_ablation_convergence.html")
chart += 1

# Combined summary table
print(f"\n{'='*60}")
print("  PHASE 1 RESULTS")
print(f"{'='*60}")
for sc_name, d in phase1_data.items():
    print(f"\n  [{sc_name}]")
    hdr = f"  {'Algorithm':<16} {'Fitness':>16} {'Coverage':>16} {'vs Random':>12}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    rc = d["random"]["mean_coverage"]
    sn = sorted(d["algorithms"].keys(), key=lambda n: d["algorithms"][n]["mean_fitness"], reverse=True)
    for i, name in enumerate(sn):
        ad = d["algorithms"][name]
        vs_r = (ad["mean_coverage"] - rc) * 100
        marker = " <<<" if i == 0 else ""
        print(f"  {name:<16} {ad['mean_fitness']:>10.4f}+/-{ad['std_fitness']:.4f}  "
              f"{ad['mean_coverage']:>10.4f}+/-{ad['std_coverage']:.4f}  +{vs_r:>8.1f}pp{marker}")

print(f"\n{'='*60}")
print("  PHASE 2: ABLATION RESULTS (Medium)")
print(f"{'='*60}")
print(f"  {'Variant':<24} {'Fitness':>16} {'Coverage':>16} {'vs Baseline':>14}")
print(f"  {'─'*72}")
base_cov = ablation_data["PSO (baseline)"]["mean_coverage"]
base_fit = ablation_data["PSO (baseline)"]["mean_fitness"]
for name in ab_names:
    ad = ablation_data[name]
    dc = (ad["mean_coverage"] - base_cov) * 100
    df = (ad["mean_fitness"] - base_fit) * 100
    m = " <<<" if name == "ImprovedPSO (all)" else ""
    print(f"  {name:<24} {ad['mean_fitness']:>10.4f}+/-{ad['std_fitness']:.4f}  "
          f"{ad['mean_coverage']:>10.4f}+/-{ad['std_coverage']:.4f}  "
          f"{'+' if dc>=0 else ''}{dc:>5.1f}pp cov{'+' if df>=0 else ''}{df:>5.1f}pp fit{m}")

print(f"\n  {chart} charts saved to results/")
print(f"  Total: {time.perf_counter()-t_start:.0f}s")
