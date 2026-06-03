# WSN Coverage Optimization using Improved Swarm Intelligence Algorithms

基于改进群体智能算法的无线传感器网络（WSN）覆盖优化研究。在给定监测区域内，利用 PSO、GWO、WOA 及其改进变体自动优化传感器节点部署位置，最大化覆盖率、最小化冗余度并均衡网络连通性。

## Architecture

```
┌─────────────────────────┐
│  WSNEnvironment         │  网格化监测区域 + 二元感知模型
│  - grid-based area      │
│  - binary sensing       │
│  - communication graph  │
└───────────┬─────────────┘
            │
┌───────────▼─────────────┐
│  FitnessEvaluator       │  F = w1·Cov + w2·(1-Red) + w3·Conn
│  - coverage rate        │
│  - redundancy rate      │
│  - connectivity rate    │
└───────────┬─────────────┘
            │
┌───────────▼─────────────┐
│  SwarmOptimizer (ABC)   │
│  ├─ PSO  ── ImprovedPSO │  粒子群 / 灰狼 / 鲸鱼优化
│  ├─ GWO  ── ImprovedGWO │  + 混沌初始化
│  └─ WOA  ── ImprovedWOA │  + 自适应参数 + 变异
└───────────┬─────────────┘
            │
┌───────────▼─────────────┐
│  Visualization          │  静态图 · 动态GIF · Plotly交互
│  ExperimentRunner       │  随机/网格Baseline · 算法对比
└─────────────────────────┘
```

## Installation

```bash
# Create conda environment
conda env create -f environment.yml
conda activate wsn

# Or use pip + virtualenv
python -m venv venv && source venv/bin/activate   # Linux/Mac
python -m venv venv && venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Quick Start

```python
from wsn.environment import WSNEnvironment
from wsn.fitness import FitnessEvaluator
from wsn.algorithms import ImprovedPSO

# 100x100 area, 50 nodes, sensing radius 12
env = WSNEnvironment(
    width=100, height=100, grid_resolution=2.0,
    n_nodes=50, sensing_radius=12.0,
    communication_radius=24.0, seed=42,
)
evaluator = FitnessEvaluator(w_coverage=0.5, w_redundancy=0.3, w_connectivity=0.2)

# Run Improved PSO
ipso = ImprovedPSO(env, evaluator, n_particles=30, max_iter=200, seed=42)
best_positions, best_fitness = ipso.optimize(verbose=True)
print(f"Best fitness: {best_fitness:.4f}")
```

## Comparing Algorithms

```python
from wsn.algorithms import PSO, ImprovedPSO, GWO, ImprovedGWO
from wsn.experiments import RandomBaseline, GridBaseline, ExperimentRunner

runner = ExperimentRunner(
    env_config=dict(width=100, height=100, grid_resolution=2.0,
                    n_nodes=50, sensing_radius=12.0, communication_radius=24.0),
    fitness_weights=(0.5, 0.3, 0.2),
)

runner.add_baseline("Random", RandomBaseline(env_rand, evaluator, n_trials=30))
runner.add_baseline("Grid", GridBaseline(env_grid, evaluator))
runner.add_algorithm("PSO", PSO(env_pso, evaluator, max_iter=200, seed=42))
runner.add_algorithm("ImprovedGWO", ImprovedGWO(env_igwo, evaluator, max_iter=200, seed=42))

results = runner.run_all(verbose=True)
runner.print_summary()
```

## Project Structure

```
wsn/
├── environment.py          # Grid-based monitoring area, binary sensing model
├── fitness.py              # Multi-objective fitness evaluator
├── algorithms/
│   ├── _base.py            # SwarmOptimizer abstract base
│   ├── pso.py              # PSO & ImprovedPSO
│   ├── gwo.py              # GWO & ImprovedGWO
│   └── woa.py              # WOA & ImprovedWOA
├── improvements/
│   ├── chaotic_init.py     # Logistic map initialization
│   ├── adaptive_params.py  # Adaptive inertia / convergence factors
│   └── mutation.py         # Gaussian mutation & Levy flight
├── visualization/
│   ├── static_plots.py     # Scatter, heatmap, convergence curves
│   ├── dynamic_plots.py    # GIF generation
│   └── interactive_dash.py # Plotly interactive dashboard
└── experiments/
    ├── baselines.py        # Random & grid deployment baselines
    └── runner.py           # Experiment comparison framework
```

## Improvement Strategies

| Strategy | Description |
|---|---|
| **Chaotic Initialization** | Logistic map generates diverse initial population |
| **Adaptive Parameters** | Nonlinear decay of inertia weight (PSO), convergence factor (GWO/WOA) |
| **Gaussian Mutation** | Perturbation in later iterations to escape local optima |
| **Levy Flight** | Occasional large jumps for deep local-optima escape |

## Evaluation Metrics

- **Coverage Rate** — fraction of grid points covered by at least one node
- **Redundancy Rate** — average excess coverage per covered point
- **Connectivity Rate** — fraction of node pairs within communication radius
- **Convergence Speed** — iterations to reach stable fitness
- **Stability** — standard deviation across repeated runs

## Running Tests

```bash
conda activate wsn
python -m pytest tests/ -v
```
