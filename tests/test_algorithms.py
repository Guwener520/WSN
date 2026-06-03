"""Smoke tests for swarm intelligence algorithms."""

import numpy as np
import pytest

from wsn.environment import WSNEnvironment
from wsn.fitness import FitnessEvaluator
from wsn.algorithms import PSO, ImprovedPSO, GWO, ImprovedGWO, WOA, ImprovedWOA


@pytest.fixture
def setup():
    env = WSNEnvironment(
        width=100, height=100, grid_resolution=5.0,
        n_nodes=20, sensing_radius=15.0, communication_radius=30.0,
        seed=42,
    )
    evaluator = FitnessEvaluator(w_coverage=0.5, w_redundancy=0.3, w_connectivity=0.2)
    return env, evaluator


def test_pso_basic(setup):
    env, evaluator = setup
    pso = PSO(env, evaluator, n_particles=10, max_iter=20, seed=42)
    best_pos, best_fit = pso.optimize(verbose=False)
    assert best_pos.shape == (env.n_nodes, 2)
    assert 0 <= best_fit <= 1
    assert len(pso.history) == 20


def test_gwo_basic(setup):
    env, evaluator = setup
    gwo = GWO(env, evaluator, n_wolves=10, max_iter=20, seed=42)
    best_pos, best_fit = gwo.optimize(verbose=False)
    assert best_pos.shape == (env.n_nodes, 2)
    assert len(gwo.history) == 20


def test_woa_basic(setup):
    env, evaluator = setup
    woa = WOA(env, evaluator, n_whales=10, max_iter=20, seed=42)
    best_pos, best_fit = woa.optimize(verbose=False)
    assert best_pos.shape == (env.n_nodes, 2)
    assert len(woa.history) == 20


def test_improved_pso(setup):
    env, evaluator = setup
    ipso = ImprovedPSO(env, evaluator, n_particles=10, max_iter=20, seed=42)
    best_pos, best_fit = ipso.optimize(verbose=False)
    assert best_pos.shape == (env.n_nodes, 2)
    assert 0 <= best_fit <= 1


def test_improved_gwo(setup):
    env, evaluator = setup
    igwo = ImprovedGWO(env, evaluator, n_wolves=10, max_iter=20, seed=42)
    best_pos, best_fit = igwo.optimize(verbose=False)
    assert best_pos.shape == (env.n_nodes, 2)


def test_improved_woa(setup):
    env, evaluator = setup
    iwoa = ImprovedWOA(env, evaluator, n_whales=10, max_iter=20, seed=42)
    best_pos, best_fit = iwoa.optimize(verbose=False)
    assert best_pos.shape == (env.n_nodes, 2)
