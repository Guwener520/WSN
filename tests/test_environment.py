"""Smoke tests for WSN environment module."""

import numpy as np
import pytest

from wsn.environment import WSNEnvironment
from wsn.fitness import FitnessEvaluator


def test_env_creation():
    env = WSNEnvironment(width=100, height=100, grid_resolution=2.0, n_nodes=30)
    assert env.n_nodes == 30
    assert env.grid_points.shape[1] == 2


def test_random_deploy():
    env = WSNEnvironment(width=100, height=100, n_nodes=20)
    pos = env.random_deploy()
    assert pos.shape == (20, 2)
    assert np.all(pos >= 0) and np.all(pos <= 100)


def test_grid_deploy():
    env = WSNEnvironment(width=100, height=100, n_nodes=25)
    pos = env.grid_deploy()
    assert pos.shape == (25, 2)


def test_coverage_matrix():
    env = WSNEnvironment(width=100, height=100, grid_resolution=5.0, n_nodes=20, sensing_radius=15)
    env.random_deploy(seed=42)
    cov = env.coverage_matrix()
    assert cov.shape == (20, env.n_grid_points)


def test_covered_grid_mask():
    env = WSNEnvironment(width=100, height=100, grid_resolution=5.0, n_nodes=20, sensing_radius=15)
    env.random_deploy(seed=42)
    mask = env.covered_grid_mask()
    assert mask.dtype == bool
    assert len(mask) == env.n_grid_points


def test_fitness_evaluator():
    env = WSNEnvironment(width=100, height=100, grid_resolution=5.0, n_nodes=20, sensing_radius=15)
    env.random_deploy(seed=42)
    evaluator = FitnessEvaluator()
    metrics = evaluator.evaluate_from_environment(env)
    assert 0 <= metrics.coverage_rate <= 1
    assert 0 <= metrics.fitness <= 1
    assert metrics.redundancy_rate >= 0
