import numpy as np
import pandas as pd

from src.portfolio.constraints import PortfolioConstraints
from src.portfolio.optimizer import PortfolioOptimizer
from src.portfolio.rebalancer import Rebalancer


def test_max_sharpe():
    np.random.seed(42)
    n = 5
    mu = np.random.randn(n) * 0.001
    cov = np.eye(n) * 0.0004

    opt = PortfolioOptimizer(method="max_sharpe")
    constraints = PortfolioConstraints(max_position_weight=0.40)
    weights = opt.optimize(mu, cov, constraints)

    assert len(weights) == n
    assert abs(np.sum(weights) - 1.0) < 1e-6
    assert all(w >= -1e-6 for w in weights)
    assert all(w <= 0.40 + 1e-6 for w in weights)


def test_min_variance():
    np.random.seed(42)
    n = 5
    mu = np.random.randn(n) * 0.001
    cov = np.eye(n) * 0.0004

    opt = PortfolioOptimizer(method="min_variance")
    constraints = PortfolioConstraints(max_position_weight=0.40)
    weights = opt.optimize(mu, cov, constraints)

    assert abs(np.sum(weights) - 1.0) < 1e-6
    # With identity cov, min variance should be roughly equal weight
    assert np.std(weights) < 0.1


def test_risk_parity():
    np.random.seed(42)
    n = 5
    mu = np.random.randn(n) * 0.001
    cov = np.eye(n) * 0.0004

    opt = PortfolioOptimizer(method="risk_parity")
    constraints = PortfolioConstraints(max_position_weight=0.40)
    weights = opt.optimize(mu, cov, constraints)

    assert abs(np.sum(weights) - 1.0) < 1e-6


def test_covariance_estimation():
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-01", periods=252)
    returns = pd.DataFrame(
        np.random.randn(252, 5) * 0.02,
        index=dates,
        columns=["A", "B", "C", "D", "E"],
    )

    opt = PortfolioOptimizer()
    cov = opt.estimate_covariance(returns)
    assert cov.shape == (5, 5)
    # Covariance should be symmetric
    assert np.allclose(cov, cov.T)


def test_rebalancer_monthly():
    dates = pd.bdate_range("2023-01-01", "2023-06-30")
    reb = Rebalancer(frequency="monthly")
    rebal_dates = reb.get_rebalance_dates(dates)
    assert len(rebal_dates) == 6  # Jan through Jun


def test_rebalancer_quarterly():
    dates = pd.bdate_range("2023-01-01", "2023-12-31")
    reb = Rebalancer(frequency="quarterly")
    rebal_dates = reb.get_rebalance_dates(dates)
    assert len(rebal_dates) == 4  # Q1 through Q4


def test_constraints_bounds():
    constraints = PortfolioConstraints(max_position_weight=0.10, long_only=True)
    bounds = constraints.get_bounds(5)
    assert len(bounds) == 5
    assert bounds[0] == (0.0, 0.10)


def test_turnover_constraint():
    constraints = PortfolioConstraints(max_turnover_per_rebalance=0.30)
    current = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    tc = constraints.get_turnover_constraint(current)

    # Same weights = 0 turnover, should satisfy constraint
    assert tc["fun"](current) >= 0

    # Large change = high turnover, should violate constraint
    new = np.array([0.5, 0.5, 0.0, 0.0, 0.0])
    assert tc["fun"](new) < 0
