from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from src.portfolio.constraints import PortfolioConstraints

logger = logging.getLogger("making_money")


class PortfolioOptimizer:
    """Portfolio optimizer using scipy with Ledoit-Wolf covariance estimation."""

    def __init__(self, method="max_sharpe", risk_free_rate=0.04):
        self.method = method
        self.risk_free_rate = risk_free_rate

    def estimate_covariance(self, returns):
        """Estimate covariance matrix using Ledoit-Wolf shrinkage."""
        clean = returns.dropna()
        if len(clean) < 30:
            logger.warning("Insufficient data for covariance estimation, using sample cov")
            return clean.cov().values
        lw = LedoitWolf().fit(clean.values)
        return lw.covariance_

    def optimize(self, expected_returns, cov_matrix, constraints_obj, current_weights=None, sectors=None):
        """Run portfolio optimization.

        Args:
            expected_returns: array of expected returns per asset.
            cov_matrix: NxN covariance matrix.
            constraints_obj: PortfolioConstraints instance.
            current_weights: current portfolio weights (for turnover constraint).
            sectors: list of sector labels per asset (for sector constraints).

        Returns:
            Optimized weight array.
        """
        n = len(expected_returns)
        bounds = constraints_obj.get_bounds(n)
        constraints = constraints_obj.get_base_constraints(n)

        if sectors is not None:
            unique_sectors = list(set(s for s in sectors if s))
            constraints.extend(constraints_obj.get_sector_constraints(sectors, unique_sectors))

        if current_weights is not None:
            constraints.append(constraints_obj.get_turnover_constraint(current_weights))

        # Initial guess: equal weight
        w0 = np.ones(n) / n

        if self.method == "max_sharpe":
            weights = self._max_sharpe(w0, expected_returns, cov_matrix, bounds, constraints)
        elif self.method == "min_variance":
            weights = self._min_variance(w0, cov_matrix, bounds, constraints)
        elif self.method == "risk_parity":
            weights = self._risk_parity(w0, cov_matrix, bounds, constraints)
        elif self.method == "max_score":
            weights = self._max_score(w0, expected_returns, cov_matrix, bounds, constraints)
        else:
            raise ValueError(f"Unknown optimization method: {self.method}")

        return weights

    def _max_sharpe(self, w0, mu, cov, bounds, constraints):
        """Maximize Sharpe ratio."""
        rf_daily = (1 + self.risk_free_rate) ** (1 / 252) - 1

        def neg_sharpe(w):
            port_ret = np.dot(w, mu)
            port_vol = np.sqrt(np.dot(w, np.dot(cov, w)))
            if port_vol < 1e-10:
                return 1e6
            return -(port_ret - rf_daily) / port_vol

        result = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints)
        if not result.success:
            logger.warning(f"Max Sharpe optimization did not converge: {result.message}")
        return result.x

    def _min_variance(self, w0, cov, bounds, constraints):
        """Minimize portfolio variance."""
        def variance(w):
            return np.dot(w, np.dot(cov, w))

        result = minimize(variance, w0, method="SLSQP", bounds=bounds, constraints=constraints)
        if not result.success:
            logger.warning(f"Min variance optimization did not converge: {result.message}")
        return result.x

    def _risk_parity(self, w0, cov, bounds, constraints):
        """Risk parity: equal risk contribution from each asset."""
        n = len(w0)
        target_risk = 1.0 / n

        def risk_contribution_diff(w):
            port_vol = np.sqrt(np.dot(w, np.dot(cov, w)))
            if port_vol < 1e-10:
                return 1e6
            marginal = np.dot(cov, w) / port_vol
            rc = w * marginal / port_vol
            return np.sum((rc - target_risk) ** 2)

        result = minimize(risk_contribution_diff, w0, method="SLSQP", bounds=bounds, constraints=constraints)
        if not result.success:
            logger.warning(f"Risk parity optimization did not converge: {result.message}")
        return result.x

    def _max_score(self, w0, scores, cov, bounds, constraints):
        """Maximize composite factor score subject to risk constraint."""
        def neg_score(w):
            return -np.dot(w, scores)

        # Add a volatility ceiling constraint (20% annualized)
        vol_ceiling = 0.20 / np.sqrt(252)
        constraints = list(constraints) + [
            {"type": "ineq", "fun": lambda w: vol_ceiling**2 - np.dot(w, np.dot(cov, w))}
        ]

        result = minimize(neg_score, w0, method="SLSQP", bounds=bounds, constraints=constraints)
        if not result.success:
            logger.warning(f"Max score optimization did not converge: {result.message}")
        return result.x
