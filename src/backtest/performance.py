from __future__ import annotations

import numpy as np
import pandas as pd


class PerformanceAnalytics:
    """Compute risk/return metrics from a returns series."""

    def __init__(self, risk_free_rate=0.04):
        self.rf = risk_free_rate
        self.rf_daily = (1 + risk_free_rate) ** (1 / 252) - 1

    def compute_all(self, returns, benchmark_returns=None):
        """Compute all performance metrics.

        Args:
            returns: Series of daily portfolio returns.
            benchmark_returns: Optional Series of daily benchmark returns.

        Returns:
            Dict of metric name to value.
        """
        stats = {}
        stats["total_return"] = self._total_return(returns)
        stats["cagr"] = self._cagr(returns)
        stats["annualized_volatility"] = self._annualized_vol(returns)
        stats["sharpe_ratio"] = self._sharpe(returns)
        stats["sortino_ratio"] = self._sortino(returns)
        stats["max_drawdown"] = self._max_drawdown(returns)
        stats["calmar_ratio"] = self._calmar(returns)

        if benchmark_returns is not None:
            aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
            if len(aligned) > 10:
                alpha, beta = self._alpha_beta(aligned.iloc[:, 0], aligned.iloc[:, 1])
                stats["alpha"] = alpha
                stats["beta"] = beta

        return stats

    def _total_return(self, returns):
        return (1 + returns).prod() - 1

    def _cagr(self, returns):
        n_years = len(returns) / 252
        if n_years <= 0:
            return 0.0
        total = (1 + returns).prod()
        return total ** (1 / n_years) - 1

    def _annualized_vol(self, returns):
        return returns.std() * np.sqrt(252)

    def _sharpe(self, returns):
        excess = returns - self.rf_daily
        if excess.std() < 1e-10:
            return 0.0
        return excess.mean() / excess.std() * np.sqrt(252)

    def _sortino(self, returns):
        excess = returns - self.rf_daily
        downside = returns[returns < 0]
        if len(downside) == 0 or downside.std() < 1e-10:
            return 0.0
        return excess.mean() / downside.std() * np.sqrt(252)

    def _max_drawdown(self, returns):
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()

    def _calmar(self, returns):
        cagr = self._cagr(returns)
        mdd = abs(self._max_drawdown(returns))
        if mdd < 1e-10:
            return 0.0
        return cagr / mdd

    def _alpha_beta(self, portfolio_returns, benchmark_returns):
        """OLS regression: r_p = alpha + beta * r_b."""
        cov = np.cov(portfolio_returns, benchmark_returns)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 1e-10 else 0.0
        alpha_daily = portfolio_returns.mean() - beta * benchmark_returns.mean()
        alpha_annual = alpha_daily * 252
        return alpha_annual, beta

    def monthly_returns_table(self, returns):
        """Create a monthly returns pivot table."""
        monthly = (1 + returns).resample("M").prod() - 1
        table = pd.DataFrame({
            "year": monthly.index.year,
            "month": monthly.index.month,
            "return": monthly.values,
        })
        return table.pivot(index="year", columns="month", values="return")

    def drawdown_series(self, returns):
        """Compute drawdown time series."""
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        return (cumulative - peak) / peak
