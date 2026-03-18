from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

from src.backtest.broker import SimulatedBroker
from src.backtest.performance import PerformanceAnalytics
from src.backtest.report import BacktestReport
from src.portfolio.constructor import PortfolioConstructor
from src.portfolio.rebalancer import Rebalancer

logger = logging.getLogger("making_money")


class BacktestEngine:
    """Calendar-driven multi-period backtest engine."""

    def __init__(
        self,
        constructor: PortfolioConstructor,
        rebalancer: Rebalancer,
        broker: SimulatedBroker,
        initial_capital: float = 1_000_000,
        benchmark_ticker: str = "SPY",
        risk_free_rate: float = 0.04,
    ):
        self.constructor = constructor
        self.rebalancer = rebalancer
        self.broker = broker
        self.initial_capital = initial_capital
        self.benchmark_ticker = benchmark_ticker
        self.risk_free_rate = risk_free_rate

    def run(self, prices, fundamentals, start_date, end_date):
        """Run the backtest.

        Args:
            prices: DataFrame of clean close prices (date index, ticker columns).
            fundamentals: DataFrame of fundamental data (ticker index).
            start_date: Backtest start date string or Timestamp.
            end_date: Backtest end date string or Timestamp.

        Returns:
            BacktestReport with all results.
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        # Filter to backtest period
        bt_prices = prices.loc[start:end]
        trading_dates = bt_prices.index
        rebalance_dates = set(self.rebalancer.get_rebalance_dates(trading_dates))

        # State
        cash = self.initial_capital
        positions = {}  # ticker -> shares
        daily_returns = []
        daily_dates = []
        weights_snapshots = []
        turnover_records = []
        cost_records = []
        prev_portfolio_value = self.initial_capital

        logger.info(f"Starting backtest: {start.date()} to {end.date()}, capital=${self.initial_capital:,.0f}")

        for date in trading_dates:
            today_prices = bt_prices.loc[date].dropna().to_dict()

            # 1. Mark to market
            position_value = sum(
                positions.get(t, 0) * today_prices.get(t, 0)
                for t in positions
            )
            portfolio_value = position_value + cash

            # 2. Check rebalance
            if date in rebalance_dates:
                # Get current weights
                current_weights = {}
                if portfolio_value > 0 and positions:
                    for t, s in positions.items():
                        p = today_prices.get(t, 0)
                        current_weights[t] = (s * p) / portfolio_value if portfolio_value > 0 else 0

                # Construct target portfolio
                target_weights = self.constructor.construct(
                    prices, fundamentals, date, current_weights
                )

                if target_weights:
                    # Compute target shares
                    target_shares = {}
                    for t, w in target_weights.items():
                        p = today_prices.get(t, 0)
                        if p > 0:
                            target_shares[t] = (w * portfolio_value) / p

                    # Compute trades
                    all_tickers = set(list(positions.keys()) + list(target_shares.keys()))
                    trades = {}
                    for t in all_tickers:
                        delta = target_shares.get(t, 0) - positions.get(t, 0)
                        if abs(delta) > 0.01:
                            trades[t] = delta

                    # Execute trades
                    if trades:
                        results, cash_impact = self.broker.execute_trades(trades, today_prices)
                        cash += cash_impact

                        # Update positions
                        for t in all_tickers:
                            positions[t] = target_shares.get(t, 0)
                        # Remove zero positions
                        positions = {t: s for t, s in positions.items() if abs(s) > 0.01}

                        # Record turnover
                        turnover = sum(abs(r.shares * r.fill_price) for r in results) / max(portfolio_value, 1)
                        turnover_records.append((date, turnover))
                        cost = sum(r.commission + r.slippage_cost for r in results)
                        cost_records.append((date, cost))

                    # Record weights
                    weights_snapshots.append((date, dict(target_weights)))

            # 3. Record daily return
            if prev_portfolio_value > 0:
                daily_ret = (portfolio_value - prev_portfolio_value) / prev_portfolio_value
            else:
                daily_ret = 0.0
            daily_returns.append(daily_ret)
            daily_dates.append(date)
            prev_portfolio_value = portfolio_value

        # Build return series
        returns = pd.Series(daily_returns, index=pd.DatetimeIndex(daily_dates), name="portfolio")

        # Benchmark returns
        benchmark_returns = pd.Series(dtype=float)
        if self.benchmark_ticker in prices.columns:
            bench = prices.loc[start:end, self.benchmark_ticker].dropna()
            benchmark_returns = bench.pct_change().dropna()

        # Performance analytics
        perf = PerformanceAnalytics(risk_free_rate=self.risk_free_rate)
        stats = perf.compute_all(returns, benchmark_returns if not benchmark_returns.empty else None)

        # Final value
        final_position_value = sum(
            positions.get(t, 0) * bt_prices.iloc[-1].get(t, 0)
            for t in positions
        )
        final_value = final_position_value + cash

        # Build weights history DataFrame
        if weights_snapshots:
            w_dates, w_dicts = zip(*weights_snapshots)
            weights_df = pd.DataFrame(list(w_dicts), index=pd.DatetimeIndex(w_dates)).fillna(0)
        else:
            weights_df = pd.DataFrame()

        # Build turnover and cost series
        turnover_series = pd.Series(dtype=float)
        cost_series = pd.Series(dtype=float)
        if turnover_records:
            t_dates, t_vals = zip(*turnover_records)
            turnover_series = pd.Series(t_vals, index=pd.DatetimeIndex(t_dates))
        if cost_records:
            c_dates, c_vals = zip(*cost_records)
            cost_series = pd.Series(c_vals, index=pd.DatetimeIndex(c_dates))

        logger.info(f"Backtest complete. Final value: ${final_value:,.0f}")
        logger.info(f"CAGR: {stats.get('cagr', 0):.2%}, Sharpe: {stats.get('sharpe_ratio', 0):.2f}")

        return BacktestReport(
            summary_stats=stats,
            returns_series=returns,
            benchmark_returns=benchmark_returns,
            weights_history=weights_df,
            turnover_history=turnover_series,
            transaction_costs=cost_series,
            initial_capital=self.initial_capital,
            final_value=final_value,
        )
