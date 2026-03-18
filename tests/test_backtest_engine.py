import numpy as np
import pandas as pd

from src.backtest.broker import SimulatedBroker
from src.backtest.engine import BacktestEngine
from src.backtest.performance import PerformanceAnalytics
from src.factors.composite import CompositeScorer
from src.factors.value import PERatio, PBRatio
from src.portfolio.constraints import PortfolioConstraints
from src.portfolio.constructor import PortfolioConstructor
from src.portfolio.optimizer import PortfolioOptimizer
from src.portfolio.rebalancer import Rebalancer


def _make_test_data():
    """Create synthetic test data for backtest."""
    np.random.seed(42)
    tickers = ["A", "B", "C", "D", "E"]
    dates = pd.bdate_range("2022-01-01", "2023-12-31")

    # Generate prices
    prices_data = {}
    for t in tickers:
        base = 50 + np.random.randn() * 10
        rets = np.random.randn(len(dates)) * 0.015
        prices_data[t] = base * np.cumprod(1 + rets)
    prices = pd.DataFrame(prices_data, index=dates)

    # Generate fundamentals
    fundamentals = pd.DataFrame(
        {
            "pe_ratio": [15, 20, 10, 25, 18],
            "pb_ratio": [2.0, 3.5, 1.5, 4.0, 2.5],
            "enterprise_value": [1e10, 2e10, 5e9, 3e10, 1.5e10],
            "ebitda": [1e9, 2e9, 8e8, 2.5e9, 1.2e9],
            "market_cap": [1e10, 2e10, 5e9, 3e10, 1.5e10],
            "sector": ["Tech", "Health", "Tech", "Finance", "Health"],
        },
        index=tickers,
    )

    return prices, fundamentals


def test_broker():
    broker = SimulatedBroker(commission_per_share=0.005, half_spread_bps=5)
    trades = {"AAPL": 100, "MSFT": -50}
    prices = {"AAPL": 150.0, "MSFT": 300.0}

    results, cash_impact = broker.execute_trades(trades, prices)
    assert len(results) == 2
    assert cash_impact < 0  # Net buy

    buy = [r for r in results if r.side == "buy"][0]
    assert buy.fill_price > 150.0  # Slippage on buy
    sell = [r for r in results if r.side == "sell"][0]
    assert sell.fill_price < 300.0  # Slippage on sell


def test_performance_analytics():
    np.random.seed(42)
    returns = pd.Series(np.random.randn(252) * 0.01, index=pd.bdate_range("2023-01-01", periods=252))

    perf = PerformanceAnalytics(risk_free_rate=0.04)
    stats = perf.compute_all(returns)

    assert "cagr" in stats
    assert "sharpe_ratio" in stats
    assert "max_drawdown" in stats
    assert stats["max_drawdown"] <= 0  # Drawdown is negative


def test_drawdown_series():
    returns = pd.Series([0.01, 0.02, -0.05, -0.03, 0.01], index=pd.bdate_range("2023-01-01", periods=5))
    perf = PerformanceAnalytics()
    dd = perf.drawdown_series(returns)
    assert dd.min() < 0


def test_full_backtest():
    """Run a mini end-to-end backtest on synthetic data."""
    prices, fundamentals = _make_test_data()

    scorer = CompositeScorer([(PERatio(), 0.5), (PBRatio(), 0.5)])
    optimizer = PortfolioOptimizer(method="min_variance")
    constraints = PortfolioConstraints(max_position_weight=0.40, max_holdings=5)
    constructor = PortfolioConstructor(scorer, optimizer, constraints, top_n=5, lookback_days=126)
    rebalancer = Rebalancer(frequency="quarterly")
    broker = SimulatedBroker()

    engine = BacktestEngine(
        constructor=constructor,
        rebalancer=rebalancer,
        broker=broker,
        initial_capital=100_000,
    )

    report = engine.run(prices, fundamentals, "2022-07-01", "2023-12-31")

    assert len(report.returns_series) > 0
    assert report.final_value > 0
    assert "cagr" in report.summary_stats
    assert "sharpe_ratio" in report.summary_stats
    assert len(report.weights_history) > 0
