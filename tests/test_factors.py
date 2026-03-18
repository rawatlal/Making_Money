import numpy as np
import pandas as pd

from src.factors.composite import CompositeScorer
from src.factors.momentum import MomentumFactor
from src.factors.registry import get_factor, list_factors
from src.factors.value import EVToEBITDA, PBRatio, PERatio


def test_pe_ratio(sample_prices, sample_fundamentals):
    factor = PERatio()
    assert factor.direction == "lower_is_better"
    result = factor.compute(sample_prices, sample_fundamentals, sample_prices.index[-1])
    assert len(result) == len(sample_fundamentals)
    assert all(result > 0)


def test_pb_ratio(sample_prices, sample_fundamentals):
    factor = PBRatio()
    result = factor.compute(sample_prices, sample_fundamentals, sample_prices.index[-1])
    assert len(result) == len(sample_fundamentals)


def test_ev_ebitda(sample_prices, sample_fundamentals):
    factor = EVToEBITDA()
    result = factor.compute(sample_prices, sample_fundamentals, sample_prices.index[-1])
    assert len(result) > 0
    assert all(result > 0)


def test_momentum(sample_prices, sample_fundamentals):
    factor = MomentumFactor(63, name="momentum_3m")
    assert factor.direction == "higher_is_better"
    date = sample_prices.index[-1]
    result = factor.compute(sample_prices, sample_fundamentals, date)
    assert len(result) > 0


def test_momentum_insufficient_data(sample_fundamentals):
    """Momentum should return empty if not enough data."""
    dates = pd.bdate_range("2023-01-01", periods=30)
    prices = pd.DataFrame(
        np.random.randn(30, 3) * 0.01 + 1,
        index=dates,
        columns=["A", "B", "C"],
    ).cumprod() * 100

    factor = MomentumFactor(252, name="momentum_12m")
    result = factor.compute(prices, sample_fundamentals, dates[-1])
    assert result.empty


def test_composite_scorer(sample_prices, sample_fundamentals):
    factors = [
        (PERatio(), 0.35),
        (PBRatio(), 0.30),
        (EVToEBITDA(), 0.35),
    ]
    scorer = CompositeScorer(factors)
    date = sample_prices.index[-1]
    scores = scorer.score(sample_prices, sample_fundamentals, date)

    assert len(scores) > 0
    # Scores should be sorted descending
    assert scores.iloc[0] >= scores.iloc[-1]
    # Weights should produce non-zero scores
    assert scores.abs().sum() > 0


def test_registry():
    names = list_factors()
    assert "pe_ratio" in names
    assert "momentum_3m" in names
    assert "momentum_12m" in names

    factor = get_factor("pe_ratio")
    assert factor.name == "pe_ratio"
