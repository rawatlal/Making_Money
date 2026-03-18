import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_tickers():
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]


@pytest.fixture
def sample_prices(sample_tickers):
    """Generate synthetic daily close prices for testing."""
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-01", periods=252)
    data = {}
    for ticker in sample_tickers:
        base = 100 + np.random.randn() * 20
        returns = np.random.randn(252) * 0.02
        prices = base * np.cumprod(1 + returns)
        data[ticker] = prices
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_fundamentals(sample_tickers):
    """Generate synthetic fundamental data for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "pe_ratio": np.random.uniform(10, 40, len(sample_tickers)),
            "pb_ratio": np.random.uniform(1, 10, len(sample_tickers)),
            "enterprise_value": np.random.uniform(1e10, 1e12, len(sample_tickers)),
            "ebitda": np.random.uniform(1e9, 5e10, len(sample_tickers)),
            "market_cap": np.random.uniform(1e10, 3e12, len(sample_tickers)),
            "sector": ["Technology", "Technology", "Technology", "Consumer Discretionary", "Technology"],
        },
        index=sample_tickers,
    )
