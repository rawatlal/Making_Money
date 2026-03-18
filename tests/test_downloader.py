from pathlib import Path

import pandas as pd

from src.data.cleaner import DataCleaner
from src.data.storage import DataStore
from src.universe.provider import Russell1000Provider
from src.utils.config import load_config


def test_load_config():
    config = load_config()
    assert "universe" in config
    assert "factors" in config
    assert "backtest" in config
    assert config["portfolio"]["top_n"] == 50


def test_fallback_csv_exists():
    csv_path = Path("src/universe/russell1000_tickers.csv")
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert "ticker" in df.columns
    assert len(df) > 50


def test_russell1000_fallback():
    provider = Russell1000Provider(cache_dir="data/universe")
    tickers = provider._load_fallback()
    assert len(tickers) > 50
    assert all(isinstance(t, str) for t in tickers)


def test_data_cleaner(sample_prices):
    # Inject some NaNs
    prices = sample_prices.copy()
    prices.iloc[5:10, 0] = None  # 5 consecutive NaN in first ticker

    cleaner = DataCleaner(max_fill_days=5, max_missing_pct=0.20)
    clean = cleaner.clean_prices(prices)

    assert clean.isnull().sum().sum() == 0  # forward-filled
    assert len(clean.columns) == len(sample_prices.columns)


def test_data_cleaner_drops_sparse(sample_prices):
    prices = sample_prices.copy()
    # Make one ticker 50% NaN
    prices.iloc[: len(prices) // 2, 0] = None

    cleaner = DataCleaner(max_fill_days=5, max_missing_pct=0.20)
    clean = cleaner.clean_prices(prices)

    assert len(clean.columns) == len(sample_prices.columns) - 1


def test_data_store(tmp_path):
    store = DataStore(cache_dir=str(tmp_path))
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

    store.save_prices(df, label="test")
    loaded = store.load_prices(label="test")
    assert loaded is not None
    assert loaded.equals(df)

    assert not store.is_stale(label="test", max_age_hours=1)
