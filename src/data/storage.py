import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger("making_money")


class DataStore:
    """Parquet-based persistence and caching for price and fundamental data."""

    def __init__(self, cache_dir="data"):
        self.base = Path(cache_dir)
        self.raw_dir = self.base / "raw"
        self.processed_dir = self.base / "processed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def save_prices(self, df, label="prices"):
        path = self.raw_dir / f"{label}.parquet"
        df.to_parquet(path)
        logger.info(f"Saved prices to {path}")

    def load_prices(self, label="prices"):
        path = self.raw_dir / f"{label}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return None

    def save_fundamentals(self, df):
        path = self.raw_dir / "fundamentals.parquet"
        df.to_parquet(path)
        logger.info(f"Saved fundamentals to {path}")

    def load_fundamentals(self):
        path = self.raw_dir / "fundamentals.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return None

    def save_processed(self, df, name):
        path = self.processed_dir / f"{name}.parquet"
        df.to_parquet(path)
        logger.info(f"Saved processed data to {path}")

    def load_processed(self, name):
        path = self.processed_dir / f"{name}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return None

    def is_stale(self, label="prices", max_age_hours=24):
        """Check if cached data is older than max_age_hours."""
        path = self.raw_dir / f"{label}.parquet"
        if not path.exists():
            return True
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        return datetime.now() - mtime > timedelta(hours=max_age_hours)
