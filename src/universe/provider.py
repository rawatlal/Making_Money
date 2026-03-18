from __future__ import annotations

import abc
import logging
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger("making_money")


class UniverseProvider(abc.ABC):
    @abc.abstractmethod
    def get_tickers(self) -> list[str]:
        ...


class Russell1000Provider(UniverseProvider):
    """Provides Russell 1000 tickers via Wikipedia scrape with CSV fallback."""

    WIKI_URL = "https://en.wikipedia.org/wiki/Russell_1000_Index"

    def __init__(self, cache_dir="data/universe", fallback_csv=None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.fallback_csv = fallback_csv or (
            Path(__file__).parent / "russell1000_tickers.csv"
        )

    def get_tickers(self) -> list[str]:
        cached = self._load_cached()
        if cached is not None:
            logger.info(f"Loaded {len(cached)} tickers from cache")
            return cached

        tickers = self._scrape_wikipedia()
        if tickers:
            self._save_cache(tickers)
            logger.info(f"Scraped {len(tickers)} tickers from Wikipedia")
            return tickers

        return self._load_fallback()

    def _load_cached(self) -> Optional[list[str]]:
        cache_file = self.cache_dir / f"russell1000_{date.today().isoformat()}.csv"
        if cache_file.exists():
            df = pd.read_csv(cache_file)
            return df["ticker"].tolist()
        return None

    def _scrape_wikipedia(self) -> list[str]:
        try:
            tables = pd.read_html(self.WIKI_URL)
            for table in tables:
                cols = [c.lower() for c in table.columns]
                if "ticker" in cols or "symbol" in cols:
                    col = "ticker" if "ticker" in cols else "symbol"
                    idx = cols.index(col)
                    tickers = table.iloc[:, idx].dropna().astype(str).tolist()
                    tickers = sorted(set(t.strip().upper() for t in tickers if t.strip()))
                    if len(tickers) > 500:
                        return tickers
        except Exception as e:
            logger.warning(f"Wikipedia scrape failed: {e}")
        return []

    def _load_fallback(self) -> list[str]:
        path = Path(self.fallback_csv)
        if path.exists():
            df = pd.read_csv(path)
            tickers = df["ticker"].tolist()
            logger.info(f"Loaded {len(tickers)} tickers from fallback CSV")
            return tickers
        logger.error("No ticker source available")
        return []

    def _save_cache(self, tickers: list[str]):
        cache_file = self.cache_dir / f"russell1000_{date.today().isoformat()}.csv"
        pd.DataFrame({"ticker": tickers}).to_csv(cache_file, index=False)
