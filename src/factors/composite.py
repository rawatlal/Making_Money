from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.factors.base import Factor

logger = logging.getLogger("making_money")


class CompositeScorer:
    """Combines multiple factors into a single composite score per ticker."""

    def __init__(
        self,
        factors: List[Tuple[Factor, float]],
        winsorize_pct: Tuple[float, float] = (1, 99),
    ):
        """
        Args:
            factors: List of (Factor, weight) tuples.
            winsorize_pct: Lower and upper percentile for winsorization.
        """
        self.factors = factors
        self.winsorize_pct = winsorize_pct

    def score(self, prices, fundamentals, date):
        """Compute composite scores for all tickers.

        Returns:
            Series of composite scores, sorted descending (best first).
        """
        all_scores = {}
        for factor, weight in self.factors:
            raw = factor.compute(prices, fundamentals, date)
            if raw.empty:
                logger.warning(f"Factor {factor.name} returned empty, skipping")
                continue

            # Winsorize
            lower = np.percentile(raw.dropna(), self.winsorize_pct[0])
            upper = np.percentile(raw.dropna(), self.winsorize_pct[1])
            clipped = raw.clip(lower, upper)

            # Cross-sectional z-score
            z = (clipped - clipped.mean()) / clipped.std()

            # Flip sign for "lower_is_better" factors
            if factor.direction == "lower_is_better":
                z = -z

            all_scores[factor.name] = z * weight

        if not all_scores:
            return pd.Series(dtype=float)

        # Combine: sum weighted z-scores
        score_df = pd.DataFrame(all_scores)
        composite = score_df.sum(axis=1)

        # Drop NaN and sort descending
        composite = composite.dropna().sort_values(ascending=False)
        logger.info(f"Computed composite scores for {len(composite)} tickers")
        return composite
