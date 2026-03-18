from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.factors.composite import CompositeScorer
from src.portfolio.constraints import PortfolioConstraints
from src.portfolio.optimizer import PortfolioOptimizer

logger = logging.getLogger("making_money")


class PortfolioConstructor:
    """End-to-end pipeline: factor scores → optimized portfolio weights."""

    def __init__(
        self,
        scorer: CompositeScorer,
        optimizer: PortfolioOptimizer,
        constraints: PortfolioConstraints,
        top_n: int = 50,
        lookback_days: int = 252,
    ):
        self.scorer = scorer
        self.optimizer = optimizer
        self.constraints = constraints
        self.top_n = top_n
        self.lookback_days = lookback_days

    def construct(
        self,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame,
        date: pd.Timestamp,
        current_weights: dict = None,
    ) -> Dict[str, float]:
        """Build target portfolio for a given date.

        Returns:
            Dict mapping ticker to target weight.
        """
        # 1. Score all tickers
        scores = self.scorer.score(prices, fundamentals, date)
        if scores.empty:
            logger.warning(f"No scores computed for {date}")
            return {}

        # 2. Select top N
        top_tickers = scores.head(self.top_n).index.tolist()

        # 3. Get trailing returns for covariance estimation
        available_prices = prices.loc[:date, top_tickers]
        trailing = available_prices.tail(self.lookback_days)
        returns = trailing.pct_change().dropna()

        if len(returns) < 30:
            logger.warning(f"Insufficient return data ({len(returns)} days) for optimization")
            # Fall back to equal weight
            n = len(top_tickers)
            return {t: 1.0 / n for t in top_tickers}

        # 4. Estimate covariance and expected returns
        cov_matrix = self.optimizer.estimate_covariance(returns)
        expected_returns = returns.mean().values

        # 5. Get sector info for constraints
        sectors = None
        if "sector" in fundamentals.columns:
            sectors = [fundamentals.loc[t, "sector"] if t in fundamentals.index else "" for t in top_tickers]

        # 6. Prepare current weights array
        curr_w = None
        if current_weights:
            curr_w = np.array([current_weights.get(t, 0.0) for t in top_tickers])

        # 7. Optimize
        weights = self.optimizer.optimize(
            expected_returns, cov_matrix, self.constraints,
            current_weights=curr_w, sectors=sectors,
        )

        # 8. Build result dict, filter near-zero weights
        result = {}
        for ticker, w in zip(top_tickers, weights):
            if w > 1e-6:
                result[ticker] = float(w)

        logger.info(f"Constructed portfolio: {len(result)} holdings on {date.date()}")
        return result
