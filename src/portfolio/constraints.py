from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class PortfolioConstraints:
    """Portfolio construction constraints for the optimizer."""

    long_only: bool = True
    max_position_weight: float = 0.05
    min_position_weight: float = 0.005
    max_sector_weight: float = 0.30
    max_turnover_per_rebalance: float = 0.50
    max_holdings: int = 50

    def get_bounds(self, n_assets):
        """Return scipy-compatible bounds for each asset weight."""
        lb = 0.0 if self.long_only else -self.max_position_weight
        return [(lb, self.max_position_weight)] * n_assets

    def get_base_constraints(self, n_assets):
        """Return list of scipy constraint dicts (weights sum to 1)."""
        return [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        ]

    def get_sector_constraints(self, sectors, unique_sectors):
        """Return sector weight constraints.

        Args:
            sectors: array of sector labels for each asset.
            unique_sectors: list of unique sector names.
        """
        constraints = []
        for sector in unique_sectors:
            mask = np.array([s == sector for s in sectors], dtype=float)
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w, m=mask: self.max_sector_weight - np.dot(m, w),
                }
            )
        return constraints

    def get_turnover_constraint(self, current_weights):
        """Return turnover constraint: sum(|w_new - w_current|) <= max_turnover."""
        return {
            "type": "ineq",
            "fun": lambda w: self.max_turnover_per_rebalance - np.sum(np.abs(w - current_weights)),
        }
