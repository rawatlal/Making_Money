from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import pandas as pd


@dataclass
class BacktestReport:
    """Structured container for backtest results."""

    summary_stats: Dict[str, float] = field(default_factory=dict)
    returns_series: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    benchmark_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    weights_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    turnover_history: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    transaction_costs: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    initial_capital: float = 1_000_000
    final_value: float = 0.0

    def to_dict(self):
        """Serialize report for Streamlit consumption."""
        return {
            "summary": self.summary_stats,
            "initial_capital": self.initial_capital,
            "final_value": self.final_value,
        }
