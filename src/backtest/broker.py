from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict

import numpy as np

logger = logging.getLogger("making_money")


@dataclass
class TradeResult:
    ticker: str
    shares: float
    fill_price: float
    commission: float
    slippage_cost: float
    side: str  # "buy" or "sell"

    @property
    def total_cost(self):
        return abs(self.shares * self.fill_price) + self.commission


class SimulatedBroker:
    """Simulates trade execution with slippage and commissions."""

    def __init__(
        self,
        commission_per_share=0.005,
        min_commission=1.00,
        half_spread_bps=5,
        impact_coefficient=0.1,
        slippage_model="sqrt_impact",
    ):
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        self.half_spread_bps = half_spread_bps
        self.impact_coefficient = impact_coefficient
        self.slippage_model = slippage_model
        self.total_commissions = 0.0
        self.total_slippage = 0.0

    def execute_trades(self, trades, prices, volumes=None):
        """Execute a dict of trades.

        Args:
            trades: Dict[ticker, shares_delta]. Positive = buy, negative = sell.
            prices: Dict[ticker, close_price].
            volumes: Dict[ticker, average_daily_volume] (optional, for slippage).

        Returns:
            List of TradeResult, total cash impact (negative = spent).
        """
        results = []
        cash_impact = 0.0

        for ticker, shares in trades.items():
            if abs(shares) < 1e-6:
                continue

            price = prices.get(ticker, 0)
            if price <= 0:
                continue

            side = "buy" if shares > 0 else "sell"
            adv = volumes.get(ticker, 1e8) if volumes else 1e8
            trade_value = abs(shares * price)

            # Compute slippage
            slippage_bps = self._compute_slippage(trade_value, adv)
            if side == "buy":
                fill_price = price * (1 + slippage_bps / 10000)
            else:
                fill_price = price * (1 - slippage_bps / 10000)

            # Commission
            commission = max(abs(shares) * self.commission_per_share, self.min_commission)

            # Cash impact
            if side == "buy":
                cash_impact -= abs(shares) * fill_price + commission
            else:
                cash_impact += abs(shares) * fill_price - commission

            slippage_cost = abs(shares) * abs(fill_price - price)
            self.total_commissions += commission
            self.total_slippage += slippage_cost

            results.append(TradeResult(
                ticker=ticker,
                shares=shares,
                fill_price=fill_price,
                commission=commission,
                slippage_cost=slippage_cost,
                side=side,
            ))

        return results, cash_impact

    def _compute_slippage(self, trade_value, adv):
        """Compute slippage in basis points."""
        if self.slippage_model == "fixed":
            return self.half_spread_bps
        elif self.slippage_model == "sqrt_impact":
            impact = self.impact_coefficient * np.sqrt(trade_value / max(adv, 1))
            return self.half_spread_bps + impact * 10000
        return self.half_spread_bps
