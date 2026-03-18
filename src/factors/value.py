from __future__ import annotations

import pandas as pd

from src.factors.base import Factor


class PERatio(Factor):
    """Price-to-Earnings ratio factor. Lower is better (cheap stocks)."""

    name = "pe_ratio"
    direction = "lower_is_better"

    def compute(self, prices, fundamentals, date):
        if "pe_ratio" in fundamentals.columns:
            return fundamentals["pe_ratio"].dropna()
        return pd.Series(dtype=float)


class PBRatio(Factor):
    """Price-to-Book ratio factor. Lower is better (cheap stocks)."""

    name = "pb_ratio"
    direction = "lower_is_better"

    def compute(self, prices, fundamentals, date):
        if "pb_ratio" in fundamentals.columns:
            return fundamentals["pb_ratio"].dropna()
        return pd.Series(dtype=float)


class EVToEBITDA(Factor):
    """Enterprise Value to EBITDA factor. Lower is better."""

    name = "ev_ebitda"
    direction = "lower_is_better"

    def compute(self, prices, fundamentals, date):
        if "enterprise_value" in fundamentals.columns and "ebitda" in fundamentals.columns:
            ev = fundamentals["enterprise_value"]
            ebitda = fundamentals["ebitda"]
            # Avoid division by zero or negative EBITDA
            valid = (ebitda > 0) & ev.notna() & ebitda.notna()
            ratio = pd.Series(dtype=float, index=fundamentals.index)
            ratio[valid] = ev[valid] / ebitda[valid]
            return ratio.dropna()
        return pd.Series(dtype=float)
