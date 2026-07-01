"""Regime service. MACRO bot owns the full macro state vector; this module ships
the shared, always-available price-based classifier the whole fleet consumes as a
Tier-C gate.

v0 generalizes the existing TS `isRiskOn` two-gate SPY filter into a labeled state.
The MACRO bot upgrades this with FRED macro features and a GMM classifier; the
interface (produce a RegimeState) stays fixed so downstream code never changes.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from .indicators import atr, ema
from .types import RegimeState, RegimeTag


def classify_price_regime(spy: pd.DataFrame, now: datetime) -> RegimeState:
    """Label the market from SPY alone. Auditable, no external data required.

    Gates mirror the original two-gate risk-on check, extended into four labels
    plus a high-vol overlay:
      - trending-up   : above 200-EMA AND within 8% of 52-wk high
      - risk-off      : below 200-EMA
      - range         : above 200-EMA but well off highs, low vol
      - high-vol      : ATR% elevated regardless of trend
    """
    df = spy.loc[spy.index <= now]
    if len(df) < 200:
        return RegimeState(at=now, label=RegimeTag.CHOP, vector={}, conviction=0.0)

    close = df["c"]
    ema200 = ema(close, 200)
    last = float(close.iloc[-1])
    ema200_last = float(ema200.iloc[-1])
    ema200_rising = float(ema200.iloc[-1]) > float(ema200.iloc[-21])
    high_52w = float(close.tail(252).max())
    pct_of_high = last / high_52w if high_52w else 0.0

    atr_pct = float((atr(df["h"], df["l"], close, 14).iloc[-1] / last) * 100)
    vector = {
        "above_200ema": float(last > ema200_last),
        "ema200_rising": float(ema200_rising),
        "pct_of_52w_high": pct_of_high,
        "atr_pct": atr_pct,
    }

    if last < ema200_last:
        label = RegimeTag.RISK_OFF
    elif atr_pct > 2.5:
        label = RegimeTag.HIGH_VOL
    elif pct_of_high >= 0.92 and ema200_rising:
        label = RegimeTag.TRENDING_UP
    else:
        label = RegimeTag.RANGE

    # Conviction: distance of pct-of-high from the 0.92 gate, clipped to 0..1.
    conviction = float(np.clip(abs(pct_of_high - 0.92) / 0.08, 0.0, 1.0))
    return RegimeState(at=now, label=label, vector=vector, conviction=conviction)
