"""Vectorized technical indicators. Pandas/NumPy port of the TS `lib/indicators.ts`.

All functions take and return pandas Series aligned to the input index, so they
compose cleanly inside the backtest engine and the bots.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    up = high.diff()
    down = -low.diff()
    plus_dm = ((up > down) & (up > 0)) * up
    minus_dm = ((down > up) & (down > 0)) * down
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    atr_ = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr_
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def donchian(high: pd.Series, low: pd.Series, n: int) -> tuple[pd.Series, pd.Series]:
    return high.rolling(n).max(), low.rolling(n).min()


def clenow_momentum(close: pd.Series, n: int = 90) -> pd.Series:
    """Clenow momentum: annualized slope of the log-price regression * R^2.

    Rewards smooth, persistent trends and penalizes choppy ones. Returns a Series
    of the score computed over a trailing window of length `n`.
    """
    log_close = np.log(close)
    idx = np.arange(n)

    def _score(window: np.ndarray) -> float:
        if np.isnan(window).any():
            return np.nan
        slope, intercept = np.polyfit(idx, window, 1)
        fitted = slope * idx + intercept
        ss_res = np.sum((window - fitted) ** 2)
        ss_tot = np.sum((window - window.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        annualized = (np.exp(slope * 252) - 1) * 100
        return annualized * r2

    return log_close.rolling(n).apply(_score, raw=True)


def rolling_rank(df_returns: pd.DataFrame, at_col: str | None = None) -> pd.Series:
    """Cross-sectional percentile rank (0..100) across symbols for the latest row.

    `df_returns` is symbols-as-columns. Used for O'Neil-style RS ranking.
    """
    row = df_returns.iloc[-1] if at_col is None else df_returns.loc[at_col]
    return row.rank(pct=True) * 100
