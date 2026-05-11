"""
src/backtest_strategies.py
Event-driven backtest: long-only Q5 and long-short Q5-Q1.
Supports composite_score, sue_proxy, or ML p_hat as signal.
"""

import numpy as np
import pandas as pd
from typing import Optional

from config.experiment import COST_BPS, SLIPPAGE_BPS, BORROW_BPS, N_QUANTILES, HOLD_HORIZON


def assign_quantiles(ef: pd.DataFrame,
                     signal_col: str = "composite_score",
                     n_q: int = N_QUANTILES) -> pd.DataFrame:
    """
    Assign quantile rank (1 = bottom, n_q = top) within each event_date cross-section.
    """
    df = ef.copy()
    df["signal_q"] = df.groupby("event_date")[signal_col].transform(
        lambda s: pd.qcut(s.rank(method="first"), n_q,
                          labels=False, duplicates="drop") + 1
        if s.nunique() >= n_q else np.nan
    )
    return df


def compute_portfolio_returns(
    ef: pd.DataFrame,
    daily_panel: pd.DataFrame,
    signal_col: str = "composite_score",
    horizon: int = HOLD_HORIZON,
    n_q: int = N_QUANTILES,
    long_short: bool = True,
    cost_bps: float = COST_BPS,
    slippage_bps: float = SLIPPAGE_BPS,
    borrow_bps: float = BORROW_BPS,
) -> pd.DataFrame:
    """
    Compute daily portfolio returns for Q5 (long) and Q1 (short).
    Each event opens a position at event_trading_date+1 open, closes at +H.
    Costs applied at entry and exit.
    Returns a daily returns DataFrame.
    """
    ef_q = assign_quantiles(ef, signal_col, n_q)
    ef_q = ef_q.dropna(subset=["signal_q", "event_trading_date"] if "event_trading_date" in ef_q.columns else ["signal_q"])

    price_idx = daily_panel.set_index(["ticker", "date"])["ret_1d"]

    positions = []  # list of (date, ticker, ret, leg)

    for _, row in ef_q.iterrows():
        if row["signal_q"] not in [1, n_q]:
            continue
        leg = "long" if row["signal_q"] == n_q else "short"
        tk  = row["ticker"]
        try:
            entry_date = pd.Timestamp(row.get("event_trading_date", row["event_date"]))
        except Exception:
            continue

        fwd = daily_panel[
            (daily_panel["ticker"] == tk) &
            (daily_panel["date"] > entry_date)
        ].sort_values("date").head(horizon)

        if len(fwd) == 0:
            continue

        rt_cost = (cost_bps + slippage_bps) / 1e4  # round-trip
        borrow_daily = (borrow_bps / 1e4) / 252 if leg == "short" else 0.0

        for i, (_, pr) in enumerate(fwd.iterrows()):
            rt = pr["ret_1d"] if not np.isnan(pr["ret_1d"]) else 0.0
            if leg == "short":
                rt = -rt - borrow_daily
            if i == 0:
                rt -= rt_cost / 2  # entry
            if i == len(fwd) - 1:
                rt -= rt_cost / 2  # exit
            positions.append({"date": pr["date"], "ticker": tk, "ret": rt, "leg": leg})

    if not positions:
        return pd.DataFrame(columns=["date", "long_ret", "short_ret", "ls_ret"])

    pos_df = pd.DataFrame(positions)
    long_r  = pos_df[pos_df["leg"]=="long"].groupby("date")["ret"].mean().rename("long_ret")
    short_r = pos_df[pos_df["leg"]=="short"].groupby("date")["ret"].mean().rename("short_ret")
    ls = pd.concat([long_r, short_r], axis=1).fillna(0)
    ls["ls_ret"] = ls["long_ret"] + ls["short_ret"]
    return ls.reset_index()


def performance_summary(port_rets: pd.DataFrame,
                        col: str = "long_ret",
                        freq: int = 252) -> dict:
    """Compute CAGR, Sharpe, Sortino, max drawdown, hit ratio."""
    r = port_rets.set_index("date")[col].dropna()
    if len(r) == 0:
        return {}
    cum   = (1 + r).cumprod()
    cagr  = cum.iloc[-1] ** (freq / len(r)) - 1
    sharpe= r.mean() / (r.std() + 1e-9) * np.sqrt(freq)
    neg   = r[r < 0]
    sortino = r.mean() / (neg.std() + 1e-9) * np.sqrt(freq)
    roll_max = cum.cummax()
    dd    = (cum - roll_max) / roll_max
    mdd   = dd.min()
    hit   = (r > 0).mean()
    return {"cagr": cagr, "sharpe": sharpe, "sortino": sortino,
            "mdd": mdd, "hit_ratio": hit, "n_obs": len(r)}
