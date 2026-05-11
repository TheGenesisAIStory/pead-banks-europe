"""
src/data_loading.py
Data ingestion: prices (yfinance), synthetic earnings events, macro proxies.
All data is fetched with strict date ordering and no future information.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
from typing import Dict, List, Optional

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.experiment import UNIVERSE, START_DATE, END_DATE, EARNINGS_APPROX, MARKET_PROXY


def load_prices(tickers: Optional[List[str]] = None,
               start: str = START_DATE,
               end: str = END_DATE) -> pd.DataFrame:
    """
    Download adjusted daily OHLCV for all universe banks + market proxy.
    Returns a MultiIndex DataFrame (ticker, field) → long panel.
    """
    if tickers is None:
        tickers = list(UNIVERSE.keys()) + [MARKET_PROXY]
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False, threads=True)
    # flatten MultiIndex columns: (field, ticker) → ticker_field
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [f"{ticker}_{field}" for field, ticker in raw.columns]
    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index()
    return raw


def build_long_panel(prices_wide: pd.DataFrame,
                     tickers: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Reshape wide prices DataFrame into long panel:
    (date, ticker) with columns: open, high, low, close, volume.
    """
    if tickers is None:
        tickers = list(UNIVERSE.keys())
    records = []
    for tk in tickers:
        cols = {"close": f"{tk}_Close", "open": f"{tk}_Open",
                "high": f"{tk}_High",  "low": f"{tk}_Low",
                "volume": f"{tk}_Volume"}
        avail = {k: v for k, v in cols.items() if v in prices_wide.columns}
        if "close" not in avail:
            continue
        sub = prices_wide[[avail[k] for k in avail]].copy()
        sub.columns = list(avail.keys())
        sub["ticker"] = tk
        sub["name"]   = UNIVERSE.get(tk, tk)
        sub.index.name = "date"
        records.append(sub.reset_index())
    panel = pd.concat(records, ignore_index=True)
    panel = panel.dropna(subset=["close"])
    panel = panel[panel["close"] > 0]
    panel["ret_1d"] = panel.groupby("ticker")["close"].pct_change()
    return panel


def build_events_synthetic(years_range: Optional[range] = None) -> pd.DataFrame:
    """
    Build synthetic earnings event dates from config.EARNINGS_APPROX.
    For each (ticker, year, quarter) generates an event_date.
    Flag: event_type = 'synthetic'.
    """
    if years_range is None:
        years_range = range(
            int(START_DATE[:4]),
            int(END_DATE[:4]) + 1
        )
    rows = []
    for ticker, dates_list in EARNINGS_APPROX.items():
        for year in years_range:
            for q_idx, (month, day) in enumerate(dates_list, start=1):
                try:
                    evt_date = pd.Timestamp(year=year, month=month, day=day)
                except ValueError:
                    continue
                rows.append({
                    "ticker":     ticker,
                    "event_date": evt_date,
                    "quarter":    f"{year}Q{q_idx}",
                    "event_type": "synthetic",
                })
    ev = pd.DataFrame(rows)
    ev = ev.sort_values(["ticker", "event_date"]).reset_index(drop=True)
    ev["event_id"] = ev.index
    return ev


def load_market_returns(prices_wide: pd.DataFrame,
                        market_col: str = f"{MARKET_PROXY}_Close") -> pd.Series:
    """
    Extract daily market returns from the wide panel.
    """
    if market_col not in prices_wide.columns:
        cols = [c for c in prices_wide.columns if MARKET_PROXY in c and "Close" in c]
        if not cols:
            return pd.Series(dtype=float, name="market_ret")
        market_col = cols[0]
    mkt = prices_wide[market_col].pct_change()
    mkt.name = "market_ret"
    return mkt


def merge_events_with_prices(events: pd.DataFrame,
                             panel: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join events onto the panel: for each event find the
    nearest trading day (±3 days) in the price panel.
    """
    trading_days = (
        panel.groupby("ticker")["date"]
        .apply(lambda s: sorted(s.unique()))
        .to_dict()
    )

    def nearest_trading_day(ticker, target_date):
        days = trading_days.get(ticker, [])
        if not days:
            return None
        deltas = [(abs((pd.Timestamp(d) - target_date).days), d) for d in days]
        best = min(deltas, key=lambda x: x[0])
        if best[0] <= 3:
            return best[1]
        return None

    events = events.copy()
    events["event_trading_date"] = events.apply(
        lambda r: nearest_trading_day(r["ticker"], r["event_date"]), axis=1
    )
    return events.dropna(subset=["event_trading_date"])
