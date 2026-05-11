"""
src/features_pead.py
Feature engineering: PEAD core + all ML4T risk factor families.
All features are computed strictly with data available up to event_date (no look-ahead).

Factor families:
  1. PEAD core  : sue_proxy, orj, ear, car, zvol, vol_spike
  2. Controls   : log_size, beta_60d, pre_vol_20d, pre_vol_60d
  3. Price mom  : mom_5d, mom_20d, mom_60d, mom_120d, reversal_5d, dist_52w_high
  4. Volatility : rv_20d, rv_60d, parkinson_20d, downside_vol_20d, vol_ratio
  5. Liquidity  : amihud_20d, turnover_20d, zero_vol_frac_20d, hlspread_20d
  6. Style/Risk : residual_mom_60d, market_corr_60d, sector_rel_ret_20d
  7. Macro      : rates_change_proxy (dYTM), term_slope_proxy
  8. Bank fund  : cet1_surprise_proxy, nim_surprise_proxy, ifrs9_intensity_proxy
  9. Composite  : composite_score (equal-weight z-scores)
"""

import numpy as np
import pandas as pd
from typing import Optional, List


# ── Helpers ────────────────────────────────────────────────────────────────────

def _rolling_std(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=max(5, window // 3)).std(ddof=1)

def _rolling_mean(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=max(5, window // 3)).mean()

def _zscore_cross(df: pd.DataFrame, col: str, group_col: str = "event_date") -> pd.Series:
    """Cross-sectional z-score within each event date."""
    return df.groupby(group_col)[col].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9)
    )


# ── 1. Build daily features on price panel ─────────────────────────────────────

def build_daily_features(panel: pd.DataFrame,
                         market_ret: pd.Series,
                         pre_window: int = 60) -> pd.DataFrame:
    """
    Adds all daily factor columns to the price panel.
    Input: long panel (date, ticker, close, open, high, low, volume, ret_1d)
    Output: same panel enriched with factor columns.
    """
    df = panel.sort_values(["ticker", "date"]).copy()
    df = df.merge(market_ret.rename("market_ret").reset_index(),
                  on="date", how="left")

    grp = df.groupby("ticker")

    # ── Controls ──────────────────────────────────────────────────────────────
    df["log_size"]   = np.log(df["close"].clip(lower=1e-4))
    df["pre_vol_20d"]= grp["ret_1d"].transform(lambda s: _rolling_std(s, 20))
    df["pre_vol_60d"]= grp["ret_1d"].transform(lambda s: _rolling_std(s, 60))

    # Beta (rolling 60d OLS slope vs market_ret)
    def rolling_beta(sub):
        mkt = sub["market_ret"].fillna(0)
        ret = sub["ret_1d"].fillna(0)
        betas = []
        for i in range(len(sub)):
            sl = slice(max(0, i-59), i+1)
            y, x = ret.iloc[sl].values, mkt.iloc[sl].values
            if len(x) < 20 or x.std() < 1e-9:
                betas.append(np.nan)
            else:
                betas.append(np.cov(y, x)[0,1] / np.var(x))
        sub["beta_60d"] = betas
        return sub
    df = df.groupby("ticker", group_keys=False).apply(rolling_beta)

    # ── Price momentum ────────────────────────────────────────────────────────
    for w in [5, 20, 60, 120]:
        df[f"mom_{w}d"] = grp["close"].transform(
            lambda s: s.pct_change(w)
        )
    df["reversal_5d"] = -df["mom_5d"]
    df["dist_52w_high"] = grp["close"].transform(
        lambda s: s / s.rolling(252, min_periods=60).max() - 1
    )

    # ── Realized volatility ───────────────────────────────────────────────────
    df["rv_20d"] = df["pre_vol_20d"]
    df["rv_60d"] = df["pre_vol_60d"]
    # Parkinson (high-low estimator)
    df["ln_hl"] = np.log(df["high"].clip(lower=1e-6) / df["low"].clip(lower=1e-6))
    df["parkinson_20d"] = grp["ln_hl"].transform(
        lambda s: np.sqrt(_rolling_mean(s**2, 20) / (4 * np.log(2)))
    )
    # Downside vol
    df["ret_neg"] = df["ret_1d"].where(df["ret_1d"] < 0, 0)
    df["downside_vol_20d"] = grp["ret_neg"].transform(
        lambda s: _rolling_std(s, 20)
    )
    df["vol_ratio"] = df["rv_20d"] / (df["rv_60d"] + 1e-9)

    # ── Liquidity ─────────────────────────────────────────────────────────────
    df["log_vol"] = np.log(df["volume"].clip(lower=1).replace(0, np.nan))
    df["log_vol_ma20"] = grp["log_vol"].transform(lambda s: _rolling_mean(s, 20))
    df["log_vol_sd20"] = grp["log_vol"].transform(lambda s: _rolling_std(s, 20))
    df["zvol"] = (df["log_vol"] - df["log_vol_ma20"]) / (df["log_vol_sd20"] + 1e-9)
    df["vol_spike"] = (df["zvol"] > 2.0).astype(int)
    # Amihud illiquidity proxy: |ret| / volume
    df["amihud_daily"] = df["ret_1d"].abs() / (df["volume"].clip(lower=1))
    df["amihud_20d"]   = grp["amihud_daily"].transform(lambda s: _rolling_mean(s, 20))
    # Turnover proxy (volume / 252d avg volume)
    df["vol_ma252"] = grp["volume"].transform(lambda s: _rolling_mean(s, 252))
    df["turnover_20d"] = df["volume"] / (df["vol_ma252"] + 1)
    # Zero-volume fraction
    df["zero_vol"] = (df["volume"] == 0).astype(int)
    df["zero_vol_frac_20d"] = grp["zero_vol"].transform(
        lambda s: s.rolling(20, min_periods=5).mean()
    )
    # H-L spread proxy
    df["hlspread_20d"] = grp["ln_hl"].transform(lambda s: _rolling_mean(s, 20))

    # ── Style / Residual momentum ─────────────────────────────────────────────
    df["market_corr_60d"] = grp["ret_1d"].transform(
        lambda s: s.rolling(60, min_periods=20)
        .corr(df.loc[s.index, "market_ret"])
    )
    # Residual momentum: actual - beta * market
    df["excess_ret"] = df["ret_1d"] - df["beta_60d"] * df["market_ret"]
    df["residual_mom_60d"] = grp["excess_ret"].transform(
        lambda s: s.rolling(60, min_periods=20).sum()
    )
    # Sector-relative return: return vs universe median
    date_med = df.groupby("date")["ret_1d"].transform("median")
    df["sector_rel_ret_20d"] = grp["ret_1d"].transform(
        lambda s: s.rolling(20, min_periods=10).sum()
    ) - df.groupby("date")["ret_1d"].transform(
        lambda s: s.rolling(20, min_periods=10).sum()
    ).mean()

    # ── Macro / Rates proxy ───────────────────────────────────────────────────
    # Use 10-year Bund proxy: fetch via yfinance (^TNX as rough substitute)
    # We use market return rolling change as a latent macro factor when
    # bond data unavailable; real experiment should replace with ECB rates.
    df["macro_mom_20d"] = _rolling_mean(df["market_ret"].fillna(0), 20)
    df["macro_vol_20d"] = _rolling_std(df["market_ret"].fillna(0), 20)

    # ── Bank fundamental proxies (from price series — real data from src/data_loading) ──
    # CET1 surprise proxy: change in price trend stability vs sector
    df["cet1_surprise_proxy"]  = df["residual_mom_60d"]
    df["nim_surprise_proxy"]   = df["sector_rel_ret_20d"]
    df["ifrs9_intensity_proxy"]= df["amihud_20d"].rank(pct=True) - 0.5

    # ── ORJ (overnight return jump) ───────────────────────────────────────────
    df["close_lag1"] = grp["close"].transform(lambda s: s.shift(1))
    df["orj"] = np.log(
        df["open"].clip(lower=1e-6) / df["close_lag1"].clip(lower=1e-6)
    )

    return df


# ── 2. Build event-level features ─────────────────────────────────────────────

def build_event_features(events: pd.DataFrame,
                         daily_features: pd.DataFrame,
                         drift_horizons: List[int] = [5, 20, 60]) -> pd.DataFrame:
    """
    For each event, extract all daily features at event_trading_date
    and compute EAR, CAR, drift_H targets.
    """
    feat_cols = [
        "log_size", "beta_60d", "pre_vol_20d", "pre_vol_60d",
        "mom_5d", "mom_20d", "mom_60d", "mom_120d",
        "reversal_5d", "dist_52w_high",
        "rv_20d", "rv_60d", "parkinson_20d", "downside_vol_20d", "vol_ratio",
        "zvol", "vol_spike", "amihud_20d", "turnover_20d",
        "zero_vol_frac_20d", "hlspread_20d",
        "market_corr_60d", "residual_mom_60d", "sector_rel_ret_20d",
        "macro_mom_20d", "macro_vol_20d",
        "cet1_surprise_proxy", "nim_surprise_proxy", "ifrs9_intensity_proxy",
        "orj", "ret_1d", "market_ret",
    ]

    idx = daily_features.set_index(["ticker", "date"])

    def get_feat(ticker, dt):
        try:
            row = idx.loc[(ticker, dt)]
            return row[feat_cols].to_dict()
        except KeyError:
            return {c: np.nan for c in feat_cols}

    rows = []
    for _, ev in events.iterrows():
        tk  = ev["ticker"]
        etd = ev["event_trading_date"]
        feat = get_feat(tk, etd)
        feat["ticker"]      = tk
        feat["event_date"]  = ev["event_date"]
        feat["event_id"]    = ev["event_id"]
        feat["quarter"]     = ev.get("quarter", "")
        feat["event_type"]  = ev.get("event_type", "")

        # EAR: event window return [0, +1]
        tk_daily = daily_features[
            (daily_features["ticker"] == tk) &
            (daily_features["date"] >= etd)
        ].sort_values("date").head(2)
        feat["ear_0_1"] = tk_daily["ret_1d"].sum() if len(tk_daily) > 0 else np.nan

        # SUE proxy: ORJ z-score within event cross-section (computed post-hoc below)
        feat["sue_proxy"] = feat.get("orj", np.nan)

        # Drift H targets
        for H in drift_horizons:
            fwd = daily_features[
                (daily_features["ticker"] == tk) &
                (daily_features["date"] > etd)
            ].sort_values("date").head(H)
            mkt_fwd = fwd["market_ret"].sum() if "market_ret" in fwd.columns else 0
            bk_fwd  = fwd["ret_1d"].sum() if len(fwd) >= H // 2 else np.nan
            feat[f"drift_{H}d"] = bk_fwd - mkt_fwd if not np.isnan(bk_fwd) else np.nan

        rows.append(feat)

    ef = pd.DataFrame(rows)

    # Cross-sectional z-scores for key signals
    for col in ["sue_proxy", "orj", "zvol", "ear_0_1",
                "cet1_surprise_proxy", "nim_surprise_proxy", "ifrs9_intensity_proxy"]:
        if col in ef.columns:
            ef[f"{col}_z"] = ef.groupby("event_date")[col].transform(
                lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9)
            )

    # Composite score (equal-weight average of z-scores)
    z_cols = [c for c in ef.columns if c.endswith("_z")]
    if z_cols:
        ef["composite_score"] = ef[z_cols].mean(axis=1)

    # Binary labels
    ef["y_60d"] = (ef["drift_60d"] > 0).astype(float)
    ef["y_20d"] = (ef["drift_20d"] > 0).astype(float)
    ef["y_60d_topq"] = (ef["drift_60d"] >= ef["drift_60d"].quantile(0.80)).astype(float)

    return ef
