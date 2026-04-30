"""features_pead.py
Calcola metriche PEAD: ORJ, EAR, SUE proxy, OFI, vol spike, score lineari.
"""
import pandas as pd
import numpy as np
import yaml


def load_config(path: str = "config/params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def zscore(x: pd.Series) -> pd.Series:
    std = x.std()
    if std == 0 or np.isnan(std):
        return x * 0
    return (x - x.mean()) / std


def compute_orj(ticker: str, earn_date: pd.Timestamp,
                close: pd.DataFrame, open_: pd.DataFrame) -> float:
    """Overnight Return Jump: (open post - close pre) / close pre."""
    if earn_date not in close.index:
        return np.nan
    idx = close.index.get_loc(earn_date)
    if idx == 0 or idx + 1 >= len(close.index):
        return np.nan
    pre_close = close.iloc[idx - 1][ticker]
    post_open = open_.iloc[idx + 1][ticker]
    if np.isnan(pre_close) or np.isnan(post_open) or pre_close == 0:
        return np.nan
    return (post_open - pre_close) / pre_close


def compute_ear(ticker: str, earn_date: pd.Timestamp,
                close: pd.DataFrame, bench: pd.Series,
                window: int = 1) -> float:
    """Earnings Announcement Return: ritorno titolo - ritorno benchmark su finestra [-w, +w]."""
    if earn_date not in close.index:
        return np.nan
    idx = close.index.get_loc(earn_date)
    s = max(idx - window, 0)
    e = min(idx + window, len(close.index) - 1)
    dates = close.index[s:e + 1]
    p0 = close.loc[dates[0], ticker]
    p1 = close.loc[dates[-1], ticker]
    b0 = bench.loc[dates[0]]
    b1 = bench.loc[dates[-1]]
    if any(np.isnan([p0, p1, b0, b1])) or p0 == 0 or b0 == 0:
        return np.nan
    return (p1 / p0 - 1) - (b1 / b0 - 1)


def compute_sue_proxy(series: pd.Series, lookback: int = 8) -> pd.Series:
    """SUE proxy: standardizza ogni osservazione rispetto alla storia recente."""
    sue = [np.nan] * len(series)
    for i in range(lookback, len(series)):
        hist = series.iloc[i - lookback:i].dropna()
        if hist.empty or hist.std() == 0:
            continue
        sue[i] = (series.iloc[i] - hist.mean()) / hist.std()
    return pd.Series(sue, index=series.index)


def compute_ofi_and_volspike(ticker: str, earn_date: pd.Timestamp,
                              close: pd.DataFrame, open_: pd.DataFrame,
                              volume: pd.DataFrame,
                              vol_lookback: int = 20) -> tuple[float, float]:
    """Proxy OFI e Volume Spike."""
    if earn_date not in close.index:
        return np.nan, np.nan
    idx = close.index.get_loc(earn_date)
    o = open_.iloc[idx][ticker]
    c = close.iloc[idx][ticker]
    v = volume.iloc[idx][ticker]
    if np.isnan(o) or np.isnan(c) or o == 0:
        return np.nan, np.nan
    intraday = (c / o) - 1
    start_idx = max(idx - vol_lookback, 0)
    v_mean = volume.iloc[start_idx:idx][ticker].mean()
    if np.isnan(v_mean) or v_mean == 0:
        return np.nan, np.nan
    ofi = np.sign(intraday) * (v / v_mean)
    spike = v / v_mean
    return ofi, spike


def build_signals(earnings_cal: pd.DataFrame,
                  close: pd.DataFrame, open_: pd.DataFrame,
                  volume: pd.DataFrame, bench: pd.Series,
                  fund_df: pd.DataFrame | None,
                  cfg: dict) -> pd.DataFrame:
    """Pipeline completa: costruisce signals_df con tutte le metriche e score."""
    records = []
    for _, row in earnings_cal.iterrows():
        t = row["ticker"]
        d = row["earn_date"]
        orj = compute_orj(t, d, close, open_)
        ear = compute_ear(t, d, close, bench, window=cfg.get("ear_window", 1))
        ofi, spike = compute_ofi_and_volspike(
            t, d, close, open_, volume,
            vol_lookback=cfg.get("vol_spike_lookback", 20)
        )
        rec = {"ticker": t, "earn_date": d,
               "ORJ": orj, "EAR": ear,
               "OFI_raw": ofi, "vol_spike": spike,
               "guidance_score": 0, "cet1_surprise": 0,
               "nim_surprise": 0, "prov_surprise": 0}
        if fund_df is not None:
            from src.data_loading import get_fundamentals_at
            f = get_fundamentals_at(fund_df, t, d)
            if f:
                rec["guidance_score"] = f.get("guidance_score", 0)
                rec["cet1_surprise"] = f.get("cet1_surprise", 0)
                rec["nim_surprise"] = f.get("nim_surprise", 0)
                rec["prov_surprise"] = f.get("prov_surprise", 0)
                rec["roe"] = f.get("roe", np.nan)
                rec["pe"] = f.get("pe", np.nan)
                rec["cet1"] = f.get("cet1", np.nan)
        records.append(rec)

    df = pd.DataFrame(records)

    # SUE proxy per ticker
    sue_list = []
    for t, grp in df.groupby("ticker"):
        grp = grp.copy()
        grp["SUE_proxy"] = compute_sue_proxy(
            grp["ORJ"], lookback=cfg.get("sue_lookback", 8)
        ).values
        sue_list.append(grp)
    df = pd.concat(sue_list).reset_index(drop=True)

    # Z-score features
    for col, z_col in [("SUE_proxy", "SUE_z"), ("ORJ", "ORJ_z"),
                       ("OFI_raw", "OFI_z"), ("guidance_score", "Guidance_z"),
                       ("cet1_surprise", "CET1surp_z"), ("vol_spike", "VOL_z")]:
        df[z_col] = df.groupby("ticker")[col].transform(zscore)

    # Score lineari
    w1 = cfg.get("w_orj", 0.6)
    w2 = cfg.get("w_sue", 0.4)
    df["Score_PEAD"] = w1 * df["ORJ_z"] + w2 * df["SUE_z"]

    ws = cfg.get
    df["CompositeBankScore"] = (
        cfg.get("weight_sue", 0.30) * df["SUE_z"]
        + cfg.get("weight_ofi", 0.25) * df["OFI_z"]
        + cfg.get("weight_guidance", 0.20) * df["Guidance_z"]
        + cfg.get("weight_cet1surp", 0.15) * df["CET1surp_z"]
        + cfg.get("weight_volspike", 0.10) * df["VOL_z"]
    )

    return df


if __name__ == "__main__":
    print("features_pead: usa build_signals() dal notebook o da backtest_strategies.")
