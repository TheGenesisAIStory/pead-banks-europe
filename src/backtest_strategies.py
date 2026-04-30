"""backtest_strategies.py
Strategie PEAD: overnight (0-3gg) e drift 60 giorni.
Include sizing risk-based, stop loss dinamici e costi di transazione.
"""
import numpy as np
import pandas as pd
import yaml


def load_config(path: str = "config/params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_costs(gross_ret: float, cfg: dict, event: bool = False) -> float:
    comm = cfg.get("commission_rate", 0.0005)
    slip = cfg.get("slippage_event", 0.0010) if event else cfg.get("slippage_normal", 0.0003)
    return gross_ret - 2 * (comm + slip)


def run_overnight_strategy(signals_df: pd.DataFrame,
                            close: pd.DataFrame, open_: pd.DataFrame,
                            cfg: dict,
                            score_col: str = "Score_ens") -> pd.DataFrame:
    """Long overnight: chiude a open del giorno successivo all'annuncio."""
    high_thr = cfg.get("grade_high", 0.70)
    orj_thr = cfg.get("overnight_orj_thr", 0.0)

    trades = []
    for _, row in signals_df.dropna(subset=[score_col]).iterrows():
        t = row["ticker"]
        d = row["earn_date"]
        if row[score_col] < high_thr or row.get("ORJ", 0) <= orj_thr:
            continue
        if d not in close.index:
            continue
        idx = close.index.get_loc(d)
        if idx + 1 >= len(close.index):
            continue
        entry_date = close.index[idx]
        exit_date = close.index[idx + 1]
        entry_price = close.loc[entry_date, t]
        exit_price = open_.loc[exit_date, t]
        if any(np.isnan([entry_price, exit_price])) or entry_price == 0:
            continue
        gross = (exit_price / entry_price) - 1
        net = apply_costs(gross, cfg, event=True)
        trades.append({"ticker": t, "entry_date": entry_date, "exit_date": exit_date,
                       "gross_ret": gross, "ret_net": net, "strategy": "overnight"})
    return pd.DataFrame(trades)


def run_drift60_strategy(signals_df: pd.DataFrame,
                          close: pd.DataFrame,
                          cfg: dict,
                          score_col: str = "Score_ens") -> pd.DataFrame:
    """Long drift 60gg su titoli High grade con stop loss dinamico."""
    hold_days = cfg.get("drift_hold_days", 60)
    high_thr = cfg.get("grade_high", 0.70)
    K = cfg.get("K_vol_stop", 2.0)
    risk_capital = cfg.get("risk_capital", 1.0)
    risk_per_trade = cfg.get("risk_per_trade", 0.02)

    # Volatilità rolling 60gg
    vol_rolling = close.pct_change().rolling(cfg.get("vol_lookback", 60)).std()

    trades = []
    for _, row in signals_df.dropna(subset=[score_col, "SUE_z"]).iterrows():
        t = row["ticker"]
        d = row["earn_date"]
        if row[score_col] < high_thr or row["SUE_z"] <= 0:
            continue
        if d not in close.index:
            continue
        idx = close.index.get_loc(d)
        exit_idx = idx + hold_days
        if exit_idx >= len(close.index):
            continue
        entry_date = close.index[idx]
        entry_price = close.loc[entry_date, t]

        vol_here = vol_rolling.loc[entry_date, t] if entry_date in vol_rolling.index else np.nan
        if np.isnan(vol_here) or vol_here == 0:
            stop_level = entry_price * 0.90  # fallback -10%
        else:
            stop_level = entry_price * (1 - K * vol_here)

        # Sizing risk-based
        risk_per_unit = entry_price - stop_level
        units = (risk_capital * risk_per_trade) / risk_per_unit if risk_per_unit > 0 else 0

        # Simula giorno per giorno fino a stop o exit
        trade_slice = close.loc[entry_date:close.index[exit_idx], t]
        exit_date = trade_slice.index[-1]
        exit_price = trade_slice.iloc[-1]
        stopped = False
        for date, price in trade_slice.items():
            if price <= stop_level:
                exit_date, exit_price, stopped = date, price, True
                break

        gross = (exit_price / entry_price) - 1
        net = apply_costs(gross, cfg, event=False)
        trades.append({"ticker": t, "entry_date": entry_date, "exit_date": exit_date,
                       "entry_price": entry_price, "exit_price": exit_price,
                       "gross_ret": gross, "ret_net": net,
                       "stopped": stopped, "units": units, "strategy": "drift60"})
    return pd.DataFrame(trades)


def build_combined_equity(ov_df: pd.DataFrame, dr_df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Costruisce equity combinata overnight + drift60 con pesi da config."""
    w_ov = cfg.get("weight_overnight", 0.30)
    w_dr = cfg.get("weight_drift60", 0.70)

    ov = ov_df.sort_values("entry_date").reset_index(drop=True)
    dr = dr_df.sort_values("entry_date").reset_index(drop=True)

    ov["equity"] = (1 + ov["ret_net"]).cumprod()
    dr["equity"] = (1 + dr["ret_net"]).cumprod()

    all_dates = pd.to_datetime(sorted(set(ov["entry_date"]) | set(dr["entry_date"])))
    ov_eq = ov.set_index("entry_date")["equity"].reindex(all_dates).ffill().fillna(1.0)
    dr_eq = dr.set_index("entry_date")["equity"].reindex(all_dates).ffill().fillna(1.0)
    combined = 1.0 + w_ov * (ov_eq - 1.0) + w_dr * (dr_eq - 1.0)
    return combined


if __name__ == "__main__":
    print("backtest_strategies: usa run_overnight_strategy() e run_drift60_strategy().")
