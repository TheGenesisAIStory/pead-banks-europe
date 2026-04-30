"""reporting.py
Metriche di performance e report: Sharpe, Sortino, Max Drawdown, grade report.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ------------------------------------------------------------------
# Metriche
# ------------------------------------------------------------------

def max_drawdown(equity: pd.Series) -> tuple[float, pd.Series]:
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return dd.min(), dd


def sharpe_ratio(returns: pd.Series, rf: float = 0.0, periods: int = 252) -> float:
    excess = returns - rf / periods
    if excess.std() == 0:
        return np.nan
    return (excess.mean() / excess.std()) * np.sqrt(periods)


def sortino_ratio(returns: pd.Series, rf: float = 0.0, periods: int = 252) -> float:
    excess = returns - rf / periods
    down = excess[excess < 0]
    if down.std() == 0:
        return np.nan
    return (excess.mean() / down.std()) * np.sqrt(periods)


def calmar_ratio(equity: pd.Series, periods: int = 252) -> float:
    returns = equity.pct_change().dropna()
    annual_ret = (equity.iloc[-1] / equity.iloc[0]) ** (periods / len(equity)) - 1
    mdd, _ = max_drawdown(equity)
    if mdd == 0:
        return np.nan
    return annual_ret / abs(mdd)


# ------------------------------------------------------------------
# Report riassuntivo
# ------------------------------------------------------------------

def strategy_stats(trades_df: pd.DataFrame,
                   ret_col: str = "ret_net",
                   name: str = "",
                   periods_per_year: int = 12) -> dict:
    rets = trades_df[ret_col].dropna()
    equity = (1 + rets).cumprod()
    mdd, _ = max_drawdown(equity)
    sh = sharpe_ratio(rets, periods=periods_per_year)
    so = sortino_ratio(rets, periods=periods_per_year)
    cal = calmar_ratio(equity, periods=periods_per_year)
    return {
        "strategy": name,
        "n_trades": len(rets),
        "mean_ret": rets.mean(),
        "med_ret": rets.median(),
        "std_ret": rets.std(),
        "sharpe": sh,
        "sortino": so,
        "calmar": cal,
        "max_drawdown": mdd,
        "total_return": equity.iloc[-1] - 1 if len(equity) else np.nan,
    }


def full_report(ov_df: pd.DataFrame, dr_df: pd.DataFrame,
                combined_equity: pd.Series) -> pd.DataFrame:
    """Genera tabella riassuntiva per tutte e tre le configurazioni."""
    stats_ov = strategy_stats(ov_df, name="Overnight ORJ/DriftScore", periods_per_year=52)
    stats_dr = strategy_stats(dr_df, name="Drift 60gg (PEAD)", periods_per_year=252 // 60)
    comb_rets = combined_equity.pct_change().dropna()
    mdd_c, _ = max_drawdown(combined_equity)
    stats_comb = {
        "strategy": "Combined (overnight + drift60)",
        "n_trades": stats_ov["n_trades"] + stats_dr["n_trades"],
        "mean_ret": comb_rets.mean(),
        "med_ret": comb_rets.median(),
        "std_ret": comb_rets.std(),
        "sharpe": sharpe_ratio(comb_rets),
        "sortino": sortino_ratio(comb_rets),
        "calmar": calmar_ratio(combined_equity),
        "max_drawdown": mdd_c,
        "total_return": combined_equity.iloc[-1] - 1,
    }
    return pd.DataFrame([stats_ov, stats_dr, stats_comb])


# ------------------------------------------------------------------
# Grade report
# ------------------------------------------------------------------

def grade_report(signals_df: pd.DataFrame,
                 grade_col: str = "grade_ens",
                 ret_col: str = "drift_60d",
                 it_tickers: list = None) -> pd.DataFrame:
    if it_tickers is None:
        it_tickers = ["ISP.MI", "UCG.MI"]
    df = signals_df.dropna(subset=[grade_col, ret_col]).copy()
    df["bucket"] = np.where(df["ticker"].isin(it_tickers), "IT_banks", "Other_banks")
    return (
        df.groupby([grade_col, "bucket"])[ret_col]
        .agg(n_trades="count", mean_ret="mean", med_ret="median", std_ret="std")
        .reset_index()
    )


# ------------------------------------------------------------------
# Visualizzazioni
# ------------------------------------------------------------------

def plot_equity(ov_df: pd.DataFrame, dr_df: pd.DataFrame,
                combined: pd.Series, save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(11, 5))
    ov_s = ov_df.sort_values("entry_date")
    dr_s = dr_df.sort_values("entry_date")
    ov_s["equity"] = (1 + ov_s["ret_net"]).cumprod()
    dr_s["equity"] = (1 + dr_s["ret_net"]).cumprod()
    ax.plot(ov_s["entry_date"], ov_s["equity"], label="Overnight ORJ/DriftScore", alpha=0.75)
    ax.plot(dr_s["entry_date"], dr_s["equity"], label="Drift 60gg (PEAD)", alpha=0.75)
    ax.plot(combined.index, combined.values, label="Portafoglio Combinato", lw=2)
    ax.set_title("Equity line strategie PEAD - Banche Europee")
    ax.set_ylabel("Valore (base 1)")
    ax.legend()
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_grade_returns(grade_rep: pd.DataFrame, save_path: str | None = None):
    pivot = grade_rep.pivot_table(index="grade_logit" if "grade_logit" in grade_rep.columns else grade_rep.columns[0],
                                   columns="bucket", values="mean_ret")
    pivot.plot(kind="bar", figsize=(8, 4), title="Rendimento medio per grado e bucket (IT vs Other banks)")
    plt.ylabel("Mean return (drift 60gg)")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    print("reporting: usa full_report(), grade_report(), plot_equity() dal notebook.")
