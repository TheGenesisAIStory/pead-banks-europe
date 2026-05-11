"""
src/reporting.py
All publication-grade plots and table exports.
Designed to be called from the Colab notebook.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve
from typing import Dict, List, Optional

PLT_STYLE = {
    "figure.facecolor":  "#0e1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#c9d1d9",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#c9d1d9",
    "grid.color":        "#21262d",
    "grid.linestyle":    "--",
    "legend.facecolor":  "#161b22",
    "legend.edgecolor":  "#30363d",
}
PALETTE = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff",
           "#ffa657", "#79c0ff", "#56d364", "#ff7b72"]


def _apply_style():
    plt.rcParams.update(PLT_STYLE)


def plot_universe_coverage(panel: pd.DataFrame, events: pd.DataFrame,
                            save_path: Optional[str] = None):
    """Bar chart: tickers with prices vs tickers with events."""
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: observations per ticker
    obs = panel.groupby("ticker").size().sort_values(ascending=False)
    axes[0].barh(obs.index, obs.values, color=PALETTE[0], alpha=0.85)
    axes[0].set_title("Trading Days per Bank", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Number of trading days")
    axes[0].invert_yaxis()

    # Right: events per ticker
    evobs = events.groupby("ticker").size().sort_values(ascending=False)
    axes[1].barh(evobs.index, evobs.values, color=PALETTE[1], alpha=0.85)
    axes[1].set_title("Earnings Events per Bank", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Number of events")
    axes[1].invert_yaxis()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_events_timeline(events: pd.DataFrame, save_path: Optional[str] = None):
    """Events per quarter timeline."""
    _apply_style()
    ev = events.copy()
    ev["ym"] = pd.to_datetime(ev["event_date"]).dt.to_period("Q").astype(str)
    cnt = ev.groupby("ym").size()
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.bar(range(len(cnt)), cnt.values, color=PALETTE[2], alpha=0.85)
    ax.set_xticks(range(0, len(cnt), 4))
    ax.set_xticklabels(cnt.index[::4], rotation=45, ha="right", fontsize=9)
    ax.set_title("Earnings Events per Quarter", fontsize=13, fontweight="bold")
    ax.set_ylabel("N events")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_drift_distributions(ef: pd.DataFrame, save_path: Optional[str] = None):
    """Distribution of drift_5d, drift_20d, drift_60d."""
    _apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, h in enumerate([5, 20, 60]):
        col = f"drift_{h}d"
        if col not in ef.columns:
            continue
        d = ef[col].dropna()
        axes[i].hist(d, bins=40, color=PALETTE[i], alpha=0.8, edgecolor="none")
        axes[i].axvline(0, color="white", linestyle="--", linewidth=1)
        axes[i].axvline(d.mean(), color=PALETTE[4], linestyle="-", linewidth=1.5,
                        label=f"mean={d.mean():.3f}")
        axes[i].set_title(f"Drift {h}d distribution", fontweight="bold")
        axes[i].set_xlabel("Abnormal return")
        axes[i].legend(fontsize=9)
    plt.suptitle("Post-Earnings Drift Distributions", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_car_by_quantile(ef: pd.DataFrame, signal_col: str = "composite_score",
                         drift_col: str = "drift_60d", n_q: int = 5,
                         save_path: Optional[str] = None):
    """Mean drift per quantile of signal — event study style."""
    _apply_style()
    df = ef.dropna(subset=[signal_col, drift_col]).copy()
    df["q"] = pd.qcut(df[signal_col], n_q, labels=[f"Q{i+1}" for i in range(n_q)],
                       duplicates="drop")
    stats = df.groupby("q")[drift_col].agg(["mean", "sem"])
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [PALETTE[0] if i < n_q-1 else PALETTE[1] for i in range(len(stats))]
    bars = ax.bar(stats.index, stats["mean"], yerr=1.96*stats["sem"],
                  capsize=4, color=colors, alpha=0.85, edgecolor="none")
    ax.axhline(0, color="white", linestyle="--", linewidth=1)
    ax.set_title(f"Mean {drift_col} by {signal_col} Quintile",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Quintile")
    ax.set_ylabel("Mean abnormal return")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_ic_barh(ic_df: pd.DataFrame, save_path: Optional[str] = None):
    """Horizontal bar chart of IC (Pearson corr with drift_60d) per feature."""
    _apply_style()
    df = ic_df.sort_values("ic", ascending=True)
    colors = [PALETTE[1] if v > 0 else PALETTE[2] for v in df["ic"]]
    fig, ax = plt.subplots(figsize=(10, max(6, len(df)*0.35)))
    ax.barh(df["feature"], df["ic"], color=colors, alpha=0.85, edgecolor="none")
    ax.axvline(0, color="white", linestyle="--", linewidth=1)
    ax.set_title("Information Coefficient (IC) — Pearson corr with drift_60d",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("IC")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_roc_curves(oos: pd.DataFrame, model_names: List[str],
                   save_path: Optional[str] = None):
    """ROC curves for all models."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, mn in enumerate(model_names):
        col = f"p_{mn}"
        if col not in oos.columns:
            continue
        sub = oos.dropna(subset=[col, "y_true"])
        if len(sub) < 10:
            continue
        fpr, tpr, _ = roc_curve(sub["y_true"], sub[col])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=PALETTE[i],
                label=f"{mn.upper()} (AUC={roc_auc:.3f})", linewidth=2)
    ax.plot([0,1],[0,1],"--",color="#8b949e",linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — OOS Walk-Forward", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_calibration(oos: pd.DataFrame, model_names: List[str],
                     save_path: Optional[str] = None):
    """Calibration reliability diagrams."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, mn in enumerate(model_names):
        col = f"p_{mn}"
        if col not in oos.columns:
            continue
        sub = oos.dropna(subset=[col, "y_true"])
        if len(sub) < 10:
            continue
        frac_pos, mean_pred = calibration_curve(sub["y_true"], sub[col], n_bins=8)
        ax.plot(mean_pred, frac_pos, "s-", color=PALETTE[i],
                label=mn.upper(), linewidth=2)
    ax.plot([0,1],[0,1],"--",color="#8b949e",linewidth=1,label="Perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration Curves", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_equity_curve(port_rets: pd.DataFrame,
                      benchmark_rets: Optional[pd.Series] = None,
                      save_path: Optional[str] = None):
    """Equity curve + drawdown panel."""
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={"height_ratios": [3, 1]})

    for col, label, color in [
        ("long_ret",  "Long-only Q5",  PALETTE[1]),
        ("ls_ret",    "Long-Short Q5-Q1", PALETTE[0]),
    ]:
        if col not in port_rets.columns:
            continue
        r = port_rets.set_index("date")[col].fillna(0)
        cum = (1 + r).cumprod()
        ax1.plot(cum.index, cum.values, label=label, color=color, linewidth=1.8)
        dd = (cum / cum.cummax() - 1)
        ax2.fill_between(dd.index, dd.values, 0, alpha=0.4, color=color)

    if benchmark_rets is not None:
        bm = (1 + benchmark_rets.fillna(0)).cumprod()
        ax1.plot(bm.index, bm.values, "--", color="#8b949e",
                 linewidth=1.2, label="Benchmark")

    ax1.set_title("Portfolio Equity Curve (1 EUR invested)",
                  fontsize=13, fontweight="bold")
    ax1.set_ylabel("Cumulative return")
    ax1.legend(fontsize=10)
    ax1.axhline(1, color="#8b949e", linestyle=":", linewidth=0.8)
    ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_ablation(ablation_df: pd.DataFrame, save_path: Optional[str] = None):
    """Horizontal bar chart of OOS AUC by feature block set."""
    _apply_style()
    df = ablation_df.sort_values("mean_auc")
    fig, ax = plt.subplots(figsize=(10, max(5, len(df)*0.55)))
    ax.barh(df["block_set"], df["mean_auc"],
            color=PALETTE[0], alpha=0.85, edgecolo