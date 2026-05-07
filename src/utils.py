import logging
from pathlib import Path

import pandas as pd

from .config import TABLES_DIR, FIGURES_DIR, LOGS_DIR


def setup_logging(name: str) -> logging.Logger:
    """Configure a basic file logger for the research pipeline."""

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(LOGS_DIR / f"{name}.log")
        fh.setLevel(logging.INFO)
        fmt = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def save_table(df: pd.DataFrame, name: str) -> None:
    """Save a pandas DataFrame as CSV in reports/tables.

    Parameters
    ----------
    df : pd.DataFrame
        Table to be saved.
    name : str
        Base file name without extension (e.g. "Table_I_descriptive").
    """

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / f"{name}.csv"
    df.to_csv(path, index=False)


def save_figure(fig, name: str, fmt: str = "pdf") -> None:
    """Save a Matplotlib figure to reports/figures.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    name : str
        Base file name without extension.
    fmt : str, default "pdf"
        File format ("pdf", "png", ...).
    """

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.{fmt}"
    fig.savefig(path, bbox_inches="tight")


def winsorize_series(s: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    """Apply simple quantile winsorization to a pandas Series."""

    lower = s.quantile(lower_q)
    upper = s.quantile(upper_q)
    return s.clip(lower, upper)
