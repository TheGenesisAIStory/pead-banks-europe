"""
src/data_loading.py
====================
Caricamento e download dati: prezzi OHLCV, earnings calendar, fondamentali.

TODO (dati reali):
 - Sostituire generate_earnings_calendar() con CSV reale da:
   MarketScreener, Bloomberg, Refinitiv, FactSet, Yahoo Finance Earnings
   Formato: ticker,earn_date  (es. ISP.MI,2024-02-08)
 - Sostituire load_fundamentals() con dataset reale contenente:
   EPS actual/consensus, CET1 actual/target, NIM, provisioni IFRS9, guidance flag
   Formato: ticker,date,pe,roe,div_yield,quality_score,cet1,cet1_surprise,
            nim,nim_surprise,prov_surprise,guidance_score
"""

from __future__ import annotations
from pathlib import Path
import logging
import yaml
import numpy as np
import pandas as pd
import yfinance as yf

ROOT     = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
RAW_DIR  = ROOT / "data" / "raw"
CFG_PATH = ROOT / "config" / "params.yaml"

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

def load_config() -> dict:
    """Carica params.yaml."""
    with open(CFG_PATH, "r") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------
# Prezzi
# ------------------------------------------------------------------

def download_prices(save: bool = True) -> pd.DataFrame:
    """
    Scarica OHLCV giornaliero per tutti i ticker + benchmark via yfinance.
    Salva in data/processed/prices.parquet.
    """
    cfg       = load_config()
    tickers   = cfg["universe"]["tickers"]
    benchmark = cfg["universe"]["benchmark"]
    start     = cfg["project"]["start_date"]
    end       = cfg["project"]["end_date"]
    all_t     = tickers + [benchmark]

    logger.info("Download prezzi: %s", all_t)
    data = yf.download(all_t, start=start, end=end, auto_adjust=False)

    if save:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        data.to_parquet(DATA_DIR / "prices.parquet")
        logger.info("Salvati in %s", DATA_DIR / "prices.parquet")
    return data


def load_prices() -> pd.DataFrame:
    """Carica prezzi da parquet (scarica se mancante)."""
    p = DATA_DIR / "prices.parquet"
    if not p.exists():
        logger.warning("prices.parquet mancante, scarico ora...")
        return download_prices(save=True)
    return pd.read_parquet(p)


def extract_price_series(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Estrae Close, Open, Volume, AdjClose da DataFrame multi-level yfinance.
    Ritorna (close, open_, volume, adj_close).
    """
    close     = data["Close"]
    open_     = data["Open"]
    volume    = data["Volume"]
    adj_close = data["Adj Close"] if "Adj Close" in data.columns.get_level_values(0) else close
    return close, open_, volume, adj_close


# ------------------------------------------------------------------
# Earnings calendar
# ------------------------------------------------------------------

def generate_earnings_calendar(prices_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Genera calendario earnings SINTETICO (placeholder).

    TODO: sostituire con dati reali.
    CSV atteso in data/raw/earnings_calendar_raw.csv:
        ticker,earn_date
        ISP.MI,2024-02-08
        UCG.MI,2024-02-07
    """
    cfg        = load_config()
    tickers    = cfg["universe"]["tickers"]
    months     = cfg["pead"]["earnings_months"]
    day        = cfg["pead"]["earnings_day"]
    start_year = prices_index.min().year
    end_year   = prices_index.max().year

    rows = []
    for t in tickers:
        for y in range(start_year, end_year + 1):
            for m in months:
                try:
                    d = pd.Timestamp(year=y, month=m, day=day)
                    if prices_index.min() <= d <= prices_index.max():
                        rows.append({"ticker": t, "earn_date": d})
                except ValueError:
                    pass
    return pd.DataFrame(rows).sort_values(["ticker", "earn_date"]).reset_index(drop=True)


def load_earnings_calendar(prices_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Carica calendario earnings:
    - usa data/raw/earnings_calendar_raw.csv se presente (dati REALI),
    - altrimenti genera calendario sintetico (PLACEHOLDER).

    TODO: popolare earnings_calendar_raw.csv con dati reali prima del go-live.
    """
    raw = RAW_DIR / "earnings_calendar_raw.csv"
    if raw.exists():
        df = pd.read_csv(raw, parse_dates=["earn_date"])
        logger.info("Calendario reale: %s (%d righe)", raw, len(df))
        return df.sort_values(["ticker", "earn_date"]).reset_index(drop=True)

    logger.warning(
        "TODO: earnings_calendar_raw.csv non trovato. "
        "Uso PLACEHOLDER sintetico - risultati NON realistici."
    )
    return generate_earnings_calendar(prices_index)


# ------------------------------------------------------------------
# Fondamentali (IFRS9 / CET1 / NIM)
# ------------------------------------------------------------------

def load_fundamentals() -> pd.DataFrame:
    """
    Carica fondamentali per filtro qualita/IFRS9.

    TODO: sostituire con dataset reale.
    CSV atteso in data/raw/fundamentals_raw.csv con colonne:
        ticker, date, pe, roe, div_yield, quality_score,
        cet1, cet1_surprise, nim, nim_surprise, prov_surprise, guidance_score

    I valori placeholder ISP/UCG sono indicativi basati su:
    - Intesa Sanpaolo Annual Report 2023/2024/2025
    - UniCredit Investor Relations 2023/2024/2025
    NON usare in produzione senza sostituire con dati reali verificati.
    """
    raw = RAW_DIR / "fundamentals_raw.csv"
    if raw.exists():
        df = pd.read_csv(raw, parse_dates=["date"])
        logger.info("Fondamentali reali: %s (%d righe)", raw, len(df))
        return df.sort_values(["ticker", "date"]).reset_index(drop=True)

    logger.warning(
        "TODO: fundamentals_raw.csv non trovato. "
        "Uso PLACEHOLDER ISP/UCG 2023-2025 - NON usare in produzione."
    )
    # --- PLACEHOLDER ---
    rows = [
        # ISP.MI - Intesa Sanpaolo
        # CET1 FL ~13.6-13.9%, ROE ~13-15%, div yield ~7.5-9.8%
        # Fonte: Intesa Annual Report / Comunicati stampa risultati
        {"ticker":"ISP.MI","date":pd.Timestamp("2023-12-31"),
         "pe":8.5, "roe":0.130,"div_yield":0.075,"quality_score":0.80,
         "cet1":0.136,"cet1_surprise":0.001,"nim":0.019,"nim_surprise":0.001,
         "prov_surprise":-0.03,"guidance_score":1},
        {"ticker":"ISP.MI","date":pd.Timestamp("2024-12-31"),
         "pe":7.0, "roe":0.140,"div_yield":0.098,"quality_score":0.85,
         "cet1":0.138,"cet1_surprise":0.002,"nim":0.020,"nim_surprise":0.001,
         "prov_surprise":-0.04,"guidance_score":1},
        {"ticker":"ISP.MI","date":pd.Timestamp("2025-12-31"),
         "pe":6.5, "roe":0.150,"div_yield":0.090,"quality_score":0.88,
         "cet1":0.139,"cet1_surprise":0.002,"nim":0.020,"nim_surprise":0.001,
         "prov_surprise":-0.05,"guidance_score":1},
        # UCG.MI - UniCredit
        # CET1 FL ~15.8-16.0%, ROE ~15-16.2%, div yield ~4.5-5.5%
        # Fonte: UniCredit Investor Relations / Comunicati stampa risultati
        {"ticker":"UCG.MI","date":pd.Timestamp("2023-12-31"),
         "pe":7.5, "roe":0.150,"div_yield":0.045,"quality_score":0.80,
         "cet1":0.158,"cet1_surprise":0.002,"nim":0.020,"nim_surprise":0.001,
         "prov_surprise":-0.06,"guidance_score":1},
        {"ticker":"UCG.MI","date":pd.Timestamp("2024-12-31"),
         "pe":6.8, "roe":0.155,"div_yield":0.050,"quality_score":0.85,
         "cet1":0.159,"cet1_surprise":0.003,"nim":0.021,"nim_surprise":0.002,
         "prov_surprise":-0.07,"guidance_score":1},
        {"ticker":"UCG.MI","date":pd.Timestamp("2025-12-31"),
         "pe":6.0, "roe":0.162,"div_yield":0.055,"quality_score":0.90,
         "cet1":0.160,"cet1_surprise":0.003,"nim":0.021,"nim_surprise":0.002,
         "prov_surprise":-0.08,"guidance_score":1},
        # Banche peer - TODO: dati reali
        {"ticker":"SAN.MC","date":pd.Timestamp("2024-12-31"),
         "pe":7.0, "roe":0.120,"div_yield":0.050,"quality_score":0.75,
         "cet1":0.126,"cet1_surprise":0.001,"nim":0.018,"nim_surprise":0.001,
         "prov_surprise":-0.02,"guidance_score":0},
        {"ticker":"BNP.PA","date":pd.Timestamp("2024-12-31"),
         "pe":6.5, "roe":0.100,"div_yield":0.060,"quality_score":0.75,
         "cet1":0.130,"cet1_surprise":0.001,"nim":0.017,"nim_surprise":0.000,
         "prov_surprise":-0.01,"guidance_score":0},
        {"ticker":"DBK.DE","date":pd.Timestamp("2024-12-31"),
         "pe":8.0, "roe":0.085,"div_yield":0.030,"quality_score":0.65,
         "cet1":0.135,"cet1_surprise":0.001,"nim":0.016,"nim_surprise":0.000,
         "prov_surprise": 0.01,"guidance_score":0},
    ]
    return pd.DataFrame(rows).sort_values(["ticker","date"]).reset_index(drop=True)


def get_fundamentals_at(fund_df: pd.DataFrame, ticker: str, date: pd.Timestamp) -> dict | None:
    """Fondamentali piu recenti per ticker alla data."""
    sub = fund_df[(fund_df["ticker"] == ticker) & (fund_df["date"] <= date)]
    if sub.empty:
        return None
    return sub.iloc[-1].to_dict()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    d = download_prices(save=True)
    logger.info("Done. Shape: %s", d.shape)
