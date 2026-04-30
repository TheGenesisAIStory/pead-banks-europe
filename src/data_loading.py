"""data_loading.py
Scarica prezzi e fondamentali, salva in data/processed/.
"""
from pathlib import Path
import pandas as pd
import yfinance as yf
import yaml


def load_config(path: str = "config/params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def download_prices(tickers: list, benchmark: str, start: str, end: str) -> pd.DataFrame:
    """Scarica dati OHLCV via yfinance per tickers + benchmark."""
    all_tickers = list(set(tickers + [benchmark]))
    data = yf.download(all_tickers, start=start, end=end, auto_adjust=True)
    return data


def save_prices(data: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(path)
    print(f"Prezzi salvati in {path}")


def load_prices(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_earnings_calendar(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["earn_date"])
    return df.sort_values(["ticker", "earn_date"]).reset_index(drop=True)


def load_fundamentals(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def get_fundamentals_at(fund_df: pd.DataFrame, ticker: str, date: pd.Timestamp) -> dict | None:
    """Ritorna i fondamentali piu' recenti disponibili alla data specificata."""
    sub = fund_df[(fund_df["ticker"] == ticker) & (fund_df["date"] <= date)]
    if sub.empty:
        return None
    return sub.iloc[-1].to_dict()


if __name__ == "__main__":
    cfg = load_config()
    print("Scarico prezzi...")
    data = download_prices(cfg["tickers"], cfg["benchmark"], cfg["start_date"], cfg["end_date"])
    save_prices(data, cfg["paths"]["prices_parquet"])
    print("Done.")
