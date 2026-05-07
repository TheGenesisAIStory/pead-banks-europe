import pandas as pd

from .config import DRIFT_HORIZONS, EARNINGS_EVENT_WINDOWS
from .utils import winsorize_series


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning of daily prices.

    - Remove non-positive prices
    - Flag suspensions (zero or missing volume)
    - Compute preliminary daily returns with simple winsorization
    """

    df = prices.copy()
    df = df[df["close"] > 0]
    df["suspension_flag"] = (df["volume"].fillna(0) == 0).astype(int)
    df = df.sort_values(["bank_id", "date"])
    df["ret_1d"] = df.groupby("bank_id")["close"].pct_change()
    df["ret_1d_w"] = df.groupby("bank_id")["ret_1d"].transform(winsorize_series)
    return df


def validate_events(events: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Validate earnings events and derive basic timing information.

    Notes
    -----
    This function currently assumes all announcements are after-close and
    derives an `ea_date` from the UTC timestamp. More granular rules
    (before-open / intraday / after-close by exchange) should be added
    once detailed timestamp and calendar information is available.
    """

    ev = events.copy()
    ev["ea_timing"] = "after_close"
    ev["ea_date"] = ev["ea_datetime_utc"].dt.date
    return ev
