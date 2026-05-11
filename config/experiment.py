"""
config/experiment.py
Central configuration for the PEAD Banks Europe experiment.
All parameters are controllable from here — never hardcode in notebooks.
"""

from dataclasses import dataclass, field
from typing import List, Dict

# ── Universe ───────────────────────────────────────────────────────────────────
UNIVERSE: Dict[str, str] = {
    # ticker_yfinance : display_name
    "ISP.MI":   "Intesa Sanpaolo",
    "UCG.MI":   "UniCredit",
    "BNPP.PA":  "BNP Paribas",
    "SAN.PA":   "Société Générale",
    "DBK.DE":   "Deutsche Bank",
    "CBK.DE":   "Commerzbank",
    "BBVA.MC":  "BBVA",
    "SAN.MC":   "Banco Santander",
    "ABI.BR":   "ABN AMRO",
    "ING.AS":   "ING Groep",
    "BARC.L":   "Barclays",
    "LLOY.L":   "Lloyds Banking Group",
    "NDA-SE.ST":"Nordea",
    "CABK.MC":  "CaixaBank",
    "BNM.MC":   "Bankinter",
    "BAMI.MI":  "Banco BPM",
    "MB.MI":    "Mediobanca",
}

MARKET_INDEX: str = "SX7E"       # STOXX Europe 600 Banks proxy (via ^STOXX...)
MARKET_PROXY: str = "^STOXX50E"  # EURO STOXX 50 as broad market

# ── Time range ─────────────────────────────────────────────────────────────────
START_DATE: str = "2016-01-01"
END_DATE:   str = "2025-12-31"

# ── Earnings events ────────────────────────────────────────────────────────────
# Approximate annual earnings dates (month, day) per ticker — used for synthetic
# event generation when real timestamps unavailable.
# Format: (month, day) for Q4/FY results (typically Feb), Q2 (Aug), Q1 (May), Q3 (Nov)
EARNINGS_APPROX: Dict[str, list] = {
    "ISP.MI":  [(2,7),(5,10),(8,6),(11,8)],
    "UCG.MI":  [(2,1),(5,3),(8,1),(11,7)],
    "BNPP.PA": [(2,6),(5,4),(8,1),(11,3)],
    "SAN.PA":  [(2,8),(5,5),(8,2),(11,4)],
    "DBK.DE":  [(1,27),(4,25),(7,24),(10,25)],
    "CBK.DE":  [(2,14),(5,9),(8,7),(11,9)],
    "BBVA.MC": [(2,1),(5,3),(7,28),(10,27)],
    "SAN.MC":  [(2,2),(4,30),(7,31),(10,29)],
    "ABI.BR":  [(2,12),(5,15),(8,14),(11,13)],
    "ING.AS":  [(2,1),(5,3),(8,2),(11,7)],
    "BARC.L":  [(2,22),(5,2),(8,1),(10,31)],
    "LLOY.L":  [(2,22),(5,1),(7,31),(10,30)],
    "NDA-SE.ST":[(1,25),(4,24),(7,18),(10,22)],
    "CABK.MC": [(2,2),(5,5),(8,5),(11,5)],
    "BNM.MC":  [(2,3),(4,27),(7,26),(10,25)],
    "BAMI.MI": [(2,9),(5,8),(8,8),(11,8)],
    "MB.MI":   [(1,27),(4,28),(7,27),(10,27)],
}

# ── Feature engineering ────────────────────────────────────────────────────────
DRIFT_HORIZONS: List[int] = [5, 20, 60]
EVENT_WINDOWS: Dict[str, tuple] = {
    "ear_m1_p1": (-1, 1),
    "ear_0_p1":  (0, 1),
    "car_p2_p20":( 2,20),
    "car_p2_p60":( 2,60),
}
PRE_EVENT_WINDOW: int = 60          # days for vol/volume pre-event baseline
MOM_WINDOWS: List[int] = [5, 20, 60, 120]  # momentum lookbacks
VOL_WINDOWS: List[int] = [20, 60]   # realized volatility lookbacks

# ── ML ────────────────────────────────────────────────────────────────────────
@dataclass
class MLConfig:
    target_horizon: int = 60
    target_scheme: str = "binary_pos"       # binary_pos | top_quantile
    top_q_threshold: float = 0.8
    test_start_year: int = 2020
    embargo_days: int = 60
    models: tuple = ("logit", "rf", "xgb", "lgbm")
    random_state: int = 42
    feature_blocks: List[str] = field(default_factory=lambda: [
        "controls", "pead", "price_momentum", "volatility",
        "liquidity", "style_risk", "macro_rates", "bank_fundamentals",
    ])

ML = MLConfig()

# ── Backtest ──────────────────────────────────────────────────────────────────
COST_BPS: float   = 15.0     # round-trip cost in basis points
SLIPPAGE_BPS: float = 5.0
BORROW_BPS: float  = 80.0    # annual borrow cost (short leg)
N_QUANTILES: int   = 5
HOLD_HORIZON: int  = 60      # default holding period in calendar days
