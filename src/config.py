from pathlib import Path
from dataclasses import dataclass, field

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
LOGS_DIR = OUTPUT_DIR / "logs"

EARNINGS_EVENT_WINDOWS = {
    "EAR_m1_p1": (-1, 1),
    "EAR_0_p1": (0, 1),
    "CAR_p2_p20": (2, 20),
    "CAR_p2_p60": (2, 60),
}

DRIFT_HORIZONS = [5, 20, 60]


@dataclass
class MLConfig:
    target_horizon: int = 60
    target_definition: str = "drift_pos_binary"  # or "top_quantile"
    test_start_year: int = 2019
    min_events_per_fold: int = 30
    embargo_days: int = 60
    models: tuple = ("logit", "rf", "xgb", "lgbm")
    random_state: int = 42
    base_features: list = field(
        default_factory=lambda: [
            "sue",
            "orj",
            "ear_0_1",
            "zvol",
            "cet1_surprise",
            "nim_surprise",
            "prov_surprise",
            "ifrs9_intensity",
            "size",
            "beta",
            "momentum_60d",
            "pre_volatility_60d",
            "liq_turnover",
        ]
    )


ML_CONFIG = MLConfig()
