"""
src/models_scoring.py
ML walk-forward training: Logit, RF, XGBoost, LightGBM.
Ablation by feature block.
SHAP / permutation importance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Feature block definitions (subset of features_pead.py output columns)
FEATURE_BLOCKS: Dict[str, List[str]] = {
    "controls": [
        "log_size", "beta_60d", "pre_vol_20d", "pre_vol_60d",
    ],
    "pead": [
        "sue_proxy", "orj", "ear_0_1", "zvol", "vol_spike",
    ],
    "price_momentum": [
        "mom_5d", "mom_20d", "mom_60d", "mom_120d",
        "reversal_5d", "dist_52w_high",
    ],
    "volatility": [
        "rv_20d", "rv_60d", "parkinson_20d", "downside_vol_20d", "vol_ratio",
    ],
    "liquidity": [
        "amihud_20d", "turnover_20d", "zero_vol_frac_20d", "hlspread_20d",
    ],
    "style_risk": [
        "market_corr_60d", "residual_mom_60d", "sector_rel_ret_20d",
    ],
    "macro_rates": [
        "macro_mom_20d", "macro_vol_20d",
    ],
    "bank_fundamentals": [
        "cet1_surprise_proxy", "nim_surprise_proxy", "ifrs9_intensity_proxy",
    ],
}


def get_features_for_blocks(blocks: List[str]) -> List[str]:
    feats = []
    for b in blocks:
        feats.extend(FEATURE_BLOCKS.get(b, []))
    return list(dict.fromkeys(feats))  # deduplicate preserving order


def _build_model(name: str, seed: int = 42):
    if name == "logit":
        return LogisticRegression(C=1.0, max_iter=500, random_state=seed)
    elif name == "rf":
        return RandomForestClassifier(n_estimators=200, max_depth=4,
                                      min_samples_leaf=5, random_state=seed)
    elif name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             eval_metric="logloss", verbosity=0,
                             random_state=seed)
    elif name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                              num_leaves=15, subsample=0.8,
                              verbose=-1, random_state=seed)
    else:
        raise ValueError(f"Unknown model: {name}")


def walk_forward(
    event_features: pd.DataFrame,
    target_col: str = "y_60d",
    feature_blocks: Optional[List[str]] = None,
    model_names: Tuple[str, ...] = ("logit", "rf", "xgb", "lgbm"),
    test_start_year: int = 2020,
    embargo_days: int = 60,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Walk-forward experiment.
    Returns:
      - oos_preds: DataFrame with event_id, ticker, event_date, y_true, p_hat per model
      - perf_summary: dict {model: {year: {auc, brier, acc}}}
    """
    if feature_blocks is None:
        feature_blocks = list(FEATURE_BLOCKS.keys())

    feat_cols = get_features_for_blocks(feature_blocks)
    df = event_features.dropna(subset=[target_col]).copy()
    df["event_date"] = pd.to_datetime(df["event_date"])
    df = df.sort_values("event_date").reset_index(drop=True)

    years = sorted(df["event_date"].dt.year.unique())
    test_years = [y for y in years if y >= test_start_year]

    all_preds = []
    perf: Dict = {m: {} for m in model_names}

    for ty in test_years:
        cutoff = pd.Timestamp(year=ty, month=1, day=1) - pd.Timedelta(days=embargo_days)
        train = df[df["event_date"] < cutoff]
        test  = df[df["event_date"].dt.year == ty]
        if len(train) < 30 or len(test) < 5:
            continue

        avail = [c for c in feat_cols if c in df.columns]
        X_tr = train[avail].fillna(0).values
        y_tr = train[target_col].values
        X_te = test[avail].fillna(0).values
        y_te = test[target_col].values

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        row_base = test[["event_id", "ticker", "event_date", target_col]].copy()
        row_base["y_true"] = y_te

        for mn in model_names:
            model = _build_model(mn, seed)
            try:
                model.fit(X_tr, y_tr)
                p = model.predict_proba(X_te)[:, 1]
                row_base[f"p_{mn}"] = p
                if len(np.unique(y_te)) > 1:
                    auc   = roc_auc_score(y_te, p)
                    brier = brier_score_loss(y_te, p)
                    acc   = ((p >= 0.5) == y_te).mean()
                    perf[mn][ty] = {"auc": auc, "brier": brier, "acc": acc}
            except Exception as e:
                row_base[f"p_{mn}"] = np.nan
        all_preds.append(row_base)

    if all_preds:
        oos = pd.concat(all_preds, ignore_index=True)
    else:
        oos = pd.DataFrame()

    return oos, perf


def ablation_by_block(
    event_features: pd.DataFrame,
    target_col: str = "y_60d",
    model_name: str = "xgb",
    test_start_year: int = 2020,
    embargo_days: int = 60,
) -> pd.DataFrame:
    """
    Run walk-forward for each block combination:
    1) controls only  2) controls + pead  3) cumulative blocks  4) all blocks
    Returns DataFrame with AUC per block set.
    """
    block_sets = {
        "controls": ["controls"],
        "controls+pead": ["controls", "pead"],
        "controls+pead+mom": ["controls", "pead", "price_momentum"],
        "controls+pead+mom+vol": ["controls", "pead", "price_momentum", "volatility"],
        "controls+pead+mom+vol+liq": ["controls", "pead", "price_momentum", "volatility", "liquidity"],
        "controls+pead+style": ["controls", "pead", "price_momentum", "volatility", "liquidity", "style_risk"],
        "controls+pead+macro": ["controls", "pead", "price_momentum", "volatility", "liquidity", "style_risk", "macro_rates"],
        "all_blocks":  list(FEATURE_BLOCKS.keys()),
    }
    results = []
    for label, blocks in block_sets.items():
        _, perf = walk_forward(
            event_features, target_col=target_col,
            feature_blocks=blocks, model_names=(model_name,),
            test_start_year=test_start_year, embargo_days=embargo_days
        )
        aucs = [v["auc"] for v in perf.get(model_name, {}).values() if "auc" in v]
        results.append({
            "block_set": label,
            "n_blocks": len(blocks),
            "mean_auc": np.mean(aucs) if aucs else np.nan,
            "aucs": aucs,
        })
    return pd.DataFrame(results)


def compute_permutation_importance(
    event_features: pd.DataFrame,
    target_col: str = "y_60d",
    model_name: str = "rf",
    n_repeats: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Train on all data (no walk-forward) and compute permutation importance.
    Use for interpretability only — not for OOS evaluation.
    """
    feat_cols = get_features_for_blocks(list(FEATURE_BLOCKS.keys()))
    df = event_features.dropna(subset=[target_col]).copy()
    avail = [c for c in feat_cols if c in df.columns]
    X = StandardScaler().fit_transform(df[avail].fillna(0).values)
    y = df[target_col].values
    model = _build_model(model_name, seed)
    model.fit(X, y)
    result = permutation_importance(model, X, y,
                                    n_repeats=n_repeats,
                                    random_state=seed,
                                    scoring="roc_auc")
    imp = pd.DataFrame({
        "feature": avail,
        "importance_mean": result.importances_mean,
        "importance_std":  result.importances_std,
    }).sort_values("importance_mean", ascending=False)
    return imp
