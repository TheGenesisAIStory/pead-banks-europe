"""models_scoring.py
Logit + Random Forest per stimare PD(drift_60d > 0).
Walk-forward OOS con TimeSeriesSplit.
"""
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, brier_score_loss


def load_config(path: str = "config/params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


FEATURES = ["SUE_z", "ORJ_z", "OFI_z", "Guidance_z", "CET1surp_z", "VOL_z"]


def prepare_dataset(signals_df: pd.DataFrame,
                    ret_col: str = "drift_60d") -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    aligned = signals_df.dropna(subset=FEATURES + [ret_col]).copy()
    aligned["y"] = (aligned[ret_col] > 0).astype(int)
    X = aligned[FEATURES].values
    y = aligned["y"].values
    return X, y, aligned


def walk_forward_eval(X: np.ndarray, y: np.ndarray, cfg: dict) -> dict:
    """Valutazione AUC e Brier score con walk-forward TimeSeriesSplit."""
    tscv = TimeSeriesSplit(n_splits=cfg.get("tscv_splits", 5))
    logit_aucs, rf_aucs = [], []
    logit_briers, rf_briers = [], []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        logit = LogisticRegression(max_iter=1000)
        logit.fit(X_train, y_train)
        p_logit = logit.predict_proba(X_test)[:, 1]
        logit_aucs.append(roc_auc_score(y_test, p_logit))
        logit_briers.append(brier_score_loss(y_test, p_logit))

        rf = RandomForestClassifier(
            n_estimators=cfg.get("rf_n_estimators", 300),
            max_depth=cfg.get("rf_max_depth", 5),
            min_samples_leaf=cfg.get("rf_min_samples_leaf", 50),
            random_state=cfg.get("rf_random_state", 42),
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        p_rf = rf.predict_proba(X_test)[:, 1]
        rf_aucs.append(roc_auc_score(y_test, p_rf))
        rf_briers.append(brier_score_loss(y_test, p_rf))

    return {
        "logit_auc_mean": np.mean(logit_aucs),
        "logit_brier_mean": np.mean(logit_briers),
        "rf_auc_mean": np.mean(rf_aucs),
        "rf_brier_mean": np.mean(rf_briers),
    }


def fit_final_models(X: np.ndarray, y: np.ndarray, cfg: dict):
    """Fit finale su tutto il campione."""
    logit = LogisticRegression(max_iter=1000)
    logit.fit(X, y)

    rf = RandomForestClassifier(
        n_estimators=cfg.get("rf_n_estimators", 300),
        max_depth=cfg.get("rf_max_depth", 5),
        min_samples_leaf=cfg.get("rf_min_samples_leaf", 50),
        random_state=cfg.get("rf_random_state", 42),
        n_jobs=-1,
    )
    rf.fit(X, y)
    return logit, rf


def add_scores(signals_df: pd.DataFrame, logit, rf,
               aligned: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge Score_logit, Score_rf, Score_ens e grade al signals_df."""
    X_full = aligned[FEATURES].fillna(0).values
    scores = aligned.copy()
    scores["Score_logit"] = logit.predict_proba(X_full)[:, 1]
    scores["Score_rf"] = rf.predict_proba(X_full)[:, 1]
    scores["Score_ens"] = 0.5 * scores["Score_logit"] + 0.5 * scores["Score_rf"]
    merged = signals_df.merge(
        scores[["ticker", "earn_date", "Score_logit", "Score_rf", "Score_ens"]],
        on=["ticker", "earn_date"],
        how="left",
    )
    for col in ["Score_logit", "Score_rf", "Score_ens"]:
        merged[f"grade_{col.split('_')[1]}"] = merged[col].apply(grade_label)
    return merged


def grade_label(score: float, high_thr: float = 0.70, low_thr: float = 0.30) -> str:
    """Ritorna 'High', 'Medium' o 'Low' in base allo score."""
    if pd.isna(score):
        return "Unknown"
    if score >= high_thr:
        return "High"
    elif score <= low_thr:
        return "Low"
    return "Medium"


if __name__ == "__main__":
    print("models_scoring: usa walk_forward_eval() e fit_final_models() dal notebook.")
