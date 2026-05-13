import numpy as np
import pandas as pd


PSI_THRESHOLDS = {"stable": 0.10, "watch": 0.25}


def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    def bucket_pct(arr: np.ndarray) -> np.ndarray:
        counts = np.histogram(arr, bins=breakpoints)[0]
        pct = counts / len(arr)
        return np.where(pct == 0, 1e-4, pct)

    exp_pct = bucket_pct(expected)
    act_pct = bucket_pct(actual)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def psi_status(psi_value: float) -> str:
    if psi_value < PSI_THRESHOLDS["stable"]:
        return "STABLE"
    elif psi_value < PSI_THRESHOLDS["watch"]:
        return "WATCH"
    return "RETRAIN"


def monitor_drift(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    feature_df_ref: pd.DataFrame,
    feature_df_cur: pd.DataFrame,
) -> dict:
    score_psi = compute_psi(reference_scores, current_scores)
    feature_psi = {
        col: compute_psi(feature_df_ref[col].values, feature_df_cur[col].values)
        for col in feature_df_ref.columns
    }
    top_drifted = sorted(feature_psi.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "score_psi": round(score_psi, 4),
        "score_status": psi_status(score_psi),
        "top_drifted_features": top_drifted,
        "recommendation": psi_status(score_psi),
    }
