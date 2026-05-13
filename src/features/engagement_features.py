import numpy as np
import pandas as pd


def add_rfm_features(df: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    df = df.copy()
    rfm = df.groupby("customer_id").agg(
        recency=("event_date", lambda x: (snapshot_date - x.max()).days),
        frequency=("event_id", "nunique"),
        monetary=("revenue", "sum"),
    )
    return df.merge(rfm, on="customer_id", how="left")


def add_engagement_decay(df: pd.DataFrame, halflife_days: int = 30) -> pd.DataFrame:
    """Exponentially weighted activity score — captures disengagement before churn."""
    df = df.copy()
    alpha = 1 - np.exp(-np.log(2) / halflife_days)
    df["engagement_score"] = (
        df.groupby("customer_id")["activity_flag"]
        .transform(lambda x: x.ewm(alpha=alpha, adjust=False).mean())
    )
    return df


def add_delinquency_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for days in [30, 60, 90]:
        df[f"ever_delinquent_{days}d"] = (
            df.groupby("customer_id")["days_past_due"]
            .transform(lambda x: (x >= days).rolling(6, min_periods=1).max())
        )
    df["dpd_trend"] = (
        df.groupby("customer_id")["days_past_due"]
        .transform(lambda x: x.diff().rolling(3, min_periods=1).mean())
    )
    return df


def add_product_penetration(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["product_depth"] = df.groupby("customer_id")["product_id"].transform("nunique")
    df["cross_sell_ratio"] = df["product_depth"] / df["total_products_available"]
    return df


def build_churn_features(df: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    df = add_rfm_features(df, snapshot_date)
    df = add_engagement_decay(df)
    df = add_delinquency_trajectory(df)
    df = add_product_penetration(df)
    df.dropna(inplace=True)
    return df
