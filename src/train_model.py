# src/train_model.py
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


def brier(y_true, p):
    y_true = np.asarray(y_true).astype(float)
    p = np.asarray(p).astype(float)
    return float(np.mean((p - y_true) ** 2))


def main():
    os.makedirs("outputs/models", exist_ok=True)

    df = pd.read_csv("data/processed/train_games.csv")

    y = df["home_win"].astype(int)
    drop_cols = {"season", "week", "home_team", "away_team", "home_win"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # time-style split: last season is validation
    last_season = df["season"].max()
    train_idx = df["season"] < last_season
    val_idx = df["season"] == last_season

    # fallback if only 1 season exists
    if train_idx.sum() == 0 or val_idx.sum() == 0:
        split = int(len(df) * 0.8)
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        X_val, y_val = X.iloc[split:], y.iloc[split:]
    else:
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_val, y_val = X.loc[val_idx], y.loc[val_idx]

    # ---- Base models ----
    lr = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False)),
            ("lr", LogisticRegression(max_iter=10000, solver="lbfgs")),
        ]
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    xgb = XGBClassifier(
        n_estimators=700,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=2,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    models = {"lr": lr, "rf": rf, "xgb": xgb}

    for m in models.values():
        m.fit(X_train, y_train)

    def ensemble_prob(X_df: pd.DataFrame) -> np.ndarray:
        ps = [m.predict_proba(X_df)[:, 1] for m in models.values()]
        return np.mean(np.vstack(ps), axis=0)

    # raw ensemble
    p_val_raw = ensemble_prob(X_val)

    # ---- Calibrator (Platt scaling) ----
    # Calibrate raw ensemble probs -> better probabilities
    cal_X_train = ensemble_prob(X_train).reshape(-1, 1)
    cal_X_val = p_val_raw.reshape(-1, 1)

    calibrator = LogisticRegression(max_iter=10000, solver="lbfgs")
    calibrator.fit(cal_X_train, y_train)
    p_val_cal = calibrator.predict_proba(cal_X_val)[:, 1]

    print("VAL raw logloss:", log_loss(y_val, p_val_raw), "brier:", brier(y_val, p_val_raw))
    print("VAL cal logloss:", log_loss(y_val, p_val_cal), "brier:", brier(y_val, p_val_cal))

    blob = {"models": models, "calibrator": calibrator, "feature_cols": feature_cols}
    joblib.dump(blob, "outputs/models/win_model.joblib")
    print("Saved outputs/models/win_model.joblib")


if __name__ == "__main__":
    main()
