# src/make_train_games.py
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

Path("data/processed").mkdir(parents=True, exist_ok=True)

SCHED_PATH = "data/raw/nfl_schedules_2022_2025.csv"
STATS_PATH = "data/raw/nfl_team_stats_2022_2025.csv"

W3 = 3
W8 = 8


def add_rolling_features(stats: pd.DataFrame) -> pd.DataFrame:
    stats = stats.sort_values(["season", "team", "week"]).copy()

    # numeric cols excluding keys
    num_cols = stats.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ["season", "week"]]

    # force numeric columns to float to avoid LossySetitemError in shift()
    for c in num_cols:
        stats[c] = pd.to_numeric(stats[c], errors="coerce").astype("float64")

    g = stats.groupby(["season", "team"], group_keys=False)

    # build new rolling columns
    new_cols = {}
    for c in num_cols:
        x = g[c]
        new_cols[f"w3_{c}"] = x.apply(lambda s: s.shift(1).rolling(W3, min_periods=1).mean())
        new_cols[f"w8_{c}"] = x.apply(lambda s: s.shift(1).rolling(W8, min_periods=1).mean())
        new_cols[f"s_{c}"] = x.apply(lambda s: s.shift(1).expanding(min_periods=1).mean())

    feat = pd.concat([stats[["season", "week", "team"]], pd.DataFrame(new_cols)], axis=1)
    return feat


def moneyline_to_prob(ml: float) -> float:
    """
    Convert American moneyline odds to implied win probability (no vig removal).
    Example: -150 -> 0.60, +130 -> ~0.435
    """
    if ml is None or (isinstance(ml, float) and np.isnan(ml)):
        return 0.0
    try:
        ml = float(ml)
    except Exception:
        return 0.0

    if ml < 0:
        return (-ml) / ((-ml) + 100.0)
    else:
        return 100.0 / (ml + 100.0)


def main():
    sched = pd.read_csv(SCHED_PATH)
    stats = pd.read_csv(STATS_PATH)

    # training games: REG with scores present
    games = sched[
        (sched["game_type"] == "REG")
        & sched["home_score"].notna()
        & sched["away_score"].notna()
    ].copy()

    games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)

    feat = add_rolling_features(stats)

    # merge home rolling features
    g = games.merge(
        feat.rename(columns={"team": "home_team"}),
        on=["season", "week", "home_team"],
        how="left",
    )

    # merge away rolling features
    g = g.merge(
        feat.rename(columns={"team": "away_team"}),
        on=["season", "week", "away_team"],
        how="left",
        suffixes=("_home", "_away"),
    )

    # diff features (home - away) for all windows
    diff_cols = []
    for col in g.columns:
        if col.startswith(("w3_", "w8_", "s_")) and col.endswith("_home"):
            away_col = col.replace("_home", "_away")
            if away_col in g.columns:
                diff = col.replace("_home", "_diff")
                g[diff] = g[col] - g[away_col]
                diff_cols.append(diff)

    # context features
    g["rest_diff"] = g["home_rest"] - g["away_rest"]
    g["div_game"] = g["div_game"].astype(int)
    g["home_field"] = 1

    # spread feature (already in your pipeline)
    g["spread_line_home"] = pd.to_numeric(g.get("spread_line", np.nan), errors="coerce").fillna(0.0)

    # ---- NEW: MONEYLINE / MARKET IMPLIED PROB FEATURES ----
    # Try common column names; if missing, fill with 0 so the pipeline still runs.
    home_ml_col = None
    away_ml_col = None

    for c in ["home_moneyline", "home_ml", "moneyline_home", "home_money_line"]:
        if c in g.columns:
            home_ml_col = c
            break

    for c in ["away_moneyline", "away_ml", "moneyline_away", "away_money_line"]:
        if c in g.columns:
            away_ml_col = c
            break

    if home_ml_col is not None:
        g["home_moneyline"] = pd.to_numeric(g[home_ml_col], errors="coerce")
    else:
        g["home_moneyline"] = np.nan

    if away_ml_col is not None:
        g["away_moneyline"] = pd.to_numeric(g[away_ml_col], errors="coerce")
    else:
        g["away_moneyline"] = np.nan

    g["market_home_prob"] = g["home_moneyline"].apply(moneyline_to_prob).astype("float64")
    g["market_away_prob"] = g["away_moneyline"].apply(moneyline_to_prob).astype("float64")
    g["market_prob_diff"] = (g["market_home_prob"] - g["market_away_prob"]).astype("float64")
    # ------------------------------------------------------

    feature_cols = diff_cols + [
        "rest_diff",
        "div_game",
        "home_field",
        "spread_line_home",
        "market_prob_diff",
    ]

    out = g[["season", "week", "home_team", "away_team", "home_win"] + feature_cols].copy()
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out.to_csv("data/processed/train_games.csv", index=False)

    print("Saved: data/processed/train_games.csv")
    print("Feature columns:", len(feature_cols))
    print("Has spread_line_home:", "spread_line_home" in out.columns)
    print("Has market_prob_diff:", "market_prob_diff" in out.columns)

import pandas as pd

df = pd.read_csv("data/processed/train_games.csv")

print(df["market_prob_diff"].describe())
print("Percent zero:", (df["market_prob_diff"] == 0).mean())

if __name__ == "__main__":
    main()