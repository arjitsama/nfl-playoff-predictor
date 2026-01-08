import pandas as pd
import numpy as np
from pathlib import Path

Path("data/processed").mkdir(parents=True, exist_ok=True)

STATS_PATH = "data/raw/nfl_team_stats_2022_2025.csv"
SEASON = 2025  # change if needed


def main():
    stats = pd.read_csv(STATS_PATH)
    stats = stats[(stats["season"] == SEASON) & (stats["season_type"] == "REG")].copy()
    stats = stats.sort_values(["team", "week"])

    # numeric columns only
    num_cols = stats.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ["season", "week"]]
    for c in num_cols:
        stats[c] = pd.to_numeric(stats[c], errors="coerce").astype("float64")

    def roll_feats(g):
        g = g.sort_values("week").copy()
        last = g.iloc[-1]

        out = {"team": last["team"], "season": int(last["season"]), "week": int(last["week"])}

        for c in num_cols:
            s = g[c]
            out[f"w3_{c}"] = float(s.shift(1).rolling(3, min_periods=1).mean().iloc[-1])
            out[f"w8_{c}"] = float(s.shift(1).rolling(8, min_periods=1).mean().iloc[-1])
            out[f"s_{c}"] = float(s.shift(1).expanding(min_periods=1).mean().iloc[-1])

        return pd.Series(out)

    team_feats = stats.groupby("team", group_keys=False).apply(roll_feats).reset_index(drop=True)
    team_feats = team_feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out_path = "data/processed/team_features_end_reg.csv"
    team_feats.to_csv(out_path, index=False)
    print("Saved:", out_path)
    print(team_feats.head())


if __name__ == "__main__":
    main()