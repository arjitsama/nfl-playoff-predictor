# src/simulate_playoffs.py
import sys
from pathlib import Path
import pandas as pd

# ensure src folder is importable when running from project root
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import joblib
from playoff_bracket import AFC, NFC

SIMS = 10000 #simulations
SEASON = 2025
RANDOM_SEED = 42

RAW_SCHED = "data/raw/nfl_schedules_2022_2025.csv"
RAW_STATS = "data/raw/nfl_team_stats_2022_2025.csv"
MODEL_PATH = "outputs/models/win_model.joblib"


def load_model():
    blob = joblib.load(MODEL_PATH)

    # Format A
    if "base_models" in blob:
        models = blob["base_models"]
        calibrator = blob["calibrator"]
        feature_cols = blob["feature_cols"]
    # Format B
    elif "lr" in blob and "rf" in blob:
        models = {"lr": blob["lr"], "rf": blob["rf"]}
        calibrator = blob["calibrator"]
        feature_cols = blob["feature_cols"]
    else:
        raise KeyError(f"Unknown model file format. Keys found: {list(blob.keys())}")

    for m in models.values():
        if hasattr(m, "n_jobs") and m.n_jobs is not None and m.n_jobs != 1:
            m.n_jobs = 1

    return models, calibrator, feature_cols


def ensemble_raw_prob(models, X_df: pd.DataFrame) -> np.ndarray:
    ps = []
    for m in models.values():
        ps.append(m.predict_proba(X_df)[:, 1])
    return np.mean(np.vstack(ps), axis=0)


def calibrated_prob(models, calibrator, X_df: pd.DataFrame) -> np.ndarray:
    raw = ensemble_raw_prob(models, X_df)
    return calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]


def build_end_of_regular_season_features(season=SEASON) -> pd.DataFrame:
    sched = pd.read_csv(RAW_SCHED)
    stats = pd.read_csv(RAW_STATS)

    reg = sched[(sched["season"] == season) & (sched["game_type"] == "REG")].copy()
    last_week = int(reg["week"].max())

    s = stats[(stats["season"] == season) & (stats["week"] <= last_week)].copy()

    num_cols = s.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ["season", "week"]]

    team_avg = s.groupby("team")[num_cols].mean()
    team_avg = team_avg.rename(columns={c: f"roll_{c}" for c in num_cols})
    return team_avg


def make_feature_row(home_team: str, away_team: str, team_feats: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    row = {}
    for c in feature_cols:
        if c.endswith("_diff"):
            base = c[:-5]  # remove _diff
            if base in team_feats.columns:
                row[c] = float(team_feats.loc[home_team, base] - team_feats.loc[away_team, base])
            else:
                row[c] = 0.0
        elif c == "home_field":
            row[c] = 1.0
        elif c in ("rest_diff", "div_game"):
            row[c] = 0.0
        else:
            row[c] = 0.0

    return pd.DataFrame([[row.get(col, 0.0) for col in feature_cols]], columns=feature_cols)


def precompute_matchup_probs(bracket_teams, models, calibrator, team_feats, feature_cols):
    """
    Precompute P(home wins) for every ordered pair (home, away) in bracket teams.
    """
    probs = {}
    teams = sorted(bracket_teams)
    for h in teams:
        for a in teams:
            if h == a:
                continue
            X = make_feature_row(h, a, team_feats, feature_cols).fillna(0.0)
            p_home = float(calibrated_prob(models, calibrator, X)[0])
            probs[(h, a)] = p_home
    return probs


def play_game_fast(home_team, away_team, probs, rng):
    p_home = probs[(home_team, away_team)]
    return home_team if (rng.random() < p_home) else away_team


def simulate_conference(conf_def, probs, rng):
    seed_to_team = conf_def["seed_to_team"]

    # Wild Card winners (store seeds)
    wc_winner_seeds = []
    for hi, lo in conf_def["wildcard"]:
        home = seed_to_team[hi]
        away = seed_to_team[lo]
        winner_team = play_game_fast(home, away, probs, rng)
        wc_winner_seeds.append(hi if winner_team == home else lo)

    # reseed
    lowest = max(wc_winner_seeds)

    # Divisional: 1 vs lowest
    div1_home_team = seed_to_team[1]
    div1_away_team = seed_to_team[lowest]
    div1_winner_team = play_game_fast(div1_home_team, div1_away_team, probs, rng)
    div1_winner_seed = 1 if div1_winner_team == div1_home_team else lowest

    # Other divisional
    other = sorted([s for s in wc_winner_seeds if s != lowest])
    sA, sB = other[0], other[1]
    div2_home_seed = min(sA, sB)
    div2_away_seed = max(sA, sB)

    div2_home_team = seed_to_team[div2_home_seed]
    div2_away_team = seed_to_team[div2_away_seed]
    div2_winner_team = play_game_fast(div2_home_team, div2_away_team, probs, rng)
    div2_winner_seed = div2_home_seed if div2_winner_team == div2_home_team else div2_away_seed

    # Conference championship: higher seed hosts
    champ_home_seed = min(div1_winner_seed, div2_winner_seed)
    champ_away_seed = max(div1_winner_seed, div2_winner_seed)

    champ_home_team = seed_to_team[champ_home_seed]
    champ_away_team = seed_to_team[champ_away_seed]
    return play_game_fast(champ_home_team, champ_away_team, probs, rng)


def main():
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)

    models, calibrator, feature_cols = load_model()
    team_feats = build_end_of_regular_season_features(SEASON)

    bracket_teams = set(AFC["seed_to_team"].values()) | set(NFC["seed_to_team"].values())
    missing = [t for t in bracket_teams if t not in team_feats.index]
    if missing:
        raise ValueError(f"Bracket teams missing from team stats: {missing}")

    rng = np.random.default_rng(RANDOM_SEED)

    # compute all matchup probs once
    probs = precompute_matchup_probs(bracket_teams, models, calibrator, team_feats, feature_cols)

    sb_wins = {t: 0 for t in bracket_teams}
    afc_titles = {t: 0 for t in AFC["seed_to_team"].values()}
    nfc_titles = {t: 0 for t in NFC["seed_to_team"].values()}

    for i in range(SIMS):
        if i % 500 == 0:
            print(f"sim {i}/{SIMS}")

        afc_champ = simulate_conference(AFC, probs, rng)
        nfc_champ = simulate_conference(NFC, probs, rng)

        afc_titles[afc_champ] += 1
        nfc_titles[nfc_champ] += 1

        # Super Bowl (neutral site; still treat AFC as "home" for direction consistency)
        winner = play_game_fast(afc_champ, nfc_champ, probs, rng)
        sb_wins[winner] += 1

    sb_df = pd.DataFrame({
        "team": list(sb_wins.keys()),
        "superbowl_win_pct": [sb_wins[t] / SIMS for t in sb_wins]
    }).sort_values("superbowl_win_pct", ascending=False)

    # --- PRINT RESULTS (PERCENTAGES) ---
    display_df = sb_df.copy()
    display_df["superbowl_win_pct"] = (display_df["superbowl_win_pct"] * 100).round(2)
    print("\nSuper Bowl Win Probabilities:")
    print(display_df.to_string(index=False))
    # --- END PRINT ---

    round_df = pd.DataFrame({
        "team": list(sb_wins.keys()),
        "conference_win_pct": [
            (afc_titles.get(t, 0) + nfc_titles.get(t, 0)) / SIMS for t in sb_wins
        ],
        "superbowl_win_pct": [sb_wins[t] / SIMS for t in sb_wins]
    }).sort_values("superbowl_win_pct", ascending=False)

    sb_df.to_csv("outputs/tables/superbowl_odds.csv", index=False)
    round_df.to_csv("outputs/tables/round_odds.csv", index=False)

    print("Wrote:")
    print(" - outputs/tables/superbowl_odds.csv")
    print(" - outputs/tables/round_odds.csv")

def load_model(path="outputs/models/win_model.joblib"):
    import joblib

    blob = joblib.load(path)

    # New format (current train_model.py)
    if "models" in blob and "calibrator" in blob and "feature_cols" in blob:
        models = blob["models"]
        calibrator = blob["calibrator"]
        feature_cols = blob["feature_cols"]
        return models, calibrator, feature_cols

    # Old format fallback (if you ever load an older file)
    if "lr" in blob or "rf" in blob:
        models = {}
        if "lr" in blob:
            models["lr"] = blob["lr"]
        if "rf" in blob:
            models["rf"] = blob["rf"]
        if "xgb" in blob:
            models["xgb"] = blob["xgb"]

        calibrator = blob.get("calibrator", None)
        feature_cols = blob.get("feature_cols", [])
        return models, calibrator, feature_cols

    raise KeyError(f"Unknown model file format. Keys found: {list(blob.keys())}")


if __name__ == "__main__":
    main()