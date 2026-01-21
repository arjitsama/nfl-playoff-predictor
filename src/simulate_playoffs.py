# src/simulate_playoffs.py
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# ensure src folder is importable when running from project root
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from playoff_bracket import AFC_CHAMPIONSHIP, NFC_CHAMPIONSHIP

SIMS = 10000
RANDOM_SEED = 42
MODEL_PATH = "outputs/models/win_model.joblib"


def load_model(path=MODEL_PATH):
    blob = joblib.load(path)

    # Current format (train_model.py)
    if "models" in blob and "calibrator" in blob and "feature_cols" in blob:
        models = blob["models"]
        calibrator = blob["calibrator"]
        feature_cols = blob["feature_cols"]

        # keep CPU sane if any model uses parallelism
        for m in models.values():
            if hasattr(m, "n_jobs") and m.n_jobs is not None and m.n_jobs != 1:
                m.n_jobs = 1

        return models, calibrator, feature_cols

    raise KeyError(f"Unknown model file format. Keys found: {list(blob.keys())}")


def ensemble_raw_prob(models, X_df: pd.DataFrame) -> np.ndarray:
    ps = [m.predict_proba(X_df)[:, 1] for m in models.values()]
    return np.mean(np.vstack(ps), axis=0)


def calibrated_prob(models, calibrator, X_df: pd.DataFrame) -> np.ndarray:
    raw = ensemble_raw_prob(models, X_df)
    return calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]


def make_feature_row(
    home_team: str,
    away_team: str,
    team_feats: pd.DataFrame,
    feature_cols: list[str],
    home_field_value: float,
) -> pd.DataFrame:
    row = {}

    for c in feature_cols:
        if c.endswith("_diff"):
            base = c[:-5]  # remove _diff
            if base in team_feats.columns:
                row[c] = float(team_feats.loc[home_team, base] - team_feats.loc[away_team, base])
            else:
                row[c] = 0.0

        elif c == "home_field":
            row[c] = float(home_field_value)

        # We don't have playoff-specific rest/division/spread/moneyline for these matchups here,
        # so keep them neutral (0). The team strength is captured by the rolling stats diffs.
        elif c in ("rest_diff", "div_game", "spread_line_home", "market_prob_diff"):
            row[c] = 0.0

        else:
            row[c] = 0.0

    return pd.DataFrame([[row.get(col, 0.0) for col in feature_cols]], columns=feature_cols)


def matchup_home_win_prob(home_team, away_team, team_feats, feature_cols, models, calibrator, home_field_value: float):
    X = make_feature_row(home_team, away_team, team_feats, feature_cols, home_field_value=home_field_value).fillna(0.0)
    return float(calibrated_prob(models, calibrator, X)[0])


def main():
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)

    models, calibrator, feature_cols = load_model(MODEL_PATH)

    # Load end-of-regular-season team features (built by make_team_features.py)
    team_feats = pd.read_csv("data/processed/team_features_end_reg.csv")
    team_feats = team_feats.set_index("team")

    # Only remaining teams
    afc_home, afc_away = AFC_CHAMPIONSHIP
    nfc_home, nfc_away = NFC_CHAMPIONSHIP
    remaining = {afc_home, afc_away, nfc_home, nfc_away}

    missing = [t for t in remaining if t not in team_feats.index]
    if missing:
        raise ValueError(f"Teams missing from team_features_end_reg.csv: {missing}")

    rng = np.random.default_rng(RANDOM_SEED)

    # Single-game probabilities for the two conference championships (home_field=1)
    p_afc_home = matchup_home_win_prob(afc_home, afc_away, team_feats, feature_cols, models, calibrator, home_field_value=1.0)
    p_nfc_home = matchup_home_win_prob(nfc_home, nfc_away, team_feats, feature_cols, models, calibrator, home_field_value=1.0)

    # Print only winner % for those two games
    afc_winner = afc_home if p_afc_home >= 0.5 else afc_away
    afc_win_pct = (p_afc_home if afc_winner == afc_home else (1.0 - p_afc_home)) * 100.0
    print(f"AFC Winner (DEN vs NE): {afc_winner} {afc_win_pct:.2f}%")

    nfc_winner = nfc_home if p_nfc_home >= 0.5 else nfc_away
    nfc_win_pct = (p_nfc_home if nfc_winner == nfc_home else (1.0 - p_nfc_home)) * 100.0
    print(f"NFC Winner (SEA vs LA): {nfc_winner} {nfc_win_pct:.2f}%")

    # Super Bowl is neutral site => home_field=0
    # We'll simulate: sample AFC champ + NFC champ, then play SB with neutral probability
    sb_wins = {t: 0 for t in remaining}

    for i in range(SIMS):
        if i % 500 == 0:
            print(f"sim {i}/{SIMS}")

        # simulate AFC championship
        afc_champ = afc_home if (rng.random() < p_afc_home) else afc_away

        # simulate NFC championship
        nfc_champ = nfc_home if (rng.random() < p_nfc_home) else nfc_away

        # SB neutral probability depends on finalists
        # We treat afc_champ as "home" just for direction consistency, but home_field=0 so it's neutral.
        p_sb_afc = matchup_home_win_prob(afc_champ, nfc_champ, team_feats, feature_cols, models, calibrator, home_field_value=0.0)
        sb_winner = afc_champ if (rng.random() < p_sb_afc) else nfc_champ
        sb_wins[sb_winner] += 1

    # Super Bowl winner (most likely) among only the 4 remaining teams
    sb_probs = {t: sb_wins[t] / SIMS for t in sb_wins}
    top_team = max(sb_probs.items(), key=lambda kv: kv[1])[0]
    top_pct = sb_probs[top_team] * 100.0
    print(f"Super Bowl Winner (most likely): {top_team} {top_pct:.2f}%")

    # Save a small CSV with ONLY the remaining four teams
    sb_df = pd.DataFrame(
        {"team": list(sb_probs.keys()), "superbowl_win_pct": list(sb_probs.values())}
    ).sort_values("superbowl_win_pct", ascending=False)

    sb_df.to_csv("outputs/tables/superbowl_odds.csv", index=False)
    print("Wrote:")
    print(" - outputs/tables/superbowl_odds.csv")


if __name__ == "__main__":
    main()