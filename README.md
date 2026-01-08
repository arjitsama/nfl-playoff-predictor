# NFL Playoff & Super Bowl Win Probability Simulator

A machine learning project that predicts NFL game win probabilities and simulates the NFL playoffs thousands of times to estimate Super Bowl championship odds.

This project focuses on **probabilistic modeling, feature engineering, and Monte Carlo simulation**, rather than single-game prediction accuracy.

---

## Project Overview

- Trained calibrated machine learning models on historical NFL team statistics
- Engineered 280+ rolling statistical features using recent and season-long performance
- Integrated Vegas betting spreads to improve probability calibration
- Simulated full AFC and NFC playoff brackets with reseeding logic
- Estimated Super Bowl win probabilities via 10,000+ Monte Carlo simulations

The goal is to produce **interpretable, well-calibrated probabilities**, not deterministic predictions.

---

## Data Pipeline

Scripts are run directly from the project root (no Python packaging):

~~~bash
python src/make_train_games.py
python src/train_model.py
python src/simulate_playoffs.py
~~~

---

## Feature Engineering

- Rolling windows: last 3 games, last 8 games, season averages
- Home vs away statistical differentials
- Rest differential features
- Vegas betting spread features
- Team-level aggregation for playoff simulation

---

## Models

- Logistic Regression (baseline)
- Random Forest
- Ensemble averaging across models
- Probability calibration (Platt / Isotonic)

Models are trained on historical regular-season games and saved for reuse during playoff simulation.

---

## Playoff Simulation

- Full AFC and NFC playoff brackets
- Wild Card, Divisional, Conference Championship rounds
- Reseeding after the Wild Card round
- Neutral-site Super Bowl simulation
- 10,000 Monte Carlo simulations per run

The simulator outputs team-level probabilities for:
- Conference championships
- Super Bowl wins

---

## Example Output — Super Bowl Win Probabilities

Below is an example output from a 10,000-simulation run:

| Team | Super Bowl Win % |
|------|------------------|
| GB   | 53.49% |
| SF   | 20.70% |
| LAC  | 9.00% |
| LA   | 6.97% |
| PIT  | 3.72% |
| SEA  | 1.87% |
| CAR  | 1.17% |
| BUF  | 1.17% |
| PHI  | 0.71% |
| CHI  | 0.51% |
| DEN  | 0.36% |
| HOU  | 0.20% |
| JAX  | 0.07% |
| NE   | 0.06% |

Probabilities reflect both team strength and playoff structure (seeding, home-field advantage, and matchup paths).

---

## Outputs

Generated artifacts (not tracked in GitHub):

~~~text
outputs/tables/superbowl_odds.csv
outputs/tables/round_odds.csv
outputs/models/win_model.joblib
~~~

These files are reproducible by running the pipeline scripts.

---

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- joblib

---

## Future Improvements

- XGBoost model integration
- Playoff-specific features (seed number, experience proxies)
- Backtesting on historical playoff seasons
- Confidence intervals on simulation outputs

---

## Notes

This project prioritizes **clarity, reproducibility, and statistical rigor** over over-engineering or black-box predictions.
