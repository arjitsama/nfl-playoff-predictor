# NFL Playoff & Super Bowl Win Probability Simulator

A machine learning project that predicts NFL game win probabilities and simulates the final stage of the NFL playoffs to estimate Super Bowl championship odds.

This project emphasizes **probabilistic modeling, feature engineering, and Monte Carlo simulation**, rather than deterministic single-game picks.

---

## Project Overview

- Trained calibrated machine learning models on historical NFL team statistics
- Engineered 280+ rolling statistical features using recent and season-long performance
- Integrated Vegas betting information (spreads + moneyline implied probabilities)
- Simulated the **final four teams** (Conference Championships → Super Bowl)
- Estimated championship probabilities via **10,000 Monte Carlo simulations**

The goal is to produce **interpretable, well-calibrated probabilities** that reflect both team strength and matchup context.

---

## Data Pipeline

Scripts are run directly from the project root (no Python packaging):

~~~bash
python src/make_train_games.py
python src/train_model.py
python src/make_team_features.py
python src/simulate_playoffs.py
~~~

---

## Feature Engineering

- Rolling windows: last 3 games, last 8 games, season averages (`w3_`, `w8_`, `s_`)
- Home vs away statistical differentials (`*_diff`)
- Rest differential (`rest_diff`)
- Home-field indicator (`home_field`)
- Vegas spread feature (`spread_line_home`)
- Moneyline implied probability differential (`market_prob_diff`)

---

## Models

- Logistic Regression (baseline)
- Random Forest
- XGBoost
- Ensemble averaging across models
- Probability calibration (Platt scaling)

Models are trained on historical regular-season games and reused during playoff simulation.

---

## Final 4 Simulation Results

The simulation considers **only the remaining four teams**:

- **AFC Championship:** Denver Broncos vs New England Patriots  
- **NFC Championship:** Seattle Seahawks vs Los Angeles Rams  

### Conference Championship Win Probabilities

| Matchup | Predicted Winner | Win Probability |
|-------|------------------|----------------|
| DEN vs NE (AFC) | **New England Patriots** | **86.25%** |
| SEA vs LA (NFC) | **Los Angeles Rams** | **53.93%** |

Probabilities represent **single-game win likelihoods** from the calibrated ensemble model.

---

### Super Bowl Outcome (Monte Carlo Simulation)

After simulating the conference championships and Super Bowl **10,000 times**:

| Team | Super Bowl Win Probability |
|----|-----------------------------|
| **New England Patriots** | **46.60%** |
| Los Angeles Rams | 28.3% |
| Seattle Seahawks | 18.7% |
| Denver Broncos | 6.4% |

The Patriots emerge as the **most likely Super Bowl winner**, driven by a dominant AFC Championship win probability and favorable neutral-site matchup outcomes.

---

## Outputs

Generated artifacts (not tracked in GitHub):

~~~text
outputs/tables/superbowl_odds.csv
outputs/models/win_model.joblib
~~~

All outputs are fully reproducible by running the pipeline scripts.

---

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- joblib

---

## Future Improvements

- De-vigging market probabilities to remove sportsbook margin
- Adding playoff-specific features (seed number, experience proxies)
- Backtesting on historical playoff seasons
- Confidence intervals on Monte Carlo outputs

---

## Notes

This project prioritizes **clarity, reproducibility, and statistical rigor** over over-engineering or black-box predictions.  
All outputs are probabilistic and intended for analytical purposes, not betting advice.
