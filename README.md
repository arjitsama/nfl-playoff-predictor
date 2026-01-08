# NFL Playoff Predictor

A machine learning project to predict NFL playoff outcomes and simulate playoff brackets.

## Project Structure

```
.
├── main.py                 # Main entry point
├── data/
│   ├── raw/               # Raw NFL data (schedules, team stats)
│   └── processed/         # Processed datasets for training
├── src/
│   ├── make_team_features.py    # Feature engineering
│   ├── make_train_games.py      # Training data preparation
│   ├── train_model.py           # Model training
│   ├── playoff_bracket.py       # Playoff bracket logic
│   └── simulate_playoffs.py     # Playoff simulation
└── outputs/
    ├── models/            # Trained models
    └── tables/            # Output tables (odds, predictions)
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/nfl-predictor.git
cd nfl-predictor
```

2. Create a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
# source .venv/bin/activate  # On macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python main.py
```

Or run specific modules:
```bash
python src/simulate_playoffs.py
```

## Features

- Team feature engineering from NFL statistics
- Machine learning model for win prediction
- Playoff bracket simulation
- Generates betting odds and predictions

## Requirements

See `requirements.txt` for dependencies.
