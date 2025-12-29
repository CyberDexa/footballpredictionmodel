# Football Match Prediction Model ⚽

A machine learning system that predicts football match outcomes across **7 major European leagues** using **real data** from API-Football and StatsBomb.

## 🔑 Getting Real Data (Required for Current Season)

### Option 1: API-Football (Recommended - Current 2025/26 Data)

1. **Get a FREE API key** at [api-football.com](https://www.api-football.com/)
   - Free tier: 100 requests/day
   - All leagues & competitions
   - Live scores & current fixtures

2. **Set your API key:**
   ```bash
   # Option A: Environment variable
   export API_FOOTBALL_KEY='your-api-key-here'
   
   # Option B: In the app sidebar
   # Just paste your key in the API Key field
   ```

### Option 2: StatsBomb Open Data (Free, No Key Required)

StatsBomb provides free historical data including:
- La Liga (2004-2021)
- Premier League (2003-2016)  
- Bundesliga (2015-2024)
- Serie A (2015-2016, 1986-1987)
- Ligue 1 (2015-2023)
- Champions League, World Cup, and more!

## 🏆 Supported Leagues

| League | Country | Flag |
|--------|---------|------|
| Premier League | England | 🏴󠁧󠁢󠁥󠁮󠁧󠁿 |
| La Liga | Spain | 🇪🇸 |
| Serie A | Italy | 🇮🇹 |
| Bundesliga | Germany | 🇩🇪 |
| Ligue 1 | France | 🇫🇷 |
| Eredivisie | Netherlands | 🇳🇱 |
| Primeira Liga | Portugal | 🇵🇹 |

## Predictions Available

- **Match Result** (Home Win / Draw / Away Win)
- **Over/Under 1.5 Goals**
- **Over/Under 2.5 Goals**

## Features

- 🌐 **Web UI** - Beautiful Streamlit interface
- 📊 Automatically fetches historical data from football-data.co.uk
- 🔧 Advanced feature engineering including:
  - Team form (last 5 matches)
  - Home/Away specific performance
  - Head-to-head records
  - Season statistics
- 🤖 Ensemble of ML models (Random Forest, XGBoost, LightGBM, Gradient Boosting)
- 💾 Model persistence for quick predictions

## Installation

```bash
cd footballpredictionmodel

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or: venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start - Web UI

Launch the beautiful web interface:

```bash
streamlit run app.py
```

This opens a browser with the prediction app where you can:
- Select any supported league
- Pick home and away teams
- Get instant AI predictions
- View probability distributions

![Web UI Screenshot](docs/screenshot.png)

## Command Line Usage

### 1. Train the Models

Run the full training pipeline:

```bash
python main.py --train
```

### 2. Make Predictions (CLI)

Predict a specific match:

```bash
python main.py --predict Arsenal Chelsea
python main.py --predict "Man United" Liverpool
python main.py --predict "Man City" "Aston Villa"
```

### 3. List Available Teams

```bash
python main.py --teams
```

### 4. Force Refresh Data

```bash
python main.py --train --refresh
```

## Example Output

```
PREDICTION: Arsenal vs Chelsea
============================================================

🏆 MATCH RESULT: Home Win
   Confidence: 54.2%
   Probabilities:
      Home Win: 54.2%
      Draw: 24.3%
      Away Win: 21.5%

⚽ GOALS 1.5: Over 1.5
   Over 1.5: 78.3%
   Under 1.5: 21.7%

⚽ GOALS 2.5: Over 2.5
   Over 2.5: 56.8%
   Under 2.5: 43.2%
```

## Using in Python Code

```python
from main import EPLPredictionSystem

# Initialize the system
system = EPLPredictionSystem()

# Train models (first time only)
system.run_full_pipeline()

# Make predictions
prediction = system.predict_upcoming_match("Liverpool", "Man City")
system.print_prediction(prediction)

# Access raw prediction data
print(prediction['predictions']['match_result']['probabilities'])
print(prediction['predictions']['goals_2.5']['over_probability'])
```

## Model Performance

Typical accuracy ranges (may vary based on training data):
- Match Result: 50-55% (3-way classification)
- Over/Under 2.5: 55-62%
- Over/Under 1.5: 65-75%

## Project Structure

```
footballpredictionmodel/
├── app.py                 # Streamlit Web UI
├── main.py                # CLI entry point
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py         # EPL data fetcher
│   ├── multi_league_fetcher.py # Multi-league data fetcher
│   ├── feature_engineering.py  # Creates features
│   └── models.py               # ML models
├── data/                  # Cached data (created automatically)
└── models/                # Saved models (created automatically)
```

## Data Source

Historical match data is sourced from [football-data.co.uk](https://www.football-data.co.uk/), which provides free historical football data including:
- Match results
- Goals scored
- Shots, shots on target
- Corners, fouls
- Cards (yellow/red)

## Features Used for Prediction

| Feature Category | Description |
|-----------------|-------------|
| Team Form | Points, goals scored/conceded in last 5 matches |
| Home/Away Form | Performance specifically at home or away |
| Head-to-Head | Historical record between the two teams |
| Season Stats | Current season points per game, goal metrics |
| Derived | Form difference, attack/defense strength differential |

## Notes

- The model uses time-based train/test split to avoid data leakage
- Predictions are probabilistic - use the probabilities for better decision making
- Models are retrained on the full dataset after initial validation
- First run may take a few minutes to download data and train models

## License

MIT License - Feel free to use and modify for your own projects!
