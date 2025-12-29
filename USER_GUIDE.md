# User Guide: Football Match Predictor

**Version**: 1.0  
**Last Updated**: January 2025

---

## Overview

The Football Match Predictor is an AI-powered application that uses machine learning to predict football match outcomes. It analyzes historical match data to provide probability estimates for various betting markets across 19 major football leagues.

**Key Features:**
- 🎯 17 different prediction markets
- ⚽ 19 major football leagues worldwide
- 📅 Automatic upcoming fixture predictions
- 📊 League statistics and analytics
- 🆓 Free data source - no API key required

---

## Getting Started

### Starting the Application

1. Open a terminal in the project directory
2. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```
4. Open your browser to `http://localhost:8501`

---

## Features

### 🔮 Predict Match

This is the main prediction interface where you can select any two teams from the chosen league and get predictions.

**How to Use:**
1. Select a league from the sidebar dropdown (e.g., English Premier League)
2. Choose the **Home Team** from the first dropdown
3. Choose the **Away Team** from the second dropdown
4. Click **🔮 Get Prediction**

**What You'll See:**
- **Match Result**: Probability for Home Win, Draw, and Away Win
- **Goals Markets**: Over/Under 1.5, 2.5, and 3.5 goals
- **BTTS**: Both Teams to Score (Yes/No)
- **Team Goals**: Home and Away team goal predictions
- **Half Time**: HT Over 0.5 and 1.5 goals
- **Goal Ranges**: 0-1, 2-3, or 4+ total goals

### 📅 Upcoming Matches

Automatically fetches upcoming fixtures from the selected league's schedule.

**How to Use:**
1. Navigate to the **📅 Upcoming Matches** tab
2. Browse the list of upcoming fixtures
3. Click on any fixture to expand it
4. Click **🔮 Predict** to get predictions for that match

**Available Information:**
- Match date and time
- Home and Away teams
- Full prediction analysis for each fixture

### 📊 Stats

View comprehensive statistics for the selected league.

**Available Statistics:**
- Total goals scored in the season
- Average goals per match
- Home win percentage
- Away win percentage
- Goals per match distribution chart
- Recent match results table

---

## 17 Prediction Markets Explained

| Market | Description |
|--------|-------------|
| **Home Win** | Probability the home team wins |
| **Draw** | Probability the match ends in a draw |
| **Away Win** | Probability the away team wins |
| **Over 1.5 Goals** | Probability of 2+ total goals |
| **Over 2.5 Goals** | Probability of 3+ total goals |
| **Over 3.5 Goals** | Probability of 4+ total goals |
| **BTTS Yes** | Both teams score at least one goal |
| **BTTS No** | At least one team fails to score |
| **Home Over 0.5** | Home team scores at least 1 goal |
| **Home Over 1.5** | Home team scores at least 2 goals |
| **Home Over 2.5** | Home team scores at least 3 goals |
| **Away Over 0.5** | Away team scores at least 1 goal |
| **Away Over 1.5** | Away team scores at least 2 goals |
| **Away Over 2.5** | Away team scores at least 3 goals |
| **HT Over 0.5** | At least 1 goal scored in first half |
| **HT Over 1.5** | At least 2 goals scored in first half |
| **0-1 Goals** | Total match goals between 0-1 |
| **2-3 Goals** | Total match goals between 2-3 |
| **4+ Goals** | 4 or more total goals in match |

---

## 19 Supported Leagues

| Country | League |
|---------|--------|
| 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England | Premier League, Championship, League One, League Two |
| 🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scotland | Premiership |
| 🇪🇸 Spain | La Liga, La Liga 2 |
| 🇮🇹 Italy | Serie A, Serie B |
| 🇩🇪 Germany | Bundesliga, Bundesliga 2 |
| 🇫🇷 France | Ligue 1, Ligue 2 |
| 🇳🇱 Netherlands | Eredivisie |
| 🇵🇹 Portugal | Primeira Liga |
| 🇨🇭 Switzerland | Super League |
| 🇧🇪 Belgium | Jupiler League |
| 🇹🇷 Turkey | Süper Lig |
| 🇷🇺 Russia | Premier League |

---

## Data Management

### Refreshing Data

Data is sourced from OpenFootball (free, open public domain).

- **Auto-refresh**: Enable "Auto-refresh weekly" checkbox in sidebar
- **Manual refresh**: Click **🔄 Refresh** for current league
- **Refresh all**: Click **🔄 All Leagues** to update all 19 leagues

### Data Freshness

The app shows data age in the sidebar:
- 🟢 Green: Data is less than 7 days old
- 🔴 Red: Data is older than 7 days

---

## Model Training

### Retrain Models

If you've refreshed data and want to update the ML models:

1. Click **🔄 Retrain Model** in the sidebar
2. Wait for training to complete (typically 2-5 minutes per league)
3. New predictions will use the updated models

### Training Details

- **Algorithms**: Random Forest, Gradient Boosting, Logistic Regression
- **Features**: 37 engineered features per match
- **Model files**: Stored in `models/` directory

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| App won't start | Ensure virtual environment is activated |
| No teams showing | Refresh data for the selected league |
| Predictions are 50% | Model needs more training data - retrain |
| Upcoming matches empty | League schedule may not be published yet |

### Error Messages

- **"No data available"**: Click Refresh to fetch data
- **"Model not found"**: Click Retrain Model
- **"Team not found"**: Team name may have changed - refresh data

---

## Best Practices

1. **Keep data fresh**: Refresh weekly for best predictions
2. **Use multiple markets**: Don't rely on a single prediction
3. **Check confidence levels**: Higher percentages = more confident predictions
4. **Compare with stats**: Use the Stats tab to validate predictions
5. **Remember the disclaimer**: Predictions are informational only

---

## Technical Specifications

| Component | Details |
|-----------|---------|
| Framework | Streamlit |
| ML Library | scikit-learn |
| Data Source | OpenFootball (openfootball.github.io) |
| Python Version | 3.10+ |
| Model Format | joblib (.joblib files) |

---

## Support

- **GitHub Repository**: https://github.com/CyberDexa/footballpredictionmodel
- **Data Source**: https://openfootball.github.io/
- **Issues**: Open an issue on GitHub

---

## Disclaimer

⚠️ **Important**: These predictions are for informational and entertainment purposes only. Past performance does not guarantee future results. Do not use these predictions for gambling or financial decisions.

---

*User Guide v1.0 - Football Match Predictor*
