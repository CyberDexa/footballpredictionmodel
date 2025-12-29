# Summary: Model Training Pipeline

> Quick reference card for the Football Prediction Model training system.

---

## 🏗️ Architecture Overview

```
Data Source → Feature Engineering → Model Training → Model Storage → Prediction
```

---

## 📊 Key Numbers

| Metric | Value |
|--------|-------|
| Leagues | 19 |
| Features per match | 37 |
| Prediction targets | 5 |
| Models trained per target | 3 (pick best) |
| Files saved per league | 7 |
| Total model files | 133 |

---

## 📁 File Structure

```
footballpredictionmodel/
├── src/
│   ├── openfootball_fetcher.py  # Data acquisition
│   ├── feature_engineering.py   # 37 features
│   └── models.py                # ML models
├── data/
│   └── {league}_data.csv        # Cached match data
├── models/
│   ├── {league}_scaler.joblib
│   ├── {league}_match_result_model.joblib
│   ├── {league}_home_win_model.joblib
│   ├── {league}_away_win_model.joblib
│   ├── {league}_over_1.5_model.joblib
│   ├── {league}_over_2.5_model.joblib
│   └── {league}_metadata.joblib
└── train_models.py              # Batch training script
```

---

## 🎯 The 5 Prediction Targets

| Target | Type | Output |
|--------|------|--------|
| `match_result` | 3-class | Home Win / Draw / Away Win |
| `home_win` | Binary | Yes / No |
| `away_win` | Binary | Yes / No |
| `over_1.5` | Binary | Over / Under 1.5 goals |
| `over_2.5` | Binary | Over / Under 2.5 goals |

---

## 🔢 The 37 Features

### Team Form (14 features)
- `home_form_points`, `away_form_points`
- `home_form_goals_scored`, `away_form_goals_scored`
- `home_form_goals_conceded`, `away_form_goals_conceded`
- `home_form_clean_sheets`, `away_form_clean_sheets`
- `home_form_wins`, `away_form_wins`
- `home_form_losses`, `away_form_losses`
- `home_matches_played`, `away_matches_played`

### Venue-Specific Form (6 features)
- `home_team_home_form_points`, `home_team_home_form_goals_scored`, `home_team_home_form_goals_conceded`
- `away_team_away_form_points`, `away_team_away_form_goals_scored`, `away_team_away_form_goals_conceded`

### Head-to-Head (4 features)
- `h2h_home_wins`, `h2h_away_wins`, `h2h_draws`, `h2h_total_goals`

### Season Stats (8 features)
- `home_season_points_per_game`, `away_season_points_per_game`
- `home_season_goals_scored_per_game`, `away_season_goals_scored_per_game`
- `home_season_goals_conceded_per_game`, `away_season_goals_conceded_per_game`
- `home_season_position_estimate`, `away_season_position_estimate`

### Derived (5 features)
- `form_diff`, `goals_diff`, `season_points_diff`, `attack_vs_defense`, `h2h_dominance`

---

## 🤖 The 3 Model Types

| Model | Best For | Parameters |
|-------|----------|------------|
| Random Forest | Binary classification | 200 trees, max_depth=10 |
| Gradient Boosting | Multi-class | 150 stages, learning_rate=0.1 |
| Logistic Regression | Baseline/interpretability | max_iter=1000 |

---

## ⚡ Quick Commands

```bash
# Train all models locally
python train_models.py

# Check trained models
ls models/ | grep -o "^[a-z_]*_" | sort -u

# Count total model files
ls models/*.joblib | wc -l
```

---

## 🔑 Key Concepts

### Data Leakage Prevention
Always filter: `df[df['Date'] < current_date]`

### Time-Series Validation
Use `TimeSeriesSplit(n_splits=5)` not random split

### Feature Scaling
`StandardScaler` normalizes to mean=0, std=1

### Model Selection
Train 3 models, pick highest test accuracy

---

## 🌐 Data Flow

```
OpenFootball (GitHub) 
    ↓ HTTP GET
DataFrame (raw matches)
    ↓ FeatureEngineer
DataFrame (37 features)
    ↓ StandardScaler
Scaled features
    ↓ Train 3 models × 5 targets
Best models selected
    ↓ joblib.dump()
133 .joblib files
```

---

## 📚 Related Files

- [01-data-pipeline.md](./01-data-pipeline.md) - Data fetching
- [02-feature-engineering.md](./02-feature-engineering.md) - Creating features
- [03-model-architecture.md](./03-model-architecture.md) - ML models
- [04-training-process.md](./04-training-process.md) - Training pipeline
