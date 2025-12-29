# Module 4: Training Process

> By the end of this module, you'll understand the complete training pipeline and how models are persisted.

## 🎯 Learning Objective

Understand how training works, including data splitting, cross-validation, and model saving.

## 📚 The Training Philosophy

Football is **time-series data** - we can't randomly shuffle matches because:
- Past matches should predict future matches
- Random splits would leak future information into training

## 💻 Code Walkthrough

### The Training Script

```python
# FILE: train_models.py
# PURPOSE: Train all 19 leagues before deployment

def train_all_models():
    fetcher = OpenFootballFetcher()
    engineer = FeatureEngineer(n_last_matches=5)
    predictor = EPLPredictor(models_dir="models")
    
    for league in fetcher.get_available_leagues().keys():
        # 1. Fetch data
        df = fetcher.get_or_fetch_league_data(league)
        
        # 2. Create features (37 per match)
        df_features = engineer.create_features(df)
        
        # 3. Prepare data (remove insufficient history)
        feature_cols = engineer.get_feature_columns()
        df_clean = predictor.prepare_data(df_features, feature_cols)
        
        # 4. Train all 5 target models
        predictor.train(df_clean, feature_cols)
        
        # 5. Save models
        predictor.save_models(prefix=league.lower())
```

### Time-Based Train/Test Split

```python
def train(self, df: pd.DataFrame, feature_columns: list):
    X = df[feature_columns].values
    X_scaled = self.scaler.fit_transform(X)
    
    for target_name, target_config in self.targets.items():
        y = df[target_config['column']].values
        
        # TIME-BASED SPLIT: Use first 80% for training, last 20% for testing
        # This simulates real prediction: train on past, test on "future"
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
```

### Why 80/20 Split?

```
Timeline:
│◄─────── 80% Training Data ───────►│◄── 20% Test ──►│
│                                    │                │
│  2022-23     2023-24      2024-25 │    2024-25     │
│  Season      Season       (early)  │    (late)      │
│                                    │                │
└────────────────────────────────────┴────────────────┘
                                     ▲
                              "Cutoff Date"
```

### Cross-Validation with TimeSeriesSplit

```python
# Even within training data, we validate properly
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = cross_val_score(model, X_train, y_train, cv=tscv)

# This creates 5 folds, each using only PAST data to validate:
# Fold 1: Train on matches 1-100,   Validate on 101-150
# Fold 2: Train on matches 1-150,   Validate on 151-200
# Fold 3: Train on matches 1-200,   Validate on 201-250
# ... and so on
```

### Model Competition

```python
best_model = None
best_score = 0

for model_name, model in base_models.items():
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"  {model_name}: {accuracy:.4f}")
    
    # Keep the best
    if accuracy > best_score:
        best_score = accuracy
        best_model = model

# Store winner
self.models[target_name] = best_model
```

### Model Persistence

```python
def save_models(self, prefix: str = "epl"):
    """Save everything needed to make predictions later"""
    
    # 1. Save the scaler (needed to transform new features)
    joblib.dump(self.scaler, f"{prefix}_scaler.joblib")
    
    # 2. Save each of the 5 winning models
    for target_name, model in self.models.items():
        joblib.dump(model, f"{prefix}_{target_name}_model.joblib")
    
    # 3. Save metadata (feature columns, target configs)
    joblib.dump({
        'feature_columns': self.feature_columns,
        'targets': self.targets
    }, f"{prefix}_metadata.joblib")
```

### What Gets Saved

For each league, we save 7 files:

```
models/
├── epl_scaler.joblib              # Feature normalizer
├── epl_match_result_model.joblib  # 3-class prediction
├── epl_home_win_model.joblib      # Binary: home win?
├── epl_away_win_model.joblib      # Binary: away win?
├── epl_over_1.5_model.joblib      # Binary: 2+ goals?
├── epl_over_2.5_model.joblib      # Binary: 3+ goals?
└── epl_metadata.joblib            # Feature columns list
```

For 19 leagues: **133 total files** (19 × 7)

## 🔗 Complete Training Flow

```
┌─────────────────────────────────────────────────────────────┐
│  train_models.py                                            │
│  "python train_models.py"                                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
           ┌──────────────────────────────────┐
           │  For each of 19 leagues:         │
           │  EPL, LA_LIGA, SERIE_A, ...      │
           └──────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   ┌─────────┐         ┌───────────┐         ┌───────────┐
   │ Fetch   │   →     │ Engineer  │    →    │ Train     │
   │ Data    │         │ Features  │         │ Models    │
   └─────────┘         └───────────┘         └───────────┘
        │                     │                     │
        ▼                     ▼                     ▼
   ┌─────────┐         ┌───────────┐         ┌───────────┐
   │ ~500    │         │ 37 cols   │         │ 5 targets │
   │ matches │         │ per row   │         │ 3 models  │
   └─────────┘         └───────────┘         │ each      │
                                             └───────────┘
                                                   │
                                                   ▼
                                      ┌───────────────────────┐
                                      │ Save 7 .joblib files  │
                                      │ per league            │
                                      └───────────────────────┘
```

## 🔑 Key Insight

> **Why Train Locally?** Each league takes 30-60 seconds to train. With 19 leagues, that's 10-20 minutes total. On Streamlit Cloud, this would timeout. By pre-training locally and committing the .joblib files, the app loads instantly.

## ✋ Pause & Reflect

1. Why can't we use random train/test split for time-series data?
2. What would happen if we deployed without pre-training?
3. Why do we save the scaler along with the models?
