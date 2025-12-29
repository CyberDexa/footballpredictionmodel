# Module 3: Model Architecture

> By the end of this module, you'll understand the ML models we use and why we train 5 different targets.

## 🎯 Learning Objective

Understand the multi-model architecture and why we use an ensemble approach.

## 📚 The Problem

Football prediction isn't one problem - it's **five**:

| Target | Type | Question |
|--------|------|----------|
| `match_result` | 3-class | Who wins? (Home/Draw/Away) |
| `home_win` | Binary | Will home team win? |
| `away_win` | Binary | Will away team win? |
| `over_1.5` | Binary | Will there be 2+ goals? |
| `over_2.5` | Binary | Will there be 3+ goals? |

Each target is a separate prediction model.

## 💻 Code Walkthrough

### The EPLPredictor Class

```python
# FILE: src/models.py
# PURPOSE: Train and predict for multiple targets

class EPLPredictor:
    def __init__(self, models_dir: str = "models"):
        self.models = {}  # Will hold 5 trained models
        self.scaler = StandardScaler()  # Normalizes features
        
        # Define all prediction targets
        self.targets = {
            'match_result': {
                'column': 'MatchResult',
                'type': 'multiclass',
                'labels': {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}
            },
            'over_1.5': {
                'column': 'Over1.5',
                'type': 'binary',
                'labels': {0: 'Under 1.5', 1: 'Over 1.5'}
            },
            # ... 3 more targets
        }
```

### The Model Ensemble

For each target, we train **3 different algorithms** and keep the best:

```python
def _get_base_models(self) -> Dict[str, Any]:
    """Three model types compete for best accuracy"""
    return {
        'rf': RandomForestClassifier(
            n_estimators=200,     # 200 decision trees
            max_depth=10,         # Prevent overfitting
            min_samples_split=10, # Require 10 samples to split
            random_state=42
        ),
        'gb': GradientBoostingClassifier(
            n_estimators=150,     # 150 boosting stages
            max_depth=5,          # Shallower trees
            learning_rate=0.1,    # Conservative learning
            random_state=42
        ),
        'lr': LogisticRegression(
            max_iter=1000,        # Enough iterations to converge
            solver='lbfgs'        # Good for multiclass
        )
    }
```

### Why These Three Models?

| Model | Strengths | Weaknesses |
|-------|-----------|------------|
| **Random Forest** | Handles non-linear patterns, robust to outliers | Can overfit with too many trees |
| **Gradient Boosting** | Sequential learning, often highest accuracy | Slower training, sensitive to noise |
| **Logistic Regression** | Fast, interpretable, good baseline | Assumes linear relationships |

### Feature Scaling

```python
# Before training, we normalize all features
X_scaled = self.scaler.fit_transform(X)

# Why? Features have different scales:
# - form_points: 0-3
# - goals_scored: 0-5
# - season_position: 1-20

# After scaling, all features have mean=0, std=1
# This helps models treat all features equally
```

## 🔗 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    37 Input Features                        │
│  [form_points, goals_scored, h2h_wins, season_ppg, ...]    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    StandardScaler                           │
│              (normalize to mean=0, std=1)                   │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Random Forest   │ │ Gradient Boost  │ │ Logistic Reg    │
│ (200 trees)     │ │ (150 stages)    │ │                 │
└─────────────────┘ └─────────────────┘ └─────────────────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   SELECT BEST MODEL                         │
│            (highest test accuracy wins)                     │
└─────────────────────────────────────────────────────────────┘
                              │
     ┌────────────┬───────────┼───────────┬────────────┐
     ▼            ▼           ▼           ▼            ▼
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│ match   │ │ home    │ │ away    │ │ over    │ │ over    │
│ result  │ │ win     │ │ win     │ │ 1.5     │ │ 2.5     │
│ model   │ │ model   │ │ model   │ │ model   │ │ model   │
└─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘
```

## 🔑 Key Insight

> **Competition = Better Models**: By training 3 different algorithms and picking the winner, we get the best of each approach. Random Forest often wins for over/under predictions, while Gradient Boosting tends to win for match result.

## ✋ Pause & Reflect

1. Why do we have separate models for `home_win` and `away_win` instead of using `match_result`?
2. What does the StandardScaler do and why is it important?
3. Why might different algorithms win for different targets?
