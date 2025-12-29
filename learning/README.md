# 📚 Learning Materials Index

> Your personal knowledge base for the Football Prediction Model

## Learning Paths

| Topic | Modules | Time | Description |
|-------|---------|------|-------------|
| [Model Training](./model-training/00-overview.md) | 4 | 20 min | How we train ML models for 19 leagues |

## Quick Reference
- [Training Summary Card](./model-training/summary.md) - Quick reference for training pipeline
- [Knowledge Check](./model-training/knowledge-check.md) - Test yourself

## How to Use

1. Start with the overview (`00-overview.md`)
2. Work through modules in order
3. Test yourself with `knowledge-check.md`
4. Keep `summary.md` handy for reference

## Project Structure Explained

```
footballpredictionmodel/
├── src/                    # Core logic
│   ├── openfootball_fetcher.py  # Data acquisition
│   ├── feature_engineering.py   # 37 features
│   └── models.py                # ML models
├── data/                   # Cached match data
├── models/                 # Trained models (133 files)
├── app.py                  # Streamlit UI
├── train_models.py         # Batch training script
└── learning/               # 📚 You are here!
```

## Key Technologies

- **Python 3.13** - Programming language
- **scikit-learn** - ML algorithms (RandomForest, GradientBoosting, LogisticRegression)
- **pandas** - Data manipulation
- **Streamlit** - Web UI
- **OpenFootball** - Free data source (no API key!)
