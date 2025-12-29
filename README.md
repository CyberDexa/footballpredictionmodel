# ⚽ Football Match Prediction Model

AI-powered football match predictions using machine learning. Predicts match outcomes, over/under goals for major European leagues.

## 🎯 Features

- **Multi-League Support**: EPL, La Liga, Serie A, Bundesliga, Ligue 1, Championship
- **Real Data**: Uses OpenFootball - free, no API key required
- **ML Predictions**:
  - Match Result (Home/Draw/Away)
  - Over/Under 1.5 Goals
  - Over/Under 2.5 Goals
  - Home Win / Away Win probability
- **Beautiful UI**: Modern Streamlit interface
- **Auto-Refresh**: Weekly data updates
- **37 Features**: Team form, head-to-head, season stats

## 🚀 Quick Start

### Local Installation

```bash
# Clone the repository
git clone https://github.com/CyberDexa/footballpredictionmodel.git
cd footballpredictionmodel

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Deploy to Streamlit Cloud (Free Hosting)

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select repository: `CyberDexa/footballpredictionmodel`
5. Set main file path: `app.py`
6. Click "Deploy"

## 📊 Data Source

Uses [OpenFootball](https://openfootball.github.io/) - free, open public domain football data.
- No API key required
- Updated regularly with current season matches
- Historical data from 2010+

## 🏆 Supported Leagues

| League | Country |
|--------|---------|
| English Premier League | England |
| English Championship | England |
| La Liga | Spain |
| Serie A | Italy |
| Bundesliga | Germany |
| Ligue 1 | France |

## 🤖 Model Performance

The models use Random Forest, Gradient Boosting, and Logistic Regression ensembles:

| Prediction | Typical Accuracy |
|------------|------------------|
| Match Result | ~50-55% |
| Over 1.5 Goals | ~70-80% |
| Over 2.5 Goals | ~50-55% |
| Home Win | ~65-70% |
| Away Win | ~70-75% |

## 📁 Project Structure

```
footballpredictionmodel/
├── app.py                 # Streamlit web app
├── main.py                # CLI interface
├── requirements.txt       # Python dependencies
├── src/
│   ├── openfootball_fetcher.py  # Data fetching
│   ├── feature_engineering.py   # 37 features
│   └── models.py                # ML models
├── data/                  # Cached match data
└── models/                # Trained models
```

## 📝 License

MIT License - feel free to use and modify!

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.
