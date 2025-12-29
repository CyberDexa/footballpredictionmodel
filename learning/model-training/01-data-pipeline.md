# Module 1: Data Pipeline

> By the end of this module, you'll understand how raw match data flows from the internet to your training pipeline.

## 🎯 Learning Objective

Understand the data acquisition layer and how OpenFootball provides free, current-season data.

## 📚 The Problem

We needed real football match data to train our ML models. Options considered:

| Source | Pros | Cons | Decision |
|--------|------|------|----------|
| API-Football | Official, comprehensive | Free tier: only 2021-2023, no current season | ❌ |
| StatsBomb | Professional grade | No EPL data in free tier | ❌ |
| OpenFootball | Free, current season, 19+ leagues | Community maintained | ✅ |

## 💻 Code Walkthrough

### The Fetcher Class

```python
# FILE: src/openfootball_fetcher.py
# PURPOSE: Fetch match data from openfootball.github.io

class OpenFootballFetcher:
    # Base URL - raw JSON from GitHub
    BASE_URL = "https://raw.githubusercontent.com/openfootball/football.json/master"
    
    # Each league maps to a JSON file
    LEAGUE_FILES = {
        'EPL': 'en.1.json',           # English Premier League
        'CHAMPIONSHIP': 'en.2.json',   # English Championship
        'SCOTTISH_PREM': 'sco.1.json', # Scottish Premiership
        'LA_LIGA': 'es.1.json',        # Spanish La Liga
        # ... 19 leagues total
    }
```

### Fetching a Season

```python
def fetch_season(self, league: str, season: str) -> pd.DataFrame:
    """
    Fetch one season of data for a league.
    
    The URL pattern is: {BASE_URL}/{season}/{league_file}
    Example: .../2024-25/en.1.json for EPL 2024-25 season
    """
    file_name = self.LEAGUE_FILES.get(league)
    url = f"{self.BASE_URL}/{season}/{file_name}"
    
    response = requests.get(url, timeout=30)
    data = response.json()
    
    # Parse the JSON structure into rows
    matches = []
    for match_round in data.get('rounds', []):
        for match in match_round.get('matches', []):
            matches.append({
                'Date': match['date'],
                'HomeTeam': match['team1']['name'],
                'AwayTeam': match['team2']['name'],
                'FTHG': match['score']['ft'][0],  # Full Time Home Goals
                'FTAG': match['score']['ft'][1],  # Full Time Away Goals
                'Season': season
            })
    
    return pd.DataFrame(matches)
```

### The Data Flow

```
┌──────────────────────────────────────────────────────────────┐
│  openfootball.github.io (GitHub Pages)                       │
│  ├── 2025-26/                                                │
│  │   ├── en.1.json  ← EPL current season                     │
│  │   ├── es.1.json  ← La Liga current season                 │
│  │   └── ...                                                 │
│  ├── 2024-25/                                                │
│  └── 2023-24/                                                │
└──────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP GET (no API key!)
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  OpenFootballFetcher.fetch_league()                          │
│  - Fetches multiple seasons                                  │
│  - Concatenates into single DataFrame                        │
│  - Adds FTR column (H/D/A result)                           │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  data/epl_data.csv                                           │
│  - Cached locally for fast access                            │
│  - ~500 matches per major league                             │
│  - Auto-refreshes if older than 7 days                       │
└──────────────────────────────────────────────────────────────┘
```

## 🔑 Key Insight

> **Why OpenFootball?** It's the only free data source that includes the **current 2025-26 season**. API-Football's free tier only has 2021-2023 data, making predictions useless for current matches.

## ✋ Pause & Reflect

Before continuing, can you answer:
1. Why do we fetch multiple seasons (not just current)?
2. What does the cached CSV prevent?
3. What triggers an auto-refresh?
