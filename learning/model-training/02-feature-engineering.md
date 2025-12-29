# Module 2: Feature Engineering

> By the end of this module, you'll understand how raw match data becomes 37 predictive features.

## 🎯 Learning Objective

Understand why feature engineering is critical and what features we create.

## 📚 The Problem

Raw match data only contains:
- Date, HomeTeam, AwayTeam, FTHG (home goals), FTAG (away goals)

But ML models need **patterns**. We need to answer:
- How is each team performing lately? (Form)
- How do these teams perform against each other? (Head-to-head)
- How are they doing this season? (Season stats)

## 💻 Code Walkthrough

### The FeatureEngineer Class

```python
# FILE: src/feature_engineering.py
# PURPOSE: Transform raw match data into ML-ready features

class FeatureEngineer:
    def __init__(self, n_last_matches: int = 5):
        """
        n_last_matches: How many recent games to consider for form.
        We use 5 because it balances recency with stability.
        """
        self.n_last_matches = n_last_matches
```

### Feature Categories

We create **37 features** in 4 categories:

#### 1. Team Form Features (14 features)
```python
def _calculate_team_form(self, df, team, date, is_home):
    """
    Look at the team's last N matches BEFORE this date.
    This prevents data leakage - we only use past data!
    """
    # Get matches before current date
    all_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)]
    all_matches = all_matches[all_matches['Date'] < date]
    last_n = all_matches.tail(self.n_last_matches)
    
    return {
        'form_points': last_n['Points'].mean(),        # Avg points (0-3)
        'form_goals_scored': last_n['GS'].mean(),      # Avg goals scored
        'form_goals_conceded': last_n['GC'].mean(),    # Avg goals conceded
        'form_clean_sheets': (last_n['GC'] == 0).mean(), # % clean sheets
        'form_wins': (last_n['Points'] == 3).mean(),   # Win rate
        'form_losses': (last_n['Points'] == 0).mean(), # Loss rate
        'matches_played': len(last_n)
    }
```

We calculate this for BOTH home and away team = 14 features.

#### 2. Home/Away Specific Form (6 features)
```python
def _calculate_home_away_form(self, df, team, date, is_home):
    """
    Some teams are strong at home but weak away.
    This captures venue-specific performance.
    """
    if is_home:
        matches = df[(df['HomeTeam'] == team) & (df['Date'] < date)]
    else:
        matches = df[(df['AwayTeam'] == team) & (df['Date'] < date)]
    
    return {
        f'{prefix}_form_points': matches['Points'].mean(),
        f'{prefix}_form_goals_scored': matches['GS'].mean(),
        f'{prefix}_form_goals_conceded': matches['GC'].mean(),
    }
```

#### 3. Head-to-Head Stats (4 features)
```python
def _calculate_head_to_head(self, df, home_team, away_team, date):
    """
    Historical matchups between these exact teams.
    Liverpool vs Man United has different dynamics than vs Brighton.
    """
    h2h = df[
        ((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
        ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))
    ]
    h2h = h2h[h2h['Date'] < date].tail(6)  # Last 6 meetings
    
    return {
        'h2h_home_wins': home_wins / len(h2h),
        'h2h_away_wins': away_wins / len(h2h),
        'h2h_draws': draws / len(h2h),
        'h2h_total_goals': (h2h['FTHG'] + h2h['FTAG']).mean()
    }
```

#### 4. Season Statistics (8 features)
```python
def _calculate_season_stats(self, df, team, date, season):
    """
    Current season form - are they fighting for title or relegation?
    """
    season_df = df[(df['Season'] == season) & (df['Date'] < date)]
    
    return {
        'season_points_per_game': total_points / matches,
        'season_goals_scored_per_game': goals_scored / matches,
        'season_goals_conceded_per_game': goals_conceded / matches,
        'season_position_estimate': estimated_league_position
    }
```

#### 5. Derived Features (5 features)
```python
# Differences between teams - captures relative strength
features['form_diff'] = home_form - away_form
features['goals_diff'] = home_goals - away_goals
features['season_points_diff'] = home_ppg - away_ppg
features['attack_vs_defense'] = home_attack - away_defense
features['h2h_dominance'] = h2h_home_wins - h2h_away_wins
```

## 🔗 Visual: Feature Creation Flow

```
┌─────────────────────────────────────────────────────────────┐
│  Raw Match: Liverpool vs Chelsea, 2025-01-15                │
└─────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
   ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
   │ Liverpool's     │ │ Chelsea's       │ │ Head-to-Head    │
   │ last 5 matches  │ │ last 5 matches  │ │ last 6 meetings │
   │ (BEFORE Jan 15) │ │ (BEFORE Jan 15) │ │ (BEFORE Jan 15) │
   └─────────────────┘ └─────────────────┘ └─────────────────┘
            │                 │                 │
            ▼                 ▼                 ▼
   ┌─────────────────────────────────────────────────────────┐
   │  37 Features for this match                             │
   │  [2.4, 1.8, 0.6, 0.4, 1.9, 1.2, ...]                   │
   └─────────────────────────────────────────────────────────┘
```

## 🔑 Key Insight

> **Preventing Data Leakage**: Notice every function uses `date < current_date`. We ONLY use data that would have been available BEFORE the match. This is critical - otherwise we'd be "cheating" by using future information.

## ✋ Pause & Reflect

1. Why do we calculate both overall form AND home/away specific form?
2. What would happen if we didn't filter by `date < current_date`?
3. Why use the last 5 matches instead of 10 or 3?
