"""
Feature Engineering Module
Creates features for EPL match prediction
"""

import pandas as pd
import numpy as np
from typing import Tuple


class FeatureEngineer:
    """Creates features for football match prediction"""
    
    def __init__(self, n_last_matches: int = 5):
        """
        Args:
            n_last_matches: Number of previous matches to consider for form calculation
        """
        self.n_last_matches = n_last_matches
        self.team_stats = {}
    
    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for prediction - Extended targets"""
        df = df.copy()
        
        # Calculate total goals
        df['TotalGoals'] = df['FTHG'] + df['FTAG']
        
        # Match result targets
        df['HomeWin'] = (df['FTR'] == 'H').astype(int)
        df['Draw'] = (df['FTR'] == 'D').astype(int)
        df['AwayWin'] = (df['FTR'] == 'A').astype(int)
        df['MatchResult'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})
        
        # Full Time Total Goals
        df['Over1.5'] = (df['TotalGoals'] > 1.5).astype(int)
        df['Over2.5'] = (df['TotalGoals'] > 2.5).astype(int)
        df['Over3.5'] = (df['TotalGoals'] > 3.5).astype(int)
        df['Under1.5'] = (df['TotalGoals'] < 1.5).astype(int)
        df['Under2.5'] = (df['TotalGoals'] < 2.5).astype(int)
        
        # Goal Ranges
        df['Goals0_1'] = (df['TotalGoals'] <= 1).astype(int)
        df['Goals2_3'] = ((df['TotalGoals'] >= 2) & (df['TotalGoals'] <= 3)).astype(int)
        df['Goals4Plus'] = (df['TotalGoals'] >= 4).astype(int)
        
        # Both Teams To Score
        df['BTTS'] = ((df['FTHG'] > 0) & (df['FTAG'] > 0)).astype(int)
        
        # Home Team Goals
        df['HomeOver0.5'] = (df['FTHG'] > 0.5).astype(int)
        df['HomeOver1.5'] = (df['FTHG'] > 1.5).astype(int)
        df['HomeOver2.5'] = (df['FTHG'] > 2.5).astype(int)
        
        # Away Team Goals
        df['AwayOver0.5'] = (df['FTAG'] > 0.5).astype(int)
        df['AwayOver1.5'] = (df['FTAG'] > 1.5).astype(int)
        df['AwayOver2.5'] = (df['FTAG'] > 2.5).astype(int)
        
        # Half Time Goals (if available)
        if 'HTHG' in df.columns and 'HTAG' in df.columns:
            df['HTGoals'] = df['HTHG'] + df['HTAG']
            df['HTOver0.5'] = (df['HTGoals'] > 0.5).astype(int)
            df['HTOver1.5'] = (df['HTGoals'] > 1.5).astype(int)
            df['HTOver2.5'] = (df['HTGoals'] > 2.5).astype(int)
            # Home HT Goals
            df['HomeHTOver0.5'] = (df['HTHG'] > 0.5).astype(int)
            # Away HT Goals
            df['AwayHTOver0.5'] = (df['HTAG'] > 0.5).astype(int)
        else:
            # Default values if HT data not available
            df['HTGoals'] = 0
            df['HTOver0.5'] = 0
            df['HTOver1.5'] = 0
            df['HTOver2.5'] = 0
            df['HomeHTOver0.5'] = 0
            df['AwayHTOver0.5'] = 0
        
        return df
    
    def _calculate_team_form(self, df: pd.DataFrame, team: str, 
                             date: pd.Timestamp, is_home: bool) -> dict:
        """Calculate team form based on last N matches"""
        # Get all matches for the team before this date
        home_matches = df[(df['HomeTeam'] == team) & (df['Date'] < date)].copy()
        away_matches = df[(df['AwayTeam'] == team) & (df['Date'] < date)].copy()
        
        # Add goals scored/conceded from team perspective
        home_matches['GoalsScored'] = home_matches['FTHG']
        home_matches['GoalsConceded'] = home_matches['FTAG']
        home_matches['Points'] = home_matches['FTR'].map({'H': 3, 'D': 1, 'A': 0})
        home_matches['IsHome'] = True
        
        away_matches['GoalsScored'] = away_matches['FTAG']
        away_matches['GoalsConceded'] = away_matches['FTHG']
        away_matches['Points'] = away_matches['FTR'].map({'H': 0, 'D': 1, 'A': 3})
        away_matches['IsHome'] = False
        
        # Combine and sort by date
        all_matches = pd.concat([home_matches, away_matches]).sort_values('Date', ascending=False)
        last_n = all_matches.head(self.n_last_matches)
        
        if len(last_n) == 0:
            return {
                'form_points': 0,
                'form_goals_scored': 0,
                'form_goals_conceded': 0,
                'form_clean_sheets': 0,
                'form_wins': 0,
                'form_losses': 0,
                'matches_played': 0
            }
        
        return {
            'form_points': last_n['Points'].mean(),
            'form_goals_scored': last_n['GoalsScored'].mean(),
            'form_goals_conceded': last_n['GoalsConceded'].mean(),
            'form_clean_sheets': (last_n['GoalsConceded'] == 0).sum() / len(last_n),
            'form_wins': (last_n['Points'] == 3).sum() / len(last_n),
            'form_losses': (last_n['Points'] == 0).sum() / len(last_n),
            'matches_played': len(last_n)
        }
    
    def _calculate_home_away_form(self, df: pd.DataFrame, team: str, 
                                   date: pd.Timestamp, is_home: bool) -> dict:
        """Calculate team's home or away specific form"""
        if is_home:
            matches = df[(df['HomeTeam'] == team) & (df['Date'] < date)].copy()
            matches['GoalsScored'] = matches['FTHG']
            matches['GoalsConceded'] = matches['FTAG']
            matches['Points'] = matches['FTR'].map({'H': 3, 'D': 1, 'A': 0})
        else:
            matches = df[(df['AwayTeam'] == team) & (df['Date'] < date)].copy()
            matches['GoalsScored'] = matches['FTAG']
            matches['GoalsConceded'] = matches['FTHG']
            matches['Points'] = matches['FTR'].map({'H': 0, 'D': 1, 'A': 3})
        
        last_n = matches.sort_values('Date', ascending=False).head(self.n_last_matches)
        
        prefix = 'home' if is_home else 'away'
        
        if len(last_n) == 0:
            return {
                f'{prefix}_form_points': 0,
                f'{prefix}_form_goals_scored': 0,
                f'{prefix}_form_goals_conceded': 0,
            }
        
        return {
            f'{prefix}_form_points': last_n['Points'].mean(),
            f'{prefix}_form_goals_scored': last_n['GoalsScored'].mean(),
            f'{prefix}_form_goals_conceded': last_n['GoalsConceded'].mean(),
        }
    
    def _calculate_head_to_head(self, df: pd.DataFrame, home_team: str, 
                                 away_team: str, date: pd.Timestamp) -> dict:
        """Calculate head-to-head statistics"""
        # Matches between these two teams
        h2h = df[
            ((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team) |
             (df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team)) &
            (df['Date'] < date)
        ].sort_values('Date', ascending=False).head(6)
        
        if len(h2h) == 0:
            return {
                'h2h_home_wins': 0,
                'h2h_away_wins': 0,
                'h2h_draws': 0,
                'h2h_total_goals': 0
            }
        
        home_wins = len(h2h[
            ((h2h['HomeTeam'] == home_team) & (h2h['FTR'] == 'H')) |
            ((h2h['AwayTeam'] == home_team) & (h2h['FTR'] == 'A'))
        ])
        
        away_wins = len(h2h[
            ((h2h['HomeTeam'] == away_team) & (h2h['FTR'] == 'H')) |
            ((h2h['AwayTeam'] == away_team) & (h2h['FTR'] == 'A'))
        ])
        
        return {
            'h2h_home_wins': home_wins / len(h2h),
            'h2h_away_wins': away_wins / len(h2h),
            'h2h_draws': (len(h2h) - home_wins - away_wins) / len(h2h),
            'h2h_total_goals': (h2h['FTHG'] + h2h['FTAG']).mean()
        }
    
    def _calculate_season_stats(self, df: pd.DataFrame, team: str, 
                                 date: pd.Timestamp, season: str) -> dict:
        """Calculate team's current season statistics"""
        season_df = df[(df['Season'] == season) & (df['Date'] < date)]
        
        home_matches = season_df[season_df['HomeTeam'] == team]
        away_matches = season_df[season_df['AwayTeam'] == team]
        
        total_matches = len(home_matches) + len(away_matches)
        
        if total_matches == 0:
            return {
                'season_points_per_game': 0,
                'season_goals_scored_per_game': 0,
                'season_goals_conceded_per_game': 0,
                'season_position_estimate': 10
            }
        
        # Calculate points
        home_points = home_matches['FTR'].map({'H': 3, 'D': 1, 'A': 0}).sum()
        away_points = away_matches['FTR'].map({'H': 0, 'D': 1, 'A': 3}).sum()
        total_points = home_points + away_points
        
        # Goals
        home_scored = home_matches['FTHG'].sum()
        away_scored = away_matches['FTAG'].sum()
        home_conceded = home_matches['FTAG'].sum()
        away_conceded = away_matches['FTHG'].sum()
        
        return {
            'season_points_per_game': total_points / total_matches,
            'season_goals_scored_per_game': (home_scored + away_scored) / total_matches,
            'season_goals_conceded_per_game': (home_conceded + away_conceded) / total_matches,
            'season_position_estimate': max(1, 20 - int(total_points / total_matches * 6))
        }
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features for the dataset"""
        df = df.copy()
        
        # Parse dates
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Create target variables
        df = self.create_target_variables(df)
        
        print("Creating features for each match...")
        features_list = []
        
        for idx, row in df.iterrows():
            if idx % 500 == 0:
                print(f"  Processing match {idx}/{len(df)}")
            
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            date = row['Date']
            season = row['Season']
            
            features = {}
            
            # Home team overall form
            home_form = self._calculate_team_form(df, home_team, date, is_home=True)
            for k, v in home_form.items():
                features[f'home_{k}'] = v
            
            # Away team overall form
            away_form = self._calculate_team_form(df, away_team, date, is_home=False)
            for k, v in away_form.items():
                features[f'away_{k}'] = v
            
            # Home-specific form for home team
            home_specific = self._calculate_home_away_form(df, home_team, date, is_home=True)
            for k, v in home_specific.items():
                features[f'home_team_{k}'] = v
            
            # Away-specific form for away team
            away_specific = self._calculate_home_away_form(df, away_team, date, is_home=False)
            for k, v in away_specific.items():
                features[f'away_team_{k}'] = v
            
            # Head to head
            h2h = self._calculate_head_to_head(df, home_team, away_team, date)
            features.update(h2h)
            
            # Season stats
            home_season = self._calculate_season_stats(df, home_team, date, season)
            for k, v in home_season.items():
                features[f'home_{k}'] = v
            
            away_season = self._calculate_season_stats(df, away_team, date, season)
            for k, v in away_season.items():
                features[f'away_{k}'] = v
            
            # Derived features
            features['form_diff'] = features['home_form_points'] - features['away_form_points']
            features['goals_diff'] = (features['home_form_goals_scored'] - features['away_form_goals_scored'])
            features['attack_strength_diff'] = (
                features['home_season_goals_scored_per_game'] - 
                features['away_season_goals_scored_per_game']
            )
            features['defense_strength_diff'] = (
                features['away_season_goals_conceded_per_game'] - 
                features['home_season_goals_conceded_per_game']
            )
            features['position_diff'] = (
                features['away_season_position_estimate'] - 
                features['home_season_position_estimate']
            )
            
            features_list.append(features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Combine with original data
        result = pd.concat([df.reset_index(drop=True), features_df], axis=1)
        
        print(f"✓ Created {len(features_df.columns)} features")
        return result
    
    def get_feature_columns(self) -> list:
        """Return list of feature column names used for prediction"""
        return [
            # Home team form
            'home_form_points', 'home_form_goals_scored', 'home_form_goals_conceded',
            'home_form_clean_sheets', 'home_form_wins', 'home_form_losses',
            # Away team form
            'away_form_points', 'away_form_goals_scored', 'away_form_goals_conceded',
            'away_form_clean_sheets', 'away_form_wins', 'away_form_losses',
            # Home/Away specific form
            'home_team_home_form_points', 'home_team_home_form_goals_scored', 
            'home_team_home_form_goals_conceded',
            'away_team_away_form_points', 'away_team_away_form_goals_scored', 
            'away_team_away_form_goals_conceded',
            # Head to head
            'h2h_home_wins', 'h2h_away_wins', 'h2h_draws', 'h2h_total_goals',
            # Season stats
            'home_season_points_per_game', 'home_season_goals_scored_per_game',
            'home_season_goals_conceded_per_game', 'home_season_position_estimate',
            'away_season_points_per_game', 'away_season_goals_scored_per_game',
            'away_season_goals_conceded_per_game', 'away_season_position_estimate',
            # Derived features
            'form_diff', 'goals_diff', 'attack_strength_diff', 
            'defense_strength_diff', 'position_diff'
        ]


if __name__ == "__main__":
    # Test feature engineering
    from data_fetcher import EPLDataFetcher
    
    fetcher = EPLDataFetcher(data_dir="../data")
    df = fetcher.get_or_fetch_data()
    
    engineer = FeatureEngineer(n_last_matches=5)
    df_with_features = engineer.create_features(df)
    
    print(f"\nDataset shape with features: {df_with_features.shape}")
    print(f"\nFeature columns: {engineer.get_feature_columns()}")
