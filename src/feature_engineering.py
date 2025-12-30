"""
Feature Engineering Module
Creates features for EPL match prediction
Enhanced with advanced predictive features
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from datetime import timedelta


class FeatureEngineer:
    """Creates features for football match prediction - Enhanced Version"""
    
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
    
    def _calculate_rest_days(self, df: pd.DataFrame, team: str, 
                              date: pd.Timestamp) -> dict:
        """Calculate days since last match (fatigue factor)"""
        # Find last match for this team
        last_match = df[
            ((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) &
            (df['Date'] < date)
        ].sort_values('Date', ascending=False).head(1)
        
        if len(last_match) == 0:
            return {'rest_days': 7}  # Default to a week
        
        days = (date - last_match['Date'].iloc[0]).days
        return {'rest_days': min(days, 30)}  # Cap at 30 days
    
    def _calculate_streak(self, df: pd.DataFrame, team: str, 
                          date: pd.Timestamp) -> dict:
        """Calculate current win/loss/unbeaten streak"""
        matches = df[
            ((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) &
            (df['Date'] < date)
        ].sort_values('Date', ascending=False).head(10)
        
        if len(matches) == 0:
            return {
                'win_streak': 0,
                'loss_streak': 0,
                'unbeaten_streak': 0,
                'winless_streak': 0
            }
        
        win_streak = 0
        loss_streak = 0
        unbeaten_streak = 0
        winless_streak = 0
        
        for _, match in matches.iterrows():
            is_home = match['HomeTeam'] == team
            result = match['FTR']
            
            won = (is_home and result == 'H') or (not is_home and result == 'A')
            lost = (is_home and result == 'A') or (not is_home and result == 'H')
            drew = result == 'D'
            
            # Win streak
            if won:
                win_streak += 1
            else:
                break
        
        # Reset and calculate loss streak
        for _, match in matches.iterrows():
            is_home = match['HomeTeam'] == team
            result = match['FTR']
            lost = (is_home and result == 'A') or (not is_home and result == 'H')
            if lost:
                loss_streak += 1
            else:
                break
        
        # Unbeaten streak
        for _, match in matches.iterrows():
            is_home = match['HomeTeam'] == team
            result = match['FTR']
            lost = (is_home and result == 'A') or (not is_home and result == 'H')
            if not lost:
                unbeaten_streak += 1
            else:
                break
        
        # Winless streak
        for _, match in matches.iterrows():
            is_home = match['HomeTeam'] == team
            result = match['FTR']
            won = (is_home and result == 'H') or (not is_home and result == 'A')
            if not won:
                winless_streak += 1
            else:
                break
        
        return {
            'win_streak': win_streak,
            'loss_streak': loss_streak,
            'unbeaten_streak': unbeaten_streak,
            'winless_streak': winless_streak
        }
    
    def _calculate_scoring_patterns(self, df: pd.DataFrame, team: str, 
                                     date: pd.Timestamp) -> dict:
        """Calculate team's scoring patterns"""
        matches = df[
            ((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) &
            (df['Date'] < date)
        ].sort_values('Date', ascending=False).head(10)
        
        if len(matches) == 0:
            return {
                'scored_first_rate': 0.5,
                'clean_sheet_rate': 0.2,
                'failed_to_score_rate': 0.2,
                'conceded_first_rate': 0.5,
                'btts_rate': 0.5,
                'over_2_5_rate': 0.5,
                'avg_goals_scored': 1.0,
                'avg_goals_conceded': 1.0,
                'first_half_goals_rate': 0.4,
                'second_half_goals_rate': 0.6
            }
        
        scored_first = 0
        clean_sheets = 0
        failed_to_score = 0
        conceded_first = 0
        btts = 0
        over_25 = 0
        total_scored = 0
        total_conceded = 0
        first_half_goals = 0
        total_goals = 0
        
        for _, match in matches.iterrows():
            is_home = match['HomeTeam'] == team
            
            if is_home:
                goals_for = match['FTHG']
                goals_against = match['FTAG']
                ht_goals_for = match.get('HTHG', 0) or 0
                ht_goals_against = match.get('HTAG', 0) or 0
            else:
                goals_for = match['FTAG']
                goals_against = match['FTHG']
                ht_goals_for = match.get('HTAG', 0) or 0
                ht_goals_against = match.get('HTHG', 0) or 0
            
            total_scored += goals_for
            total_conceded += goals_against
            
            # Scoring patterns
            if goals_against == 0:
                clean_sheets += 1
            if goals_for == 0:
                failed_to_score += 1
            if goals_for > 0 and goals_against > 0:
                btts += 1
            if goals_for + goals_against > 2.5:
                over_25 += 1
            
            # First half scoring
            if goals_for > 0:
                first_half_goals += ht_goals_for
                total_goals += goals_for
            
            # Scored/conceded first (estimate based on HT)
            if ht_goals_for > 0:
                scored_first += 1
            if ht_goals_against > 0:
                conceded_first += 1
        
        n = len(matches)
        return {
            'scored_first_rate': scored_first / n,
            'clean_sheet_rate': clean_sheets / n,
            'failed_to_score_rate': failed_to_score / n,
            'conceded_first_rate': conceded_first / n,
            'btts_rate': btts / n,
            'over_2_5_rate': over_25 / n,
            'avg_goals_scored': total_scored / n,
            'avg_goals_conceded': total_conceded / n,
            'first_half_goals_rate': first_half_goals / max(total_goals, 1),
            'second_half_goals_rate': 1 - (first_half_goals / max(total_goals, 1))
        }
    
    def _calculate_goal_timing(self, df: pd.DataFrame, team: str, 
                                date: pd.Timestamp) -> dict:
        """Calculate goal timing patterns (first half vs second half)"""
        matches = df[
            ((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) &
            (df['Date'] < date)
        ].sort_values('Date', ascending=False).head(10)
        
        if len(matches) == 0 or 'HTHG' not in matches.columns:
            return {
                'ht_scoring_rate': 0.4,
                'second_half_scoring_rate': 0.6,
                'ht_conceding_rate': 0.4,
                'second_half_conceding_rate': 0.6
            }
        
        ht_goals_for = 0
        sh_goals_for = 0
        ht_goals_against = 0
        sh_goals_against = 0
        
        for _, match in matches.iterrows():
            is_home = match['HomeTeam'] == team
            
            if is_home:
                ht_for = match.get('HTHG', 0) or 0
                ft_for = match['FTHG']
                ht_against = match.get('HTAG', 0) or 0
                ft_against = match['FTAG']
            else:
                ht_for = match.get('HTAG', 0) or 0
                ft_for = match['FTAG']
                ht_against = match.get('HTHG', 0) or 0
                ft_against = match['FTHG']
            
            ht_goals_for += ht_for
            sh_goals_for += (ft_for - ht_for)
            ht_goals_against += ht_against
            sh_goals_against += (ft_against - ht_against)
        
        total_for = ht_goals_for + sh_goals_for
        total_against = ht_goals_against + sh_goals_against
        
        return {
            'ht_scoring_rate': ht_goals_for / max(total_for, 1),
            'second_half_scoring_rate': sh_goals_for / max(total_for, 1),
            'ht_conceding_rate': ht_goals_against / max(total_against, 1),
            'second_half_conceding_rate': sh_goals_against / max(total_against, 1)
        }
    
    def _calculate_momentum(self, df: pd.DataFrame, team: str, 
                            date: pd.Timestamp) -> dict:
        """Calculate team momentum (weighted recent form)"""
        matches = df[
            ((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) &
            (df['Date'] < date)
        ].sort_values('Date', ascending=False).head(5)
        
        if len(matches) == 0:
            return {
                'momentum_score': 0.5,
                'recent_form_weighted': 0.5
            }
        
        # Exponential weighting - more recent matches count more
        weights = [1.0, 0.8, 0.6, 0.4, 0.2]
        weighted_points = 0
        total_weight = 0
        
        for i, (_, match) in enumerate(matches.iterrows()):
            if i >= len(weights):
                break
                
            is_home = match['HomeTeam'] == team
            result = match['FTR']
            
            if (is_home and result == 'H') or (not is_home and result == 'A'):
                points = 3
            elif result == 'D':
                points = 1
            else:
                points = 0
            
            weighted_points += points * weights[i]
            total_weight += 3 * weights[i]  # Max points * weight
        
        momentum = weighted_points / max(total_weight, 1)
        
        return {
            'momentum_score': momentum,
            'recent_form_weighted': momentum
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
            
            # === NEW ADVANCED FEATURES ===
            
            # Rest days (fatigue)
            home_rest = self._calculate_rest_days(df, home_team, date)
            features['home_rest_days'] = home_rest['rest_days']
            
            away_rest = self._calculate_rest_days(df, away_team, date)
            features['away_rest_days'] = away_rest['rest_days']
            
            # Streaks
            home_streak = self._calculate_streak(df, home_team, date)
            for k, v in home_streak.items():
                features[f'home_{k}'] = v
            
            away_streak = self._calculate_streak(df, away_team, date)
            for k, v in away_streak.items():
                features[f'away_{k}'] = v
            
            # Scoring patterns
            home_scoring = self._calculate_scoring_patterns(df, home_team, date)
            for k, v in home_scoring.items():
                features[f'home_{k}'] = v
            
            away_scoring = self._calculate_scoring_patterns(df, away_team, date)
            for k, v in away_scoring.items():
                features[f'away_{k}'] = v
            
            # Goal timing
            home_timing = self._calculate_goal_timing(df, home_team, date)
            for k, v in home_timing.items():
                features[f'home_{k}'] = v
            
            away_timing = self._calculate_goal_timing(df, away_team, date)
            for k, v in away_timing.items():
                features[f'away_{k}'] = v
            
            # Momentum
            home_momentum = self._calculate_momentum(df, home_team, date)
            for k, v in home_momentum.items():
                features[f'home_{k}'] = v
            
            away_momentum = self._calculate_momentum(df, away_team, date)
            for k, v in away_momentum.items():
                features[f'away_{k}'] = v
            
            # === DERIVED FEATURES ===
            
            # Basic diffs
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
            
            # Advanced diffs
            features['rest_diff'] = features['home_rest_days'] - features['away_rest_days']
            features['momentum_diff'] = features['home_momentum_score'] - features['away_momentum_score']
            features['streak_diff'] = features['home_win_streak'] - features['away_win_streak']
            features['clean_sheet_diff'] = features['home_clean_sheet_rate'] - features['away_clean_sheet_rate']
            features['scoring_rate_diff'] = features['home_avg_goals_scored'] - features['away_avg_goals_scored']
            
            # Combined strength scores
            features['home_attack_defense'] = (
                features['home_avg_goals_scored'] - features['home_avg_goals_conceded']
            )
            features['away_attack_defense'] = (
                features['away_avg_goals_scored'] - features['away_avg_goals_conceded']
            )
            
            # Expected goals proxy
            features['expected_total_goals'] = (
                features['home_avg_goals_scored'] + features['away_avg_goals_scored'] +
                features['home_avg_goals_conceded'] + features['away_avg_goals_conceded']
            ) / 2
            
            # BTTS likelihood
            features['btts_combined'] = (
                (1 - features['home_clean_sheet_rate']) * (1 - features['away_failed_to_score_rate']) +
                (1 - features['away_clean_sheet_rate']) * (1 - features['home_failed_to_score_rate'])
            ) / 2
            
            features_list.append(features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Combine with original data
        result = pd.concat([df.reset_index(drop=True), features_df], axis=1)
        
        print(f"✓ Created {len(features_df.columns)} features")
        return result
    
    def get_feature_columns(self) -> list:
        """Return list of feature column names used for prediction - ENHANCED"""
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
            
            # === NEW FEATURES ===
            # Rest days
            'home_rest_days', 'away_rest_days',
            # Streaks
            'home_win_streak', 'home_loss_streak', 'home_unbeaten_streak', 'home_winless_streak',
            'away_win_streak', 'away_loss_streak', 'away_unbeaten_streak', 'away_winless_streak',
            # Scoring patterns
            'home_scored_first_rate', 'home_clean_sheet_rate', 'home_failed_to_score_rate',
            'home_btts_rate', 'home_over_2_5_rate', 'home_avg_goals_scored', 'home_avg_goals_conceded',
            'away_scored_first_rate', 'away_clean_sheet_rate', 'away_failed_to_score_rate',
            'away_btts_rate', 'away_over_2_5_rate', 'away_avg_goals_scored', 'away_avg_goals_conceded',
            # Goal timing
            'home_ht_scoring_rate', 'home_second_half_scoring_rate',
            'away_ht_scoring_rate', 'away_second_half_scoring_rate',
            # Momentum
            'home_momentum_score', 'away_momentum_score',
            
            # === DERIVED FEATURES ===
            'form_diff', 'goals_diff', 'attack_strength_diff', 
            'defense_strength_diff', 'position_diff',
            # Advanced derived
            'rest_diff', 'momentum_diff', 'streak_diff', 'clean_sheet_diff', 'scoring_rate_diff',
            'home_attack_defense', 'away_attack_defense',
            'expected_total_goals', 'btts_combined'
        ]
    
    def get_odds_feature_columns(self) -> list:
        """Return list of odds-based feature columns (for enhanced predictions)"""
        return [
            'odds_home_implied_prob',
            'odds_draw_implied_prob',
            'odds_away_implied_prob',
            'odds_over25_implied_prob',
            'odds_under25_implied_prob',
            'odds_home_value',  # Our model vs bookmaker
            'odds_away_value',
            'odds_draw_value',
            'odds_margin',  # Bookmaker margin
        ]
    
    def add_odds_features(self, match_data: Dict, odds_data: Dict) -> Dict:
        """
        Add odds-based features to match data for live prediction
        
        Args:
            match_data: Dictionary of match features
            odds_data: Dictionary with bookmaker odds
                Expected format: {
                    'home_odds': 2.1,
                    'draw_odds': 3.4,
                    'away_odds': 3.2,
                    'over25_odds': 1.85,  # Optional
                    'under25_odds': 2.0    # Optional
                }
        
        Returns:
            match_data with added odds features
        """
        if not odds_data:
            # Return defaults if no odds available
            match_data.update({
                'odds_home_implied_prob': 0.4,
                'odds_draw_implied_prob': 0.27,
                'odds_away_implied_prob': 0.33,
                'odds_over25_implied_prob': 0.5,
                'odds_under25_implied_prob': 0.5,
                'odds_home_value': 0,
                'odds_away_value': 0,
                'odds_draw_value': 0,
                'odds_margin': 0.05,
            })
            return match_data
        
        # Convert odds to implied probabilities
        home_odds = odds_data.get('home_odds', 2.0)
        draw_odds = odds_data.get('draw_odds', 3.5)
        away_odds = odds_data.get('away_odds', 3.0)
        
        # Implied probabilities (1/odds)
        home_prob = 1 / home_odds if home_odds > 0 else 0.4
        draw_prob = 1 / draw_odds if draw_odds > 0 else 0.27
        away_prob = 1 / away_odds if away_odds > 0 else 0.33
        
        # Normalize to remove margin
        total_prob = home_prob + draw_prob + away_prob
        margin = total_prob - 1  # Bookmaker margin
        
        # Fair probabilities
        fair_home = home_prob / total_prob
        fair_draw = draw_prob / total_prob
        fair_away = away_prob / total_prob
        
        # Over/Under 2.5 (if available)
        over25_odds = odds_data.get('over25_odds', 1.9)
        under25_odds = odds_data.get('under25_odds', 1.9)
        
        over25_prob = 1 / over25_odds if over25_odds > 0 else 0.5
        under25_prob = 1 / under25_odds if under25_odds > 0 else 0.5
        
        total_ou = over25_prob + under25_prob
        fair_over25 = over25_prob / total_ou if total_ou > 0 else 0.5
        fair_under25 = under25_prob / total_ou if total_ou > 0 else 0.5
        
        # Calculate value (our prediction - market probability)
        # Positive value = we think it's more likely than market
        our_home_prob = match_data.get('home_form_wins', 0.4)
        our_away_prob = match_data.get('away_form_wins', 0.3)
        our_draw_prob = 1 - our_home_prob - our_away_prob
        
        match_data.update({
            'odds_home_implied_prob': fair_home,
            'odds_draw_implied_prob': fair_draw,
            'odds_away_implied_prob': fair_away,
            'odds_over25_implied_prob': fair_over25,
            'odds_under25_implied_prob': fair_under25,
            'odds_home_value': our_home_prob - fair_home,
            'odds_away_value': our_away_prob - fair_away,
            'odds_draw_value': our_draw_prob - fair_draw,
            'odds_margin': margin,
        })
        
        return match_data
    
    def calculate_elo_rating(self, df: pd.DataFrame, team: str, 
                             date: pd.Timestamp, k_factor: float = 32) -> float:
        """
        Calculate ELO rating for a team
        
        Args:
            df: Historical match data
            team: Team name
            date: Date to calculate ELO up to
            k_factor: ELO K-factor (higher = more responsive)
        
        Returns:
            ELO rating (1500 is average)
        """
        matches = df[
            ((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) &
            (df['Date'] < date)
        ].sort_values('Date')
        
        if len(matches) == 0:
            return 1500.0  # Default ELO
        
        elo = 1500.0
        
        for _, match in matches.tail(20).iterrows():  # Last 20 matches
            is_home = match['HomeTeam'] == team
            opponent = match['AwayTeam'] if is_home else match['HomeTeam']
            result = match['FTR']
            
            # Estimate opponent ELO (simplified)
            opponent_elo = 1500.0  # Could be recursive but expensive
            
            # Expected score
            expected = 1 / (1 + 10 ** ((opponent_elo - elo) / 400))
            
            # Actual score
            if (is_home and result == 'H') or (not is_home and result == 'A'):
                actual = 1.0
            elif result == 'D':
                actual = 0.5
            else:
                actual = 0.0
            
            # Home advantage
            if is_home:
                expected += 0.05  # Home teams have ~5% advantage
            
            # Update ELO
            elo += k_factor * (actual - expected)
        
        return elo


if __name__ == "__main__":
    # Test feature engineering
    from data_fetcher import EPLDataFetcher
    
    fetcher = EPLDataFetcher(data_dir="../data")
    df = fetcher.get_or_fetch_data()
    
    engineer = FeatureEngineer(n_last_matches=5)
    df_with_features = engineer.create_features(df)
    
    print(f"\nDataset shape with features: {df_with_features.shape}")
    print(f"\nFeature columns: {engineer.get_feature_columns()}")
