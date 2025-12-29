"""
Upcoming Fixtures Fetcher
Fetches upcoming/scheduled matches for prediction
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json


class UpcomingFixturesFetcher:
    """
    Fetches upcoming fixtures from OpenFootball and other sources.
    Provides matches ready for prediction.
    """
    
    # OpenFootball base URL
    BASE_URL = "https://raw.githubusercontent.com/openfootball/football.json/master"
    
    # League mappings
    LEAGUES = {
        'EPL': {'file': 'en.1.json', 'name': 'English Premier League', 'flag': '🏴󠁧󠁢󠁥󠁮󠁧󠁿'},
        'CHAMPIONSHIP': {'file': 'en.2.json', 'name': 'English Championship', 'flag': '🏴󠁧󠁢󠁥󠁮󠁧󠁿'},
        'LEAGUE_ONE': {'file': 'en.3.json', 'name': 'English League One', 'flag': '🏴󠁧󠁢󠁥󠁮󠁧󠁿'},
        'LEAGUE_TWO': {'file': 'en.4.json', 'name': 'English League Two', 'flag': '🏴󠁧󠁢󠁥󠁮󠁧󠁿'},
        'SCOTTISH_PREM': {'file': 'sco.1.json', 'name': 'Scottish Premiership', 'flag': '🏴󠁧󠁢󠁳󠁣󠁴󠁿'},
        'LA_LIGA': {'file': 'es.1.json', 'name': 'La Liga', 'flag': '🇪🇸'},
        'LA_LIGA_2': {'file': 'es.2.json', 'name': 'La Liga 2', 'flag': '🇪🇸'},
        'SERIE_A': {'file': 'it.1.json', 'name': 'Serie A', 'flag': '🇮🇹'},
        'SERIE_B': {'file': 'it.2.json', 'name': 'Serie B', 'flag': '🇮🇹'},
        'BUNDESLIGA': {'file': 'de.1.json', 'name': 'Bundesliga', 'flag': '🇩🇪'},
        'BUNDESLIGA_2': {'file': 'de.2.json', 'name': '2. Bundesliga', 'flag': '🇩🇪'},
        'LIGUE_1': {'file': 'fr.1.json', 'name': 'Ligue 1', 'flag': '🇫🇷'},
        'LIGUE_2': {'file': 'fr.2.json', 'name': 'Ligue 2', 'flag': '🇫🇷'},
        'EREDIVISIE': {'file': 'nl.1.json', 'name': 'Eredivisie', 'flag': '🇳🇱'},
        'PRIMEIRA_LIGA': {'file': 'pt.1.json', 'name': 'Primeira Liga', 'flag': '🇵🇹'},
        'BELGIAN_PRO': {'file': 'be.1.json', 'name': 'Belgian Pro League', 'flag': '🇧🇪'},
        'AUSTRIAN_BL': {'file': 'at.1.json', 'name': 'Austrian Bundesliga', 'flag': '🇦🇹'},
        'SUPER_LIG': {'file': 'tr.1.json', 'name': 'Süper Lig', 'flag': '🇹🇷'},
        'GREEK_SL': {'file': 'gr.1.json', 'name': 'Super League', 'flag': '🇬🇷'},
    }
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'FootballPredictionModel/1.0'
        })
    
    def get_current_season(self) -> str:
        """Get current season string (e.g., '2025-26')"""
        today = datetime.now()
        if today.month >= 8:  # Season starts in August
            return f"{today.year}-{str(today.year + 1)[2:]}"
        else:
            return f"{today.year - 1}-{str(today.year)[2:]}"
    
    def fetch_upcoming_fixtures(self, league: str, days_ahead: int = 14) -> pd.DataFrame:
        """
        Fetch upcoming fixtures for a league.
        
        Args:
            league: League code (e.g., 'EPL')
            days_ahead: Number of days ahead to fetch (default 14)
            
        Returns:
            DataFrame with upcoming fixtures
        """
        if league not in self.LEAGUES:
            raise ValueError(f"Unknown league: {league}")
        
        season = self.get_current_season()
        file_name = self.LEAGUES[league]['file']
        url = f"{self.BASE_URL}/{season}/{file_name}"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Error fetching fixtures: {e}")
            return pd.DataFrame()
        
        today = datetime.now().date()
        cutoff_date = today + timedelta(days=days_ahead)
        
        fixtures = []
        for match_round in data.get('rounds', []):
            round_name = match_round.get('name', 'Unknown Round')
            
            for match in match_round.get('matches', []):
                try:
                    match_date_str = match.get('date')
                    if not match_date_str:
                        continue
                    
                    match_date = datetime.strptime(match_date_str, '%Y-%m-%d').date()
                    
                    # Check if match is upcoming (no score yet or date in future)
                    score = match.get('score', {})
                    has_score = score and score.get('ft') is not None
                    
                    if match_date >= today and match_date <= cutoff_date and not has_score:
                        fixtures.append({
                            'Date': match_date_str,
                            'Round': round_name,
                            'HomeTeam': match['team1']['name'],
                            'AwayTeam': match['team2']['name'],
                            'Time': match.get('time', 'TBD'),
                            'League': league,
                            'LeagueName': self.LEAGUES[league]['name'],
                            'Flag': self.LEAGUES[league]['flag']
                        })
                except Exception:
                    continue
        
        df = pd.DataFrame(fixtures)
        if len(df) > 0:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
        
        return df
    
    def fetch_all_upcoming(self, days_ahead: int = 7) -> pd.DataFrame:
        """
        Fetch upcoming fixtures for all leagues.
        
        Args:
            days_ahead: Number of days ahead to fetch
            
        Returns:
            DataFrame with all upcoming fixtures across leagues
        """
        all_fixtures = []
        
        for league in self.LEAGUES.keys():
            try:
                df = self.fetch_upcoming_fixtures(league, days_ahead)
                if len(df) > 0:
                    all_fixtures.append(df)
            except Exception as e:
                print(f"Error fetching {league}: {e}")
                continue
        
        if all_fixtures:
            combined = pd.concat(all_fixtures, ignore_index=True)
            combined = combined.sort_values(['Date', 'League'])
            return combined
        
        return pd.DataFrame()
    
    def fetch_todays_matches(self) -> pd.DataFrame:
        """Fetch matches scheduled for today."""
        return self.fetch_all_upcoming(days_ahead=0)
    
    def fetch_weekend_matches(self) -> pd.DataFrame:
        """Fetch matches for the upcoming weekend (Fri-Sun)."""
        today = datetime.now().date()
        days_to_friday = (4 - today.weekday()) % 7
        days_to_sunday = days_to_friday + 2
        
        all_fixtures = self.fetch_all_upcoming(days_ahead=days_to_sunday + 1)
        
        if len(all_fixtures) == 0:
            return pd.DataFrame()
        
        friday = today + timedelta(days=days_to_friday)
        sunday = today + timedelta(days=days_to_sunday)
        
        weekend = all_fixtures[
            (all_fixtures['Date'].dt.date >= friday) & 
            (all_fixtures['Date'].dt.date <= sunday)
        ]
        
        return weekend
    
    def get_league_info(self, league: str) -> Dict:
        """Get information about a league."""
        return self.LEAGUES.get(league, {})
    
    def get_available_leagues(self) -> Dict:
        """Get all available leagues."""
        return self.LEAGUES
    
    def fetch_live_scores(self, league: str) -> pd.DataFrame:
        """
        Fetch current live scores (matches in progress).
        Note: OpenFootball doesn't have real-time data, so this returns
        today's matches that may be in progress.
        """
        today = datetime.now().date()
        all_today = self.fetch_upcoming_fixtures(league, days_ahead=0)
        
        # For true live scores, you'd need a real-time API
        # This is a placeholder showing today's scheduled matches
        return all_today


class FixturesManager:
    """
    Manages fixtures and provides predictions for upcoming matches.
    """
    
    def __init__(self):
        from src.openfootball_fetcher import OpenFootballFetcher
        from src.feature_engineering import FeatureEngineer
        from src.models import EPLPredictor
        
        self.fixtures_fetcher = UpcomingFixturesFetcher()
        self.data_fetcher = OpenFootballFetcher()
        self.engineer = FeatureEngineer(n_last_matches=5)
        self.predictor = EPLPredictor()
    
    def get_predictions_for_fixtures(self, league: str, days_ahead: int = 7) -> pd.DataFrame:
        """
        Get predictions for upcoming fixtures in a league.
        
        Args:
            league: League code
            days_ahead: Days ahead to fetch fixtures
            
        Returns:
            DataFrame with fixtures and their predictions
        """
        # Get upcoming fixtures
        fixtures = self.fixtures_fetcher.fetch_upcoming_fixtures(league, days_ahead)
        
        if len(fixtures) == 0:
            return pd.DataFrame()
        
        # Load models for this league
        try:
            self.predictor.load_models(prefix=league.lower())
        except:
            return fixtures  # Return fixtures without predictions if no model
        
        # Get historical data for feature calculation
        try:
            historical_df = self.data_fetcher.load_league_data(league)
            df_with_features = self.engineer.create_features(historical_df)
        except:
            return fixtures
        
        # Add predictions for each fixture
        predictions_list = []
        
        for _, fixture in fixtures.iterrows():
            home_team = fixture['HomeTeam']
            away_team = fixture['AwayTeam']
            
            try:
                # Get team stats from historical data
                pred = self._predict_match(df_with_features, home_team, away_team)
                predictions_list.append(pred)
            except Exception as e:
                predictions_list.append({
                    'home_win_prob': None,
                    'draw_prob': None,
                    'away_win_prob': None,
                    'over_2.5_prob': None,
                    'btts_prob': None,
                    'prediction': 'N/A'
                })
        
        # Merge predictions with fixtures
        pred_df = pd.DataFrame(predictions_list)
        result = pd.concat([fixtures.reset_index(drop=True), pred_df], axis=1)
        
        return result
    
    def _predict_match(self, df_with_features: pd.DataFrame, 
                       home_team: str, away_team: str) -> Dict:
        """Generate prediction for a single match."""
        import numpy as np
        
        feature_cols = self.engineer.get_feature_columns()
        
        # Get latest stats for home team
        home_matches = df_with_features[
            (df_with_features['HomeTeam'] == home_team) | 
            (df_with_features['AwayTeam'] == home_team)
        ]
        
        away_matches = df_with_features[
            (df_with_features['HomeTeam'] == away_team) | 
            (df_with_features['AwayTeam'] == away_team)
        ]
        
        if len(home_matches) == 0 or len(away_matches) == 0:
            raise ValueError("Team not found in historical data")
        
        # Use latest match data as proxy for team form
        home_latest = home_matches.iloc[-1]
        away_latest = away_matches.iloc[-1]
        
        # Build feature vector
        features = []
        for col in feature_cols:
            if col.startswith('home_'):
                features.append(home_latest.get(col, 0))
            elif col.startswith('away_'):
                features.append(away_latest.get(col, 0))
            else:
                features.append(home_latest.get(col, away_latest.get(col, 0)))
        
        features = np.array(features).reshape(1, -1)
        
        # Get predictions
        predictions = self.predictor.predict(features)
        
        # Extract probabilities
        result = {
            'prediction': predictions.get('match_result', {}).get('prediction', ['N/A'])[0],
            'confidence': predictions.get('match_result', {}).get('confidence', [0])[0],
        }
        
        # Add probabilities for each target
        if 'match_result' in predictions:
            probs = predictions['match_result'].get('probabilities', [[0, 0, 0]])[0]
            result['home_win_prob'] = probs[0] if len(probs) > 0 else 0
            result['draw_prob'] = probs[1] if len(probs) > 1 else 0
            result['away_win_prob'] = probs[2] if len(probs) > 2 else 0
        
        if 'over_2.5' in predictions:
            probs = predictions['over_2.5'].get('probabilities', [[0, 0]])[0]
            result['over_2.5_prob'] = probs[1] if len(probs) > 1 else 0
        
        if 'btts' in predictions:
            probs = predictions['btts'].get('probabilities', [[0, 0]])[0]
            result['btts_prob'] = probs[1] if len(probs) > 1 else 0
        
        return result


# Test
if __name__ == "__main__":
    fetcher = UpcomingFixturesFetcher()
    
    print("="*60)
    print("UPCOMING FIXTURES")
    print("="*60)
    
    # Fetch EPL fixtures
    fixtures = fetcher.fetch_upcoming_fixtures('EPL', days_ahead=14)
    print(f"\nEPL Upcoming ({len(fixtures)} matches):")
    
    if len(fixtures) > 0:
        for _, row in fixtures.head(10).iterrows():
            print(f"  {row['Date'].strftime('%Y-%m-%d')} | {row['HomeTeam']} vs {row['AwayTeam']}")
    else:
        print("  No upcoming fixtures found")
    
    # Fetch all weekend matches
    print("\n" + "="*60)
    print("WEEKEND FIXTURES (All Leagues)")
    print("="*60)
    
    weekend = fetcher.fetch_weekend_matches()
    if len(weekend) > 0:
        for league in weekend['League'].unique():
            league_fixtures = weekend[weekend['League'] == league]
            print(f"\n{fetcher.LEAGUES[league]['flag']} {fetcher.LEAGUES[league]['name']}:")
            for _, row in league_fixtures.head(5).iterrows():
                print(f"  {row['Date'].strftime('%a %d')} | {row['HomeTeam']} vs {row['AwayTeam']}")
