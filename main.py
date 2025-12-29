"""
EPL Match Prediction System
Main entry point for training and making predictions
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.openfootball_fetcher import OpenFootballFetcher
from src.feature_engineering import FeatureEngineer
from src.models import EPLPredictor
import pandas as pd
import argparse


class EPLPredictionSystem:
    """Main system for EPL match predictions"""
    
    def __init__(self, data_dir: str = "data", models_dir: str = "models", league: str = "EPL"):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.league = league
        
        self.fetcher = OpenFootballFetcher(data_dir=data_dir)
        self.engineer = FeatureEngineer(n_last_matches=5)
        self.predictor = EPLPredictor(models_dir=models_dir)
        
        self.df_with_features = None
        self.is_trained = False
    
    def fetch_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """Fetch league historical data"""
        print("\n" + "="*60)
        print("STEP 1: FETCHING DATA")
        print("="*60)
        return self.fetcher.get_or_fetch_league_data(self.league, force_refresh=force_refresh)
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for prediction"""
        print("\n" + "="*60)
        print("STEP 2: ENGINEERING FEATURES")
        print("="*60)
        self.df_with_features = self.engineer.create_features(df)
        return self.df_with_features
    
    def train_models(self, df: pd.DataFrame = None):
        """Train all prediction models"""
        print("\n" + "="*60)
        print("STEP 3: TRAINING MODELS")
        print("="*60)
        
        if df is None:
            df = self.df_with_features
        
        if df is None:
            raise ValueError("No data available. Run fetch_data and engineer_features first.")
        
        feature_cols = self.engineer.get_feature_columns()
        df_clean = self.predictor.prepare_data(df, feature_cols)
        
        print(f"\nTraining on {len(df_clean)} matches...")
        results = self.predictor.train(df_clean, feature_cols)
        
        # Save models
        self.predictor.save_models()
        self.is_trained = True
        
        return results
    
    def load_trained_models(self):
        """Load previously trained models"""
        self.predictor.load_models(prefix=self.league.lower())
        self.is_trained = True
    
    def predict_upcoming_match(self, home_team: str, away_team: str) -> dict:
        """
        Predict an upcoming match between two teams
        Uses their most recent form data
        """
        if not self.is_trained:
            try:
                self.load_trained_models()
            except:
                raise ValueError("No trained models available. Run train_models first.")
        
        if self.df_with_features is None:
            # Load data for current season stats
            df = self.fetch_data(force_refresh=False)
            self.df_with_features = self.engineer.create_features(df)
        
        # Get latest stats for both teams
        df = self.df_with_features.copy()
        
        # Normalize team names
        home_team_normalized = self._normalize_team_name(home_team, df)
        away_team_normalized = self._normalize_team_name(away_team, df)
        
        if home_team_normalized is None:
            available = df['HomeTeam'].unique().tolist()
            raise ValueError(f"Team '{home_team}' not found. Available teams: {available}")
        
        if away_team_normalized is None:
            available = df['AwayTeam'].unique().tolist()
            raise ValueError(f"Team '{away_team}' not found. Available teams: {available}")
        
        # Get the most recent match for each team to extract their latest stats
        home_latest = df[
            (df['HomeTeam'] == home_team_normalized) | 
            (df['AwayTeam'] == home_team_normalized)
        ].iloc[-1]
        
        away_latest = df[
            (df['HomeTeam'] == away_team_normalized) | 
            (df['AwayTeam'] == away_team_normalized)
        ].iloc[-1]
        
        # Build feature vector
        feature_cols = self.engineer.get_feature_columns()
        features = []
        
        for col in feature_cols:
            if col in home_latest.index:
                features.append(home_latest[col])
            elif col in away_latest.index:
                features.append(away_latest[col])
            else:
                features.append(0)
        
        import numpy as np
        features = np.array(features).reshape(1, -1)
        
        # Get predictions
        predictions = self.predictor.predict(features)
        
        return self._format_predictions(home_team_normalized, away_team_normalized, predictions)
    
    def _normalize_team_name(self, team: str, df: pd.DataFrame) -> str:
        """Try to match team name to dataset"""
        all_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
        
        # Exact match
        if team in all_teams:
            return team
        
        # Case-insensitive match
        team_lower = team.lower()
        for t in all_teams:
            if t.lower() == team_lower:
                return t
        
        # Partial match
        for t in all_teams:
            if team_lower in t.lower() or t.lower() in team_lower:
                return t
        
        # Common abbreviations
        abbreviations = {
            'man utd': 'Man United',
            'man united': 'Man United',
            'manchester united': 'Man United',
            'man city': 'Man City',
            'manchester city': 'Man City',
            'arsenal': 'Arsenal',
            'chelsea': 'Chelsea',
            'liverpool': 'Liverpool',
            'spurs': 'Tottenham',
            'tottenham': 'Tottenham',
            'newcastle': 'Newcastle',
            'west ham': 'West Ham',
            'aston villa': 'Aston Villa',
            'brighton': 'Brighton',
            'wolves': 'Wolves',
            'everton': 'Everton',
            'crystal palace': 'Crystal Palace',
            'brentford': 'Brentford',
            'fulham': 'Fulham',
            'bournemouth': 'Bournemouth',
            'nottingham': "Nott'm Forest",
            'nottingham forest': "Nott'm Forest",
            'forest': "Nott'm Forest",
            'leicester': 'Leicester',
            'southampton': 'Southampton',
            'ipswich': 'Ipswich',
            'leeds': 'Leeds',
            'luton': 'Luton',
            'sheffield': 'Sheffield United',
            'burnley': 'Burnley'
        }
        
        if team_lower in abbreviations:
            abbrev_name = abbreviations[team_lower]
            for t in all_teams:
                if abbrev_name.lower() in t.lower():
                    return t
        
        return None
    
    def _format_predictions(self, home_team: str, away_team: str, predictions: dict) -> dict:
        """Format predictions for display"""
        result = {
            'match': f"{home_team} vs {away_team}",
            'predictions': {}
        }
        
        # Match result
        match_pred = predictions['match_result']
        result['predictions']['match_result'] = {
            'prediction': match_pred['prediction'][0],
            'probabilities': {
                'Home Win': f"{match_pred['probabilities'][0][0]*100:.1f}%",
                'Draw': f"{match_pred['probabilities'][0][1]*100:.1f}%",
                'Away Win': f"{match_pred['probabilities'][0][2]*100:.1f}%"
            },
            'confidence': f"{match_pred['confidence'][0]*100:.1f}%"
        }
        
        # Over/Under 1.5
        over15 = predictions['over_1.5']
        result['predictions']['goals_1.5'] = {
            'prediction': over15['prediction'][0],
            'over_probability': f"{over15['probabilities'][0][1]*100:.1f}%",
            'under_probability': f"{over15['probabilities'][0][0]*100:.1f}%",
            'confidence': f"{over15['confidence'][0]*100:.1f}%"
        }
        
        # Over/Under 2.5
        over25 = predictions['over_2.5']
        result['predictions']['goals_2.5'] = {
            'prediction': over25['prediction'][0],
            'over_probability': f"{over25['probabilities'][0][1]*100:.1f}%",
            'under_probability': f"{over25['probabilities'][0][0]*100:.1f}%",
            'confidence': f"{over25['confidence'][0]*100:.1f}%"
        }
        
        return result
    
    def print_prediction(self, prediction: dict):
        """Pretty print a prediction"""
        print("\n" + "="*60)
        print(f"PREDICTION: {prediction['match']}")
        print("="*60)
        
        # Match result
        mr = prediction['predictions']['match_result']
        print(f"\n🏆 MATCH RESULT: {mr['prediction']}")
        print(f"   Confidence: {mr['confidence']}")
        print(f"   Probabilities:")
        for outcome, prob in mr['probabilities'].items():
            print(f"      {outcome}: {prob}")
        
        # Goals 1.5
        g15 = prediction['predictions']['goals_1.5']
        print(f"\n⚽ GOALS 1.5: {g15['prediction']}")
        print(f"   Over 1.5: {g15['over_probability']}")
        print(f"   Under 1.5: {g15['under_probability']}")
        
        # Goals 2.5
        g25 = prediction['predictions']['goals_2.5']
        print(f"\n⚽ GOALS 2.5: {g25['prediction']}")
        print(f"   Over 2.5: {g25['over_probability']}")
        print(f"   Under 2.5: {g25['under_probability']}")
    
    def run_full_pipeline(self, force_refresh: bool = False):
        """Run the complete training pipeline"""
        df = self.fetch_data(force_refresh=force_refresh)
        df_features = self.engineer_features(df)
        results = self.train_models(df_features)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        
        for target, result in results.items():
            print(f"\n{target}: Accuracy = {result['accuracy']*100:.2f}%")
        
        return results
    
    def get_available_teams(self) -> list:
        """Get list of available teams in the dataset"""
        if self.df_with_features is None:
            df = self.fetch_data()
            self.df_with_features = self.engineer.create_features(df)
        
        teams = sorted(set(self.df_with_features['HomeTeam'].unique()) | 
                      set(self.df_with_features['AwayTeam'].unique()))
        return teams


def main():
    parser = argparse.ArgumentParser(description='EPL Match Prediction System')
    parser.add_argument('--train', action='store_true', help='Train the models')
    parser.add_argument('--predict', nargs=2, metavar=('HOME', 'AWAY'), 
                        help='Predict a match (e.g., --predict Arsenal Chelsea)')
    parser.add_argument('--teams', action='store_true', help='List available teams')
    parser.add_argument('--refresh', action='store_true', help='Force refresh data')
    
    args = parser.parse_args()
    
    system = EPLPredictionSystem()
    
    if args.teams:
        print("\nAvailable teams:")
        for team in system.get_available_teams():
            print(f"  - {team}")
        return
    
    if args.train:
        system.run_full_pipeline(force_refresh=args.refresh)
    
    if args.predict:
        home_team, away_team = args.predict
        try:
            prediction = system.predict_upcoming_match(home_team, away_team)
            system.print_prediction(prediction)
        except ValueError as e:
            print(f"\nError: {e}")
    
    if not any([args.train, args.predict, args.teams]):
        # Default: run full pipeline
        system.run_full_pipeline()
        
        # Show example prediction
        print("\n" + "="*60)
        print("EXAMPLE PREDICTION")
        print("="*60)
        try:
            prediction = system.predict_upcoming_match("Arsenal", "Chelsea")
            system.print_prediction(prediction)
        except Exception as e:
            print(f"Could not make example prediction: {e}")


if __name__ == "__main__":
    main()
