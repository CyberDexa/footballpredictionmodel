"""
Automated Prediction Scheduler
Automatically generates predictions for upcoming matches
"""

import os
import sys
import time
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.odds_api import get_odds_api
from src.database import Database
from src.models import EPLPredictor
from src.openfootball_fetcher import OpenFootballFetcher
from src.feature_engineering import FeatureEngineer

# Create logs directory
os.makedirs('logs', exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/scheduler.log', mode='a')
    ]
)
logger = logging.getLogger('PredictionScheduler')


class PredictionScheduler:
    """
    Automated scheduler for football predictions.
    Fetches upcoming fixtures and generates predictions automatically.
    """
    
    # Leagues to auto-predict (must be supported by both Odds API and our model)
    SUPPORTED_LEAGUES = [
        'EPL', 'CHAMPIONSHIP', 'LA_LIGA', 'SERIE_A', 'BUNDESLIGA', 'LIGUE_1',
        'EREDIVISIE', 'PRIMEIRA_LIGA', 'SCOTTISH_PREM'
    ]
    
    def __init__(self, days_ahead: int = 2, check_interval_hours: int = 6):
        """
        Initialize the scheduler.
        
        Args:
            days_ahead: How many days before a match to generate predictions (default: 2)
            check_interval_hours: How often to check for new fixtures (default: 6 hours)
        """
        self.days_ahead = days_ahead
        self.check_interval = check_interval_hours * 3600  # Convert to seconds
        self.odds_api = get_odds_api()
        self.db = Database()
        self.fetcher = OpenFootballFetcher(data_dir="data")
        self.engineer = FeatureEngineer(n_last_matches=5)
        self.predictor = EPLPredictor(models_dir="models")
        self._running = False
        self._thread = None
        self._features_cache = {}
        
    def _load_league_features(self, league: str):
        """Load and cache features for a league"""
        if league in self._features_cache:
            return self._features_cache[league]
        
        try:
            # Load match data
            df = self.fetcher.load_league_data(league)
            if df is None or df.empty:
                logger.warning(f"No data available for {league}")
                return None
            
            # Create features
            df_features = self.engineer.create_features(df)
            self._features_cache[league] = df_features
            
            # Get list of teams
            teams = sorted(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))
            
            logger.info(f"Loaded {league} with {len(df_features)} matches, {len(teams)} teams")
            return df_features
            
        except Exception as e:
            logger.error(f"Error loading features for {league}: {e}")
            return None
    
    def _match_team_name(self, api_name: str, teams: List[str]) -> Optional[str]:
        """Match an API team name to a team in our dataset"""
        api_lower = api_name.lower().strip()
        
        for team in teams:
            team_lower = team.lower().strip()
            # Direct match
            if api_lower == team_lower:
                return team
            # Contains match
            if api_lower in team_lower or team_lower in api_lower:
                return team
            # Word match (e.g., "Manchester United" matches "Man United")
            api_words = set(api_lower.split())
            team_words = set(team_lower.split())
            if len(api_words & team_words) >= 1 and (
                'united' in api_words or 'city' in api_words or 
                'fc' in api_words or len(api_words & team_words) >= 2
            ):
                return team
        
        return None
    
    def get_upcoming_fixtures(self, league: str, days_ahead: int = None) -> List[Dict]:
        """
        Fetch upcoming fixtures from The Odds API.
        
        Args:
            league: League code (EPL, LA_LIGA, etc.)
            days_ahead: Only return matches within this many days
            
        Returns:
            List of upcoming fixtures
        """
        if days_ahead is None:
            days_ahead = self.days_ahead
            
        if not self.odds_api.is_configured():
            logger.warning("Odds API not configured - cannot fetch fixtures")
            return []
        
        try:
            # Use the dedicated fixtures endpoint (FREE - no quota cost)
            fixtures = self.odds_api.get_upcoming_fixtures(league, days_ahead)
            logger.info(f"Found {len(fixtures)} fixtures for {league} in next {days_ahead} days")
            return fixtures
            
        except Exception as e:
            logger.error(f"Error fetching fixtures for {league}: {e}")
            return []
    
    def has_prediction(self, home_team: str, away_team: str, match_date: str = None) -> bool:
        """Check if a prediction already exists for this match"""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        
        # Check for existing prediction with same teams (fuzzy match)
        home_pattern = f'%{home_team.lower()[:15]}%'
        away_pattern = f'%{away_team.lower()[:15]}%'
        
        cursor.execute('''
            SELECT COUNT(*) FROM predictions
            WHERE LOWER(home_team) LIKE ? 
            AND LOWER(away_team) LIKE ?
            AND match_played = FALSE
        ''', (home_pattern, away_pattern))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count > 0
    
    def generate_prediction(self, league: str, home_team: str, away_team: str, 
                          df_features, match_date: str = None) -> Optional[Dict]:
        """
        Generate a prediction for a match using our model.
        
        Args:
            league: League code
            home_team: Home team name (from our dataset)
            away_team: Away team name (from our dataset)
            df_features: DataFrame with engineered features
            match_date: Match date string
            
        Returns:
            Prediction data dict or None if failed
        """
        try:
            feature_cols = self.engineer.get_feature_columns()
            
            # Get latest data for each team
            home_matches = df_features[
                (df_features['HomeTeam'] == home_team) | (df_features['AwayTeam'] == home_team)
            ]
            away_matches = df_features[
                (df_features['HomeTeam'] == away_team) | (df_features['AwayTeam'] == away_team)
            ]
            
            if home_matches.empty or away_matches.empty:
                logger.warning(f"No match data for {home_team} or {away_team}")
                return None
            
            home_latest = home_matches.iloc[-1]
            away_latest = away_matches.iloc[-1]
            
            # Build feature vector
            features = []
            for col in feature_cols:
                if col in home_latest.index:
                    val = home_latest[col]
                    features.append(float(val) if not (isinstance(val, float) and np.isnan(val)) else 0)
                elif col in away_latest.index:
                    val = away_latest[col]
                    features.append(float(val) if not (isinstance(val, float) and np.isnan(val)) else 0)
                else:
                    features.append(0)
            
            features = np.array(features).reshape(1, -1)
            
            # Get predictions from model
            predictions = self.predictor.predict(features)
            
            if predictions is None:
                return None
            
            # Extract match result probabilities
            match_result = predictions.get('match_result', {})
            probs = match_result.get('probabilities', [[0.33, 0.34, 0.33]])[0]
            home_prob, draw_prob, away_prob = probs[0], probs[1], probs[2]
            
            # Determine predicted result
            if home_prob >= draw_prob and home_prob >= away_prob:
                predicted_result = 'Home Win'
                confidence = home_prob
            elif away_prob >= draw_prob:
                predicted_result = 'Away Win'
                confidence = away_prob
            else:
                predicted_result = 'Draw'
                confidence = draw_prob
            
            # Extract other predictions
            over_2_5 = predictions.get('over_2.5', {})
            over_2_5_prob = over_2_5.get('probabilities', [[0.5, 0.5]])[0][1] if over_2_5 else 0.5
            
            btts = predictions.get('btts', {})
            btts_prob = btts.get('probabilities', [[0.5, 0.5]])[0][1] if btts else 0.5
            
            over_1_5 = predictions.get('over_1.5', {})
            over_1_5_prob = over_1_5.get('probabilities', [[0.5, 0.5]])[0][1] if over_1_5 else 0.5
            
            over_3_5 = predictions.get('over_3.5', {})
            over_3_5_prob = over_3_5.get('probabilities', [[0.5, 0.5]])[0][1] if over_3_5 else 0.5
            
            return {
                'league': league,
                'home_team': home_team,
                'away_team': away_team,
                'match_date': match_date,
                'home_win_prob': home_prob,
                'draw_prob': draw_prob,
                'away_win_prob': away_prob,
                'predicted_result': predicted_result,
                'result_confidence': confidence,
                'over_1_5_prob': over_1_5_prob,
                'over_2_5_prob': over_2_5_prob,
                'over_3_5_prob': over_3_5_prob,
                'btts_prob': btts_prob,
                'auto_generated': True
            }
            
        except Exception as e:
            logger.error(f"Error generating prediction for {home_team} vs {away_team}: {e}")
            return None
    
    def save_prediction(self, prediction_data: Dict) -> bool:
        """Save a prediction to the database"""
        try:
            # Format data for database
            db_data = {
                'league': prediction_data['league'],
                'home_team': prediction_data['home_team'],
                'away_team': prediction_data['away_team'],
                'match_date': prediction_data.get('match_date'),
                'home_win_prob': prediction_data['home_win_prob'],
                'draw_prob': prediction_data['draw_prob'],
                'away_win_prob': prediction_data['away_win_prob'],
                'predicted_result': prediction_data['predicted_result'],
                'result_confidence': prediction_data['result_confidence'],
                'over_1_5_prob': prediction_data.get('over_1_5_prob', 0),
                'over_2_5_prob': prediction_data.get('over_2_5_prob', 0),
                'over_3_5_prob': prediction_data.get('over_3_5_prob', 0),
                'btts_prob': prediction_data.get('btts_prob', 0),
                'home_over_0_5_prob': prediction_data.get('home_over_0_5_prob', 0),
                'home_over_1_5_prob': prediction_data.get('home_over_1_5_prob', 0),
                'away_over_0_5_prob': prediction_data.get('away_over_0_5_prob', 0),
                'away_over_1_5_prob': prediction_data.get('away_over_1_5_prob', 0),
                'ht_over_0_5_prob': prediction_data.get('ht_over_0_5_prob', 0),
                'ht_over_1_5_prob': prediction_data.get('ht_over_1_5_prob', 0),
                'goals_0_1_prob': prediction_data.get('goals_0_1_prob', 0),
                'goals_2_3_prob': prediction_data.get('goals_2_3_prob', 0),
                'goals_4_plus_prob': prediction_data.get('goals_4_plus_prob', 0),
            }
            
            self.db.save_prediction(db_data, user_id=0)  # System user (0 = auto-generated)
            
            logger.info(f"✓ Saved: {prediction_data['home_team']} vs {prediction_data['away_team']} -> {prediction_data['predicted_result']} ({prediction_data['result_confidence']*100:.1f}%)")
            return True
            
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")
            return False
    
    def run_predictions_for_league(self, league: str) -> Dict:
        """
        Run predictions for all upcoming fixtures in a league.
        
        Returns:
            Statistics about predictions made
        """
        stats = {
            'league': league,
            'fixtures_found': 0,
            'predictions_made': 0,
            'skipped': 0,
            'errors': 0,
            'no_team_match': 0
        }
        
        # Load features for this league
        df_features = self._load_league_features(league)
        if df_features is None:
            logger.warning(f"Skipping {league} - no data available")
            return stats
        
        # Get team list
        teams = sorted(set(df_features['HomeTeam'].unique()) | set(df_features['AwayTeam'].unique()))
        
        # Check if model is loaded
        if not self.predictor.models:
            try:
                self.predictor.load_models(league.lower())
            except Exception as e:
                logger.error(f"Could not load model for {league}: {e}")
                return stats
        
        # Get upcoming fixtures
        fixtures = self.get_upcoming_fixtures(league)
        stats['fixtures_found'] = len(fixtures)
        
        if not fixtures:
            logger.info(f"No upcoming fixtures for {league}")
            return stats
        
        logger.info(f"Processing {len(fixtures)} fixtures for {league}")
        
        for fixture in fixtures:
            try:
                api_home = fixture['home_team']
                api_away = fixture['away_team']
                match_date = fixture.get('match_date')
                
                # Match API team names to our dataset
                home_team = self._match_team_name(api_home, teams)
                away_team = self._match_team_name(api_away, teams)
                
                if not home_team or not away_team:
                    logger.debug(f"Could not match teams: {api_home} vs {api_away}")
                    stats['no_team_match'] += 1
                    continue
                
                # Check if already predicted
                if self.has_prediction(home_team, away_team, match_date):
                    logger.debug(f"Already predicted: {home_team} vs {away_team}")
                    stats['skipped'] += 1
                    continue
                
                # Generate prediction
                prediction = self.generate_prediction(
                    league, home_team, away_team, df_features, match_date
                )
                
                if prediction:
                    if self.save_prediction(prediction):
                        stats['predictions_made'] += 1
                    else:
                        stats['errors'] += 1
                else:
                    stats['errors'] += 1
                    
            except Exception as e:
                logger.error(f"Error processing fixture: {e}")
                stats['errors'] += 1
        
        return stats
    
    def run_all_predictions(self, leagues: List[str] = None) -> List[Dict]:
        """
        Run predictions for all supported leagues.
        
        Args:
            leagues: List of leagues to process (default: all supported)
            
        Returns:
            List of statistics per league
        """
        if leagues is None:
            leagues = self.SUPPORTED_LEAGUES
        
        all_stats = []
        total_made = 0
        
        logger.info(f"Starting prediction run for {len(leagues)} leagues...")
        
        for league in leagues:
            if league in self.SUPPORTED_LEAGUES or league in self.odds_api.SPORT_KEYS:
                logger.info(f"--- Processing {league} ---")
                stats = self.run_predictions_for_league(league)
                all_stats.append(stats)
                total_made += stats['predictions_made']
                
                # Small delay between leagues
                time.sleep(0.5)
        
        # Log summary
        total_fixtures = sum(s['fixtures_found'] for s in all_stats)
        total_skipped = sum(s['skipped'] for s in all_stats)
        total_errors = sum(s['errors'] for s in all_stats)
        
        logger.info(f"=== Prediction Run Complete ===")
        logger.info(f"Total: {total_made} new predictions from {total_fixtures} fixtures")
        logger.info(f"Skipped: {total_skipped}, Errors: {total_errors}")
        
        return all_stats
    
    def run_verification(self, leagues: List[str] = None) -> Dict:
        """Run verification of pending predictions"""
        if leagues is None:
            leagues = self.SUPPORTED_LEAGUES
        
        logger.info("Running prediction verification...")
        result = self.db.verify_predictions_with_odds_api(self.odds_api, leagues)
        logger.info(f"Verified {result.get('verified', 0)} predictions, {result.get('correct', 0)} correct")
        return result
    
    def _scheduler_loop(self):
        """Main scheduler loop (runs in background thread)"""
        logger.info("=== Prediction Scheduler Started ===")
        logger.info(f"Will check for fixtures every {self.check_interval/3600:.1f} hours")
        logger.info(f"Predictions generated {self.days_ahead} days before matches")
        
        while self._running:
            try:
                # Run predictions
                logger.info("Running scheduled predictions...")
                self.run_all_predictions()
                
                # Verify pending predictions
                self.run_verification()
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
            
            # Wait for next check
            logger.info(f"Sleeping for {self.check_interval/3600:.1f} hours...")
            for _ in range(int(self.check_interval)):
                if not self._running:
                    break
                time.sleep(1)
        
        logger.info("=== Prediction Scheduler Stopped ===")
    
    def start_background(self):
        """Start the scheduler in a background thread"""
        if self._running:
            logger.warning("Scheduler already running")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._thread.start()
        logger.info("Background scheduler started")
    
    def stop_background(self):
        """Stop the background scheduler"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Background scheduler stopped")
    
    def is_running(self) -> bool:
        """Check if scheduler is running"""
        return self._running


# Singleton instance
_scheduler = None

def get_scheduler(days_ahead: int = 2, check_interval_hours: int = 6) -> PredictionScheduler:
    """Get singleton scheduler instance"""
    global _scheduler
    if _scheduler is None:
        _scheduler = PredictionScheduler(days_ahead, check_interval_hours)
    return _scheduler


def main():
    """Main entry point for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Football Prediction Scheduler")
    parser.add_argument('--leagues', nargs='+', help='Leagues to process (default: all supported)')
    parser.add_argument('--days', type=int, default=2, help='Days ahead to predict (default: 2)')
    parser.add_argument('--daemon', action='store_true', help='Run as background daemon')
    parser.add_argument('--interval', type=int, default=6, help='Check interval in hours (default: 6)')
    parser.add_argument('--verify', action='store_true', help='Only run verification')
    
    args = parser.parse_args()
    
    scheduler = PredictionScheduler(days_ahead=args.days, check_interval_hours=args.interval)
    
    if args.verify:
        print("Running verification only...")
        result = scheduler.run_verification(args.leagues)
        print(f"\nVerification complete:")
        print(f"  Verified: {result.get('verified', 0)}")
        print(f"  Correct: {result.get('correct', 0)}")
        
    elif args.daemon:
        print("Starting scheduler daemon...")
        print(f"  Check interval: {args.interval} hours")
        print(f"  Predict {args.days} days ahead")
        print("Press Ctrl+C to stop")
        
        scheduler.start_background()
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("\nStopping scheduler...")
            scheduler.stop_background()
    else:
        print(f"Running predictions for matches in next {args.days} days...")
        stats = scheduler.run_all_predictions(args.leagues)
        
        print("\n=== Results ===")
        for s in stats:
            if s['fixtures_found'] > 0 or s['predictions_made'] > 0:
                print(f"{s['league']}: {s['predictions_made']} predictions ({s['fixtures_found']} fixtures, {s['skipped']} skipped, {s['no_team_match']} unmatched)")
        
        total = sum(s['predictions_made'] for s in stats)
        print(f"\nTotal new predictions: {total}")


if __name__ == "__main__":
    main()
