"""
Football Match Prediction App
A beautiful Streamlit UI for multi-league football predictions
Uses real data from OpenFootball (openfootball.github.io)

Features:
- 17 prediction targets (match result, goals, BTTS, half-time)
- Upcoming/Live matches auto-prediction
- 19 leagues across 11 countries
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
from typing import Tuple, List, Dict
from scipy.stats import poisson
import math

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.openfootball_fetcher import OpenFootballFetcher
from src.feature_engineering import FeatureEngineer
from src.models import EPLPredictor
from src.upcoming_fixtures import UpcomingFixturesFetcher, FixturesManager
from src.database import get_database
from src.value_bets import get_value_calculator
from src.auth import get_auth_manager
from src.payments import get_payment_manager
from src.odds_api import get_odds_api
from src.football_api import get_football_api
from src.thesportsdb_api import get_sportsdb_api

# Page configuration
st.set_page_config(
    page_title="⚽ Football Match Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .stat-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .win-home {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .win-away {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .win-draw {
        background: linear-gradient(135deg, #4568dc 0%, #b06ab3 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .btts-yes {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .goals-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .ht-card {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .upcoming-match {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .prob-bar {
        background: #e9ecef;
        border-radius: 10px;
        overflow: hidden;
        height: 25px;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    .api-key-box {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class FootballPredictorApp:
    """Main application class for the football predictor"""
    
    def __init__(self):
        # Initialize session state
        if 'trained_leagues' not in st.session_state:
            st.session_state.trained_leagues = {}
        if 'current_league' not in st.session_state:
            st.session_state.current_league = 'EPL'
        if 'features_cache' not in st.session_state:
            st.session_state.features_cache = {}
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True
        if 'last_refresh_check' not in st.session_state:
            st.session_state.last_refresh_check = {}
        if 'active_tab' not in st.session_state:
            st.session_state.active_tab = "predict"
        
        # Accumulator session state
        if 'accumulator' not in st.session_state:
            st.session_state.accumulator = []  # List of {match, selection, odds}
        
        # Auth session state
        if 'session_token' not in st.session_state:
            st.session_state.session_token = None
        if 'user' not in st.session_state:
            st.session_state.user = None
        if 'auth_mode' not in st.session_state:
            st.session_state.auth_mode = 'login'  # 'login' or 'register'
        
        # Initialize fetcher (no API key needed for OpenFootball!)
        self.fetcher = OpenFootballFetcher(data_dir="data")
        self.engineer = FeatureEngineer(n_last_matches=5)
        self.predictor = EPLPredictor(models_dir="models")
        self.fixtures_fetcher = UpcomingFixturesFetcher()
        self.fixtures_manager = FixturesManager()
        self.db = get_database()
        self.value_calculator = get_value_calculator()
        self.auth = get_auth_manager()
        self.payments = get_payment_manager()
        
        # External APIs (optional - will work without keys)
        self.odds_api = get_odds_api(os.getenv('ODDS_API_KEY'))
        self.football_api = get_football_api(os.getenv('FOOTBALL_API_KEY'))
        self.sportsdb = get_sportsdb_api()  # Free - no API key needed!
    
    def check_auto_refresh(self, league: str, max_age_days: int = 7):
        """Check if data needs auto-refresh and refresh if needed."""
        if not st.session_state.get('auto_refresh', True):
            return
        
        # Only check once per session per league
        if league in st.session_state.last_refresh_check:
            return
        
        if self.fetcher.needs_refresh(league, max_age_days):
            with st.spinner(f"📥 Auto-refreshing {league} data (older than {max_age_days} days)..."):
                try:
                    self.fetcher.get_or_fetch_league_data(league, force_refresh=True)
                    st.session_state.last_refresh_check[league] = True
                    st.toast(f"✅ {league} data refreshed!")
                except Exception as e:
                    st.warning(f"Could not refresh {league}: {e}")
        else:
            st.session_state.last_refresh_check[league] = True
    
    def load_or_train_model(self, league: str, force_retrain: bool = False):
        """Load existing model or train new one for a league"""
        model_path = f"models/{league.lower()}_match_result_model.joblib"
        
        if not force_retrain and os.path.exists(model_path):
            try:
                self.predictor.load_models(prefix=league.lower())
                return True
            except:
                pass
        
        # Need to train
        with st.spinner(f"🔄 Training models for {league}... This may take a minute."):
            try:
                # Fetch data
                df = self.fetcher.get_or_fetch_league_data(league)
                
                # Engineer features
                df_features = self.engineer.create_features(df)
                st.session_state.features_cache[league] = df_features
                
                # Train models
                feature_cols = self.engineer.get_feature_columns()
                df_clean = self.predictor.prepare_data(df_features, feature_cols)
                self.predictor.train(df_clean, feature_cols)
                self.predictor.save_models(prefix=league.lower())
                
                return True
            except Exception as e:
                st.error(f"Error training model: {e}")
                return False
    
    def get_prediction(self, league: str, home_team: str, away_team: str) -> dict:
        """Get prediction for a match"""
        # Get features data
        if league not in st.session_state.features_cache:
            df = self.fetcher.get_or_fetch_league_data(league)
            df_features = self.engineer.create_features(df)
            st.session_state.features_cache[league] = df_features
        
        df = st.session_state.features_cache[league]
        feature_cols = self.engineer.get_feature_columns()
        
        # Get latest data for prediction
        home_latest = df[
            (df['HomeTeam'] == home_team) | (df['AwayTeam'] == home_team)
        ].iloc[-1] if len(df[(df['HomeTeam'] == home_team) | (df['AwayTeam'] == home_team)]) > 0 else None
        
        away_latest = df[
            (df['HomeTeam'] == away_team) | (df['AwayTeam'] == away_team)
        ].iloc[-1] if len(df[(df['HomeTeam'] == away_team) | (df['AwayTeam'] == away_team)]) > 0 else None
        
        if home_latest is None or away_latest is None:
            return None
        
        # Build feature vector
        features = []
        for col in feature_cols:
            if col in home_latest.index:
                features.append(home_latest[col])
            elif col in away_latest.index:
                features.append(away_latest[col])
            else:
                features.append(0)
        
        features = np.array(features).reshape(1, -1)
        predictions = self.predictor.predict(features)
        
        return predictions
    
    def get_team_form(self, league: str, team: str, n_matches: int = 5) -> List[Dict]:
        """Get team's recent form (last N matches with results)"""
        try:
            df = self.fetcher.load_league_data(league)
            if df is None or df.empty:
                return []
            
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date', ascending=False)
            
            # Get matches where team played
            team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].head(n_matches)
            
            form = []
            for _, match in team_matches.iterrows():
                is_home = match['HomeTeam'] == team
                goals_scored = match['FTHG'] if is_home else match['FTAG']
                goals_conceded = match['FTAG'] if is_home else match['FTHG']
                opponent = match['AwayTeam'] if is_home else match['HomeTeam']
                
                if goals_scored > goals_conceded:
                    result = 'W'
                elif goals_scored < goals_conceded:
                    result = 'L'
                else:
                    result = 'D'
                
                form.append({
                    'result': result,
                    'goals_scored': goals_scored,
                    'goals_conceded': goals_conceded,
                    'opponent': opponent,
                    'is_home': is_home,
                    'date': match['Date'].strftime('%Y-%m-%d') if pd.notna(match['Date']) else ''
                })
            
            return form
        except:
            return []
    
    def get_correct_score_predictions(self, home_expected_goals: float, away_expected_goals: float, top_n: int = 10) -> List[Dict]:
        """Calculate correct score probabilities using Poisson distribution"""
        scores = []
        
        # Limit expected goals to reasonable range
        home_xg = max(0.5, min(4.0, home_expected_goals))
        away_xg = max(0.3, min(3.5, away_expected_goals))
        
        for home_goals in range(6):
            for away_goals in range(6):
                # Poisson probability for each scoreline
                home_prob = poisson.pmf(home_goals, home_xg)
                away_prob = poisson.pmf(away_goals, away_xg)
                score_prob = home_prob * away_prob
                
                scores.append({
                    'score': f"{home_goals}-{away_goals}",
                    'home_goals': home_goals,
                    'away_goals': away_goals,
                    'probability': score_prob * 100,
                    'odds': 1 / score_prob if score_prob > 0 else 100
                })
        
        # Sort by probability and return top N
        scores.sort(key=lambda x: x['probability'], reverse=True)
        return scores[:top_n]
    
    def get_prediction_explanation(self, league: str, home_team: str, away_team: str) -> Dict:
        """Get explanation for why the model made its prediction"""
        try:
            # Get feature importance
            importance_df = self.predictor.get_feature_importance('match_result')
            if importance_df.empty:
                return {'factors': [], 'summary': 'Feature importance not available'}
            
            # Get the current features for context
            if league in st.session_state.features_cache:
                df = st.session_state.features_cache[league]
                
                home_data = df[(df['HomeTeam'] == home_team) | (df['AwayTeam'] == home_team)].iloc[-1] if len(df[(df['HomeTeam'] == home_team) | (df['AwayTeam'] == home_team)]) > 0 else None
                away_data = df[(df['HomeTeam'] == away_team) | (df['AwayTeam'] == away_team)].iloc[-1] if len(df[(df['HomeTeam'] == away_team) | (df['AwayTeam'] == away_team)]) > 0 else None
                
                if home_data is not None and away_data is not None:
                    # Top factors with their values
                    top_features = importance_df.head(8)
                    factors = []
                    
                    feature_explanations = {
                        'form_diff': ('Form Difference', 'Higher = Home team in better form'),
                        'home_form_points': ('Home Team Form', 'Avg points per game recently'),
                        'away_form_points': ('Away Team Form', 'Avg points per game recently'),
                        'position_diff': ('Table Position Gap', 'Positive = Home team higher'),
                        'home_form_goals_scored': ('Home Scoring Form', 'Avg goals scored recently'),
                        'away_form_goals_scored': ('Away Scoring Form', 'Avg goals scored recently'),
                        'home_form_goals_conceded': ('Home Defensive Form', 'Avg goals conceded'),
                        'away_form_goals_conceded': ('Away Defensive Form', 'Avg goals conceded'),
                        'h2h_home_wins': ('H2H Home Advantage', 'Historical home wins %'),
                        'home_season_points_per_game': ('Home Season PPG', 'Season points per game'),
                        'away_season_points_per_game': ('Away Season PPG', 'Season points per game'),
                        'attack_strength_diff': ('Attack Strength Gap', 'Scoring ability difference'),
                        'defense_strength_diff': ('Defense Strength Gap', 'Defensive ability difference'),
                    }
                    
                    for _, row in top_features.iterrows():
                        feat_name = row['feature']
                        importance = row['importance']
                        
                        # Get value
                        value = home_data.get(feat_name, away_data.get(feat_name, 0))
                        
                        exp = feature_explanations.get(feat_name, (feat_name, ''))
                        factors.append({
                            'name': exp[0],
                            'feature': feat_name,
                            'importance': importance * 100,
                            'value': value,
                            'explanation': exp[1]
                        })
                    
                    return {'factors': factors, 'summary': 'Key factors influencing this prediction'}
            
            return {'factors': [], 'summary': 'Could not generate explanation'}
        except Exception as e:
            return {'factors': [], 'summary': f'Error: {str(e)}'}
    
    def get_confidence_level(self, probability: float) -> Tuple[str, str, str]:
        """Get confidence level label, color, and emoji based on probability"""
        if probability >= 0.65:
            return "HIGH", "#28a745", "🟢"
        elif probability >= 0.50:
            return "MEDIUM", "#ffc107", "🟡"
        elif probability >= 0.40:
            return "LOW", "#fd7e14", "🟠"
        else:
            return "VERY LOW", "#dc3545", "🔴"
    
    def render_sidebar(self):
        """Render the sidebar"""
        with st.sidebar:
            # User menu at top
            self.render_user_menu()
            
            st.divider()
            
            st.markdown("## ⚙️ Settings")
            
            # Data source info
            st.markdown("### 📊 Data Source")
            st.success("✅ Using OpenFootball (No API key needed!)")
            st.markdown("""
            Free, open public domain football data from 
            [openfootball.github.io](https://openfootball.github.io/)
            
            **Coverage:** 2010-2026 seasons
            """)
            
            st.divider()
            
            # League selection - filter by user access
            leagues = self.fetcher.get_available_leagues()
            
            # Filter leagues based on user tier
            accessible_leagues = {
                code: info for code, info in leagues.items() 
                if self.check_league_access(code)
            }
            
            if not accessible_leagues:
                accessible_leagues = {'EPL': leagues['EPL']}  # Fallback to EPL
            
            league_options = {f"{info['flag']} {info['name']}": code 
                           for code, info in accessible_leagues.items()}
            
            # Show upgrade hint if limited leagues
            if len(accessible_leagues) < len(leagues):
                st.caption(f"📊 {len(accessible_leagues)}/{len(leagues)} leagues available")
                st.caption("💎 Upgrade for all leagues")
            
            selected_display = st.selectbox(
                "🏆 Select League",
                options=list(league_options.keys()),
                index=0
            )
            selected_league = league_options[selected_display]
            st.session_state.current_league = selected_league
            
            st.divider()
            
            # Data refresh options
            st.markdown("### 📊 Data Management")
            
            # Show data freshness
            data_age = self.fetcher.get_data_age_days(selected_league)
            if data_age is not None:
                if data_age <= 7:
                    st.success(f"✅ Data is {data_age} days old")
                elif data_age <= 14:
                    st.warning(f"⚠️ Data is {data_age} days old")
                else:
                    st.error(f"🔴 Data is {data_age} days old")
            else:
                st.info("📥 No data yet - click Refresh")
            
            # Auto-refresh setting
            auto_refresh = st.checkbox(
                "Auto-refresh weekly",
                value=True,
                help="Automatically refresh data if older than 7 days"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Refresh", width='stretch'):
                    with st.spinner("Fetching latest data..."):
                        try:
                            df = self.fetcher.get_or_fetch_league_data(selected_league, force_refresh=True)
                            st.success(f"✅ {len(df)} matches!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
            
            with col2:
                if st.button("🔄 All Leagues", width='stretch'):
                    with st.spinner("Refreshing all leagues..."):
                        for league_code in leagues.keys():
                            try:
                                self.fetcher.get_or_fetch_league_data(league_code, force_refresh=True)
                            except:
                                pass
                        st.success("✅ All leagues refreshed!")
                        st.rerun()
            
            st.divider()
            
            # Training options
            st.markdown("### 🤖 Model Training")
            
            if st.button("🔄 Retrain Model", width='stretch'):
                self.load_or_train_model(selected_league, force_retrain=True)
                st.success("✅ Model retrained!")
            
            st.divider()
            
            # League info
            league_info = leagues[selected_league]
            st.markdown(f"### {league_info['flag']} {league_info['name']}")
            st.markdown(f"**Country:** {league_info['country']}")
            
            # Get team count from data
            try:
                teams = self.fetcher.get_league_teams(selected_league)
                st.markdown(f"**Teams:** {len(teams)}")
            except:
                st.markdown("**Teams:** Loading...")
            
            st.divider()
            
            # About
            st.markdown("### ℹ️ About")
            st.markdown("""
            This app uses machine learning to predict football match outcomes:
            
            **Match Results:**
            - Home Win / Draw / Away Win
            
            **Goals Markets:**
            - Over/Under 1.5, 2.5, 3.5 Goals
            - Home Over 0.5, 1.5, 2.5 Goals
            - Away Over 0.5, 1.5, 2.5 Goals
            - BTTS (Both Teams to Score)
            
            **Half Time:**
            - HT Over 0.5, 1.5 Goals
            
            **Goal Ranges:**
            - 0-1 Goals, 2-3 Goals, 4+ Goals
            
            **Data Source:** OpenFootball
            - Free, open public domain data
            - No API key required!
            """)
            
            return selected_league
    
    def render_main_content(self, league: str):
        """Render the main content area"""
        # Header
        st.markdown('<h1 class="main-header">⚽ Football Match Predictor</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">AI-powered predictions using real football data • 17 prediction markets</p>', unsafe_allow_html=True)
        
        # Auto-refresh check (weekly)
        self.check_auto_refresh(league, max_age_days=7)
        
        # Check for data and show info
        try:
            df = self.fetcher.load_league_data(league)
            df['Date'] = pd.to_datetime(df['Date'])
            latest_date = df['Date'].max()
            
            # Show data info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Total Matches", len(df))
            with col2:
                st.metric("📅 Latest Data", latest_date.strftime("%Y-%m-%d"))
            with col3:
                teams_count = len(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))
                st.metric("👥 Teams", teams_count)
            with col4:
                st.metric("🎯 Predictions", "17 Markets")
            
            # Warning if data is old
            from datetime import datetime
            days_old = (datetime.now() - latest_date.to_pydatetime()).days
            if days_old > 60:
                st.warning(f"⚠️ Data is {days_old} days old. Click 'Refresh Data' in the sidebar for the latest matches.")
        except:
            st.info("📥 Click 'Refresh Data' in the sidebar to fetch match data.")
        
        # Tabs for different functionality
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13 = st.tabs([
            "🔮 Predict Match", 
            "📅 Upcoming Matches", 
            "📊 Team Form",
            "🎰 Accumulator",
            "💹 Live Odds",
            "⚽ Player Stats",
            "🏥 Injuries",
            "🏟️ Team Explorer",
            "📈 Track Record",
            "📋 My Predictions",
            "🏆 Leaderboard",
            "⚔️ Head-to-Head",
            "📊 Stats"
        ])
        
        with tab1:
            self.render_prediction_tab(league)
        
        with tab2:
            self.render_upcoming_matches_tab(league)
        
        with tab3:
            self.render_team_form_tab(league)
        
        with tab4:
            self.render_accumulator_tab(league)
        
        with tab5:
            self.render_live_odds_tab(league)
        
        with tab6:
            self.render_player_stats_tab(league)
        
        with tab7:
            self.render_injuries_tab(league)
        
        with tab8:
            self.render_team_explorer_tab(league)
        
        with tab9:
            self.render_accuracy_tab(league)
        
        with tab10:
            self.render_my_predictions_tab(league)
        
        with tab11:
            self.render_leaderboard_tab()
        
        with tab12:
            self.render_head_to_head_tab(league)
        
        with tab13:
            self.render_stats_tab(league)
    
    def render_prediction_tab(self, league: str):
        """Render the prediction selection tab"""
        
        # Ensure model is loaded
        try:
            self.load_or_train_model(league)
        except Exception as e:
            st.error(f"Could not load model: {e}")
            st.info("Click 'Refresh Data' to fetch match data first.")
            return
        
        # Get teams from loaded data
        leagues = self.fetcher.get_available_leagues()
        league_info = leagues[league]
        
        # Fallback teams by league
        fallback_teams = {
            'EPL': ['Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton', 'Chelsea', 
                   'Crystal Palace', 'Everton', 'Fulham', 'Liverpool', 'Manchester City', 
                   'Manchester United', 'Newcastle', 'Nottingham Forest', 'Southampton', 
                   'Tottenham', 'West Ham', 'Wolves', 'Ipswich', 'Leicester'],
            'LA_LIGA': ['Real Madrid', 'Barcelona', 'Atletico Madrid', 'Sevilla', 'Real Betis',
                       'Real Sociedad', 'Villarreal', 'Athletic Bilbao', 'Valencia', 'Osasuna'],
            'SERIE_A': ['Juventus', 'Inter', 'AC Milan', 'Napoli', 'Roma', 'Lazio', 'Atalanta',
                       'Fiorentina', 'Bologna', 'Torino'],
            'BUNDESLIGA': ['Bayern Munich', 'Borussia Dortmund', 'RB Leipzig', 'Bayer Leverkusen',
                          'Wolfsburg', 'Eintracht Frankfurt', 'Union Berlin', 'Freiburg'],
            'LIGUE_1': ['Paris Saint-Germain', 'Marseille', 'Monaco', 'Lyon', 'Nice', 'Lille',
                       'Lens', 'Rennes', 'Strasbourg', 'Toulouse'],
        }
        
        # Get teams from actual data
        try:
            teams = self.fetcher.get_league_teams(league)
            if not teams:
                teams = fallback_teams.get(league, fallback_teams['EPL'])
        except:
            teams = fallback_teams.get(league, fallback_teams['EPL'])
        
        if not teams:
            st.warning("No teams available. Please refresh data.")
            return
        
        # Match Selection
        st.markdown(f"### {league_info['flag']} Select Match - {league_info['name']}")
        
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            home_team = st.selectbox(
                "🏠 Home Team",
                options=teams,
                index=0,
                key="home_team"
            )
        
        with col2:
            st.markdown("<div style='text-align: center; padding-top: 2rem; font-size: 2rem;'>⚔️</div>", unsafe_allow_html=True)
        
        with col3:
            away_options = [t for t in teams if t != home_team]
            away_team = st.selectbox(
                "✈️ Away Team",
                options=away_options,
                index=min(1, len(away_options)-1) if away_options else 0,
                key="away_team"
            )
        
        # Predict button
        st.markdown("")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_clicked = st.button(
                "🔮 Get Prediction",
                width='stretch',
                type="primary"
            )
        
        if predict_clicked:
            # Check prediction limit
            can_predict, limit_msg = self.check_prediction_limit()
            if not can_predict:
                st.warning(f"⚠️ {limit_msg}")
                st.info("💎 Upgrade your plan for more predictions!")
            else:
                self.render_prediction(league, home_team, away_team)
                self.record_prediction_usage()
    
    def render_prediction(self, league: str, home_team: str, away_team: str):
        """Render the prediction results with all 17 prediction targets"""
        with st.spinner("🔮 Analyzing match data..."):
            predictions = self.get_prediction(league, home_team, away_team)
        
        if predictions is None:
            st.error("Could not generate prediction. Please try different teams.")
            return
        
        st.markdown("---")
        st.markdown(f"## 📊 Prediction: {home_team} vs {away_team}")
        
        # ========== MATCH RESULT ==========
        st.markdown("### 🏆 Match Result Prediction")
        
        match_pred = predictions.get('match_result', {})
        if match_pred and 'probabilities' in match_pred:
            probs = match_pred['probabilities'][0]
            home_prob, draw_prob, away_prob = probs[0], probs[1], probs[2]
        else:
            home_prob, draw_prob, away_prob = 0.33, 0.34, 0.33
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if home_prob == max([home_prob, draw_prob, away_prob]):
                st.markdown(f"""
                <div class="win-home">
                    <h3 style="margin:0; color: white;">🏠 {home_team}</h3>
                    <h2 style="margin:0; color: white;">{home_prob*100:.1f}%</h2>
                    <p style="margin:0; color: rgba(255,255,255,0.8);">HOME WIN</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.metric(f"🏠 {home_team}", f"{home_prob*100:.1f}%", "Home Win")
        
        with col2:
            if draw_prob == max([home_prob, draw_prob, away_prob]):
                st.markdown(f"""
                <div class="win-draw">
                    <h3 style="margin:0; color: white;">🤝 Draw</h3>
                    <h2 style="margin:0; color: white;">{draw_prob*100:.1f}%</h2>
                    <p style="margin:0; color: rgba(255,255,255,0.8);">DRAW</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.metric("🤝 Draw", f"{draw_prob*100:.1f}%")
        
        with col3:
            if away_prob == max([home_prob, draw_prob, away_prob]):
                st.markdown(f"""
                <div class="win-away">
                    <h3 style="margin:0; color: white;">✈️ {away_team}</h3>
                    <h2 style="margin:0; color: white;">{away_prob*100:.1f}%</h2>
                    <p style="margin:0; color: rgba(255,255,255,0.8);">AWAY WIN</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.metric(f"✈️ {away_team}", f"{away_prob*100:.1f}%", "Away Win")
        
        # ========== GOALS PREDICTIONS ==========
        st.markdown("### ⚽ Total Goals Prediction")
        
        col1, col2, col3 = st.columns(3)
        
        def get_over_prob(key):
            pred = predictions.get(key, {})
            if pred and 'probabilities' in pred:
                return pred['probabilities'][0][1] * 100
            return 50.0
        
        with col1:
            over15_prob = get_over_prob('over_1.5')
            st.markdown("#### Over/Under 1.5")
            st.metric("📈 Over 1.5", f"{over15_prob:.1f}%")
            st.progress(over15_prob / 100)
        
        with col2:
            over25_prob = get_over_prob('over_2.5')
            st.markdown("#### Over/Under 2.5")
            st.metric("📈 Over 2.5", f"{over25_prob:.1f}%")
            st.progress(over25_prob / 100)
        
        with col3:
            over35_prob = get_over_prob('over_3.5')
            st.markdown("#### Over/Under 3.5")
            st.metric("📈 Over 3.5", f"{over35_prob:.1f}%")
            st.progress(over35_prob / 100)
        
        # ========== BTTS ==========
        st.markdown("### 🔄 Both Teams To Score (BTTS)")
        
        btts_pred = predictions.get('btts', {})
        if btts_pred and 'probabilities' in btts_pred:
            btts_yes = btts_pred['probabilities'][0][1] * 100
        else:
            btts_yes = 50.0
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="btts-yes">
                <h3 style="margin:0; color: white;">✅ BTTS Yes</h3>
                <h2 style="margin:0; color: white;">{btts_yes:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <h3 style="margin:0;">❌ BTTS No</h3>
                <h2 style="margin:0;">{100-btts_yes:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # ========== HOME/AWAY GOALS ==========
        st.markdown("### 🏠 Home Team Goals")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            home_o05 = get_over_prob('home_over_0.5')
            st.metric("Home Over 0.5", f"{home_o05:.1f}%")
        with col2:
            home_o15 = get_over_prob('home_over_1.5')
            st.metric("Home Over 1.5", f"{home_o15:.1f}%")
        with col3:
            # Calculate home over 2.5 from the data if available
            st.metric("Home Over 2.5", f"{max(0, home_o15-15):.1f}%")
        
        st.markdown("### ✈️ Away Team Goals")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            away_o05 = get_over_prob('away_over_0.5')
            st.metric("Away Over 0.5", f"{away_o05:.1f}%")
        with col2:
            away_o15 = get_over_prob('away_over_1.5')
            st.metric("Away Over 1.5", f"{away_o15:.1f}%")
        with col3:
            st.metric("Away Over 2.5", f"{max(0, away_o15-15):.1f}%")
        
        # ========== HALF TIME ==========
        st.markdown("### ⏱️ Half Time Goals")
        
        col1, col2 = st.columns(2)
        with col1:
            ht_o05 = get_over_prob('ht_over_0.5')
            st.markdown(f"""
            <div class="ht-card">
                <h4 style="margin:0; color: white;">HT Over 0.5 Goals</h4>
                <h2 style="margin:0; color: white;">{ht_o05:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            ht_o15 = get_over_prob('ht_over_1.5')
            st.markdown(f"""
            <div class="ht-card">
                <h4 style="margin:0; color: white;">HT Over 1.5 Goals</h4>
                <h2 style="margin:0; color: white;">{ht_o15:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # ========== GOAL RANGES ==========
        st.markdown("### 📊 Goal Range Prediction")
        
        col1, col2, col3 = st.columns(3)
        
        goals_01 = get_over_prob('goals_0_1')
        goals_23 = get_over_prob('goals_2_3')
        goals_4p = get_over_prob('goals_4_plus')
        
        # Normalize if needed
        total = goals_01 + goals_23 + goals_4p
        if total > 0:
            goals_01 = (goals_01 / total) * 100
            goals_23 = (goals_23 / total) * 100
            goals_4p = (goals_4p / total) * 100
        
        with col1:
            st.metric("0-1 Goals", f"{goals_01:.1f}%")
        with col2:
            st.metric("2-3 Goals", f"{goals_23:.1f}%")
        with col3:
            st.metric("4+ Goals", f"{goals_4p:.1f}%")
        
        # ========== VALUE BETS ==========
        st.markdown("### 💰 Value Bet Analysis")
        
        # Analyze value
        value_analysis = self.value_calculator.analyze_value(predictions)
        best_value = self.value_calculator.get_best_value_bets(value_analysis, top_n=3)
        
        if best_value:
            st.markdown("**🎯 Potential Value Bets** (Edge vs typical market odds)")
            
            cols = st.columns(len(best_value))
            for i, (market, details) in enumerate(best_value):
                market_name = self.value_calculator.format_market_name(market)
                with cols[i]:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                                padding: 1rem; border-radius: 10px; color: white; text-align: center;">
                        <h4 style="margin:0; font-size:0.9rem;">{market_name}</h4>
                        <h2 style="margin:0.5rem 0;">+{details['edge']:.1f}%</h2>
                        <p style="margin:0; font-size:0.8rem; opacity:0.9;">
                            Our: {details['model_prob']:.0f}% vs Market: {details['market_prob']:.0f}%
                        </p>
                        <p style="margin:0.3rem 0 0 0; font-size:0.8rem;">
                            📊 Odds: {details['decimal_odds']:.2f}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("📊 No significant value detected vs typical market odds for this match.")
        
        # ========== TEAM FORM ==========
        st.markdown("### 📈 Team Form (Last 5 Games)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**🏠 {home_team}**")
            home_form = self.get_team_form(league, home_team, 5)
            if home_form:
                form_str = ""
                for match in home_form:
                    if match['result'] == 'W':
                        form_str += "🟢 "
                    elif match['result'] == 'D':
                        form_str += "🟡 "
                    else:
                        form_str += "🔴 "
                st.markdown(f"Form: {form_str}")
                
                # Recent results details
                for match in home_form[:3]:
                    venue = "🏠" if match['is_home'] else "✈️"
                    result_color = "green" if match['result'] == 'W' else "orange" if match['result'] == 'D' else "red"
                    st.markdown(f"{venue} vs {match['opponent']}: **{match['goals_scored']}-{match['goals_conceded']}** :{result_color}[{match['result']}]")
            else:
                st.caption("Form data not available")
        
        with col2:
            st.markdown(f"**✈️ {away_team}**")
            away_form = self.get_team_form(league, away_team, 5)
            if away_form:
                form_str = ""
                for match in away_form:
                    if match['result'] == 'W':
                        form_str += "🟢 "
                    elif match['result'] == 'D':
                        form_str += "🟡 "
                    else:
                        form_str += "🔴 "
                st.markdown(f"Form: {form_str}")
                
                # Recent results details
                for match in away_form[:3]:
                    venue = "🏠" if match['is_home'] else "✈️"
                    result_color = "green" if match['result'] == 'W' else "orange" if match['result'] == 'D' else "red"
                    st.markdown(f"{venue} vs {match['opponent']}: **{match['goals_scored']}-{match['goals_conceded']}** :{result_color}[{match['result']}]")
            else:
                st.caption("Form data not available")
        
        # ========== CORRECT SCORE PREDICTIONS ==========
        st.markdown("### 🎯 Correct Score Predictions")
        
        # Estimate expected goals from the model
        home_xg = 1.5 + (home_prob - 0.33) * 2  # Rough estimate
        away_xg = 1.2 + (away_prob - 0.33) * 2
        
        # Adjust based on over 2.5 probability
        if over25_prob > 60:
            home_xg *= 1.1
            away_xg *= 1.1
        elif over25_prob < 40:
            home_xg *= 0.9
            away_xg *= 0.9
        
        correct_scores = self.get_correct_score_predictions(home_xg, away_xg, top_n=9)
        
        # Display in 3 columns
        score_cols = st.columns(3)
        for i, score in enumerate(correct_scores[:9]):
            with score_cols[i % 3]:
                bg_color = "#e8f5e9" if i == 0 else "#f8f9fa"
                border = "2px solid #28a745" if i == 0 else "1px solid #dee2e6"
                st.markdown(f"""
                <div style="background: {bg_color}; padding: 0.75rem; border-radius: 8px; 
                            text-align: center; margin-bottom: 0.5rem; border: {border};">
                    <h3 style="margin:0; color: #333;">{score['score']}</h3>
                    <p style="margin:0; color: #666; font-size: 0.9rem;">{score['probability']:.1f}%</p>
                    <small style="color: #999;">Odds: {score['odds']:.1f}</small>
                </div>
                """, unsafe_allow_html=True)
        
        st.caption("📊 Based on Poisson distribution model using team scoring patterns")
        
        # ========== PREDICTION EXPLANATION ==========
        st.markdown("### 🧠 Why This Prediction?")
        
        explanation = self.get_prediction_explanation(league, home_team, away_team)
        
        if explanation['factors']:
            st.markdown("**Key factors influencing this prediction:**")
            
            for factor in explanation['factors'][:6]:
                importance_bar = "█" * int(factor['importance'] / 3) + "░" * (10 - int(factor['importance'] / 3))
                
                # Determine if factor favors home or away
                if 'home' in factor['feature'].lower() and factor['value'] > 0:
                    favor_emoji = "🏠"
                elif 'away' in factor['feature'].lower() and factor['value'] > 0:
                    favor_emoji = "✈️"
                elif 'diff' in factor['feature'].lower():
                    favor_emoji = "🏠" if factor['value'] > 0 else "✈️" if factor['value'] < 0 else "⚖️"
                else:
                    favor_emoji = "📊"
                
                st.markdown(f"""
                {favor_emoji} **{factor['name']}** `{importance_bar}` {factor['importance']:.1f}%  
                <small style="color: #666;">Value: {factor['value']:.2f} • {factor['explanation']}</small>
                """, unsafe_allow_html=True)
        else:
            st.caption("Prediction explanation not available")
        
        # ========== ADD TO ACCUMULATOR ==========
        st.markdown("### 🎰 Add to Accumulator")
        
        acca_col1, acca_col2, acca_col3 = st.columns(3)
        
        with acca_col1:
            if st.button(f"➕ {home_team} Win ({home_prob*100:.0f}%)", key=f"acca_home_{home_team}_{away_team}"):
                selection = {
                    'match': f"{home_team} vs {away_team}",
                    'selection': f"{home_team} Win",
                    'odds': round(1 / home_prob, 2) if home_prob > 0 else 1.01,
                    'probability': home_prob * 100
                }
                if selection not in st.session_state.accumulator:
                    st.session_state.accumulator.append(selection)
                    st.success(f"✅ Added to accumulator!")
        
        with acca_col2:
            if st.button(f"➕ Draw ({draw_prob*100:.0f}%)", key=f"acca_draw_{home_team}_{away_team}"):
                selection = {
                    'match': f"{home_team} vs {away_team}",
                    'selection': "Draw",
                    'odds': round(1 / draw_prob, 2) if draw_prob > 0 else 1.01,
                    'probability': draw_prob * 100
                }
                if selection not in st.session_state.accumulator:
                    st.session_state.accumulator.append(selection)
                    st.success(f"✅ Added to accumulator!")
        
        with acca_col3:
            if st.button(f"➕ {away_team} Win ({away_prob*100:.0f}%)", key=f"acca_away_{home_team}_{away_team}"):
                selection = {
                    'match': f"{home_team} vs {away_team}",
                    'selection': f"{away_team} Win",
                    'odds': round(1 / away_prob, 2) if away_prob > 0 else 1.01,
                    'probability': away_prob * 100
                }
                if selection not in st.session_state.accumulator:
                    st.session_state.accumulator.append(selection)
                    st.success(f"✅ Added to accumulator!")
        
        # Goals markets
        goals_col1, goals_col2 = st.columns(2)
        with goals_col1:
            if st.button(f"➕ Over 2.5 Goals ({over25_prob:.0f}%)", key=f"acca_o25_{home_team}_{away_team}"):
                selection = {
                    'match': f"{home_team} vs {away_team}",
                    'selection': "Over 2.5 Goals",
                    'odds': round(100 / over25_prob, 2) if over25_prob > 0 else 1.01,
                    'probability': over25_prob
                }
                if selection not in st.session_state.accumulator:
                    st.session_state.accumulator.append(selection)
                    st.success(f"✅ Added to accumulator!")
        
        with goals_col2:
            if st.button(f"➕ BTTS Yes ({btts_yes:.0f}%)", key=f"acca_btts_{home_team}_{away_team}"):
                selection = {
                    'match': f"{home_team} vs {away_team}",
                    'selection': "BTTS Yes",
                    'odds': round(100 / btts_yes, 2) if btts_yes > 0 else 1.01,
                    'probability': btts_yes
                }
                if selection not in st.session_state.accumulator:
                    st.session_state.accumulator.append(selection)
                    st.success(f"✅ Added to accumulator!")
        
        if st.session_state.accumulator:
            st.info(f"🎰 You have {len(st.session_state.accumulator)} selection(s) in your accumulator. Go to the Accumulator tab to view.")
        
        # ========== SUMMARY ==========
        st.markdown("### 💡 Prediction Summary")
        
        result_pred = match_pred.get('prediction', ['Draw'])[0] if match_pred else 'Draw'
        confidence = match_pred.get('confidence', [0.5])[0] * 100 if match_pred else 50.0
        
        # Calculate overall confidence
        overall_confidence = max(home_prob, draw_prob, away_prob) * 100
        
        goals_pred = "Over 2.5 Goals" if over25_prob > 50 else "Under 2.5 Goals"
        btts_pred_text = "Yes" if btts_yes > 50 else "No"
        
        # Confidence badge
        if overall_confidence >= 60:
            confidence_badge = "🟢 HIGH CONFIDENCE"
            confidence_color = "green"
        elif overall_confidence >= 50:
            confidence_badge = "🟡 MEDIUM CONFIDENCE"
            confidence_color = "orange"
        else:
            confidence_badge = "🔴 LOW CONFIDENCE"
            confidence_color = "red"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1rem; border-radius: 10px; color: white; margin-bottom: 1rem;">
            <h4 style="margin:0;">Confidence Level: {confidence_badge}</h4>
            <p style="margin:0.5rem 0 0 0; opacity: 0.9;">Model confidence: {overall_confidence:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.success(f"""
        **🏆 Match Outcome:** {result_pred} (Confidence: {overall_confidence:.1f}%)
        
        **⚽ Goals:** {goals_pred} ({over25_prob:.1f}%)
        
        **🔄 BTTS:** {btts_pred_text} ({btts_yes:.1f}%)
        
        **⏱️ Half Time:** {'Over' if ht_o05 > 50 else 'Under'} 0.5 goals ({ht_o05:.1f}%)
        
        **📊 Most Likely Goal Range:** {'0-1' if goals_01 == max(goals_01, goals_23, goals_4p) else '2-3' if goals_23 == max(goals_01, goals_23, goals_4p) else '4+'} Goals
        """)
        
        # Disclaimer
        st.caption("⚠️ These predictions are for informational purposes only. Past performance does not guarantee future results.")
        
        # Save prediction to database
        try:
            prediction_data = {
                'league': league,
                'home_team': home_team,
                'away_team': away_team,
                'home_win_prob': home_prob,
                'draw_prob': draw_prob,
                'away_win_prob': away_prob,
                'over_1_5_prob': over15_prob / 100,
                'over_2_5_prob': over25_prob / 100,
                'over_3_5_prob': over35_prob / 100,
                'btts_prob': btts_yes / 100,
                'home_over_0_5_prob': home_o05 / 100,
                'home_over_1_5_prob': home_o15 / 100,
                'away_over_0_5_prob': away_o05 / 100,
                'away_over_1_5_prob': away_o15 / 100,
                'ht_over_0_5_prob': ht_o05 / 100,
                'ht_over_1_5_prob': ht_o15 / 100,
                'goals_0_1_prob': goals_01 / 100,
                'goals_2_3_prob': goals_23 / 100,
                'goals_4_plus_prob': goals_4p / 100,
            }
            prediction_id = self.db.save_prediction(prediction_data)
            st.caption(f"📝 Prediction saved (ID: {prediction_id})")
        except Exception as e:
            pass  # Silently fail if DB save fails
    
    def render_upcoming_matches_tab(self, league: str):
        """Render the upcoming matches tab with auto-predictions"""
        st.markdown("### 📅 Upcoming Matches")
        
        # Get league info
        leagues = self.fetcher.get_available_leagues()
        league_info = leagues.get(league, {'flag': '🏆', 'name': league})
        
        st.markdown(f"**{league_info['flag']} {league_info['name']}** - Upcoming fixtures with predictions")
        
        # Try to get upcoming fixtures
        try:
            fixtures_df = self.fixtures_fetcher.fetch_upcoming_fixtures(league, days_ahead=14)
            
            if fixtures_df.empty:
                st.info("📭 No upcoming fixtures found. Try refreshing the data or check back closer to match day.")
                
                # Show sample upcoming matches interface
                st.markdown("#### 🔮 Quick Predict")
                st.markdown("Select teams above in the 'Predict Match' tab to get predictions for any matchup.")
                return
            
            # Display fixtures with predictions
            for idx, fixture in fixtures_df.iterrows():
                home_team = fixture.get('HomeTeam', fixture.get('home_team', 'Unknown'))
                away_team = fixture.get('AwayTeam', fixture.get('away_team', 'Unknown'))
                match_date = fixture.get('Date', fixture.get('date', 'TBD'))
                
                with st.expander(f"⚽ {home_team} vs {away_team} - {match_date}"):
                    col1, col2, col3 = st.columns([2, 1, 2])
                    
                    with col1:
                        st.markdown(f"### 🏠 {home_team}")
                    with col2:
                        st.markdown("<div style='text-align:center; font-size:1.5rem;'>VS</div>", unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"### ✈️ {away_team}")
                    
                    # Get prediction for this fixture
                    if st.button(f"🔮 Predict", key=f"pred_{home_team}_{away_team}_{idx}"):
                        can_predict, limit_msg = self.check_prediction_limit()
                        if not can_predict:
                            st.warning(f"⚠️ {limit_msg}")
                        else:
                            self.render_prediction(league, home_team, away_team)
                            self.record_prediction_usage()
        
        except Exception as e:
            st.warning(f"Could not load upcoming fixtures: {e}")
            st.markdown("Use the 'Predict Match' tab to manually select teams for prediction.")
    
    def render_stats_tab(self, league: str):
        """Render league statistics tab"""
        st.markdown("### 📊 League Statistics")
        
        try:
            df = self.fetcher.load_league_data(league)
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Basic stats
            col1, col2, col3, col4 = st.columns(4)
            
            total_goals = df['FTHG'].sum() + df['FTAG'].sum()
            avg_goals = total_goals / len(df) if len(df) > 0 else 0
            home_wins = len(df[df['FTR'] == 'H'])
            away_wins = len(df[df['FTR'] == 'A'])
            draws = len(df[df['FTR'] == 'D'])
            
            with col1:
                st.metric("⚽ Total Goals", f"{total_goals:,}")
            with col2:
                st.metric("📊 Avg Goals/Match", f"{avg_goals:.2f}")
            with col3:
                st.metric("🏠 Home Win %", f"{(home_wins/len(df)*100):.1f}%")
            with col4:
                st.metric("✈️ Away Win %", f"{(away_wins/len(df)*100):.1f}%")
            
            # Goals distribution
            st.markdown("#### ⚽ Goals Per Match Distribution")
            df['TotalGoals'] = df['FTHG'] + df['FTAG']
            goals_dist = df['TotalGoals'].value_counts().sort_index()
            st.bar_chart(goals_dist)
            
            # Recent form table
            st.markdown("#### 📈 Recent Results")
            recent = df.sort_values('Date', ascending=False).head(10)
            recent_display = recent[['Date', 'HomeTeam', 'FTHG', 'FTAG', 'AwayTeam', 'FTR']].copy()
            recent_display.columns = ['Date', 'Home', 'HG', 'AG', 'Away', 'Result']
            st.dataframe(recent_display, width='stretch', hide_index=True)
            
        except Exception as e:
            st.warning(f"Could not load statistics: {e}")
            st.info("Click 'Refresh Data' in the sidebar to fetch match data.")
    
    def render_accuracy_tab(self, league: str):
        """Render the prediction accuracy tracking tab"""
        st.markdown("### 📈 Prediction Track Record")
        st.markdown("View our prediction accuracy and performance metrics")
        
        # Get overall stats
        total_stats = self.db.get_total_stats()
        accuracy_stats = self.db.get_accuracy_stats(league=league)
        
        # Header metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📊 Total Predictions",
                total_stats['total_predictions'],
                help="Total predictions made"
            )
        
        with col2:
            st.metric(
                "✅ Verified Results",
                total_stats['verified'],
                help="Predictions with known outcomes"
            )
        
        with col3:
            st.metric(
                "🎯 Correct Predictions",
                total_stats['correct'],
                help="Number of correct match result predictions"
            )
        
        with col4:
            accuracy = total_stats['accuracy'] if total_stats['verified'] > 0 else 0
            st.metric(
                "📈 Overall Accuracy",
                f"{accuracy:.1f}%",
                help="Match result prediction accuracy"
            )
        
        st.divider()
        
        # Accuracy by market
        st.markdown("#### 🎯 Accuracy by Market")
        
        if accuracy_stats['total_predictions'] > 0:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="stat-card">
                    <h4>🏆 Match Result</h4>
                    <h2>{:.1f}%</h2>
                    <p>Win/Draw/Loss predictions</p>
                </div>
                """.format(accuracy_stats['result_accuracy']), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="stat-card">
                    <h4>⚽ Over 2.5 Goals</h4>
                    <h2>{:.1f}%</h2>
                    <p>Goals predictions</p>
                </div>
                """.format(accuracy_stats['over_2_5_accuracy']), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="stat-card">
                    <h4>🔄 BTTS</h4>
                    <h2>{:.1f}%</h2>
                    <p>Both teams to score</p>
                </div>
                """.format(accuracy_stats['btts_accuracy']), unsafe_allow_html=True)
        else:
            st.info("📭 No verified predictions yet. Start making predictions and check back after matches are played!")
        
        st.divider()
        
        # Accuracy by confidence tier
        st.markdown("#### 📊 Accuracy by Confidence Level")
        
        confidence_tiers = self.db.get_predictions_by_confidence_tier()
        
        if any(t['total'] > 0 for t in confidence_tiers):
            tier_data = pd.DataFrame(confidence_tiers)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create a simple bar chart
                chart_data = tier_data[tier_data['total'] > 0][['tier', 'accuracy']].set_index('tier')
                st.bar_chart(chart_data)
            
            with col2:
                st.markdown("**Performance by Tier**")
                for tier in confidence_tiers:
                    if tier['total'] > 0:
                        emoji = "🟢" if tier['accuracy'] >= 55 else "🟡" if tier['accuracy'] >= 45 else "🔴"
                        st.markdown(f"{emoji} **{tier['tier']}**: {tier['accuracy']:.1f}% ({tier['correct']}/{tier['total']})")
        else:
            st.info("Make predictions to see accuracy by confidence level")
        
        st.divider()
        
        # League performance comparison
        st.markdown("#### 🌍 Performance by League")
        
        league_perf = self.db.get_league_performance()
        
        if league_perf:
            league_df = pd.DataFrame(league_perf)
            st.dataframe(
                league_df,
                column_config={
                    "league": "League",
                    "total": st.column_config.NumberColumn("Predictions"),
                    "correct": st.column_config.NumberColumn("Correct"),
                    "accuracy": st.column_config.NumberColumn("Accuracy %", format="%.1f%%"),
                    "avg_confidence": st.column_config.NumberColumn("Avg Confidence %", format="%.1f%%")
                },
                hide_index=True,
                width="stretch"
            )
        else:
            st.info("No league performance data yet")
        
        st.divider()
        
        # High confidence predictions performance
        st.markdown("#### 🎯 High Confidence Predictions (>60%)")
        
        high_conf = self.db.get_high_confidence_accuracy(min_confidence=0.6)
        
        if high_conf['total'] > 0:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total High-Confidence", high_conf['total'])
            with col2:
                st.metric("Correct", high_conf['correct'])
            with col3:
                st.metric("Accuracy", f"{high_conf['accuracy']:.1f}%")
            
            if high_conf['accuracy'] >= 55:
                st.success(f"✅ High confidence predictions are performing above baseline ({high_conf['accuracy']:.1f}% vs 33% random)")
            else:
                st.warning(f"⚠️ Model needs more training data for better high-confidence predictions")
        else:
            st.info("No high-confidence predictions verified yet")
        
        st.divider()
        
        # Recent predictions
        st.markdown("#### 📋 Recent Predictions")
        
        recent = self.db.get_recent_predictions(limit=20, league=league)
        
        if recent:
            recent_df = pd.DataFrame(recent)
            
            # Format for display
            display_cols = ['created_at', 'home_team', 'away_team', 'predicted_result', 
                           'result_confidence', 'actual_result', 'result_correct']
            
            available_cols = [c for c in display_cols if c in recent_df.columns]
            display_df = recent_df[available_cols].copy()
            
            # Rename columns
            col_names = {
                'created_at': 'Date',
                'home_team': 'Home',
                'away_team': 'Away',
                'predicted_result': 'Prediction',
                'result_confidence': 'Confidence',
                'actual_result': 'Actual',
                'result_correct': 'Correct'
            }
            display_df = display_df.rename(columns=col_names)
            
            # Format confidence as percentage
            if 'Confidence' in display_df.columns:
                display_df['Confidence'] = display_df['Confidence'].apply(
                    lambda x: f"{x*100:.0f}%" if pd.notna(x) else "-"
                )
            
            # Format correct column
            if 'Correct' in display_df.columns:
                display_df['Correct'] = display_df['Correct'].apply(
                    lambda x: "✅" if x == True else "❌" if x == False else "⏳"
                )
            
            st.dataframe(display_df, hide_index=True, width="stretch")
        else:
            st.info("No predictions made yet. Start predicting matches to build your track record!")
        
        # Tips for users
        st.divider()
        st.markdown("#### 💡 Tips")
        st.markdown("""
        - **Higher confidence = better accuracy**: Focus on predictions above 60%
        - **Track record matters**: We log every prediction for transparency
        - **Verified results**: Results are verified after matches are played
        - **Multiple markets**: Don't rely on just match result - check goals and BTTS too
        """)
    
    def render_my_predictions_tab(self, league: str):
        """Render the user's prediction history tab"""
        st.markdown("### 📋 My Predictions")
        
        user = st.session_state.user
        
        if not user or user.get('is_guest'):
            st.info("🔐 Please login or register to view your prediction history")
            if st.button("🔐 Login / Register"):
                st.session_state.user = None
                st.rerun()
            return
        
        # Get user stats
        user_stats = self.db.get_user_stats(user['id'])
        
        # User performance header
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Predictions", user_stats['total_predictions'])
        with col2:
            st.metric("✅ Verified", user_stats['verified'])
        with col3:
            accuracy_delta = user_stats['result_accuracy'] - 33.3 if user_stats['verified'] > 0 else 0
            st.metric("🎯 Accuracy", f"{user_stats['result_accuracy']}%", 
                     f"{accuracy_delta:+.1f}% vs random" if user_stats['verified'] > 0 else None)
        with col4:
            st.metric("💪 Avg Confidence", f"{user_stats['avg_confidence']}%")
        
        st.divider()
        
        # Accuracy breakdown by market
        if user_stats['verified'] > 0:
            st.markdown("#### 🎯 Your Accuracy by Market")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                result_color = "🟢" if user_stats['result_accuracy'] >= 50 else "🟡" if user_stats['result_accuracy'] >= 40 else "🔴"
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; text-align: center;">
                    <h4>{result_color} Match Results</h4>
                    <h2>{user_stats['result_accuracy']}%</h2>
                    <small>{user_stats['result_correct']}/{user_stats['verified']} correct</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                goals_color = "🟢" if user_stats['over_2_5_accuracy'] >= 55 else "🟡" if user_stats['over_2_5_accuracy'] >= 45 else "🔴"
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; text-align: center;">
                    <h4>{goals_color} Over 2.5 Goals</h4>
                    <h2>{user_stats['over_2_5_accuracy']}%</h2>
                    <small>{user_stats['over_2_5_correct']}/{user_stats['verified']} correct</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                btts_color = "🟢" if user_stats['btts_accuracy'] >= 55 else "🟡" if user_stats['btts_accuracy'] >= 45 else "🔴"
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; text-align: center;">
                    <h4>{btts_color} BTTS</h4>
                    <h2>{user_stats['btts_accuracy']}%</h2>
                    <small>{user_stats['btts_correct']}/{user_stats['verified']} correct</small>
                </div>
                """, unsafe_allow_html=True)
        
        st.divider()
        
        # Prediction history
        st.markdown("#### 📜 Prediction History")
        
        predictions = self.db.get_user_predictions(user['id'], limit=50)
        
        if predictions:
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                filter_status = st.selectbox("Filter by", ["All", "Verified Only", "Pending"])
            with col2:
                filter_result = st.selectbox("Result", ["All", "Correct ✅", "Incorrect ❌"])
            
            # Apply filters
            filtered = predictions
            if filter_status == "Verified Only":
                filtered = [p for p in filtered if p.get('match_played')]
            elif filter_status == "Pending":
                filtered = [p for p in filtered if not p.get('match_played')]
            
            if filter_result == "Correct ✅":
                filtered = [p for p in filtered if p.get('result_correct') == True]
            elif filter_result == "Incorrect ❌":
                filtered = [p for p in filtered if p.get('result_correct') == False]
            
            # Display predictions
            for pred in filtered:
                status_icon = "✅" if pred.get('result_correct') else "❌" if pred.get('result_correct') == False else "⏳"
                confidence = pred.get('result_confidence', 0) * 100
                
                conf_color = "🟢" if confidence >= 60 else "🟡" if confidence >= 50 else "🔴"
                
                with st.expander(f"{status_icon} {pred['home_team']} vs {pred['away_team']} - {pred.get('created_at', '')[:10]}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**League:** {pred.get('league', 'Unknown')}")
                        st.markdown(f"**Predicted:** {pred.get('predicted_result', 'Unknown')}")
                        st.markdown(f"**Confidence:** {conf_color} {confidence:.1f}%")
                    
                    with col2:
                        if pred.get('match_played'):
                            st.markdown(f"**Actual Result:** {pred.get('actual_result', 'Unknown')}")
                            st.markdown(f"**Score:** {pred.get('actual_home_goals', '?')} - {pred.get('actual_away_goals', '?')}")
                            st.markdown(f"**Status:** {status_icon} {'Correct!' if pred.get('result_correct') else 'Incorrect'}")
                        else:
                            st.markdown("**Status:** ⏳ Pending result")
                    
                    # Additional markets
                    st.markdown("---")
                    goals_cols = st.columns(4)
                    with goals_cols[0]:
                        over_25 = pred.get('over_2_5_prob', 0.5) * 100
                        st.markdown(f"**O2.5:** {'Over' if over_25 > 50 else 'Under'} ({over_25:.0f}%)")
                    with goals_cols[1]:
                        btts = pred.get('btts_prob', 0.5) * 100
                        st.markdown(f"**BTTS:** {'Yes' if btts > 50 else 'No'} ({btts:.0f}%)")
                    with goals_cols[2]:
                        over_15 = pred.get('over_1_5_prob', 0.5) * 100
                        st.markdown(f"**O1.5:** {'Over' if over_15 > 50 else 'Under'} ({over_15:.0f}%)")
                    with goals_cols[3]:
                        over_35 = pred.get('over_3_5_prob', 0.5) * 100
                        st.markdown(f"**O3.5:** {'Over' if over_35 > 50 else 'Under'} ({over_35:.0f}%)")
            
            if not filtered:
                st.info("No predictions match your filter criteria")
        else:
            st.info("📭 You haven't made any predictions yet. Go to the Predict Match tab to get started!")
    
    def render_leaderboard_tab(self):
        """Render the user leaderboard tab"""
        st.markdown("### 🏆 Leaderboard")
        st.markdown("See how you rank against other predictors!")
        
        # Get leaderboard data
        min_predictions = st.slider("Minimum predictions to qualify", 1, 20, 5)
        leaderboard = self.db.get_leaderboard(min_predictions=min_predictions)
        
        if leaderboard:
            # Get usernames from auth module
            user_ids = [entry['user_id'] for entry in leaderboard if entry['user_id']]
            usernames = self.auth.get_usernames_batch(user_ids) if user_ids else {}
            
            # Update leaderboard entries with usernames
            for entry in leaderboard:
                if entry['user_id'] in usernames:
                    entry['username'] = usernames[entry['user_id']]
                    entry['display_name'] = entry['username'][:10] + '...' if len(entry['username']) > 10 else entry['username']
            
            # Current user's rank
            user = st.session_state.user
            user_rank = None
            
            if user and not user.get('is_guest'):
                for entry in leaderboard:
                    if entry['user_id'] == user['id']:
                        user_rank = entry
                        break
            
            if user_rank:
                st.success(f"🎉 Your Rank: **#{user_rank['rank']}** with {user_rank['accuracy']}% accuracy ({user_rank['correct']}/{user_rank['verified']} correct)")
            
            st.divider()
            
            # Top 3 podium
            if len(leaderboard) >= 3:
                st.markdown("#### 🥇🥈🥉 Top Performers")
                
                cols = st.columns(3)
                medals = ["🥇", "🥈", "🥉"]
                podium_order = [1, 0, 2]  # 2nd, 1st, 3rd for visual effect
                
                for i, pos in enumerate(podium_order):
                    if pos < len(leaderboard):
                        entry = leaderboard[pos]
                        with cols[i]:
                            height = "150px" if pos == 0 else "120px" if pos == 1 else "100px"
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, {'#FFD700' if pos == 0 else '#C0C0C0' if pos == 1 else '#CD7F32'} 0%, 
                                        {'#FFA500' if pos == 0 else '#A9A9A9' if pos == 1 else '#8B4513'} 100%);
                                        padding: 1rem; border-radius: 10px; text-align: center; height: {height};
                                        display: flex; flex-direction: column; justify-content: center;">
                                <h2 style="margin:0;">{medals[pos]}</h2>
                                <h4 style="margin:0.5rem 0;">{entry['username']}</h4>
                                <p style="margin:0;"><strong>{entry['accuracy']}%</strong></p>
                                <small>{entry['correct']}/{entry['verified']} correct</small>
                            </div>
                            """, unsafe_allow_html=True)
            
            st.divider()
            
            # Full leaderboard table
            st.markdown("#### 📊 Full Rankings")
            
            # Convert to DataFrame for display
            df = pd.DataFrame(leaderboard)
            
            # Add rank emoji
            def rank_emoji(rank):
                if rank == 1:
                    return "🥇"
                elif rank == 2:
                    return "🥈"
                elif rank == 3:
                    return "🥉"
                elif rank <= 10:
                    return "⭐"
                return ""
            
            df['rank_display'] = df['rank'].apply(lambda x: f"{rank_emoji(x)} #{x}")
            
            # Highlight current user
            if user and not user.get('is_guest'):
                df['is_you'] = df['user_id'] == user['id']
            else:
                df['is_you'] = False
            
            # Display columns
            display_df = df[['rank_display', 'username', 'accuracy', 'correct', 'verified', 'total_predictions', 'avg_confidence']].copy()
            display_df.columns = ['Rank', 'User', 'Accuracy %', 'Correct', 'Verified', 'Total', 'Avg Confidence %']
            
            st.dataframe(
                display_df,
                column_config={
                    "Accuracy %": st.column_config.NumberColumn(format="%.1f%%"),
                    "Avg Confidence %": st.column_config.NumberColumn(format="%.1f%%")
                },
                hide_index=True,
                width="stretch"
            )
            
            # Your stats if not on leaderboard
            if user and not user.get('is_guest') and not user_rank:
                user_stats = self.db.get_user_stats(user['id'])
                if user_stats['verified'] < min_predictions:
                    st.info(f"📊 You need {min_predictions - user_stats['verified']} more verified predictions to appear on the leaderboard!")
        else:
            st.info("📭 No users have enough verified predictions yet. Be the first to make the leaderboard!")
            st.markdown(f"**Requirement:** At least {min_predictions} verified predictions")
    
    def render_head_to_head_tab(self, league: str):
        """Render the head-to-head analysis tab"""
        st.markdown("### ⚔️ Head-to-Head Analysis")
        st.markdown("Analyze historical matchups between two teams")
        
        # Get teams for the league
        try:
            data = self.fetcher.load_league_data(league)
            if data is None or data.empty:
                st.warning("No match data available. Please refresh data first.")
                return
            
            # Handle column name variations
            home_col = 'HomeTeam' if 'HomeTeam' in data.columns else 'home_team'
            away_col = 'AwayTeam' if 'AwayTeam' in data.columns else 'away_team'
            home_goals_col = 'FTHG' if 'FTHG' in data.columns else 'home_goals'
            away_goals_col = 'FTAG' if 'FTAG' in data.columns else 'away_goals'
            date_col = 'Date' if 'Date' in data.columns else 'date'
            
            teams = sorted(set(data[home_col].unique()) | set(data[away_col].unique()))
        except Exception as e:
            st.error(f"Could not load team data: {e}")
            return
        
        # Team selection
        col1, col2 = st.columns(2)
        
        with col1:
            team1 = st.selectbox("🏠 Select Team 1", teams, key="h2h_team1")
        with col2:
            team2_options = [t for t in teams if t != team1]
            team2 = st.selectbox("✈️ Select Team 2", team2_options, key="h2h_team2")
        
        if st.button("⚔️ Analyze Head-to-Head", type="primary"):
            # Get H2H matches from data
            h2h_matches = data[
                ((data[home_col] == team1) & (data[away_col] == team2)) |
                ((data[home_col] == team2) & (data[away_col] == team1))
            ].sort_values(date_col, ascending=False)
            
            if len(h2h_matches) == 0:
                st.info(f"No historical matches found between {team1} and {team2}")
                return
            
            st.divider()
            
            # Calculate H2H stats
            team1_wins = 0
            team2_wins = 0
            draws = 0
            team1_goals = 0
            team2_goals = 0
            
            for _, match in h2h_matches.iterrows():
                home = match[home_col]
                home_goals = match.get(home_goals_col, 0) or 0
                away_goals = match.get(away_goals_col, 0) or 0
                
                if home == team1:
                    team1_goals += home_goals
                    team2_goals += away_goals
                    if home_goals > away_goals:
                        team1_wins += 1
                    elif away_goals > home_goals:
                        team2_wins += 1
                    else:
                        draws += 1
                else:
                    team1_goals += away_goals
                    team2_goals += home_goals
                    if away_goals > home_goals:
                        team1_wins += 1
                    elif home_goals > away_goals:
                        team2_wins += 1
                    else:
                        draws += 1
            
            total_matches = len(h2h_matches)
            total_goals = team1_goals + team2_goals
            
            # H2H Summary
            st.markdown(f"#### 📊 {team1} vs {team2} - Last {total_matches} Meetings")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                win_pct = team1_wins / total_matches * 100
                st.markdown(f"""
                <div style="background: #e8f5e9; padding: 1rem; border-radius: 10px; text-align: center;">
                    <h4>🏆 {team1} Wins</h4>
                    <h2>{team1_wins}</h2>
                    <small>{win_pct:.1f}% of matches</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                draw_pct = draws / total_matches * 100
                st.markdown(f"""
                <div style="background: #fff3e0; padding: 1rem; border-radius: 10px; text-align: center;">
                    <h4>🤝 Draws</h4>
                    <h2>{draws}</h2>
                    <small>{draw_pct:.1f}% of matches</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                win_pct2 = team2_wins / total_matches * 100
                st.markdown(f"""
                <div style="background: #e3f2fd; padding: 1rem; border-radius: 10px; text-align: center;">
                    <h4>🏆 {team2} Wins</h4>
                    <h2>{team2_wins}</h2>
                    <small>{win_pct2:.1f}% of matches</small>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            # Goals analysis
            st.markdown("#### ⚽ Goals Analysis")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(f"{team1} Goals", team1_goals, f"{team1_goals/total_matches:.1f} per game")
            with col2:
                st.metric(f"{team2} Goals", team2_goals, f"{team2_goals/total_matches:.1f} per game")
            with col3:
                st.metric("Total Goals", total_goals, f"{total_goals/total_matches:.1f} per game")
            with col4:
                over_25 = sum(1 for _, m in h2h_matches.iterrows() 
                             if (m.get(home_goals_col, 0) or 0) + (m.get(away_goals_col, 0) or 0) > 2.5)
                st.metric("Over 2.5 Goals", f"{over_25}/{total_matches}", f"{over_25/total_matches*100:.0f}%")
            
            st.divider()
            
            # BTTS analysis
            btts_yes = sum(1 for _, m in h2h_matches.iterrows() 
                          if (m.get(home_goals_col, 0) or 0) > 0 and (m.get(away_goals_col, 0) or 0) > 0)
            
            st.markdown("#### 🔄 Both Teams to Score")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("BTTS Yes", btts_yes, f"{btts_yes/total_matches*100:.0f}%")
            with col2:
                st.metric("BTTS No", total_matches - btts_yes, f"{(total_matches-btts_yes)/total_matches*100:.0f}%")
            
            st.divider()
            
            # Recent matches
            st.markdown("#### 📜 Recent Matches")
            
            for idx, match in h2h_matches.head(10).iterrows():
                home = match[home_col]
                away = match[away_col]
                hg = match.get(home_goals_col, '?')
                ag = match.get(away_goals_col, '?')
                dt = match.get(date_col, 'Unknown')
                
                # Determine winner emoji
                try:
                    if hg > ag:
                        result_emoji = "🏠" if home == team1 else "✈️"
                    elif ag > hg:
                        result_emoji = "✈️" if away == team1 else "🏠"
                    else:
                        result_emoji = "🤝"
                except:
                    result_emoji = "⚽"
                
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 0.5rem 1rem; border-radius: 5px; margin-bottom: 0.5rem;">
                    {result_emoji} <strong>{home}</strong> {hg} - {ag} <strong>{away}</strong>
                    <span style="float: right; color: #666;">{str(dt)[:10]}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            # Prediction recommendation
            st.markdown("#### 🔮 AI Recommendation")
            
            # Simple recommendation based on H2H
            if team1_wins > team2_wins:
                rec = f"{team1} has the edge in H2H ({team1_wins} wins vs {team2_wins})"
                rec_result = "Home Win" if team1 == teams[0] else "Away Win"
            elif team2_wins > team1_wins:
                rec = f"{team2} has the edge in H2H ({team2_wins} wins vs {team1_wins})"
                rec_result = "Away Win" if team2 == teams[0] else "Home Win"
            else:
                rec = f"Evenly matched ({draws} draws in {total_matches} matches)"
                rec_result = "Draw"
            
            avg_goals = total_goals / total_matches
            goals_rec = "Over 2.5 Goals" if avg_goals > 2.5 else "Under 2.5 Goals"
            btts_rec = "BTTS Yes" if btts_yes / total_matches > 0.5 else "BTTS No"
            
            st.info(f"""
            **Based on H2H history:**
            - 🏆 **Result:** {rec}
            - ⚽ **Goals:** {goals_rec} (avg: {avg_goals:.1f} per game)
            - 🔄 **BTTS:** {btts_rec} ({btts_yes/total_matches*100:.0f}% of matches)
            
            *For more accurate predictions, use the Predict Match tab which considers recent form and full statistics.*
            """)

    def render_team_form_tab(self, league: str):
        """Render the team form analysis tab"""
        st.markdown("### 📊 Team Form Analysis")
        st.markdown("View recent performance and trends for any team")
        
        try:
            df = self.fetcher.load_league_data(league)
            if df is None or df.empty:
                st.warning("No match data available. Please refresh data first.")
                return
            
            teams = sorted(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))
        except Exception as e:
            st.error(f"Could not load team data: {e}")
            return
        
        # Team selection
        selected_team = st.selectbox("🏟️ Select Team", teams, key="form_team")
        n_matches = st.slider("Number of matches to analyze", 5, 20, 10)
        
        if st.button("📈 Analyze Form", type="primary"):
            form = self.get_team_form(league, selected_team, n_matches)
            
            if not form:
                st.warning("No recent matches found for this team.")
                return
            
            st.divider()
            
            # Form summary
            wins = sum(1 for m in form if m['result'] == 'W')
            draws = sum(1 for m in form if m['result'] == 'D')
            losses = sum(1 for m in form if m['result'] == 'L')
            goals_scored = sum(m['goals_scored'] for m in form)
            goals_conceded = sum(m['goals_conceded'] for m in form)
            
            st.markdown(f"#### 📊 {selected_team} - Last {len(form)} Games")
            
            # Form string
            form_str = ""
            for match in form:
                if match['result'] == 'W':
                    form_str += "🟢 "
                elif match['result'] == 'D':
                    form_str += "🟡 "
                else:
                    form_str += "🔴 "
            
            st.markdown(f"**Form:** {form_str}")
            
            # Stats
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("🏆 Wins", wins, f"{wins/len(form)*100:.0f}%")
            with col2:
                st.metric("🤝 Draws", draws, f"{draws/len(form)*100:.0f}%")
            with col3:
                st.metric("❌ Losses", losses, f"{losses/len(form)*100:.0f}%")
            with col4:
                st.metric("⚽ Goals For", goals_scored, f"{goals_scored/len(form):.1f}/game")
            with col5:
                st.metric("🥅 Goals Against", goals_conceded, f"{goals_conceded/len(form):.1f}/game")
            
            # Points per game
            points = wins * 3 + draws
            ppg = points / len(form)
            
            st.markdown(f"**Points per game:** {ppg:.2f}")
            
            # Form chart
            st.markdown("#### 📈 Recent Results")
            
            # Create form chart data
            form_data = []
            for i, match in enumerate(reversed(form)):
                pts = 3 if match['result'] == 'W' else 1 if match['result'] == 'D' else 0
                form_data.append({
                    'Match': len(form) - i,
                    'Points': pts,
                    'Goals Scored': match['goals_scored'],
                    'Goals Conceded': match['goals_conceded']
                })
            
            form_df = pd.DataFrame(form_data)
            st.line_chart(form_df.set_index('Match')[['Points', 'Goals Scored', 'Goals Conceded']])
            
            # Recent matches list
            st.markdown("#### 📜 Match Details")
            
            for match in form:
                venue = "🏠 Home" if match['is_home'] else "✈️ Away"
                result_emoji = "🟢" if match['result'] == 'W' else "🟡" if match['result'] == 'D' else "🔴"
                
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 0.75rem; border-radius: 8px; margin-bottom: 0.5rem;">
                    {result_emoji} <strong>{match['result']}</strong> - {venue} vs {match['opponent']}
                    <span style="float: right;">{match['goals_scored']} - {match['goals_conceded']}</span>
                    <br><small style="color: #666;">{match['date']}</small>
                </div>
                """, unsafe_allow_html=True)
            
            # Trend analysis
            st.markdown("#### 📊 Trend Analysis")
            
            recent_3 = form[:3]
            older_3 = form[3:6] if len(form) >= 6 else form[3:]
            
            if older_3:
                recent_ppg = sum(3 if m['result'] == 'W' else 1 if m['result'] == 'D' else 0 for m in recent_3) / len(recent_3)
                older_ppg = sum(3 if m['result'] == 'W' else 1 if m['result'] == 'D' else 0 for m in older_3) / len(older_3)
                
                ppg_diff = recent_ppg - older_ppg
                
                if ppg_diff > 0.3:
                    st.success(f"📈 **Improving Form!** Recent PPG: {recent_ppg:.2f} vs Earlier: {older_ppg:.2f} (+{ppg_diff:.2f})")
                elif ppg_diff < -0.3:
                    st.error(f"📉 **Declining Form!** Recent PPG: {recent_ppg:.2f} vs Earlier: {older_ppg:.2f} ({ppg_diff:.2f})")
                else:
                    st.info(f"📊 **Stable Form** - Recent PPG: {recent_ppg:.2f} vs Earlier: {older_ppg:.2f}")
    
    def render_accumulator_tab(self, league: str):
        """Render the accumulator builder tab"""
        st.markdown("### 🎰 Accumulator Builder")
        st.markdown("Build your accumulator by adding selections from predictions")
        
        accumulator = st.session_state.accumulator
        
        if not accumulator:
            st.info("📭 Your accumulator is empty. Add selections from the Predict Match tab!")
            st.markdown("""
            **How to use:**
            1. Go to the **Predict Match** tab
            2. Select a match and get predictions
            3. Click the **➕ Add to Accumulator** buttons
            4. Come back here to view your accumulator
            """)
            return
        
        # Display accumulator
        st.markdown(f"#### 📋 Your Selections ({len(accumulator)})")
        
        total_odds = 1.0
        combined_probability = 1.0
        
        for i, selection in enumerate(accumulator):
            total_odds *= selection['odds']
            combined_probability *= (selection['probability'] / 100)
            
            col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
            
            with col1:
                st.markdown(f"**{selection['match']}**")
            with col2:
                st.markdown(f"{selection['selection']}")
            with col3:
                st.markdown(f"@{selection['odds']:.2f}")
            with col4:
                if st.button("❌", key=f"remove_{i}"):
                    st.session_state.accumulator.pop(i)
                    st.rerun()
        
        st.divider()
        
        # Accumulator summary
        st.markdown("#### 📊 Accumulator Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📈 Total Odds", f"{total_odds:.2f}")
        with col2:
            st.metric("🎯 Combined Probability", f"{combined_probability*100:.2f}%")
        with col3:
            # Confidence level
            if combined_probability >= 0.20:
                conf_level = "🟢 HIGH"
            elif combined_probability >= 0.10:
                conf_level = "🟡 MEDIUM"
            elif combined_probability >= 0.05:
                conf_level = "🟠 LOW"
            else:
                conf_level = "🔴 VERY LOW"
            st.metric("💪 Confidence", conf_level)
        
        # Potential returns calculator
        st.markdown("#### 💰 Potential Returns Calculator")
        
        stake = st.number_input("Enter stake amount", min_value=1.0, max_value=10000.0, value=10.0, step=1.0)
        
        potential_return = stake * total_odds
        potential_profit = potential_return - stake
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
                <h4 style="margin:0;">Potential Return</h4>
                <h2 style="margin:0.5rem 0;">${potential_return:.2f}</h2>
                <small>on ${stake:.2f} stake</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                        padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
                <h4 style="margin:0;">Potential Profit</h4>
                <h2 style="margin:0.5rem 0;">${potential_profit:.2f}</h2>
                <small>{total_odds:.2f}x multiplier</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Expected value analysis
        st.markdown("#### 📉 Expected Value Analysis")
        
        ev = (combined_probability * potential_return) - stake
        
        if ev > 0:
            st.success(f"✅ **Positive Expected Value:** ${ev:.2f}")
            st.markdown("This accumulator has positive expected value based on our model probabilities.")
        else:
            st.warning(f"⚠️ **Negative Expected Value:** ${ev:.2f}")
            st.markdown("This accumulator has negative expected value. Consider removing low-probability selections.")
        
        # Actions
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear All Selections", type="secondary", width="stretch"):
                st.session_state.accumulator = []
                st.rerun()
        
        with col2:
            if st.button("📋 Copy Accumulator", width="stretch"):
                # Format accumulator as text
                acca_text = "🎰 Accumulator:\n"
                for sel in accumulator:
                    acca_text += f"- {sel['match']}: {sel['selection']} @{sel['odds']:.2f}\n"
                acca_text += f"\nTotal Odds: {total_odds:.2f}"
                acca_text += f"\nStake: ${stake:.2f} → Returns: ${potential_return:.2f}"
                
                st.code(acca_text, language=None)
                st.success("Accumulator formatted above - copy and share!")
        
        # Tips
        st.divider()
        st.markdown("#### 💡 Tips for Better Accumulators")
        st.markdown("""
        - 🎯 **Fewer selections = Higher chance of winning** - Keep it to 3-5 selections
        - 📊 **Focus on high-confidence picks** - Look for 60%+ probability selections
        - 🔍 **Diversify your markets** - Mix results with goals/BTTS predictions
        - ⚠️ **Avoid long shots** - Low probability selections tank your EV
        """)

    def render_live_odds_tab(self, league: str):
        """Render live odds comparison tab"""
        st.markdown("### 💹 Live Odds Comparison")
        st.markdown("Compare real-time odds from multiple bookmakers")
        
        # League mapping for odds API - use OddsAPI's mapping
        # Include alternate names for flexibility
        league_mapping = {
            # English leagues
            'EPL': 'soccer_epl',
            'CHAMPIONSHIP': 'soccer_efl_champ',
            'EFL_CHAMPIONSHIP': 'soccer_efl_champ',
            'LEAGUE_ONE': 'soccer_england_league1',
            'EFL_LEAGUE1': 'soccer_england_league1',
            'LEAGUE_TWO': 'soccer_england_league2',
            'EFL_LEAGUE2': 'soccer_england_league2',
            'SCOTTISH_PREM': 'soccer_spl',
            # Spanish leagues
            'LA_LIGA': 'soccer_spain_la_liga',
            'LA_LIGA_2': 'soccer_spain_segunda_division',
            'LA_LIGA2': 'soccer_spain_segunda_division',
            # Italian leagues
            'SERIE_A': 'soccer_italy_serie_a',
            # German leagues
            'BUNDESLIGA': 'soccer_germany_bundesliga',
            'BUNDESLIGA_2': 'soccer_germany_bundesliga2',
            'BUNDESLIGA2': 'soccer_germany_bundesliga2',
            # French leagues
            'LIGUE_1': 'soccer_france_ligue_one',
            'LIGUE_2': 'soccer_france_ligue_two',
            # Other European leagues
            'EREDIVISIE': 'soccer_netherlands_eredivisie',
            'PRIMEIRA_LIGA': 'soccer_portugal_primeira_liga',
            'PRIMEIRA': 'soccer_portugal_primeira_liga',
            'SUPER_LIG': 'soccer_turkey_super_league',
            'SUPER_LEAGUE_GR': 'soccer_greece_super_league',
            'GREEK_SL': 'soccer_greece_super_league',
        }
        
        sport_key = league_mapping.get(league)
        
        if not sport_key:
            st.info(f"📭 Live odds not available for this league yet")
            st.caption("Live odds are available for: EPL, Championship, La Liga, Serie A, Bundesliga, Ligue 1, and more.")
            return
        
        # Check API key
        if not self.odds_api.api_key:
            st.warning("⚠️ Odds API not configured")
            st.markdown("""
            **To enable live odds:**
            1. Get a free API key from [The Odds API](https://the-odds-api.com/)
            2. Set environment variable: `ODDS_API_KEY=your_key`
            3. Restart the application
            
            *Free tier includes 500 requests/month*
            """)
            
            # Demo mode
            st.divider()
            st.markdown("#### 📊 Demo Mode - Sample Odds Display")
            demo_odds = [
                {"home_team": "Manchester City", "away_team": "Arsenal", "bookmakers": [
                    {"title": "Bet365", "home": 1.75, "draw": 3.80, "away": 4.20},
                    {"title": "Unibet", "home": 1.72, "draw": 3.90, "away": 4.30}
                ]},
                {"home_team": "Liverpool", "away_team": "Chelsea", "bookmakers": [
                    {"title": "Bet365", "home": 1.90, "draw": 3.50, "away": 3.80},
                    {"title": "Unibet", "home": 1.85, "draw": 3.60, "away": 3.90}
                ]}
            ]
            
            for match in demo_odds:
                st.markdown(f"**{match['home_team']} vs {match['away_team']}**")
                odds_data = []
                for bookie in match['bookmakers']:
                    odds_data.append({
                        "Bookmaker": bookie['title'],
                        "Home": f"{bookie['home']:.2f}",
                        "Draw": f"{bookie['draw']:.2f}",
                        "Away": f"{bookie['away']:.2f}"
                    })
                st.dataframe(pd.DataFrame(odds_data), hide_index=True)
                st.divider()
            return
        
        # Fetch live odds
        with st.spinner("Fetching live odds..."):
            odds_data = self.odds_api.get_odds(sport_key)
        
        if not odds_data:
            st.info("📭 No upcoming matches with odds available")
            return
        
        # Display usage info
        st.caption(f"📊 API Requests: {self.odds_api.requests_used}/{500} used this month")
        
        # Market selection
        market = st.selectbox(
            "Select Market",
            options=["h2h", "totals", "spreads"],
            format_func=lambda x: {"h2h": "1X2 (Match Result)", "totals": "Over/Under", "spreads": "Asian Handicap"}[x]
        )
        
        st.divider()
        
        # Display odds for each match
        for match in odds_data[:10]:  # Limit to 10 matches
            home = match.get('home_team', 'Home')
            away = match.get('away_team', 'Away')
            
            with st.expander(f"⚽ {home} vs {away}", expanded=False):
                bookmakers = match.get('bookmakers', [])
                
                if not bookmakers:
                    st.info("No odds available for this match")
                    continue
                
                odds_table = []
                best_home = 0
                best_draw = 0
                best_away = 0
                
                for bookie in bookmakers:
                    bookie_name = bookie.get('title', 'Unknown')
                    markets = bookie.get('markets', [])
                    
                    for m in markets:
                        if m.get('key') == market:
                            outcomes = m.get('outcomes', [])
                            row = {"Bookmaker": bookie_name}
                            
                            for outcome in outcomes:
                                name = outcome.get('name', '')
                                price = outcome.get('price', 0)
                                
                                if name == home:
                                    row['Home'] = price
                                    best_home = max(best_home, price)
                                elif name == 'Draw':
                                    row['Draw'] = price
                                    best_draw = max(best_draw, price)
                                elif name == away:
                                    row['Away'] = price
                                    best_away = max(best_away, price)
                                elif 'Over' in name:
                                    row['Over 2.5'] = price
                                elif 'Under' in name:
                                    row['Under 2.5'] = price
                            
                            odds_table.append(row)
                
                if odds_table:
                    df = pd.DataFrame(odds_table)
                    st.dataframe(df, hide_index=True, width="stretch")
                    
                    if market == 'h2h' and best_home > 0:
                        st.markdown(f"""
                        **🏆 Best Odds:** Home: **{best_home:.2f}** | Draw: **{best_draw:.2f}** | Away: **{best_away:.2f}**
                        """)
        
        # Value Bets section
        st.divider()
        st.markdown("#### 💎 Value Bets Finder")
        st.markdown("Find bets where bookmaker odds exceed our predicted fair odds")
        
        if st.button("🔍 Find Value Bets", type="primary"):
            try:
                df = self.fetcher.load_league_data(league)
                
                if df is not None and len(df) > 0:
                    teams = sorted(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))
                    
                    value_bets = []
                    for match in odds_data[:5]:
                        home = match.get('home_team', '')
                        away = match.get('away_team', '')
                        
                        # Try to match team names
                        home_match = next((t for t in teams if t.lower() in home.lower() or home.lower() in t.lower()), None)
                        away_match = next((t for t in teams if t.lower() in away.lower() or away.lower() in t.lower()), None)
                        
                        if home_match and away_match:
                            # Get our prediction
                            prediction = self.predictor.predict_match(df, home_match, away_match)
                            
                            if prediction:
                                our_probs = prediction['probabilities']
                                
                                # Get best odds
                                best = self.odds_api.get_best_odds(match)
                                
                                for outcome, prob in [('Home', our_probs['home']), 
                                                      ('Draw', our_probs['draw']), 
                                                      ('Away', our_probs['away'])]:
                                    if outcome in best:
                                        implied_prob = 1 / best[outcome]
                                        value = (prob / 100) * best[outcome] - 1
                                        
                                        if value > 0.05:  # 5%+ edge
                                            value_bets.append({
                                                'Match': f"{home} vs {away}",
                                                'Bet': outcome,
                                                'Odds': f"{best[outcome]:.2f}",
                                                'Our Prob': f"{prob:.1f}%",
                                                'Implied Prob': f"{implied_prob*100:.1f}%",
                                                'Value': f"+{value*100:.1f}%"
                                            })
                    
                    if value_bets:
                        st.success(f"Found {len(value_bets)} value bet(s)!")
                        st.dataframe(pd.DataFrame(value_bets), hide_index=True)
                    else:
                        st.info("No significant value bets found in current odds")
                else:
                    st.warning("Load match data first to find value bets")
            except Exception as e:
                st.error(f"Error finding value bets: {e}")

    def render_player_stats_tab(self, league: str):
        """Render player statistics tab"""
        st.markdown("### ⚽ Player Statistics")
        st.markdown("Top scorers, assists, and player performance data")
        
        # Check API key
        if not self.football_api.api_key:
            st.warning("⚠️ Football API not configured")
            st.markdown("""
            **To enable player stats:**
            1. Get a free API key from [API-Football on RapidAPI](https://rapidapi.com/api-sports/api/api-football)
            2. Set environment variable: `FOOTBALL_API_KEY=your_key`
            3. Restart the application
            
            *Free tier includes 100 requests/day*
            """)
            
            # Demo mode
            st.divider()
            st.markdown("#### 📊 Demo Mode - Sample Stats")
            
            demo_scorers = [
                {"Player": "Erling Haaland", "Team": "Manchester City", "Goals": 27, "Penalties": 5},
                {"Player": "Mohamed Salah", "Team": "Liverpool", "Goals": 19, "Penalties": 3},
                {"Player": "Cole Palmer", "Team": "Chelsea", "Goals": 18, "Penalties": 4},
                {"Player": "Alexander Isak", "Team": "Newcastle", "Goals": 17, "Penalties": 0},
                {"Player": "Ollie Watkins", "Team": "Aston Villa", "Goals": 15, "Penalties": 2}
            ]
            
            demo_assists = [
                {"Player": "Kevin De Bruyne", "Team": "Manchester City", "Assists": 14},
                {"Player": "Bukayo Saka", "Team": "Arsenal", "Assists": 12},
                {"Player": "Bruno Fernandes", "Team": "Man United", "Assists": 11},
                {"Player": "Cole Palmer", "Team": "Chelsea", "Assists": 10},
                {"Player": "Jarrod Bowen", "Team": "West Ham", "Assists": 9}
            ]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### 🥇 Top Scorers")
                st.dataframe(pd.DataFrame(demo_scorers), hide_index=True)
            
            with col2:
                st.markdown("##### 🎯 Top Assists")
                st.dataframe(pd.DataFrame(demo_assists), hide_index=True)
            return
        
        # API is configured - fetch real data
        current_season = datetime.now().year if datetime.now().month >= 8 else datetime.now().year - 1
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 🥇 Top Scorers")
            
            with st.spinner("Fetching top scorers..."):
                scorers = self.football_api.get_top_scorers(league, current_season)
            
            if scorers:
                scorers_data = []
                for player in scorers[:10]:
                    player_info = player.get('player', {})
                    stats = player.get('statistics', [{}])[0]
                    goals_data = stats.get('goals', {})
                    team_info = stats.get('team', {})
                    
                    scorers_data.append({
                        "Player": player_info.get('name', 'Unknown'),
                        "Team": team_info.get('name', 'Unknown'),
                        "Goals": goals_data.get('total', 0) or 0,
                        "Penalties": goals_data.get('penalties', 0) or 0
                    })
                
                st.dataframe(pd.DataFrame(scorers_data), hide_index=True)
            else:
                st.info("No scorer data available")
        
        with col2:
            st.markdown("##### 🎯 Top Assists")
            
            with st.spinner("Fetching top assists..."):
                assists = self.football_api.get_top_assists(league, current_season)
            
            if assists:
                assists_data = []
                for player in assists[:10]:
                    player_info = player.get('player', {})
                    stats = player.get('statistics', [{}])[0]
                    goals_data = stats.get('goals', {})
                    team_info = stats.get('team', {})
                    
                    assists_data.append({
                        "Player": player_info.get('name', 'Unknown'),
                        "Team": team_info.get('name', 'Unknown'),
                        "Assists": goals_data.get('assists', 0) or 0
                    })
                
                st.dataframe(pd.DataFrame(assists_data), hide_index=True)
            else:
                st.info("No assist data available")
        
        # Usage info
        st.caption(f"📊 API Requests Today: {self.football_api.requests_today}/100")

    def render_injuries_tab(self, league: str):
        """Render injuries and suspensions tab"""
        st.markdown("### 🏥 Injuries & Suspensions")
        st.markdown("Check team injury news before making predictions")
        
        # Check API key
        if not self.football_api.api_key:
            st.warning("⚠️ Football API not configured")
            st.markdown("""
            **To enable injury news:**
            1. Get a free API key from [API-Football on RapidAPI](https://rapidapi.com/api-sports/api/api-football)
            2. Set environment variable: `FOOTBALL_API_KEY=your_key`
            3. Restart the application
            
            *Free tier includes 100 requests/day*
            """)
            
            # Demo mode
            st.divider()
            st.markdown("#### 📊 Demo Mode - Sample Injury Report")
            
            demo_injuries = {
                "Manchester City": [
                    {"Player": "Rodri", "Type": "ACL", "Status": "Out for season"},
                    {"Player": "Nathan Ake", "Type": "Hamstring", "Status": "2-3 weeks"}
                ],
                "Arsenal": [
                    {"Player": "Bukayo Saka", "Type": "Muscle", "Status": "Doubtful"},
                    {"Player": "Takehiro Tomiyasu", "Type": "Knee", "Status": "4-6 weeks"}
                ],
                "Liverpool": [
                    {"Player": "Diogo Jota", "Type": "Muscle", "Status": "1-2 weeks"}
                ]
            }
            
            for team, injuries in demo_injuries.items():
                with st.expander(f"🏥 {team} ({len(injuries)} injured)"):
                    st.dataframe(pd.DataFrame(injuries), hide_index=True)
            return
        
        # API is configured
        try:
            df = self.fetcher.load_league_data(league)
            
            if df is not None and len(df) > 0:
                teams = sorted(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))
                
                selected_team = st.selectbox("Select Team", options=teams)
                
                if selected_team:
                    with st.spinner(f"Fetching injury report for {selected_team}..."):
                        injuries = self.football_api.get_team_injuries(selected_team, league)
                    
                    if injuries:
                        st.markdown(f"#### 🏥 {selected_team} Injury Report")
                        
                        injury_data = []
                        for inj in injuries:
                            player = inj.get('player', {})
                            injury_info = inj.get('player', {}).get('reason', 'Unknown')
                            
                            injury_data.append({
                                "Player": player.get('name', 'Unknown'),
                                "Type": injury_info,
                                "Status": inj.get('player', {}).get('type', 'Unknown')
                            })
                        
                        st.dataframe(pd.DataFrame(injury_data), hide_index=True, width="stretch")
                        
                        st.warning(f"⚠️ {len(injuries)} player(s) currently unavailable")
                    else:
                        st.success(f"✅ No injuries reported for {selected_team}")
            else:
                st.info("Load match data first to view team injuries")
                
        except Exception as e:
            st.error(f"Error fetching injuries: {e}")
        
        # Usage info
        st.caption(f"📊 API Requests Today: {self.football_api.requests_today}/100")

    def render_team_explorer_tab(self, league: str):
        """Render Team Explorer tab with TheSportsDB data"""
        st.markdown("### 🏟️ Team Explorer")
        st.caption("Explore teams, players, and stadiums • Powered by TheSportsDB (FREE)")
        
        # League mapping for TheSportsDB
        sportsdb_leagues = {
            'EPL': 'EPL',
            'CHAMPIONSHIP': 'CHAMPIONSHIP', 
            'EFL_CHAMPIONSHIP': 'CHAMPIONSHIP',
            'LA_LIGA': 'LA_LIGA',
            'SERIE_A': 'SERIE_A',
            'BUNDESLIGA': 'BUNDESLIGA',
            'LIGUE_1': 'LIGUE_1',
            'EREDIVISIE': 'EREDIVISIE',
            'SCOTTISH_PREM': 'SCOTTISH_PREM',
        }
        
        sdb_league = sportsdb_leagues.get(league, 'EPL')
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Get league teams
            with st.spinner("Loading teams..."):
                teams = self.sportsdb.get_league_teams(sdb_league)
            
            if teams:
                team_options = [t['name'] for t in teams]
                selected_team = st.selectbox("🏟️ Select Team", team_options, key="sdb_team")
                
                # Find team info
                selected_info = next((t for t in teams if t['name'] == selected_team), None)
                
                if selected_info and selected_info.get('logo'):
                    st.image(selected_info['logo'], width=150)
            else:
                # Fallback: manual search
                st.info(f"League not in database. Search manually:")
                selected_team = st.text_input("🔍 Search Team", placeholder="e.g. Arsenal")
                selected_info = None
        
        with col2:
            if selected_team:
                with st.spinner(f"Loading {selected_team} details..."):
                    team_data = self.sportsdb.get_team_info(selected_team)
                
                if team_data:
                    st.markdown(f"## {team_data['name']}")
                    
                    # Team info cards
                    info_col1, info_col2, info_col3 = st.columns(3)
                    
                    with info_col1:
                        st.markdown("**🏟️ Stadium**")
                        st.markdown(f"{team_data.get('stadium', 'N/A')}")
                        if team_data.get('stadium_capacity'):
                            try:
                                capacity = int(team_data['stadium_capacity'])
                                st.caption(f"Capacity: {capacity:,}")
                            except (ValueError, TypeError):
                                st.caption(f"Capacity: {team_data['stadium_capacity']}")
                    
                    with info_col2:
                        st.markdown("**📍 Location**")
                        st.markdown(f"{team_data.get('location', 'N/A')}")
                        st.caption(f"{team_data.get('country', '')}")
                    
                    with info_col3:
                        st.markdown("**📅 Founded**")
                        st.markdown(f"{team_data.get('formed_year', 'N/A')}")
                    
                    # Stadium image
                    if team_data.get('stadium_image'):
                        st.image(team_data['stadium_image'], caption=f"{team_data.get('stadium', 'Stadium')}", use_container_width=True)
                    
                    # Banner
                    if team_data.get('banner'):
                        st.image(team_data['banner'], use_container_width=True)
                    
                    # Social links
                    socials = []
                    if team_data.get('website'):
                        socials.append(f"[🌐 Website]({team_data['website']})")
                    if team_data.get('twitter'):
                        socials.append(f"[🐦 Twitter]({team_data['twitter']})")
                    if team_data.get('instagram'):
                        socials.append(f"[📸 Instagram]({team_data['instagram']})")
                    if team_data.get('facebook'):
                        socials.append(f"[📘 Facebook]({team_data['facebook']})")
                    
                    if socials:
                        st.markdown(" • ".join(socials))
                else:
                    st.warning(f"Could not find details for {selected_team}")
        
        st.divider()
        
        # Squad section
        if selected_team:
            st.markdown("### 👥 Squad")
            
            with st.spinner("Loading squad..."):
                players = self.sportsdb.get_team_players(selected_team)
            
            if players:
                # Group by position
                positions = {}
                for p in players:
                    pos = p.get('position', 'Unknown') or 'Unknown'
                    if pos not in positions:
                        positions[pos] = []
                    positions[pos].append(p)
                
                # Display in columns by position
                position_order = ['Goalkeeper', 'Defender', 'Midfielder', 'Forward', 'Unknown']
                
                for pos in position_order:
                    if pos in positions:
                        st.markdown(f"**{pos}s**")
                        cols = st.columns(min(5, len(positions[pos])))
                        
                        for i, player in enumerate(positions[pos][:10]):  # Max 10 per position
                            with cols[i % 5]:
                                if player.get('cutout') or player.get('photo'):
                                    st.image(player.get('cutout') or player.get('photo'), width=80)
                                st.markdown(f"**{player['name']}**")
                                if player.get('nationality'):
                                    st.caption(f"🏳️ {player['nationality']}")
                                if player.get('number'):
                                    st.caption(f"#{player['number']}")
            else:
                st.info("No squad data available")
        
        st.divider()
        
        # Recent results & next match
        if selected_team:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Recent Results")
                results = self.sportsdb.get_team_last_matches(selected_team, limit=5)
                
                if results:
                    for match in results:
                        home = match.get('home_team', 'TBD')
                        away = match.get('away_team', 'TBD')
                        h_score = match.get('home_score', '-')
                        a_score = match.get('away_score', '-')
                        date = match.get('date', '')
                        
                        # Determine if selected team won/lost/drew
                        is_home = home == selected_team
                        if h_score != '-' and a_score != '-':
                            h_score, a_score = int(h_score), int(a_score)
                            if (is_home and h_score > a_score) or (not is_home and a_score > h_score):
                                result_emoji = "🟢"  # Win
                            elif h_score == a_score:
                                result_emoji = "🟡"  # Draw
                            else:
                                result_emoji = "🔴"  # Loss
                        else:
                            result_emoji = "⚪"
                        
                        st.markdown(f"{result_emoji} **{home}** {h_score} - {a_score} **{away}**")
                        st.caption(f"📅 {date}")
                else:
                    st.info("No recent results available")
            
            with col2:
                st.markdown("### ⏭️ Next Match")
                next_match = self.sportsdb.get_team_next_match(selected_team)
                
                if next_match:
                    st.markdown(f"**{next_match.get('home_team', 'TBD')}** vs **{next_match.get('away_team', 'TBD')}**")
                    st.markdown(f"📅 {next_match.get('date', 'TBD')} at {next_match.get('time', 'TBD')}")
                    st.markdown(f"🏟️ {next_match.get('venue', 'TBD')}")
                    st.caption(f"🏆 {next_match.get('league', '')}")
                else:
                    st.info("No upcoming matches scheduled")
        
        # Search player
        st.divider()
        st.markdown("### 🔍 Search Player")
        
        player_search = st.text_input("Search for any player", placeholder="e.g. Salah, Haaland, Mbappe")
        
        if player_search:
            with st.spinner(f"Searching for {player_search}..."):
                player_info = self.sportsdb.get_player_info(player_search)
            
            if player_info:
                pcol1, pcol2 = st.columns([1, 2])
                
                with pcol1:
                    if player_info.get('cutout'):
                        st.image(player_info['cutout'], width=200)
                    elif player_info.get('photo'):
                        st.image(player_info['photo'], width=200)
                
                with pcol2:
                    st.markdown(f"## {player_info['name']}")
                    st.markdown(f"**Team:** {player_info.get('team', 'N/A')}")
                    st.markdown(f"**Position:** {player_info.get('position', 'N/A')}")
                    st.markdown(f"**Nationality:** {player_info.get('nationality', 'N/A')}")
                    
                    if player_info.get('date_of_birth'):
                        st.markdown(f"**Born:** {player_info['date_of_birth']}")
                    if player_info.get('height'):
                        st.markdown(f"**Height:** {player_info['height']}")
                    
                    # Social links
                    psocials = []
                    if player_info.get('twitter'):
                        psocials.append(f"[🐦 Twitter]({player_info['twitter']})")
                    if player_info.get('instagram'):
                        psocials.append(f"[📸 Instagram]({player_info['instagram']})")
                    if psocials:
                        st.markdown(" • ".join(psocials))
            else:
                st.warning(f"Could not find player: {player_search}")
        
        st.caption("📊 Data from TheSportsDB • Free API • No rate limits")

    def render_auth_page(self):
        """Render the authentication page (login/register)"""
        st.markdown('<h1 class="main-header">⚽ Football Match Predictor</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">AI-powered predictions using real football data</p>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Auth mode tabs
            tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
            
            with tab1:
                st.markdown("### Welcome Back!")
                
                login_email = st.text_input("Email", key="login_email", placeholder="your@email.com")
                login_password = st.text_input("Password", type="password", key="login_password")
                
                if st.button("🔐 Login", type="primary", width="stretch"):
                    if login_email and login_password:
                        success, message, token = self.auth.login(login_email, login_password)
                        if success:
                            st.session_state.session_token = token
                            user = self.auth.validate_session(token)
                            st.session_state.user = user
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
                    else:
                        st.warning("Please enter email and password")
                
                st.markdown("---")
                st.markdown("**Or continue as guest** (limited to 3 predictions/day, EPL only)")
                
                if st.button("👤 Continue as Guest", width="stretch"):
                    # Create a guest session
                    st.session_state.user = {
                        'id': None,
                        'email': 'guest',
                        'subscription_tier': 'free',
                        'tier_info': self.auth.SUBSCRIPTION_TIERS['free'],
                        'predictions_today': 0,
                        'is_guest': True
                    }
                    st.rerun()
            
            with tab2:
                st.markdown("### Create Your Account")
                st.markdown("Get started with **3 free predictions per day**!")
                
                reg_username = st.text_input("Username", key="reg_username", placeholder="Your display name")
                reg_email = st.text_input("Email", key="reg_email", placeholder="your@email.com")
                reg_password = st.text_input("Password", type="password", key="reg_password", 
                                            help="Min 8 chars, 1 uppercase, 1 lowercase, 1 number")
                reg_password2 = st.text_input("Confirm Password", type="password", key="reg_password2")
                
                if st.button("📝 Create Account", type="primary", width="stretch"):
                    if not reg_email or not reg_password:
                        st.warning("Please fill in all fields")
                    elif reg_password != reg_password2:
                        st.error("Passwords do not match")
                    else:
                        success, message, user_id = self.auth.register(reg_email, reg_password, reg_username or None)
                        if success:
                            st.success(message)
                            # Auto-login after registration
                            success, _, token = self.auth.login(reg_email, reg_password)
                            if success:
                                st.session_state.session_token = token
                                st.session_state.user = self.auth.validate_session(token)
                                st.rerun()
                        else:
                            st.error(message)
            
            # Pricing info
            st.markdown("---")
            st.markdown("### 💎 Upgrade for More")
            
            pricing = self.payments.get_pricing_display()
            
            cols = st.columns(3)
            for i, (tier, info) in enumerate(pricing.items()):
                with cols[i]:
                    popular_badge = "⭐ POPULAR" if info.get('popular') else ""
                    st.markdown(f"""
                    <div style="background: {'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' if info.get('popular') else '#f8f9fa'}; 
                                padding: 1rem; border-radius: 10px; text-align: center;
                                color: {'white' if info.get('popular') else 'black'};">
                        <small>{popular_badge}</small>
                        <h4 style="margin:0;">{info['name']}</h4>
                        <h2 style="margin:0.5rem 0;">${info['monthly']}/mo</h2>
                        <small>or ${info['yearly']}/year (save {info['savings']}%)</small>
                    </div>
                    """, unsafe_allow_html=True)
    
    def render_user_menu(self):
        """Render user menu in sidebar"""
        user = st.session_state.user
        
        if user:
            tier = user.get('subscription_tier', 'free')
            tier_info = user.get('tier_info', self.auth.SUBSCRIPTION_TIERS['free'])
            
            st.markdown("### 👤 Account")
            
            if user.get('is_guest'):
                st.info("👤 Guest Mode")
                st.caption("Limited features")
                if st.button("🔐 Login / Register", width="stretch"):
                    st.session_state.user = None
                    st.rerun()
            else:
                st.markdown(f"**{user['email'][:20]}...**" if len(user.get('email', '')) > 20 else f"**{user.get('email', '')}**")
                
                # Tier badge
                tier_colors = {
                    'free': '🆓', 'basic': '🥉', 'pro': '🥈', 'unlimited': '🥇'
                }
                st.markdown(f"{tier_colors.get(tier, '🆓')} **{tier_info['name']}** Plan")
                
                # Usage
                can_predict, msg, remaining = self.auth.can_make_prediction(user['id'])
                if remaining == -1:
                    st.caption("♾️ Unlimited predictions")
                else:
                    st.caption(f"📊 {remaining} predictions left today")
                    st.progress(remaining / tier_info['predictions_per_day'])
                
                # Upgrade button (if not unlimited)
                if tier != 'unlimited':
                    if st.button("💎 Upgrade Plan", width="stretch"):
                        st.session_state.show_pricing = True
                
                st.divider()
                
                if st.button("🚪 Logout", width="stretch"):
                    if st.session_state.session_token:
                        self.auth.logout(st.session_state.session_token)
                    st.session_state.session_token = None
                    st.session_state.user = None
                    st.rerun()
    
    def check_prediction_limit(self) -> Tuple[bool, str]:
        """Check if user can make a prediction"""
        user = st.session_state.user
        
        if not user:
            return False, "Please login to make predictions"
        
        if user.get('is_guest'):
            # Track guest predictions in session
            guest_predictions = st.session_state.get('guest_predictions', 0)
            if guest_predictions >= 3:
                return False, "Guest limit reached (3/day). Please register for more!"
            return True, f"{3 - guest_predictions} predictions remaining"
        
        # Logged in user
        can_predict, msg, remaining = self.auth.can_make_prediction(user['id'])
        return can_predict, msg
    
    def record_prediction_usage(self):
        """Record that a prediction was made"""
        user = st.session_state.user
        
        if not user:
            return
        
        if user.get('is_guest'):
            st.session_state.guest_predictions = st.session_state.get('guest_predictions', 0) + 1
        else:
            self.auth.record_prediction(user['id'])
    
    def check_league_access(self, league: str) -> bool:
        """Check if user can access a league"""
        user = st.session_state.user
        
        if not user:
            return league == 'EPL'  # Guests only get EPL
        
        if user.get('is_guest'):
            return league == 'EPL'
        
        return self.auth.can_access_league(user['id'], league)
    
    def run(self):
        """Run the application"""
        # Check for valid session
        if st.session_state.session_token:
            user = self.auth.validate_session(st.session_state.session_token)
            if user:
                st.session_state.user = user
            else:
                # Session expired
                st.session_state.session_token = None
                st.session_state.user = None
        
        # Show auth page if not logged in
        if not st.session_state.user:
            self.render_auth_page()
            return
        
        # Normal app flow
        league = self.render_sidebar()
        self.render_main_content(league)


def main():
    app = FootballPredictorApp()
    app.run()


if __name__ == "__main__":
    main()
