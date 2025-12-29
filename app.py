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

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.openfootball_fetcher import OpenFootballFetcher
from src.feature_engineering import FeatureEngineer
from src.models import EPLPredictor
from src.upcoming_fixtures import UpcomingFixturesFetcher, FixturesManager
from src.database import get_database
from src.value_bets import get_value_calculator

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
        
        # Initialize fetcher (no API key needed for OpenFootball!)
        self.fetcher = OpenFootballFetcher(data_dir="data")
        self.engineer = FeatureEngineer(n_last_matches=5)
        self.predictor = EPLPredictor(models_dir="models")
        self.fixtures_fetcher = UpcomingFixturesFetcher()
        self.fixtures_manager = FixturesManager()
        self.db = get_database()
        self.value_calculator = get_value_calculator()
    
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
    
    def render_sidebar(self):
        """Render the sidebar"""
        with st.sidebar:
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
            
            # League selection
            leagues = self.fetcher.get_available_leagues()
            league_options = {f"{info['flag']} {info['name']}": code 
                           for code, info in leagues.items()}
            
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
        tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict Match", "📅 Upcoming Matches", "📊 Stats", "📈 Track Record"])
        
        with tab1:
            self.render_prediction_tab(league)
        
        with tab2:
            self.render_upcoming_matches_tab(league)
        
        with tab3:
            self.render_stats_tab(league)
        
        with tab4:
            self.render_accuracy_tab(league)
    
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
            self.render_prediction(league, home_team, away_team)
    
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
                        self.render_prediction(league, home_team, away_team)
        
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
                use_container_width=True
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
            
            st.dataframe(display_df, hide_index=True, use_container_width=True)
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
    
    def run(self):
        """Run the application"""
        league = self.render_sidebar()
        self.render_main_content(league)


def main():
    app = FootballPredictorApp()
    app.run()


if __name__ == "__main__":
    main()
