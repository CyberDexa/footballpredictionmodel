"""
Football Match Prediction App
A beautiful Streamlit UI for multi-league football predictions
Uses real data from OpenFootball (openfootball.github.io)
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.openfootball_fetcher import OpenFootballFetcher
from src.feature_engineering import FeatureEngineer
from src.models import EPLPredictor

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
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
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
        
        # Initialize fetcher (no API key needed for OpenFootball!)
        self.fetcher = OpenFootballFetcher(data_dir="data")
        self.engineer = FeatureEngineer(n_last_matches=5)
        self.predictor = EPLPredictor(models_dir="models")
    
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
                if st.button("🔄 Refresh", use_container_width=True):
                    with st.spinner("Fetching latest data..."):
                        try:
                            df = self.fetcher.get_or_fetch_league_data(selected_league, force_refresh=True)
                            st.success(f"✅ {len(df)} matches!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
            
            with col2:
                if st.button("🔄 All Leagues", use_container_width=True):
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
            
            if st.button("🔄 Retrain Model", use_container_width=True):
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
            - **Match Result** (Home/Draw/Away)
            - **Over/Under 1.5 Goals**
            - **Over/Under 2.5 Goals**
            
            **Data Source:** OpenFootball
            - Free, open public domain data
            - No API key required!
            - Updated regularly with current season
            """)
            
            return selected_league
    
    def render_main_content(self, league: str):
        """Render the main content area"""
        # Header
        st.markdown('<h1 class="main-header">⚽ Football Match Predictor</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">AI-powered predictions using real football data</p>', unsafe_allow_html=True)
        
        # Auto-refresh check (weekly)
        self.check_auto_refresh(league, max_age_days=7)
        
        # Check for data and show info
        try:
            df = self.fetcher.load_league_data(league)
            df['Date'] = pd.to_datetime(df['Date'])
            latest_date = df['Date'].max()
            
            # Show data info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total Matches", len(df))
            with col2:
                st.metric("📅 Latest Data", latest_date.strftime("%Y-%m-%d"))
            with col3:
                teams_count = len(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))
                st.metric("👥 Teams", teams_count)
            
            # Warning if data is old
            from datetime import datetime
            days_old = (datetime.now() - latest_date.to_pydatetime()).days
            if days_old > 60:
                st.warning(f"⚠️ Data is {days_old} days old. Click 'Refresh Data' in the sidebar for the latest matches.")
        except:
            st.info("📥 Click 'Refresh Data' in the sidebar to fetch match data.")
        
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
                use_container_width=True,
                type="primary"
            )
        
        if predict_clicked:
            self.render_prediction(league, home_team, away_team)
    
    def render_prediction(self, league: str, home_team: str, away_team: str):
        """Render the prediction results"""
        with st.spinner("🔮 Analyzing match data..."):
            predictions = self.get_prediction(league, home_team, away_team)
        
        if predictions is None:
            st.error("Could not generate prediction. Please try different teams.")
            return
        
        st.markdown("---")
        st.markdown(f"## 📊 Prediction: {home_team} vs {away_team}")
        
        # Match Result
        st.markdown("### 🏆 Match Result Prediction")
        
        match_pred = predictions['match_result']
        probs = match_pred['probabilities'][0]
        home_prob, draw_prob, away_prob = probs[0], probs[1], probs[2]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if home_prob == max(probs):
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
            if draw_prob == max(probs):
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
            if away_prob == max(probs):
                st.markdown(f"""
                <div class="win-away">
                    <h3 style="margin:0; color: white;">✈️ {away_team}</h3>
                    <h2 style="margin:0; color: white;">{away_prob*100:.1f}%</h2>
                    <p style="margin:0; color: rgba(255,255,255,0.8);">AWAY WIN</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.metric(f"✈️ {away_team}", f"{away_prob*100:.1f}%", "Away Win")
        
        # Probability bars
        st.markdown("#### Win Probability Distribution")
        
        prob_data = pd.DataFrame({
            'Outcome': [f'🏠 {home_team}', '🤝 Draw', f'✈️ {away_team}'],
            'Probability': [home_prob * 100, draw_prob * 100, away_prob * 100]
        })
        
        st.bar_chart(prob_data.set_index('Outcome'), horizontal=True, height=200)
        
        # Goals Predictions
        st.markdown("### ⚽ Goals Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            over15 = predictions['over_1.5']
            over15_prob = over15['probabilities'][0][1] * 100
            under15_prob = over15['probabilities'][0][0] * 100
            
            st.markdown("#### Over/Under 1.5 Goals")
            
            goal_col1, goal_col2 = st.columns(2)
            with goal_col1:
                st.metric("📈 Over 1.5", f"{over15_prob:.1f}%")
            with goal_col2:
                st.metric("📉 Under 1.5", f"{under15_prob:.1f}%")
            
            # Progress bar for over 1.5
            st.progress(over15_prob / 100)
        
        with col2:
            over25 = predictions['over_2.5']
            over25_prob = over25['probabilities'][0][1] * 100
            under25_prob = over25['probabilities'][0][0] * 100
            
            st.markdown("#### Over/Under 2.5 Goals")
            
            goal_col1, goal_col2 = st.columns(2)
            with goal_col1:
                st.metric("📈 Over 2.5", f"{over25_prob:.1f}%")
            with goal_col2:
                st.metric("📉 Under 2.5", f"{under25_prob:.1f}%")
            
            # Progress bar for over 2.5
            st.progress(over25_prob / 100)
        
        # Recommendation Box
        st.markdown("### 💡 Prediction Summary")
        
        # Determine best prediction
        result_pred = match_pred['prediction'][0]
        confidence = match_pred['confidence'][0] * 100
        
        goals_pred = "Over 2.5 Goals" if over25_prob > 50 else "Under 2.5 Goals"
        goals_conf = max(over25_prob, under25_prob)
        
        st.info(f"""
        **Match Outcome:** {result_pred} (Confidence: {confidence:.1f}%)
        
        **Goals:** {goals_pred} ({goals_conf:.1f}% probability)
        
        **Analysis:**
        - The model predicts **{result_pred}** as the most likely outcome
        - There's a **{over25_prob:.1f}%** chance of seeing more than 2.5 goals
        - {home_team} has a **{home_prob*100:.1f}%** chance of winning at home
        """)
        
        # Disclaimer
        st.caption("⚠️ These predictions are for informational purposes only. Past performance does not guarantee future results.")
    
    def run(self):
        """Run the application"""
        league = self.render_sidebar()
        self.render_main_content(league)


def main():
    app = FootballPredictorApp()
    app.run()


if __name__ == "__main__":
    main()
