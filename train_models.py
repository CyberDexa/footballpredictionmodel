"""
Train all league models
Run this script before deploying to ensure all models are ready
"""

from src.openfootball_fetcher import OpenFootballFetcher
from src.feature_engineering import FeatureEngineer
from src.models import EPLPredictor
import os

def train_all_models():
    fetcher = OpenFootballFetcher()
    engineer = FeatureEngineer(n_last_matches=5)
    predictor = EPLPredictor(models_dir="models")
    
    leagues = list(fetcher.get_available_leagues().keys())
    
    # Check status
    missing = []
    trained = []
    
    for league in leagues:
        model_path = f"models/{league.lower()}_match_result_model.joblib"
        if os.path.exists(model_path):
            trained.append(league)
        else:
            missing.append(league)
    
    print("=" * 60)
    print("MODEL TRAINING STATUS")
    print("=" * 60)
    print(f"Already trained ({len(trained)}): {', '.join(trained)}")
    print(f"Need training ({len(missing)}): {', '.join(missing)}")
    print("=" * 60)
    
    if not missing:
        print("\n✅ All models are already trained!")
        return
    
    # Train missing ones
    for league in missing:
        print(f"\n🔄 Training {league}...")
        try:
            # Fetch data
            df = fetcher.get_or_fetch_league_data(league)
            
            if len(df) < 50:
                print(f"  ⚠️ Not enough data ({len(df)} matches), skipping")
                continue
            
            # Create features
            df_features = engineer.create_features(df)
            
            # Train
            feature_cols = engineer.get_feature_columns()
            df_clean = predictor.prepare_data(df_features, feature_cols)
            predictor.train(df_clean, feature_cols)
            predictor.save_models(prefix=league.lower())
            
            print(f"  ✅ {league} trained successfully!")
            
        except Exception as e:
            print(f"  ❌ Error training {league}: {e}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    train_all_models()
