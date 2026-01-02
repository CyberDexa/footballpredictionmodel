"""
Train all league models with enhanced features
Run this script before deploying to ensure all models are ready

Enhanced model features:
- 74 base features (form, streaks, momentum, goal timing, etc.)
- 17 prediction targets (results, goals, BTTS, half-time, etc.)
- Stacking ensemble for improved accuracy
- Probability calibration for reliable predictions
"""

from src.openfootball_fetcher import OpenFootballFetcher
from src.feature_engineering import FeatureEngineer
from src.models import EPLPredictor
import os
import sys
from datetime import datetime

# All 17 prediction targets
ENHANCED_TARGETS = [
    'match_result', 'home_win', 'away_win', 'draw',
    'over_1.5', 'over_2.5', 'over_3.5',
    'btts', 'goals_0_1', 'goals_2_3', 'goals_4_plus',
    'home_over_0.5', 'home_over_1.5', 
    'away_over_0.5', 'away_over_1.5',
    'ht_over_0.5', 'ht_over_1.5'
]


def check_model_status(league: str, models_dir: str = "models") -> dict:
    """Check which models exist for a league"""
    status = {'complete': True, 'missing': [], 'existing': []}
    
    for target in ENHANCED_TARGETS:
        model_path = f"{models_dir}/{league.lower()}_{target}_model.joblib"
        if os.path.exists(model_path):
            status['existing'].append(target)
        else:
            status['missing'].append(target)
            status['complete'] = False
    
    return status


def train_league(league: str, fetcher, engineer, predictor, force: bool = False):
    """Train all models for a single league"""
    print(f"\n{'='*60}")
    print(f"🏟️  Training {league}")
    print(f"{'='*60}")
    
    # Check if already trained
    status = check_model_status(league)
    if status['complete'] and not force:
        print(f"  ✅ Already has all {len(ENHANCED_TARGETS)} models, skipping")
        print(f"  (Use --force to retrain)")
        return True
    
    if status['existing'] and not force:
        print(f"  ⚠️  Has {len(status['existing'])}/{len(ENHANCED_TARGETS)} models")
        print(f"  Missing: {', '.join(status['missing'][:5])}...")
    
    try:
        # Fetch data
        print(f"  📥 Fetching historical data...")
        df = fetcher.get_or_fetch_league_data(league)
        
        if df is None or len(df) < 100:
            print(f"  ⚠️ Not enough data ({len(df) if df is not None else 0} matches), need 100+")
            return False
        
        print(f"  📊 Loaded {len(df)} matches")
        
        # Create enhanced features (74 features)
        print(f"  🔧 Engineering {len(engineer.get_feature_columns())} features...")
        df_features = engineer.create_features(df)
        
        # Get feature columns
        feature_cols = engineer.get_feature_columns()
        
        # Prepare data
        print(f"  🧹 Preparing data...")
        df_clean = predictor.prepare_data(df_features, feature_cols)
        
        if len(df_clean) < 50:
            print(f"  ⚠️ Not enough clean data ({len(df_clean)} rows after processing)")
            return False
        
        print(f"  📈 Training on {len(df_clean)} samples with {len(feature_cols)} features")
        
        # Train all 17 targets with stacking ensemble
        print(f"  🤖 Training {len(predictor.targets)} models (stacking ensemble + calibration)...")
        predictor.train(df_clean, feature_cols)
        
        # Save models
        print(f"  💾 Saving models...")
        predictor.save_models(prefix=league.lower())
        
        print(f"  ✅ {league} trained successfully with all {len(ENHANCED_TARGETS)} targets!")
        return True
        
    except Exception as e:
        print(f"  ❌ Error training {league}: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_all_models(leagues_to_train=None, force=False):
    """Train models for all leagues (or specified ones)"""
    
    print("=" * 70)
    print("🚀 ENHANCED MODEL TRAINING")
    print(f"   Features: 74 | Targets: 17 | Stacking: ✅ | Calibration: ✅")
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Initialize components
    fetcher = OpenFootballFetcher()
    engineer = FeatureEngineer(n_last_matches=5)
    predictor = EPLPredictor(models_dir="models")
    
    # Get leagues
    all_leagues = list(fetcher.get_available_leagues().keys())
    
    if leagues_to_train:
        # Validate specified leagues
        invalid = [l for l in leagues_to_train if l not in all_leagues]
        if invalid:
            print(f"⚠️  Invalid leagues: {', '.join(invalid)}")
            print(f"   Available: {', '.join(all_leagues)}")
        leagues = [l for l in leagues_to_train if l in all_leagues]
    else:
        leagues = all_leagues
    
    print(f"\n📋 Leagues to train: {len(leagues)}")
    for league in leagues:
        info = fetcher.get_available_leagues().get(league, {})
        flag = info.get('flag', '🏳️')
        name = info.get('name', league)
        print(f"   {flag} {name}")
    
    # Check current status
    print(f"\n📊 Current Model Status:")
    complete_count = 0
    partial_count = 0
    missing_count = 0
    
    for league in leagues:
        status = check_model_status(league)
        if status['complete']:
            complete_count += 1
            symbol = "✅"
        elif status['existing']:
            partial_count += 1
            symbol = "⚠️"
        else:
            missing_count += 1
            symbol = "❌"
        print(f"   {symbol} {league}: {len(status['existing'])}/{len(ENHANCED_TARGETS)} models")
    
    print(f"\n   Summary: ✅ Complete: {complete_count} | ⚠️ Partial: {partial_count} | ❌ Missing: {missing_count}")
    
    if force:
        print(f"\n⚠️  Force mode: Will retrain ALL leagues")
    
    # Train each league
    results = {'success': [], 'failed': [], 'skipped': []}
    
    for i, league in enumerate(leagues, 1):
        print(f"\n[{i}/{len(leagues)}] Processing {league}...")
        
        if train_league(league, fetcher, engineer, predictor, force=force):
            if check_model_status(league)['complete']:
                results['success'].append(league)
            else:
                results['skipped'].append(league)
        else:
            results['failed'].append(league)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TRAINING SUMMARY")
    print("=" * 70)
    print(f"✅ Successfully trained: {len(results['success'])}")
    for league in results['success']:
        print(f"   • {league}")
    
    if results['skipped']:
        print(f"⏭️  Skipped (already complete): {len(results['skipped'])}")
        for league in results['skipped']:
            print(f"   • {league}")
    
    if results['failed']:
        print(f"❌ Failed: {len(results['failed'])}")
        for league in results['failed']:
            print(f"   • {league}")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train football prediction models')
    parser.add_argument('leagues', nargs='*', help='Specific leagues to train (optional)')
    parser.add_argument('--force', '-f', action='store_true', help='Force retrain even if models exist')
    parser.add_argument('--list', '-l', action='store_true', help='List available leagues')
    
    args = parser.parse_args()
    
    if args.list:
        fetcher = OpenFootballFetcher()
        print("\n📋 Available Leagues:")
        for code, info in fetcher.get_available_leagues().items():
            print(f"   {info.get('flag', '🏳️')} {code}: {info.get('name', code)}")
        sys.exit(0)
    
    train_all_models(
        leagues_to_train=args.leagues if args.leagues else None,
        force=args.force
    )
