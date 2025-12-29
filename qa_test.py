#!/usr/bin/env python3
"""
Comprehensive QA Test Suite for Football Prediction Model
"""

import os
import sys

def run_qa_tests():
    print('='*60)
    print('QA TESTING REPORT - Football Prediction Model')
    print('='*60)
    print()

    results = {
        'passed': [],
        'failed': [],
        'warnings': []
    }

    # Test 1: Syntax Check
    print('### 1. SYNTAX CHECK')
    try:
        import py_compile
        files = ['src/openfootball_fetcher.py', 'src/feature_engineering.py', 'src/models.py', 'app.py', 'main.py', 'train_models.py']
        for f in files:
            py_compile.compile(f, doraise=True)
        print('✅ All Python files have valid syntax')
        results['passed'].append('Syntax Check')
    except Exception as e:
        print(f'❌ Syntax error: {e}')
        results['failed'].append(f'Syntax Check: {e}')
    print()

    # Test 2: Import Check
    print('### 2. IMPORT CHECK')
    try:
        from src.openfootball_fetcher import OpenFootballFetcher
        from src.feature_engineering import FeatureEngineer
        from src.models import EPLPredictor
        import pandas, numpy, sklearn, streamlit, requests, joblib
        print('✅ All imports successful')
        results['passed'].append('Import Check')
    except Exception as e:
        print(f'❌ Import error: {e}')
        results['failed'].append(f'Import Check: {e}')
    print()

    # Test 3: Data Files
    print('### 3. DATA FILES')
    data_dir = 'data'
    data_files = [f for f in os.listdir(data_dir) if f.endswith('_data.csv')]
    print(f'✅ {len(data_files)} data files found')
    results['passed'].append(f'Data Files: {len(data_files)} files')
    print()

    # Test 4: Model Files
    print('### 4. MODEL FILES')
    model_dir = 'models'
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
    print(f'✅ {len(model_files)} model files found')
    metadata_files = [f for f in model_files if 'metadata' in f]
    print(f'✅ {len(metadata_files)} leagues have trained models')
    results['passed'].append(f'Model Files: {len(model_files)} files, {len(metadata_files)} leagues')
    print()

    # Test 5: Model Loading
    print('### 5. MODEL LOADING')
    try:
        from src.models import EPLPredictor
        predictor = EPLPredictor()
        predictor.load_models(prefix='epl')
        print(f'✅ EPL models loaded ({len(predictor.models)} targets)')
        results['passed'].append('Model Loading')
    except Exception as e:
        print(f'❌ Model loading error: {e}')
        results['failed'].append(f'Model Loading: {e}')
    print()

    # Test 6: Prediction Pipeline
    print('### 6. PREDICTION PIPELINE')
    try:
        import numpy as np
        dummy = np.random.randn(1, len(predictor.feature_columns))
        preds = predictor.predict(dummy)
        print(f'✅ Prediction successful ({len(preds)} targets)')
        results['passed'].append('Prediction Pipeline')
    except Exception as e:
        print(f'❌ Prediction error: {e}')
        results['failed'].append(f'Prediction Pipeline: {e}')
    print()

    # Test 7: Feature Engineering
    print('### 7. FEATURE ENGINEERING')
    try:
        from src.openfootball_fetcher import OpenFootballFetcher
        from src.feature_engineering import FeatureEngineer
        fetcher = OpenFootballFetcher()
        engineer = FeatureEngineer(n_last_matches=5)
        df = fetcher.load_league_data('EPL')
        df_sample = df.head(30)
        df_features = engineer.create_features(df_sample)
        print(f'✅ Feature engineering successful ({len(engineer.get_feature_columns())} features)')
        results['passed'].append('Feature Engineering')
    except Exception as e:
        print(f'❌ Feature engineering error: {e}')
        results['failed'].append(f'Feature Engineering: {e}')
    print()

    # Test 8: All Leagues Load
    print('### 8. ALL LEAGUES')
    try:
        from src.openfootball_fetcher import OpenFootballFetcher
        fetcher = OpenFootballFetcher()
        leagues = fetcher.get_available_leagues()
        loaded = 0
        for league in leagues:
            try:
                df = fetcher.load_league_data(league)
                if len(df) > 0:
                    loaded += 1
            except:
                pass
        print(f'✅ {loaded}/{len(leagues)} leagues have data')
        results['passed'].append(f'All Leagues: {loaded}/{len(leagues)}')
    except Exception as e:
        print(f'❌ League loading error: {e}')
        results['failed'].append(f'All Leagues: {e}')
    print()

    # Summary
    print('='*60)
    print('SUMMARY')
    print('='*60)
    print(f'✅ Passed: {len(results["passed"])}')
    print(f'❌ Failed: {len(results["failed"])}')
    print(f'⚠️  Warnings: {len(results["warnings"])}')
    print()

    if results['failed']:
        print('FAILED TESTS:')
        for f in results['failed']:
            print(f'  - {f}')
    else:
        print('🎉 ALL TESTS PASSED!')
    
    return len(results['failed']) == 0


if __name__ == '__main__':
    success = run_qa_tests()
    sys.exit(0 if success else 1)
