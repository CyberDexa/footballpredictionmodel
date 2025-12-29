"""
ML Models Module
Contains models for EPL match prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Try to import optional boosting libraries
HAS_XGBOOST = False
HAS_LIGHTGBM = False

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except (ImportError, Exception):
    pass  # XGBoost not available or can't load (missing libomp on macOS)

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except (ImportError, Exception):
    pass  # LightGBM not available or can't load


class EPLPredictor:
    """
    Multi-model predictor for EPL matches
    Predicts: Match Result, Over/Under 1.5 goals, Over/Under 2.5 goals
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_columns = []
        
        # Target configurations
        self.targets = {
            'match_result': {
                'column': 'MatchResult',
                'type': 'multiclass',
                'labels': {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}
            },
            'over_1.5': {
                'column': 'Over1.5',
                'type': 'binary',
                'labels': {0: 'Under 1.5', 1: 'Over 1.5'}
            },
            'over_2.5': {
                'column': 'Over2.5',
                'type': 'binary',
                'labels': {0: 'Under 2.5', 1: 'Over 2.5'}
            },
            'home_win': {
                'column': 'HomeWin',
                'type': 'binary',
                'labels': {0: 'Not Home Win', 1: 'Home Win'}
            },
            'away_win': {
                'column': 'AwayWin',
                'type': 'binary',
                'labels': {0: 'Not Away Win', 1: 'Away Win'}
            }
        }
    
    def _get_base_models(self) -> Dict[str, Any]:
        """Get dictionary of base models to ensemble"""
        models = {
            'rf': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            'lr': LogisticRegression(
                max_iter=1000,
                random_state=42,
                solver='lbfgs'
            )
        }
        
        if HAS_XGBOOST:
            models['xgb'] = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbosity=0,
                use_label_encoder=False
            )
        
        if HAS_LIGHTGBM:
            models['lgbm'] = LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbosity=-1
            )
        
        return models
    
    def prepare_data(self, df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
        """Prepare data for training by removing rows with insufficient history"""
        self.feature_columns = feature_columns
        
        # Remove rows where teams haven't played enough matches
        df_clean = df[df['home_matches_played'] >= 3].copy()
        df_clean = df_clean[df_clean['away_matches_played'] >= 3]
        
        # Drop any rows with NaN in features or targets
        all_cols = feature_columns + [t['column'] for t in self.targets.values()]
        df_clean = df_clean.dropna(subset=[c for c in all_cols if c in df_clean.columns])
        
        return df_clean
    
    def train(self, df: pd.DataFrame, feature_columns: list) -> Dict[str, Dict]:
        """Train models for all targets"""
        self.feature_columns = feature_columns
        
        # Prepare features
        X = df[feature_columns].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        results = {}
        
        for target_name, target_config in self.targets.items():
            print(f"\n{'='*50}")
            print(f"Training models for: {target_name}")
            print(f"{'='*50}")
            
            y = df[target_config['column']].values
            
            # Time-based split (use last 20% as test)
            split_idx = int(len(X_scaled) * 0.8)
            X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            best_model = None
            best_score = 0
            model_scores = {}
            
            base_models = self._get_base_models()
            
            for model_name, model in base_models.items():
                print(f"\n  Training {model_name}...")
                
                try:
                    model.fit(X_train, y_train)
                    
                    # Evaluate
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # Cross-validation with time series split
                    tscv = TimeSeriesSplit(n_splits=5)
                    cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='accuracy')
                    
                    model_scores[model_name] = {
                        'test_accuracy': accuracy,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std()
                    }
                    
                    print(f"    Test Accuracy: {accuracy:.4f}")
                    print(f"    CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
                    
                    if accuracy > best_score:
                        best_score = accuracy
                        best_model = model
                        
                except Exception as e:
                    print(f"    Error: {e}")
            
            # Store best model
            self.models[target_name] = best_model
            
            # Final evaluation on best model
            y_pred = best_model.predict(X_test)
            
            results[target_name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'model_scores': model_scores,
                'classification_report': classification_report(
                    y_test, y_pred, 
                    target_names=list(target_config['labels'].values()),
                    output_dict=True
                )
            }
            
            print(f"\n  Best model accuracy: {results[target_name]['accuracy']:.4f}")
            print(f"\n  Classification Report:")
            print(classification_report(
                y_test, y_pred,
                target_names=list(target_config['labels'].values())
            ))
        
        return results
    
    def predict(self, features: np.ndarray) -> Dict[str, Dict]:
        """Make predictions for a single match or batch of matches"""
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        predictions = {}
        
        for target_name, model in self.models.items():
            target_config = self.targets[target_name]
            
            # Get prediction and probabilities
            pred = model.predict(features_scaled)
            proba = model.predict_proba(features_scaled)
            
            predictions[target_name] = {
                'prediction': [target_config['labels'][p] for p in pred],
                'probabilities': proba.tolist(),
                'confidence': np.max(proba, axis=1).tolist()
            }
        
        return predictions
    
    def predict_match(self, home_features: dict, away_features: dict, 
                      derived_features: dict) -> Dict:
        """Predict a single match given feature dictionaries"""
        # Combine features in correct order
        features = []
        for col in self.feature_columns:
            if col in home_features:
                features.append(home_features[col])
            elif col in away_features:
                features.append(away_features[col])
            elif col in derived_features:
                features.append(derived_features[col])
            else:
                features.append(0)  # Default
        
        features = np.array(features)
        return self.predict(features)
    
    def save_models(self, prefix: str = "epl"):
        """Save all trained models"""
        # Save scaler
        scaler_path = os.path.join(self.models_dir, f"{prefix}_scaler.joblib")
        joblib.dump(self.scaler, scaler_path)
        
        # Save models
        for target_name, model in self.models.items():
            model_path = os.path.join(self.models_dir, f"{prefix}_{target_name}_model.joblib")
            joblib.dump(model, model_path)
        
        # Save feature columns
        meta_path = os.path.join(self.models_dir, f"{prefix}_metadata.joblib")
        joblib.dump({
            'feature_columns': self.feature_columns,
            'targets': self.targets
        }, meta_path)
        
        print(f"✓ Models saved to {self.models_dir}/")
    
    def load_models(self, prefix: str = "epl"):
        """Load trained models"""
        # Load scaler
        scaler_path = os.path.join(self.models_dir, f"{prefix}_scaler.joblib")
        self.scaler = joblib.load(scaler_path)
        
        # Load metadata
        meta_path = os.path.join(self.models_dir, f"{prefix}_metadata.joblib")
        metadata = joblib.load(meta_path)
        self.feature_columns = metadata['feature_columns']
        
        # Load models
        for target_name in self.targets.keys():
            model_path = os.path.join(self.models_dir, f"{prefix}_{target_name}_model.joblib")
            if os.path.exists(model_path):
                self.models[target_name] = joblib.load(model_path)
        
        print(f"✓ Models loaded from {self.models_dir}/")
    
    def get_feature_importance(self, target_name: str = 'match_result') -> pd.DataFrame:
        """Get feature importance for a specific model"""
        model = self.models.get(target_name)
        if model is None:
            return pd.DataFrame()
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).mean(axis=0)
        else:
            return pd.DataFrame()
        
        return pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)


if __name__ == "__main__":
    # Test the models
    from data_fetcher import EPLDataFetcher
    from feature_engineering import FeatureEngineer
    
    fetcher = EPLDataFetcher(data_dir="../data")
    df = fetcher.get_or_fetch_data()
    
    engineer = FeatureEngineer(n_last_matches=5)
    df_with_features = engineer.create_features(df)
    
    predictor = EPLPredictor(models_dir="../models")
    feature_cols = engineer.get_feature_columns()
    
    df_clean = predictor.prepare_data(df_with_features, feature_cols)
    results = predictor.train(df_clean, feature_cols)
    
    predictor.save_models()
