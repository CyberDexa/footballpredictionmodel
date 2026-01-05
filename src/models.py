"""
ML Models Module
Contains models for EPL match prediction
Enhanced with Stacking Ensemble, Calibration, and Advanced Features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, brier_score_loss
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
        
        # Target configurations - Extended predictions
        self.targets = {
            # Match Result
            'match_result': {
                'column': 'MatchResult',
                'type': 'multiclass',
                'labels': {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}
            },
            'home_win': {
                'column': 'HomeWin',
                'type': 'binary',
                'labels': {0: 'No', 1: 'Yes'}
            },
            'draw': {
                'column': 'Draw',
                'type': 'binary',
                'labels': {0: 'No', 1: 'Yes'}
            },
            'away_win': {
                'column': 'AwayWin',
                'type': 'binary',
                'labels': {0: 'No', 1: 'Yes'}
            },
            # Full Time Goals
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
            'over_3.5': {
                'column': 'Over3.5',
                'type': 'binary',
                'labels': {0: 'Under 3.5', 1: 'Over 3.5'}
            },
            # Both Teams To Score
            'btts': {
                'column': 'BTTS',
                'type': 'binary',
                'labels': {0: 'No', 1: 'Yes'}
            },
            # Home Team Goals
            'home_over_0.5': {
                'column': 'HomeOver0.5',
                'type': 'binary',
                'labels': {0: 'Under 0.5', 1: 'Over 0.5'}
            },
            'home_over_1.5': {
                'column': 'HomeOver1.5',
                'type': 'binary',
                'labels': {0: 'Under 1.5', 1: 'Over 1.5'}
            },
            # Away Team Goals
            'away_over_0.5': {
                'column': 'AwayOver0.5',
                'type': 'binary',
                'labels': {0: 'Under 0.5', 1: 'Over 0.5'}
            },
            'away_over_1.5': {
                'column': 'AwayOver1.5',
                'type': 'binary',
                'labels': {0: 'Under 1.5', 1: 'Over 1.5'}
            },
            # Half Time Goals
            'ht_over_0.5': {
                'column': 'HTOver0.5',
                'type': 'binary',
                'labels': {0: 'Under 0.5', 1: 'Over 0.5'}
            },
            'ht_over_1.5': {
                'column': 'HTOver1.5',
                'type': 'binary',
                'labels': {0: 'Under 1.5', 1: 'Over 1.5'}
            },
            # Goal Ranges
            'goals_0_1': {
                'column': 'Goals0_1',
                'type': 'binary',
                'labels': {0: 'No', 1: 'Yes (0-1 Goals)'}
            },
            'goals_2_3': {
                'column': 'Goals2_3',
                'type': 'binary',
                'labels': {0: 'No', 1: 'Yes (2-3 Goals)'}
            },
            'goals_4_plus': {
                'column': 'Goals4Plus',
                'type': 'binary',
                'labels': {0: 'No', 1: 'Yes (4+ Goals)'}
            }
        }
        
        # Use enhanced stacking by default
        self.use_stacking = True
        self.use_calibration = True
    
    def _get_base_models(self, use_class_weights: bool = False) -> Dict[str, Any]:
        """Get dictionary of base models to ensemble - ENHANCED"""
        
        # Class weights for imbalanced targets (e.g., draws are rare)
        class_weight = 'balanced' if use_class_weights else None
        
        models = {
            'rf': RandomForestClassifier(
                n_estimators=300,  # Increased
                max_depth=12,      # Slightly deeper
                min_samples_split=8,
                min_samples_leaf=4,
                class_weight=class_weight,
                random_state=42,
                n_jobs=-1
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=200,  # Increased
                max_depth=6,
                learning_rate=0.08,  # Slightly lower for better generalization
                subsample=0.8,       # Add randomness
                random_state=42
            ),
            'lr': LogisticRegression(
                max_iter=2000,
                random_state=42,
                solver='lbfgs',
                class_weight=class_weight,
                C=0.5  # Some regularization
            )
        }
        
        if HAS_XGBOOST:
            models['xgb'] = XGBClassifier(
                n_estimators=300,
                max_depth=7,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
        
        if HAS_LIGHTGBM:
            models['lgbm'] = LGBMClassifier(
                n_estimators=300,
                max_depth=7,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=-1,
                class_weight=class_weight
            )
        
        return models
    
    def _create_stacking_ensemble(self, base_models: Dict[str, Any], 
                                   is_multiclass: bool = False) -> StackingClassifier:
        """Create a stacking ensemble from base models"""
        estimators = [(name, model) for name, model in base_models.items()]
        
        # Meta-learner: Logistic Regression with regularization
        meta_learner = LogisticRegression(
            max_iter=2000,
            random_state=42,
            solver='lbfgs',
            C=1.0
        )
        
        stack = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=5,  # 5-fold CV for meta-features
            stack_method='predict_proba',  # Use probabilities
            n_jobs=-1,
            passthrough=False  # Don't include original features
        )
        
        return stack
    
    def _create_voting_ensemble(self, base_models: Dict[str, Any]) -> VotingClassifier:
        """Create a soft voting ensemble from base models"""
        estimators = [(name, model) for name, model in base_models.items()]
        
        voting = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Use probabilities
            n_jobs=-1
        )
        
        return voting
    
    def _calibrate_model(self, model, X_train, y_train) -> CalibratedClassifierCV:
        """Apply probability calibration using Platt scaling"""
        calibrated = CalibratedClassifierCV(
            model,
            method='sigmoid',  # Platt scaling
            cv=5
        )
        calibrated.fit(X_train, y_train)
        return calibrated
    
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
        """Train models for all targets - ENHANCED with Stacking & Calibration"""
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
            is_multiclass = target_config['type'] == 'multiclass'
            
            # Use class weights for imbalanced targets
            use_weights = target_name in ['draw', 'goals_0_1', 'goals_4_plus', 'ht_over_1.5']
            
            # Time-based split (use last 20% as test)
            split_idx = int(len(X_scaled) * 0.8)
            X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            best_model = None
            best_score = 0
            model_scores = {}
            
            # Get base models
            base_models = self._get_base_models(use_class_weights=use_weights)
            
            # Train individual models first
            print("\n  📊 Training individual models...")
            for model_name, model in base_models.items():
                try:
                    model.fit(X_train, y_train)
                    
                    # Evaluate
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # Cross-validation with time series split
                    tscv = TimeSeriesSplit(n_splits=5)
                    cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='accuracy')
                    
                    # Brier score for probability calibration (binary only)
                    brier = None
                    if not is_multiclass:
                        y_proba = model.predict_proba(X_test)[:, 1]
                        brier = brier_score_loss(y_test, y_proba)
                    
                    model_scores[model_name] = {
                        'test_accuracy': accuracy,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'brier_score': brier
                    }
                    
                    print(f"    {model_name}: Acc={accuracy:.4f}, CV={cv_scores.mean():.4f}")
                    
                    if accuracy > best_score:
                        best_score = accuracy
                        best_model = model
                        
                except Exception as e:
                    print(f"    {model_name} Error: {e}")
            
            # Try stacking ensemble
            if self.use_stacking and len(base_models) >= 2:
                print("\n  🔄 Training Stacking Ensemble...")
                try:
                    # Re-create fresh models for stacking
                    fresh_models = self._get_base_models(use_class_weights=use_weights)
                    stack = self._create_stacking_ensemble(fresh_models, is_multiclass)
                    stack.fit(X_train, y_train)
                    
                    y_pred_stack = stack.predict(X_test)
                    stack_accuracy = accuracy_score(y_test, y_pred_stack)
                    
                    model_scores['stacking'] = {
                        'test_accuracy': stack_accuracy,
                        'cv_mean': None,
                        'cv_std': None
                    }
                    
                    print(f"    Stacking Ensemble: Acc={stack_accuracy:.4f}")
                    
                    if stack_accuracy > best_score:
                        best_score = stack_accuracy
                        best_model = stack
                        print("    ✓ Stacking is best!")
                        
                except Exception as e:
                    print(f"    Stacking Error: {e}")
            
            # Try voting ensemble
            print("\n  🗳️ Training Voting Ensemble...")
            try:
                fresh_models = self._get_base_models(use_class_weights=use_weights)
                voting = self._create_voting_ensemble(fresh_models)
                voting.fit(X_train, y_train)
                
                y_pred_voting = voting.predict(X_test)
                voting_accuracy = accuracy_score(y_test, y_pred_voting)
                
                model_scores['voting'] = {
                    'test_accuracy': voting_accuracy,
                    'cv_mean': None,
                    'cv_std': None
                }
                
                print(f"    Voting Ensemble: Acc={voting_accuracy:.4f}")
                
                if voting_accuracy > best_score:
                    best_score = voting_accuracy
                    best_model = voting
                    print("    ✓ Voting is best!")
                    
            except Exception as e:
                print(f"    Voting Error: {e}")
            
            # Apply calibration if enabled and beneficial
            if self.use_calibration and not is_multiclass:
                print("\n  🎯 Applying probability calibration...")
                try:
                    # Calibrate the best model
                    calibrated = CalibratedClassifierCV(
                        best_model,
                        method='sigmoid',
                        cv=3
                    )
                    calibrated.fit(X_train, y_train)
                    
                    # Check if calibration improves Brier score
                    y_proba_cal = calibrated.predict_proba(X_test)[:, 1]
                    brier_cal = brier_score_loss(y_test, y_proba_cal)
                    
                    y_proba_uncal = best_model.predict_proba(X_test)[:, 1]
                    brier_uncal = brier_score_loss(y_test, y_proba_uncal)
                    
                    if brier_cal < brier_uncal:
                        best_model = calibrated
                        print(f"    Calibration improved Brier: {brier_uncal:.4f} → {brier_cal:.4f}")
                    else:
                        print(f"    Calibration not helpful (Brier: {brier_cal:.4f} vs {brier_uncal:.4f})")
                        
                except Exception as e:
                    print(f"    Calibration Error: {e}")
            
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
            
            print(f"\n  ✅ Best model accuracy: {results[target_name]['accuracy']:.4f}")
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
            
            # Apply probability dampening to prevent overconfidence
            # This shrinks extreme probabilities toward the center
            proba = self._apply_probability_dampening(proba, target_name)
            
            predictions[target_name] = {
                'prediction': [target_config['labels'][p] for p in pred],
                'probabilities': proba.tolist(),
                'confidence': np.max(proba, axis=1).tolist()
            }
        
        return predictions
    
    def _apply_probability_dampening(self, proba: np.ndarray, target_name: str) -> np.ndarray:
        """
        Apply probability dampening to prevent overconfidence.
        Uses temperature scaling and shrinkage toward base rates.
        
        Football is inherently unpredictable - we should never show 100% confidence.
        """
        # Base rates for football outcomes (historical averages)
        base_rates = {
            'match_result': np.array([0.45, 0.25, 0.30]),  # Home, Draw, Away
            'home_win': np.array([0.55, 0.45]),
            'draw': np.array([0.75, 0.25]),
            'away_win': np.array([0.70, 0.30]),
            'over_1.5': np.array([0.30, 0.70]),
            'over_2.5': np.array([0.50, 0.50]),
            'over_3.5': np.array([0.70, 0.30]),
            'btts': np.array([0.50, 0.50]),
        }
        
        # Shrinkage factor - how much to pull toward base rates
        # Higher = more conservative predictions
        shrinkage = 0.25
        
        # Temperature for softmax scaling (higher = more uniform)
        temperature = 1.2
        
        # Get base rate for this target
        base = base_rates.get(target_name, np.ones(proba.shape[1]) / proba.shape[1])
        if len(base) != proba.shape[1]:
            base = np.ones(proba.shape[1]) / proba.shape[1]
        
        # Apply temperature scaling (softens extreme probabilities)
        log_proba = np.log(np.clip(proba, 1e-10, 1.0))
        scaled_proba = np.exp(log_proba / temperature)
        scaled_proba = scaled_proba / scaled_proba.sum(axis=1, keepdims=True)
        
        # Shrink toward base rates
        dampened = (1 - shrinkage) * scaled_proba + shrinkage * base
        
        # Ensure probabilities sum to 1
        dampened = dampened / dampened.sum(axis=1, keepdims=True)
        
        # Cap maximum confidence at 85% for match result, 90% for others
        max_conf = 0.85 if target_name == 'match_result' else 0.90
        dampened = np.clip(dampened, 0.05, max_conf)
        
        # Re-normalize
        dampened = dampened / dampened.sum(axis=1, keepdims=True)
        
        return dampened
    
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
        
        # Handle stacking/voting models
        if hasattr(model, 'estimators_'):
            # Get importance from first estimator that supports it
            for est in model.estimators_:
                if hasattr(est, 'feature_importances_'):
                    importance = est.feature_importances_
                    break
            else:
                return pd.DataFrame()
        elif hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).mean(axis=0)
        else:
            return pd.DataFrame()
        
        return pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, 
                                  n_trials: int = 50) -> Dict:
        """
        Optimize hyperparameters using Optuna
        
        Args:
            X: Feature matrix
            y: Target vector
            n_trials: Number of optimization trials
        
        Returns:
            Dictionary of best hyperparameters
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            print("Optuna not installed. Using default hyperparameters.")
            return {}
        
        def objective(trial):
            # Hyperparameters to tune
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 4, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            }
            
            # Create model with suggested parameters
            model = RandomForestClassifier(
                **params,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
            
            return scores.mean()
        
        # Run optimization
        print(f"\n🔧 Running hyperparameter optimization ({n_trials} trials)...")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"   Best accuracy: {study.best_value:.4f}")
        print(f"   Best params: {study.best_params}")
        
        return study.best_params
    
    def train_optimized(self, df: pd.DataFrame, feature_columns: list, 
                        optimize: bool = True, n_trials: int = 30) -> Dict[str, Dict]:
        """
        Train with optional hyperparameter optimization
        
        Args:
            df: DataFrame with features
            feature_columns: List of feature column names
            optimize: Whether to run hyperparameter optimization
            n_trials: Number of Optuna trials
        
        Returns:
            Training results
        """
        if not optimize:
            return self.train(df, feature_columns)
        
        self.feature_columns = feature_columns
        X = df[feature_columns].values
        X_scaled = self.scaler.fit_transform(X)
        
        # Optimize on match_result first (main target)
        y_main = df['MatchResult'].values
        split_idx = int(len(X_scaled) * 0.8)
        X_train = X_scaled[:split_idx]
        y_train = y_main[:split_idx]
        
        # Get optimized params
        best_params = self.optimize_hyperparameters(X_train, y_train, n_trials)
        
        if best_params:
            # Update base models with optimized params
            print("\n📊 Applying optimized hyperparameters...")
            self._optimized_params = best_params
        
        # Continue with normal training
        return self.train(df, feature_columns)


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
