"""
Baseline models for optimal-stopping sell signal prediction
LightGBM and XGBoost implementations for upside-left regression and drawdown classification
"""
import numpy as np
import pandas as pd
import polars as pl
from typing import Dict, List, Tuple, Optional
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, roc_auc_score
import joblib
import os
from datetime import datetime, timedelta

from ..config import MODEL_CONFIG, OPTIMAL_STOPPING_CONFIG, MODEL_DIR

class UpsideLeftRegressor:
    """LightGBM/XGBoost ensemble for predicting upside-left (remaining profit potential)"""
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.models = {}
        self.feature_importance = {}
        
        # LightGBM parameters optimized for your RTX Pro 4500 setup
        self.lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 127,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'device_type': 'gpu' if use_gpu else 'cpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'max_bin': 255,
            'verbose': -1
        }
        
        # XGBoost parameters
        self.xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 10,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'tree_method': 'gpu_hist' if use_gpu else 'hist',
            'gpu_id': 0 if use_gpu else None,
            'predictor': 'gpu_predictor' if use_gpu else 'cpu_predictor'
        }
        
    def prepare_features(self, df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare feature matrix from extracted data"""
        
        # Define feature columns (exclude metadata and target)
        exclude_cols = {'timestamp', 'token', 'pool', 'bucket', 'upside_left', 
                       'net_return_current', 'entry_price', 'current_price'}
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Convert to pandas for model training (more stable with ML libs)
        df_pd = df.to_pandas()
        
        # Handle missing values
        X = df_pd[feature_cols].fillna(0).values
        y = df_pd['upside_left'].fillna(0).values
        
        return X, y, feature_cols
        
    def train_bucket_models(self, train_data: pl.DataFrame, 
                           validation_data: Optional[pl.DataFrame] = None) -> Dict[str, float]:
        """Train separate models for each liquidity bucket"""
        
        results = {}
        
        for bucket in OPTIMAL_STOPPING_CONFIG.horizon_hours.keys():
            print(f"Training models for {bucket} bucket...")
            
            # Filter data for this bucket
            bucket_train = train_data.filter(pl.col('bucket') == bucket)
            if len(bucket_train) < 1000:  # Skip if insufficient data
                print(f"Skipping {bucket}: insufficient data ({len(bucket_train)} samples)")
                continue
                
            X_train, y_train, feature_names = self.prepare_features(bucket_train)
            
            # Prepare validation data if provided
            X_val, y_val = None, None
            if validation_data is not None:
                bucket_val = validation_data.filter(pl.col('bucket') == bucket)
                if len(bucket_val) > 0:
                    X_val, y_val, _ = self.prepare_features(bucket_val)
            
            # Train LightGBM
            train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
            val_sets = [train_set]
            if X_val is not None:
                val_set = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_set)
                val_sets.append(val_set)
                
            lgb_model = lgb.train(
                self.lgb_params,
                train_set,
                valid_sets=val_sets,
                num_boost_round=1000,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(period=100)
                ]
            )
            
            # Train XGBoost
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
            evals = [(dtrain, 'train')]
            if X_val is not None:
                dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
                evals.append((dval, 'val'))
                
            xgb_model = xgb.train(
                self.xgb_params,
                dtrain,
                num_boost_round=1000,
                evals=evals,
                early_stopping_rounds=50,
                verbose_eval=100
            )
            
            # Store models
            self.models[bucket] = {
                'lgb': lgb_model,
                'xgb': xgb_model,
                'feature_names': feature_names
            }
            
            # Calculate validation metrics
            if X_val is not None:
                lgb_pred = lgb_model.predict(X_val)
                xgb_pred = xgb_model.predict(X_val)
                ensemble_pred = (lgb_pred + xgb_pred) / 2
                
                rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
                mae = mean_absolute_error(y_val, ensemble_pred)
                
                results[bucket] = {'rmse': rmse, 'mae': mae}
                print(f"{bucket} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
                
            # Store feature importance
            lgb_importance = lgb_model.feature_importance(importance_type='gain')
            self.feature_importance[bucket] = dict(zip(feature_names, lgb_importance))
            
        return results
        
    def predict(self, data: pl.DataFrame, bucket: str) -> np.ndarray:
        """Make predictions for given bucket"""
        
        if bucket not in self.models:
            raise ValueError(f"No trained model for bucket: {bucket}")
            
        X, _, _ = self.prepare_features(data)
        
        # Ensemble prediction
        lgb_pred = self.models[bucket]['lgb'].predict(X)
        xgb_pred = self.models[bucket]['xgb'].predict(X)
        
        return (lgb_pred + xgb_pred) / 2
        
    def save_models(self, base_path: str = None):
        """Save trained models to disk"""
        if base_path is None:
            base_path = os.path.join(MODEL_DIR, f'upside_left_models_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
            
        os.makedirs(base_path, exist_ok=True)
        
        for bucket, model_dict in self.models.items():
            bucket_path = os.path.join(base_path, bucket)
            os.makedirs(bucket_path, exist_ok=True)
            
            # Save LightGBM
            model_dict['lgb'].save_model(os.path.join(bucket_path, 'lgb_model.txt'))
            
            # Save XGBoost  
            model_dict['xgb'].save_model(os.path.join(bucket_path, 'xgb_model.json'))
            
            # Save metadata
            joblib.dump({
                'feature_names': model_dict['feature_names'],
                'feature_importance': self.feature_importance[bucket]
            }, os.path.join(bucket_path, 'metadata.pkl'))
            
        print(f"Models saved to: {base_path}")
        return base_path
        
    def load_models(self, base_path: str):
        """Load models from disk"""
        self.models = {}
        self.feature_importance = {}
        
        for bucket in os.listdir(base_path):
            bucket_path = os.path.join(base_path, bucket)
            if not os.path.isdir(bucket_path):
                continue
                
            # Load LightGBM
            lgb_model = lgb.Booster(model_file=os.path.join(bucket_path, 'lgb_model.txt'))
            
            # Load XGBoost
            xgb_model = xgb.Booster()
            xgb_model.load_model(os.path.join(bucket_path, 'xgb_model.json'))
            
            # Load metadata
            metadata = joblib.load(os.path.join(bucket_path, 'metadata.pkl'))
            
            self.models[bucket] = {
                'lgb': lgb_model,
                'xgb': xgb_model,
                'feature_names': metadata['feature_names']
            }
            self.feature_importance[bucket] = metadata['feature_importance']


class DrawdownClassifier:
    """Binary classifier to predict risk of significant drawdown"""
    
    def __init__(self, drawdown_threshold: float = 0.05, use_gpu: bool = True):
        self.drawdown_threshold = drawdown_threshold  # 5% drawdown threshold
        self.use_gpu = use_gpu
        self.models = {}
        
        # Parameters optimized for binary classification
        self.lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 50,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'device_type': 'gpu' if use_gpu else 'cpu',
            'verbose': -1
        }
        
    def create_drawdown_labels(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create binary labels for drawdown prediction"""
        
        # Calculate future drawdown for each position
        df_with_labels = []
        
        for (token, pool, entry_time), group in df.group_by(['token', 'pool', 'entry_price']):
            group_sorted = group.sort('timestamp')
            
            for i, row in enumerate(group_sorted.iter_rows(named=True)):
                current_time = row['timestamp']
                current_return = row['net_return_current']
                
                # Look at future returns to calculate max drawdown
                future_returns = group_sorted.slice(i, None).select('net_return_current').to_series()
                
                if len(future_returns) > 1:
                    # Calculate drawdown from current point
                    peak = current_return
                    max_drawdown = 0
                    
                    for future_return in future_returns[1:]:  # Skip current
                        if future_return > peak:
                            peak = future_return
                        drawdown = peak - future_return
                        max_drawdown = max(max_drawdown, drawdown)
                    
                    # Binary label: 1 if max future drawdown > threshold
                    drawdown_label = 1 if max_drawdown > self.drawdown_threshold else 0
                else:
                    drawdown_label = 0
                    
                row_dict = row.copy()
                row_dict['drawdown_label'] = drawdown_label
                df_with_labels.append(row_dict)
                
        return pl.DataFrame(df_with_labels)
        
    def train(self, train_data: pl.DataFrame) -> Dict[str, float]:
        """Train drawdown classifier"""
        
        # Create labels
        labeled_data = self.create_drawdown_labels(train_data)
        
        results = {}
        
        for bucket in OPTIMAL_STOPPING_CONFIG.horizon_hours.keys():
            bucket_data = labeled_data.filter(pl.col('bucket') == bucket)
            if len(bucket_data) < 500:
                continue
                
            # Prepare features (same as upside-left regressor)
            exclude_cols = {'timestamp', 'token', 'pool', 'bucket', 'upside_left', 
                           'net_return_current', 'entry_price', 'current_price', 'drawdown_label'}
            
            feature_cols = [col for col in bucket_data.columns if col not in exclude_cols]
            
            df_pd = bucket_data.to_pandas()
            X = df_pd[feature_cols].fillna(0).values
            y = df_pd['drawdown_label'].values
            
            # Check class balance
            pos_ratio = np.mean(y)
            print(f"{bucket} - Positive class ratio: {pos_ratio:.3f}")
            
            if pos_ratio < 0.05 or pos_ratio > 0.95:  # Skip if too imbalanced
                print(f"Skipping {bucket}: class imbalance")
                continue
                
            # Train with class weights
            train_set = lgb.Dataset(X, label=y, feature_name=feature_cols)
            
            params = self.lgb_params.copy()
            params['scale_pos_weight'] = (1 - pos_ratio) / pos_ratio  # Balance classes
            
            model = lgb.train(
                params,
                train_set,
                num_boost_round=500,
                callbacks=[lgb.log_evaluation(period=100)]
            )
            
            self.models[bucket] = {
                'model': model,
                'feature_names': feature_cols
            }
            
            # Evaluate on training data (for debugging)
            pred_proba = model.predict(X)
            pred_binary = (pred_proba > 0.5).astype(int)
            
            accuracy = accuracy_score(y, pred_binary)
            try:
                auc = roc_auc_score(y, pred_proba)
                results[bucket] = {'accuracy': accuracy, 'auc': auc}
                print(f"{bucket} - Accuracy: {accuracy:.3f}, AUC: {auc:.3f}")
            except ValueError:
                results[bucket] = {'accuracy': accuracy, 'auc': np.nan}
                print(f"{bucket} - Accuracy: {accuracy:.3f}, AUC: N/A (single class)")
                
        return results
        
    def predict_proba(self, data: pl.DataFrame, bucket: str) -> np.ndarray:
        """Predict probability of significant drawdown"""
        
        if bucket not in self.models:
            return np.zeros(len(data))  # Conservative: assume no risk if no model
            
        exclude_cols = {'timestamp', 'token', 'pool', 'bucket', 'upside_left', 
                       'net_return_current', 'entry_price', 'current_price'}
        
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        df_pd = data.to_pandas()
        X = df_pd[feature_cols].fillna(0).values
        
        return self.models[bucket]['model'].predict(X)


class SellDecisionEngine:
    """Combined decision engine using upside-left regression and drawdown classification"""
    
    def __init__(self):
        self.upside_regressor = UpsideLeftRegressor()
        self.drawdown_classifier = DrawdownClassifier()
        
    def train(self, train_data: pl.DataFrame, validation_data: Optional[pl.DataFrame] = None):
        """Train both models"""
        
        print("Training upside-left regressor...")
        upside_results = self.upside_regressor.train_bucket_models(train_data, validation_data)
        
        print("\nTraining drawdown classifier...")
        drawdown_results = self.drawdown_classifier.train(train_data)
        
        return {
            'upside_results': upside_results,
            'drawdown_results': drawdown_results
        }
        
    def should_sell(self, data: pl.DataFrame, bucket: str) -> np.ndarray:
        """Make sell/hold decisions based on ensemble of models"""
        
        # Get predictions
        upside_pred = self.upside_regressor.predict(data, bucket)
        drawdown_proba = self.drawdown_classifier.predict_proba(data, bucket)
        
        # Decision logic from configuration
        upside_threshold = OPTIMAL_STOPPING_CONFIG.upside_threshold_pct / 100
        drawdown_threshold = OPTIMAL_STOPPING_CONFIG.drawdown_prob_threshold
        
        # Sell if: low upside AND high drawdown risk
        sell_signals = (upside_pred < upside_threshold) & (drawdown_proba > drawdown_threshold)
        
        return sell_signals.astype(int)
        
    def save_models(self, base_path: str = None):
        """Save both models"""
        if base_path is None:
            base_path = os.path.join(MODEL_DIR, f'sell_decision_engine_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
            
        upside_path = self.upside_regressor.save_models(os.path.join(base_path, 'upside_models'))
        
        # Save drawdown classifier
        drawdown_path = os.path.join(base_path, 'drawdown_models')
        os.makedirs(drawdown_path, exist_ok=True)
        joblib.dump(self.drawdown_classifier.models, os.path.join(drawdown_path, 'models.pkl'))
        
        return base_path
