"""
Tennis Match Prediction - Model Training (v4)

This script trains the XGBoost model for tennis match prediction:
1. Loads historical match data and features
2. Performs time-based train/val/test split
3. Tunes hyperparameters
4. Trains final model
5. Evaluates performance
6. Saves model and metadata for prediction pipeline
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
import xgboost as xgb
import optuna
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import psycopg2
import json
import multiprocessing

# Add project root to path to allow imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{project_root}/predictor/v4/output/logs/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
MODEL_VERSION = "v4_" + datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = project_root / "predictor/v4"
MODEL_DIR = OUTPUT_DIR / "models"
PLOTS_DIR = OUTPUT_DIR / "output/plots"

# Create necessary directories
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Check GPU availability
def check_gpu_availability():
    """Check if CUDA GPU is available for XGBoost"""
    try:
        # Check XGBoost version
        xgb_version = tuple(map(int, xgb.__version__.split('.')))
        logger.info(f"XGBoost version: {xgb.__version__}")
        
        # For XGBoost 2.0.0 and newer, use different detection methods
        if xgb_version >= (2, 0, 0):
            # Try to create a simple DMatrix and train with device='cuda'
            try:
                test_data = np.random.rand(10, 10)
                test_labels = np.random.randint(0, 2, 10)
                test_dmatrix = xgb.DMatrix(test_data, label=test_labels)
                
                # Try to train a small model with GPU
                test_params = {'tree_method': 'hist', 'device': 'cuda'}
                xgb.train(test_params, test_dmatrix, num_boost_round=1)
                return True
            except Exception as e:
                logger.info(f"GPU detection failed with newer API: {e}")
                return False
        else:
            # For older XGBoost versions, use older API
            # Check if CUDA is available
            try:
                gpu_available = xgb.config.get_config().get('use_cuda', False)
                
                if not gpu_available:
                    # Try to create a simple DMatrix with tree_method='gpu_hist'
                    test_data = np.random.rand(10, 10)
                    test_labels = np.random.randint(0, 2, 10)
                    test_dmatrix = xgb.DMatrix(test_data, label=test_labels)
                    
                    # Try to train a small model with GPU
                    test_params = {'tree_method': 'gpu_hist'}
                    xgb.train(test_params, test_dmatrix, num_boost_round=1)
                    gpu_available = True
                
                return gpu_available
            except Exception as e:
                logger.info(f"GPU detection failed with older API: {e}")
                return False
    
    except Exception as e:
        logger.info(f"GPU detection failed: {e}")
        return False

# Get optimal number of CPU threads
def get_optimal_cpu_threads():
    """Get optimal number of CPU threads (total cores - 4, minimum 1)"""
    total_cores = multiprocessing.cpu_count()
    # Use total cores minus 4, with minimum of 1
    return max(1, total_cores - 4)

class ModelTrainer:
    def __init__(self):
        """Initialize the model trainer"""
        load_dotenv()
        
        # Database settings
        self.db_url = os.getenv("DATABASE_URL")
        if self.db_url and self.db_url.startswith('postgres://'):
            self.db_url = self.db_url.replace('postgres://', 'postgresql://', 1)
        
        # Check for GPU availability
        self.use_gpu = check_gpu_availability()
        if self.use_gpu:
            logger.info("CUDA GPU available and will be used for training")
            self.device = 'cuda'
            self.tree_method = 'hist'  # Use histogram method (works for both CPU and GPU)
        else:
            logger.info("No GPU detected, falling back to CPU")
            self.device = 'cpu'
            self.tree_method = 'hist'  # Use CPU histogram for training
            
            # Set optimal number of threads for CPU
            self.n_jobs = get_optimal_cpu_threads()
            logger.info(f"Using {self.n_jobs} CPU threads for training")
    
    def get_db_connection(self):
        """Create a database connection"""
        return psycopg2.connect(self.db_url)
    
    def load_training_data(self) -> pd.DataFrame:
        """
        Load historical match data with features
        
        Returns:
            DataFrame with match data and features
        """
        # This query gets completed matches from match_features
        # Important note on match ID relationships:
        # - For historical matches, match_features.match_id = matches.id (auto-incremented PK)
        # - For future matches (excluded here with is_future IS NOT TRUE), we'd use scheduled_matches.match_id
        query = """
            SELECT 
                f.id,
                f.match_id,
                f.player1_id,
                f.player2_id,
                f.surface,
                f.tournament_date,
                f.tournament_level,
                f.is_future,
                -- All the diff features
                f.player_elo_diff,
                f.win_rate_5_diff,
                f.win_streak_diff,
                f.loss_streak_diff,
                f.win_rate_hard_5_diff,
                f.win_rate_clay_5_diff,
                f.win_rate_grass_5_diff,
                f.win_rate_carpet_5_diff,
                f.win_rate_hard_overall_diff,
                f.win_rate_clay_overall_diff,
                f.win_rate_grass_overall_diff,
                f.win_rate_carpet_overall_diff,
                f.serve_efficiency_5_diff,
                f.first_serve_pct_5_diff,
                f.first_serve_win_pct_5_diff,
                f.second_serve_win_pct_5_diff,
                f.ace_pct_5_diff,
                f.bp_saved_pct_5_diff,
                f.return_efficiency_5_diff,
                f.bp_conversion_pct_5_diff,
                -- All player1 features
                f.player1_win_rate_5,
                f.player1_win_streak,
                f.player1_loss_streak,
                f.player1_win_rate_hard_5,
                f.player1_win_rate_clay_5,
                f.player1_win_rate_grass_5,
                f.player1_win_rate_carpet_5,
                f.player1_win_rate_hard_overall,
                f.player1_win_rate_clay_overall,
                f.player1_win_rate_grass_overall,
                f.player1_win_rate_carpet_overall,
                f.player1_serve_efficiency_5,
                f.player1_first_serve_pct_5,
                f.player1_first_serve_win_pct_5,
                f.player1_second_serve_win_pct_5,
                f.player1_ace_pct_5,
                f.player1_bp_saved_pct_5,
                f.player1_return_efficiency_5,
                f.player1_bp_conversion_pct_5,
                -- All player2 features
                f.player2_win_rate_5,
                f.player2_win_streak,
                f.player2_loss_streak,
                f.player2_win_rate_hard_5,
                f.player2_win_rate_clay_5,
                f.player2_win_rate_grass_5,
                f.player2_win_rate_carpet_5,
                f.player2_win_rate_hard_overall,
                f.player2_win_rate_clay_overall,
                f.player2_win_rate_grass_overall,
                f.player2_win_rate_carpet_overall,
                f.player2_serve_efficiency_5,
                f.player2_first_serve_pct_5,
                f.player2_first_serve_win_pct_5,
                f.player2_second_serve_win_pct_5,
                f.player2_ace_pct_5,
                f.player2_bp_saved_pct_5,
                f.player2_return_efficiency_5,
                f.player2_bp_conversion_pct_5,
                -- Result
                m.winner_id = f.player1_id as result
            FROM match_features f
            JOIN matches m ON f.match_id = m.id
            WHERE f.is_future IS NOT TRUE
            AND m.winner_id IS NOT NULL
            ORDER BY f.tournament_date ASC
        """
        
        with self.get_db_connection() as conn:
            df = pd.read_sql(query, conn)
        
        # Additional safety check to ensure no future matches are included
        if 'is_future' in df.columns:
            future_matches = df[df['is_future'] == True]
            if not future_matches.empty:
                logger.error(f"Found {len(future_matches)} future matches in training data! Removing them.")
                df = df[df['is_future'] != True]
        
        # Verify we have proper class balance
        positive_rate = df['result'].mean()
        logger.info(f"Class distribution: {positive_rate:.2%} positive (player1 wins)")
        
        logger.info(f"Loaded {len(df)} matches for training")
        return df
    
    def create_train_val_test_split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create time-based train/validation/test split
        
        Args:
            df: Input DataFrame
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Ensure we have datetime format for tournament_date
        if pd.api.types.is_object_dtype(df['tournament_date']):
            df['tournament_date'] = pd.to_datetime(df['tournament_date'])

        # Sort by date (enforcing strict chronological order)
        df = df.sort_values('tournament_date')
        
        # Calculate split points based on dates
        min_date = df['tournament_date'].min()
        max_date = df['tournament_date'].max()
        date_range = (max_date - min_date).days
        
        train_end_date = min_date + pd.Timedelta(days=int(date_range * train_ratio))
        val_end_date = min_date + pd.Timedelta(days=int(date_range * (train_ratio + val_ratio)))
        
        # Split data by dates
        train_df = df[df['tournament_date'] <= train_end_date]
        val_df = df[(df['tournament_date'] > train_end_date) & (df['tournament_date'] <= val_end_date)]
        test_df = df[df['tournament_date'] > val_end_date]
        
        # Log the date ranges
        logger.info(f"Training data: {min_date.strftime('%Y-%m-%d')} to {train_end_date.strftime('%Y-%m-%d')}")
        logger.info(f"Validation data: {(train_end_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')} to {val_end_date.strftime('%Y-%m-%d')}")
        logger.info(f"Test data: {(val_end_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
        
        # Verify there is no overlap
        train_max = train_df['tournament_date'].max()
        val_min = val_df['tournament_date'].min() if not val_df.empty else None
        val_max = val_df['tournament_date'].max() if not val_df.empty else None
        test_min = test_df['tournament_date'].min() if not test_df.empty else None
        
        if val_min is not None and val_min <= train_max:
            logger.warning(f"Time overlap between train and validation sets! Train max: {train_max}, Val min: {val_min}")
        if test_min is not None and val_max is not None and test_min <= val_max:
            logger.warning(f"Time overlap between validation and test sets! Val max: {val_max}, Test min: {test_min}")
        
        logger.info(f"Split data into train ({len(train_df)}), val ({len(val_df)}), test ({len(test_df)}) sets")
        return train_df, val_df, test_df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature columns for training
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of feature column names
        """
        # Exclude non-feature columns and date/timestamp columns
        exclude_cols = [
            'id', 'match_id', 'player1_id', 'player2_id', 'tournament_date',
            'surface', 'tournament_level', 'result', 'is_future',
            'created_at', 'updated_at',  # Exclude timestamp columns
        ]
        
        # Check data types and exclude any date, datetime or timestamp columns
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]) or 'date' in col.lower() or 'time' in col.lower():
                if col not in exclude_cols:
                    exclude_cols.append(col)
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Log categorical features - XGBoost can handle these properly
        for col in feature_cols:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                # Convert to categorical type to ensure proper handling
                df[col] = df[col].astype('category')
                logger.info(f"Found categorical feature: {col}")
        
        logger.info(f"Using {len(feature_cols)} features for training")
        return feature_cols
    
    def prepare_features(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: List[str]
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Prepare feature matrices and labels for training
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            feature_cols: List of feature columns
            
        Returns:
            Tuple of ((X_train, X_val, X_test), (y_train, y_val, y_test))
        """
        # Ensure categorical columns are properly handled
        categorical_cols = []
        for col in feature_cols:
            if (train_df[col].dtype.name == 'category' or 
                train_df[col].dtype == 'object'):
                categorical_cols.append(col)
                # Convert to categorical in all dataframes
                train_df[col] = train_df[col].astype('category')
                val_df[col] = val_df[col].astype('category')
                test_df[col] = test_df[col].astype('category')
        
        if categorical_cols:
            logger.info(f"Handling {len(categorical_cols)} categorical features: {categorical_cols}")
            
            # Ensure categories in validation and test sets match those in training
            for col in categorical_cols:
                # Get categories from training set
                categories = train_df[col].cat.categories
                
                # Update categories in validation and test sets
                val_df[col] = pd.Categorical(val_df[col], categories=categories)
                test_df[col] = pd.Categorical(test_df[col], categories=categories)
        
        # Create feature matrices
        X_train = train_df[feature_cols].values
        X_val = val_df[feature_cols].values
        X_test = test_df[feature_cols].values
        
        # Create label vectors
        y_train = train_df['result'].values
        y_val = val_df['result'].values
        y_test = test_df['result'].values
        
        return (X_train, X_val, X_test), (y_train, y_val, y_test)
    
    def tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str],
        n_trials: int = 100
    ) -> Dict[str, Any]:
        """
        Tune model hyperparameters using Optuna
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            feature_names: List of feature names
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary of best hyperparameters
        """
        def objective(trial):
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'gamma': trial.suggest_float('gamma', 0, 1),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                'tree_method': self.tree_method,
                'device': self.device,  # Set device parameter for GPU/CPU
            }
            
            # Add device-specific parameters
            if self.device == 'cpu':
                params['nthread'] = self.n_jobs
            
            # Get number of boosting rounds separately
            num_boost_round = trial.suggest_int('n_estimators', 100, 1000)
            
            # Create validation set - enable categorical features
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names, enable_categorical=True)
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names, enable_categorical=True)
            
            # Train model
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                evals=[(dval, 'val')],
                early_stopping_rounds=50,
                verbose_eval=False
            )
            
            # Get validation score
            y_pred = model.predict(dval)
            val_auc = roc_auc_score(y_val, y_pred)
            
            return val_auc
        
        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # Get best parameters and add n_estimators separately
        best_params = study.best_params.copy()
        n_estimators = best_params.pop('n_estimators', 100)  # Remove n_estimators and get its value
        
        # Log best parameters
        logger.info(f"Best hyperparameters: {best_params}, num_boost_round: {n_estimators}")
        logger.info(f"Best validation AUC: {study.best_value:.4f}")
        
        # Return complete parameters including n_estimators
        best_params['n_estimators'] = n_estimators
        return best_params
    
    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str],
        params: Dict[str, Any]
    ) -> xgb.Booster:
        """
        Train final model with best hyperparameters
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            feature_names: List of feature names
            params: Model hyperparameters
            
        Returns:
            Trained XGBoost model
        """
        # Extract num_boost_round and remove n_estimators from params
        num_boost_round = params.pop('n_estimators', 100)
        
        # Add tree method and device-specific parameters
        params['tree_method'] = self.tree_method
        params['device'] = self.device  # Set device parameter for GPU/CPU
        
        if self.device == 'cpu':
            params['nthread'] = self.n_jobs
        
        # Create datasets - enable categorical features
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names, enable_categorical=True)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names, enable_categorical=True)
        
        # Set up early stopping callback
        early_stopping_rounds = 50
        
        # Dictionary to store best iteration
        evals_result = {}
        
        # Train model
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=10,
            evals_result=evals_result
        )
        
        # Get best iteration - compatible with newer XGBoost versions
        best_iteration = getattr(model, 'best_iteration', 0)
        if best_iteration == 0:
            # Try alternative method for newer versions
            try:
                best_iteration = model.best_ntree_limit
            except AttributeError:
                # For newest versions, try attributes or len(evals_result)
                best_iteration = getattr(model, 'best_iteration', num_boost_round)
        
        logger.info(f"Trained model with {best_iteration} trees")
        
        # Store best_iteration for future reference
        model.best_iteration = best_iteration
        
        return model
    
    def evaluate_model(
        self,
        model: xgb.Booster,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        dataset_name: str = "Test"
    ) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            X: Feature matrix
            y: True labels
            feature_names: List of feature names
            dataset_name: Name of dataset for logging
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Get predictions
        dmatrix = xgb.DMatrix(X, feature_names=feature_names, enable_categorical=True)
        
        # Use best iteration if available (handles different XGBoost versions)
        iteration_to_use = None
        if hasattr(model, 'best_iteration'):
            iteration_to_use = model.best_iteration
        elif hasattr(model, 'best_ntree_limit'):
            iteration_to_use = model.best_ntree_limit
            
        # Make predictions
        if iteration_to_use is not None:
            logger.info(f"Using best iteration {iteration_to_use} for prediction")
            y_pred_proba = model.predict(dmatrix, iteration_range=(0, iteration_to_use))
        else:
            logger.info("No best iteration found, using all trees")
            y_pred_proba = model.predict(dmatrix)
            
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'auc': roc_auc_score(y, y_pred_proba)
        }
        
        # Log results
        logger.info(f"\n{dataset_name} Set Performance:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # Plot confusion matrix
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"{dataset_name} Set Confusion Matrix")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(PLOTS_DIR / f"confusion_matrix_{dataset_name.lower()}.png")
        plt.close()
        
        # Plot feature importance
        importance_scores = model.get_score(importance_type='gain')
        importance_df = pd.DataFrame(
            list(importance_scores.items()),
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=importance_df.head(20), x='Importance', y='Feature')
        plt.title(f"Top 20 Feature Importance ({dataset_name} Set)")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"feature_importance_{dataset_name.lower()}.png")
        plt.close()
        
        return metrics
    
    def save_model(
        self,
        model: xgb.Booster,
        feature_names: List[str],
        metrics: Dict[str, float]
    ):
        """
        Save model and metadata
        
        Args:
            model: Trained model
            feature_names: List of feature names
            metrics: Dictionary of evaluation metrics
        """
        # Save model
        model_path = MODEL_DIR / f"model_{MODEL_VERSION}.json"
        model.save_model(str(model_path))
        
        # Save feature names
        features_path = MODEL_DIR / f"features_{MODEL_VERSION}.json"
        with open(features_path, 'w') as f:
            json.dump(feature_names, f)
        
        # Save metrics
        metrics_path = MODEL_DIR / f"metrics_{MODEL_VERSION}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)
        
        # Save metadata including device used for training
        metadata = {
            'model_version': MODEL_VERSION,
            'training_date': datetime.now().isoformat(),
            'device': self.device,
            'tree_method': self.tree_method,
            'metrics': metrics
        }
        
        metadata_path = MODEL_DIR / f"metadata_{MODEL_VERSION}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        logger.info(f"Saved model and metadata to {MODEL_DIR}")
    
    def validate_training_data(self, df: pd.DataFrame) -> bool:
        """
        Perform thorough data validation to check for data leakage
        
        Args:
            df: Input DataFrame with training data
            
        Returns:
            True if validation passes, False otherwise
        """
        validation_passed = True
        
        # Check 1: Ensure no future matches in training data
        future_matches = None
        if 'is_future' in df.columns:
            future_matches = df[df['is_future'] == True]
            if not future_matches.empty:
                logger.error(f"DATA LEAKAGE: Found {len(future_matches)} future matches in training data")
                logger.error("This would cause the model to train on data it shouldn't have access to")
                validation_passed = False
                
                # Additional critical check: future matches with results
                future_with_results = future_matches[future_matches['result'].notnull()]
                if not future_with_results.empty:
                    logger.error(f"CRITICAL DATA LEAKAGE: {len(future_with_results)} future matches have result values!")
                    logger.error("This indicates a severe pipeline issue that must be fixed immediately")
                    logger.error("Future matches should NEVER have result values set")
                    
                    # Sample of problematic matches
                    if len(future_with_results) > 0:
                        sample = future_with_results.head(min(5, len(future_with_results)))
                        logger.error(f"Sample of problematic matches:\n{sample[['match_id', 'player1_id', 'player2_id', 'result']]}")
        
        # Check 2: Ensure no matches with dates in the future
        if 'tournament_date' in df.columns:
            current_date = datetime.now().date()
            if pd.api.types.is_object_dtype(df['tournament_date']):
                df['tournament_date_temp'] = pd.to_datetime(df['tournament_date']).dt.date
            else:
                df['tournament_date_temp'] = df['tournament_date'].dt.date
                
            future_date_matches = df[df['tournament_date_temp'] > current_date]
            if not future_date_matches.empty:
                logger.error(f"DATA LEAKAGE: Found {len(future_date_matches)} matches with dates in the future")
                earliest_future = future_date_matches['tournament_date_temp'].min()
                logger.error(f"Earliest future date: {earliest_future}")
                validation_passed = False
        
        # Check 3: Verify class balance to detect potential bias
        if 'result' in df.columns:
            pos_rate = df['result'].mean()
            if pos_rate < 0.45 or pos_rate > 0.55:
                logger.warning(f"POTENTIAL BIAS: Class distribution is imbalanced: {pos_rate:.2%} positive")
                logger.warning("This may indicate data issues or bias in match selection")
                # Don't fail validation for this, but warn
        
        # Check 4: Verify all result values are valid (0 or 1, or NULL for future matches)
        if 'result' in df.columns:
            # Future matches should have NULL result
            future_matches_invalid_results = 0
            if 'is_future' in df.columns:
                future_matches_invalid_results = df[(df['is_future'] == True) & (df['result'].notnull())].shape[0]
                if future_matches_invalid_results > 0:
                    logger.error(f"DATA ERROR: {future_matches_invalid_results} future matches have non-NULL results")
                    validation_passed = False
            
            # Historical matches should have 0 or 1 result
            historical_invalid_results = 0
            if 'is_future' in df.columns:
                historical_invalid_results = df[
                    (df['is_future'] != True) & 
                    (df['result'].notnull()) & 
                    (~df['result'].isin([0, 1]))
                ].shape[0]
            else:
                historical_invalid_results = df[
                    (df['result'].notnull()) & 
                    (~df['result'].isin([0, 1]))
                ].shape[0]
                
            if historical_invalid_results > 0:
                logger.error(f"DATA ERROR: {historical_invalid_results} historical matches have invalid result values")
                logger.error("Result values should be 0 or 1 for historical matches")
                validation_passed = False
        
        if validation_passed:
            logger.info("Data validation passed: No data leakage detected")
        else:
            logger.error("DATA VALIDATION FAILED: Please fix data issues before training")
            
        return validation_passed

    def train(self):
        """Main training method"""
        try:
            # Log training device
            logger.info(f"Training on {self.device.upper()}")
            
            # Load data
            df = self.load_training_data()
            
            # Validate data to check for leakage
            validation_passed = self.validate_training_data(df)
            if not validation_passed:
                logger.error("Aborting training due to data validation failures")
                return
            
            # Double-check that no future matches are included
            assert 'is_future' not in df.columns or df[df['is_future'] == True].empty, "Training data contains future matches!"
            
            # Get feature columns
            feature_cols = self.get_feature_columns(df)
            
            # Split data
            train_df, val_df, test_df = self.create_train_val_test_split(df)
            
            # Final verification that we're not training on future data
            current_date = datetime.now()
            if pd.api.types.is_object_dtype(train_df['tournament_date']):
                train_df['tournament_date'] = pd.to_datetime(train_df['tournament_date'])
            
            future_training_matches = train_df[train_df['tournament_date'] > current_date]
            if not future_training_matches.empty:
                logger.error(f"Found {len(future_training_matches)} matches in training data with dates in the future!")
                logger.error("Removing these matches from training data.")
                train_df = train_df[train_df['tournament_date'] <= current_date]
            
            # Prepare features
            (X_train, X_val, X_test), (y_train, y_val, y_test) = self.prepare_features(
                train_df, val_df, test_df, feature_cols
            )
            
            # Verify class balance in training set
            train_positive_rate = np.mean(y_train)
            logger.info(f"Training data class distribution: {train_positive_rate:.2%} positive (player1 wins)")
            
            # Tune hyperparameters
            best_params = self.tune_hyperparameters(
                X_train, y_train, X_val, y_val, feature_cols
            )
            
            # Train final model
            model = self.train_model(
                X_train, y_train, X_val, y_val, feature_cols, best_params
            )
            
            # Evaluate on test set
            test_metrics = self.evaluate_model(
                model, X_test, y_test, feature_cols, "Test"
            )
            
            # Save model and metadata
            self.save_model(model, feature_cols, test_metrics)
            
            logger.info("Training completed successfully")
            
        except Exception as e:
            logger.error(f"Error in training: {e}")
            raise

def main():
    """Main execution function"""
    try:
        trainer = ModelTrainer()
        trainer.train()
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 