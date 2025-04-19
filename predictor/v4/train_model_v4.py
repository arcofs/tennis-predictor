"""
Tennis Match Prediction - Model Training (v4 - Fixed)

This script trains the XGBoost model for tennis match prediction:
1. Loads historical match data and features
2. Performs time-based train/val/test split with strict temporal separation
3. Tunes hyperparameters
4. Trains final model
5. Evaluates performance
6. Saves model and metadata for prediction pipeline

Important Changes to Fix Data Leakage:
1. Modified data loading to NOT calculate result directly from matches table
2. Added enhanced validation checks to detect signs of data leakage
3. Improved time-based train/val/test splitting to ensure strict temporal separation
4. Added checks for suspiciously high accuracy that may indicate leakage
5. Added detection of features that might directly leak results
6. Enhanced metadata to track data split information

These changes address the issue where the v4 model was achieving 99% accuracy
due to data leakage from directly using match outcomes during training.
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
        Load historical match data with features, removing any ID fields
        that might cause data leakage
        
        Returns:
            DataFrame with match data and features
        """
        # Modified query to prevent data leakage:
        # 1. Not joining with matches table to get results directly
        # 2. Explicitly excluding match_id and other ID fields
        query = """
            SELECT 
                -- Exclude ID columns that are causing data leakage
                -- f.id,
                -- f.match_id,  -- Removed as it correlates with result
                -- f.player1_id,
                -- f.player2_id,
                f.surface,
                f.tournament_date,
                f.tournament_level,
                f.result, -- Use pre-calculated result from match_features
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
                f.player2_bp_conversion_pct_5
            FROM match_features f
            WHERE f.result IS NOT NULL -- Ensure we have valid results
            ORDER BY f.tournament_date ASC
        """
        
        with self.get_db_connection() as conn:
            df = pd.read_sql(query, conn)
        
        # Verify class balance
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
        Create time-based train/validation/test split with strict temporal separation
        to prevent any data leakage between sets.
        
        Args:
            df: Input DataFrame
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Debug logging for datetime operations
        logger.info(f"Tournament date column type: {type(df['tournament_date'].iloc[0])}")
        logger.info(f"Tournament date sample: {df['tournament_date'].iloc[0]}")
        
        # Ensure we have datetime format for tournament_date
        if pd.api.types.is_object_dtype(df['tournament_date']):
            df['tournament_date'] = pd.to_datetime(df['tournament_date'])
            logger.info("Converted tournament_date from object to datetime")

        # Sort by date (enforcing strict chronological order)
        df = df.sort_values('tournament_date')
        
        # Find date cutoffs rather than using row indices
        # This handles multiple matches on same day better
        dates = df['tournament_date'].unique()
        logger.info(f"Unique dates type: {type(dates[0])}")
        dates = np.sort(dates)
        
        train_end_idx = int(len(dates) * train_ratio)
        val_end_idx = int(len(dates) * (train_ratio + val_ratio))
        
        train_end_date = dates[train_end_idx - 1]
        val_end_date = dates[val_end_idx - 1]
        
        logger.info(f"Train end date type: {type(train_end_date)}")
        logger.info(f"Train end date value: {train_end_date}")
        
        # Add one day to ensure strict separation
        train_end_date_exclusive = pd.Timestamp(train_end_date) + pd.Timedelta(days=1)
        val_end_date_exclusive = pd.Timestamp(val_end_date) + pd.Timedelta(days=1)
        
        # Split data by dates with explicit separation
        train_df = df[df['tournament_date'] < train_end_date_exclusive].copy()
        val_df = df[(df['tournament_date'] >= train_end_date_exclusive) & 
                   (df['tournament_date'] < val_end_date_exclusive)].copy()
        test_df = df[df['tournament_date'] >= val_end_date_exclusive].copy()
        
        # Log the date ranges using proper conversion
        logger.info(f"Training data: {pd.Timestamp(df['tournament_date'].min()).strftime('%Y-%m-%d')} to {pd.Timestamp(train_end_date).strftime('%Y-%m-%d')}")
        logger.info(f"Validation data: {pd.Timestamp(train_end_date_exclusive).strftime('%Y-%m-%d')} to {pd.Timestamp(val_end_date).strftime('%Y-%m-%d')}")
        logger.info(f"Test data: {pd.Timestamp(val_end_date_exclusive).strftime('%Y-%m-%d')} to {pd.Timestamp(df['tournament_date'].max()).strftime('%Y-%m-%d')}")
        
        # Verify split sizes
        logger.info(f"Split data into train ({len(train_df)}), val ({len(val_df)}), test ({len(test_df)}) sets")
        
        # Verify strict time separation
        train_max = train_df['tournament_date'].max()
        val_min = val_df['tournament_date'].min() if not val_df.empty else None
        val_max = val_df['tournament_date'].max() if not val_df.empty else None
        test_min = test_df['tournament_date'].min() if not test_df.empty else None
        
        if val_min is not None and val_min <= train_max:
            logger.error(f"CRITICAL ERROR: Time overlap between train and validation sets!")
            logger.error(f"Train max: {pd.Timestamp(train_max).strftime('%Y-%m-%d')}, Val min: {pd.Timestamp(val_min).strftime('%Y-%m-%d')}")
            raise ValueError("Train and validation sets have time overlap")
            
        if test_min is not None and val_max is not None and test_min <= val_max:
            logger.error(f"CRITICAL ERROR: Time overlap between validation and test sets!")
            logger.error(f"Val max: {pd.Timestamp(val_max).strftime('%Y-%m-%d')}, Test min: {pd.Timestamp(test_min).strftime('%Y-%m-%d')}")
            raise ValueError("Validation and test sets have time overlap")
        
        # Return the splits
        return train_df, val_df, test_df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature columns for training, explicitly excluding ID columns
        and any columns that could cause data leakage
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of feature column names
        """
        # Exclude non-feature columns and those that might cause data leakage
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
        
        # Exclude any columns that might be directly related to match outcome
        leakage_patterns = ['winner', 'loser', 'score', 'match_id', 'match_num']
        for pattern in leakage_patterns:
            for col in df.columns:
                if pattern in col.lower() and col not in exclude_cols:
                    logger.warning(f"Excluding potential leakage column: {col}")
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
        metrics: Dict[str, float],
        training_metadata: Dict[str, Any] = None
    ):
        """
        Save model and detailed metadata
        
        Args:
            model: Trained model
            feature_names: List of feature names
            metrics: Dictionary of evaluation metrics
            training_metadata: Optional dictionary of training metadata
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
        
        # Combine all metadata
        metadata = {
            'model_version': MODEL_VERSION,
            'training_date': datetime.now().isoformat(),
            'device': self.device,
            'tree_method': self.tree_method,
            'metrics': metrics,
            'feature_count': len(feature_names),
            'features': feature_names,
            'training_notes': [
                "Model trained with strict time-based split to prevent data leakage",
                "Enhanced validation performed to detect potential data leakage",
                f"Model accuracy: {metrics.get('accuracy', None)}"
            ]
        }
        
        # Add training_metadata if provided
        if training_metadata:
            metadata.update(training_metadata)
        
        # Save metadata
        metadata_path = MODEL_DIR / f"metadata_{MODEL_VERSION}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        logger.info(f"Saved model and metadata to {MODEL_DIR}")
        
        # Add a warning if accuracy is suspicious
        if metrics.get('accuracy', 0) > 0.90:
            logger.warning("NOTE: The model's high accuracy may indicate data leakage.")
            logger.warning("Review the training process and data preparation carefully.")
            logger.warning("Consider using the model only after thorough analysis and validation.")
    
    def validate_training_data(self, df: pd.DataFrame) -> bool:
        """
        Perform thorough data validation to check for data leakage
        
        Args:
            df: Input DataFrame with training data
            
        Returns:
            True if validation passes, False otherwise
        """
        validation_passed = True
        
        # Check 1: Ensure no matches with dates in the future
        if 'tournament_date' in df.columns:
            current_date = pd.Timestamp.now().normalize()  # Get current date without time
            
            # Debug logging
            logger.info(f"Validation - tournament_date type: {type(df['tournament_date'].iloc[0])}")
            
            # Convert to pandas datetime if needed
            if pd.api.types.is_object_dtype(df['tournament_date']):
                df['tournament_date'] = pd.to_datetime(df['tournament_date'])
            
            # Create normalized dates for comparison
            df['tournament_date_temp'] = df['tournament_date'].dt.normalize()
                
            future_date_matches = df[df['tournament_date_temp'] > current_date]
            if not future_date_matches.empty:
                logger.error(f"DATA LEAKAGE: Found {len(future_date_matches)} matches with dates in the future")
                earliest_future = future_date_matches['tournament_date_temp'].min()
                logger.error(f"Earliest future date: {pd.Timestamp(earliest_future).strftime('%Y-%m-%d')}")
                validation_passed = False
            
            # Clean up temporary column
            df.drop('tournament_date_temp', axis=1, inplace=True)
        
        # Check 2: Verify class balance to detect potential bias or leakage
        if 'result' in df.columns:
            pos_rate = df['result'].mean()
            if pos_rate < 0.40 or pos_rate > 0.60:
                logger.warning(f"POTENTIAL BIAS OR LEAKAGE: Class distribution is significantly imbalanced: {pos_rate:.2%} positive")
                logger.warning("This may indicate data issues, bias in match selection, or data leakage")
                # Treat extreme imbalance as a potential sign of leakage
                if pos_rate < 0.35 or pos_rate > 0.65:
                    logger.error("SEVERE CLASS IMBALANCE: This level of imbalance often indicates data leakage")
                    validation_passed = False
        
        # Check 3: Verify all result values are valid (0 or 1)
        if 'result' in df.columns:
            # Historical matches should have 0 or 1 result
            historical_invalid_results = df[
                (df['result'].notnull()) & 
                (~df['result'].isin([0, 1]))
            ].shape[0]
                
            if historical_invalid_results > 0:
                logger.error(f"DATA ERROR: {historical_invalid_results} historical matches have invalid result values")
                logger.error("Result values should be 0 or 1 for historical matches")
                validation_passed = False
        
        # Check 4: Check for near-perfect correlation between features and result (sign of leakage)
        corr_threshold = 0.85  # Threshold for concerning correlation
        # Calculate correlation of result with numeric features
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if 'result' in numeric_cols:
            numeric_cols.remove('result')
        
        if numeric_cols and 'result' in df.columns:
            correlations = df[numeric_cols + ['result']].corr()['result'].abs().sort_values(ascending=False)
            high_corr_features = correlations[correlations > corr_threshold].index.tolist()
            
            if 'result' in high_corr_features:
                high_corr_features.remove('result')
                
            if high_corr_features:
                logger.error(f"POTENTIAL DATA LEAKAGE: Found {len(high_corr_features)} features with suspiciously high correlation to result")
                logger.error(f"Top correlated features: {high_corr_features[:5]}")
                logger.error(f"These correlations (>{corr_threshold}) indicate potential data leakage")
                validation_passed = False
        
        # Check 5: Check for unrealistic accuracy in a simple model (quick check for obvious leakage)
        try:
            if len(df) > 1000 and 'result' in df.columns:
                from sklearn.model_selection import train_test_split
                from sklearn.ensemble import RandomForestClassifier
                
                # Use a subset of the data for quick validation
                sample_size = min(5000, len(df))
                df_sample = df.sample(sample_size, random_state=42)
                
                # Get numeric features only for quick check
                feature_cols = df_sample.select_dtypes(include=['number']).columns.tolist()
                if 'result' in feature_cols:
                    feature_cols.remove('result')
                
                # Split data
                X = df_sample[feature_cols].fillna(-999)  # Simple imputation for quick check
                y = df_sample['result']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                
                # Train a simple model
                clf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
                clf.fit(X_train, y_train)
                
                # Check accuracy
                train_accuracy = clf.score(X_train, y_train)
                test_accuracy = clf.score(X_test, y_test)
                
                if train_accuracy > 0.95:
                    logger.error(f"CRITICAL DATA LEAKAGE INDICATOR: Simple model has {train_accuracy:.4f} training accuracy")
                    logger.error("This is an almost certain sign of data leakage")
                    validation_passed = False
                
                if test_accuracy > 0.90:
                    logger.error(f"CRITICAL DATA LEAKAGE INDICATOR: Simple model has {test_accuracy:.4f} test accuracy")
                    logger.error("This is an almost certain sign of data leakage")
                    validation_passed = False
                
                logger.info(f"Data leakage quick check: Train accuracy={train_accuracy:.4f}, Test accuracy={test_accuracy:.4f}")
        except Exception as e:
            logger.warning(f"Could not perform leakage quick check with simple model: {e}")
        
        # Check 6: Special check for the match_features table structure
        try:
            if 'result' in df.columns:
                # Run a query to check if the match_features table is consistently biased
                with self.get_db_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute("""
                            SELECT 
                                COUNT(*) FILTER (WHERE result = 1) AS player1_win_count,
                                COUNT(*) FILTER (WHERE result = 0) AS player1_loss_count,
                                COUNT(*) AS total_count
                            FROM match_features
                            WHERE result IS NOT NULL
                        """)
                        result = cursor.fetchone()
                        
                        if result:
                            player1_win_count, player1_loss_count, total_count = result
                            if total_count > 0:
                                win_rate = player1_win_count / total_count
                                logger.info(f"Database check: Player1 win rate = {win_rate:.2%} ({player1_win_count}/{total_count})")
                                
                                # If player1 almost always wins or loses, there's a systematic issue in data generation
                                if win_rate > 0.95 or win_rate < 0.05:
                                    logger.error("CRITICAL DATABASE ISSUE: match_features table has systematic bias")
                                    logger.error(f"Player1 win rate = {win_rate:.2%}, which indicates the data is not properly generated")
                                    logger.error("This requires fixing the data generation pipeline in generate_historical_features.py")
                                    validation_passed = False
                        
                        # Check for genuine duplicates (more than 2 entries per match) in the feature table
                        cursor.execute("""
                            WITH match_counts AS (
                                SELECT 
                                    player1_id, 
                                    player2_id, 
                                    tournament_date,
                                    COUNT(*) as cnt
                                FROM match_features
                                WHERE player1_id IS NOT NULL 
                                    AND player2_id IS NOT NULL 
                                    AND tournament_date IS NOT NULL
                                GROUP BY player1_id, player2_id, tournament_date
                                HAVING COUNT(*) > 2  -- More than 2 because we expect 2 due to symmetric generation
                            )
                            SELECT COUNT(*) as duplicate_count
                            FROM match_counts;
                        """)
                        duplicate_count = cursor.fetchone()[0]
                        
                        if duplicate_count > 0:
                            logger.warning(f"Found {duplicate_count} matches with more than 2 entries (1 original + 1 symmetric)")
                            logger.warning("This affects approximately {:.2%} of the dataset".format(duplicate_count / (total_count / 2)))
                            # Only fail validation if duplicates affect more than 1% of the dataset
                            if duplicate_count > (total_count / 2) * 0.01:
                                logger.error("Duplicate matches affect more than 1% of the dataset")
                                validation_passed = False
                            else:
                                logger.info("Duplicate matches affect less than 1% of the dataset, proceeding with training")
        except Exception as e:
            logger.warning(f"Could not perform database structure check: {e}")
        
        if validation_passed:
            logger.info("Data validation passed: No data leakage detected")
        else:
            logger.error("DATA VALIDATION FAILED: Please fix data issues before training")
            
        return validation_passed

    def train(self):
        """Main training method with safeguards against data leakage"""
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
            
            # Additional safeguard: Remove any columns that might directly leak the result
            suspicious_cols = [col for col in feature_cols if 'winner' in col.lower() or 'loser' in col.lower()]
            if suspicious_cols:
                logger.warning(f"Removing potentially leaky features: {suspicious_cols}")
                feature_cols = [col for col in feature_cols if col not in suspicious_cols]
            
            # Split data with strict temporal separation
            train_df, val_df, test_df = self.create_train_val_test_split(df)
            
            # Final verification that we're not training on future data
            current_date = pd.Timestamp.now()
            if pd.api.types.is_object_dtype(train_df['tournament_date']):
                train_df['tournament_date'] = pd.to_datetime(train_df['tournament_date'])
            
            # Remove any matches with dates in the future
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
            
            # Check for perfect separation or unrealistic class balance in splits
            val_positive_rate = np.mean(y_val) if len(y_val) > 0 else 0
            test_positive_rate = np.mean(y_test) if len(y_test) > 0 else 0
            
            logger.info(f"Class distribution comparison:")
            logger.info(f"  - Train: {train_positive_rate:.2%} positive")
            logger.info(f"  - Validation: {val_positive_rate:.2%} positive")
            logger.info(f"  - Test: {test_positive_rate:.2%} positive")
            
            # Check for severe deviation in class distribution across splits
            max_deviation = 0.15  # Maximum allowable deviation in class distribution
            if (abs(train_positive_rate - val_positive_rate) > max_deviation or 
                abs(train_positive_rate - test_positive_rate) > max_deviation or
                abs(val_positive_rate - test_positive_rate) > max_deviation):
                logger.warning("POTENTIAL DATA ISSUE: Severe class imbalance differences between splits")
                logger.warning("This could indicate temporal data shifts or potential data leakage")
                
                # Continue anyway but note the warning
                logger.warning("Continuing with training, but results may be unreliable")
            
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
            
            # Check for unrealistically high performance (data leakage indicator)
            test_accuracy = test_metrics.get('accuracy', 0)
            if test_accuracy > 0.90:
                logger.warning(f"SUSPICIOUS PERFORMANCE: Test accuracy is {test_accuracy:.4f}, which is unusually high")
                logger.warning("This level of accuracy often indicates data leakage. Review your pipeline!")
            
            # Prepare training metadata
            training_metadata = {
                'data_splits': {
                    'train_size': len(train_df),
                    'val_size': len(val_df),
                    'test_size': len(test_df),
                    'train_date_range': [
                        pd.Timestamp(train_df['tournament_date'].min()).strftime('%Y-%m-%d'),
                        pd.Timestamp(train_df['tournament_date'].max()).strftime('%Y-%m-%d')
                    ],
                    'val_date_range': [
                        pd.Timestamp(val_df['tournament_date'].min()).strftime('%Y-%m-%d'),
                        pd.Timestamp(val_df['tournament_date'].max()).strftime('%Y-%m-%d')
                    ],
                    'test_date_range': [
                        pd.Timestamp(test_df['tournament_date'].min()).strftime('%Y-%m-%d'),
                        pd.Timestamp(test_df['tournament_date'].max()).strftime('%Y-%m-%d')
                    ],
                    'class_distribution': {
                        'train': float(train_positive_rate),
                        'val': float(val_positive_rate),
                        'test': float(test_positive_rate)
                    }
                },
                'hyperparameters': best_params,
                'validation_checks': {
                    'future_matches_removed': len(future_training_matches) if 'future_training_matches' in locals() else 0,
                    'suspicious_features_removed': len(suspicious_cols) if 'suspicious_cols' in locals() else 0
                }
            }
            
            # Save model and metadata
            self.save_model(model, feature_cols, test_metrics, training_metadata)
            
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