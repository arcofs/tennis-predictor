import os
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional, Union, Any
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, brier_score_loss, log_loss,
    classification_report
)
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap
from tqdm import tqdm
import json
import warnings
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
import multiprocessing
from functools import partial
import optuna
from optuna.visualization import plot_param_importances, plot_optimization_history
import pickle
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# For reproducibility
np.random.seed(42)

# Set up multiprocessing - use half of available cores by default
NUM_CORES = max(1, multiprocessing.cpu_count() // 2)

# Project directories
PROJECT_ROOT = Path(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "predictor" / "output" / "v3"
MODELS_DIR = PROJECT_ROOT / "models" / "v3"

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# File paths
INPUT_FILE = DATA_DIR / "v3" / "features_v3.csv"
MODEL_OUTPUT = MODELS_DIR / "tennis_model_v3.json"
PIPELINE_OUTPUT = MODELS_DIR / "model_pipeline_v3.pkl"
METRICS_OUTPUT = OUTPUT_DIR / "metrics_v3.json"
HYPERPARAMS_OUTPUT = OUTPUT_DIR / "hyperparameters_v3.json"
PLOTS_DIR = OUTPUT_DIR / "plots"
OPTUNA_STUDY_OUTPUT = OUTPUT_DIR / "optuna_study_v3.pkl"

# Create plots directory if it doesn't exist
os.makedirs(PLOTS_DIR, exist_ok=True)

# Constants
SEED = 42
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SURFACES = ['Hard', 'Clay', 'Grass', 'Carpet']

# XGBoost model parameters - baseline parameters, will be tuned
MODEL_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'],
    'eta': 0.01,  # Lower learning rate for better generalization
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'gamma': 0.1,
    'scale_pos_weight': 1,
    'seed': SEED,
    'tree_method': 'hist',  # More efficient for large datasets
    'grow_policy': 'lossguide',
    'max_bin': 256,  # Reduces memory usage
    'verbosity': 0,
    'reg_alpha': 0.1,  # L1 regularization
    'reg_lambda': 1.0,  # L2 regularization
    'num_parallel_tree': 1,  # Single tree to start with
    'missing': float('nan')  # Explicitly handle missing values
}

# Define hyperparameter search space
HYPERPARAMETER_SPACE = {
    'max_depth': [3, 4, 5, 6, 7, 8],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'reg_alpha': [0, 0.001, 0.01, 0.1, 1],
    'reg_lambda': [0.1, 0.5, 1, 5, 10],
    'eta': [0.01, 0.03, 0.05, 0.1]
}

# Define serve and return feature names
SERVE_RETURN_FEATURES = [
    'serve_efficiency_5_diff',
    'first_serve_pct_5_diff',
    'first_serve_win_pct_5_diff',
    'second_serve_win_pct_5_diff',
    'ace_pct_5_diff',
    'bp_saved_pct_5_diff',
    'return_efficiency_5_diff',
    'bp_conversion_pct_5_diff'
]

# Raw player features
RAW_PLAYER_FEATURES = [
    'win_rate_5',
    'win_streak',
    'loss_streak'
]

# Surface-specific features will be dynamically generated for each surface

# Database settings
DB_BATCH_SIZE = 20000
DB_TIMEOUT_SECONDS = 300

def get_database_connection() -> psycopg2.extensions.connection:
    """
    Create database connection using environment variables.
    
    Returns:
        psycopg2 connection object
    """
    load_dotenv()
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL not found in environment variables")
    
    try:
        # Add connection timeout and statement timeout settings
        connection = psycopg2.connect(
            database_url,
            connect_timeout=10,
            options=f"-c statement_timeout={DB_TIMEOUT_SECONDS * 1000}"
        )
        logger.info("Successfully connected to database")
        return connection
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        raise

def load_data_from_database(limit: Optional[int] = None, 
                           progress_tracker: Optional['ProgressTracker'] = None) -> pd.DataFrame:
    """
    Load the tennis match features from the database efficiently.
    
    Args:
        limit: Optional limit on number of rows to fetch
        progress_tracker: Optional progress tracker
        
    Returns:
        DataFrame with tennis match features
    """
    logger.info("Loading data from database...")
    
    # Connect to database
    conn = get_database_connection()
    
    try:
        # Define the query with dynamic row limit
        limit_clause = f"LIMIT {limit}" if limit else ""
        
        # Use batched loading to handle large datasets efficiently
        offset = 0
        dataframes = []
        total_rows = 0
        
        if limit:
            total_to_fetch = limit
        else:
            # Get total count first
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM match_features")
                total_to_fetch = cursor.fetchone()[0]
        
        logger.info(f"Fetching up to {total_to_fetch} rows from database")
        pbar = tqdm(total=total_to_fetch, desc="Loading data from database")
        
        while True:
            # Define batch query
            query = f"""
            SELECT * FROM match_features
            ORDER BY tournament_date
            {limit_clause}
            OFFSET {offset}
            LIMIT {DB_BATCH_SIZE}
            """
            
            # Load batch
            batch_df = pd.read_sql(query, conn)
            
            # If batch is empty, we're done
            if len(batch_df) == 0:
                break
                
            # Append to list of dataframes
            dataframes.append(batch_df)
            
            # Update counts
            rows_fetched = len(batch_df)
            total_rows += rows_fetched
            pbar.update(rows_fetched)
            
            # Check if we've reached the limit
            if limit and total_rows >= limit:
                break
                
            # Update offset for next batch
            offset += DB_BATCH_SIZE
        
        pbar.close()
        
        # Combine all batches
        if dataframes:
            df = pd.concat(dataframes, ignore_index=True)
            
            # Convert date columns to datetime
            if 'tournament_date' in df.columns:
                df['tournament_date'] = pd.to_datetime(df['tournament_date'])
            
            # Sort by date
            df = df.sort_values(by='tournament_date').reset_index(drop=True)
            
            logger.info(f"Loaded {len(df)} match features")
            
            if progress_tracker:
                progress_tracker.update("Data loading complete")
            
            return df
        else:
            logger.warning("No data retrieved from database")
            return pd.DataFrame()
    
    finally:
        conn.close()

def load_data_from_file(file_path: Union[str, Path], 
                       progress_tracker: Optional['ProgressTracker'] = None) -> pd.DataFrame:
    """
    Load the tennis match features dataset from a file.
    
    Args:
        file_path: Path to the input CSV file
        progress_tracker: Optional progress tracker
        
    Returns:
        DataFrame with tennis match features
    """
    logger.info(f"Loading data from {file_path}")
    
    try:
        # Use chunked reading for large files
        chunks = pd.read_csv(file_path, chunksize=DB_BATCH_SIZE)
        df_list = []
        
        # Process each chunk
        for chunk in tqdm(chunks, desc="Reading data chunks"):
            df_list.append(chunk)
        
        # Combine chunks
        df = pd.concat(df_list, ignore_index=True)
        
        # Convert date columns to datetime
        if 'tournament_date' in df.columns:
            df['tournament_date'] = pd.to_datetime(df['tournament_date'])
        
        # Sort by date
        df = df.sort_values(by='tournament_date').reset_index(drop=True)
        
        logger.info(f"Loaded {len(df)} match features")
        
        if progress_tracker:
            progress_tracker.update("Data loading complete")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading data from file: {e}")
        raise

class ProgressTracker:
    """
    Class to track and log progress during model training.
    """
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.description = description
    
    def update(self, description: str = None):
        """Update progress and log status."""
        self.current_step += 1
        elapsed = time.time() - self.start_time
        
        if description:
            message = f"{description}: "
        else:
            message = f"{self.description}: "
            
        if self.current_step < self.total_steps:
            est_remaining = (elapsed / self.current_step) * (self.total_steps - self.current_step)
            logger.info(f"{message}Progress: {self.percent_complete}% complete. Est. remaining time: {est_remaining:.1f}s")
        else:
            logger.info(f"{message}Progress: 100% complete. Total time: {elapsed:.1f}s")
    
    @property
    def percent_complete(self) -> int:
        """Calculate percentage of completion."""
        return int((self.current_step / self.total_steps) * 100)


class XGBoostProgressCallback(xgb.callback.TrainingCallback):
    """
    Callback to track XGBoost training progress.
    """
    def __init__(self, total_rounds: int):
        self.total_rounds = total_rounds
        self.current_round = 0
        self.start_time = time.time()
        
    def after_iteration(self, model, epoch, evals_log):
        """Log progress after each iteration."""
        self.current_round += 1
        if self.current_round % 10 == 0 or self.current_round == self.total_rounds:
            elapsed = time.time() - self.start_time
            progress = (self.current_round / self.total_rounds) * 100
            
            if self.current_round < self.total_rounds:
                est_remaining = (elapsed / self.current_round) * (self.total_rounds - self.current_round)
                logger.info(f"XGBoost training: {progress:.1f}% complete. Est. remaining time: {est_remaining:.1f}s")
            else:
                logger.info(f"XGBoost training: 100% complete. Total time: {elapsed:.1f}s")
                
        return False  # Continue training


def create_time_based_train_val_test_split(df: pd.DataFrame, 
                                        train_size: float = 0.7,
                                        val_size: float = 0.15,
                                        test_size: float = 0.15,
                                        progress_tracker: Optional[ProgressTracker] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create a strict time-based train-validation-test split to prevent any data leakage.
    
    Args:
        df: DataFrame with match features
        train_size: Proportion of data to use for training
        val_size: Proportion of data to use for validation
        test_size: Proportion of data to use for testing
        progress_tracker: Optional progress tracker
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info("Creating time-based train-validation-test split")
    
    # Verify split proportions sum to 1
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        logger.warning(f"Split proportions sum to {total}, not 1.0. Normalizing.")
        train_size = train_size / total
        val_size = val_size / total
        test_size = test_size / total
    
    # Sort by date to ensure chronological ordering
    df = df.sort_values(by='tournament_date').reset_index(drop=True)
    
    # Calculate split indices
    n = len(df)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)
    
    # Split data
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    # Calculate date ranges for reporting
    train_start = train_df['tournament_date'].min().strftime('%Y-%m-%d')
    train_end_date = train_df['tournament_date'].max().strftime('%Y-%m-%d')
    
    val_start = val_df['tournament_date'].min().strftime('%Y-%m-%d')
    val_end_date = val_df['tournament_date'].max().strftime('%Y-%m-%d')
    
    test_start = test_df['tournament_date'].min().strftime('%Y-%m-%d')
    test_end_date = test_df['tournament_date'].max().strftime('%Y-%m-%d')
    
    # Log split information
    logger.info(f"Train set: {len(train_df)} matches ({train_size*100:.1f}%) from {train_start} to {train_end_date}")
    logger.info(f"Validation set: {len(val_df)} matches ({val_size*100:.1f}%) from {val_start} to {val_end_date}")
    logger.info(f"Test set: {len(test_df)} matches ({test_size*100:.1f}%) from {test_start} to {test_end_date}")
    
    if progress_tracker:
        progress_tracker.update("Time-based split complete")
    
    return train_df, val_df, test_df


def get_feature_columns(df: pd.DataFrame, progress_tracker: Optional[ProgressTracker] = None) -> List[str]:
    """
    Get the list of feature columns for training.
    Following XGBoost best practices:
    - Keep numerical features as is (no scaling needed)
    - Handle categorical features appropriately
    - Focus on feature selection rather than transformation
    
    Args:
        df: DataFrame with match features
        progress_tracker: Optional progress tracker
        
    Returns:
        List of feature column names
    """
    logger.info("Identifying feature columns with XGBoost best practices")
    
    # Columns to exclude from features
    exclude_cols = [
        'id', 'match_id', 'tournament_date', 'player1_id', 'player2_id', 
        'surface', 'result', 'created_at', 'updated_at'
    ]
    
    # Get all numeric columns except excluded ones
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Generate list of surface-specific serve and return features
    surface_specific_features = []
    for feature in SERVE_RETURN_FEATURES:
        base_feature = feature.split('_diff')[0]
        for surface in SURFACES:
            surface_feature = f"{base_feature}_{surface}_diff"
            if surface_feature in df.columns and surface_feature not in feature_cols:
                surface_specific_features.append(surface_feature)
    
    # Generate list of surface-specific win rate features
    for surface in SURFACES:
        # Most recent win rate on specific surface
        surface_win_rate = f"win_rate_{surface}_5_diff"
        if surface_win_rate in df.columns and surface_win_rate not in feature_cols:
            surface_specific_features.append(surface_win_rate)
        
        # Overall win rate on specific surface
        surface_overall_win_rate = f"win_rate_{surface}_overall_diff"
        if surface_overall_win_rate in df.columns and surface_overall_win_rate not in feature_cols:
            surface_specific_features.append(surface_overall_win_rate)
    
    # Generate list of raw player features (for both player1 and player2)
    raw_player_features = []
    for feature in RAW_PLAYER_FEATURES:
        player1_feature = f"player1_{feature}"
        player2_feature = f"player2_{feature}"
        
        if player1_feature in df.columns and player1_feature not in feature_cols:
            raw_player_features.append(player1_feature)
        
        if player2_feature in df.columns and player2_feature not in feature_cols:
            raw_player_features.append(player2_feature)
    
    # Generate raw player serve/return features
    for feature in SERVE_RETURN_FEATURES:
        base_feature = feature.split('_diff')[0]
        player1_feature = f"player1_{base_feature}"
        player2_feature = f"player2_{base_feature}"
        
        if player1_feature in df.columns and player1_feature not in feature_cols:
            raw_player_features.append(player1_feature)
        
        if player2_feature in df.columns and player2_feature not in feature_cols:
            raw_player_features.append(player2_feature)
    
    # Ensure all surface-specific and raw player features are included
    for feature in surface_specific_features + raw_player_features:
        if feature not in feature_cols:
            feature_cols.append(feature)
    
    # Check for categorical features in the data
    categorical_features = []
    for col in feature_cols:
        if col in df.columns and df[col].dtype == 'category':
            categorical_features.append(col)
    
    if categorical_features:
        logger.info(f"Identified {len(categorical_features)} categorical features: {categorical_features}")
        logger.info("XGBoost will handle categorical features based on numerical splits")
    
    # Log feature statistics
    numeric_feature_count = len(feature_cols) - len(categorical_features)
    logger.info(f"Selected {len(feature_cols)} features ({numeric_feature_count} numeric, {len(categorical_features)} categorical)")
    
    if progress_tracker:
        progress_tracker.update("Feature selection complete following XGBoost best practices")
    
    return feature_cols


def prepare_features(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    feature_cols: List[str],
    progress_tracker: Optional[ProgressTracker] = None
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray], List[int]]:
    """
    Prepare feature matrices and labels for training, validation, and testing.
    Following XGBoost best practices:
    - Let XGBoost handle missing values by keeping NaNs
    - Identify categorical features but leave them as is
    - No scaling needed for tree-based models
    
    Args:
        train_df: Training data
        val_df: Validation data
        test_df: Test data
        feature_cols: List of feature columns
        progress_tracker: Optional progress tracker
        
    Returns:
        Tuple of:
        - (X_train, X_val, X_test): Feature matrices
        - (y_train, y_val, y_test): Target vectors
        - categorical_feature_indices: Indices of categorical features
    """
    logger.info("Preparing features following XGBoost best practices")
    
    # Identify categorical features
    categorical_features = []
    categorical_indices = []
    
    for i, col in enumerate(feature_cols):
        # Check if column is categorical
        if col in train_df.columns and (
            train_df[col].dtype.name == 'category' or 
            (train_df[col].dtype == 'object' and train_df[col].nunique() < 100)
        ):
            categorical_features.append(col)
            categorical_indices.append(i)
    
    if categorical_features:
        logger.info(f"Found {len(categorical_features)} categorical features: {categorical_features}")
    
    # Extract features and labels
    X_train = train_df[feature_cols].values
    y_train = train_df['result'].values
    
    X_val = val_df[feature_cols].values
    y_val = val_df['result'].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df['result'].values
    
    # Log feature statistics
    logger.info(f"Training set shape: {X_train.shape}, {Counter(y_train)}")
    logger.info(f"Validation set shape: {X_val.shape}, {Counter(y_val)}")
    logger.info(f"Test set shape: {X_test.shape}, {Counter(y_test)}")
    
    # Log missing value statistics
    train_missing_rates = (np.isnan(X_train).sum(axis=0) / X_train.shape[0]) * 100
    high_missing_features = [(feature_cols[i], rate) for i, rate in enumerate(train_missing_rates) if rate > 10]
    
    if high_missing_features:
        logger.info("Features with high missing rates (>10%):")
        for feature, rate in high_missing_features:
            logger.info(f"  - {feature}: {rate:.2f}%")
        logger.info("XGBoost will handle these missing values optimally")
    
    if progress_tracker:
        progress_tracker.update("Feature preparation complete")
    
    return (X_train, X_val, X_test), (y_train, y_val, y_test), categorical_indices


def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    categorical_indices: List[int] = None,
    progress_tracker: Optional[ProgressTracker] = None,
    n_trials: int = 50,
    timeout: int = 3600
) -> Dict[str, Any]:
    """
    Tune XGBoost hyperparameters using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        feature_names: List of feature names
        categorical_indices: Indices of categorical features (if any)
        progress_tracker: Optional progress tracker
        n_trials: Number of trials for optimization
        timeout: Timeout in seconds
        
    Returns:
        Dictionary of optimized hyperparameters
    """
    logger.info(f"Tuning hyperparameters with Optuna (n_trials={n_trials}, timeout={timeout}s)")
    
    # Define objective function for Optuna
    def objective(trial: optuna.Trial) -> float:
        """Objective function for hyperparameter optimization."""
        # Define hyperparameter search space
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.8, 1.2),
            'missing': float('nan')  # XGBoost will handle missing values optimally
        }
        
        # Enable categorical feature support if needed
        if categorical_indices and len(categorical_indices) > 0:
            params['enable_categorical'] = True
        
        # Create DMatrix objects
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names,
                             enable_categorical=params.get('enable_categorical', False))
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names,
                           enable_categorical=params.get('enable_categorical', False))
        
        # Set up categorical features if present
        if categorical_indices:
            for cat_idx in categorical_indices:
                dtrain.set_feature_types(['c' if i == cat_idx else 'q' for i in range(X_train.shape[1])])
                dval.set_feature_types(['c' if i == cat_idx else 'q' for i in range(X_val.shape[1])])
        
        # Train model with early stopping
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-logloss")
        evals = [(dtrain, "train"), (dval, "validation")]
        
        # Use a moderate number of boosting rounds for tuning
        bst = xgb.train(
            params, dtrain, num_boost_round=1000, 
            evals=evals, early_stopping_rounds=50, 
            callbacks=[pruning_callback], verbose_eval=False
        )
        
        # Return validation metric as objective value
        eval_result = float(bst.best_score)
        return eval_result
    
    # Create Optuna study
    study_name = f"xgboost_tennis_prediction_{time.strftime('%Y%m%d_%H%M%S')}"
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    
    # Run hyperparameter optimization
    try:
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
        
        # Get best parameters
        best_params = study.best_params
        
        # Add fixed parameters
        best_params.update({
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'missing': float('nan')
        })
        
        # Enable categorical support if needed
        if categorical_indices and len(categorical_indices) > 0:
            best_params['enable_categorical'] = True
        
        # Log best parameters
        logger.info(f"Best hyperparameters: {best_params}")
        logger.info(f"Best validation score: {study.best_value:.6f}")
        
        # Save study results
        with open(OPTUNA_STUDY_OUTPUT, 'wb') as f:
            pickle.dump(study, f)
        logger.info(f"Optuna study saved to {OPTUNA_STUDY_OUTPUT}")
        
        # Create optimization history plot
        try:
            fig = plot_optimization_history(study)
            fig.write_image(str(PLOTS_DIR / "optuna_history.png"))
            
            fig = plot_param_importances(study)
            fig.write_image(str(PLOTS_DIR / "optuna_param_importance.png"))
            
            logger.info("Optuna plots saved to plots directory")
        except Exception as e:
            logger.warning(f"Failed to create Optuna plots: {e}")
        
        if progress_tracker:
            progress_tracker.update(f"Hyperparameter tuning complete - best validation score: {study.best_value:.6f}")
        
        return best_params
    
    except Exception as e:
        logger.error(f"Error in hyperparameter tuning: {e}")
        logger.info("Falling back to default parameters")
        
        # Return default parameters on error
        default_params = MODEL_PARAMS.copy()
        
        # Enable categorical support if needed
        if categorical_indices and len(categorical_indices) > 0:
            default_params['enable_categorical'] = True
            
        if progress_tracker:
            progress_tracker.update("Using default parameters due to tuning error")
            
        return default_params


def train_model(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_val: np.ndarray, 
    y_val: np.ndarray, 
    feature_names: List[str],
    categorical_features: List[int] = None,
    params: Dict[str, Any] = None,
    early_stopping_rounds: int = 50,
    progress_tracker: Optional[ProgressTracker] = None
) -> Tuple[xgb.Booster, Dict[str, List[float]]]:
    """
    Train XGBoost model with early stopping using validation data.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        feature_names: List of feature names
        categorical_features: Indices of categorical features (if any)
        params: XGBoost parameters (uses MODEL_PARAMS if None)
        early_stopping_rounds: Number of rounds for early stopping
        progress_tracker: Optional progress tracker
        
    Returns:
        Trained XGBoost model and dict with evaluation history
    """
    logger.info("Training XGBoost model with early stopping")
    
    start_time = time.time()
    
    if params is None:
        params = MODEL_PARAMS.copy()
    
    # Enable categorical feature support if categorical features are present
    if categorical_features and len(categorical_features) > 0:
        params['enable_categorical'] = True
        logger.info(f"Enabling categorical feature support for {len(categorical_features)} features")
    
    dtrain = xgb.DMatrix(
        X_train, 
        label=y_train, 
        feature_names=feature_names,
        enable_categorical=params.get('enable_categorical', False)
    )
    
    if categorical_features:
        for cat_idx in categorical_features:
            dtrain.set_feature_types(['c' if i == cat_idx else 'q' for i in range(X_train.shape[1])])
    
    dval = xgb.DMatrix(
        X_val, 
        label=y_val, 
        feature_names=feature_names,
        enable_categorical=params.get('enable_categorical', False)
    )
    
    if categorical_features:
        for cat_idx in categorical_features:
            dval.set_feature_types(['c' if i == cat_idx else 'q' for i in range(X_val.shape[1])])
    
    # Setup evaluation metrics
    evals = [(dtrain, 'train'), (dval, 'val')]
    
    # Train model with early stopping
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=5000,  # Maximum number of rounds
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=100  # Print evaluation every 100 rounds
    )
    
    training_time = time.time() - start_time
    best_iteration = model.best_iteration
    best_score = model.best_score
    
    logger.info(f"Model training completed in {training_time:.2f} seconds")
    logger.info(f"Best iteration: {best_iteration}, Best validation score: {best_score:.6f}")
    
    if progress_tracker:
        progress_tracker.update(f"XGBoost model trained (iterations: {best_iteration})")
    
    # Return model and evaluation history
    evals_result = model.evals_result()
    
    return model, evals_result


def evaluate_model(model: xgb.Booster, X: np.ndarray, y: np.ndarray, 
                  df: pd.DataFrame, feature_cols: List[str],
                  dataset_name: str = "Test",
                  threshold: float = 0.5,
                  progress_tracker: Optional[ProgressTracker] = None) -> Dict[str, Any]:
    """
    Evaluate the trained model with comprehensive metrics.
    
    Args:
        model: Trained XGBoost model
        X: Features
        y: Labels
        df: Original DataFrame with metadata
        feature_cols: List of feature column names
        dataset_name: Name of the dataset for logging
        threshold: Classification threshold
        progress_tracker: Optional progress tracker
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info(f"Evaluating model on {dataset_name} data")
    
    # Create DMatrix
    dmatrix = xgb.DMatrix(X, label=y, feature_names=feature_cols)
    
    # Make predictions
    y_pred_proba = model.predict(dmatrix)
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate standard classification metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    # Calculate ROC AUC
    try:
        roc_auc = auc(*roc_curve(y, y_pred_proba)[:2])
    except Exception as e:
        logger.warning(f"Could not calculate ROC AUC: {e}")
        roc_auc = None
    
    # Calculate average precision (PR AUC)
    try:
        pr_auc = average_precision_score(y, y_pred_proba)
    except Exception as e:
        logger.warning(f"Could not calculate PR AUC: {e}")
        pr_auc = None
    
    # Calculate log loss
    try:
        logloss = log_loss(y, y_pred_proba)
    except Exception as e:
        logger.warning(f"Could not calculate log loss: {e}")
        logloss = None
    
    # Calculate Brier score (calibration metric)
    try:
        brier = brier_score_loss(y, y_pred_proba)
    except Exception as e:
        logger.warning(f"Could not calculate Brier score: {e}")
        brier = None
    
    # Log metrics
    logger.info(f"{dataset_name} metrics:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    if roc_auc:
        logger.info(f"  ROC AUC: {roc_auc:.4f}")
    if pr_auc:
        logger.info(f"  PR AUC: {pr_auc:.4f}")
    if logloss:
        logger.info(f"  Log Loss: {logloss:.4f}")
    if brier:
        logger.info(f"  Brier Score: {brier:.4f}")
    
    # Calculate surface-specific metrics
    surface_metrics = {}
    
    for surface in SURFACES:
        surface_idx = df['surface'] == surface
        
        if sum(surface_idx) > 0:
            y_true_surface = y[surface_idx]
            y_pred_surface = y_pred[surface_idx]
            y_proba_surface = y_pred_proba[surface_idx]
            
            if len(y_true_surface) > 0:
                accuracy_surface = accuracy_score(y_true_surface, y_pred_surface)
                precision_surface = precision_score(y_true_surface, y_pred_surface, zero_division=0)
                recall_surface = recall_score(y_true_surface, y_pred_surface, zero_division=0)
                f1_surface = f1_score(y_true_surface, y_pred_surface, zero_division=0)
                
                # ROC AUC for this surface
                try:
                    roc_auc_surface = auc(*roc_curve(y_true_surface, y_proba_surface)[:2])
                except Exception:
                    roc_auc_surface = None
                
                surface_metrics[surface] = {
                    'accuracy': accuracy_surface,
                    'precision': precision_surface,
                    'recall': recall_surface,
                    'f1': f1_surface,
                    'roc_auc': roc_auc_surface,
                    'count': int(sum(surface_idx))
                }
                
                logger.info(f"Metrics for {surface} surface (n={sum(surface_idx)}):")
                logger.info(f"  Accuracy: {accuracy_surface:.4f}")
                logger.info(f"  F1 Score: {f1_surface:.4f}")
                if roc_auc_surface:
                    logger.info(f"  ROC AUC: {roc_auc_surface:.4f}")
        else:
            logger.info(f"No {dataset_name} samples for {surface} surface")
    
    # Calculate metrics by confidence range
    confidence_bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    confidence_metrics = []
    
    for i in range(len(confidence_bins) - 1):
        lower = confidence_bins[i]
        upper = confidence_bins[i + 1]
        
        # Find predictions in this confidence range (from both sides of 0.5)
        mask = (
            ((y_pred_proba >= lower) & (y_pred_proba < upper)) | 
            ((y_pred_proba <= (1 - lower)) & (y_pred_proba > (1 - upper)))
        )
        
        if sum(mask) > 0:
            bin_acc = accuracy_score(y[mask], y_pred[mask])
            
            confidence_metrics.append({
                'confidence_range': f"{lower:.1f}-{upper:.1f}",
                'accuracy': bin_acc,
                'count': int(sum(mask)),
                'percentage': float(sum(mask) / len(y) * 100)
            })
            
            logger.info(f"Accuracy for confidence {lower:.1f}-{upper:.1f}: {bin_acc:.4f} " 
                      f"(n={sum(mask)}, {sum(mask) / len(y) * 100:.1f}%)")
    
    # Compile all metrics
    metrics = {
        'overall': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'log_loss': logloss,
            'brier_score': brier,
            'count': len(y)
        },
        'by_surface': surface_metrics,
        'by_confidence': confidence_metrics,
        'threshold': threshold,
        'dataset': dataset_name
    }
    
    if progress_tracker:
        progress_tracker.update(f"Model evaluation on {dataset_name} data complete")
    
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                          output_path: Optional[Union[str, Path]] = None,
                          title: str = "Confusion Matrix",
                          progress_tracker: Optional[ProgressTracker] = None) -> None:
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_path: Path to save the plot
        title: Plot title
        progress_tracker: Optional progress tracker
    """
    logger.info(f"Plotting {title}")
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title(f'{title} (Counts)')
    ax1.set_xticklabels(['Loss', 'Win'])
    ax1.set_yticklabels(['Loss', 'Win'])
    
    # Plot percentages
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues', cbar=False, ax=ax2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title(f'{title} (Percentages)')
    ax2.set_xticklabels(['Loss', 'Win'])
    ax2.set_yticklabels(['Loss', 'Win'])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {output_path}")
    
    plt.close()
    
    if progress_tracker:
        progress_tracker.update(f"{title} plotting complete")


def plot_feature_importance(importances: Dict[str, float], top_n: int = 20, 
                           output_path: Optional[Union[str, Path]] = None,
                           progress_tracker: Optional[ProgressTracker] = None) -> None:
    """
    Plot and save feature importance.
    
    Args:
        importances: Dictionary of feature importances
        top_n: Number of top features to plot
        output_path: Path to save the plot
        progress_tracker: Optional progress tracker
    """
    logger.info(f"Plotting top {top_n} feature importances")
    
    # Sort and limit to top N
    sorted_importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n])
    
    # Calculate total importance for normalization
    total_importance = sum(sorted_importances.values())
    normalized_importances = {k: v/total_importance*100 for k, v in sorted_importances.items()}
    
    # Reverse order for horizontal bar chart (to have highest at top)
    features = list(reversed(list(normalized_importances.keys())))
    values = list(reversed(list(normalized_importances.values())))
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Plot horizontal bar chart with percentages
    bars = plt.barh(features, values, color='cornflowerblue')
    
    # Add percentage labels
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f'{values[i]:.1f}%',
                va='center', fontsize=10)
    
    plt.xlabel('Relative Importance (%)')
    plt.title(f'Top {top_n} Feature Importances')
    plt.gca().spines[['right', 'top']].set_visible(False)  # Clean up
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {output_path}")
    
    plt.close()
    
    if progress_tracker:
        progress_tracker.update("Feature importance plotting complete")


def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                  output_path: Optional[Union[str, Path]] = None,
                  title: str = "ROC Curve",
                  progress_tracker: Optional[ProgressTracker] = None) -> None:
    """
    Plot and save ROC curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        output_path: Path to save the plot
        title: Plot title
        progress_tracker: Optional progress tracker
    """
    logger.info(f"Plotting {title}")
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    # Plot thresholds
    threshold_markers = [0.1, 0.3, 0.5, 0.7, 0.9]
    for threshold in threshold_markers:
        # Find closest threshold
        idx = (np.abs(thresholds - threshold)).argmin()
        plt.plot(fpr[idx], tpr[idx], 'o', markersize=8, 
                label=f'Threshold = {thresholds[idx]:.2f}')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved ROC curve to {output_path}")
    
    plt.close()
    
    if progress_tracker:
        progress_tracker.update(f"{title} plotting complete")


def plot_precision_recall_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                              output_path: Optional[Union[str, Path]] = None,
                              title: str = "Precision-Recall Curve",
                              progress_tracker: Optional[ProgressTracker] = None) -> None:
    """
    Plot and save precision-recall curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        output_path: Path to save the plot
        title: Plot title
        progress_tracker: Optional progress tracker
    """
    logger.info(f"Plotting {title}")
    
    # Calculate PR curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {pr_auc:.3f})')
    
    # Add baseline
    baseline = sum(y_true) / len(y_true)
    plt.axhline(y=baseline, color='navy', lw=2, linestyle='--', label=f'Baseline (= {baseline:.3f})')
    
    # Plot thresholds
    threshold_markers = [0.1, 0.3, 0.5, 0.7, 0.9]
    for threshold in threshold_markers:
        if len(thresholds) > 0:  # Check if we have thresholds
            # Find closest threshold (handling edge case)
            idx = min(len(thresholds) - 1, (np.abs(thresholds - threshold)).argmin())
            idx2 = min(len(precision) - 1, idx)  # Ensure we don't exceed precision length
            plt.plot(recall[idx2], precision[idx2], 'o', markersize=8, 
                    label=f'Threshold = {thresholds[idx]:.2f}')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='lower left')
    plt.grid(alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved precision-recall curve to {output_path}")
    
    plt.close()
    
    if progress_tracker:
        progress_tracker.update(f"{title} plotting complete")


def plot_calibration_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                          output_path: Optional[Union[str, Path]] = None,
                          title: str = "Calibration Curve",
                          progress_tracker: Optional[ProgressTracker] = None) -> None:
    """
    Plot and save calibration curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        output_path: Path to save the plot
        title: Plot title
        progress_tracker: Optional progress tracker
    """
    logger.info(f"Plotting {title}")
    
    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
    
    # Calculate Brier score
    brier = brier_score_loss(y_true, y_pred_proba)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Plot calibration curve
    plt.plot(prob_pred, prob_true, 's-', color='darkgreen', lw=2, 
            label=f'Calibration curve (Brier score = {brier:.3f})')
    
    # Plot perfect calibration
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives (Empirical probability)')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved calibration curve to {output_path}")
    
    plt.close()
    
    if progress_tracker:
        progress_tracker.update(f"{title} plotting complete")


def save_metrics(metrics: Dict[str, Any], output_path: Union[str, Path],
                progress_tracker: Optional[ProgressTracker] = None) -> None:
    """
    Save evaluation metrics to JSON.
    
    Args:
        metrics: Dictionary of metrics
        output_path: Path to save metrics
        progress_tracker: Optional progress tracker
    """
    logger.info(f"Saving metrics to {output_path}")
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy_types(x) for x in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return convert_numpy_types(obj.tolist())
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif obj is None:
            return None
        else:
            return obj
    
    metrics_json = convert_numpy_types(metrics)
    
    # Add timestamp
    metrics_json['timestamp'] = datetime.now().isoformat()
    
    # Add model information
    metrics_json['model_info'] = {
        'model_path': str(MODEL_OUTPUT),
        'feature_count': metrics_json.get('feature_count', None),
        'data_size': metrics_json.get('data_size', None)
    }
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    logger.info(f"Metrics saved to {output_path}")
    
    if progress_tracker:
        progress_tracker.update("Metrics saving complete")


def evaluate_player_position_bias(model: xgb.Booster, test_df: pd.DataFrame, feature_cols: List[str],
                                 progress_tracker: Optional[ProgressTracker] = None) -> Dict[str, float]:
    """
    Evaluate if the model has any bias based on player position.
    
    Args:
        model: Trained XGBoost model
        test_df: Test DataFrame
        feature_cols: List of feature column names
        progress_tracker: Optional progress tracker
        
    Returns:
        Dictionary with bias metrics
    """
    logger.info("Evaluating player position bias")
    
    # Get unique match IDs
    match_ids = test_df['match_id'].unique()
    
    # Initialize counters
    same_prediction_count = 0
    different_prediction_count = 0
    
    # Create a progress bar
    pbar = tqdm(match_ids, desc="Checking player position bias")
    
    for match_id in pbar:
        # Get the two rows for this match (player1 as winner and player1 as loser)
        match_rows = test_df[test_df['match_id'] == match_id]
        
        if len(match_rows) != 2:
            continue
        
        # Get features for both scenarios
        X1 = match_rows.iloc[0][feature_cols].values.reshape(1, -1)
        X2 = match_rows.iloc[1][feature_cols].values.reshape(1, -1)
        
        # Create DMatrix objects
        dmat1 = xgb.DMatrix(X1, feature_names=feature_cols)
        dmat2 = xgb.DMatrix(X2, feature_names=feature_cols)
        
        # Get predictions
        pred1 = model.predict(dmat1)[0]
        pred2 = model.predict(dmat2)[0]
        
        # Check if predictions are consistent
        # pred1 > 0.5 should mean pred2 < 0.5 (and vice versa)
        # If both are on the same side of 0.5, we have inconsistency
        if (pred1 > 0.5 and pred2 > 0.5) or (pred1 < 0.5 and pred2 < 0.5):
            different_prediction_count += 1
        else:
            same_prediction_count += 1
    
    # Calculate consistency percentage
    total_matches = same_prediction_count + different_prediction_count
    if total_matches > 0:
        consistency_pct = (same_prediction_count / total_matches) * 100
    else:
        consistency_pct = 0
    
    logger.info(f"Player position consistency: {consistency_pct:.2f}%")
    logger.info(f"Consistent predictions: {same_prediction_count} / {total_matches}")
    logger.info(f"Inconsistent predictions: {different_prediction_count} / {total_matches}")
    
    bias_metrics = {
        'player_position_consistency_pct': consistency_pct,
        'consistent_predictions': same_prediction_count,
        'inconsistent_predictions': different_prediction_count,
        'total_matches': total_matches
    }
    
    if progress_tracker:
        progress_tracker.update("Player position bias analysis complete")
    
    return bias_metrics


def select_features_by_importance(
    importances: Dict[str, float], 
    threshold: float = 0.95,
    min_features: int = 10,
    progress_tracker: Optional[ProgressTracker] = None
) -> List[str]:
    """
    Select top features based on cumulative importance.
    
    Args:
        importances: Dictionary of feature importances {feature: importance_value}
        threshold: Cumulative importance threshold (0.0 to 1.0)
        min_features: Minimum number of features to include
        progress_tracker: Optional progress tracker
        
    Returns:
        List of selected feature names
    """
    logger.info(f"Selecting features with cumulative importance threshold {threshold}")
    
    # Sort features by importance
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate total importance
    total_importance = sum(importances.values())
    
    # Select features with cumulative importance up to threshold
    cum_importance = 0
    selected_features = []
    
    for feature, importance in sorted_features:
        selected_features.append(feature)
        cum_importance += importance / total_importance
        
        if cum_importance >= threshold and len(selected_features) >= min_features:
            break
    
    # If we have fewer than min_features, add more based on importance
    if len(selected_features) < min_features:
        remaining_features = [f for f, _ in sorted_features if f not in selected_features]
        selected_features.extend(remaining_features[:min_features - len(selected_features)])
    
    # Log selected features
    logger.info(f"Selected {len(selected_features)} features out of {len(importances)} "
                f"(cumulative importance: {cum_importance:.2f})")
    logger.info(f"Top 10 selected features: {selected_features[:10]}")
    
    if progress_tracker:
        progress_tracker.update(f"Selected {len(selected_features)} features")
    
    return selected_features


def save_pipeline(model: xgb.Booster, feature_cols: List[str], scaler: Optional[StandardScaler] = None,
                 output_path: Union[str, Path] = PIPELINE_OUTPUT,
                 progress_tracker: Optional[ProgressTracker] = None) -> None:
    """
    Save model pipeline artifacts for later use.
    With XGBoost, we don't need scaling so scaler will typically be None.
    
    Args:
        model: Trained XGBoost model
        feature_cols: Feature column names
        scaler: Optional feature scaler (not needed for XGBoost)
        output_path: Path to save the pipeline
        progress_tracker: Optional progress tracker
    """
    logger.info(f"Saving model pipeline to {output_path}")
    
    # Create pipeline dictionary
    pipeline = {
        'model': model,
        'feature_cols': feature_cols,
        'scaler': scaler,  # Will be None for XGBoost as scaling is unnecessary
        'created_at': datetime.now().isoformat(),
        'surfaces': SURFACES,
        'model_params': model.get_params() if hasattr(model, 'get_params') else None
    }
    
    # Save to file
    with open(output_path, 'wb') as f:
        pickle.dump(pipeline, f)
    
    logger.info(f"Model pipeline saved to {output_path}")
    
    if progress_tracker:
        progress_tracker.update("Model pipeline saving complete")


def plot_shap_values(
    model: xgb.Booster,
    X: np.ndarray,
    feature_names: List[str],
    output_path: Path,
    max_display: int = 20,
    progress_tracker: Optional[ProgressTracker] = None
) -> None:
    """
    Plot SHAP values for feature importance explanation.
    
    Args:
        model: Trained XGBoost model
        X: Feature matrix (usually test set)
        feature_names: List of feature names
        output_path: Path to save the plot
        max_display: Maximum number of features to display
        progress_tracker: Optional progress tracker
    """
    logger.info("Generating SHAP value plots for model explanation")
    
    try:
        # Create explainer and calculate SHAP values
        explainer = shap.TreeExplainer(model)
        dmatrix = xgb.DMatrix(X, feature_names=feature_names)
        shap_values = explainer.shap_values(X)
        
        # Create and save summary plot
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X, feature_names=feature_names, 
                         max_display=min(max_display, len(feature_names)),
                         show=False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create and save beeswarm plot
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X, feature_names=feature_names, 
                         plot_type="bar", max_display=min(max_display, len(feature_names)),
                         show=False)
        plt.tight_layout()
        plt.savefig(str(output_path).replace('.png', '_bar.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create and save dependence plots for top features
        top_features = list(zip(feature_names, np.abs(shap_values).mean(0)))[:]
        top_features = sorted(top_features, key=lambda x: x[1], reverse=True)[:5]
        
        for feature, _ in top_features:
            plt.figure(figsize=(10, 7))
            feature_idx = feature_names.index(feature)
            shap.dependence_plot(feature_idx, shap_values, X, feature_names=feature_names, 
                                show=False)
            plt.tight_layout()
            plt.savefig(str(output_path).replace('.png', f'_dependence_{feature}.png'), 
                      dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"SHAP plots saved to {output_path} and related files")
        
        if progress_tracker:
            progress_tracker.update("SHAP value plots created")
    
    except Exception as e:
        logger.error(f"Error generating SHAP plots: {e}")
        logger.exception("Exception details:")
        if progress_tracker:
            progress_tracker.update("Failed to create SHAP plots")
    
    # Create a waterfall plot for a sample prediction
    try:
        if X.shape[0] > 0:
            plt.figure(figsize=(12, 8))
            sample_idx = 0  # Use first sample
            expected_value = explainer.expected_value
            sample_shap_values = explainer.shap_values(X[sample_idx:sample_idx+1])[0]
            
            shap.plots._waterfall.waterfall_legacy(
                expected_value, 
                sample_shap_values, 
                feature_names=feature_names,
                max_display=min(max_display, len(feature_names)),
                show=False
            )
            plt.tight_layout()
            plt.savefig(str(output_path).replace('.png', '_waterfall.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Waterfall plot saved to {str(output_path).replace('.png', '_waterfall.png')}")
    except Exception as e:
        logger.error(f"Error generating waterfall plot: {e}")
        logger.exception("Exception details:")


def main() -> None:
    """Main function to execute the entire training workflow."""
    start_time = time.time()
    
    # Configure directories
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    
    # Set up progress tracking
    total_steps = 14
    progress_tracker = ProgressTracker("Training tennis prediction model", total_steps)
    
    try:
        # Step 1: Load data
        logger.info(f"Step 1/{total_steps}: Loading data...")
        if os.path.exists(INPUT_FILE):
            df = load_data_from_file(INPUT_FILE, progress_tracker)
        else:
            df = load_data_from_database(progress_tracker=progress_tracker)
            # Save to file for future use
            df.to_csv(INPUT_FILE, index=False)
            logger.info(f"Saved data to {INPUT_FILE}")
        
        # Step 2: Create time-based train/val/test split
        logger.info(f"Step 2/{total_steps}: Creating time-based split...")
        train_df, val_df, test_df = create_time_based_train_val_test_split(
            df, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, progress_tracker
        )
        
        # Step 3: Get feature columns
        logger.info(f"Step 3/{total_steps}: Identifying feature columns...")
        feature_cols = get_feature_columns(df, progress_tracker)
        
        # Step 4: Prepare features and labels - no scaling for XGBoost
        logger.info(f"Step 4/{total_steps}: Preparing features and labels...")
        (X_train, X_val, X_test), (y_train, y_val, y_test), categorical_indices = prepare_features(
            train_df, val_df, test_df, feature_cols, progress_tracker
        )
        
        # Step 5: Tune hyperparameters
        logger.info(f"Step 5/{total_steps}: Tuning hyperparameters...")
        best_params = tune_hyperparameters(
            X_train, y_train, X_val, y_val, feature_cols, categorical_indices, progress_tracker
        )
        
        # Step 6: Train model with best parameters
        logger.info(f"Step 6/{total_steps}: Training model with tuned hyperparameters...")
        model, evals_result = train_model(
            X_train, y_train, X_val, y_val, feature_cols, categorical_indices, best_params, 50, progress_tracker
        )
        
        # Step 7: Extract feature importance
        logger.info(f"Step 7/{total_steps}: Extracting feature importance...")
        importance_scores = model.get_score(importance_type='gain')
        importances = {feature: score for feature, score in importance_scores.items() if feature in feature_cols}
        importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
        
        # Step 8: Select most important features
        logger.info(f"Step 8/{total_steps}: Selecting top features...")
        selected_features = select_features_by_importance(importances, threshold=0.95, min_features=20)
        
        # Step 9: Retrain with selected features
        logger.info(f"Step 9/{total_steps}: Retraining with selected features...")
        X_train_selected = X_train[:, [feature_cols.index(f) for f in selected_features]]
        X_val_selected = X_val[:, [feature_cols.index(f) for f in selected_features]]
        X_test_selected = X_test[:, [feature_cols.index(f) for f in selected_features]]
        
        # Update categorical indices for selected features
        selected_categorical_indices = []
        for i, feature in enumerate(selected_features):
            if feature_cols.index(feature) in categorical_indices:
                selected_categorical_indices.append(i)
        
        # Retrain
        final_model, final_evals_result = train_model(
            X_train_selected, y_train, X_val_selected, y_val, selected_features, 
            selected_categorical_indices, best_params, 50, progress_tracker
        )
        
        # Step 10: Evaluate model
        logger.info(f"Step 10/{total_steps}: Evaluating model...")
        metrics = evaluate_model(
            final_model, X_test_selected, y_test, selected_features, 
            progress_tracker=progress_tracker,
            surfaces=train_df['surface'].unique()
        )
        
        # Step 11: Save metrics
        logger.info(f"Step 11/{total_steps}: Saving evaluation metrics...")
        save_metrics(metrics, METRICS_OUTPUT, progress_tracker)
        
        # Step 12: Save hyperparameters
        logger.info(f"Step 12/{total_steps}: Saving hyperparameters...")
        with open(HYPERPARAMS_OUTPUT, "w") as f:
            json.dump(best_params, f, indent=4)
        progress_tracker.update("Hyperparameters saved")
        
        # Step 13: Plot feature importance
        logger.info(f"Step 13/{total_steps}: Plotting feature importance...")
        plot_feature_importance(
            importances, min(20, len(selected_features)), 
            PLOTS_DIR / "feature_importance.png", progress_tracker
        )
        
        # Step 14: Save model and pipeline
        logger.info(f"Step 14/{total_steps}: Saving model and pipeline...")
        final_model.save_model(str(MODEL_OUTPUT))
        logger.info(f"Model saved to {MODEL_OUTPUT}")
        
        # Save full pipeline
        save_pipeline(final_model, selected_features, None, PIPELINE_OUTPUT, progress_tracker)
        
        # Print final message
        total_time = time.time() - start_time
        logger.info(f"Model training completed in {total_time:.2f} seconds")
        logger.info(f"Model accuracy on test set: {metrics['accuracy']:.4f}")
        
        # Plot SHAP values for feature explanation
        plot_shap_values(final_model, X_test_selected, selected_features, PLOTS_DIR / "shap_summary.png")
        
    except Exception as e:
        logger.error(f"Error in training process: {str(e)}")
        logger.exception("Exception details:")
        raise


if __name__ == "__main__":
    main() 