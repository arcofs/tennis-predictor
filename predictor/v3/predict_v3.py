import os
import time
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from tqdm import tqdm
from dotenv import load_dotenv
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project directories
PROJECT_ROOT = Path(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
DATA_DIR = PROJECT_ROOT / "data"
V3_DATA_DIR = DATA_DIR / "v3"
OUTPUT_DIR = PROJECT_ROOT / "predictor" / "output" / "v3"
MODELS_DIR = PROJECT_ROOT / "models" / "v3"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# File paths
MODEL_PATH = MODELS_DIR / "tennis_model_v3.json"
FEATURES_PATH = V3_DATA_DIR / "features_v3.csv"
PREDICTIONS_OUTPUT = OUTPUT_DIR / "predictions_v3.csv"

# Constants
SURFACES = ['hard', 'clay', 'grass', 'carpet']

# Database constants
DB_BATCH_SIZE = 10000  # Number of records to fetch in each database batch
DB_TIMEOUT_SECONDS = 30  # Database query timeout in seconds

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


def get_database_connection() -> psycopg2.extensions.connection:
    """
    Create a database connection using environment variables.
    
    Returns:
        Database connection
    """
    # Load environment variables
    load_dotenv()
    
    # Get database URL from environment
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL not found in environment variables")
    
    try:
        # Convert postgres:// to postgresql:// if needed (psycopg2 requirement)
        if database_url.startswith('postgres://'):
            database_url = database_url.replace('postgres://', 'postgresql://', 1)
            
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


class ProgressTracker:
    """
    Class to track and log progress during prediction.
    """
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        
    def update(self, description: str = None):
        """Update progress and log status."""
        self.current_step += 1
        elapsed = time.time() - self.start_time
        
        if description:
            message = f"{description}: "
        else:
            message = ""
            
        if self.current_step < self.total_steps:
            est_remaining = (elapsed / self.current_step) * (self.total_steps - self.current_step)
            logger.info(f"{message}Progress: {self.percent_complete}% complete. Est. remaining time: {est_remaining:.1f}s")
        else:
            logger.info(f"{message}Progress: 100% complete. Total time: {elapsed:.1f}s")
    
    @property
    def percent_complete(self) -> int:
        """Calculate percentage of completion."""
        return int((self.current_step / self.total_steps) * 100)


def load_model(model_path: Union[str, Path], 
              progress_tracker: Optional[ProgressTracker] = None) -> xgb.Booster:
    """
    Load the trained XGBoost model.
    
    Args:
        model_path: Path to the saved model
        progress_tracker: Optional progress tracker
        
    Returns:
        Loaded XGBoost model
    """
    logger.info(f"Loading model from {model_path}")
    
    try:
        model = xgb.Booster()
        model.load_model(str(model_path))
        logger.info("Model loaded successfully")
        
        if progress_tracker:
            progress_tracker.update("Model loading complete")
        
        return model
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def load_test_data_from_database(training_cutoff_date: Optional[str] = None, 
                               limit: Optional[int] = None,
                               progress_tracker: Optional[ProgressTracker] = None) -> pd.DataFrame:
    """
    Load test data from the PostgreSQL database, ensuring we only get matches
    that occurred after the training cutoff date to prevent data leakage.
    
    Args:
        training_cutoff_date: ISO format date string (YYYY-MM-DD) after which to get test data
        limit: Optional limit on number of rows to fetch
        progress_tracker: Optional progress tracker
        
    Returns:
        DataFrame with test data
    """
    logger.info("Loading test data from database")
    
    try:
        # Connect to database
        conn = get_database_connection()
        
        # Define query to get model training date
        if training_cutoff_date is None:
            try:
                # Try to get the training cutoff date from model metrics
                metrics_path = OUTPUT_DIR / "model_metrics_v3.json"
                if os.path.exists(metrics_path):
                    import json
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                    
                    # Get the exact date range from the test set used during training
                    if 'training_info' in metrics and 'data_date_range' in metrics['training_info']:
                        # Calculate the start date for test set based on splits
                        # Assuming TRAIN_SPLIT = 0.7, VAL_SPLIT = 0.15, TEST_SPLIT = 0.15
                        data_start = metrics['training_info']['data_date_range']['start'].split('T')[0]
                        data_end = metrics['training_info']['data_date_range']['end'].split('T')[0]
                        logger.info(f"Data date range from metrics: {data_start} to {data_end}")
                        
                        # Get the test size - default to 0.15 if not specified
                        test_size = 0.15
                        val_size = 0.15
                        
                        # Load the total dataset to calculate dates proportionally
                        count_query = """
                        SELECT COUNT(*) as total_count,
                               MIN(tournament_date) as min_date,
                               MAX(tournament_date) as max_date
                        FROM match_features
                        WHERE tournament_date BETWEEN '{0}' AND '{1}'
                        """.format(data_start, data_end)
                        
                        df_count = pd.read_sql(count_query, conn)
                        total_count = df_count.iloc[0]['total_count']
                        min_date = df_count.iloc[0]['min_date']
                        max_date = df_count.iloc[0]['max_date']
                        
                        # Calculate the test split start date
                        # This query finds the date that corresponds to the start of the test set
                        # by ordering all matches and finding the date at the position where test set begins
                        test_start_query = """
                        WITH ordered_matches AS (
                            SELECT tournament_date,
                                   ROW_NUMBER() OVER (ORDER BY tournament_date ASC) as rn
                            FROM match_features
                            WHERE tournament_date BETWEEN '{0}' AND '{1}'
                        )
                        SELECT tournament_date
                        FROM ordered_matches
                        WHERE rn >= FLOOR({2} * {3})
                        ORDER BY rn ASC
                        LIMIT 1
                        """.format(data_start, data_end, total_count, 1 - test_size)
                        
                        df_test_start = pd.read_sql(test_start_query, conn)
                        
                        if not df_test_start.empty:
                            test_start_date = df_test_start.iloc[0]['tournament_date']
                            if isinstance(test_start_date, str):
                                test_start_date = test_start_date.split('T')[0]
                            elif hasattr(test_start_date, 'date'):
                                test_start_date = test_start_date.date().isoformat()
                                
                            logger.info(f"Using test start date: {test_start_date}")
                            training_cutoff_date = test_start_date
                        else:
                            logger.warning("Could not determine test start date from query")
                            training_cutoff_date = data_end
                    else:
                        # Default to end date if no specific info available
                        training_cutoff_date = metrics['training_info']['data_date_range']['end'].split('T')[0]
                        logger.info(f"Using training cutoff date from metrics: {training_cutoff_date}")
            except Exception as e:
                logger.warning(f"Error loading training cutoff date from metrics: {e}")
                logger.info("Using recent data for testing")
        
        # Define the base query
        if training_cutoff_date:
            base_query = f"""
            SELECT *
            FROM match_features
            WHERE tournament_date >= '{training_cutoff_date}'
            ORDER BY tournament_date ASC
            """
            logger.info(f"Getting test data from {training_cutoff_date}")
        else:
            # If no cutoff date, get the most recent 15% of matches (matching TEST_SPLIT in training)
            base_query = """
            WITH ranked_matches AS (
                SELECT 
                    *, 
                    NTILE(20) OVER (ORDER BY tournament_date ASC) AS date_quintile
                FROM match_features
            )
            SELECT * FROM ranked_matches
            WHERE date_quintile >= 18
            ORDER BY tournament_date ASC
            """
            logger.info("Getting most recent 15% of matches as test data (matching TEST_SPLIT in training)")
        
        # Use batched loading to handle large datasets efficiently
        offset = 0
        dataframes = []
        total_rows = 0
        
        if limit:
            total_to_fetch = limit
        else:
            # Get total count first
            with conn.cursor() as cursor:
                if training_cutoff_date:
                    cursor.execute(f"SELECT COUNT(*) FROM match_features WHERE tournament_date >= '{training_cutoff_date}'")
                else:
                    cursor.execute("SELECT COUNT(*) FROM match_features")
                    total_count = cursor.fetchone()[0]
                    # We want the most recent 15% if no cutoff date
                    total_to_fetch = max(1, int(total_count * 0.15))
        
        logger.info(f"Fetching up to {total_to_fetch} rows from database")
        pbar = tqdm(total=total_to_fetch, desc="Loading test data from database")
        
        # Define the overall limit
        remaining_to_fetch = limit if limit else total_to_fetch
        
        while remaining_to_fetch > 0:
            # Define batch size (either DB_BATCH_SIZE or remaining limit)
            batch_size = min(DB_BATCH_SIZE, remaining_to_fetch)
            
            # Define batch query with proper LIMIT and OFFSET
            query = f"""
            {base_query}
            LIMIT {batch_size} OFFSET {offset}
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
            
            # Update remaining to fetch and offset
            remaining_to_fetch -= rows_fetched
            offset += rows_fetched
            
            # If we've reached the limit, we're done
            if limit and total_rows >= limit:
                break
        
        pbar.close()
        
        # Combine all batches
        if dataframes:
            df = pd.concat(dataframes, ignore_index=True)
            
            # Convert date columns to datetime
            if 'tournament_date' in df.columns:
                df['tournament_date'] = pd.to_datetime(df['tournament_date'])
            
            # Sort by date
            df = df.sort_values(by='tournament_date').reset_index(drop=True)
            
            # Ensure result is an integer (1 for win, 0 for loss)
            if 'result' in df.columns:
                df['result'] = df['result'].astype(int)
            
            logger.info(f"Loaded {len(df)} test matches spanning from "
                      f"{df['tournament_date'].min().date()} to {df['tournament_date'].max().date()}")
            
            if progress_tracker:
                progress_tracker.update("Test data loading complete")
            
            return df
        else:
            logger.warning("No test data retrieved from database")
            return pd.DataFrame()
    
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        raise
    finally:
        conn.close()


def load_test_data(features_path: Union[str, Path], test_size: float = 0.2, 
                  progress_tracker: Optional[ProgressTracker] = None) -> pd.DataFrame:
    """
    Load test data from features file. 
    This is kept for backwards compatibility.
    
    Args:
        features_path: Path to the features CSV file
        test_size: Proportion of data to use for testing
        progress_tracker: Optional progress tracker
        
    Returns:
        DataFrame with test data
    """
    logger.info(f"Loading test data from {features_path}")
    
    try:
        # Load features
        df = pd.read_csv(features_path)
        
        # Convert date columns to datetime
        if 'tourney_date' in df.columns:
            df['tourney_date'] = pd.to_datetime(df['tourney_date'])
            
        # Sort by date
        df = df.sort_values(by='tourney_date').reset_index(drop=True)
        
        # Calculate split index - always use the most recent matches for testing
        # to ensure we're evaluating on future matches not seen during training
        split_idx = int(len(df) * (1 - test_size))
        
        # Get test data
        test_df = df.iloc[split_idx:].copy()
        
        # Check if we have enough test data
        if len(test_df) < 100:
            logger.warning(f"Small test set: only {len(test_df)} matches. Consider using a larger test_size.")
        
        # Calculate date range
        if 'tourney_date' in test_df.columns:
            start_date = test_df['tourney_date'].min().strftime('%Y-%m-%d')
            end_date = test_df['tourney_date'].max().strftime('%Y-%m-%d')
            logger.info(f"Test data date range: {start_date} to {end_date}")
        
        logger.info(f"Loaded {len(test_df)} test matches from a total of {len(df)} matches")
        logger.info(f"Using the most recent {test_size*100:.1f}% of matches for testing")
        
        if progress_tracker:
            progress_tracker.update("Test data loading complete")
        
        return test_df
    
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        raise


def get_feature_columns(df: pd.DataFrame, 
                       progress_tracker: Optional[ProgressTracker] = None) -> List[str]:
    """
    Get the list of feature columns for prediction.
    
    Args:
        df: DataFrame with match features
        progress_tracker: Optional progress tracker
        
    Returns:
        List of feature column names
    """
    logger.info("Identifying feature columns")
    
    # Columns to exclude from features
    exclude_cols = ['id', 'match_id', 'tournament_date', 'player1_id', 'player2_id', 'surface', 'result', 'created_at', 'updated_at']
    
    # Get all numeric columns except excluded ones
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Generate list of surface-specific serve and return features
    surface_specific_features = []
    for feature in SERVE_RETURN_FEATURES:
        base_feature = feature.split('_diff')[0]
        # Try both lowercase and uppercase surface names
        for surface in SURFACES:
            surface_lower = surface.lower()
            # Check for lowercase version (from database)
            surface_feature_lower = f"{base_feature}_{surface_lower}_diff"
            if surface_feature_lower in df.columns:
                surface_specific_features.append(surface_feature_lower)
            
            # Also add uppercase version (model might use this)
            surface_feature_upper = f"{base_feature}_{surface}_diff"
            if surface_feature_upper in df.columns and surface_feature_upper != surface_feature_lower:
                surface_specific_features.append(surface_feature_upper)
    
    # Generate list of surface-specific win rate features
    for surface in SURFACES:
        surface_lower = surface.lower()
        
        # Try both lowercase and uppercase versions
        for surface_format in [surface_lower, surface]:
            # Most recent win rate on specific surface
            surface_win_rate = f"win_rate_{surface_format}_5_diff"
            if surface_win_rate in df.columns and surface_win_rate not in feature_cols:
                surface_specific_features.append(surface_win_rate)
            
            # Overall win rate on specific surface
            surface_overall_win_rate = f"win_rate_{surface_format}_overall_diff"
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
    
    # Generate raw player surface-specific features
    for surface in SURFACES:
        surface_lower = surface.lower()
        
        # Try both lowercase and uppercase versions for each player
        for surface_format in [surface_lower, surface]:
            # Win rates
            for player_prefix in ['player1_', 'player2_']:
                # Recent win rate
                feat = f"{player_prefix}win_rate_{surface_format}_5"
                if feat in df.columns and feat not in raw_player_features:
                    raw_player_features.append(feat)
                
                # Overall win rate
                feat = f"{player_prefix}win_rate_{surface_format}_overall"
                if feat in df.columns and feat not in raw_player_features:
                    raw_player_features.append(feat)
            
            # Serve and return stats
            for base_feature in [feat.split('_diff')[0] for feat in SERVE_RETURN_FEATURES]:
                for player_prefix in ['player1_', 'player2_']:
                    feat = f"{player_prefix}{base_feature}_{surface_format}"
                    if feat in df.columns and feat not in raw_player_features:
                        raw_player_features.append(feat)
    
    # Ensure all surface-specific and raw player features are included
    for feature in surface_specific_features + raw_player_features:
        if feature not in feature_cols:
            feature_cols.append(feature)
    
    logger.info(f"Selected {len(feature_cols)} feature columns")
    
    if progress_tracker:
        progress_tracker.update("Feature selection complete")
    
    return feature_cols


def make_predictions(model: xgb.Booster, test_df: pd.DataFrame, feature_cols: List[str],
                    progress_tracker: Optional[ProgressTracker] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Make predictions on test data.
    
    Args:
        model: Trained XGBoost model
        test_df: Test DataFrame
        feature_cols: List of feature columns
        progress_tracker: Optional progress tracker
        
    Returns:
        Tuple of (DataFrame with predictions, metrics dictionary)
    """
    logger.info("Making predictions on test data")
    
    try:
        # Create a copy of test data
        df_pred = test_df.copy()
        
        # Get model feature names
        model_features = model.feature_names
        
        # Create a mapping between database columns and model features
        # This is needed in case column names differ slightly (e.g., case sensitivity)
        feature_mapping = {}
        missing_features = []
        
        for model_feat in model_features:
            found = False
            
            # Direct match - feature exists exactly as named
            if model_feat in df_pred.columns:
                feature_mapping[model_feat] = model_feat
                found = True
            else:
                # Try case-insensitive match
                model_feat_lower = model_feat.lower()
                for db_col in df_pred.columns:
                    if db_col.lower() == model_feat_lower:
                        feature_mapping[model_feat] = db_col
                        found = True
                        logger.info(f"Mapped model feature '{model_feat}' to database column '{db_col}'")
                        break
            
            if not found:
                missing_features.append(model_feat)
        
        # Log the mapping and missing features
        logger.info(f"Successfully mapped {len(feature_mapping)} features between model and database")
        
        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features expected by model")
            
            # Add missing features as NaN values
            for feat in missing_features:
                if feat not in df_pred.columns:
                    logger.info(f"Adding missing feature '{feat}' as NaN (will be handled by XGBoost)")
                    df_pred[feat] = np.nan
        
        # Create a new DataFrame with the exact column names the model expects
        X_test_mapped = pd.DataFrame()
        
        # For existing mappings, copy data from df_pred using the correct database column name
        for model_feat, db_feat in feature_mapping.items():
            X_test_mapped[model_feat] = df_pred[db_feat]
        
        # For missing features, copy from the added NaN columns
        for feat in missing_features:
            if feat in df_pred.columns:
                X_test_mapped[feat] = df_pred[feat]
        
        # Ensure we have all features the model expects
        assert set(X_test_mapped.columns) == set(model_features), "Feature mismatch after mapping"
        
        # Log feature information
        logger.info(f"Using {len(model_features)} features for prediction")
        
        # Check for features with all missing values
        null_counts = X_test_mapped.isnull().sum()
        complete_null_features = null_counts[null_counts == len(X_test_mapped)].index.tolist()
        if complete_null_features:
            logger.warning(f"Features with all missing values: {complete_null_features[:5]}...")
        
        # Extract target
        y_test = df_pred['result'].values
        
        # Create DMatrix with feature names and specify missing value handling
        dtest = xgb.DMatrix(X_test_mapped.values, label=y_test, feature_names=model_features, missing=np.nan)
        
        # Make predictions
        y_pred_proba = model.predict(dtest)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Add predictions to DataFrame
        df_pred['predicted_proba'] = y_pred_proba
        df_pred['predicted'] = y_pred
        df_pred['correct'] = (df_pred['predicted'] == df_pred['result']).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        logger.info(f"Overall accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        
        # Calculate surface-specific metrics
        surface_metrics = {}
        
        for surface in SURFACES:
            surface_idx = df_pred['surface'] == surface
            
            if sum(surface_idx) > 0:
                surface_acc = accuracy_score(
                    df_pred.loc[surface_idx, 'result'], 
                    df_pred.loc[surface_idx, 'predicted']
                )
                
                surface_metrics[surface] = {
                    'accuracy': surface_acc,
                    'count': sum(surface_idx)
                }
                
                logger.info(f"{surface} surface accuracy: {surface_acc:.4f} (n={sum(surface_idx)})")
        
        # Calculate confidence-based metrics
        confidence_bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        confidence_metrics = []
        
        for i in range(len(confidence_bins) - 1):
            lower = confidence_bins[i]
            upper = confidence_bins[i + 1]
            
            # Filter predictions in this confidence range
            mask = (df_pred['predicted_proba'] >= lower) & (df_pred['predicted_proba'] < upper) | \
                  (df_pred['predicted_proba'] <= (1 - lower)) & (df_pred['predicted_proba'] > (1 - upper))
            
            bin_count = sum(mask)
            
            if bin_count > 0:
                bin_acc = accuracy_score(
                    df_pred.loc[mask, 'result'], 
                    df_pred.loc[mask, 'predicted']
                )
                
                confidence_metrics.append({
                    'confidence_range': f"{lower:.1f}-{upper:.1f}",
                    'accuracy': bin_acc,
                    'count': bin_count,
                    'percentage': bin_count / len(df_pred) * 100
                })
                
                logger.info(f"Confidence {lower:.1f}-{upper:.1f} accuracy: {bin_acc:.4f} (n={bin_count}, {bin_count / len(df_pred) * 100:.1f}%)")
        
        # Compile metrics
        metrics = {
            'overall': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'count': len(test_df)
            },
            'by_surface': surface_metrics,
            'by_confidence': confidence_metrics
        }
        
        if progress_tracker:
            progress_tracker.update("Prediction complete")
        
        return df_pred, metrics
    
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise


def calculate_match_level_metrics(predictions_df: pd.DataFrame,
                                progress_tracker: Optional[ProgressTracker] = None) -> Dict[str, Any]:
    """
    Calculate match-level metrics for player position symmetry.
    
    Args:
        predictions_df: DataFrame with predictions
        progress_tracker: Optional progress tracker
        
    Returns:
        Dictionary with match-level metrics
    """
    logger.info("Calculating match-level metrics")
    
    try:
        # Get unique match IDs
        match_ids = predictions_df['match_id'].unique()
        
        # Initialize counters
        consistent_predictions = 0
        inconsistent_predictions = 0
        
        # Initialize match-level accuracy counter
        match_correct = 0
        total_matches = 0
        
        # Create progress bar
        pbar = tqdm(match_ids, desc="Analyzing match predictions")
        
        for match_id in pbar:
            # Get predictions for this match
            match_rows = predictions_df[predictions_df['match_id'] == match_id]
            
            if len(match_rows) != 2:
                continue
            
            # Check if predictions are consistent
            p1_pred = match_rows.iloc[0]['predicted']
            p2_pred = match_rows.iloc[1]['predicted']
            
            # p1_pred and p2_pred should be opposite for consistent predictions
            if (p1_pred == 1 and p2_pred == 0) or (p1_pred == 0 and p2_pred == 1):
                consistent_predictions += 1
            else:
                inconsistent_predictions += 1
            
            # Check if either prediction is correct (only need to check one)
            if match_rows.iloc[0]['correct'] == 1 or match_rows.iloc[1]['correct'] == 1:
                match_correct += 1
            
            total_matches += 1
        
        # Calculate match-level accuracy
        match_level_accuracy = match_correct / total_matches if total_matches > 0 else 0
        
        # Calculate consistency percentage
        consistency_pct = consistent_predictions / total_matches * 100 if total_matches > 0 else 0
        
        logger.info(f"Match-level accuracy: {match_level_accuracy:.4f}")
        logger.info(f"Prediction consistency: {consistency_pct:.2f}%")
        logger.info(f"Consistent predictions: {consistent_predictions} / {total_matches}")
        logger.info(f"Inconsistent predictions: {inconsistent_predictions} / {total_matches}")
        
        # Compile metrics
        match_metrics = {
            'match_level_accuracy': match_level_accuracy,
            'prediction_consistency_pct': consistency_pct,
            'consistent_predictions': consistent_predictions,
            'inconsistent_predictions': inconsistent_predictions,
            'total_matches': total_matches
        }
        
        if progress_tracker:
            progress_tracker.update("Match-level metrics calculation complete")
        
        return match_metrics
    
    except Exception as e:
        logger.error(f"Error calculating match-level metrics: {e}")
        raise


def plot_results(predictions_df: pd.DataFrame, metrics: Dict[str, Any], output_dir: Union[str, Path],
                progress_tracker: Optional[ProgressTracker] = None) -> None:
    """
    Plot prediction results.
    
    Args:
        predictions_df: DataFrame with predictions
        metrics: Dictionary with metrics
        output_dir: Directory to save plots
        progress_tracker: Optional progress tracker
    """
    logger.info("Plotting results")
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(predictions_df['result'], predictions_df['predicted'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Player 2 Wins', 'Player 1 Wins'],
                   yticklabels=['Player 2 Wins', 'Player 1 Wins'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(output_dir / "prediction_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(predictions_df['result'], predictions_df['predicted_proba'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(output_dir / "roc_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot accuracy by confidence
        if 'by_confidence' in metrics and metrics['by_confidence']:
            plt.figure(figsize=(10, 6))
            conf_ranges = [item['confidence_range'] for item in metrics['by_confidence']]
            accuracies = [item['accuracy'] for item in metrics['by_confidence']]
            counts = [item['count'] for item in metrics['by_confidence']]
            
            # Plot bar chart
            bars = plt.bar(conf_ranges, accuracies, color='skyblue')
            
            # Add count labels
            for i, bar in enumerate(bars):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                      f"n={counts[i]}", ha='center')
            
            plt.axhline(y=metrics['overall']['accuracy'], color='r', linestyle='--', 
                      label=f"Overall Accuracy: {metrics['overall']['accuracy']:.3f}")
            plt.xlabel('Confidence Range')
            plt.ylabel('Accuracy')
            plt.title('Prediction Accuracy by Confidence Range')
            plt.ylim(0, 1.1)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / "accuracy_by_confidence.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot accuracy by surface
        if 'by_surface' in metrics and metrics['by_surface']:
            plt.figure(figsize=(10, 6))
            surfaces = list(metrics['by_surface'].keys())
            surface_accs = [metrics['by_surface'][s]['accuracy'] for s in surfaces]
            surface_counts = [metrics['by_surface'][s]['count'] for s in surfaces]
            
            # Plot bar chart
            bars = plt.bar(surfaces, surface_accs, color='lightgreen')
            
            # Add count labels
            for i, bar in enumerate(bars):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                      f"n={surface_counts[i]}", ha='center')
            
            plt.axhline(y=metrics['overall']['accuracy'], color='r', linestyle='--',
                      label=f"Overall Accuracy: {metrics['overall']['accuracy']:.3f}")
            plt.xlabel('Surface')
            plt.ylabel('Accuracy')
            plt.title('Prediction Accuracy by Surface')
            plt.ylim(0, 1.1)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / "accuracy_by_surface.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot prediction distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(predictions_df['predicted_proba'], bins=20, kde=True)
        plt.xlabel('Prediction Probability')
        plt.ylabel('Count')
        plt.title('Distribution of Prediction Probabilities')
        plt.savefig(output_dir / "prediction_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved plots to {output_dir}")
        
        if progress_tracker:
            progress_tracker.update("Results plotting complete")
    
    except Exception as e:
        logger.error(f"Error plotting results: {e}")


def save_results(predictions_df: pd.DataFrame, metrics: Dict[str, Any], 
                output_path: Union[str, Path], metrics_path: Union[str, Path],
                progress_tracker: Optional[ProgressTracker] = None) -> None:
    """
    Save prediction results.
    
    Args:
        predictions_df: DataFrame with predictions
        metrics: Dictionary with metrics
        output_path: Path to save predictions
        metrics_path: Path to save metrics
        progress_tracker: Optional progress tracker
    """
    logger.info("Saving results")
    
    try:
        # Save predictions
        predictions_df.to_csv(output_path, index=False)
        logger.info(f"Saved predictions to {output_path}")
        
        # Save metrics
        import json
        
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
            else:
                return obj
        
        metrics_json = convert_numpy_types(metrics)
        
        # Add timestamp
        metrics_json['timestamp'] = datetime.now().isoformat()
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        
        logger.info(f"Saved metrics to {metrics_path}")
        
        if progress_tracker:
            progress_tracker.update("Results saving complete")
    
    except Exception as e:
        logger.error(f"Error saving results: {e}")


def predict_match(model: xgb.Booster, player1_id: str, player2_id: str, 
                 surface: str, player_stats: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """
    Predict the outcome of a single match.
    
    Args:
        model: Trained XGBoost model
        player1_id: ID of player 1
        player2_id: ID of player 2
        surface: Match surface
        player_stats: Dictionary with player statistics
        
    Returns:
        Dictionary with prediction results
    """
    logger.info(f"Predicting match between player {player1_id} and player {player2_id} on {surface}")
    
    try:
        # Check if players exist in stats
        if player1_id not in player_stats:
            raise ValueError(f"Player {player1_id} not found in player stats")
        
        if player2_id not in player_stats:
            raise ValueError(f"Player {player2_id} not found in player stats")
        
        # Get player stats
        p1_stats = player_stats[player1_id]
        p2_stats = player_stats[player2_id]
        
        # Create feature dictionary
        features = {}
        
        # Add Elo difference if available
        if 'elo' in p1_stats and 'elo' in p2_stats:
            features['player_elo_diff'] = p1_stats['elo'] - p2_stats['elo']
        else:
            # Leave as NaN instead of 0.0
            features['player_elo_diff'] = np.nan
        
        # Add win rate differences
        for stat in RAW_PLAYER_FEATURES:
            if stat in p1_stats and stat in p2_stats:
                features[f"{stat}_diff"] = p1_stats[stat] - p2_stats[stat]
                
                # Add raw player features
                features[f"player1_{stat}"] = p1_stats[stat]
                features[f"player2_{stat}"] = p2_stats[stat]
            else:
                # Use NaN for missing stats
                features[f"{stat}_diff"] = np.nan
                if stat not in p1_stats:
                    features[f"player1_{stat}"] = np.nan
                if stat not in p2_stats:
                    features[f"player2_{stat}"] = np.nan
        
        # Surface name handling - ensure we use lowercase consistently
        surface_lower = surface.lower()
        
        # Win rate features
        for timeframe in ["5", "overall"]:
            # Define feature name for database feature (always lowercase)
            db_feature = f"win_rate_{surface_lower}_{timeframe}"
            
            # Define feature names in model format
            model_feature = f"win_rate_{surface_lower}_{timeframe}_diff"
            p1_model_feature = f"player1_win_rate_{surface_lower}_{timeframe}"
            p2_model_feature = f"player2_win_rate_{surface_lower}_{timeframe}"
            
            if db_feature in p1_stats and db_feature in p2_stats:
                features[model_feature] = p1_stats[db_feature] - p2_stats[db_feature]
                features[p1_model_feature] = p1_stats[db_feature]
                features[p2_model_feature] = p2_stats[db_feature]
            else:
                # Use NaN for missing stats
                features[model_feature] = np.nan
                features[p1_model_feature] = np.nan if db_feature not in p1_stats else p1_stats[db_feature]
                features[p2_model_feature] = np.nan if db_feature not in p2_stats else p2_stats[db_feature]
        
        # Add serve and return stats
        for stat in SERVE_RETURN_FEATURES:
            base_stat = stat.split('_diff')[0]
            
            if base_stat in p1_stats and base_stat in p2_stats:
                features[stat] = p1_stats[base_stat] - p2_stats[base_stat]
                
                # Add raw player serve/return stats
                features[f"player1_{base_stat}"] = p1_stats[base_stat]
                features[f"player2_{base_stat}"] = p2_stats[base_stat]
            else:
                # Use NaN for missing stats
                features[stat] = np.nan
                if base_stat not in p1_stats:
                    features[f"player1_{base_stat}"] = np.nan
                if base_stat not in p2_stats:
                    features[f"player2_{base_stat}"] = np.nan
            
            # Add surface-specific serve/return stats
            # Database feature name (always lowercase)
            db_surface_stat = f"{base_stat}_{surface_lower}"
            
            # Create feature names as model might expect them
            model_surface_stat = f"{base_stat}_{surface_lower}_diff"
            p1_model_surface_stat = f"player1_{base_stat}_{surface_lower}"
            p2_model_surface_stat = f"player2_{base_stat}_{surface_lower}"
            
            if db_surface_stat in p1_stats and db_surface_stat in p2_stats:
                features[model_surface_stat] = p1_stats[db_surface_stat] - p2_stats[db_surface_stat]
                features[p1_model_surface_stat] = p1_stats[db_surface_stat]
                features[p2_model_surface_stat] = p2_stats[db_surface_stat]
            else:
                # Use NaN for missing stats
                features[model_surface_stat] = np.nan
                features[p1_model_surface_stat] = np.nan if db_surface_stat not in p1_stats else p1_stats[db_surface_stat]
                features[p2_model_surface_stat] = np.nan if db_surface_stat not in p2_stats else p2_stats[db_surface_stat]
        
        # Add surface - use lowercase for consistency
        features['surface'] = surface_lower
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Get feature columns that match the model's features
        feature_names = model.feature_names
        
        # Map feature names to ensure we have exact columns the model expects
        X_test_mapped = pd.DataFrame()
        missing_features = []
        
        for model_feat in feature_names:
            if model_feat in df.columns:
                # Direct match
                X_test_mapped[model_feat] = df[model_feat]
            else:
                # Check for case-insensitive matches
                model_feat_lower = model_feat.lower()
                found = False
                for col in df.columns:
                    if col.lower() == model_feat_lower:
                        X_test_mapped[model_feat] = df[col]
                        found = True
                        break
                
                if not found:
                    missing_features.append(model_feat)
        
        # Add any missing features as NaN
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            
            for feat in missing_features:
                X_test_mapped[feat] = np.nan
                logger.info(f"Adding missing feature '{feat}' as NaN (will be handled by XGBoost)")
        
        # Create DMatrix with missing value handling
        dmat = xgb.DMatrix(X_test_mapped.values, feature_names=feature_names, missing=np.nan)
        
        # Make prediction
        prob = model.predict(dmat)[0]
        pred = 1 if prob > 0.5 else 0
        
        # Prepare result
        result = {
            'player1_id': player1_id,
            'player2_id': player2_id,
            'surface': surface,
            'predicted_winner': player1_id if pred == 1 else player2_id,
            'predicted_proba': float(prob) if pred == 1 else float(1 - prob),
            'player1_win_proba': float(prob),
            'player2_win_proba': float(1 - prob),
            'features_used': list(feature_names)
        }
        
        logger.info(f"Prediction: {result['predicted_winner']} will win with {result['predicted_proba']:.2f} probability")
        
        return result
    
    except Exception as e:
        logger.error(f"Error predicting match: {e}")
        raise


def get_player_stats_from_db(player_ids: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Get player statistics from the database for prediction.
    
    Args:
        player_ids: List of player IDs to get stats for
        
    Returns:
        Dictionary mapping player IDs to their statistics
    """
    logger.info(f"Getting stats for {len(player_ids)} players from database")
    
    try:
        # Connect to database
        conn = get_database_connection()
        
        # Create empty result dictionary
        player_stats = {}
        
        for player_id in tqdm(player_ids, desc="Fetching player stats"):
            # Query to get the latest match features for this player
            query = f"""
            WITH player_matches AS (
                SELECT * FROM match_features
                WHERE player1_id = {player_id} OR player2_id = {player_id}
                ORDER BY tournament_date DESC
                LIMIT 1
            )
            SELECT * FROM player_matches
            """
            
            # Execute query
            player_df = pd.read_sql(query, conn)
            
            if len(player_df) == 0:
                logger.warning(f"No matches found for player {player_id}")
                continue
            
            # Extract player stats
            stats = {}
            
            # Get all player-specific columns
            if str(player_id) == str(player_df['player1_id'].iloc[0]):
                # Player is player1 in the latest match
                prefix = 'player1_'
                opponent_prefix = 'player2_'
            else:
                # Player is player2 in the latest match
                prefix = 'player2_'
                opponent_prefix = 'player1_'
            
            # Get all raw player features
            for col in player_df.columns:
                if col.startswith(prefix):
                    feature_name = col[len(prefix):]  # Remove prefix
                    stats[feature_name] = float(player_df[col].iloc[0])
            
            # Get Elo if available (from _diff features)
            if 'player_elo_diff' in player_df.columns:
                # Positive diff means player1 has higher Elo, negative means player2 has higher Elo
                elo_diff = float(player_df['player_elo_diff'].iloc[0])
                
                if prefix == 'player1_':
                    # If player is player1, their Elo is higher by elo_diff
                    if 'elo' in stats:
                        # If we already have an Elo value, leave it
                        pass
                    else:
                        # Assume a base Elo and add the diff (this is approximate)
                        stats['elo'] = 1500.0 + abs(elo_diff) / 2
                else:
                    # If player is player2, their Elo is lower by elo_diff
                    if 'elo' in stats:
                        pass
                    else:
                        stats['elo'] = 1500.0 - abs(elo_diff) / 2
            
            # Store player stats
            player_stats[str(player_id)] = stats
            
        logger.info(f"Retrieved stats for {len(player_stats)} players")
        return player_stats
    
    except Exception as e:
        logger.error(f"Error getting player stats: {e}")
        raise
    
    finally:
        conn.close()


def predict_upcoming_matches(model_path: Union[str, Path], matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Predict outcomes for a list of upcoming matches.
    
    Args:
        model_path: Path to the trained model
        matches: List of dictionaries with player1_id, player2_id, and surface
        
    Returns:
        List of dictionaries with prediction results
    """
    logger.info(f"Predicting outcomes for {len(matches)} upcoming matches")
    
    try:
        # Load model
        model = load_model(model_path)
        
        # Get unique player IDs
        player_ids = []
        for match in matches:
            player_ids.append(str(match['player1_id']))
            player_ids.append(str(match['player2_id']))
        player_ids = list(set(player_ids))
        
        # Get player stats
        player_stats = get_player_stats_from_db(player_ids)
        
        # Make predictions
        predictions = []
        for match in tqdm(matches, desc="Predicting matches"):
            try:
                result = predict_match(
                    model, 
                    str(match['player1_id']), 
                    str(match['player2_id']), 
                    match['surface'], 
                    player_stats
                )
                
                # Add match info to result
                result.update({
                    'tournament_name': match.get('tournament_name', ''),
                    'match_date': match.get('match_date', '')
                })
                
                predictions.append(result)
                
            except Exception as e:
                logger.error(f"Error predicting match: {e}")
                
                # Add error match to predictions with error message
                predictions.append({
                    'player1_id': match['player1_id'],
                    'player2_id': match['player2_id'],
                    'surface': match['surface'],
                    'error': str(e),
                    'tournament_name': match.get('tournament_name', ''),
                    'match_date': match.get('match_date', '')
                })
        
        return predictions
    
    except Exception as e:
        logger.error(f"Error in prediction process: {e}")
        raise


def main():
    """Make predictions using the trained model on fresh data from the database."""
    start_time = time.time()
    
    # Define total steps for progress tracking
    total_steps = 7
    progress_tracker = ProgressTracker(total_steps)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Make predictions using the trained model.')
    parser.add_argument('--use_file', action='store_true', help='Use file data instead of database')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data to use for testing')
    parser.add_argument('--limit', type=int, default=None, help='Limit on number of test matches to use')
    parser.add_argument('--cutoff_date', type=str, default=None, 
                        help='Use matches after this date for testing (YYYY-MM-DD)')
    
    # Add arguments for predicting specific matches
    parser.add_argument('--predict_match', action='store_true', 
                        help='Predict a specific upcoming match')
    parser.add_argument('--player1_id', type=int, help='ID of player 1')
    parser.add_argument('--player2_id', type=int, help='ID of player 2')
    parser.add_argument('--surface', type=str, choices=SURFACES, 
                        help='Match surface (hard, clay, grass, carpet)')
    parser.add_argument('--tournament', type=str, default='', help='Tournament name')
    
    args = parser.parse_args()
    
    # If predicting a specific match
    if args.predict_match:
        if not (args.player1_id and args.player2_id and args.surface):
            logger.error("Must provide player1_id, player2_id, and surface for match prediction")
            return
        
        try:
            # Prepare match information
            match = {
                'player1_id': args.player1_id,
                'player2_id': args.player2_id,
                'surface': args.surface,
                'tournament_name': args.tournament,
                'match_date': datetime.now().strftime("%Y-%m-%d")
            }
            
            # Predict match outcome
            predictions = predict_upcoming_matches(MODEL_PATH, [match])
            
            # Print prediction results
            if predictions and len(predictions) > 0:
                prediction = predictions[0]
                
                # Check for errors
                if 'error' in prediction:
                    logger.error(f"Error predicting match: {prediction['error']}")
                    return
                
                # Print formatted prediction
                print("\n---- Match Prediction ----")
                print(f"Player 1 ID: {prediction['player1_id']}")
                print(f"Player 2 ID: {prediction['player2_id']}")
                print(f"Surface: {prediction['surface']}")
                print(f"Tournament: {prediction.get('tournament_name', 'Unknown')}")
                print(f"Date: {prediction.get('match_date', 'Unknown')}")
                print("\nPrediction Results:")
                print(f"Predicted Winner: {prediction['predicted_winner']}")
                print(f"Win Probability: {prediction['predicted_proba']:.2f} ({prediction['predicted_proba']*100:.1f}%)")
                print(f"Player 1 Win Probability: {prediction['player1_win_proba']:.2f} ({prediction['player1_win_proba']*100:.1f}%)")
                print(f"Player 2 Win Probability: {prediction['player2_win_proba']:.2f} ({prediction['player2_win_proba']*100:.1f}%)")
                print(f"Key Features Used: {', '.join(prediction['features_used'][:5])}...")
                print("------------------------\n")
            else:
                logger.error("No prediction results returned")
            
            return
        
        except Exception as e:
            logger.error(f"Error in match prediction: {e}")
            raise
    
    try:
        # Step 1: Load model
        logger.info(f"Step 1/{total_steps}: Loading model...")
        model = load_model(MODEL_PATH, progress_tracker)
        
        # Step 2: Load test data
        logger.info(f"Step 2/{total_steps}: Loading test data...")
        if args.use_file:
            logger.info("Using file data as specified by --use_file flag")
            test_df = load_test_data(FEATURES_PATH, args.test_size, progress_tracker)
        else:
            # First check if we have a saved test set from training
            saved_test_path = OUTPUT_DIR / "test_data_v3.csv"
            if os.path.exists(saved_test_path):
                logger.info(f"Loading saved test set from {saved_test_path}")
                test_df = pd.read_csv(saved_test_path)
                if 'tournament_date' in test_df.columns:
                    test_df['tournament_date'] = pd.to_datetime(test_df['tournament_date'])
                logger.info(f"Loaded {len(test_df)} matches from saved test set")
                if progress_tracker:
                    progress_tracker.update("Test data loading complete")
            else:
                logger.info("No saved test set found. Using database data")
                test_df = load_test_data_from_database(args.cutoff_date, args.limit, progress_tracker)
            
        if len(test_df) == 0:
            logger.error("No test data found. Exiting.")
            return
            
        # Step 3: Get feature columns
        logger.info(f"Step 3/{total_steps}: Identifying feature columns...")
        feature_cols = get_feature_columns(test_df, progress_tracker)
        
        # Check if we have all features the model expects
        model_features = model.feature_names
        missing_features = [f for f in model_features if f not in feature_cols]
        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features expected by model: {missing_features[:5]}...")
            logger.warning("This may lead to poor predictions. Consider regenerating features.")
        
        # Step 4: Make predictions
        logger.info(f"Step 4/{total_steps}: Making predictions...")
        predictions_df, metrics = make_predictions(model, test_df, feature_cols, progress_tracker)
        
        # Step 5: Calculate match-level metrics
        logger.info(f"Step 5/{total_steps}: Calculating match-level metrics...")
        match_metrics = calculate_match_level_metrics(predictions_df, progress_tracker)
        metrics['match_level'] = match_metrics
        
        # Step 6: Plot results
        logger.info(f"Step 6/{total_steps}: Plotting results...")
        plot_results(predictions_df, metrics, OUTPUT_DIR / "plots", progress_tracker)
        
        # Step 7: Save results
        logger.info(f"Step 7/{total_steps}: Saving results...")
        save_results(
            predictions_df, 
            metrics, 
            PREDICTIONS_OUTPUT, 
            OUTPUT_DIR / "prediction_metrics_v3.json", 
            progress_tracker
        )
        
        # Print final message
        elapsed_time = time.time() - start_time
        logger.info(f"Prediction completed in {elapsed_time:.2f} seconds")
        logger.info(f"Overall accuracy: {metrics['overall']['accuracy']:.4f}")
        
        # Return metrics for potential further use
        return metrics
    
    except Exception as e:
        logger.error(f"Error in prediction process: {e}")
        raise


if __name__ == "__main__":
    main() 