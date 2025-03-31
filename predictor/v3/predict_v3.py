#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tennis Match Prediction Script - Version 3

This script loads the trained XGBoost model from train_model_v3.py and makes
predictions on test data from the database, ensuring we don't use training data.

The script:
1. Connects to the database and loads recent test data
2. Loads the trained model pipeline
3. Makes predictions with probabilities
4. Generates evaluation metrics and visualizations
5. Analyzes prediction quality and identifies discrepancies
6. Outputs results to the output/v3 directory
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import time
import pickle
import logging
import warnings
import argparse
from datetime import datetime, timedelta
from typing import Optional, Any, Dict, List, Tuple, Union, cast

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import psycopg2
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve, auc,
    precision_recall_curve, log_loss, brier_score_loss,
    confusion_matrix, balanced_accuracy_score, classification_report
)
from sklearn.calibration import calibration_curve
from psycopg2 import pool
from dotenv import load_dotenv
import psutil
from predictor.v3.data_cache import get_cache_key, get_cached_data, save_to_cache

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("predict_v3.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("tennis_predictor")

# Project directories
PROJECT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output" / "v3"
MODELS_DIR = OUTPUT_DIR / "models"
PLOTS_DIR = OUTPUT_DIR / "plots"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

# Create output directories if they don't exist
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Input/output file paths
MODEL_PATH = MODELS_DIR / "tennis_model_v3.json"
PIPELINE_PATH = MODELS_DIR / "tennis_pipeline_v3.pkl"
PREDICTIONS_OUTPUT = PREDICTIONS_DIR / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
METRICS_OUTPUT = PREDICTIONS_DIR / f"prediction_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# Tennis surfaces for surface-specific evaluation
SURFACES = ["hard", "clay", "grass", "carpet"]

# Database configuration
DB_BATCH_SIZE = 10000  # Number of records to fetch in each database batch
DB_TIMEOUT_SECONDS = 30  # Database query timeout in seconds

# Default CPU threads to use if GPU is not available
DEFAULT_CPU_THREADS = max(4, os.cpu_count() - 1) if os.cpu_count() else 4


def detect_optimal_device() -> dict:
    """
    Detect the optimal device for XGBoost prediction.
    Tries GPU first with CUDA, then falls back to CPU with multithreading.
    
    Returns:
        Dict with optimal parameters for tree_method and nthread
    """
    # Try GPU first
    try:
        # Create small test matrix to verify GPU works
        test_matrix = xgb.DMatrix(np.array([[1, 2], [3, 4]]), label=np.array([0, 1]))
        test_params = {'tree_method': 'gpu_hist'}
        xgb.train(test_params, test_matrix, num_boost_round=1)
        
        logger.info("CUDA GPU acceleration available and will be used")
        return {
            'tree_method': 'gpu_hist',
            'gpu_id': 0
        }
    except Exception as e:
        logger.info(f"GPU acceleration not available ({str(e)}), falling back to CPU with multithreading")
        
        # Determine optimal number of threads for CPU - use all cores minus 1
        cpu_threads = DEFAULT_CPU_THREADS
        logger.info(f"Using {cpu_threads} CPU threads for XGBoost prediction")
        
        return {
            'tree_method': 'hist',
            'nthread': cpu_threads
        }


class ProgressTracker:
    """Helper class to track and log progress during model prediction."""
    
    def __init__(self, total_steps: int, task_description: str):
        """
        Initialize the progress tracker.
        
        Args:
            total_steps: Total number of steps in the process
            task_description: Description of the task
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.task_description = task_description
        self.start_time = time.time()
        self.last_update_time = self.start_time
        
        logger.info(f"Starting {task_description} with {total_steps} steps")
    
    def update(self, step_description: str) -> None:
        """
        Update the progress tracker.
        
        Args:
            step_description: Description of the current step
        """
        self.current_step += 1
        current_time = time.time()
        elapsed = current_time - self.last_update_time
        total_elapsed = current_time - self.start_time
        
        logger.info(f"[{self.current_step}/{self.total_steps}] {step_description} "
                   f"(step: {elapsed:.2f}s, total: {total_elapsed:.2f}s)")
        
        self.last_update_time = current_time
    
    def get_progress(self) -> float:
        """
        Get the current progress as a fraction.
        
        Returns:
            Progress as a float between 0 and 1
        """
        return self.current_step / self.total_steps


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


def load_test_data_from_database(
    training_end_date: Optional[datetime] = None,
    limit: Optional[int] = None,
    progress_tracker: Optional[ProgressTracker] = None
) -> pd.DataFrame:
    """
    Load test data from the database, ensuring we don't use training data.
    
    Args:
        training_end_date: End date of training data to avoid using training data
        limit: Optional limit on number of rows to fetch
        progress_tracker: Optional progress tracker
        
    Returns:
        DataFrame with tennis match features for testing
    """
    # If training_end_date is not provided, try to get it from metrics file
    if training_end_date is None:
        metrics_files = list(OUTPUT_DIR.glob("model_metrics_v3.json"))
        if metrics_files:
            try:
                with open(metrics_files[0], 'r') as f:
                    metrics = json.load(f)
                    if 'training_info' in metrics and 'data_date_range' in metrics['training_info']:
                        date_str = metrics['training_info']['data_date_range'].get('end')
                        if date_str:
                            training_end_date = datetime.fromisoformat(date_str)
                            logger.info(f"Using training end date from metrics: {training_end_date.date()}")
            except Exception as e:
                logger.warning(f"Could not load training end date from metrics: {e}")
    
    # If still no training_end_date, use default of 1 year ago
    if training_end_date is None:
        training_end_date = datetime.now() - timedelta(days=365)
        logger.warning(f"No training end date found. Using default: {training_end_date.date()}")
    
    # Define the base query to fetch data after training period
    base_query = f"""
    SELECT *
    FROM match_features
    WHERE tournament_date > '{training_end_date.date()}'
    ORDER BY tournament_date ASC
    """
    
    # Generate cache key
    cache_key = get_cache_key(base_query, limit)
    
    # Try to load from cache first
    cached_df = get_cached_data(cache_key)
    if cached_df is not None:
        logger.info("Using cached test data")
        if progress_tracker:
            progress_tracker.update("Test data loaded from cache")
        return cached_df
    
    # If not in cache, load from database
    logger.info("Loading test data from database (match_features table)...")
    
    # Connect to database
    conn = get_database_connection()
    
    try:
        # Define the query with dynamic row limit
        limit_clause = f"LIMIT {limit}" if limit else ""
        
        # Use batched loading to handle large datasets efficiently
        offset = 0
        dataframes = []
        total_rows = 0
        
        # Get total count first
        with conn.cursor() as cursor:
            count_query = f"""
            SELECT COUNT(*) 
            FROM match_features 
            WHERE tournament_date > '{training_end_date.date()}'
            """
            cursor.execute(count_query)
            total_to_fetch = cursor.fetchone()[0]
            
            if limit and limit < total_to_fetch:
                total_to_fetch = limit
        
        logger.info(f"Fetching up to {total_to_fetch} rows of test data from database")
        pbar = tqdm(total=total_to_fetch, desc="Loading test data from database")
        
        while True:
            # Define batch query
            query = f"""
            {base_query}
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
            
            # Ensure result is an integer (1 for win, 0 for loss)
            if 'result' in df.columns:
                df['result'] = df['result'].astype(int)
            
            # Sort by date
            df = df.sort_values(by='tournament_date').reset_index(drop=True)
            
            logger.info(f"Loaded {len(df)} test matches from {df['tournament_date'].min().date()} to {df['tournament_date'].max().date()}")
            
            # Save to cache for future use
            save_to_cache(df, cache_key)
            
            if progress_tracker:
                progress_tracker.update("Test data loading complete")
            
            return df
        else:
            logger.warning("No test data retrieved from database")
            return pd.DataFrame()
    
    finally:
        conn.close()


def load_model_pipeline() -> dict:
    """
    Load the trained model pipeline.
    
    Returns:
        Dictionary containing model, features, and metadata
    """
    logger.info(f"Loading model pipeline from {PIPELINE_PATH}")
    
    try:
        with open(PIPELINE_PATH, 'rb') as f:
            pipeline = pickle.load(f)
        
        # Verify pipeline structure
        required_keys = ['model', 'features', 'metadata']
        for key in required_keys:
            if key not in pipeline:
                raise ValueError(f"Invalid pipeline format: missing '{key}'")
        
        # Log model metadata
        logger.info(f"Model type: {pipeline['metadata'].get('model_type', 'unknown')}")
        logger.info(f"Model features: {pipeline['metadata'].get('feature_count', 'unknown')}")
        logger.info(f"Model version: {pipeline['metadata'].get('version', 'unknown')}")
        logger.info(f"Creation date: {pipeline['metadata'].get('creation_date', 'unknown')}")
        
        return pipeline
    except Exception as e:
        logger.error(f"Error loading model pipeline: {e}")
        raise


def prepare_features_for_prediction(
    df: pd.DataFrame,
    feature_names: list,
    categorical_features: Optional[list] = None,
    progress_tracker: Optional[ProgressTracker] = None
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Prepare features for prediction using the same approach as in training.
    
    Args:
        df: DataFrame containing tennis match features
        feature_names: List of feature names to use
        categorical_features: List of categorical feature names
        progress_tracker: Optional progress tracker
        
    Returns:
        Tuple of (feature matrix, labels if available)
    """
    logger.info("Preparing features for prediction")
    
    # Ensure all required features are present
    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        logger.warning(f"Missing {len(missing_features)} features in prediction data: {missing_features[:5]}...")
        logger.warning("Will proceed with available features, but predictions may be less accurate")
        
        # Use only available features
        available_features = [f for f in feature_names if f in df.columns]
        if not available_features:
            raise ValueError("No model features found in prediction data")
        feature_names = available_features
    
    # Extract features
    X = df[feature_names].copy()
    
    # Handle categorical features
    if categorical_features:
        for col in categorical_features:
            if col in X.columns and (df[col].dtype == 'object' or df[col].dtype.name == 'category'):
                # Convert to numeric codes for XGBoost
                X[col] = pd.Categorical(X[col]).codes
                
                # Replace -1 (unknown category) with NaN for XGBoost to handle
                X[col] = X[col].replace(-1, np.nan)
    
    # Ensure all features are numeric
    for col in feature_names:
        if col in X.columns and not pd.api.types.is_numeric_dtype(X[col]):
            logger.warning(f"Converting non-numeric feature {col} to float type")
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Convert to numpy array
    X_array = X.values.astype(np.float32)
    
    # Extract labels if available
    y_array = None
    if 'result' in df.columns:
        y_array = df['result'].values
    
    # Log data info
    logger.info(f"Prepared features: {X_array.shape[1]} features, {X_array.shape[0]} samples")
    if y_array is not None:
        logger.info(f"Class distribution: {np.bincount(y_array)}")
    
    if progress_tracker:
        progress_tracker.update("Feature preparation complete")
    
    return X_array, y_array


def make_predictions(
    model: xgb.Booster,
    X: np.ndarray,
    feature_names: list,
    progress_tracker: Optional[ProgressTracker] = None
) -> np.ndarray:
    """
    Make predictions using the trained model.
    
    Args:
        model: Trained XGBoost model
        X: Feature matrix
        feature_names: List of feature names
        progress_tracker: Optional progress tracker
        
    Returns:
        Array of prediction probabilities
    """
    logger.info(f"Making predictions on {X.shape[0]} samples")
    
    # Create DMatrix for efficient prediction
    dmatrix = xgb.DMatrix(X, feature_names=feature_names)
    
    # Make predictions - use GPU if available
    device_params = detect_optimal_device()
    logger.info(f"Prediction using {device_params.get('tree_method', 'unknown')} method")
    
    start_time = time.time()
    y_pred_proba = model.predict(dmatrix)
    prediction_time = time.time() - start_time
    
    logger.info(f"Predictions completed in {prediction_time:.2f} seconds")
    logger.info(f"Average prediction probability: {y_pred_proba.mean():.4f}")
    
    if progress_tracker:
        progress_tracker.update("Predictions complete")
    
    return y_pred_proba


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
    progress_tracker: Optional[ProgressTracker] = None
) -> dict:
    """
    Evaluate prediction performance with comprehensive metrics.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold
        progress_tracker: Optional progress tracker
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info(f"Evaluating predictions with threshold {threshold}")
    
    # Convert probabilities to binary predictions
    y_pred = (y_pred_proba > threshold).astype(int)
    
    # Calculate standard classification metrics
    metrics = {}
    
    try:
        # Basic metrics
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        metrics['balanced_accuracy'] = float(balanced_accuracy_score(y_true, y_pred))
        metrics['precision'] = float(precision_score(y_true, y_pred))
        metrics['recall'] = float(recall_score(y_true, y_pred))
        metrics['f1'] = float(f1_score(y_true, y_pred))
        
        # ROC AUC
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba))
        
        # Average precision (PR AUC)
        metrics['pr_auc'] = float(average_precision_score(y_true, y_pred_proba))
        
        # Log loss
        metrics['log_loss'] = float(log_loss(y_true, y_pred_proba))
        
        # Brier score (calibration metric)
        metrics['brier_score'] = float(brier_score_loss(y_true, y_pred_proba))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        metrics['confusion_matrix'] = {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp)
        }
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        metrics['classification_report'] = report
        
        # Log results
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"PR AUC: {metrics['pr_auc']:.4f}")
        logger.info(f"Brier Score: {metrics['brier_score']:.4f}")
        logger.info(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        
    except Exception as e:
        logger.error(f"Error calculating evaluation metrics: {e}")
        metrics['error'] = str(e)
    
    if progress_tracker:
        progress_tracker.update("Evaluation complete")
    
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Optional[Path] = None,
    progress_tracker: Optional[ProgressTracker] = None
) -> None:
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_path: Path to save the plot
        progress_tracker: Optional progress tracker
    """
    logger.info("Plotting confusion matrix")
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title('Confusion Matrix (Counts)')
    ax1.set_xticklabels(['Loss', 'Win'])
    ax1.set_yticklabels(['Loss', 'Win'])
    
    # Plot percentages
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues', cbar=False, ax=ax2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title('Confusion Matrix (Percentages)')
    ax2.set_xticklabels(['Loss', 'Win'])
    ax2.set_yticklabels(['Loss', 'Win'])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {output_path}")
    
    plt.close()
    
    if progress_tracker:
        progress_tracker.update("Confusion matrix plot complete")


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: Optional[Path] = None,
    progress_tracker: Optional[ProgressTracker] = None
) -> None:
    """
    Plot and save ROC curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        output_path: Path to save the plot
        progress_tracker: Optional progress tracker
    """
    logger.info("Plotting ROC curve")
    
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
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved ROC curve to {output_path}")
    
    plt.close()
    
    if progress_tracker:
        progress_tracker.update("ROC curve plot complete")


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: Optional[Path] = None,
    progress_tracker: Optional[ProgressTracker] = None
) -> None:
    """
    Plot and save precision-recall curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        output_path: Path to save the plot
        progress_tracker: Optional progress tracker
    """
    logger.info("Plotting precision-recall curve")
    
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
                    label=f'Threshold = {threshold:.2f}')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved precision-recall curve to {output_path}")
    
    plt.close()
    
    if progress_tracker:
        progress_tracker.update("Precision-recall curve plot complete")


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: Optional[Path] = None,
    progress_tracker: Optional[ProgressTracker] = None
) -> None:
    """
    Plot and save calibration curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        output_path: Path to save the plot
        progress_tracker: Optional progress tracker
    """
    logger.info("Plotting calibration curve")
    
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
    plt.title('Calibration Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved calibration curve to {output_path}")
    
    plt.close()
    
    if progress_tracker:
        progress_tracker.update("Calibration curve plot complete")


def plot_probability_distribution(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: Optional[Path] = None,
    progress_tracker: Optional[ProgressTracker] = None
) -> None:
    """
    Plot distribution of prediction probabilities.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        output_path: Path to save the plot
        progress_tracker: Optional progress tracker
    """
    logger.info("Plotting probability distribution")
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot histograms for each class
    bins = np.linspace(0, 1, 21)  # 20 bins
    
    # Class 0 (Losses)
    class0_probs = y_pred_proba[y_true == 0]
    if len(class0_probs) > 0:
        plt.hist(class0_probs, bins=bins, alpha=0.5, color='red', 
                 label=f'Actual losses (n={len(class0_probs)})')
    
    # Class 1 (Wins)
    class1_probs = y_pred_proba[y_true == 1]
    if len(class1_probs) > 0:
        plt.hist(class1_probs, bins=bins, alpha=0.5, color='blue', 
                 label=f'Actual wins (n={len(class1_probs)})')
    
    # Add vertical line at threshold 0.5
    plt.axvline(x=0.5, color='black', linestyle='--', 
                label='Decision threshold (0.5)')
    
    plt.xlabel('Predicted probability of winning')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Probabilities by Actual Outcome')
    plt.legend()
    plt.grid(alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved probability distribution plot to {output_path}")
    
    plt.close()
    
    if progress_tracker:
        progress_tracker.update("Probability distribution plot complete")


def analyze_prediction_quality(
    df: pd.DataFrame,
    y_pred_proba: np.ndarray,
    metrics: dict,
    progress_tracker: Optional[ProgressTracker] = None
) -> dict:
    """
    Analyze prediction quality and identify discrepancies.
    
    Args:
        df: Original DataFrame with match details
        y_pred_proba: Predicted probabilities
        metrics: Evaluation metrics
        progress_tracker: Optional progress tracker
        
    Returns:
        Dictionary with analysis results
    """
    logger.info("Analyzing prediction quality")
    
    # Copy DataFrame and add predictions
    analysis_df = df.copy()
    analysis_df['prediction_proba'] = y_pred_proba
    analysis_df['predicted_winner'] = (y_pred_proba > 0.5).astype(int)
    
    # Calculate correctness
    if 'result' in analysis_df.columns:
        analysis_df['correct'] = analysis_df['predicted_winner'] == analysis_df['result']
    
    analysis = {}
    
    # 1. Surface analysis
    if 'surface' in analysis_df.columns and 'result' in analysis_df.columns:
        surface_metrics = {}
        
        for surface in SURFACES:
            surface_df = analysis_df[analysis_df['surface'].str.lower() == surface.lower()]
            if len(surface_df) > 0:
                y_true_surface = surface_df['result'].values
                y_pred_surface = surface_df['predicted_winner'].values
                y_proba_surface = surface_df['prediction_proba'].values
                
                # Calculate metrics for this surface
                try:
                    surface_metrics[surface] = {
                        'count': int(len(surface_df)),
                        'accuracy': float(accuracy_score(y_true_surface, y_pred_surface)),
                        'roc_auc': float(roc_auc_score(y_true_surface, y_proba_surface)) 
                            if len(np.unique(y_true_surface)) > 1 else None
                    }
                except Exception as e:
                    logger.warning(f"Error calculating metrics for {surface} surface: {e}")
        
        analysis['surface_analysis'] = surface_metrics
    
    # 2. Tournament level analysis
    if 'tournament_level' in analysis_df.columns and 'result' in analysis_df.columns:
        level_metrics = {}
        
        for level in analysis_df['tournament_level'].unique():
            if pd.isna(level):
                continue
                
            level_df = analysis_df[analysis_df['tournament_level'] == level]
            if len(level_df) > 0:
                y_true_level = level_df['result'].values
                y_pred_level = level_df['predicted_winner'].values
                y_proba_level = level_df['prediction_proba'].values
                
                # Calculate metrics for this tournament level
                try:
                    level_metrics[str(level)] = {
                        'count': int(len(level_df)),
                        'accuracy': float(accuracy_score(y_true_level, y_pred_level)),
                        'roc_auc': float(roc_auc_score(y_true_level, y_proba_level))
                            if len(np.unique(y_true_level)) > 1 else None
                    }
                except Exception as e:
                    logger.warning(f"Error calculating metrics for tournament level {level}: {e}")
        
        analysis['tournament_level_analysis'] = level_metrics
    
    # 3. Check for potential issues or discrepancies
    discrepancies = []
    
    # Surface performance discrepancy
    if 'surface_analysis' in analysis:
        surface_accs = [(s, m['accuracy']) for s, m in analysis['surface_analysis'].items() 
                        if m['count'] >= 50]  # Only consider surfaces with enough matches
        
        if surface_accs:
            best_surface = max(surface_accs, key=lambda x: x[1])
            worst_surface = min(surface_accs, key=lambda x: x[1])
            
            if best_surface[1] - worst_surface[1] > 0.1:  # Difference > 10%
                discrepancies.append({
                    'type': 'surface_performance_gap',
                    'description': f"Large performance gap between surfaces: {best_surface[0]} ({best_surface[1]:.2%}) vs {worst_surface[0]} ({worst_surface[1]:.2%})",
                    'severity': 'medium'
                })
    
    # Unexpected probability distributions
    if 'result' in analysis_df.columns:
        class0_probs = y_pred_proba[analysis_df['result'] == 0]
        class1_probs = y_pred_proba[analysis_df['result'] == 1]
        
        if len(class0_probs) > 0 and len(class1_probs) > 0:
            # Check if mean probability for winners is lower than expected
            if np.mean(class1_probs) < 0.6:
                discrepancies.append({
                    'type': 'low_winner_confidence',
                    'description': f"Model lacks confidence in true winners. Mean probability for winners: {np.mean(class1_probs):.2f}",
                    'severity': 'high' if np.mean(class1_probs) < 0.55 else 'medium'
                })
            
            # Check if mean probability for losers is higher than expected
            if np.mean(class0_probs) > 0.4:
                discrepancies.append({
                    'type': 'high_loser_confidence',
                    'description': f"Model overconfident in predicting losers as winners. Mean probability for losers: {np.mean(class0_probs):.2f}",
                    'severity': 'high' if np.mean(class0_probs) > 0.45 else 'medium'
                })
    
    # Overall calibration issues
    if metrics.get('brier_score', 0) > 0.25:
        discrepancies.append({
            'type': 'poor_calibration',
            'description': f"Model shows poor probability calibration (Brier score: {metrics['brier_score']:.3f})",
            'severity': 'high'
        })
    
    # Threshold analysis
    if 'roc_auc' in metrics and metrics['roc_auc'] > 0.6:
        # If AUC is decent but accuracy is lower than expected,
        # the threshold might need adjustment
        if metrics.get('accuracy', 0) < 0.6:
            discrepancies.append({
                'type': 'threshold_adjustment',
                'description': "Model has good ROC AUC but lower accuracy. Consider adjusting the prediction threshold.",
                'severity': 'medium'
            })
    
    analysis['discrepancies'] = discrepancies
    
    # 4. Timestamp analysis
    if 'tournament_date' in analysis_df.columns and 'result' in analysis_df.columns:
        # Group by month and calculate performance
        analysis_df['month'] = analysis_df['tournament_date'].dt.to_period('M')
        monthly_performance = analysis_df.groupby('month')['correct'].agg(['mean', 'count']).reset_index()
        monthly_performance['month'] = monthly_performance['month'].astype(str)
        
        # Convert to dictionary for JSON serialization
        analysis['monthly_performance'] = monthly_performance.to_dict(orient='records')
        
        # Check for performance trends
        if len(monthly_performance) >= 3:
            recent_months = monthly_performance.iloc[-3:]
            if recent_months['mean'].is_monotonic_decreasing and recent_months['mean'].iloc[-1] < 0.6:
                discrepancies.append({
                    'type': 'decreasing_performance',
                    'description': "Model performance is steadily decreasing in recent months",
                    'severity': 'high'
                })
    
    # Log major findings
    if discrepancies:
        logger.warning("Analysis found the following issues:")
        for d in discrepancies:
            logger.warning(f"  - {d['severity'].upper()}: {d['description']}")
    else:
        logger.info("No significant issues found in prediction analysis")
    
    if progress_tracker:
        progress_tracker.update("Prediction quality analysis complete")
    
    return analysis


def save_predictions_and_analysis(
    df: pd.DataFrame,
    y_pred_proba: np.ndarray,
    metrics: dict,
    analysis: dict,
    predictions_path: Path,
    metrics_path: Path,
    progress_tracker: Optional[ProgressTracker] = None
) -> None:
    """
    Save predictions, metrics, and analysis to files.
    
    Args:
        df: Original DataFrame with match details
        y_pred_proba: Predicted probabilities
        metrics: Evaluation metrics
        analysis: Analysis results
        predictions_path: Path to save predictions CSV
        metrics_path: Path to save metrics JSON
        progress_tracker: Optional progress tracker
    """
    logger.info("Saving predictions and analysis")
    
    # Save predictions to CSV
    predictions_df = df.copy()
    predictions_df['prediction_probability'] = y_pred_proba
    predictions_df['predicted_winner'] = (y_pred_proba > 0.5).astype(int)
    
    # Add correctness if actual results are available
    if 'result' in predictions_df.columns:
        predictions_df['correct'] = predictions_df['predicted_winner'] == predictions_df['result']
    
    # Save to CSV
    predictions_df.to_csv(predictions_path, index=False)
    logger.info(f"Saved predictions to {predictions_path}")
    
    # Combine metrics and analysis
    combined_results = {
        'metrics': metrics,
        'analysis': analysis,
        'timestamp': datetime.now().isoformat(),
        'prediction_count': len(y_pred_proba),
        'data_range': {
            'start': predictions_df['tournament_date'].min().isoformat() if 'tournament_date' in predictions_df.columns else None,
            'end': predictions_df['tournament_date'].max().isoformat() if 'tournament_date' in predictions_df.columns else None
        }
    }
    
    # Save to JSON
    with open(metrics_path, 'w') as f:
        json.dump(combined_results, f, indent=2)
    logger.info(f"Saved metrics and analysis to {metrics_path}")
    
    if progress_tracker:
        progress_tracker.update("Saved predictions and analysis")


def main(args: argparse.Namespace) -> None:
    """
    Main function for tennis match prediction.
    
    Args:
        args: Command-line arguments
    """
    start_time = time.time()
    
    # Create output directories if they don't exist
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Log hardware information
    logger.info(f"XGBoost version: {xgb.__version__}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"CPU count: {os.cpu_count()}")
    logger.info(f"Memory: {psutil.virtual_memory().total / (1024 ** 3):.1f} GB")
    
    # Define the total number of processing steps for progress tracking
    total_steps = 8
    progress_tracker = ProgressTracker(total_steps, "Tennis Match Prediction")
    
    try:
        # Step 1: Load model pipeline
        logger.info(f"Step 1/{total_steps}: Loading model pipeline")
        pipeline = load_model_pipeline()
        model = pipeline['model']
        feature_names = pipeline['features']
        categorical_features = [f for f in feature_names if any(cat in f.lower() for cat in ['surface', 'level', 'hand', 'court', 'round'])]
        progress_tracker.update("Model pipeline loaded")
        
        # Step 2: Load test data from database
        logger.info(f"Step 2/{total_steps}: Loading test data")
        
        # Use training end date from args if provided, otherwise None will use auto-detection
        training_end_date = None
        if args.after_date:
            try:
                training_end_date = datetime.fromisoformat(args.after_date)
                logger.info(f"Using specified date cutoff: {training_end_date.date()}")
            except ValueError:
                logger.error(f"Invalid date format: {args.after_date}, expected YYYY-MM-DD")
                logger.info("Using auto-detection for training end date")
        
        # Load test data
        df = load_test_data_from_database(
            training_end_date=training_end_date,
            limit=args.limit,
            progress_tracker=progress_tracker
        )
        
        # Check if we have data
        if len(df) == 0:
            logger.error("No test data available. Exiting.")
            return
        
        # Step 3: Prepare features for prediction
        logger.info(f"Step 3/{total_steps}: Preparing features")
        X, y = prepare_features_for_prediction(
            df, feature_names, categorical_features, progress_tracker
        )
        
        # Step 4: Make predictions
        logger.info(f"Step 4/{total_steps}: Making predictions")
        y_pred_proba = make_predictions(model, X, feature_names, progress_tracker)
        
        # Step 5: Evaluate predictions if we have ground truth
        logger.info(f"Step 5/{total_steps}: Evaluating predictions")
        if y is not None:
            metrics = evaluate_predictions(y, y_pred_proba, threshold=0.5, progress_tracker=progress_tracker)
        else:
            logger.warning("No ground truth available for evaluation")
            metrics = {}
        
        # Step 6: Generate visualization plots
        logger.info(f"Step 6/{total_steps}: Generating visualization plots")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if y is not None:
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Confusion matrix
            plot_confusion_matrix(
                y, y_pred,
                PLOTS_DIR / f"prediction_confusion_matrix_{timestamp}.png",
                progress_tracker
            )
            
            # ROC curve
            plot_roc_curve(
                y, y_pred_proba,
                PLOTS_DIR / f"prediction_roc_curve_{timestamp}.png",
                progress_tracker
            )
            
            # Precision-recall curve
            plot_precision_recall_curve(
                y, y_pred_proba,
                PLOTS_DIR / f"prediction_pr_curve_{timestamp}.png",
                progress_tracker
            )
            
            # Calibration curve
            plot_calibration_curve(
                y, y_pred_proba,
                PLOTS_DIR / f"prediction_calibration_curve_{timestamp}.png",
                progress_tracker
            )
            
            # Probability distribution
            plot_probability_distribution(
                y, y_pred_proba,
                PLOTS_DIR / f"prediction_probability_distribution_{timestamp}.png",
                progress_tracker
            )
        else:
            progress_tracker.update("Skipped visualization plots (no ground truth)")
        
        # Step 7: Analyze prediction quality
        logger.info(f"Step 7/{total_steps}: Analyzing prediction quality")
        analysis = analyze_prediction_quality(df, y_pred_proba, metrics, progress_tracker)
        
        # Step 8: Save predictions and analysis
        logger.info(f"Step 8/{total_steps}: Saving predictions and analysis")
        save_predictions_and_analysis(
            df, y_pred_proba, metrics, analysis,
            PREDICTIONS_DIR / f"predictions_{timestamp}.csv",
            PREDICTIONS_DIR / f"prediction_metrics_{timestamp}.json",
            progress_tracker
        )
        
        # Print summary
        total_time = time.time() - start_time
        logger.info(f"Prediction completed in {total_time:.2f} seconds")
        logger.info(f"Processed {len(df)} matches")
        
        # Print performance summary
        if y is not None:
            logger.info(f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
            logger.info(f"ROC AUC: {metrics.get('roc_auc', 'N/A'):.4f}")
            logger.info(f"PR AUC: {metrics.get('pr_auc', 'N/A'):.4f}")
        
        # Print discrepancies summary
        if 'discrepancies' in analysis and analysis['discrepancies']:
            logger.info("Analysis found the following potential issues:")
            for d in analysis['discrepancies']:
                logger.info(f"  - {d['severity'].upper()}: {d['description']}")
        else:
            logger.info("No significant issues found in prediction analysis")
        
    except Exception as e:
        logger.error(f"Error in prediction process: {str(e)}")
        logger.exception("Exception details:")
        raise


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Tennis Match Prediction Script - V3")
    
    # Add arguments
    parser.add_argument(
        "--after-date",
        type=str,
        help="Only use matches after this date (format: YYYY-MM-DD)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of matches to process"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run main function
    main(args)
