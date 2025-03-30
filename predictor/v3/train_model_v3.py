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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project directories
PROJECT_ROOT = Path(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "predictor" / "output" / "v3"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# File paths
INPUT_FILE = DATA_DIR / "v3" / "features_v3.csv"
MODEL_OUTPUT = OUTPUT_DIR / "tennis_model_v3.json"
METRICS_OUTPUT = OUTPUT_DIR / "metrics_v3.json"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Create plots directory if it doesn't exist
os.makedirs(PLOTS_DIR, exist_ok=True)

# Constants
SEED = 42
TEST_SIZE = 0.2
SURFACES = ['hard', 'clay', 'grass']

# XGBoost model parameters
MODEL_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'gamma': 0.1,
    'scale_pos_weight': 1,
    'seed': SEED,
    'verbosity': 0
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

# Surface-specific serve and return features will be dynamically generated


class ProgressTracker:
    """
    Class to track and log progress during model training.
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


def load_data(file_path: Union[str, Path], progress_tracker: Optional[ProgressTracker] = None) -> pd.DataFrame:
    """
    Load the tennis match features dataset.
    
    Args:
        file_path: Path to the input CSV file
        progress_tracker: Optional progress tracker
        
    Returns:
        DataFrame with tennis match features
    """
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    
    # Convert date columns to datetime
    if 'tourney_date' in df.columns:
        df['tourney_date'] = pd.to_datetime(df['tourney_date'])
    
    # Sort by date
    df = df.sort_values(by='tourney_date').reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} match features")
    
    if progress_tracker:
        progress_tracker.update("Data loading complete")
    
    return df


def create_time_based_split(df: pd.DataFrame, test_size: float = 0.2, 
                           progress_tracker: Optional[ProgressTracker] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a time-based train-test split.
    
    Args:
        df: DataFrame with match features
        test_size: Proportion of data to use for testing
        progress_tracker: Optional progress tracker
        
    Returns:
        Tuple of (train_df, test_df)
    """
    logger.info("Creating time-based train-test split")
    
    # Sort by date
    df = df.sort_values(by='tourney_date').reset_index(drop=True)
    
    # Calculate split index
    split_idx = int(len(df) * (1 - test_size))
    
    # Split data
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    logger.info(f"Train set: {len(train_df)} matches from {train_df['tourney_date'].min()} to {train_df['tourney_date'].max()}")
    logger.info(f"Test set: {len(test_df)} matches from {test_df['tourney_date'].min()} to {test_df['tourney_date'].max()}")
    
    if progress_tracker:
        progress_tracker.update("Time-based split complete")
    
    return train_df, test_df


def get_feature_columns(df: pd.DataFrame, progress_tracker: Optional[ProgressTracker] = None) -> List[str]:
    """
    Get the list of feature columns for training.
    
    Args:
        df: DataFrame with match features
        progress_tracker: Optional progress tracker
        
    Returns:
        List of feature column names
    """
    logger.info("Identifying feature columns")
    
    # Columns to exclude from features
    exclude_cols = ['match_id', 'tourney_date', 'player1_id', 'player2_id', 'surface', 'result']
    
    # Get all numeric columns except excluded ones
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Generate list of surface-specific serve and return features
    surface_specific_features = []
    for feature in SERVE_RETURN_FEATURES:
        base_feature = feature.split('_diff')[0]
        for surface in SURFACES:
            surface_feature = f"{base_feature}_{surface}_diff"
            if surface_feature in df.columns:
                surface_specific_features.append(surface_feature)
    
    # Ensure all surface-specific features are included
    for feature in surface_specific_features:
        if feature not in feature_cols:
            feature_cols.append(feature)
    
    logger.info(f"Selected {len(feature_cols)} feature columns")
    
    if progress_tracker:
        progress_tracker.update("Feature selection complete")
    
    return feature_cols


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                    feature_cols: List[str], 
                    progress_tracker: Optional[ProgressTracker] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare feature matrices and labels for training.
    
    Args:
        train_df: Training DataFrame
        test_df: Testing DataFrame
        feature_cols: List of feature column names
        progress_tracker: Optional progress tracker
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info("Preparing feature matrices and labels")
    
    # Check if there are any non-numeric values in feature columns
    non_numeric_cols = []
    for col in feature_cols:
        if train_df[col].dtype not in [np.int64, np.float64]:
            non_numeric_cols.append(col)
    
    if non_numeric_cols:
        logger.warning(f"Non-numeric columns found: {non_numeric_cols}")
        logger.warning("Converting non-numeric columns to numeric")
        
        for col in non_numeric_cols:
            train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
            test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
    
    # Extract features and labels
    X_train = train_df[feature_cols].values
    y_train = train_df['result'].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df['result'].values
    
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}")
    
    if progress_tracker:
        progress_tracker.update("Feature preparation complete")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train: np.ndarray, y_train: np.ndarray, 
               feature_cols: List[str], params: Dict[str, Any] = MODEL_PARAMS, 
               progress_tracker: Optional[ProgressTracker] = None) -> Tuple[xgb.Booster, Dict[str, Any]]:
    """
    Train an XGBoost model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        feature_cols: List of feature column names
        params: XGBoost model parameters
        progress_tracker: Optional progress tracker
        
    Returns:
        Tuple of (trained model, feature importances)
    """
    logger.info("Training XGBoost model")
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    
    # Set up training parameters
    num_rounds = 1000
    early_stopping_rounds = 50
    
    # Train model with progress tracking
    progress_callback = XGBoostProgressCallback(num_rounds)
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        callbacks=[progress_callback]
    )
    
    # Get feature importances
    importance_scores = model.get_score(importance_type='gain')
    importances = {feature: score for feature, score in importance_scores.items()}
    
    # Sort importances
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    
    logger.info(f"Top 5 important features: {list(importances.keys())[:5]}")
    
    if progress_tracker:
        progress_tracker.update("Model training complete")
    
    return model, importances


def evaluate_model(model: xgb.Booster, X_test: np.ndarray, y_test: np.ndarray, 
                  test_df: pd.DataFrame, feature_cols: List[str],
                  progress_tracker: Optional[ProgressTracker] = None) -> Dict[str, Any]:
    """
    Evaluate the trained model.
    
    Args:
        model: Trained XGBoost model
        X_test: Test features
        y_test: Test labels
        test_df: Test DataFrame
        feature_cols: List of feature column names
        progress_tracker: Optional progress tracker
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating model on test data")
    
    # Create DMatrix for testing
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)
    
    # Make predictions
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    logger.info(f"Overall metrics:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    
    # Calculate surface-specific metrics
    surface_metrics = {}
    
    for surface in SURFACES:
        surface_idx = test_df['surface'] == surface
        
        if sum(surface_idx) > 0:
            y_true_surface = y_test[surface_idx]
            y_pred_surface = y_pred[surface_idx]
            
            if len(y_true_surface) > 0:
                accuracy_surface = accuracy_score(y_true_surface, y_pred_surface)
                precision_surface = precision_score(y_true_surface, y_pred_surface, zero_division=0)
                recall_surface = recall_score(y_true_surface, y_pred_surface, zero_division=0)
                f1_surface = f1_score(y_true_surface, y_pred_surface, zero_division=0)
                
                surface_metrics[surface] = {
                    'accuracy': accuracy_surface,
                    'precision': precision_surface,
                    'recall': recall_surface,
                    'f1': f1_surface,
                    'count': sum(surface_idx)
                }
                
                logger.info(f"Metrics for {surface} surface (n={sum(surface_idx)}):")
                logger.info(f"  Accuracy: {accuracy_surface:.4f}")
                logger.info(f"  Precision: {precision_surface:.4f}")
                logger.info(f"  Recall: {recall_surface:.4f}")
                logger.info(f"  F1 Score: {f1_surface:.4f}")
    
    # Compile all metrics
    metrics = {
        'overall': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'count': len(y_test)
        },
        'by_surface': surface_metrics
    }
    
    if progress_tracker:
        progress_tracker.update("Model evaluation complete")
    
    return metrics


def plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray, 
                          output_path: Optional[Union[str, Path]] = None,
                          progress_tracker: Optional[ProgressTracker] = None) -> None:
    """
    Plot and save confusion matrix.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        output_path: Path to save the plot
        progress_tracker: Optional progress tracker
    """
    logger.info("Plotting confusion matrix")
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    
    # Plot with percentages
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {output_path}")
    
    plt.close()
    
    if progress_tracker:
        progress_tracker.update("Confusion matrix plotting complete")


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
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Plot horizontal bar chart
    plt.barh(
        list(reversed(list(sorted_importances.keys()))), 
        list(reversed(list(sorted_importances.values())))
    )
    
    plt.xlabel('Importance (Gain)')
    plt.title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {output_path}")
    
    plt.close()
    
    if progress_tracker:
        progress_tracker.update("Feature importance plotting complete")


def plot_shap_values(model: xgb.Booster, X_test: np.ndarray, feature_cols: List[str], 
                    top_n: int = 20, output_path: Optional[Union[str, Path]] = None,
                    progress_tracker: Optional[ProgressTracker] = None) -> None:
    """
    Plot and save SHAP values.
    
    Args:
        model: Trained XGBoost model
        X_test: Test features
        feature_cols: List of feature column names
        top_n: Number of top features to plot
        output_path: Path to save the plot
        progress_tracker: Optional progress tracker
    """
    logger.info("Calculating and plotting SHAP values")
    
    try:
        # Create explainer
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values (limit to 1000 samples for speed if dataset is large)
        if X_test.shape[0] > 1000:
            logger.info(f"Using {1000} random samples for SHAP calculation")
            indices = np.random.choice(X_test.shape[0], 1000, replace=False)
            X_sample = X_test[indices]
        else:
            X_sample = X_test
        
        # Create DMatrix for SHAP
        dmatrix = xgb.DMatrix(X_sample, feature_names=feature_cols)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(dmatrix)
        
        # Plot SHAP values
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, 
            X_sample, 
            feature_names=feature_cols, 
            plot_type="bar", 
            max_display=top_n, 
            show=False
        )
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved SHAP values plot to {output_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting SHAP values: {e}")
    
    if progress_tracker:
        progress_tracker.update("SHAP analysis complete")


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
        else:
            return obj
    
    metrics_json = convert_numpy_types(metrics)
    
    # Add timestamp
    metrics_json['timestamp'] = datetime.now().isoformat()
    
    # Save to file
    import json
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


def main():
    """Train and evaluate a tennis match prediction model."""
    start_time = time.time()
    
    # Define total steps for progress tracking
    total_steps = 11
    progress_tracker = ProgressTracker(total_steps)
    
    # Step 1: Load data
    logger.info(f"Step 1/{total_steps}: Loading data...")
    df = load_data(INPUT_FILE, progress_tracker)
    
    # Step 2: Create time-based split
    logger.info(f"Step 2/{total_steps}: Creating time-based train-test split...")
    train_df, test_df = create_time_based_split(df, TEST_SIZE, progress_tracker)
    
    # Step 3: Get feature columns
    logger.info(f"Step 3/{total_steps}: Identifying feature columns...")
    feature_cols = get_feature_columns(df, progress_tracker)
    
    # Step 4: Prepare features and labels
    logger.info(f"Step 4/{total_steps}: Preparing features and labels...")
    X_train, X_test, y_train, y_test = prepare_features(train_df, test_df, feature_cols, progress_tracker)
    
    # Step 5: Train model
    logger.info(f"Step 5/{total_steps}: Training model...")
    model, importances = train_model(X_train, y_train, feature_cols, MODEL_PARAMS, progress_tracker)
    
    # Step 6: Evaluate model
    logger.info(f"Step 6/{total_steps}: Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test, test_df, feature_cols, progress_tracker)
    
    # Step 7: Plot confusion matrix
    logger.info(f"Step 7/{total_steps}: Plotting confusion matrix...")
    # Convert probabilities to binary predictions
    dtest = xgb.DMatrix(X_test, feature_names=feature_cols)
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)
    plot_confusion_matrix(y_test, y_pred, PLOTS_DIR / "confusion_matrix.png", progress_tracker)
    
    # Step 8: Plot feature importance
    logger.info(f"Step 8/{total_steps}: Plotting feature importance...")
    plot_feature_importance(importances, 20, PLOTS_DIR / "feature_importance.png", progress_tracker)
    
    # Step 9: Plot SHAP values
    logger.info(f"Step 9/{total_steps}: Plotting SHAP values...")
    plot_shap_values(model, X_test, feature_cols, 20, PLOTS_DIR / "shap_values.png", progress_tracker)
    
    # Step 10: Evaluate player position bias
    logger.info(f"Step 10/{total_steps}: Evaluating player position bias...")
    bias_metrics = evaluate_player_position_bias(model, test_df, feature_cols, progress_tracker)
    metrics['player_position_bias'] = bias_metrics
    
    # Step 11: Save model and metrics
    logger.info(f"Step 11/{total_steps}: Saving model and metrics...")
    model.save_model(str(MODEL_OUTPUT))
    logger.info(f"Model saved to {MODEL_OUTPUT}")
    
    # Save metrics
    save_metrics(metrics, METRICS_OUTPUT, progress_tracker)
    
    # Print final message
    elapsed_time = time.time() - start_time
    logger.info(f"Training and evaluation completed in {elapsed_time:.2f} seconds")
    logger.info(f"Model accuracy: {metrics['overall']['accuracy']:.4f}")
    
    # Return metrics for potential further use
    return metrics


if __name__ == "__main__":
    main() 