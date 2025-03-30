import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import shap
import logging
from datetime import datetime
import json
import time
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "predictor" / "output"

# Create necessary directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Input and output files
INPUT_FILE = DATA_DIR / "features_v2.csv"
MODEL_FILE = MODELS_DIR / "tennis_predictor_v2.xgb"
FEATURE_IMPORTANCE_PLOT = OUTPUT_DIR / "v2_feature_importance.png"
SHAP_PLOT = OUTPUT_DIR / "v2_shap.png"
TRAINING_METRICS = OUTPUT_DIR / "v2_metrics.txt"
CONFUSION_MATRIX_PLOT = OUTPUT_DIR / "v2_confusion_matrix.png"

class ProgressTracker:
    """Class to track progress across multiple steps."""
    
    def __init__(self, total_steps: int, description: str = "Model Training"):
        """
        Initialize the progress tracker.
        
        Args:
            total_steps: Total number of steps in the process
            description: Description of the process
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
        self.step_start_time = time.time()
        
    def update(self, step_description: str) -> None:
        """
        Update progress to the next step.
        
        Args:
            step_description: Description of the current step
        """
        self.current_step += 1
        progress_pct = (self.current_step / self.total_steps) * 100
        
        # Calculate time statistics
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        step_time = current_time - self.step_start_time
        self.step_start_time = current_time
        
        # Estimate remaining time
        if self.current_step > 0:
            avg_step_time = elapsed_time / self.current_step
            remaining_steps = self.total_steps - self.current_step
            remaining_time = avg_step_time * remaining_steps
            
            # Format time strings
            elapsed_str = self._format_time(elapsed_time)
            remaining_str = self._format_time(remaining_time)
            step_str = self._format_time(step_time)
            
            logger.info(f"{self.description} Progress: {progress_pct:.1f}% complete - Step {self.current_step}/{self.total_steps}")
            logger.info(f"Current step: {step_description} (took {step_str})")
            logger.info(f"Elapsed time: {elapsed_str}, Estimated time remaining: {remaining_str}")
        else:
            logger.info(f"{self.description} Progress: {progress_pct:.1f}% complete - Step {self.current_step}/{self.total_steps}")
            logger.info(f"Current step: {step_description}")
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds into a readable string."""
        minutes, seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

# Try to detect GPU
def detect_gpu_availability() -> bool:
    """Check if GPU is available for XGBoost."""
    try:
        # Create a small test XGBoost model with GPU
        test_data = np.random.rand(10, 5)
        test_labels = np.random.randint(0, 2, 10)
        
        # Set up a minimal GPU model
        test_model = xgb.XGBClassifier(tree_method='hist', device='cuda', n_estimators=1)
        
        # Try to train it
        test_model.fit(test_data, test_labels)
        logger.info("GPU is available and will be used for training")
        return True
    except Exception as e:
        logger.info(f"GPU is not available (reason: {str(e)}). Using CPU instead")
        return False

# Detect GPU availability
GPU_AVAILABLE = detect_gpu_availability()
# Use modern XGBoost GPU configuration
TREE_METHOD = 'hist'  # Always use hist for efficiency
DEVICE = 'cuda' if GPU_AVAILABLE else 'cpu'

def load_data(progress_tracker: Optional[ProgressTracker] = None) -> pd.DataFrame:
    """
    Load the features dataset.
    
    Args:
        progress_tracker: Optional progress tracker to update
    
    Returns:
        Loaded DataFrame
    """
    if progress_tracker:
        progress_tracker.update("Loading data")
    
    logger.info(f"Loading data from {INPUT_FILE}...")
    
    # Load the CSV
    df = pd.read_csv(INPUT_FILE)
    
    # Convert date column to datetime
    if 'tourney_date' in df.columns:
        df['tourney_date'] = pd.to_datetime(df['tourney_date'])
    
    # Sort by date to ensure chronological order
    df = df.sort_values('tourney_date').reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} samples from {df['tourney_date'].min().date()} to {df['tourney_date'].max().date()}")
    logger.info(f"Class distribution: {df['result'].value_counts().to_dict()}")
    
    return df

def create_time_based_split(df: pd.DataFrame, val_date: str = '2022-01-01', test_date: str = '2023-01-01',
                          progress_tracker: Optional[ProgressTracker] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create a time-based split of the data with separate validation and test sets.
    
    Args:
        df: DataFrame to split
        val_date: Start date for validation set
        test_date: Start date for test set
        progress_tracker: Optional progress tracker to update
    
    Returns:
        Training, validation, and test DataFrames
    """
    if progress_tracker:
        progress_tracker.update("Creating time-based data split")
    
    # Make sure date column is datetime
    if not pd.api.types.is_datetime64_dtype(df['tourney_date']):
        df['tourney_date'] = pd.to_datetime(df['tourney_date'])
    
    # Split the data by date
    train_df = df[df['tourney_date'] < pd.to_datetime(val_date)]
    val_df = df[(df['tourney_date'] >= pd.to_datetime(val_date)) & 
               (df['tourney_date'] < pd.to_datetime(test_date))]
    test_df = df[df['tourney_date'] >= pd.to_datetime(test_date)]
    
    logger.info(f"Training set: {len(train_df)} matches ({train_df['tourney_date'].min().date()} to {train_df['tourney_date'].max().date()})")
    logger.info(f"Validation set: {len(val_df)} matches ({val_df['tourney_date'].min().date() if not val_df.empty else 'N/A'} to {val_df['tourney_date'].max().date() if not val_df.empty else 'N/A'})")
    logger.info(f"Test set: {len(test_df)} matches ({test_df['tourney_date'].min().date() if not test_df.empty else 'N/A'} to {test_df['tourney_date'].max().date() if not test_df.empty else 'N/A'})")
    
    return train_df, val_df, test_df

def get_feature_columns(df: pd.DataFrame, progress_tracker: Optional[ProgressTracker] = None) -> List[str]:
    """
    Get the list of feature columns to use for training.
    
    Args:
        df: DataFrame with features
        progress_tracker: Optional progress tracker to update
    
    Returns:
        List of feature column names
    """
    if progress_tracker:
        progress_tracker.update("Identifying feature columns")
    
    # These are the key features we identified
    diff_features = [
        'player_elo_diff',
        'win_rate_5_diff',
        'win_streak_diff',
        'loss_streak_diff'
    ]
    
    # Add surface-specific features if they exist
    for surface in ['Hard', 'Clay', 'Grass', 'Carpet']:
        for suffix in ['_5_diff', '_overall_diff']:
            feature = f'win_rate_{surface}{suffix}'
            if feature in df.columns:
                diff_features.append(feature)
    
    # Filter to features that exist in the dataframe
    feature_cols = [col for col in diff_features if col in df.columns]
    
    logger.info(f"Using {len(feature_cols)} features: {feature_cols}")
    return feature_cols

def prepare_features(df: pd.DataFrame, feature_cols: List[str], 
                    progress_tracker: Optional[ProgressTracker] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare features and target for model training.
    
    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
        progress_tracker: Optional progress tracker to update
    
    Returns:
        Tuple of feature matrix and target vector
    """
    if progress_tracker:
        progress_tracker.update("Preparing feature matrices")
    
    # Check if all feature columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}")
        feature_cols = [col for col in feature_cols if col in df.columns]
    
    # Get features - do not impute, let XGBoost handle missing values
    X = df[feature_cols].values
    
    # Get target
    y = df['result'].values
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target vector shape: {y.shape}")
    logger.info(f"Target distribution: {np.bincount(y.astype(int))}")
    
    return X, y

class XGBoostProgressCallback(xgb.callback.TrainingCallback):
    """Custom callback to track XGBoost training progress."""
    
    def __init__(self, n_estimators: int = 100, report_interval: int = 10):
        """
        Initialize the callback.
        
        Args:
            n_estimators: Total number of trees to train
            report_interval: How often to report progress (in trees)
        """
        self.n_estimators = n_estimators
        self.report_interval = report_interval
        self.start_time = time.time()
        self.pbar = tqdm(total=n_estimators, desc="Training XGBoost", unit="tree")
        
    def after_iteration(self, model, epoch, evals_log):
        """Called after each boosting iteration."""
        # Update progress bar
        self.pbar.update(1)
        
        # Report progress at specified intervals
        if (epoch + 1) % self.report_interval == 0 or epoch + 1 == self.n_estimators:
            elapsed_time = time.time() - self.start_time
            trees_per_second = (epoch + 1) / elapsed_time
            
            # Estimate remaining time
            remaining_trees = self.n_estimators - (epoch + 1)
            remaining_time = remaining_trees / trees_per_second if trees_per_second > 0 else 0
            
            # Calculate progress percentage
            progress_pct = ((epoch + 1) / self.n_estimators) * 100
            
            # Log progress
            logger.info(f"XGBoost Progress: {progress_pct:.1f}% ({epoch+1}/{self.n_estimators} trees)")
            logger.info(f"Trees per second: {trees_per_second:.2f}, Estimated time remaining: {self._format_time(remaining_time)}")
            
            # If we have evaluation results, log them
            if evals_log:
                for eval_name, eval_metrics in evals_log.items():
                    for metric_name, metric_values in eval_metrics.items():
                        latest_value = metric_values[-1]
                        logger.info(f"  {eval_name}-{metric_name}: {latest_value:.6f}")
        
        # Check for early stopping
        if hasattr(model, 'best_iteration') and model.best_iteration < epoch:
            # Close progress bar if early stopping
            self.pbar.close()
            logger.info(f"Early stopping at iteration {model.best_iteration}")
            
        return False  # Continue training
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds into a readable string."""
        minutes, seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
            
    def __del__(self):
        """Close progress bar on destruction."""
        if hasattr(self, 'pbar'):
            self.pbar.close()

def train_model(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
               feature_names: List[str], progress_tracker: Optional[ProgressTracker] = None) -> Tuple[xgb.XGBClassifier, Dict]:
    """
    Train the XGBoost model with anti-overfitting settings.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        feature_names: List of feature names
        progress_tracker: Optional progress tracker to update
    
    Returns:
        Tuple of trained model and feature importance dictionary
    """
    if progress_tracker:
        progress_tracker.update("Training XGBoost model")
    
    logger.info("Training XGBoost model...")
    
    # XGBoost parameters tuned to prevent overfitting
    n_estimators = 500
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'learning_rate': 0.01,
        'max_depth': 4,           # Shallow trees to prevent overfitting
        'min_child_weight': 3,     # Higher value prevents overfitting
        'subsample': 0.7,          # Use 70% of data per tree
        'colsample_bytree': 0.7,   # Use 70% of features per tree
        'gamma': 0.1,              # Minimum loss reduction for split
        'reg_alpha': 0.1,          # L1 regularization
        'reg_lambda': 1.0,         # L2 regularization
        'tree_method': TREE_METHOD,
        'device': DEVICE,
        'n_estimators': n_estimators,
        'early_stopping_rounds': 50,
        'random_state': 42
    }
    
    # Create and train model
    model = xgb.XGBClassifier(**params)
    
    # Progress callback
    progress_callback = XGBoostProgressCallback(n_estimators=n_estimators, report_interval=20)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[progress_callback],
        verbose=False  # We'll use our custom callback instead
    )
    
    # Get feature importance
    importance = model.feature_importances_
    feature_importance = dict(zip(feature_names, importance))
    
    return model, feature_importance

def evaluate_model(model: xgb.XGBClassifier, X_test: np.ndarray, y_test: np.ndarray, 
                 progress_tracker: Optional[ProgressTracker] = None) -> Dict[str, float]:
    """
    Evaluate model performance.
    
    Args:
        model: Trained XGBoost model
        X_test: Test features
        y_test: Test targets
        progress_tracker: Optional progress tracker to update
    
    Returns:
        Dictionary of performance metrics
    """
    if progress_tracker:
        progress_tracker.update("Evaluating model performance")
    
    logger.info("Evaluating model on test set...")
    
    # Make predictions
    logger.info("Making predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    logger.info("Calculating performance metrics...")
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_pred_proba)
    }
    
    return metrics

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path, 
                        progress_tracker: Optional[ProgressTracker] = None) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_path: Path to save the plot
        progress_tracker: Optional progress tracker to update
    """
    if progress_tracker:
        progress_tracker.update("Plotting confusion matrix")
    
    logger.info("Plotting confusion matrix...")
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Player 2 Wins', 'Player 1 Wins'],
               yticklabels=['Player 2 Wins', 'Player 1 Wins'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_feature_importance(feature_importance: Dict[str, float], output_path: Path, 
                          progress_tracker: Optional[ProgressTracker] = None) -> None:
    """
    Plot feature importance.
    
    Args:
        feature_importance: Dictionary of feature importances
        output_path: Path to save the plot
        progress_tracker: Optional progress tracker to update
    """
    if progress_tracker:
        progress_tracker.update("Plotting feature importance")
    
    logger.info("Plotting feature importance...")
    
    # Create dataframe from feature importance dict
    importance_df = pd.DataFrame({
        'feature': list(feature_importance.keys()),
        'importance': list(feature_importance.values())
    }).sort_values('importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance_df, x='importance', y='feature')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_shap_values(model: xgb.XGBClassifier, X_test: np.ndarray, feature_names: List[str], output_path: Path, 
                    progress_tracker: Optional[ProgressTracker] = None) -> None:
    """
    Plot SHAP values to understand feature impact.
    
    Args:
        model: Trained XGBoost model
        X_test: Test features
        feature_names: List of feature names
        output_path: Path to save the plot
        progress_tracker: Optional progress tracker to update
    """
    if progress_tracker:
        progress_tracker.update("Calculating and plotting SHAP values")
    
    logger.info("Calculating SHAP values...")
    
    # Limit number of samples for SHAP analysis
    max_samples = min(500, X_test.shape[0])
    X_shap = X_test[:max_samples]
    
    # Calculate SHAP values with progress tracking
    logger.info(f"Analyzing {X_shap.shape[0]} samples for SHAP values...")
    explainer = shap.Explainer(model)
    
    # Create progress bar for SHAP calculation
    with tqdm(total=1, desc="Calculating SHAP values") as pbar:
        shap_values = explainer(X_shap)
        pbar.update(1)
    
    # Plot
    logger.info("Generating SHAP plot...")
    plt.figure(figsize=(12, 8))
    shap.plots.beeswarm(shap_values, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_metrics(metrics: Dict[str, float], feature_importance: Dict[str, float], output_path: Path, 
               progress_tracker: Optional[ProgressTracker] = None) -> None:
    """
    Save model metrics and feature importance to a file.
    
    Args:
        metrics: Dictionary of performance metrics
        feature_importance: Dictionary of feature importances
        output_path: Path to save the metrics
        progress_tracker: Optional progress tracker to update
    """
    if progress_tracker:
        progress_tracker.update("Saving model metrics and importance")
    
    logger.info(f"Saving metrics to {output_path}...")
    
    with open(output_path, 'w') as f:
        f.write("Tennis Match Prediction Model v2\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Training Metrics:\n")
        f.write("-" * 40 + "\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        
        f.write("\nFeature Importance:\n")
        f.write("-" * 40 + "\n")
        
        for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{feature}: {importance:.6f}\n")

def evaluate_player_position_bias(model: xgb.XGBClassifier, test_df: pd.DataFrame, feature_cols: List[str], 
                               progress_tracker: Optional[ProgressTracker] = None) -> None:
    """
    Evaluate if the model has player position bias by checking prediction symmetry.
    
    This confirms that swapping player1 and player2 consistently flips the prediction.
    
    Args:
        model: Trained XGBoost model
        test_df: Test DataFrame
        feature_cols: List of feature column names
        progress_tracker: Optional progress tracker to update
    """
    if progress_tracker:
        progress_tracker.update("Evaluating player position bias")
    
    logger.info("Evaluating player position bias...")
    
    # Group rows by match_id
    match_ids = test_df['match_id'].unique()
    
    # Sample some matches for analysis
    sample_size = min(100, len(match_ids))
    sample_match_ids = np.random.choice(match_ids, sample_size, replace=False)
    
    # Get sample data
    sample_df = test_df[test_df['match_id'].isin(sample_match_ids)].copy()
    
    # Get predictions
    X_sample = sample_df[feature_cols].values
    sample_df['pred_proba'] = model.predict_proba(X_sample)[:, 1]
    
    # Group by match_id
    symmetry_results = []
    
    # Process each match with progress bar
    for match_id in tqdm(sample_match_ids, desc="Checking prediction symmetry", unit="match"):
        match_rows = sample_df[sample_df['match_id'] == match_id]
        
        if len(match_rows) == 2:
            # Get the two perspectives
            p1_row = match_rows[match_rows['result'] == 1].iloc[0]
            p2_row = match_rows[match_rows['result'] == 0].iloc[0]
            
            # Check prediction symmetry
            p1_win_prob = p1_row['pred_proba']
            p2_lose_prob = p2_row['pred_proba']
            
            # These should sum to approximately 1
            symmetry_sum = p1_win_prob + (1 - p2_lose_prob)
            
            symmetry_results.append({
                'match_id': match_id,
                'p1_win_prob': p1_win_prob,
                'p2_lose_prob': p2_lose_prob,
                'probability_sum': symmetry_sum
            })
    
    # Analyze results
    if symmetry_results:
        symmetry_df = pd.DataFrame(symmetry_results)
        avg_sum = symmetry_df['probability_sum'].mean()
        min_sum = symmetry_df['probability_sum'].min()
        max_sum = symmetry_df['probability_sum'].max()
        
        logger.info(f"Player position bias check - prediction probabilities should sum to 1")
        logger.info(f"Average sum: {avg_sum:.4f}, Min: {min_sum:.4f}, Max: {max_sum:.4f}")
        
        # Check if there's significant bias
        if abs(avg_sum - 1.0) > 0.05:
            logger.warning(f"Player position bias detected: Average sum = {avg_sum:.4f}")
        else:
            logger.info("No significant player position bias detected")

def main():
    """Main function to train and evaluate the model."""
    start_time = datetime.now()
    logger.info(f"Starting model training at {start_time}")
    
    # Define the total steps in the process for progress tracking
    total_steps = 12
    tracker = ProgressTracker(total_steps, "Tennis Model Training")
    
    # Print info about hardware being used
    logger.info(f"Using {'GPU' if GPU_AVAILABLE else 'CPU'} for training")
    
    # 1. Load data
    df = load_data(tracker)
    
    # 2. Create time-based train/val/test split
    train_df, val_df, test_df = create_time_based_split(df, progress_tracker=tracker)
    
    # 3. Get feature columns
    feature_cols = get_feature_columns(df, tracker)
    
    # 4. Prepare features
    X_train, y_train = prepare_features(train_df, feature_cols, tracker)
    X_val, y_val = prepare_features(val_df, feature_cols)
    X_test, y_test = prepare_features(test_df, feature_cols)
    
    # 5. Train model
    model, feature_importance = train_model(X_train, y_train, X_val, y_val, feature_cols, tracker)
    
    # 6. Evaluate model
    metrics = evaluate_model(model, X_test, y_test, tracker)
    
    logger.info("Model performance metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # 7. Plot confusion matrix
    y_pred = model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, CONFUSION_MATRIX_PLOT, tracker)
    
    # 8. Plot feature importance
    plot_feature_importance(feature_importance, FEATURE_IMPORTANCE_PLOT, tracker)
    
    # 9. Plot SHAP values
    plot_shap_values(model, X_test, feature_cols, SHAP_PLOT, tracker)
    
    # 10. Save metrics and model
    save_metrics(metrics, feature_importance, TRAINING_METRICS, tracker)
    
    # Save model
    tracker.update("Saving model")
    logger.info(f"Saving model to {MODEL_FILE}...")
    model.save_model(MODEL_FILE)
    
    # 11. Check for player position bias
    evaluate_player_position_bias(model, test_df, feature_cols, tracker)
    
    # 12. Finish up
    tracker.update("Completing training process")
    
    # Log completion
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Model training completed in {duration}")
    logger.info(f"Model saved to: {MODEL_FILE}")
    logger.info(f"Metrics saved to: {TRAINING_METRICS}")

if __name__ == "__main__":
    main() 