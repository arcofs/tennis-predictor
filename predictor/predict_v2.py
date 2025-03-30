import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import logging
from datetime import datetime
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

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Input and output files
FEATURES_FILE = DATA_DIR / "features_v2.csv"
MODEL_FILE = MODELS_DIR / "tennis_predictor_v2.xgb"
PREDICTIONS_FILE = OUTPUT_DIR / "v2_predictions.csv"
RESULTS_SUMMARY_FILE = OUTPUT_DIR / "v2_results_summary.txt"
CONFUSION_MATRIX_PLOT = OUTPUT_DIR / "v2_prediction_confusion_matrix.png"
ACCURACY_SURFACE_PLOT = OUTPUT_DIR / "v2_accuracy_by_surface.png"

# Standard surface definitions
SURFACE_HARD = 'Hard'
SURFACE_CLAY = 'Clay'
SURFACE_GRASS = 'Grass'
SURFACE_CARPET = 'Carpet'
STANDARD_SURFACES = [SURFACE_HARD, SURFACE_CLAY, SURFACE_GRASS, SURFACE_CARPET]

class ProgressTracker:
    """Class to track progress across multiple steps."""
    
    def __init__(self, total_steps: int, description: str = "Prediction Process"):
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

def load_model(model_path: Path = MODEL_FILE, progress_tracker: Optional[ProgressTracker] = None) -> xgb.XGBClassifier:
    """
    Load the trained XGBoost model.
    
    Args:
        model_path: Path to the model file
        progress_tracker: Optional progress tracker
        
    Returns:
        Loaded XGBoost model
    """
    if progress_tracker:
        progress_tracker.update("Loading model")
    
    logger.info(f"Loading model from {model_path}...")
    
    try:
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def load_test_data(features_file: Path = FEATURES_FILE, test_date: str = '2023-01-01', 
                  progress_tracker: Optional[ProgressTracker] = None) -> pd.DataFrame:
    """
    Load test data from the features file, filtering to matches after the test_date.
    
    Args:
        features_file: Path to features CSV file
        test_date: Date to filter test data, only using matches on or after this date
        progress_tracker: Optional progress tracker
        
    Returns:
        DataFrame with test data
    """
    if progress_tracker:
        progress_tracker.update("Loading test data")
    
    logger.info(f"Loading test data from {features_file}...")
    
    try:
        # Load the CSV file
        df = pd.read_csv(features_file)
        
        # Convert date to datetime
        if 'tourney_date' in df.columns:
            df['tourney_date'] = pd.to_datetime(df['tourney_date'])
            
            # Filter to matches on or after test_date
            test_df = df[df['tourney_date'] >= pd.to_datetime(test_date)]
            logger.info(f"Filtered to {len(test_df)} samples from {test_date} onwards")
            
            return test_df
        else:
            logger.warning("No tourney_date column found, returning all data")
            return df
    
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        raise

def get_feature_columns(df: pd.DataFrame, progress_tracker: Optional[ProgressTracker] = None) -> List[str]:
    """
    Get the list of feature columns to use for prediction.
    
    Args:
        df: DataFrame with features
        progress_tracker: Optional progress tracker
        
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
    for surface in STANDARD_SURFACES:
        for suffix in ['_5_diff', '_overall_diff']:
            feature = f'win_rate_{surface}{suffix}'
            if feature in df.columns:
                diff_features.append(feature)
    
    # Filter to features that exist in the dataframe
    feature_cols = [col for col in diff_features if col in df.columns]
    
    logger.info(f"Using {len(feature_cols)} features: {feature_cols}")
    return feature_cols

def make_predictions(model: xgb.XGBClassifier, df: pd.DataFrame, feature_cols: List[str], 
                    progress_tracker: Optional[ProgressTracker] = None) -> pd.DataFrame:
    """
    Make predictions on the test data.
    
    Args:
        model: Trained XGBoost model
        df: Test data with features
        feature_cols: List of feature columns to use
        progress_tracker: Optional progress tracker
        
    Returns:
        DataFrame with added predictions
    """
    if progress_tracker:
        progress_tracker.update("Making predictions")
    
    logger.info(f"Making predictions on {len(df)} samples...")
    
    # Check if all required features exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing features: {missing_cols}")
        feature_cols = [col for col in feature_cols if col in df.columns]
    
    # Prepare feature matrix
    X = df[feature_cols].values
    
    # Make predictions with progress bar
    batch_size = 10000  # Process in batches to show progress
    n_samples = X.shape[0]
    n_batches = int(np.ceil(n_samples / batch_size))
    
    y_pred_proba = np.zeros(n_samples)
    
    # Create a progress bar for prediction
    with tqdm(total=n_samples, desc="Making predictions", unit="sample") as pbar:
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            batch_X = X[start_idx:end_idx]
            batch_pred_proba = model.predict_proba(batch_X)[:, 1]
            
            y_pred_proba[start_idx:end_idx] = batch_pred_proba
            pbar.update(end_idx - start_idx)
    
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Add predictions to dataframe
    df_pred = df.copy()
    df_pred['predicted_probability'] = y_pred_proba
    df_pred['predicted_result'] = y_pred
    
    # Add model prediction about who will win
    logger.info("Determining predicted winners...")
    df_pred['predicted_winner_id'] = df_pred.apply(
        lambda row: row['player1_id'] if row['predicted_result'] == 1 else row['player2_id'],
        axis=1
    )
    
    # If we have actual results, calculate accuracy
    if 'result' in df_pred.columns:
        df_pred['is_correct'] = df_pred['predicted_result'] == df_pred['result']
        accuracy = df_pred['is_correct'].mean()
        logger.info(f"Overall prediction accuracy: {accuracy:.4f}")
        
        # Calculate metrics by surface if available
        if 'surface' in df_pred.columns:
            surface_metrics = {}
            
            logger.info("Calculating accuracy by surface:")
            for surface in df_pred['surface'].unique():
                surface_df = df_pred[df_pred['surface'] == surface]
                surface_acc = surface_df['is_correct'].mean()
                surface_count = len(surface_df)
                surface_metrics[surface] = {'accuracy': surface_acc, 'count': surface_count}
                logger.info(f"  {surface}: {surface_acc:.4f} (n={surface_count})")
    
    return df_pred

def calculate_match_level_metrics(df_pred: pd.DataFrame, progress_tracker: Optional[ProgressTracker] = None) -> Dict:
    """
    Calculate metrics at the match level, not the player perspective level.
    
    Args:
        df_pred: DataFrame with predictions for both perspectives of each match
        progress_tracker: Optional progress tracker
        
    Returns:
        Dictionary with match-level metrics
    """
    if progress_tracker:
        progress_tracker.update("Calculating match-level metrics")
    
    logger.info("Calculating match-level metrics...")
    
    # Group by match_id to get both perspectives of each match
    metrics = {}
    match_ids = df_pred['match_id'].unique()
    logger.info(f"Analyzing {len(match_ids)} unique matches")
    
    correct_predictions = 0
    total_matches = 0
    
    # Process matches with progress bar
    for match_id in tqdm(match_ids, desc="Processing matches", unit="match"):
        match_rows = df_pred[df_pred['match_id'] == match_id]
        
        # Need both perspectives (player1 wins and player1 loses)
        if len(match_rows) == 2:
            total_matches += 1
            
            # Get the actual winner (same for both rows)
            actual_winner = match_rows['player1_id'].iloc[0] if match_rows['result'].iloc[0] == 1 else match_rows['player2_id'].iloc[0]
            
            # Get perspective where player1 is the winner
            p1_wins_row = match_rows[match_rows['result'] == 1].iloc[0]
            p1_wins_prob = p1_wins_row['predicted_probability']
            
            # Get perspective where player2 is the winner
            p2_wins_row = match_rows[match_rows['result'] == 0].iloc[0]
            p2_wins_prob = 1 - p2_wins_row['predicted_probability']  # Probability that player2 wins
            
            # Average the probabilities from both perspectives
            avg_prob = (p1_wins_prob + p2_wins_prob) / 2
            
            # Make final prediction based on average probability
            predicted_winner = p1_wins_row['player1_id'] if avg_prob >= 0.5 else p2_wins_row['player2_id']
            
            # Check if prediction is correct
            is_correct = predicted_winner == actual_winner
            if is_correct:
                correct_predictions += 1
    
    # Calculate match-level accuracy
    if total_matches > 0:
        match_accuracy = correct_predictions / total_matches
        logger.info(f"Match-level accuracy: {match_accuracy:.4f} ({correct_predictions}/{total_matches})")
        metrics['match_accuracy'] = match_accuracy
    else:
        logger.warning("No complete matches found for match-level evaluation")
        metrics['match_accuracy'] = None
    
    return metrics

def plot_results(df_pred: pd.DataFrame, progress_tracker: Optional[ProgressTracker] = None) -> None:
    """
    Plot analysis of prediction results.
    
    Args:
        df_pred: DataFrame with predictions
        progress_tracker: Optional progress tracker
    """
    if progress_tracker:
        progress_tracker.update("Plotting analysis")
    
    # Only proceed if we have accuracy data
    if 'is_correct' not in df_pred.columns:
        logger.warning("No accuracy data available for plotting")
        return
    
    # Plot accuracy by surface
    if 'surface' in df_pred.columns:
        logger.info("Plotting accuracy by surface...")
        
        plt.figure(figsize=(10, 6))
        surface_accuracy = df_pred.groupby('surface')['is_correct'].mean()
        surface_counts = df_pred.groupby('surface').size()
        
        # Plot as bar chart
        ax = surface_accuracy.plot(kind='bar', color='skyblue')
        plt.title('Prediction Accuracy by Surface')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.0)
        
        # Add counts
        for i, surface in enumerate(surface_accuracy.index):
            ax.text(i, surface_accuracy[surface] + 0.02, f"n={surface_counts[surface]}", ha='center')
        
        plt.tight_layout()
        plt.savefig(ACCURACY_SURFACE_PLOT)
        plt.close()
        logger.info(f"Saved surface accuracy plot to {ACCURACY_SURFACE_PLOT}")
    
    # Plot confusion matrix
    if 'result' in df_pred.columns and 'predicted_result' in df_pred.columns:
        logger.info("Plotting confusion matrix...")
        
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(df_pred['result'], df_pred['predicted_result'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Player 2 Wins', 'Player 1 Wins'],
                   yticklabels=['Player 2 Wins', 'Player 1 Wins'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(CONFUSION_MATRIX_PLOT)
        plt.close()
        logger.info(f"Saved confusion matrix to {CONFUSION_MATRIX_PLOT}")

def save_results(df_pred: pd.DataFrame, metrics: Dict, progress_tracker: Optional[ProgressTracker] = None) -> None:
    """
    Save prediction results and metrics.
    
    Args:
        df_pred: DataFrame with predictions
        metrics: Dictionary with calculated metrics
        progress_tracker: Optional progress tracker
    """
    if progress_tracker:
        progress_tracker.update("Saving results")
    
    # Save predictions to CSV
    logger.info(f"Saving predictions to {PREDICTIONS_FILE}...")
    
    # Select columns to save
    cols_to_save = [
        'match_id', 'tourney_date', 'surface',
        'player1_id', 'player2_id',
        'predicted_probability', 'predicted_result', 'predicted_winner_id'
    ]
    
    # Add actual results if available
    if 'result' in df_pred.columns:
        cols_to_save.append('result')
    
    if 'is_correct' in df_pred.columns:
        cols_to_save.append('is_correct')
    
    # Filter to columns that exist in the dataframe
    save_cols = [col for col in cols_to_save if col in df_pred.columns]
    
    # Save to CSV
    df_pred[save_cols].to_csv(PREDICTIONS_FILE, index=False)
    logger.info(f"Saved predictions to {PREDICTIONS_FILE}")
    
    # Save summary to text file
    logger.info(f"Saving metrics summary to {RESULTS_SUMMARY_FILE}...")
    with open(RESULTS_SUMMARY_FILE, 'w') as f:
        f.write("Tennis Match Prediction Results Summary (v2)\n")
        f.write("=" * 50 + "\n\n")
        
        # Write overall metrics
        f.write("Overall Metrics:\n")
        f.write("-" * 40 + "\n")
        
        if 'is_correct' in df_pred.columns:
            accuracy = df_pred['is_correct'].mean()
            f.write(f"Perspective-level accuracy: {accuracy:.4f}\n")
        
        if 'match_accuracy' in metrics and metrics['match_accuracy'] is not None:
            f.write(f"Match-level accuracy: {metrics['match_accuracy']:.4f}\n")
        
        f.write("\n")
        
        # Write metrics by surface
        if 'surface' in df_pred.columns and 'is_correct' in df_pred.columns:
            f.write("Metrics by Surface:\n")
            f.write("-" * 40 + "\n")
            
            for surface in df_pred['surface'].unique():
                surface_df = df_pred[df_pred['surface'] == surface]
                surface_acc = surface_df['is_correct'].mean()
                surface_count = len(surface_df)
                f.write(f"{surface}: {surface_acc:.4f} (n={surface_count})\n")
    
    logger.info(f"Saved metrics summary to {RESULTS_SUMMARY_FILE}")

def predict_new_match(model: xgb.XGBClassifier, player1_id: int, player2_id: int, surface: str, 
                      player1_features: Dict = None, player2_features: Dict = None,
                      progress_tracker: Optional[ProgressTracker] = None) -> Dict:
    """
    Predict the outcome of a new match between two players.
    
    Args:
        model: Trained XGBoost model
        player1_id: ID of player 1
        player2_id: ID of player 2
        surface: Match surface
        player1_features: Dictionary with features for player 1 (optional)
        player2_features: Dictionary with features for player 2 (optional)
        progress_tracker: Optional progress tracker
        
    Returns:
        Dictionary with prediction results
    """
    if progress_tracker:
        progress_tracker.update("Predicting new match")
    
    logger.info(f"Predicting match between players {player1_id} and {player2_id} on {surface}")
    
    # Create match data
    match_data = {
        'player1_id': player1_id,
        'player2_id': player2_id,
        'surface': surface
    }
    
    # Load source data to get feature values if not provided
    if player1_features is None or player2_features is None:
        try:
            logger.info("Loading source data for feature extraction...")
            source_df = pd.read_csv(FEATURES_FILE)
            # Convert date to datetime for sorting
            if 'tourney_date' in source_df.columns:
                source_df['tourney_date'] = pd.to_datetime(source_df['tourney_date'])
                source_df = source_df.sort_values('tourney_date')
        except Exception as e:
            logger.warning(f"Could not load source data for feature extraction: {e}")
            source_df = None
    
    # Process player1 features
    if player1_features is not None:
        # Use provided features
        for feature, value in player1_features.items():
            match_data[f'player1_{feature}'] = value
    elif source_df is not None:
        # Try to extract from source data (most recent match for player)
        p1_matches = source_df[source_df['player1_id'] == player1_id]
        if not p1_matches.empty:
            p1_latest = p1_matches.iloc[-1]
            for col in p1_latest.index:
                if col.startswith('player1_') and col != 'player1_id':
                    match_data[col] = p1_latest[col]
    
    # Process player2 features
    if player2_features is not None:
        # Use provided features
        for feature, value in player2_features.items():
            match_data[f'player2_{feature}'] = value
    elif source_df is not None:
        # Try to extract from source data (most recent match for player)
        p2_matches = source_df[source_df['player2_id'] == player2_id]
        if not p2_matches.empty:
            p2_latest = p2_matches.iloc[-1]
            for col in p2_latest.index:
                if col.startswith('player2_') and col != 'player2_id':
                    match_data[col] = p2_latest[col]
    
    # Calculate difference features
    if source_df is not None:
        # Get feature columns from source data
        feature_cols = get_feature_columns(source_df)
        
        # Create diff features based on raw player values
        for feature in feature_cols:
            if feature.endswith('_diff'):
                base_feature = feature.replace('_diff', '')
                p1_col = f'player1_{base_feature}'
                p2_col = f'player2_{base_feature}'
                
                if p1_col in match_data and p2_col in match_data:
                    match_data[feature] = match_data[p1_col] - match_data[p2_col]
    
    # Create dataframe for prediction
    match_df = pd.DataFrame([match_data])
    
    # Get feature columns (same as training)
    feature_cols = get_feature_columns(match_df)
    
    # Filter out missing features
    feature_cols = [col for col in feature_cols if col in match_df.columns]
    
    # Fill missing features with NaN - XGBoost will handle them
    for col in feature_cols:
        if col not in match_df.columns:
            match_df[col] = np.nan
    
    # Make prediction
    X = match_df[feature_cols].values
    
    try:
        # Get probability
        win_probability = model.predict_proba(X)[0, 1]
        
        # Determine winner
        predicted_winner_id = player1_id if win_probability >= 0.5 else player2_id
        
        # Create result dictionary
        result = {
            'player1_id': player1_id,
            'player2_id': player2_id,
            'surface': surface,
            'player1_win_probability': float(win_probability),
            'player2_win_probability': float(1 - win_probability),
            'predicted_winner_id': int(predicted_winner_id)
        }
        
        logger.info(f"Prediction: Player 1 ({player1_id}) win probability: {win_probability:.4f}")
        logger.info(f"Predicted winner: Player {'1' if predicted_winner_id == player1_id else '2'} ({predicted_winner_id})")
        
        return result
    
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return {
            'player1_id': player1_id,
            'player2_id': player2_id,
            'surface': surface,
            'error': str(e)
        }

def main():
    """Main function to run predictions."""
    start_time = datetime.now()
    logger.info(f"Starting tennis match prediction with v2 model at {start_time}")
    
    # Define the total steps in the process for progress tracking
    total_steps = 9
    tracker = ProgressTracker(total_steps, "Tennis Prediction")
    
    try:
        # 1. Load the model
        model = load_model(progress_tracker=tracker)
        
        # 2. Load test data
        test_df = load_test_data(progress_tracker=tracker)
        
        # 3. Get feature columns
        feature_cols = get_feature_columns(test_df, progress_tracker=tracker)
        
        # 4. Make predictions
        predictions_df = make_predictions(model, test_df, feature_cols, progress_tracker=tracker)
        
        # 5. Calculate match-level metrics
        metrics = calculate_match_level_metrics(predictions_df, progress_tracker=tracker)
        
        # 6. Plot results
        plot_results(predictions_df, progress_tracker=tracker)
        
        # 7. Save results
        save_results(predictions_df, metrics, progress_tracker=tracker)
        
        # 8. Sample prediction for a specific match
        tracker.update("Making sample prediction")
        sample_player1_id = 104925  # Federer (example)
        sample_player2_id = 105657  # Nadal (example)
        sample_result = predict_new_match(
            model,
            sample_player1_id,
            sample_player2_id,
            SURFACE_HARD
        )
        
        logger.info(f"Sample prediction: {sample_result}")
        
        # 9. Try prediction from opposite perspective to verify symmetry
        tracker.update("Verifying prediction symmetry")
        logger.info("Predicting same match from opposite perspective to verify symmetry:")
        opposite_result = predict_new_match(
            model,
            sample_player2_id,  # Swapped players
            sample_player1_id,
            SURFACE_HARD
        )
        
        logger.info(f"Opposite perspective: {opposite_result}")
        logger.info(f"Probability check (should be symmetric): {sample_result['player1_win_probability']:.4f} vs {1 - opposite_result['player1_win_probability']:.4f}")
        
        # Calculate and log total time
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Tennis match prediction completed successfully in {duration}")
        
    except Exception as e:
        logger.error(f"Error running predictions: {e}")
        raise

if __name__ == "__main__":
    main() 