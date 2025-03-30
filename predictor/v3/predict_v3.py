import os
import time
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
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
V3_DATA_DIR = DATA_DIR / "v3"
OUTPUT_DIR = PROJECT_ROOT / "predictor" / "output" / "v3"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# File paths
MODEL_PATH = OUTPUT_DIR / "tennis_model_v3.json"
FEATURES_PATH = V3_DATA_DIR / "features_v3.csv"
PREDICTIONS_OUTPUT = OUTPUT_DIR / "predictions_v3.csv"

# Constants
SURFACES = ['hard', 'clay', 'grass']


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


def load_test_data(features_path: Union[str, Path], test_size: float = 0.2, 
                 progress_tracker: Optional[ProgressTracker] = None) -> pd.DataFrame:
    """
    Load test data from features file.
    
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
        
        # Calculate split index
        split_idx = int(len(df) * (1 - test_size))
        
        # Get test data
        test_df = df.iloc[split_idx:].copy()
        
        logger.info(f"Loaded {len(test_df)} test matches")
        
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
    exclude_cols = ['match_id', 'tourney_date', 'player1_id', 'player2_id', 'surface', 'result']
    
    # Get all columns except excluded ones
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
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
        
        # Extract features
        X_test = df_pred[feature_cols].values
        y_test = df_pred['result'].values
        
        # Create DMatrix
        dtest = xgb.DMatrix(X_test, feature_names=feature_cols)
        
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
                
                logger.info(f"{surface.capitalize()} surface accuracy: {surface_acc:.4f} (n={sum(surface_idx)})")
        
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
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
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
            ax = plt.bar(conf_ranges, accuracies, color='skyblue')
            
            # Add count labels
            for i, v in enumerate(accuracies):
                plt.text(i, v + 0.01, f"n={counts[i]}", ha='center')
            
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
            plt.bar(surfaces, surface_accs, color='lightgreen')
            
            # Add count labels
            for i, v in enumerate(surface_accs):
                plt.text(i, v + 0.01, f"n={surface_counts[i]}", ha='center')
            
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
        
        # Calculate differences
        for stat in p1_stats:
            if stat in p2_stats:
                features[f"{stat}_diff"] = p1_stats[stat] - p2_stats[stat]
        
        # Add raw player features
        for stat in p1_stats:
            features[f"player1_{stat}"] = p1_stats[stat]
        
        for stat in p2_stats:
            features[f"player2_{stat}"] = p2_stats[stat]
        
        # Add surface
        features['surface'] = surface
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Get feature columns that match the model's features
        feature_names = model.feature_names
        
        # Filter and check for missing features
        missing_features = [f for f in feature_names if f not in df.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            
            # Set missing features to 0
            for f in missing_features:
                df[f] = 0
        
        # Select only the features used by the model
        X = df[feature_names].values
        
        # Create DMatrix
        dmat = xgb.DMatrix(X, feature_names=feature_names)
        
        # Make prediction
        prob = model.predict(dmat)[0]
        pred = 1 if prob > 0.5 else 0
        
        # Prepare result
        result = {
            'player1_id': player1_id,
            'player2_id': player2_id,
            'surface': surface,
            'predicted_winner': player1_id if pred == 1 else player2_id,
            'predicted_proba': prob if pred == 1 else 1 - prob,
            'features_used': feature_names
        }
        
        logger.info(f"Prediction: {result['predicted_winner']} will win with {result['predicted_proba']:.2f} probability")
        
        return result
    
    except Exception as e:
        logger.error(f"Error predicting match: {e}")
        raise


def main():
    """Make predictions using the trained model."""
    start_time = time.time()
    
    # Define total steps for progress tracking
    total_steps = 7
    progress_tracker = ProgressTracker(total_steps)
    
    try:
        # Step 1: Load model
        logger.info(f"Step 1/{total_steps}: Loading model...")
        model = load_model(MODEL_PATH, progress_tracker)
        
        # Step 2: Load test data
        logger.info(f"Step 2/{total_steps}: Loading test data...")
        test_df = load_test_data(FEATURES_PATH, 0.2, progress_tracker)
        
        # Step 3: Get feature columns
        logger.info(f"Step 3/{total_steps}: Identifying feature columns...")
        feature_cols = get_feature_columns(test_df, progress_tracker)
        
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


    # Check if all required features exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing features: {missing_cols}")
        feature_cols = [col for col in feature_cols if col in df.columns]
    
    # Prepare feature matrix
    X = df[feature_cols].values
    
    # Make predictions with progress bar
    batch_size = 10000 if not GPU_AVAILABLE else 50000  # Larger batches for GPU
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
        'surface': surface,
        # Add default values for critical features to ensure model runs even with missing data
        'player_elo_diff': 0.0,  # Default to equal skills if no Elo available
        'win_rate_5_diff': 0.0,
        'win_streak_diff': 0.0,
        'loss_streak_diff': 0.0
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
    
    # If we still don't have player_elo_diff, attempt to construct it from Elo columns
    if 'player_elo_diff' not in match_data or pd.isna(match_data['player_elo_diff']):
        elo_diff = 0.0  # Default if no Elo available
        
        # Try to get Elo from source data
        if source_df is not None:
            # Look for matches with these players to get their most recent Elo
            player1_elo = None
            player2_elo = None
            
            # Find player1's most recent Elo
            p1_match = source_df[(source_df['player1_id'] == player1_id) | 
                               (source_df['player2_id'] == player1_id)].sort_values('tourney_date').tail(1)
            if not p1_match.empty:
                if p1_match['player1_id'].iloc[0] == player1_id:
                    player1_elo = p1_match['player1_elo'].iloc[0] if 'player1_elo' in p1_match.columns else 1500.0
                else:
                    player1_elo = p1_match['player2_elo'].iloc[0] if 'player2_elo' in p1_match.columns else 1500.0
            
            # Find player2's most recent Elo
            p2_match = source_df[(source_df['player1_id'] == player2_id) | 
                               (source_df['player2_id'] == player2_id)].sort_values('tourney_date').tail(1)
            if not p2_match.empty:
                if p2_match['player1_id'].iloc[0] == player2_id:
                    player2_elo = p2_match['player1_elo'].iloc[0] if 'player1_elo' in p2_match.columns else 1500.0
                else:
                    player2_elo = p2_match['player2_elo'].iloc[0] if 'player2_elo' in p2_match.columns else 1500.0
            
            # Calculate Elo difference if we found both
            if player1_elo is not None and player2_elo is not None:
                elo_diff = player1_elo - player2_elo
        
        match_data['player_elo_diff'] = elo_diff
    
    # Calculate difference features
    if source_df is not None:
        # Get feature columns from source data
        feature_cols = get_feature_columns(source_df)
        logger.info(f"Using {len(feature_cols)} features: {feature_cols}")
        
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
    logger.info(f"Using {len(feature_cols)} features: {feature_cols}")
    
    # Check for missing required features and set defaults if needed
    expected_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else []
    if expected_features and len(expected_features) > 0:
        for feature in expected_features:
            if feature not in match_df.columns:
                match_df[feature] = 0.0  # Use neutral default value
                logger.warning(f"Missing required feature '{feature}'. Using default value 0.0")
    
    # Make prediction
    try:
        # Check if we have all the expected features
        if hasattr(model, 'feature_names_in_'):
            expected_count = len(model.feature_names_in_)
            actual_count = len(feature_cols)
            if expected_count != actual_count:
                logger.warning(f"Feature count mismatch. Model expects {expected_count} features, but {actual_count} provided.")
                # Try to match the expected features
                X = np.zeros((1, expected_count))
                for i, feature in enumerate(model.feature_names_in_):
                    if feature in match_df.columns:
                        X[0, i] = match_df[feature].iloc[0]
                    else:
                        logger.warning(f"Missing feature '{feature}' for prediction. Using default value 0.0")
            else:
                X = match_df[feature_cols].values
        else:
            X = match_df[feature_cols].values
        
        # Get probability with progress tracking
        with tqdm(total=1, desc="Predicting match outcome", unit="match") as pbar:
            win_probability = model.predict_proba(X)[0, 1]
            pbar.update(1)
        
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
        
        # Check if both predictions were successful
        if ('player1_win_probability' in sample_result and 
            'player1_win_probability' in opposite_result):
            # Compare probabilities
            logger.info(f"Probability check (should be symmetric): " +
                      f"{sample_result['player1_win_probability']:.4f} vs " +
                      f"{1 - opposite_result['player1_win_probability']:.4f}")
        else:
            logger.warning("Could not verify prediction symmetry due to errors in sample predictions")
        
        # Calculate and log total time
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Tennis match prediction completed successfully in {duration}")
        
    except Exception as e:
        logger.error(f"Error running predictions: {e}")
        raise

if __name__ == "__main__":
    main() 