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
MODELS_DIR = PROJECT_ROOT / "models" / "v3"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# File paths
MODEL_PATH = MODELS_DIR / "tennis_model_v3.json"
FEATURES_PATH = V3_DATA_DIR / "features_v3.csv"
PREDICTIONS_OUTPUT = OUTPUT_DIR / "predictions_v3.csv"

# Constants
SURFACES = ['Hard', 'Clay', 'Grass', 'Carpet']

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
            features['player_elo_diff'] = 0.0  # Default if no Elo available
        
        # Add win rate differences
        for stat in RAW_PLAYER_FEATURES:
            if stat in p1_stats and stat in p2_stats:
                features[f"{stat}_diff"] = p1_stats[stat] - p2_stats[stat]
                
                # Add raw player features
                features[f"player1_{stat}"] = p1_stats[stat]
                features[f"player2_{stat}"] = p2_stats[stat]
        
        # Add surface-specific win rates if available
        surface_win_rate_5 = f"win_rate_{surface}_5"
        surface_win_rate_overall = f"win_rate_{surface}_overall"
        
        if surface_win_rate_5 in p1_stats and surface_win_rate_5 in p2_stats:
            features[f"{surface_win_rate_5}_diff"] = p1_stats[surface_win_rate_5] - p2_stats[surface_win_rate_5]
        
        if surface_win_rate_overall in p1_stats and surface_win_rate_overall in p2_stats:
            features[f"{surface_win_rate_overall}_diff"] = p1_stats[surface_win_rate_overall] - p2_stats[surface_win_rate_overall]
        
        # Add serve and return stats
        for stat in SERVE_RETURN_FEATURES:
            base_stat = stat.split('_diff')[0]
            
            if base_stat in p1_stats and base_stat in p2_stats:
                features[stat] = p1_stats[base_stat] - p2_stats[base_stat]
                
                # Add raw player serve/return stats
                features[f"player1_{base_stat}"] = p1_stats[base_stat]
                features[f"player2_{base_stat}"] = p2_stats[base_stat]
            
            # Add surface-specific serve/return stats if available
            surface_stat = f"{base_stat}_{surface}"
            
            if surface_stat in p1_stats and surface_stat in p2_stats:
                features[f"{surface_stat}_diff"] = p1_stats[surface_stat] - p2_stats[surface_stat]
        
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


if __name__ == "__main__":
    main() 