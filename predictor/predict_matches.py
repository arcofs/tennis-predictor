import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import logging
from datetime import datetime

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
ALL_FEATURES_MODEL_PATH = MODELS_DIR / "tennis_predictor_all_features.xgb"
RESULTS_PATH = OUTPUT_DIR / "all_features_prediction_results.csv"
METRICS_PATH = OUTPUT_DIR / "all_features_prediction_metrics.txt"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "all_features_confusion_matrix.png"
SURFACE_ANALYSIS_PATH = OUTPUT_DIR / "all_features_surface_analysis.png"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Standard surface definitions
SURFACE_HARD = 'Hard'
SURFACE_CLAY = 'Clay'
SURFACE_GRASS = 'Grass'
SURFACE_CARPET = 'Carpet'
STANDARD_SURFACES = [SURFACE_HARD, SURFACE_CLAY, SURFACE_GRASS, SURFACE_CARPET]

def verify_surface_name(surface: str) -> str:
    """Verify and correct surface name to ensure consistency."""
    if pd.isna(surface):
        return None
    
    surface_str = str(surface).lower()
    surface_mapping = {
        'hard': SURFACE_HARD,
        'h': SURFACE_HARD,
        'clay': SURFACE_CLAY,
        'cl': SURFACE_CLAY,
        'grass': SURFACE_GRASS,
        'gr': SURFACE_GRASS,
        'carpet': SURFACE_CARPET,
        'cpt': SURFACE_CARPET,
        'indoor': SURFACE_HARD,  # Map indoor to hard
        'outdoor': SURFACE_HARD,  # Map outdoor to hard by default
    }
    
    return surface_mapping.get(surface_str, surface)

def load_model(model_path: Path = ALL_FEATURES_MODEL_PATH) -> xgb.XGBClassifier:
    """Load the trained XGBoost model."""
    logger.info(f"Loading model from {model_path}...")
    
    try:
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def load_matches(csv_path: Path, recent_only: bool = True) -> pd.DataFrame:
    """
    Load match data from a CSV file.
    
    Args:
        csv_path: Path to the CSV file
        recent_only: Whether to only include recent matches (last 3 years)
        
    Returns:
        DataFrame with match data
    """
    logger.info(f"Loading match data from {csv_path}...")
    
    try:
        # Load the CSV file
        df = pd.read_csv(csv_path)
        
        # Convert date column to datetime
        if 'tourney_date' in df.columns:
            df['tourney_date'] = pd.to_datetime(df['tourney_date'], errors='coerce')
            
            # Filter by date if requested
            if recent_only:
                recent_date = pd.Timestamp.now() - pd.DateOffset(years=3)
                df = df[df['tourney_date'] >= recent_date]
                logger.info(f"Filtered to matches from the last 3 years: {len(df)} matches")
        
        # Standardize surface names
        if 'surface' in df.columns:
            df['surface'] = df['surface'].apply(verify_surface_name)
        
        # Create player1/player2 structure from winner/loser
        if 'winner_id' in df.columns and 'loser_id' in df.columns:
            # Create balanced dataset - each match appears twice with swapped players
            df_player1_wins = df.copy()
            df_player1_wins['player1_id'] = df_player1_wins['winner_id']
            df_player1_wins['player2_id'] = df_player1_wins['loser_id']
            df_player1_wins['result'] = 1  # player1 wins
            
            df_player2_wins = df.copy()
            df_player2_wins['player1_id'] = df_player2_wins['loser_id']
            df_player2_wins['player2_id'] = df_player2_wins['winner_id']
            df_player2_wins['result'] = 0  # player1 loses
            
            # Combine and sort by date
            df_balanced = pd.concat([df_player1_wins, df_player2_wins])
            df_balanced = df_balanced.sort_values('tourney_date').reset_index(drop=True)
            
            # Use the balanced dataset
            df = df_balanced
            logger.info(f"Created balanced dataset with {len(df)} rows")
            
        # Create difference features if needed
        # We'll use generate_difference_features for this
            
        return df
        
    except Exception as e:
        logger.error(f"Error loading match data: {e}")
        raise

def generate_difference_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Generate difference features for prediction.
    
    Args:
        df: DataFrame with match data
        
    Returns:
        Tuple of (DataFrame with difference features, list of feature columns)
    """
    logger.info("Generating difference features...")
    
    # Start with an empty list of feature columns
    feature_cols = []
    
    # Look for pairs of player1/player2 features and create difference features
    player1_cols = [col for col in df.columns if col.startswith('player1_') and col != 'player1_id']
    
    for p1_col in player1_cols:
        feature_name = p1_col[8:]  # Remove 'player1_' prefix
        p2_col = f'player2_{feature_name}'
        
        if p2_col in df.columns:
            # Check for boolean columns
            if pd.api.types.is_bool_dtype(df[p1_col]) or pd.api.types.is_bool_dtype(df[p2_col]):
                # Convert to int before subtraction
                diff_col = f'{feature_name}_diff'
                df[diff_col] = df[p1_col].astype(int) - df[p2_col].astype(int)
            else:
                # For numeric columns, perform normal subtraction
                diff_col = f'{feature_name}_diff'
                df[diff_col] = df[p1_col] - df[p2_col]
            
            feature_cols.append(diff_col)
    
    # If winner_*/loser_* features exist, also create diff features
    winner_cols = [col for col in df.columns if col.startswith('winner_') and col != 'winner_id']
    
    for w_col in winner_cols:
        feature_name = w_col[7:]  # Remove 'winner_' prefix
        l_col = f'loser_{feature_name}'
        
        if l_col in df.columns:
            if pd.api.types.is_bool_dtype(df[w_col]) or pd.api.types.is_bool_dtype(df[l_col]):
                diff_col = f'{feature_name}_diff'
                if diff_col not in df.columns:  # Don't overwrite existing diff cols
                    df[diff_col] = df[w_col].astype(int) - df[l_col].astype(int)
                    feature_cols.append(diff_col)
            else:
                diff_col = f'{feature_name}_diff'
                if diff_col not in df.columns:  # Don't overwrite existing diff cols
                    df[diff_col] = df[w_col] - df[l_col]
                    feature_cols.append(diff_col)
    
    # Use existing diff columns if present
    existing_diff_cols = [col for col in df.columns if col.endswith('_diff') and col not in feature_cols]
    feature_cols.extend(existing_diff_cols)
    
    # Filter out any potentially leaky features
    leakage_keywords = ['winner', 'loser', 'score', 'sets_won', 'games_won']
    leaky_features = []
    
    for col in feature_cols:
        for keyword in leakage_keywords:
            if keyword in col.lower() and col not in ['win_rate_diff', 'h2h_win_pct_diff'] and not col.startswith('win_rate_'):
                leaky_features.append(col)
                break
    
    # Remove leaky features
    clean_features = [col for col in feature_cols if col not in leaky_features]
    
    if leaky_features:
        logger.warning(f"Removed {len(leaky_features)} potentially leaky features: {leaky_features[:5]}...")
    
    logger.info(f"Generated {len(clean_features)} difference features")
    return df, clean_features

def predict_matches(model: xgb.XGBClassifier, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Predict match outcomes.
    
    Args:
        model: Trained XGBoost model
        df: DataFrame with match data and features
        feature_cols: List of feature columns to use
        
    Returns:
        DataFrame with predictions added
    """
    logger.info(f"Predicting match outcomes for {len(df)} matches...")
    
    # Check if we have all required features
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        logger.warning(f"Missing {len(missing_features)} features: {missing_features[:5]}...")
        
        # Only use available features
        available_features = [col for col in feature_cols if col in df.columns]
        logger.info(f"Using {len(available_features)} available features")
        feature_cols = available_features
    
    # DEBUG: Check for zero or constant values in features
    logger.info("=== FEATURE ANALYSIS ===")
    for i, feature in enumerate(feature_cols[:10]):  # Analyze first 10 features
        unique_values = df[feature].nunique()
        min_val = df[feature].min()
        max_val = df[feature].max()
        mean_val = df[feature].mean()
        null_pct = (df[feature].isnull().sum() / len(df)) * 100
        logger.info(f"Feature '{feature}': unique values={unique_values}, min={min_val}, max={max_val}, mean={mean_val:.4f}, null={null_pct:.2f}%")
    
    # Prepare feature matrix - keep NaN values for XGBoost
    X = df[feature_cols].values
    
    # Check feature array for issues
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Non-finite values in feature matrix: {np.sum(~np.isfinite(X))}")
    
    # === Do not scale features (to match training) ===
    # Make predictions (test direct prediction without scaling)
    logger.info("Making predictions without scaling...")
    y_prob = model.predict_proba(X)[:, 1]
    
    # Debug the prediction distribution
    logger.info(f"Prediction probabilities min: {y_prob.min():.6f}, max: {y_prob.max():.6f}, mean: {y_prob.mean():.6f}")
    logger.info(f"Prediction distribution: \n{np.histogram(y_prob, bins=10)[0]}")
    
    # Count predictions by threshold
    y_pred = (y_prob > 0.5).astype(int)
    logger.info(f"Predictions: 0s={np.sum(y_pred==0)}, 1s={np.sum(y_pred==1)}")
    
    # Add predictions to DataFrame
    df['predicted_probability'] = y_prob
    df['predicted_result'] = y_pred
    
    # Add interpretable columns
    df['predicted_winner'] = df.apply(
        lambda row: row['player1_id'] if row['predicted_result'] == 1 else row['player2_id'], 
        axis=1
    )
    
    # If we have actual results, calculate accuracy
    if 'result' in df.columns:
        df['correct'] = df['predicted_result'] == df['result']
        accuracy = df['correct'].mean()
        logger.info(f"Overall prediction accuracy: {accuracy:.4f}")
        
        # Calculate accuracy by probability range
        logger.info("Accuracy by prediction confidence:")
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0]
        df['prob_bin'] = pd.cut(df['predicted_probability'], bins=bins)
        bin_accuracy = df.groupby('prob_bin')['correct'].agg(['mean', 'count'])
        logger.info(f"\n{bin_accuracy}")
        
        # Calculate accuracy by surface if available
        if 'surface' in df.columns:
            surface_accuracy = df.groupby('surface')['correct'].mean()
            for surface, acc in surface_accuracy.items():
                logger.info(f"Accuracy on {surface}: {acc:.4f}")
    
    # Check for balanced results by player position (player1/2)
    player1_win_pct = df[df['player1_id'] == df['predicted_winner']].shape[0] / df.shape[0]
    logger.info(f"Percentage of predictions where player1 wins: {player1_win_pct:.4f}")
    
    return df

def save_predictions(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save match predictions to CSV.
    
    Args:
        df: DataFrame with predictions
        output_path: Path to save CSV
    """
    logger.info(f"Saving predictions to {output_path}...")
    
    # Select columns to save
    cols_to_save = [
        'tourney_date', 'tourney_name', 'surface', 
        'player1_id', 'player2_id', 
        'predicted_probability', 'predicted_winner'
    ]
    
    # Add player names if available
    if 'player1_name' in df.columns:
        cols_to_save.append('player1_name')
    if 'player2_name' in df.columns:
        cols_to_save.append('player2_name')
    
    # Add actual results if available
    if 'result' in df.columns:
        cols_to_save.append('result')
    if 'correct' in df.columns:
        cols_to_save.append('correct')
    
    # Filter to columns that exist
    save_cols = [col for col in cols_to_save if col in df.columns]
    
    # Save to CSV
    df[save_cols].to_csv(output_path, index=False)
    logger.info(f"Saved predictions to {output_path}")

def plot_prediction_analysis(df: pd.DataFrame) -> None:
    """
    Plot analysis of prediction results.
    
    Args:
        df: DataFrame with predictions
    """
    # Make sure we have the necessary columns for plotting
    if 'match_winner_correct' not in df.columns and 'correct' not in df.columns:
        logger.warning("No accuracy data available for plotting")
        return
    
    # Use match_winner_correct if available, otherwise use correct
    correct_col = 'match_winner_correct' if 'match_winner_correct' in df.columns else 'correct'
    
    # Plot accuracy by surface
    if 'surface' in df.columns:
        logger.info("Plotting accuracy by surface...")
        
        plt.figure(figsize=(12, 6))
        surface_accuracy = df.groupby('surface')[correct_col].mean()
        surface_counts = df.groupby('surface').size()
        
        # Plot as bar chart
        ax = surface_accuracy.plot(kind='bar', color='skyblue')
        
        # Add counts
        for i, surface in enumerate(surface_accuracy.index):
            ax.text(i, surface_accuracy[surface] + 0.02, 
                   f"n={surface_counts[surface]}", ha='center')
        
        plt.title('Prediction Accuracy by Surface')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig(SURFACE_ANALYSIS_PATH)
        plt.close()
    
    # Plot confusion matrix
    logger.info("Plotting confusion matrix...")
    
    # Only plot the confusion matrix if we have expected results
    if 'expected_result' in df.columns and 'predicted_result' in df.columns:
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(df['expected_result'], df['predicted_result'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Player 2 Wins', 'Player 1 Wins'],
                   yticklabels=['Player 2 Wins', 'Player 1 Wins'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(CONFUSION_MATRIX_PATH)
        plt.close()
    else:
        logger.warning("Cannot plot confusion matrix: missing expected or predicted results")

def predict_upcoming_match(model: xgb.XGBClassifier, player1_id: int, player2_id: int, 
                          surface: str, feature_dict: Dict = None) -> Dict:
    """
    Predict the outcome of an upcoming match.
    
    Args:
        model: Trained XGBoost model
        player1_id: ID of player 1
        player2_id: ID of player 2
        surface: Match surface
        feature_dict: Dictionary of pre-computed features
        
    Returns:
        Dictionary with prediction results
    """
    logger.info(f"Predicting match between player {player1_id} vs player {player2_id} on {surface}")
    
    # Create a DataFrame with the match
    match_df = pd.DataFrame({
        'player1_id': [player1_id],
        'player2_id': [player2_id],
        'surface': [surface]
    })
    
    # Add features from feature_dict if provided
    if feature_dict:
        for feature, value in feature_dict.items():
            match_df[feature] = [value]
    
    # For prediction, we need to ensure we have exactly the same features
    # that the model was trained with. Let's get the required features from the model.
    try:
        # Get feature count from model
        n_features = model.n_features_in_
        logger.info(f"Model requires {n_features} features")
        
        # Get a sample from the dataset to extract feature names
        data_path = DATA_DIR / "enhanced_features_v2.csv"
        sample_df = pd.read_csv(data_path, nrows=10)
        
        # Generate features for the sample
        sample_df_processed, all_feature_cols = generate_difference_features(sample_df)
        
        # Generate placeholder features for prediction
        for feature in all_feature_cols:
            if feature not in match_df.columns:
                match_df[feature] = 0.0  # Default value
        
        # Make prediction
        feature_cols = all_feature_cols
        X = match_df[feature_cols].values
        
        # Predict with raw features - XGBoost handles missing values
        y_prob = model.predict_proba(X)[:, 1][0]
        winner_id = player1_id if y_prob > 0.5 else player2_id
        
        # Create result dictionary
        result = {
            'player1_id': player1_id,
            'player2_id': player2_id,
            'surface': surface,
            'player1_win_probability': float(y_prob),
            'player2_win_probability': float(1 - y_prob),
            'predicted_winner_id': int(winner_id)
        }
        
        logger.info(f"Prediction: {result}")
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
    logger.info("Starting tennis match prediction using all_features model")
    
    try:
        # 1. Load the model
        model = load_model()
        
        # DEBUG: Print model info
        logger.info(f"Model info - feature count: {model.n_features_in_}")
        
        # 2. Load match data
        data_path = DATA_DIR / "enhanced_features_v2.csv"
        df = pd.read_csv(data_path)
        
        # Convert date column to datetime for time-based split
        if 'tourney_date' in df.columns:
            df['tourney_date'] = pd.to_datetime(df['tourney_date'])
            
        # Use only test data (from 2023-01-01 onwards) for predictions
        # This ensures we're predicting on data not seen during training
        test_date = '2023-01-01'
        test_df = df[df['tourney_date'] >= pd.to_datetime(test_date)].copy()
        logger.info(f"Using {len(test_df)} matches from {test_date} onwards for prediction")
        
        # Standardize surface names
        if 'surface' in test_df.columns:
            test_df['surface'] = test_df['surface'].apply(verify_surface_name)
        
        # NOTE: For proper evaluation, we need to do both perspectives
        # This is because the model might perform better on one perspective than the other
        if 'winner_id' in test_df.columns and 'loser_id' in test_df.columns:
            # Create both perspectives
            # Original perspective (winner as player1)
            winner_perspective = test_df.copy()
            winner_perspective['player1_id'] = winner_perspective['winner_id'] 
            winner_perspective['player2_id'] = winner_perspective['loser_id']
            winner_perspective['expected_winner'] = winner_perspective['winner_id']
            winner_perspective['perspective'] = 'winner_perspective'
            winner_perspective['expected_result'] = 1  # We expect player1 to win
            
            # Flipped perspective (loser as player1)
            loser_perspective = test_df.copy()
            loser_perspective['player1_id'] = loser_perspective['loser_id']
            loser_perspective['player2_id'] = loser_perspective['winner_id']
            loser_perspective['expected_winner'] = loser_perspective['winner_id'] 
            loser_perspective['perspective'] = 'loser_perspective'
            loser_perspective['expected_result'] = 0  # We expect player1 to lose
            
            # Combine perspectives for complete evaluation
            evaluation_df = pd.concat([winner_perspective, loser_perspective])
            
            # Generate features for full evaluation dataset
            logger.info(f"Prepared {len(evaluation_df)} matches for evaluation (both perspectives)")
            evaluation_df_features, eval_feature_cols = generate_difference_features(evaluation_df)
            
            # Make predictions on full evaluation dataset
            predictions_df = predict_matches(model, evaluation_df_features, eval_feature_cols)
            
            # Calculate evaluation metrics for winner perspective
            winner_preds = predictions_df[predictions_df['perspective'] == 'winner_perspective'].copy()
            winner_preds['correct'] = (winner_preds['predicted_result'] == winner_preds['expected_result'])
            winner_accuracy = winner_preds['correct'].mean()
            logger.info(f"Winner perspective accuracy: {winner_accuracy:.4f}")
            
            # Calculate evaluation metrics for loser perspective
            loser_preds = predictions_df[predictions_df['perspective'] == 'loser_perspective'].copy()
            loser_preds['correct'] = (loser_preds['predicted_result'] == loser_preds['expected_result'])
            loser_accuracy = loser_preds['correct'].mean()
            logger.info(f"Loser perspective accuracy: {loser_accuracy:.4f}")
            
            # Calculate match-level accuracy (did we predict the correct winner regardless of perspective)
            predictions_df['match_winner_correct'] = (
                predictions_df['predicted_winner'] == predictions_df['expected_winner']
            )
            match_accuracy = predictions_df['match_winner_correct'].mean()
            logger.info(f"Overall match winner accuracy: {match_accuracy:.4f}")
            
            # Copy match_winner_correct to the winner_preds DataFrame
            match_ids = predictions_df[['winner_id', 'loser_id', 'match_winner_correct']]
            winner_preds = winner_preds.merge(
                match_ids[['winner_id', 'loser_id', 'match_winner_correct']],
                on=['winner_id', 'loser_id'],
                how='left'
            )
            
            # Calculate surface-specific accuracies
            if 'surface' in predictions_df.columns:
                surface_accuracy = predictions_df.groupby('surface')['match_winner_correct'].mean()
                logger.info("Accuracy by surface:")
                for surface, acc in surface_accuracy.items():
                    count = len(predictions_df[predictions_df['surface'] == surface]) // 2
                    logger.info(f"  {surface}: {acc:.4f} (n={count})")
            
            # Plot accuracy by prediction confidence band
            logger.info("Plotting analysis...")
            plot_prediction_analysis(predictions_df)
            
            # Save predictions to CSV
            save_predictions(predictions_df, RESULTS_PATH)
            
            # Also save a simplified version with one row per match
            match_summary = winner_preds[['tourney_date', 'surface', 'winner_id', 'loser_id', 
                               'predicted_winner', 'predicted_probability', 'match_winner_correct']]
            
            match_summary.to_csv(OUTPUT_DIR / "match_prediction_summary.csv", index=False)
            logger.info(f"Saved match prediction summary to {OUTPUT_DIR / 'match_prediction_summary.csv'}")
        
        # Sample prediction for a specific match
        sample_player1_id = 104925
        sample_player2_id = 105657 
        sample_result = predict_upcoming_match(
            model, 
            sample_player1_id, 
            sample_player2_id, 
            SURFACE_HARD
        )
        
        logger.info(f"Sample match prediction: {sample_result}")
        
        # Example of predicting from opposite perspective
        logger.info("\nPredicting same match from opposite perspective:")
        flipped_result = predict_upcoming_match(
            model,
            sample_player2_id,  # Swapped player order
            sample_player1_id,
            SURFACE_HARD
        )
        
        logger.info(f"Flipped perspective prediction: {flipped_result}")
        p1_probability = sample_result['player1_win_probability']
        p2_probability = flipped_result['player2_win_probability']
        logger.info(f"Probability check - should be close to equal: {p1_probability:.4f} vs {1-p2_probability:.4f}")
        
        logger.info("Tennis match prediction completed successfully")
        
    except Exception as e:
        logger.error(f"Error running prediction: {e}")
        raise

if __name__ == "__main__":
    main() 