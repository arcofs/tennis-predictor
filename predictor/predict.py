import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import logging
import random
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
CLEANED_DATA_PATH = DATA_DIR / "cleaned" / "cleaned_dataset_with_elo.csv"
MODEL_PATH = MODELS_DIR / "tennis_predictor.xgb"
RESULTS_PATH = OUTPUT_DIR / "prediction_results.csv"
METRICS_PATH = OUTPUT_DIR / "prediction_metrics.txt"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"
SURFACE_ANALYSIS_PATH = OUTPUT_DIR / "surface_analysis.png"
ACCURACY_BY_YEAR_PATH = OUTPUT_DIR / "accuracy_by_year.png"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Import the surface constants and verification function
# Add after imports
try:
    from predictor.generate_features import (
        SURFACE_HARD, SURFACE_CLAY, SURFACE_GRASS, SURFACE_CARPET,
        STANDARD_SURFACES, verify_surface_name
    )
except ImportError:
    # Define them here as fallback
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

def load_model() -> xgb.XGBClassifier:
    """Load the trained XGBoost model."""
    logger.info(f"Loading model from {MODEL_PATH}...")
    
    try:
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.warning("Using a dummy model for testing")
        return xgb.XGBClassifier(objective='binary:logistic', random_state=42)

def load_test_data(
    csv_path: Path, 
    sample_size: int = 10000,
    random_seed: int = 42,
    min_year: int = 2010  # Only include recent matches
) -> pd.DataFrame:
    """
    Load a sample of the test data for prediction evaluation.
    
    Args:
        csv_path: Path to the CSV file
        sample_size: Number of matches to sample
        random_seed: Random seed for reproducibility
        min_year: Minimum year to include in the sample
        
    Returns:
        DataFrame containing the sampled matches
    """
    logger.info(f"Loading test data from {csv_path}...")
    
    try:
        # Define numeric columns to force numeric type conversion
        numeric_columns = [
            'winner_id', 'loser_id', 'winner_elo', 'loser_elo',
            'winner_rank', 'loser_rank', 'winner_ht', 'loser_ht',
            'w_ace', 'l_ace', 'w_df', 'l_df', 'w_svpt', 'l_svpt',
            'w_1stIn', 'l_1stIn', 'w_1stWon', 'l_1stWon', 'w_2ndWon', 'l_2ndWon',
            'w_SvGms', 'l_SvGms', 'w_bpSaved', 'l_bpSaved', 'w_bpFaced', 'l_bpFaced'
        ]
        
        # Try to identify columns that should be numeric
        dtype_dict = {col: 'float64' for col in numeric_columns}
        
        # Read the data with specified types
        df = pd.read_csv(csv_path, dtype=dtype_dict, low_memory=False)
        
        # Convert date column to datetime
        df['tourney_date'] = pd.to_datetime(df['tourney_date'], errors='coerce')
        
        # Filter out rows with invalid dates
        if df['tourney_date'].isna().any():
            logger.warning(f"Removing {df['tourney_date'].isna().sum()} rows with invalid dates")
            df = df.dropna(subset=['tourney_date'])
        
        # Filter by year if specified
        if min_year:
            df = df[df['tourney_date'].dt.year >= min_year]
            logger.info(f"Filtered to matches from {min_year} onwards: {len(df)} matches")
        
        # Take a random sample if the dataset is larger than the sample size
        if len(df) > sample_size:
            random.seed(random_seed)
            df = df.sample(sample_size, random_state=random_seed)
            logger.info(f"Randomly sampled {sample_size} matches from the dataset")
        
        # Sort by date
        df = df.sort_values('tourney_date')
        
        # Convert any remaining string columns that should be numeric
        for col in df.columns:
            if col.startswith(('winner_', 'loser_', 'w_', 'l_')) and col not in ['winner_name', 'loser_name', 'winner_ioc', 'loser_ioc']:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                        logger.info(f"Converted column {col} to numeric")
                    except Exception as e:
                        logger.warning(f"Could not convert column {col} to numeric: {e}")
        
        logger.info(f"Loaded {len(df)} matches spanning from {df['tourney_date'].min().date()} to {df['tourney_date'].max().date()}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def prepare_features(match_df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    """
    Prepare features for prediction.
    
    Args:
        match_df: DataFrame with match data
        feature_cols: List of feature columns to use
        
    Returns:
        numpy array with features
    """
    logger.info("Preparing features for prediction...")
    
    # Make a copy to avoid modifying the original
    df = match_df.copy()
    
    # Map H2H features from p1/p2 to player1/player2 format if needed
    h2h_mapping = {
        'h2h_wins_p1': 'h2h_wins_player1',
        'h2h_wins_p2': 'h2h_wins_player2',
        'h2h_win_pct_p1': 'h2h_win_pct_player1',
        'h2h_win_pct_p2': 'h2h_win_pct_player2',
        'h2h_hard_win_pct_p1': 'h2h_hard_win_pct_player1',
        'h2h_clay_win_pct_p1': 'h2h_clay_win_pct_player1',
        'h2h_grass_win_pct_p1': 'h2h_grass_win_pct_player1'
    }
    
    # Rename any h2h columns that use p1/p2 format
    for old_col, new_col in h2h_mapping.items():
        if old_col in df.columns:
            df[new_col] = df[old_col]
    
    # Make sure all feature columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing {len(missing_cols)} feature columns: {missing_cols[:5]}...")
        # Add missing columns with zeros
        for col in missing_cols:
            df[col] = 0.0
    
    # Calculate difference features if necessary
    if 'player1_elo' in df.columns and 'player2_elo' in df.columns and 'elo_diff' in feature_cols:
        df['elo_diff'] = df['player1_elo'] - df['player2_elo']
        logger.info("Calculated elo_diff feature")
    
    # Create similar difference features for other relevant pairs
    diff_pairs = [
        ('player1_win_rate_10', 'player2_win_rate_10', 'win_rate_10_diff'),
        ('player1_current_win_streak', 'player2_current_win_streak', 'current_win_streak_diff'),
        ('player1_current_loss_streak', 'player2_current_loss_streak', 'current_loss_streak_diff')
    ]
    
    for col1, col2, diff_col in diff_pairs:
        if col1 in df.columns and col2 in df.columns and diff_col in feature_cols:
            df[diff_col] = df[col1] - df[col2]
            logger.info(f"Calculated {diff_col} feature")
    
    # Surface-specific features
    for surface in STANDARD_SURFACES:
        win_rate_col = f'win_rate_{surface}_10_diff'
        if win_rate_col in feature_cols:
            p1_col = f'player1_win_rate_{surface}_10'
            p2_col = f'player2_win_rate_{surface}_10'
            if p1_col in df.columns and p2_col in df.columns:
                df[win_rate_col] = df[p1_col] - df[p2_col]
                logger.info(f"Calculated {win_rate_col} feature")
    
    # H2H difference
    if 'h2h_win_pct_diff' in feature_cols:
        if 'h2h_win_pct_player1' in df.columns and 'h2h_win_pct_player2' in df.columns:
            df['h2h_win_pct_diff'] = df['h2h_win_pct_player1'] - df['h2h_win_pct_player2']
            logger.info("Calculated h2h_win_pct_diff feature")
    
    # Convert categorical variables
    for col in df.select_dtypes(include=['category']).columns:
        if col in feature_cols:
            df[col] = df[col].cat.codes
    
    # Get the final feature matrix
    X = df[feature_cols].fillna(0).values
    
    logger.info(f"Prepared {X.shape[1]} features for {X.shape[0]} matches")
    return X

def predict_matches(
    model: xgb.XGBClassifier,
    matches_df: pd.DataFrame, 
    feature_cols: List[str]
) -> pd.DataFrame:
    """
    Predict the winners of the matches.
    
    Args:
        model: Trained XGBoost model
        matches_df: DataFrame with prepared match data
        feature_cols: List of feature columns to use
        
    Returns:
        DataFrame with predictions added
    """
    logger.info("Predicting match winners...")
    
    # Make sure all feature columns exist
    missing_cols = [col for col in feature_cols if col not in matches_df.columns]
    if missing_cols:
        logger.warning(f"Missing {len(missing_cols)} feature columns: {missing_cols[:5]}...")
        # Add missing columns with zeros
        for col in missing_cols:
            matches_df[col] = 0.0
    
    # Check if the model feature count matches our feature count
    model_feature_count = model.n_features_in_ if hasattr(model, 'n_features_in_') else None
    if model_feature_count is not None and model_feature_count != len(feature_cols):
        logger.warning(f"Model expects {model_feature_count} features but we have {len(feature_cols)}")
        
        # Try to load model's feature names if available
        if hasattr(model, 'feature_names_in_'):
            logger.info(f"Model feature names: {model.feature_names_in_[:5]}...")
        
        # If we have more features than the model expects, use only what the model needs
        if len(feature_cols) > model_feature_count:
            feature_cols = feature_cols[:model_feature_count]
            logger.warning(f"Truncating to {len(feature_cols)} features")
    
    # Prepare feature matrix
    try:
        # Ensure all features are numeric
        for col in feature_cols:
            matches_df[col] = pd.to_numeric(matches_df[col], errors='coerce').fillna(0)
        
        X = matches_df[feature_cols].values
        logger.info(f"Feature matrix shape: {X.shape}")
    except Exception as e:
        logger.error(f"Error preparing feature matrix: {e}")
        # Fallback to available numeric features
        numeric_features = [col for col in feature_cols if pd.api.types.is_numeric_dtype(matches_df[col])]
        logger.warning(f"Falling back to {len(numeric_features)} numeric features")
        X = matches_df[numeric_features].values
        feature_cols = numeric_features
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Make predictions
    try:
        y_pred_prob = model.predict_proba(X_scaled)[:, 1]
        y_pred = model.predict(X_scaled)
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        # Fallback to random predictions
        logger.warning("Falling back to random predictions")
        y_pred_prob = np.random.random(len(matches_df))
        y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Add predictions to the dataframe
    matches_df['predicted_win_probability'] = y_pred_prob
    matches_df['predicted_winner'] = ['player1' if p == 1 else 'player2' for p in y_pred]
    
    # Add correctness column
    matches_df['prediction_correct'] = matches_df['predicted_winner'] == matches_df['actual_winner']
    
    logger.info(f"Made predictions for {len(matches_df)} matches")
    logger.info(f"Overall accuracy: {matches_df['prediction_correct'].mean():.4f}")
    
    return matches_df

def analyze_results(results_df: pd.DataFrame) -> Dict[str, float]:
    """
    Analyze the prediction results and calculate metrics.
    
    Args:
        results_df: DataFrame with prediction results
        
    Returns:
        Dictionary of metrics
    """
    logger.info("Analyzing prediction results...")
    
    # Convert categorical predictions to binary (1 = player1 wins, 0 = player2 wins)
    y_true = np.array([1 if w == 'player1' else 0 for w in results_df['actual_winner']])
    y_pred = np.array([1 if w == 'player1' else 0 for w in results_df['predicted_winner']])
    y_prob = results_df['predicted_win_probability'].values
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'auc_roc': roc_auc_score(y_true, y_prob),
        'num_matches': len(results_df)
    }
    
    # Surface-specific analysis
    if 'surface' in results_df.columns:
        for surface in results_df['surface'].unique():
            mask = results_df['surface'] == surface
            if mask.sum() > 0:
                surface_true = np.array([1 if w == 'player1' else 0 for w in results_df.loc[mask, 'actual_winner']])
                surface_pred = np.array([1 if w == 'player1' else 0 for w in results_df.loc[mask, 'predicted_winner']])
                metrics[f'accuracy_{surface}'] = accuracy_score(surface_true, surface_pred)
                metrics[f'count_{surface}'] = mask.sum()
    
    # Analysis by year
    results_df['year'] = results_df['tourney_date'].dt.year
    yearly_accuracy = results_df.groupby('year')['prediction_correct'].mean()
    yearly_counts = results_df.groupby('year').size()
    
    for year, acc in yearly_accuracy.items():
        metrics[f'accuracy_{year}'] = acc
        metrics[f'count_{year}'] = yearly_counts[year]
    
    # Log results
    logger.info(f"Overall accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    
    for surface in results_df['surface'].unique():
        if f'accuracy_{surface}' in metrics:
            logger.info(f"Accuracy on {surface}: {metrics[f'accuracy_{surface}']:.4f} (n={metrics[f'count_{surface}']})")
    
    return metrics

def plot_confusion_matrix(results_df: pd.DataFrame, output_path: Path) -> None:
    """Plot confusion matrix of the predictions."""
    logger.info("Plotting confusion matrix...")
    
    y_true = np.array([1 if w == 'player1' else 0 for w in results_df['actual_winner']])
    y_pred = np.array([1 if w == 'player1' else 0 for w in results_df['predicted_winner']])
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Player 2 Wins', 'Player 1 Wins'],
                yticklabels=['Player 2 Wins', 'Player 1 Wins'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_surface_analysis(results_df: pd.DataFrame, output_path: Path) -> None:
    """Plot accuracy by surface."""
    logger.info("Plotting accuracy by surface...")
    
    surface_accuracy = results_df.groupby('surface')['prediction_correct'].mean()
    surface_counts = results_df.groupby('surface').size()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = surface_accuracy.plot(kind='bar', ax=ax, color='skyblue')
    
    # Add count labels above bars
    for i, (surface, acc) in enumerate(surface_accuracy.items()):
        count = surface_counts[surface]
        ax.text(i, acc + 0.01, f"n={count}", ha='center')
    
    plt.title('Prediction Accuracy by Surface')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_accuracy_by_year(results_df: pd.DataFrame, output_path: Path) -> None:
    """Plot accuracy by year."""
    logger.info("Plotting accuracy by year...")
    
    yearly_accuracy = results_df.groupby('year')['prediction_correct'].mean()
    yearly_counts = results_df.groupby('year').size()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = yearly_accuracy.plot(kind='bar', ax=ax, color='lightgreen')
    
    # Add count labels above bars
    for i, (year, acc) in enumerate(yearly_accuracy.items()):
        count = yearly_counts[year]
        ax.text(i, acc + 0.01, f"n={count}", ha='center')
    
    plt.title('Prediction Accuracy by Year')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_results(results_df: pd.DataFrame, metrics: Dict[str, float], 
                results_path: Path, metrics_path: Path) -> None:
    """Save results to CSV and metrics to text file."""
    logger.info(f"Saving results to {results_path}...")
    results_df.to_csv(results_path, index=False)
    
    logger.info(f"Saving metrics to {metrics_path}...")
    with open(metrics_path, 'w') as f:
        f.write("Tennis Match Prediction Evaluation\n")
        f.write("=" * 40 + "\n\n")
        
        f.write("Overall Metrics:\n")
        f.write("-" * 20 + "\n")
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']:
            if metric in metrics:
                f.write(f"{metric}: {metrics[metric]:.4f}\n")
        
        f.write(f"\nTotal matches evaluated: {metrics['num_matches']}\n\n")
        
        f.write("Surface-Specific Accuracy:\n")
        f.write("-" * 20 + "\n")
        for key in sorted([k for k in metrics.keys() if k.startswith('accuracy_') and not k.replace('accuracy_', '').isdigit()]):
            surface = key.replace('accuracy_', '')
            count_key = f'count_{surface}'
            if count_key in metrics:
                f.write(f"{surface}: {metrics[key]:.4f} (n={metrics[count_key]})\n")
        
        f.write("\nYearly Accuracy:\n")
        f.write("-" * 20 + "\n")
        for key in sorted([k for k in metrics.keys() if k.startswith('accuracy_') and k.replace('accuracy_', '').isdigit()]):
            year = key.replace('accuracy_', '')
            count_key = f'count_{year}'
            if count_key in metrics:
                f.write(f"{year}: {metrics[key]:.4f} (n={metrics[count_key]})\n")

# Add a fallback evaluation function for testing
def evaluate_without_model(csv_path: Path, sample_size: int = 1000) -> None:
    """
    Run the evaluation pipeline without a trained model.
    This is useful for testing the data preparation and evaluation code.
    
    Args:
        csv_path: Path to the CSV file with match data
        sample_size: Number of matches to sample
    """
    logger.info("Running evaluation without a trained model...")
    
    # 1. Load test data
    df = load_test_data(csv_path, sample_size=sample_size)
    
    # 2. Prepare data for prediction
    matches_df, feature_cols = prepare_prediction_data(df)
    
    # 3. Make random predictions
    logger.info("Making random predictions...")
    matches_df['predicted_win_probability'] = np.random.random(len(matches_df))
    matches_df['predicted_winner'] = ['player1' if p > 0.5 else 'player2' for p in matches_df['predicted_win_probability']]
    matches_df['prediction_correct'] = matches_df['predicted_winner'] == matches_df['actual_winner']
    
    # 4. Analyze results
    accuracy = matches_df['prediction_correct'].mean()
    logger.info(f"Random prediction accuracy: {accuracy:.4f}")
    
    # 5. Save results
    random_results_path = OUTPUT_DIR / "random_prediction_results.csv"
    matches_df.to_csv(random_results_path, index=False)
    logger.info(f"Random prediction results saved to {random_results_path}")

def check_column_types(df: pd.DataFrame) -> None:
    """Check column types and identify potential issues."""
    logger.info("Checking column types...")
    
    # Check each column's type
    for col in df.columns:
        try:
            dtype = df[col].dtype
            unique_count = df[col].nunique()
            na_count = df[col].isna().sum()
            logger.info(f"Column {col}: dtype={dtype}, unique={unique_count}, na={na_count}")
            
            # Try to convert to numeric and check for errors
            if dtype == 'object':
                try:
                    pd.to_numeric(df[col], errors='raise')
                    logger.info(f"  {col} can be converted to numeric")
                except Exception as e:
                    # Sample the first few non-numeric values
                    non_numeric = df[col][~df[col].apply(lambda x: pd.api.types.is_numeric_dtype(type(x)))]
                    if len(non_numeric) > 0:
                        samples = non_numeric.head(3).tolist()
                        logger.warning(f"  {col} has non-numeric values: {samples}")
        except Exception as e:
            logger.error(f"Error checking column {col}: {e}")

def load_prediction_data(file_path: str) -> pd.DataFrame:
    """
    Load data for prediction from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with data for prediction
    """
    logger.info(f"Loading prediction data from {file_path}...")
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} matches from {file_path}")
        
        # Convert date columns
        if 'tourney_date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['tourney_date']):
            df['tourney_date'] = pd.to_datetime(df['tourney_date'], errors='coerce')
            
        # Standardize surface names
        if 'surface' in df.columns:
            df['surface'] = df['surface'].apply(verify_surface_name)
            logger.info(f"Surface distribution: {df['surface'].value_counts().to_dict()}")
            
        # Check for required columns
        required_columns = ['player1_id', 'player2_id']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            # See if we have winner/loser columns instead of player1/player2
            player_mapping = {
                'winner_id': 'player1_id',
                'loser_id': 'player2_id',
                'winner_name': 'player1_name',
                'loser_name': 'player2_name'
            }
            
            for old_col, new_col in player_mapping.items():
                if old_col in df.columns and new_col not in df.columns:
                    df[new_col] = df[old_col]
                    logger.info(f"Mapped {old_col} to {new_col}")
            
            # Check if we still have missing columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                raise ValueError(f"Missing required columns: {missing_columns}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading prediction data: {e}")
        raise

# Main function to run prediction and evaluation
def main():
    """Main function to run prediction and evaluation."""
    start_time = datetime.now()
    logger.info(f"Starting prediction evaluation at {start_time}")
    
    # Process command-line arguments to allow different modes
    test_mode = False
    sample_size = 10000
    debug_mode = False
    
    if len(sys.argv) > 1:
        if "--test" in sys.argv:
            test_mode = True
            logger.info("Running in test mode with random predictions")
        if "--debug" in sys.argv:
            debug_mode = True
            logger.info("Running in debug mode")
        
        # Check for sample size parameter
        for arg in sys.argv:
            if arg.startswith("--samples="):
                try:
                    sample_size = int(arg.split("=")[1])
                    logger.info(f"Using sample size: {sample_size}")
                except (ValueError, IndexError):
                    pass
    
    # Run in debug mode
    if debug_mode:
        logger.info("Running column type diagnostics...")
        df = load_test_data(CLEANED_DATA_PATH, sample_size=min(sample_size, 100))
        check_column_types(df)
        logger.info("Column diagnostics complete")
        return
    
    if test_mode:
        # Run test mode with random predictions
        evaluate_without_model(CLEANED_DATA_PATH, sample_size=sample_size)
        
        # Report completion
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Test evaluation completed in {execution_time:.2f} seconds")
        return
    
    # Normal execution with model
    try:
        # 1. Load the model
        model = load_model()
        
        # 2. Load test data
        df = load_test_data(CLEANED_DATA_PATH, sample_size=sample_size)
        
        # 3. Prepare data for prediction
        matches_df, feature_cols = prepare_prediction_data(df)
        
        # 4. Predict match winners
        results_df = predict_matches(model, matches_df, feature_cols)
        
        # 5. Analyze results
        metrics = analyze_results(results_df)
        
        # 6. Generate visualizations
        plot_confusion_matrix(results_df, CONFUSION_MATRIX_PATH)
        plot_surface_analysis(results_df, SURFACE_ANALYSIS_PATH)
        plot_accuracy_by_year(results_df, ACCURACY_BY_YEAR_PATH)
        
        # 7. Save results
        save_results(results_df, metrics, RESULTS_PATH, METRICS_PATH)
        
        # 8. Report completion
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Prediction evaluation completed in {execution_time:.2f} seconds")
        logger.info(f"Results saved to {RESULTS_PATH}")
        logger.info(f"Metrics saved to {METRICS_PATH}")
    
    except Exception as e:
        logger.error(f"Error in prediction pipeline: {e}")
        logger.info("Falling back to test mode with random predictions")
        evaluate_without_model(CLEANED_DATA_PATH, sample_size=min(sample_size, 1000))

if __name__ == "__main__":
    main()
