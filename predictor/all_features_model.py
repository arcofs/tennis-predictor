import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import xgboost as xgb
import shap
import logging
from pydantic import BaseModel

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
INPUT_FILE = DATA_DIR / "enhanced_features_v2.csv"
MODEL_FILE = MODELS_DIR / "tennis_predictor_all_features.xgb"
FEATURE_IMPORTANCE_PLOT = OUTPUT_DIR / "all_features_importance.png"
SHAP_PLOT = OUTPUT_DIR / "all_features_shap.png"
TRAINING_METRICS = OUTPUT_DIR / "all_features_metrics.txt"

# Try to detect GPU
def detect_gpu_availability() -> bool:
    """Check if GPU is available for XGBoost."""
    try:
        # Create a small test XGBoost model with GPU
        test_data = np.random.rand(10, 5)
        test_labels = np.random.randint(0, 2, 10)
        
        # Set up a minimal GPU model
        test_model = xgb.XGBClassifier(tree_method='gpu_hist', n_estimators=1)
        
        # Try to train it
        test_model.fit(test_data, test_labels)
        logger.info("GPU is available and will be used for training")
        return True
    except Exception as e:
        logger.info(f"GPU is not available (reason: {str(e)}). Using CPU instead")
        return False

# Detect GPU availability
GPU_AVAILABLE = detect_gpu_availability()
TREE_METHOD = 'gpu_hist' if GPU_AVAILABLE else 'hist'

class ModelConfig(BaseModel):
    """Configuration for the XGBoost model with anti-overfitting settings"""
    learning_rate: float = 0.01
    max_depth: int = 4
    min_child_weight: int = 3
    subsample: float = 0.7
    colsample_bytree: float = 0.7
    n_estimators: int = 500
    early_stopping_rounds: int = 50
    eval_metric: str = "logloss"
    objective: str = "binary:logistic"
    tree_method: str = TREE_METHOD
    gamma: float = 0.1
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    random_state: int = 42
    n_jobs: int = -1
    missing: float = np.nan
    enable_categorical: bool = True

def load_and_prepare_data() -> Tuple[pd.DataFrame, List[str]]:
    """Load and prepare data for training with all features"""
    logger.info("Loading data...")
    
    # Load the dataset
    df = pd.read_csv(INPUT_FILE)
    
    # Convert date column to datetime
    df['tourney_date'] = pd.to_datetime(df['tourney_date'])
    
    # Sort by date to ensure chronological order
    df = df.sort_values('tourney_date').reset_index(drop=True)
    
    # First, identify if we have winner/loser specific features
    winner_cols = [col for col in df.columns if col.startswith('winner_') and col != 'winner_id']
    loser_cols = [col for col in df.columns if col.startswith('loser_') and col != 'loser_id']
    
    has_player_specific_features = len(winner_cols) > 0 and len(loser_cols) > 0
    
    if has_player_specific_features:
        logger.info("Creating a balanced dataset with player-specific features")
        
        # Create a balanced dataset
        matches = []
        
        # Columns to keep from original dataset
        common_cols = ['tourney_date', 'surface', 'tourney_level']
        
        # Get reliability weight if available
        if 'reliability_weight' in df.columns:
            common_cols.append('reliability_weight')
            logger.info("Using reliability weights for sample importance")
        
        # Process each match
        for _, row in df.iterrows():
            # Extract common features
            common_features = {col: row[col] for col in common_cols if col in row}
            
            # Create player 1 vs player 2 sample (actual match) - label is 1 (player1 won)
            p1_features = {}
            for col in winner_cols:
                feature_name = col.replace('winner_', 'player1_')
                p1_features[feature_name] = row[col]
            
            for col in loser_cols:
                feature_name = col.replace('loser_', 'player2_')
                p1_features[feature_name] = row[col]
            
            # Add player IDs
            p1_features['player1_id'] = row['winner_id'] if 'winner_id' in row else 0
            p1_features['player2_id'] = row['loser_id'] if 'loser_id' in row else 0
            
            # Add label - player1 won
            p1_features['result'] = 1
            
            # Combine with common features
            p1_features.update(common_features)
            matches.append(p1_features)
            
            # Create player 2 vs player 1 sample (flipped match) - label is 0 (player2 lost)
            p2_features = {}
            for col in loser_cols:
                feature_name = col.replace('loser_', 'player1_')
                p2_features[feature_name] = row[col]
            
            for col in winner_cols:
                feature_name = col.replace('winner_', 'player2_')
                p2_features[feature_name] = row[col]
            
            # Add player IDs
            p2_features['player1_id'] = row['loser_id'] if 'loser_id' in row else 0
            p2_features['player2_id'] = row['winner_id'] if 'winner_id' in row else 0
            
            # Add label - player1 lost
            p2_features['result'] = 0
            
            # Combine with common features
            p2_features.update(common_features)
            matches.append(p2_features)
        
        # Create a new dataframe with balanced data
        balanced_df = pd.DataFrame(matches)
        
        # Get feature columns
        feature_cols = []
        
        # Create feature differences (player1 - player2)
        # These are more useful for prediction than raw values
        for p1_col in [col for col in balanced_df.columns if col.startswith('player1_') and col != 'player1_id']:
            feature_name = p1_col[8:]  # Remove 'player1_' prefix
            p2_col = f'player2_{feature_name}'
            
            if p2_col in balanced_df.columns:
                diff_col = f'{feature_name}_diff'
                balanced_df[diff_col] = balanced_df[p1_col] - balanced_df[p2_col]
                feature_cols.append(diff_col)
        
        # Get the final dataset
        df = balanced_df
        
    else:
        logger.info("Using existing difference features from the dataset")
        
        # Get all relevant feature columns
        exclude_cols = [
            'tourney_date', 'winner_id', 'loser_id', 'surface', 'tourney_level',
            'index', 'Unnamed: 0', 'match_id', 'result'
        ]
        
        # Use ALL difference features for prediction (not just those ending with '_diff')
        # Include both raw difference features and imputed flags
        all_features = [col for col in df.columns if col not in exclude_cols]
        feature_cols = all_features
            
        # If we don't have a result column but need to create one
        if 'result' not in df.columns:
            df['result'] = 1  # All matches are wins for player1
    
    # Check for potentially leaky features and warn
    leakage_keywords = ['winner', 'loser', 'score', 'sets_won', 'games_won']
    potential_leaky_features = []
    
    for col in feature_cols:
        for keyword in leakage_keywords:
            if keyword in col.lower() and col not in ['win_rate_diff', 'h2h_win_pct_diff'] and not col.startswith('win_rate_'):
                potential_leaky_features.append(col)
                break
    
    if potential_leaky_features:
        logger.warning(f"WARNING: Potential data leakage in features: {potential_leaky_features}")
        logger.warning("Removing these features to prevent unwanted data leakage")
        
        # Exclude potential leaky features from the feature list
        safe_features = [col for col in feature_cols if col not in potential_leaky_features]
        logger.info(f"Using {len(safe_features)} safe features after removing potential leakage")
        feature_cols = safe_features
    
    logger.info(f"Processed dataset: {len(df)} samples with {len(feature_cols)} features")
    logger.info(f"Date range: {df['tourney_date'].min()} to {df['tourney_date'].max()}")
    logger.info(f"Example features: {feature_cols[:5]}...")
    
    if 'result' in df.columns:
        logger.info(f"Class distribution: {df['result'].value_counts().to_dict()}")
    
    return df, feature_cols

def create_time_based_split(df: pd.DataFrame, val_date: str = '2022-01-01', test_date: str = '2023-01-01') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a time-based split of the data with validation set using explicit dates."""
    # Convert dates to datetime if they aren't already
    if not pd.api.types.is_datetime64_dtype(df['tourney_date']):
        df['tourney_date'] = pd.to_datetime(df['tourney_date'])
    
    # Split the data by date
    train_df = df[df['tourney_date'] < pd.to_datetime(val_date)]
    val_df = df[(df['tourney_date'] >= pd.to_datetime(val_date)) & 
                (df['tourney_date'] < pd.to_datetime(test_date))]
    test_df = df[df['tourney_date'] >= pd.to_datetime(test_date)]
    
    logger.info(f"Training set: {len(train_df)} matches ({train_df['tourney_date'].min().date()} to {train_df['tourney_date'].max().date()})")
    logger.info(f"Validation set: {len(val_df)} matches ({val_df['tourney_date'].min().date()} to {val_df['tourney_date'].max().date()})")
    logger.info(f"Test set: {len(test_df)} matches ({test_df['tourney_date'].min().date()} to {test_df['tourney_date'].max().date()})")
    
    return train_df, val_df, test_df

def prepare_features(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare features and target for training."""
    # Check if we have the necessary columns
    if not all(col in df.columns for col in feature_cols):
        missing = [col for col in feature_cols if col not in df.columns]
        logger.error(f"Missing columns: {missing}")
        # Filter to only existing columns
        feature_cols = [col for col in feature_cols if col in df.columns]
    
    # Use only the feature columns - keep NaN values for XGBoost to handle properly
    X = df[feature_cols].values
    
    # Get the target variable (result)
    if 'result' in df.columns:
        y = df['result'].values
    else:
        # Fallback if no result column (should not happen with our updated logic)
        logger.warning("No 'result' column found, creating synthetic target")
        y = np.ones(len(df))
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target vector shape: {y.shape}")
    logger.info(f"Target distribution: {np.bincount(y.astype(int))}")
    
    return X, y

def scale_features(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    
    # Make a copy of original data to avoid modifying it
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()
    
    # Find columns with no missing values to scale
    non_missing_cols = ~np.isnan(X_train).any(axis=0)
    
    # Only scale columns with no missing values
    if np.any(non_missing_cols):
        X_train_scaled[:, non_missing_cols] = scaler.fit_transform(X_train[:, non_missing_cols])
        X_val_scaled[:, non_missing_cols] = scaler.transform(X_val[:, non_missing_cols])
        X_test_scaled[:, non_missing_cols] = scaler.transform(X_test[:, non_missing_cols])
    
    return X_train_scaled, X_val_scaled, X_test_scaled

def train_model(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                feature_names: List[str], sample_weights: Optional[np.ndarray] = None) -> Tuple[xgb.XGBClassifier, Dict]:
    """Train the XGBoost model with anti-overfitting configuration."""
    logger.info("Training final model...")
    
    # Use anti-overfitting configuration
    config = ModelConfig()
    model = xgb.XGBClassifier(
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        min_child_weight=config.min_child_weight,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        n_estimators=config.n_estimators,
        early_stopping_rounds=config.early_stopping_rounds,
        eval_metric=config.eval_metric,
        objective=config.objective,
        tree_method=config.tree_method,
        gamma=config.gamma,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        missing=config.missing,
        enable_categorical=config.enable_categorical
    )
    
    # Use sample weights if provided
    if sample_weights is not None:
        logger.info("Using sample weights for training")
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            sample_weight=sample_weights,
            verbose=True
        )
    else:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=True
        )
    
    # Get feature importance
    importance = model.feature_importances_
    feature_importance = dict(zip(feature_names, importance))
    
    return model, feature_importance

def evaluate_model(model: xgb.XGBClassifier, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_pred_proba)
    }
    
    return metrics

def plot_feature_importance(feature_importance: Dict[str, float], output_path: Path) -> None:
    """Plot feature importance."""
    plt.figure(figsize=(14, 10))
    importance_df = pd.DataFrame({
        'feature': list(feature_importance.keys()),
        'importance': list(feature_importance.values())
    }).sort_values('importance', ascending=False)
    
    # Plot top 30 features for better readability
    sns.barplot(data=importance_df.head(30), x='importance', y='feature')
    plt.title('Top 30 Most Important Features')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def analyze_shap_values(model: xgb.XGBClassifier, X_test: np.ndarray, feature_names: List[str],
                       output_path: Path) -> None:
    """Analyze and plot SHAP values."""
    logger.info("Calculating SHAP values...")
    
    # Choose a subset of test data for SHAP analysis if the dataset is large
    max_samples = min(500, X_test.shape[0])
    X_shap = X_test[:max_samples]
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)
    
    plt.figure(figsize=(14, 10))
    shap.summary_plot(shap_values, X_shap, feature_names=feature_names, show=False, max_display=30)
    plt.title('SHAP Feature Importance')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_training_metrics(metrics: Dict[str, float], feature_importance: Dict[str, float],
                         output_path: Path) -> None:
    """Save training metrics and feature importance to a file."""
    with open(output_path, 'w') as f:
        f.write("Tennis Match Prediction Model with All Features\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Training Metrics:\n")
        f.write("-" * 50 + "\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        
        f.write("\nTop 50 Features by Importance:\n")
        f.write("-" * 50 + "\n")
        importance_df = pd.DataFrame({
            'feature': list(feature_importance.keys()),
            'importance': list(feature_importance.values())
        }).sort_values('importance', ascending=False)
        
        for _, row in importance_df.head(50).iterrows():
            f.write(f"{row['feature']}: {row['importance']:.6f}\n")

def main() -> None:
    """Main function to train the model and analyze results with all features."""
    logger.info("Starting tennis match prediction model training with all features")
    
    # Log GPU/CPU usage
    logger.info(f"Using {'GPU' if GPU_AVAILABLE else 'CPU'} for XGBoost training (tree_method={TREE_METHOD})")
    
    # 1. Load and prepare data
    df, feature_cols = load_and_prepare_data()
    
    # 2. Create time-based split with separate validation set
    train_df, val_df, test_df = create_time_based_split(df)
    
    # 3. Prepare features
    X_train, y_train = prepare_features(train_df, feature_cols)
    X_val, y_val = prepare_features(val_df, feature_cols)
    X_test, y_test = prepare_features(test_df, feature_cols)
    
    # Extract sample weights if available
    sample_weights_train = None
    if 'reliability_weight' in train_df.columns:
        logger.info("Using reliability weights for model training")
        sample_weights_train = train_df['reliability_weight'].values
    
    # 4. Scale features
    X_train_scaled, X_val_scaled, X_test_scaled = scale_features(X_train, X_val, X_test)
    
    # 5. Analyze feature correlations to detect potentially redundant features
    logger.info("Analyzing feature correlations...")
    
    # Convert to DataFrame for correlation analysis, handling NaN values
    feature_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
    
    # 6. Train model with all features
    logger.info("Training model with all features...")
    model, feature_importance = train_model(
        X_train_scaled, y_train, 
        X_val_scaled, y_val,
        feature_cols,
        sample_weights_train
    )
    
    # 7. Evaluate model
    logger.info("Evaluating model on test set...")
    metrics = evaluate_model(model, X_test_scaled, y_test)
    
    logger.info("Model performance metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # 8. Calculate permutation importance for more stable feature importance
    logger.info("Calculating permutation feature importance...")
    perm_importance = permutation_importance(
        model, X_test_scaled, y_test,
        n_repeats=10,
        random_state=42
    )
    
    perm_importance_values = perm_importance.importances_mean
    perm_feature_importance = dict(zip(feature_cols, perm_importance_values))
    
    # 9. Plot feature importance
    plot_feature_importance(perm_feature_importance, FEATURE_IMPORTANCE_PLOT)
    
    # 10. Analyze SHAP values
    analyze_shap_values(model, X_test_scaled, feature_cols, SHAP_PLOT)
    
    # 11. Save results
    save_training_metrics(metrics, perm_feature_importance, TRAINING_METRICS)
    
    # 12. Save model
    model.save_model(MODEL_FILE)
    
    logger.info(f"Model saved to: {MODEL_FILE}")
    logger.info(f"Feature importance plot saved to: {FEATURE_IMPORTANCE_PLOT}")
    logger.info(f"SHAP values plot saved to: {SHAP_PLOT}")
    logger.info(f"Training metrics saved to: {TRAINING_METRICS}")
    
    # 13. Print top 20 most important features
    top_features = sorted(perm_feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]
    logger.info("\nTop 20 Most Important Features:")
    for feature, importance in top_features:
        logger.info(f"{feature}: {importance:.6f}")

if __name__ == "__main__":
    main() 