import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
from xgboost import plot_importance
import shap
import optuna
from pydantic import BaseModel
import logging
import warnings
warnings.filterwarnings('ignore')

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
MODEL_FILE = MODELS_DIR / "tennis_predictor.xgb"
FEATURE_IMPORTANCE_PLOT = OUTPUT_DIR / "feature_importance.png"
SHAP_PLOT = OUTPUT_DIR / "shap_importance.png"
TRAINING_METRICS = OUTPUT_DIR / "training_metrics.txt"

def detect_gpu_availability() -> bool:
    """Check if GPU is available for XGBoost."""
    try:
        # Create a small test XGBoost model with GPU
        test_data = np.random.rand(10, 5)
        test_labels = np.random.randint(0, 2, 10)
        
        # Set up a minimal GPU model
        test_model = xgb.XGBClassifier(
            tree_method='gpu_hist',
            n_estimators=1
        )
        
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
    """Configuration for the XGBoost model."""
    learning_rate: float = 0.01
    max_depth: int = 4  # Reduced from 6 to prevent overfitting
    min_child_weight: int = 3  # Increased from 1 to prevent overfitting
    subsample: float = 0.7  # Reduced from 0.8 to prevent overfitting
    colsample_bytree: float = 0.7  # Reduced from 0.8 to prevent overfitting
    n_estimators: int = 500  # Reduced from 1000 to prevent overfitting
    early_stopping_rounds: int = 50
    eval_metric: str = "logloss"
    objective: str = "binary:logistic"
    tree_method: str = TREE_METHOD  # Use GPU if available, otherwise CPU
    gamma: float = 0.1  # Added to reduce overfitting
    reg_alpha: float = 0.1  # L1 regularization
    reg_lambda: float = 1.0  # L2 regularization
    random_state: int = 42
    n_jobs: int = -1

class FeatureAnalysis:
    """Class to analyze and manage features."""
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.feature_importance: Dict[str, float] = {}
        self.shap_values: Optional[np.ndarray] = None
        self.feature_names: List[str] = []
        
    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns, excluding non-feature columns."""
        exclude_cols = [
            'tourney_date', 'winner_id', 'loser_id', 'surface', 'tourney_level',
            'index', 'Unnamed: 0', 'match_id'
        ]
        return [col for col in self.df.columns if col not in exclude_cols]
    
    def analyze_feature_correlations(self) -> pd.DataFrame:
        """Analyze correlations between features."""
        feature_cols = self.get_feature_columns()
        return self.df[feature_cols].corr()
    
    def plot_feature_correlations(self, output_path: Path) -> None:
        """Plot feature correlation heatmap."""
        corr_matrix = self.analyze_feature_correlations()
        plt.figure(figsize=(15, 12))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def select_important_features(self, X_train: np.ndarray, y_train: np.ndarray, 
                                feature_names: List[str], threshold: str = 'mean') -> List[str]:
        """Select important features using XGBoost's feature importance."""
        logger.info("Selecting important features...")
        # Train a simple model to get feature importances
        model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            tree_method=TREE_METHOD,  # Use GPU if available
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Use SelectFromModel to select features
        selector = SelectFromModel(model, threshold=threshold, prefit=True)
        support = selector.get_support()
        
        # Get selected feature names
        selected_features = [feature_names[i] for i in range(len(feature_names)) if support[i]]
        logger.info(f"Selected {len(selected_features)}/{len(feature_names)} features")
        logger.info(f"Selected features: {selected_features[:10]}...")
        
        return selected_features

def load_and_prepare_data() -> Tuple[pd.DataFrame, List[str]]:
    """Load and prepare data for training."""
    logger.info("Loading data...")
    
    # Load the dataset
    df = pd.read_csv(INPUT_FILE)
    
    # Load feature metadata if available
    feature_metadata = {}
    metadata_path = OUTPUT_DIR / "feature_metadata.json"
    if metadata_path.exists():
        try:
            import json
            with open(metadata_path, 'r') as f:
                feature_metadata = json.load(f)
            logger.info(f"Loaded feature metadata with {len(feature_metadata)} features")
        except Exception as e:
            logger.warning(f"Could not load feature metadata: {e}")
    
    # Convert date column to datetime
    df['tourney_date'] = pd.to_datetime(df['tourney_date'])
    
    # Sort by date to ensure chronological order
    df = df.sort_values('tourney_date').reset_index(drop=True)
    
    # Check data structure
    logger.info(f"Columns in the dataset: {df.columns.tolist()}")
    
    # Create a balanced dataset with both positive and negative examples
    # For each match, we'll create two rows:
    # 1. Player1 vs Player2 (actual match) - label is 1 (player1 won)
    # 2. Player2 vs Player1 (flipped match) - label is 0 (player2 lost)
    
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
        
        # If we already have pre-calculated difference features, use them
        # Get all relevant feature columns from metadata if available
        if feature_metadata:
            feature_cols = [col for col in feature_metadata if feature_metadata[col].get('is_prediction_feature', False)]
            logger.info(f"Using {len(feature_cols)} prediction features from metadata")
        else:
            # Get all relevant feature columns
            exclude_cols = [
                'tourney_date', 'winner_id', 'loser_id', 'surface', 'tourney_level',
                'index', 'Unnamed: 0', 'match_id', 'result'
            ]
            # Use only difference features for prediction
            feature_cols = [col for col in df.columns if col.endswith('_diff') and not col.endswith('_imputed') 
                           and col not in exclude_cols]
            
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
        logger.warning("Consider excluding these features to prevent unwanted data leakage")
        
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

def create_time_based_split(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a time-based split of the data with validation set."""
    # Calculate split points
    total_len = len(df)
    test_idx = int(total_len * (1 - test_size))
    val_idx = int(test_idx * (1 - val_size))
    
    # Split the data
    train_df = df.iloc[:val_idx]
    val_df = df.iloc[val_idx:test_idx]
    test_df = df.iloc[test_idx:]
    
    logger.info(f"Training set: {len(train_df)} matches ({train_df['tourney_date'].min()} to {train_df['tourney_date'].max()})")
    logger.info(f"Validation set: {len(val_df)} matches ({val_df['tourney_date'].min()} to {val_df['tourney_date'].max()})")
    logger.info(f"Test set: {len(test_df)} matches ({test_df['tourney_date'].min()} to {test_df['tourney_date'].max()})")
    
    return train_df, val_df, test_df

def prepare_features(df: pd.DataFrame, feature_cols: List[str], selected_features: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare features and target for training."""
    # Use selected features if provided
    if selected_features:
        use_cols = selected_features
    else:
        use_cols = feature_cols
    
    # Check if we have the necessary columns
    if not all(col in df.columns for col in use_cols):
        missing = [col for col in use_cols if col not in df.columns]
        logger.error(f"Missing columns: {missing}")
        # Filter to only existing columns
        use_cols = [col for col in use_cols if col in df.columns]
    
    # Use only the feature columns
    X = df[use_cols].fillna(0).values
    
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
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled

def objective(trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> float:
    """Objective function for Optuna hyperparameter optimization."""
    param = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 2, 6),  # Reduced max depth range
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),  # Increased range
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'gamma': trial.suggest_float('gamma', 0, 0.5),  # Added gamma parameter
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),  # Added L1 regularization
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0),  # Added L2 regularization
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'tree_method': TREE_METHOD,  # Use GPU if available, otherwise CPU
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'early_stopping_rounds': 50,
        'random_state': 42
    }
    
    model = xgb.XGBClassifier(**param)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Use AUC-ROC score for optimization as it's better for imbalanced data
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, y_pred_proba)

def optimize_hyperparameters(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Dict:
    """Optimize hyperparameters using Optuna."""
    logger.info("Starting hyperparameter optimization...")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=30  # Reduced from 50 to save time
    )
    
    best_params = study.best_params
    best_params.update({
        'tree_method': TREE_METHOD,  # Use GPU if available, otherwise CPU
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'early_stopping_rounds': 50,
        'random_state': 42
    })
    
    logger.info(f"Best parameters: {best_params}")
    return best_params

def train_model(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                feature_names: List[str], best_params: Dict, sample_weights: Optional[np.ndarray] = None) -> Tuple[xgb.XGBClassifier, Dict]:
    """Train the XGBoost model with the best parameters."""
    logger.info("Training final model...")
    
    model = xgb.XGBClassifier(**best_params)
    
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
    plt.figure(figsize=(12, 8))
    importance_df = pd.DataFrame({
        'feature': list(feature_importance.keys()),
        'importance': list(feature_importance.values())
    }).sort_values('importance', ascending=False)
    
    sns.barplot(data=importance_df.head(20), x='importance', y='feature')
    plt.title('Top 20 Most Important Features')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def analyze_shap_values(model: xgb.XGBClassifier, X_test: np.ndarray, feature_names: List[str],
                       output_path: Path) -> None:
    """Analyze and plot SHAP values."""
    logger.info("Calculating SHAP values...")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title('SHAP Feature Importance')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_training_metrics(metrics: Dict[str, float], feature_importance: Dict[str, float],
                         output_path: Path) -> None:
    """Save training metrics and feature importance to a file."""
    with open(output_path, 'w') as f:
        f.write("Training Metrics:\n")
        f.write("=" * 50 + "\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        
        f.write("\nFeature Importance:\n")
        f.write("=" * 50 + "\n")
        importance_df = pd.DataFrame({
            'feature': list(feature_importance.keys()),
            'importance': list(feature_importance.values())
        }).sort_values('importance', ascending=False)
        
        for _, row in importance_df.iterrows():
            f.write(f"{row['feature']}: {row['importance']:.4f}\n")

def main() -> None:
    """Main function to train the model and analyze results."""
    start_time = time.time()
    
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
    corr_matrix = pd.DataFrame(X_train_scaled, columns=feature_cols).corr()
    
    # Find highly correlated features (>0.95)
    high_corr_features = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.95:
                colname = corr_matrix.columns[i]
                high_corr_features.add(colname)
    
    if high_corr_features:
        logger.info(f"Found {len(high_corr_features)} highly correlated features")
        # Remove these features if desired
        # feature_cols = [col for col in feature_cols if col not in high_corr_features]
    
    # 6. Feature selection to reduce complexity
    # Initialize FeatureAnalysis
    feature_analyzer = FeatureAnalysis(df)
    
    # Select important features
    selected_features = feature_analyzer.select_important_features(
        X_train_scaled, y_train, feature_cols
    )
    
    # 7. Re-prepare features using only selected features
    X_train, y_train = prepare_features(train_df, feature_cols, selected_features)
    X_val, y_val = prepare_features(val_df, feature_cols, selected_features)
    X_test, y_test = prepare_features(test_df, feature_cols, selected_features)
    
    # 8. Re-scale features
    X_train_scaled, X_val_scaled, X_test_scaled = scale_features(X_train, X_val, X_test)
    
    # 9. Optimize hyperparameters
    logger.info("Optimizing hyperparameters with Optuna...")
    best_params = optimize_hyperparameters(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # 10. Train final model
    logger.info("Training final model...")
    model, feature_importance = train_model(
        X_train_scaled, y_train, 
        X_val_scaled, y_val,
        selected_features,
        best_params,
        sample_weights_train
    )
    
    # 11. Evaluate model
    logger.info("Evaluating model on test set...")
    metrics = evaluate_model(model, X_test_scaled, y_test)
    
    logger.info("Model performance metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # 12. Calculate permutation importance
    logger.info("Calculating permutation feature importance...")
    perm_importance = permutation_importance(
        model, X_test_scaled, y_test,
        n_repeats=10,
        random_state=42
    )
    
    perm_importance_values = perm_importance.importances_mean
    perm_feature_importance = dict(zip(selected_features, perm_importance_values))
    
    # 13. Compare the two methods
    logger.info("Comparing feature importance methods:")
    importance_comparison = pd.DataFrame({
        'feature': selected_features,
        'xgboost_importance': [feature_importance[f] for f in selected_features],
        'permutation_importance': [perm_feature_importance[f] for f in selected_features]
    }).sort_values('permutation_importance', ascending=False)
    
    logger.info("\nTop 10 features by permutation importance:")
    for _, row in importance_comparison.head(10).iterrows():
        logger.info(f"{row['feature']}: {row['permutation_importance']:.6f}")
    
    # 14. Plot feature importance
    plot_feature_importance(perm_feature_importance, FEATURE_IMPORTANCE_PLOT)
    
    # 15. Analyze SHAP values
    logger.info("Calculating SHAP values...")
    analyze_shap_values(model, X_test_scaled, selected_features, SHAP_PLOT)
    
    # 16. Save results
    with open(TRAINING_METRICS, 'w') as f:
        f.write("Tennis Match Prediction Model Metrics:\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Model Performance:\n")
        f.write("-" * 30 + "\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        
        f.write("\nFeature Importance Analysis:\n")
        f.write("-" * 30 + "\n")
        
        f.write("\nXGBoost Feature Importance:\n")
        for feat, imp in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{feat}: {imp:.6f}\n")
        
        f.write("\nPermutation Feature Importance:\n")
        for feat, imp in sorted(perm_feature_importance.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{feat}: {imp:.6f}\n")
    
    # 17. Save model
    model.save_model(MODEL_FILE)
    
    # 18. Print summary
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    logger.info("\nModel Training Summary:")
    logger.info("=" * 50)
    logger.info(f"Total training time: {elapsed_time:.2f} seconds")
    
    logger.info("\nModel Performance:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    logger.info("\nTop 20 Most Important Features:")
    importance_df = pd.DataFrame({
        'feature': list(perm_feature_importance.keys()),
        'importance': list(perm_feature_importance.values())
    }).sort_values('importance', ascending=False)
    
    for _, row in importance_df.head(20).iterrows():
        logger.info(f"{row['feature']}: {row['importance']:.6f}")
    
    logger.info(f"\nModel saved to: {MODEL_FILE}")
    logger.info(f"Feature importance plot saved to: {FEATURE_IMPORTANCE_PLOT}")
    logger.info(f"SHAP values plot saved to: {SHAP_PLOT}")
    logger.info(f"Training metrics saved to: {TRAINING_METRICS}")
    
    # 19. Generate correlation analysis between features
    correlation_plot_path = OUTPUT_DIR / "feature_correlation.png"
    logger.info("Generating feature correlation analysis...")
    
    # Convert scaled features back to DataFrame for correlation analysis
    train_feature_df = pd.DataFrame(X_train_scaled, columns=selected_features)
    
    # Calculate correlation matrix
    corr_matrix = train_feature_df.corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, 
        mask=mask,
        cmap='coolwarm',
        annot=False,
        center=0,
        square=True,
        linewidths=0.5
    )
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(correlation_plot_path)
    plt.close()
    
    logger.info(f"Feature correlation plot saved to: {correlation_plot_path}")

if __name__ == "__main__":
    main()
