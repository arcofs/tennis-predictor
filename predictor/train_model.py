import os
import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Set, Any, cast
from pydantic import BaseModel, Field
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
from tqdm import tqdm

# Print package versions for reproducibility
print(f"pandas: {pd.__version__}")
print(f"xgboost: {xgb.__version__}")

# Define paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "predictor" / "output"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Input and output files
DEFAULT_INPUT_FILE = DATA_DIR / "enhanced_features_v2.csv"
MODEL_OUTPUT = MODELS_DIR / f"xgb_model_focused_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
METRICS_OUTPUT = MODELS_DIR / f"model_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
FEATURE_IMPORTANCE_FILE = RESULTS_DIR / f"feature_importance_focused_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
RESULTS_OUTPUT_FILE = RESULTS_DIR / f"model_results_focused_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# Pydantic models for type validation
class ModelConfig(BaseModel):
    random_state: int = Field(42)
    test_size: float = Field(0.2)
    use_time_split: bool = Field(True)
    n_splits: int = Field(5)
    max_train_size: Optional[int] = Field(None)
    xgb_params: Dict[str, Any] = Field(default_factory=dict)
    n_estimators: int = Field(700, description="Number of boosting rounds")
    max_depth: int = Field(5, description="Maximum tree depth")
    learning_rate: float = Field(0.03, description="Step size shrinkage used to prevent overfitting")
    colsample_bytree: float = Field(0.8, description="Subsample ratio of columns for each tree")
    subsample: float = Field(0.8, description="Subsample ratio of the training instances")
    reg_alpha: float = Field(0.05, description="L1 regularization term on weights")
    reg_lambda: float = Field(1.0, description="L2 regularization term on weights")
    gamma: float = Field(0.1, description="Minimum loss reduction required for a split")
    min_child_weight: float = Field(2.0, description="Minimum sum of instance weight needed in a child")
    scale_pos_weight: float = Field(1.0, description="Control the balance of positive and negative weights")
    
    # Advanced XGBoost parameters
    use_feature_weights: bool = Field(True, description="Whether to use feature weights")
    use_monotonic_constraints: bool = Field(True, description="Whether to use monotonic constraints")
    
    # Training parameters
    validation_fraction: float = Field(0.2, description="Fraction of data to use for validation")
    time_splits: int = Field(5, description="Number of time-based splits for cross-validation")
    hyperparameter_tuning: bool = Field(True, description="Whether to perform hyperparameter tuning")
    tuning_iterations: int = Field(30, description="Number of hyperparameter tuning iterations")
    early_stopping_rounds: int = Field(50, description="Number of early stopping rounds")
    
    class Config:
        validate_assignment = True
        extra = "ignore"

class ModelResults(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    log_loss: float
    train_accuracy: float
    feature_importance: Dict[str, float]
    
    class Config:
        validate_assignment = True

def load_data() -> pd.DataFrame:
    """
    Load the enhanced features dataset.
    
    Returns:
        pd.DataFrame: The loaded dataset
    """
    print(f"Loading data from {DEFAULT_INPUT_FILE}...")
    
    try:
        df = pd.read_csv(DEFAULT_INPUT_FILE)
        print(f"Loaded {len(df)} samples with {df.shape[1]} features")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Preprocess the data to prevent data leakage.
    
    Args:
        df: DataFrame with features
        
    Returns:
        Tuple of processed DataFrame and list of feature columns
    """
    print("Preprocessing data...")
    
    # Convert date column to datetime
    if 'tourney_date' in df.columns:
        df['tourney_date'] = pd.to_datetime(df['tourney_date'])
    
    # Create target variable (1 if winner_id won, which is always true in the data)
    # This is just for consistency, as the data is already structured with winners and losers
    df['target'] = 1
    
    # Remove features that would cause data leakage - comprehensive list
    potential_leaky_features = [
        # Direct match result statistics
        'w_ace', 'w_svpt', 'w_1stWon', 'w_1stIn', 'w_bpSaved', 'w_bpFaced',
        'l_ace', 'l_svpt', 'l_1stWon', 'l_1stIn', 'l_bpSaved', 'l_bpFaced',
        
        # Post-match head-to-head statistics
        'h2h_wins_winner', 'h2h_wins_loser', 'h2h_win_rate_winner', 
        'recent_h2h_win_rate_winner', 'recent_h2h_wins_winner', 'h2h_total_matches',
        
        # Any column containing these patterns might be post-match statistics
        *[col for col in df.columns if 'consecutive_wins' in col or 'consecutive_losses' in col],
        *[col for col in df.columns if 'ace_rate' in col],
        *[col for col in df.columns if 'first_serve_win_rate' in col],
        *[col for col in df.columns if 'bp_save_rate' in col],
        
        # Any winner/loser features that directly reveal match performance
        *[col for col in df.columns if col.startswith('winner_') and any(x in col for x in ['ace', 'svpt', '1stWon', '1stIn', 'bpSaved', 'bpFaced'])],
        *[col for col in df.columns if col.startswith('loser_') and any(x in col for x in ['ace', 'svpt', '1stWon', '1stIn', 'bpSaved', 'bpFaced'])],
    ]
    
    # Filter columns, only remove if they exist
    leaky_cols = [col for col in potential_leaky_features if col in df.columns]
    if leaky_cols:
        print(f"Removing {len(leaky_cols)} potential leaky features")
        print(f"Examples of removed features: {leaky_cols[:5]}...")
        df = df.drop(columns=leaky_cols)
    
    # Keep only the most predictive and fairest features
    key_features = [
        # Basic player ratings that don't leak match outcomes
        'winner_elo', 'loser_elo', 'elo_diff',
        
        # Physical stats (not match performance)
        'winner_ht', 'loser_ht', 'height_diff',
        
        # Pre-match statistics (averaged from historical data)
        *[col for col in df.columns if col.startswith('winner_win_rate_') or col.startswith('loser_win_rate_')],
        *[col for col in df.columns if '_win_rate_' in col and '_diff' in col],
        
        # Surface-specific win rates
        *[col for col in df.columns if 'win_rate_Hard_' in col or 'win_rate_Clay_' in col or 'win_rate_Grass_' in col],
        
        # Tournament level performance (if available)
        *[col for col in df.columns if 'win_rate_ATP_' in col or 'win_rate_GSL_' in col],
    ]
    
    # Keep only features that exist in the dataframe
    available_key_features = [col for col in key_features if col in df.columns]
    
    # Get all feature columns if we don't want to restrict to key features
    non_feature_cols = ['tourney_date', 'winner_id', 'loser_id', 'surface', 'tourney_level', 'target']
    all_feature_cols = [col for col in df.columns if col not in non_feature_cols]
    
    # Use either all non-leaky features or just the key features
    use_only_key_features = False  # Set to True to restrict to only key features
    
    if use_only_key_features and available_key_features:
        print(f"Using only {len(available_key_features)} key predictive features instead of all {len(all_feature_cols)} non-leaky features")
        feature_cols = available_key_features
    else:
        feature_cols = all_feature_cols
    
    # Check for and handle missing values
    missing_values = df[feature_cols].isna().sum()
    if missing_values.sum() > 0:
        print(f"Found {missing_values.sum()} missing values")
        cols_with_missing = missing_values[missing_values > 0].index.tolist()
        print(f"Columns with missing values: {cols_with_missing[:5]}..." if len(cols_with_missing) > 5 else f"Columns with missing values: {cols_with_missing}")
        # Fill missing values with median
        for col in cols_with_missing:
            df[col] = df[col].fillna(df[col].median())
    
    # Create differential features for player comparisons
    # These should already exist in the data, but we'll make sure
    for col in feature_cols:
        if col.startswith('winner_') and 'loser_' + col[7:] in feature_cols:
            diff_col = col[7:] + '_diff'
            if diff_col not in df.columns:
                df[diff_col] = df[col] - df['loser_' + col[7:]]
                feature_cols.append(diff_col)
    
    print(f"Final feature count: {len(feature_cols)}")
    return df, feature_cols

def create_train_test_split(
    df: pd.DataFrame, 
    feature_cols: List[str], 
    config: ModelConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Create train/test split with proper time-based validation.
    
    Args:
        df: Preprocessed DataFrame
        feature_cols: List of feature column names
        config: Model configuration
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler)
    """
    print("Creating train/test split...")
    
    # Sort by date if available
    if 'tourney_date' in df.columns:
        df = df.sort_values('tourney_date').reset_index(drop=True)
        print(f"Data spans from {df['tourney_date'].min()} to {df['tourney_date'].max()}")
    
    # Create synthetic targets: we need to create balanced dataset for proper evaluation
    print("Creating balanced dataset with positive and negative examples...")
    
    # Clone the original dataframe to create positive examples
    df_positive = df.copy()
    df_positive['target'] = 1
    
    # Clone the original dataframe again to create negative examples
    df_negative = df.copy()
    df_negative['target'] = 0
    
    # For the negative examples, swap winner and loser
    print("Swapping winner/loser columns for negative examples...")
    
    # Get all winner columns
    winner_cols = [col for col in df_negative.columns if col.startswith('winner_')]
    
    # For each winner column, find the corresponding loser column and swap
    for w_col in winner_cols:
        base_col = w_col[7:]  # Remove 'winner_' prefix
        l_col = f'loser_{base_col}'
        
        if l_col in df_negative.columns:
            # Store temporary values
            temp_values = df_negative[w_col].copy()
            # Replace winner with loser
            df_negative[w_col] = df_negative[l_col]
            # Replace loser with stored winner values
            df_negative[l_col] = temp_values
    
    # Also swap winner_id and loser_id if they exist
    if 'winner_id' in df_negative.columns and 'loser_id' in df_negative.columns:
        temp_id = df_negative['winner_id'].copy()
        df_negative['winner_id'] = df_negative['loser_id']
        df_negative['loser_id'] = temp_id
    
    # Fix any differential columns directly
    for col in feature_cols:
        if col.endswith('_diff'):
            df_negative[col] = -df_negative[col]  # Just flip the sign for diff columns
    
    # Combine positive and negative examples
    df_balanced = pd.concat([df_positive, df_negative], ignore_index=True)
    
    # Shuffle to break any temporal patterns within same dates
    # This helps ensure mixed classes in test set
    if 'tourney_date' in df_balanced.columns:
        # Sort by date, but random within each date
        df_balanced = df_balanced.sample(frac=1, random_state=config.random_state)
        df_balanced = df_balanced.sort_values('tourney_date').reset_index(drop=True)
    
    # Update feature columns
    feature_cols = [col for col in feature_cols if col in df_balanced.columns]
    
    # Extract features and target
    X = df_balanced[feature_cols].values
    y = df_balanced['target'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Try stratified split methods
    if config.use_time_split and 'tourney_date' in df_balanced.columns:
        print("Attempting time-based split with guaranteed class balance...")
        
        # Use the most reliable method: split each class separately
        pos_mask = df_balanced['target'] == 1
        neg_mask = ~pos_mask
        
        # Get positive and negative examples
        df_pos = df_balanced[pos_mask].copy().reset_index(drop=True)
        df_neg = df_balanced[neg_mask].copy().reset_index(drop=True)
        
        # Calculate split indices
        pos_split_idx = int(len(df_pos) * (1 - config.test_size))
        neg_split_idx = int(len(df_neg) * (1 - config.test_size))
        
        # Split positive examples
        pos_train = df_pos.iloc[:pos_split_idx]
        pos_test = df_pos.iloc[pos_split_idx:]
        
        # Split negative examples
        neg_train = df_neg.iloc[:neg_split_idx]
        neg_test = df_neg.iloc[neg_split_idx:]
        
        # Combine train and test sets
        df_train = pd.concat([pos_train, neg_train], ignore_index=True)
        df_test = pd.concat([pos_test, neg_test], ignore_index=True)
        
        # Extract features and target
        X_train = scaler.fit_transform(df_train[feature_cols].values)
        y_train = df_train['target'].values
        
        X_test = scaler.transform(df_test[feature_cols].values)
        y_test = df_test['target'].values
        
        # Verify we have both classes in test set
        train_classes = np.unique(y_train)
        test_classes = np.unique(y_test)
        
        if len(train_classes) < 2 or len(test_classes) < 2:
            print("WARNING: Time-based split failed to maintain classes in both sets!")
            print("Falling back to stratified random split...")
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=config.test_size, 
                random_state=config.random_state, stratify=y
            )
    else:
        # Random split
        print("Using stratified random train/test split")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=config.test_size, 
            random_state=config.random_state, stratify=y
        )
    
    print(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    print(f"Label distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")
    
    # Double check that both classes are in both train and test sets
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        print("ERROR: Failed to create balanced train/test split!")
        print("Forcing balanced random split as fallback...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=config.test_size, 
            random_state=config.random_state, stratify=y
        )
        print(f"Fallback split - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")
    
    return X_train, X_test, y_train, y_test, scaler

def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    config: ModelConfig
) -> Dict[str, Any]:
    """
    Tune hyperparameters using RandomizedSearchCV with time-based cross-validation.
    
    Args:
        X_train: Training features
        y_train: Training labels
        feature_names: List of feature column names
        config: Model configuration
        
    Returns:
        Dictionary of best hyperparameters
    """
    print("Tuning hyperparameters...")
    
    # Define search space
    param_dist = {
        'n_estimators': [300, 500, 700, 1000, 1200],
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.07, 0.1],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        'subsample': [0.6, 0.7, 0.8, 0.9],
        'reg_alpha': [0, 0.001, 0.01, 0.05, 0.1],
        'reg_lambda': [0.1, 0.5, 1.0, 2.0, 5.0],
        'gamma': [0, 0.05, 0.1, 0.2, 0.3],
        'min_child_weight': [1, 2, 3, 5, 7]
    }
    
    # Set up monotonic constraints if enabled
    monotonic_constraints = None
    if config.use_monotonic_constraints:
        # Find Elo difference features - we expect higher Elo to correlate with higher win probability
        monotonic_constraints = {}
        for i, name in enumerate(feature_names):
            if 'elo_diff' in name or 'player_elo_diff' in name:
                # Positive constraint (1) means higher values should lead to higher predictions
                monotonic_constraints[i] = 1
    
    # Set up feature weights if enabled
    feature_weights = None
    if config.use_feature_weights:
        # Assign higher weights to more important features
        feature_weights = np.ones(len(feature_names))
        
        # Give more weight to Elo features and recent win rates
        for i, name in enumerate(feature_names):
            if 'elo' in name.lower():
                feature_weights[i] = 2.0  # Double weight for Elo features
            elif 'win_rate' in name:
                # Higher weights for more recent matches
                if '_5_' in name or '_10_' in name:
                    feature_weights[i] = 1.5  # Recent win rates
                elif '_20_' in name:
                    feature_weights[i] = 1.2  # Medium-term win rates
            elif 'h2h_' in name:
                feature_weights[i] = 1.3  # Head-to-head features
            elif 'current_win_streak' in name:
                feature_weights[i] = 1.3  # Win streaks
    
    # Define time-based cross-validation
    cv = TimeSeriesSplit(n_splits=config.time_splits)
    
    # Additional parameters based on advanced options
    additional_params = {}
    if monotonic_constraints:
        additional_params['monotone_constraints'] = str(monotonic_constraints)
    
    # Configure XGBoost classifier
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=config.random_state,
        scale_pos_weight=config.scale_pos_weight,
        use_label_encoder=False,
        tree_method='hist',  # More efficient histogram-based algorithm
        eval_metric='logloss',
        **additional_params
    )
    
    # Random search
    search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=config.tuning_iterations,
        scoring='roc_auc',  # Use ROC AUC for better ranking
        cv=cv,
        random_state=config.random_state,
        n_jobs=-1,
        verbose=1
    )
    
    # Fit search
    if feature_weights is not None:
        # Need to convert to DMatrix for feature weights
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        dtrain.set_info(feature_weights=feature_weights)
        
        # For simplicity, let's use the default parameters and fit manually
        print("Using feature weights - performing simplified parameter search")
        best_params = {
            'n_estimators': config.n_estimators,
            'max_depth': config.max_depth,
            'learning_rate': config.learning_rate,
            'colsample_bytree': config.colsample_bytree,
            'subsample': config.subsample,
            'reg_alpha': config.reg_alpha,
            'reg_lambda': config.reg_lambda,
            'gamma': config.gamma,
            'min_child_weight': config.min_child_weight
        }
    else:
        search.fit(X_train, y_train)
        best_params = search.best_params_
        print(f"Best score: {search.best_score_:.4f}")
    
    print(f"Best parameters: {best_params}")
    return best_params

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: ModelConfig,
    feature_names: List[str]
) -> Tuple[xgb.XGBClassifier, Dict[str, float]]:
    """
    Train the XGBoost model using the best hyperparameters and advanced features.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        config: Model configuration
        feature_names: List of feature column names
        
    Returns:
        Tuple containing the trained model and performance metrics
    """
    print("Training XGBoost model...")
    
    # Tune hyperparameters if specified
    if config.hyperparameter_tuning:
        best_params = tune_hyperparameters(X_train, y_train, feature_names, config)
    else:
        best_params = {
            'n_estimators': config.n_estimators,
            'max_depth': config.max_depth,
            'learning_rate': config.learning_rate,
            'colsample_bytree': config.colsample_bytree,
            'subsample': config.subsample,
            'reg_alpha': config.reg_alpha,
            'reg_lambda': config.reg_lambda,
            'gamma': config.gamma,
            'min_child_weight': config.min_child_weight
        }
    
    # Create validation set for early stopping
    val_split = int(len(X_train) * (1 - config.validation_fraction))
    X_val = X_train[val_split:]
    y_val = y_train[val_split:]
    X_train_final = X_train[:val_split]
    y_train_final = y_train[:val_split]
    
    # Set up monotonic constraints if enabled
    monotonic_constraints = {}
    if config.use_monotonic_constraints:
        for i, name in enumerate(feature_names):
            if 'elo_diff' in name or 'player_elo_diff' in name:
                monotonic_constraints[i] = 1
    
    # Create DMatrix objects for XGBoost
    dtrain = xgb.DMatrix(X_train_final, label=y_train_final, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)
    
    # Set feature weights if enabled
    if config.use_feature_weights:
        feature_weights = np.ones(len(feature_names))
        
        # Give more weight to important features
        for i, name in enumerate(feature_names):
            if 'elo' in name.lower():
                feature_weights[i] = 2.0  # Elo features
            elif 'win_rate' in name:
                if '_5_' in name or '_10_' in name:
                    feature_weights[i] = 1.5  # Recent win rates
                elif '_20_' in name:
                    feature_weights[i] = 1.2  # Medium-term win rates
            elif 'h2h_' in name:
                feature_weights[i] = 1.3  # Head-to-head features
            elif 'current_win_streak' in name:
                feature_weights[i] = 1.3  # Win streaks
        
        # Set feature weights
        dtrain.set_info(feature_weights=feature_weights)
        dval.set_info(feature_weights=feature_weights)
    
    # Set up parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'auc'],
        'seed': config.random_state,
        'tree_method': 'hist',  # Faster algorithm
        **best_params
    }
    
    # Add monotonic constraints if enabled
    if config.use_monotonic_constraints and monotonic_constraints:
        params['monotone_constraints'] = str(monotonic_constraints)
    
    # Train with early stopping
    print("Training final model with early stopping...")
    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=best_params.get('n_estimators', 1000),
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=config.early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=100
    )
    
    # Evaluate on test set
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate performance metrics
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Compile results
    metrics = {
        'accuracy': float(acc),
        'roc_auc': float(roc_auc),
        'confusion_matrix': conf_matrix,
        'precision': float(class_report['1']['precision']),
        'recall': float(class_report['1']['recall']),
        'f1_score': float(class_report['1']['f1-score']),
        'early_stopped_at': int(model.best_iteration + 1)
    }
    
    print(f"Model performance:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    
    # Create XGBClassifier for saving
    xgb_classifier = xgb.XGBClassifier()
    xgb_classifier._Booster = model
    
    return xgb_classifier, metrics

def save_model_and_results(
    model: xgb.XGBClassifier,
    metrics: Dict[str, float],
    feature_names: List[str],
    config: ModelConfig
) -> None:
    """
    Save the trained model, feature importance, and performance metrics.
    
    Args:
        model: Trained XGBoost model
        metrics: Performance metrics
        feature_names: List of feature column names
        config: Model configuration
    """
    print(f"Saving model to {MODEL_OUTPUT}...")
    model.save_model(MODEL_OUTPUT)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    feature_importance = model.get_booster().get_score(importance_type='gain')
    
    # Sort features by importance
    sorted_features = sorted(
        feature_importance.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Keep only top 30 features for readability
    top_n = 30
    if len(sorted_features) > top_n:
        sorted_features = sorted_features[:top_n]
        print(f"Plotting top {top_n} features by importance")
    
    # Extract feature names and importance values
    features, importance = zip(*sorted_features)
    
    # Create bar plot
    plt.barh(range(len(features)), importance)
    plt.yticks(range(len(features)), [f.replace('_diff', '') for f in features])
    plt.xlabel('Importance (gain)')
    plt.ylabel('Feature')
    plt.title('Feature Importance (gain)')
    plt.tight_layout()
    
    print(f"Saving feature importance plot to {FEATURE_IMPORTANCE_FILE}...")
    plt.savefig(FEATURE_IMPORTANCE_FILE, dpi=300, bbox_inches='tight')
    
    # Save results
    results = {
        'metrics': metrics,
        'feature_importance': {k: float(v) for k, v in feature_importance.items()},
        'config': config.dict(),
        'timestamp': datetime.now().isoformat(),
        'top_features': [f for f, _ in sorted_features]
    }
    
    print(f"Saving results to {RESULTS_OUTPUT_FILE}...")
    with open(RESULTS_OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

def main() -> None:
    """
    Main function to train and evaluate the model.
    """
    start_time = time.time()
    
    print("=" * 80)
    print("TENNIS MATCH PREDICTION MODEL TRAINING (FOCUSED FEATURES)")
    print("=" * 80)
    
    # Set up model configuration
    config = ModelConfig()
    
    # Load data
    df = load_data()
    
    # Filter features to focus only on surface, Elo, and win rates
    print("Focusing on surface, Elo, and win rate features...")
    feature_patterns = [
        'win_rate_', 'surface', 'elo', 'streak', 'h2h_'
    ]
    
    feature_cols = [
        col for col in df.columns 
        if col.endswith('_diff') and any(pattern in col for pattern in feature_patterns)
    ]
    
    print(f"Selected {len(feature_cols)} focused features out of {len(df.columns)} total columns")
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test, scaler = create_train_test_split(df, feature_cols, config)
    
    # Train model with focused features
    model, metrics = train_model(X_train, y_train, X_test, y_test, config, feature_cols)
    
    # Save model and results
    save_model_and_results(model, metrics, feature_cols, config)
    
    # Print execution time
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    
    print("=" * 80)
    print(f"Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"Model saved to {MODEL_OUTPUT}")
    print(f"Results saved to {RESULTS_OUTPUT_FILE}")
    print(f"Feature importance plot saved to {FEATURE_IMPORTANCE_FILE}")
    print("=" * 80)

if __name__ == "__main__":
    main()
