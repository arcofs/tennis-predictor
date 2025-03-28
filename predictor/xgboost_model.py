#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tennis Match Winner Prediction Model using XGBoost
-------------------------------------------------
This script trains an XGBoost model to predict tennis match winners based on
features generated from historical match data. It's designed to be run in a
Google Colab environment with access to GPU acceleration.

The model outputs the probability of a player winning a match based on
various features including Elo ratings, head-to-head statistics, recent form,
and surface-specific performance metrics.
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Check if running in Google Colab
def is_colab() -> bool:
    """Check if the code is running in Google Colab."""
    return 'google.colab' in sys.modules

# Install required packages if needed
if is_colab():
    try:
        import xgboost as xgb
        import shap
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "xgboost", "shap"])
        import xgboost as xgb
        import shap
else:
    import xgboost as xgb
    try:
        import shap
    except ImportError:
        print("SHAP not available. Feature importance visualization will be limited.")

# Print versions for debugging
print(f"Using pandas version: {pd.__version__}")
print(f"Using XGBoost version: {xgb.__version__}")

# Try to import GPU acceleration libraries, fall back to CPU if not available
USE_GPU = False
try:
    import cudf
    import cupy as cp
    # Test that GPU is actually working
    try:
        # Create a small test dataframe to verify GPU functionality
        test_df = cudf.DataFrame({'test': [1, 2, 3]})
        test_arr = cp.array([1, 2, 3])
        del test_df, test_arr  # Clean up test objects
        
        USE_GPU = True
        print("GPU acceleration enabled - using cuDF and cuPy")
    except Exception as e:
        print(f"GPU libraries imported but failed initialization test: {e}")
        print("Falling back to CPU-only mode")
        # Unload GPU libraries to prevent partial usage
        import sys
        if 'cudf' in sys.modules:
            del sys.modules['cudf']
        if 'cupy' in sys.modules:
            del sys.modules['cupy']
except ImportError:
    print("GPU libraries not available - falling back to CPU-only mode")

# Mount Google Drive if in Colab and not already mounted
if is_colab():
    from google.colab import drive
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')
        print("Google Drive mounted")
    else:
        print("Google Drive already mounted")

    # Google Drive paths for Colab
    BASE_DIR = Path('/content/drive/MyDrive/Colab Notebooks/tennis-predictor')
else:
    # Local paths if not running in Colab
    BASE_DIR = Path(__file__).parent.parent

# Define paths
DATA_DIR = BASE_DIR / "data"
CLEANED_DATA_DIR = DATA_DIR / "cleaned"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "predictor" / "output"

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Input and output files
FEATURES_FILE = CLEANED_DATA_DIR / "enhanced_features.csv"
MODEL_FILE = MODELS_DIR / "xgboost_model.pkl"
SCALER_FILE = MODELS_DIR / "feature_scaler.pkl"
FEATURE_IMPORTANCE_FILE = OUTPUT_DIR / "feature_importance.png"
MODEL_METRICS_FILE = OUTPUT_DIR / "model_metrics.json"

class TennisMatchPredictor:
    """
    A class to train and use an XGBoost model for predicting tennis match winners.
    
    Attributes:
        model: Trained XGBoost model
        scaler: StandardScaler for feature normalization
        feature_columns: List of feature column names
        categorical_columns: List of categorical column names
        numerical_columns: List of numerical column names
    """
    
    def __init__(self):
        """Initialize the predictor with empty model and scaler."""
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.categorical_columns = []
        self.numerical_columns = []
        self.feature_importance = None
        self.progress_bar = None
    
    def load_data(self, file_path: Union[str, Path] = FEATURES_FILE) -> pd.DataFrame:
        """
        Load the tennis match dataset with enhanced features.
        
        Args:
            file_path: Path to the CSV file with enhanced features
            
        Returns:
            pd.DataFrame: The loaded dataset
        """
        print(f"Loading data from {file_path}...")
        
        try:
            # Try to load with optimized dtypes
            df = pd.read_csv(file_path)
            
            # Convert date column if it exists
            if 'tourney_date' in df.columns:
                df['tourney_date'] = pd.to_datetime(df['tourney_date'])
            
            print(f"Loaded {len(df)} matches spanning from {df['tourney_date'].min()} to {df['tourney_date'].max()}")
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Prepare data for model training by splitting features and target.
        
        Args:
            df: DataFrame with enhanced features
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple containing (X_train, X_test, y_train, y_test)
        """
        print("Preparing data for model training...")
        
        # Identify feature columns (exclude non-feature columns)
        non_feature_cols = ['tourney_date', 'winner_id', 'loser_id', 'surface', 'tourney_level']
        
        # Identify categorical and numerical columns
        categorical_cols = ['surface', 'tourney_level']
        
        # All columns except non-feature columns are features
        feature_cols = [col for col in df.columns if col not in non_feature_cols]
        
        # Store feature columns for later use
        self.feature_columns = feature_cols
        self.categorical_columns = [col for col in categorical_cols if col in df.columns]
        self.numerical_columns = [col for col in feature_cols if col not in self.categorical_columns]
        
        print(f"Using {len(feature_cols)} features for model training")
        
        # Create target variable (1 for winner, 0 for loser)
        # Since our dataset already has matches with known winners, we'll create a balanced dataset
        # by duplicating each match and swapping winner/loser
        
        # First, create a copy of the original dataframe
        df_copy = df.copy()
        
        # Swap winner and loser columns in the copy
        winner_cols = [col for col in df.columns if 'winner' in col]
        loser_cols = [col for col in df.columns if 'loser' in col]
        
        # Create mapping for column swapping
        swap_dict = {}
        for w_col, l_col in zip(winner_cols, loser_cols):
            swap_dict[w_col] = l_col
            swap_dict[l_col] = w_col
        
        # Rename columns in the copy
        df_copy = df_copy.rename(columns=swap_dict)
        
        # Add target column (1 for original matches where winner won, 0 for swapped matches where winner lost)
        df['target'] = 1
        df_copy['target'] = 0
        
        # Combine original and swapped dataframes
        combined_df = pd.concat([df, df_copy], ignore_index=True)
        
        # Shuffle the data
        combined_df = combined_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        # Split features and target
        X = combined_df[feature_cols].copy()  # Create a copy to avoid SettingWithCopyWarning
        y = combined_df['target']
        
        # Split into training and testing sets (chronologically if possible)
        if 'tourney_date' in combined_df.columns:
            # Sort by date
            combined_df = combined_df.sort_values('tourney_date')
            
            # Use the most recent matches for testing
            train_size = int((1 - test_size) * len(combined_df))
            X_train = X.iloc[:train_size].copy()  # Create a copy to avoid SettingWithCopyWarning
            X_test = X.iloc[train_size:].copy()   # Create a copy to avoid SettingWithCopyWarning
            y_train = y.iloc[:train_size]
            y_test = y.iloc[train_size:]
        else:
            # If no date column, use random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            # Create copies to avoid SettingWithCopyWarning
            X_train = X_train.copy()
            X_test = X_test.copy()
        
        # Scale numerical features
        self.scaler = StandardScaler()
        X_train.loc[:, self.numerical_columns] = self.scaler.fit_transform(X_train[self.numerical_columns])
        X_test.loc[:, self.numerical_columns] = self.scaler.transform(X_test[self.numerical_columns])
        
        print(f"Data prepared: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                   X_test: pd.DataFrame = None, y_test: pd.Series = None,
                   hyperparameter_tuning: bool = True) -> None:
        """
        Train the XGBoost model with optional hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Testing features (optional, for evaluation during training)
            y_test: Testing target (optional, for evaluation during training)
            hyperparameter_tuning: Whether to perform hyperparameter tuning
        """
        print("Training XGBoost model...")
        start_time = time.time()
        
        # Set up XGBoost parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'verbosity': 1,
            'n_jobs': -1  # Use all available cores
        }
        
        # Add GPU parameters if available
        if USE_GPU:
            params['tree_method'] = 'gpu_hist'
            params['gpu_id'] = 0
            print("Using GPU acceleration for XGBoost")
        else:
            params['tree_method'] = 'hist'
            print("Using CPU for XGBoost")
        
        if hyperparameter_tuning:
            print("Performing hyperparameter tuning...")
            
            # Define parameter grid
            param_grid = {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'n_estimators': [100, 200],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'gamma': [0, 0.1]
            }
            
            # Calculate total iterations for the progress bar
            total_iterations = len(param_grid['max_depth']) * len(param_grid['learning_rate']) * \
                              len(param_grid['n_estimators']) * len(param_grid['subsample']) * \
                              len(param_grid['colsample_bytree']) * len(param_grid['gamma']) * 3  # 3-fold CV
            
            print(f"Total hyperparameter combinations to try: {total_iterations // 3}")
            self.progress_bar = tqdm(total=total_iterations, desc="Hyperparameter Tuning Progress")
            
            # Create custom scoring function that updates the progress bar
            def custom_scorer(estimator, X, y):
                score = -log_loss(y, estimator.predict_proba(X)[:, 1])
                self.progress_bar.update(1)
                return score
            
            # Create XGBoost classifier
            xgb_model = xgb.XGBClassifier(**params)
            
            # Set up grid search
            grid_search = GridSearchCV(
                estimator=xgb_model,
                param_grid=param_grid,
                scoring=custom_scorer,
                cv=3,
                verbose=0,  # Set to 0 as we use our own progress bar
                n_jobs=-1
            )
            
            # Fit grid search
            grid_search.fit(X_train, y_train)
            
            # Close progress bar
            self.progress_bar.close()
            
            # Get best parameters
            best_params = grid_search.best_params_
            print(f"Best parameters: {best_params}")
            
            # Update parameters with best values
            params.update(best_params)
            
            # Create model with best parameters
            self.model = xgb.XGBClassifier(**params)
        else:
            # Default parameters if no tuning
            default_params = {
                'max_depth': 5,
                'learning_rate': 0.1,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0
            }
            params.update(default_params)
            
            # Create model
            self.model = xgb.XGBClassifier(**params)
        
        # Train the model
        if X_test is not None and y_test is not None:
            # Use validation set during training
            eval_set = [(X_train, y_train), (X_test, y_test)]
            
            # Create progress bar for training
            n_estimators = params.get('n_estimators', 200)
            self.progress_bar = tqdm(total=n_estimators, desc="Training Progress")
            
            # Custom callback to update progress bar
            def progress_callback(env):
                self.progress_bar.update(1)
                
                # Calculate approximate completion percentage
                iteration = env.iteration
                total_iterations = env.end_iteration
                percentage = (iteration / total_iterations) * 100
                
                # Update description with percentage
                self.progress_bar.set_description(f"Training Progress: {percentage:.1f}%")
                
                # Check if we should stop early due to validation error not improving
                if len(env.evaluation_result_list) > 1:
                    # validation error not decreasing
                    if env.iteration >= 10 and env.best_iteration < env.iteration - 9:
                        return True
                return False
            
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                eval_metric='logloss',
                early_stopping_rounds=10,
                verbose=False,  # Set to False as we use our own progress
                callbacks=[progress_callback]
            )
            
            # Close progress bar
            self.progress_bar.close()
        else:
            # Train without validation
            n_estimators = params.get('n_estimators', 200)
            self.progress_bar = tqdm(total=n_estimators, desc="Training Progress")
            
            # Define custom callback for progress update
            def progress_callback(env):
                self.progress_bar.update(1)
                
                # Calculate approximate completion percentage
                iteration = env.iteration
                total_iterations = env.end_iteration
                percentage = (iteration / total_iterations) * 100
                
                # Update description with percentage
                self.progress_bar.set_description(f"Training Progress: {percentage:.1f}%")
            
            self.model.fit(
                X_train, y_train,
                verbose=False,  # Set to False as we use our own progress
                callbacks=[progress_callback]
            )
            
            # Close progress bar
            self.progress_bar.close()
        
        # Calculate feature importance
        self.feature_importance = self.model.feature_importances_
        
        # Print training time
        end_time = time.time()
        training_time = end_time - start_time
        print(f"Model training completed in {training_time:.2f} seconds")
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate the trained model on test data.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dict containing evaluation metrics
        """
        print("Evaluating model performance...")
        
        # Make predictions
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'log_loss': log_loss(y_test, y_pred_proba),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Print metrics
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"Log Loss: {metrics['log_loss']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def visualize_feature_importance(self, output_file: Union[str, Path] = FEATURE_IMPORTANCE_FILE) -> None:
        """
        Visualize feature importance from the trained model.
        
        Args:
            output_file: Path to save the visualization
        """
        if self.model is None:
            print("Model not trained yet. Cannot visualize feature importance.")
            return
        
        print("Visualizing feature importance...")
        
        # Get feature importance
        importance = self.model.feature_importances_
        
        # Create DataFrame for visualization
        importance_df = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Plot top 20 features
        plt.figure(figsize=(12, 10))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
        plt.title('Top 20 Feature Importance')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_file)
        print(f"Feature importance visualization saved to {output_file}")
        
        # Try SHAP values if available
        try:
            if 'shap' in sys.modules:
                print("Calculating SHAP values for feature importance...")
                
                # Create explainer
                explainer = shap.Explainer(self.model)
                
                # Calculate SHAP values (sample 100 instances for speed)
                X_sample = X_test.sample(min(100, len(X_test)))
                shap_values = explainer(X_sample)
                
                # Plot summary
                plt.figure(figsize=(12, 10))
                shap.summary_plot(shap_values, X_sample, show=False)
                plt.tight_layout()
                
                # Save figure
                shap_file = str(output_file).replace('.png', '_shap.png')
                plt.savefig(shap_file)
                print(f"SHAP feature importance visualization saved to {shap_file}")
        except Exception as e:
            print(f"Error calculating SHAP values: {e}")
            print("Skipping SHAP visualization")
    
    def save_model(self, model_file: Union[str, Path] = MODEL_FILE, 
                  scaler_file: Union[str, Path] = SCALER_FILE) -> None:
        """
        Save the trained model and scaler to disk.
        
        Args:
            model_file: Path to save the model
            scaler_file: Path to save the scaler
        """
        if self.model is None:
            print("Model not trained yet. Cannot save.")
            return
        
        print(f"Saving model to {model_file}...")
        
        # Save model
        with open(model_file, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save scaler if available
        if self.scaler is not None:
            print(f"Saving scaler to {scaler_file}...")
            with open(scaler_file, 'wb') as f:
                pickle.dump(self.scaler, f)
        
        # Save feature columns
        feature_file = str(model_file).replace('.pkl', '_features.pkl')
        with open(feature_file, 'wb') as f:
            pickle.dump({
                'feature_columns': self.feature_columns,
                'categorical_columns': self.categorical_columns,
                'numerical_columns': self.numerical_columns
            }, f)
        
        print("Model and associated files saved successfully")
    
    def load_model(self, model_file: Union[str, Path] = MODEL_FILE,
                 scaler_file: Union[str, Path] = SCALER_FILE) -> None:
        """
        Load a trained model and scaler from disk.
        
        Args:
            model_file: Path to the saved model
            scaler_file: Path to the saved scaler
        """
        print(f"Loading model from {model_file}...")
        
        # Load model
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load scaler if available
        if os.path.exists(scaler_file):
            print(f"Loading scaler from {scaler_file}...")
            with open(scaler_file, 'rb') as f:
                self.scaler = pickle.load(f)
        
        # Load feature columns
        feature_file = str(model_file).replace('.pkl', '_features.pkl')
        if os.path.exists(feature_file):
            with open(feature_file, 'rb') as f:
                feature_data = pickle.load(f)
                self.feature_columns = feature_data['feature_columns']
                self.categorical_columns = feature_data['categorical_columns']
                self.numerical_columns = feature_data['numerical_columns']
        
        print("Model and associated files loaded successfully")
    
    def predict_match(self, player1_features: Dict[str, Any], player2_features: Dict[str, Any]) -> Tuple[int, float]:
        """
        Predict the winner of a match between two players.
        
        Args:
            player1_features: Dictionary of features for player 1
            player2_features: Dictionary of features for player 2
            
        Returns:
            Tuple containing (predicted_winner, win_probability)
            where predicted_winner is 1 for player1 and 2 for player2
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded yet")
        
        # Create feature vector for the match
        match_features = {}
        
        # Map player1 features to winner features and player2 features to loser features
        for col in self.feature_columns:
            if 'winner' in col:
                base_col = col.replace('winner_', '')
                if base_col in player1_features:
                    match_features[col] = player1_features[base_col]
                else:
                    match_features[col] = 0  # Default value
            elif 'loser' in col:
                base_col = col.replace('loser_', '')
                if base_col in player2_features:
                    match_features[col] = player2_features[base_col]
                else:
                    match_features[col] = 0  # Default value
            else:
                # Handle diff columns or other special columns
                if col.endswith('_diff'):
                    base_col = col.replace('_diff', '')
                    if base_col in player1_features and base_col in player2_features:
                        match_features[col] = player1_features[base_col] - player2_features[base_col]
                    else:
                        match_features[col] = 0  # Default value
                else:
                    # Other columns
                    match_features[col] = 0  # Default value
        
        # Convert to DataFrame
        match_df = pd.DataFrame([match_features])
        
        # Scale numerical features if scaler is available
        if self.scaler is not None and self.numerical_columns:
            match_df[self.numerical_columns] = self.scaler.transform(match_df[self.numerical_columns])
        
        # Make prediction
        player1_win_prob = self.model.predict_proba(match_df)[0, 1]
        
        # Determine winner
        if player1_win_prob >= 0.5:
            return 1, player1_win_prob
        else:
            return 2, 1 - player1_win_prob
    
    def predict_match_from_ids(self, df: pd.DataFrame, player1_id: int, player2_id: int,
                             surface: str = 'Hard', tourney_level: str = 'ATP') -> Tuple[int, float]:
        """
        Predict the winner of a match between two players using their IDs.
        
        Args:
            df: DataFrame with player features
            player1_id: ID of player 1
            player2_id: ID of player 2
            surface: Match surface
            tourney_level: Tournament level
            
        Returns:
            Tuple containing (predicted_winner_id, win_probability)
        """
        # Get latest features for player 1
        player1_mask = (df['winner_id'] == player1_id) | (df['loser_id'] == player1_id)
        if not player1_mask.any():
            raise ValueError(f"Player with ID {player1_id} not found in the dataset")
        
        player1_data = df[player1_mask].sort_values('tourney_date').iloc[-1]
        
        # Get latest features for player 2
        player2_mask = (df['winner_id'] == player2_id) | (df['loser_id'] == player2_id)
        if not player2_mask.any():
            raise ValueError(f"Player with ID {player2_id} not found in the dataset")
        
        player2_data = df[player2_mask].sort_values('tourney_date').iloc[-1]
        
        # Extract features for each player
        player1_features = {}
        player2_features = {}
        
        # Helper function to extract features
        def extract_player_features(row, player_id, features_dict):
            if row['winner_id'] == player_id:
                # Player was the winner in this match
                prefix = 'winner_'
            else:
                # Player was the loser in this match
                prefix = 'loser_'
            
            # Extract all features with the appropriate prefix
            for col in row.index:
                if col.startswith(prefix):
                    base_col = col.replace(prefix, '')
                    features_dict[base_col] = row[col]
            
            # Add surface and tourney_level
            features_dict['surface'] = surface
            features_dict['tourney_level'] = tourney_level
            
            return features_dict
        
        # Extract features
        player1_features = extract_player_features(player1_data, player1_id, player1_features)
        player2_features = extract_player_features(player2_data, player2_id, player2_features)
        
        # Predict match
        winner, probability = self.predict_match(player1_features, player2_features)
        
        # Return winner ID and probability
        if winner == 1:
            return player1_id, probability
        else:
            return player2_id, probability

def main():
    """Main function to train and evaluate the model."""
    print("Tennis Match Winner Prediction using XGBoost")
    print("=" * 50)
    
    # Create predictor
    predictor = TennisMatchPredictor()
    
    # Load data
    df = predictor.load_data()
    
    # Prepare data
    X_train, X_test, y_train, y_test = predictor.prepare_data(df)
    
    # Train model
    predictor.train_model(X_train, y_train, X_test, y_test, hyperparameter_tuning=True)
    
    # Evaluate model
    metrics = predictor.evaluate_model(X_test, y_test)
    
    # Visualize feature importance
    predictor.visualize_feature_importance()
    
    # Save model
    predictor.save_model()
    
    # Save metrics
    with open(MODEL_METRICS_FILE, 'w') as f:
        import json
        json.dump(metrics, f, indent=4)
    
    print("=" * 50)
    print("Model training and evaluation completed")
    print(f"Model saved to {MODEL_FILE}")
    print(f"Metrics saved to {MODEL_METRICS_FILE}")
    print(f"Feature importance visualization saved to {FEATURE_IMPORTANCE_FILE}")

if __name__ == "__main__":
    main()
