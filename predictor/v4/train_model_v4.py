"""
Tennis Match Prediction - Model Training (v4)

This script trains the XGBoost model for tennis match prediction:
1. Loads historical match data and features
2. Performs time-based train/val/test split
3. Tunes hyperparameters
4. Trains final model
5. Evaluates performance
6. Saves model and metadata for prediction pipeline
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
import xgboost as xgb
import optuna
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import psycopg2
import json

# Add project root to path to allow imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{project_root}/predictor/v4/output/logs/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
MODEL_VERSION = "v4_" + datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = project_root / "predictor/v4"
MODEL_DIR = OUTPUT_DIR / "models"
PLOTS_DIR = OUTPUT_DIR / "output/plots"

# Create necessary directories
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

class ModelTrainer:
    def __init__(self):
        """Initialize the model trainer"""
        load_dotenv()
        
        # Database settings
        self.db_url = os.getenv("DATABASE_URL")
        if self.db_url and self.db_url.startswith('postgres://'):
            self.db_url = self.db_url.replace('postgres://', 'postgresql://', 1)
    
    def get_db_connection(self):
        """Create a database connection"""
        return psycopg2.connect(self.db_url)
    
    def load_training_data(self) -> pd.DataFrame:
        """
        Load historical match data with features
        
        Returns:
            DataFrame with match data and features
        """
        # This query gets completed matches from match_features
        # Important note on match ID relationships:
        # - For historical matches, match_features.match_id = matches.id (auto-incremented PK)
        # - For future matches (excluded here with is_future IS NOT TRUE), we'd use scheduled_matches.match_id
        query = """
            SELECT 
                f.id,
                f.match_id,
                f.player1_id,
                f.player2_id,
                f.surface,
                f.tournament_date,
                f.tournament_level,
                f.is_future,
                -- All the diff features
                f.player_elo_diff,
                f.win_rate_5_diff,
                f.win_streak_diff,
                f.loss_streak_diff,
                f.win_rate_hard_5_diff,
                f.win_rate_clay_5_diff,
                f.win_rate_grass_5_diff,
                f.win_rate_carpet_5_diff,
                f.win_rate_hard_overall_diff,
                f.win_rate_clay_overall_diff,
                f.win_rate_grass_overall_diff,
                f.win_rate_carpet_overall_diff,
                f.serve_efficiency_5_diff,
                f.first_serve_pct_5_diff,
                f.first_serve_win_pct_5_diff,
                f.second_serve_win_pct_5_diff,
                f.ace_pct_5_diff,
                f.bp_saved_pct_5_diff,
                f.return_efficiency_5_diff,
                f.bp_conversion_pct_5_diff,
                -- All player1 features
                f.player1_win_rate_5,
                f.player1_win_streak,
                f.player1_loss_streak,
                f.player1_win_rate_hard_5,
                f.player1_win_rate_clay_5,
                f.player1_win_rate_grass_5,
                f.player1_win_rate_carpet_5,
                f.player1_win_rate_hard_overall,
                f.player1_win_rate_clay_overall,
                f.player1_win_rate_grass_overall,
                f.player1_win_rate_carpet_overall,
                f.player1_serve_efficiency_5,
                f.player1_first_serve_pct_5,
                f.player1_first_serve_win_pct_5,
                f.player1_second_serve_win_pct_5,
                f.player1_ace_pct_5,
                f.player1_bp_saved_pct_5,
                f.player1_return_efficiency_5,
                f.player1_bp_conversion_pct_5,
                -- All player2 features
                f.player2_win_rate_5,
                f.player2_win_streak,
                f.player2_loss_streak,
                f.player2_win_rate_hard_5,
                f.player2_win_rate_clay_5,
                f.player2_win_rate_grass_5,
                f.player2_win_rate_carpet_5,
                f.player2_win_rate_hard_overall,
                f.player2_win_rate_clay_overall,
                f.player2_win_rate_grass_overall,
                f.player2_win_rate_carpet_overall,
                f.player2_serve_efficiency_5,
                f.player2_first_serve_pct_5,
                f.player2_first_serve_win_pct_5,
                f.player2_second_serve_win_pct_5,
                f.player2_ace_pct_5,
                f.player2_bp_saved_pct_5,
                f.player2_return_efficiency_5,
                f.player2_bp_conversion_pct_5,
                -- Result
                m.winner_id = f.player1_id as result
            FROM match_features f
            JOIN matches m ON f.match_id = m.id
            WHERE f.is_future IS NOT TRUE
            AND m.winner_id IS NOT NULL
            ORDER BY f.tournament_date ASC
        """
        
        with self.get_db_connection() as conn:
            df = pd.read_sql(query, conn)
        
        logger.info(f"Loaded {len(df)} matches for training")
        return df
    
    def create_train_val_test_split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create time-based train/validation/test split
        
        Args:
            df: Input DataFrame
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Sort by date
        df = df.sort_values('tournament_date')
        
        # Calculate split points
        n = len(df)
        train_idx = int(n * train_ratio)
        val_idx = int(n * (train_ratio + val_ratio))
        
        # Split data
        train_df = df.iloc[:train_idx]
        val_df = df.iloc[train_idx:val_idx]
        test_df = df.iloc[val_idx:]
        
        logger.info(f"Split data into train ({len(train_df)}), val ({len(val_df)}), test ({len(test_df)}) sets")
        return train_df, val_df, test_df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature columns for training
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of feature column names
        """
        # Exclude non-feature columns and date/timestamp columns
        exclude_cols = [
            'id', 'match_id', 'player1_id', 'player2_id', 'tournament_date',
            'surface', 'tournament_level', 'result', 'is_future',
            'created_at', 'updated_at',  # Exclude timestamp columns
        ]
        
        # Check data types and exclude any date, datetime or timestamp columns
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]) or 'date' in col.lower() or 'time' in col.lower():
                if col not in exclude_cols:
                    exclude_cols.append(col)
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Log categorical features - XGBoost can handle these properly
        for col in feature_cols:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                # Convert to categorical type to ensure proper handling
                df[col] = df[col].astype('category')
                logger.info(f"Found categorical feature: {col}")
        
        logger.info(f"Using {len(feature_cols)} features for training")
        return feature_cols
    
    def prepare_features(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: List[str]
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Prepare feature matrices and labels for training
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            feature_cols: List of feature columns
            
        Returns:
            Tuple of ((X_train, X_val, X_test), (y_train, y_val, y_test))
        """
        # Create feature matrices
        X_train = train_df[feature_cols].values
        X_val = val_df[feature_cols].values
        X_test = test_df[feature_cols].values
        
        # Create label vectors
        y_train = train_df['result'].values
        y_val = val_df['result'].values
        y_test = test_df['result'].values
        
        return (X_train, X_val, X_test), (y_train, y_val, y_test)
    
    def tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str],
        n_trials: int = 100
    ) -> Dict[str, Any]:
        """
        Tune model hyperparameters using Optuna
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            feature_names: List of feature names
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary of best hyperparameters
        """
        def objective(trial):
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'gamma': trial.suggest_float('gamma', 0, 1),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            }
            
            # Create validation set
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
            
            # Train model
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=params['n_estimators'],
                evals=[(dval, 'val')],
                early_stopping_rounds=50,
                verbose_eval=False
            )
            
            # Get validation score
            y_pred = model.predict(dval)
            val_auc = roc_auc_score(y_val, y_pred)
            
            return val_auc
        
        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Best hyperparameters: {study.best_params}")
        logger.info(f"Best validation AUC: {study.best_value:.4f}")
        
        return study.best_params
    
    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str],
        params: Dict[str, Any]
    ) -> xgb.Booster:
        """
        Train final model with best hyperparameters
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            feature_names: List of feature names
            params: Model hyperparameters
            
        Returns:
            Trained XGBoost model
        """
        # Create datasets
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
        
        # Train model
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=params['n_estimators'],
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=10
        )
        
        logger.info(f"Trained model with {model.best_ntree_limit} trees")
        return model
    
    def evaluate_model(
        self,
        model: xgb.Booster,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        dataset_name: str = "Test"
    ) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            X: Feature matrix
            y: True labels
            feature_names: List of feature names
            dataset_name: Name of dataset for logging
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Get predictions
        dmatrix = xgb.DMatrix(X, feature_names=feature_names)
        y_pred_proba = model.predict(dmatrix)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'auc': roc_auc_score(y, y_pred_proba)
        }
        
        # Log results
        logger.info(f"\n{dataset_name} Set Performance:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # Plot confusion matrix
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"{dataset_name} Set Confusion Matrix")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(PLOTS_DIR / f"confusion_matrix_{dataset_name.lower()}.png")
        plt.close()
        
        # Plot feature importance
        importance_scores = model.get_score(importance_type='gain')
        importance_df = pd.DataFrame(
            list(importance_scores.items()),
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=importance_df.head(20), x='Importance', y='Feature')
        plt.title(f"Top 20 Feature Importance ({dataset_name} Set)")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"feature_importance_{dataset_name.lower()}.png")
        plt.close()
        
        return metrics
    
    def save_model(
        self,
        model: xgb.Booster,
        feature_names: List[str],
        metrics: Dict[str, float]
    ):
        """
        Save model and metadata
        
        Args:
            model: Trained model
            feature_names: List of feature names
            metrics: Dictionary of evaluation metrics
        """
        # Save model
        model_path = MODEL_DIR / f"model_{MODEL_VERSION}.json"
        model.save_model(str(model_path))
        
        # Save feature names
        features_path = MODEL_DIR / f"features_{MODEL_VERSION}.json"
        with open(features_path, 'w') as f:
            json.dump(feature_names, f)
        
        # Save metrics
        metrics_path = MODEL_DIR / f"metrics_{MODEL_VERSION}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)
        
        logger.info(f"Saved model and metadata to {MODEL_DIR}")
    
    def train(self):
        """Main training method"""
        try:
            # Load data
            df = self.load_training_data()
            
            # Get feature columns
            feature_cols = self.get_feature_columns(df)
            
            # Split data
            train_df, val_df, test_df = self.create_train_val_test_split(df)
            
            # Prepare features
            (X_train, X_val, X_test), (y_train, y_val, y_test) = self.prepare_features(
                train_df, val_df, test_df, feature_cols
            )
            
            # Tune hyperparameters
            best_params = self.tune_hyperparameters(
                X_train, y_train, X_val, y_val, feature_cols
            )
            
            # Train final model
            model = self.train_model(
                X_train, y_train, X_val, y_val, feature_cols, best_params
            )
            
            # Evaluate on test set
            test_metrics = self.evaluate_model(
                model, X_test, y_test, feature_cols, "Test"
            )
            
            # Save model and metadata
            self.save_model(model, feature_cols, test_metrics)
            
            logger.info("Training completed successfully")
            
        except Exception as e:
            logger.error(f"Error in training: {e}")
            raise

def main():
    """Main execution function"""
    try:
        trainer = ModelTrainer()
        trainer.train()
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 