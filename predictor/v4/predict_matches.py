"""
Tennis Match Prediction - Future Match Prediction (v4)

This script makes predictions for upcoming tennis matches by:
1. Loading the latest trained model
2. Getting features for unprocessed matches
3. Generating predictions with confidence scores
4. Storing predictions in the match_predictions table

Important note about match IDs:
- matches table: 'id' is auto-incremented PK, 'match_num' is the external API match ID
- scheduled_matches table: 'match_id' is the external API match ID (as string)
- match_features table: 'match_id' refers to matches.id for historical matches,
                         and refers to scheduled_matches.match_id for future matches
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
import xgboost as xgb
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values
import json

# Add project root to path to allow imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{project_root}/predictor/v4/output/logs/predictions.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MatchPredictor:
    def __init__(self):
        """Initialize the match predictor"""
        load_dotenv()
        
        # Database settings
        self.db_url = os.getenv("DATABASE_URL")
        if self.db_url and self.db_url.startswith('postgres://'):
            self.db_url = self.db_url.replace('postgres://', 'postgresql://', 1)
        
        # Model settings
        self.model_dir = project_root / "predictor/v4/models"
        self.model = None
        self.model_version = None
        self.feature_columns = None
    
    def get_db_connection(self):
        """Create a database connection"""
        return psycopg2.connect(self.db_url)
    
    def load_latest_model(self) -> Tuple[xgb.Booster, str, List[str]]:
        """
        Load the latest trained model and its metadata
        
        Returns:
            Tuple of (model, model version, feature columns)
        """
        # Find latest model file
        model_files = list(self.model_dir.glob("model_v4_*.json"))
        if not model_files:
            raise FileNotFoundError("No v4 model files found")
        
        latest_model_file = max(model_files, key=lambda x: x.stat().st_mtime)
        model_version = latest_model_file.stem.replace("model_", "")
        
        # Load model
        model = xgb.Booster()
        model.load_model(str(latest_model_file))
        
        # Load feature columns
        features_file = latest_model_file.with_name(f"features_{model_version}.json")
        if not features_file.exists():
            raise FileNotFoundError(f"Features file not found: {features_file}")
        
        with open(features_file) as f:
            feature_columns = json.load(f)
        
        logger.info(f"Loaded model version {model_version} with {len(feature_columns)} features")
        return model, model_version, feature_columns
    
    def get_unprocessed_matches(self) -> pd.DataFrame:
        """
        Get matches that need predictions
        
        Returns:
            DataFrame with match features
        """
        # Note: For future matches, match_features.match_id == scheduled_matches.match_id
        # This query gets future matches that don't have predictions yet
        query = """
            SELECT f.*, s.scheduled_date
            FROM match_features f
            JOIN scheduled_matches s ON f.match_id = s.match_id
            WHERE f.is_future = TRUE
            AND NOT EXISTS (
                SELECT 1 FROM match_predictions p
                WHERE p.match_id = f.match_id
                AND p.model_version = %s
            )
            AND s.scheduled_date >= CURRENT_DATE
            AND s.scheduled_date <= CURRENT_DATE + INTERVAL '7 days'
            ORDER BY s.scheduled_date ASC
        """
        
        with self.get_db_connection() as conn:
            df = pd.read_sql(query, conn, params=(self.model_version,))
        
        logger.info(f"Found {len(df)} matches needing predictions")
        return df
    
    def prepare_features(self, matches_df: pd.DataFrame) -> np.ndarray:
        """
        Prepare feature matrix for prediction
        
        Args:
            matches_df: DataFrame with match features
            
        Returns:
            numpy array of features in correct order
        """
        # Ensure all required features are present
        missing_features = set(self.feature_columns) - set(matches_df.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Create feature matrix in correct order
        X = matches_df[self.feature_columns].values
        
        return X
    
    def make_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for matches
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of win probabilities for player1
        """
        # Convert to DMatrix for prediction
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_columns)
        
        # Get raw predictions
        predictions = self.model.predict(dmatrix)
        
        return predictions
    
    def store_predictions(self, matches_df: pd.DataFrame, predictions: np.ndarray):
        """
        Store predictions in the database
        
        Args:
            matches_df: DataFrame with match information
            predictions: Array of predicted probabilities
        """
        with self.get_db_connection() as conn:
            with conn.cursor() as cur:
                # Prepare prediction records
                prediction_data = []
                for idx, row in matches_df.iterrows():
                    prediction = {
                        'match_id': row['match_id'],  # This is scheduled_matches.match_id
                        'player1_id': row['player1_id'],
                        'player2_id': row['player2_id'],
                        'player1_win_probability': float(predictions[idx]),
                        'prediction_date': datetime.now(),
                        'scheduled_date': row['scheduled_date'],
                        'model_version': self.model_version,
                        'features_used': json.dumps(self.feature_columns)
                    }
                    prediction_data.append(prediction)
                
                # Insert predictions
                if prediction_data:
                    columns = prediction_data[0].keys()
                    values = [[pred[col] for col in columns] for pred in prediction_data]
                    
                    execute_values(
                        cur,
                        f"""
                        INSERT INTO match_predictions (
                            {','.join(columns)}
                        ) VALUES %s
                        """,
                        values
                    )
                    
                    conn.commit()
                    logger.info(f"Stored {len(prediction_data)} predictions")
    
    def update_prediction_accuracy(self):
        """Update accuracy for past predictions where results are now known"""
        # Update prediction accuracy for completed matches
        # This query updates match_predictions by joining scheduled_matches to matches
        # Key relationship: scheduled_matches.match_id (string) is converted to integer
        # and matched with matches.match_num (which is the external API match ID)
        query = """
            UPDATE match_predictions p
            SET 
                actual_winner_id = m.winner_id,
                prediction_accuracy = CASE 
                    WHEN (p.player1_win_probability >= 0.5 AND m.winner_id = p.player1_id)
                    OR (p.player1_win_probability < 0.5 AND m.winner_id = p.player2_id)
                    THEN 1.0
                    ELSE 0.0
                END
            FROM matches m
            JOIN scheduled_matches s ON s.match_id::integer = m.match_num
            WHERE p.match_id = s.match_id
            AND p.actual_winner_id IS NULL
            AND m.winner_id IS NOT NULL
        """
        
        with self.get_db_connection() as conn:
            with conn.cursor() as cur:
                # Update prediction accuracy
                cur.execute(query)
                updated_predictions = cur.rowcount
                
                conn.commit()
                
                if updated_predictions > 0:
                    logger.info(f"Updated accuracy for {updated_predictions} past predictions")
                    
                # Note: We deliberately don't update match_features here
                # This allows generate_historical_features.py to handle all feature generation
                # which maintains a cleaner separation of concerns
    
    def predict_matches(self):
        """Main method to generate predictions for upcoming matches"""
        try:
            # Load latest model
            self.model, self.model_version, self.feature_columns = self.load_latest_model()
            
            # Update accuracy for past predictions
            self.update_prediction_accuracy()
            
            # Get unprocessed matches
            matches_df = self.get_unprocessed_matches()
            
            if matches_df.empty:
                logger.info("No matches need predictions")
                return
            
            # Prepare features
            X = self.prepare_features(matches_df)
            
            # Generate predictions
            predictions = self.make_predictions(X)
            
            # Store predictions
            self.store_predictions(matches_df, predictions)
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise

def main():
    """Main execution function"""
    try:
        predictor = MatchPredictor()
        predictor.predict_matches()
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 