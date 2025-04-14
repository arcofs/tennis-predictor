"""
Predict Outcomes for Upcoming Tennis Matches

This script loads a trained model and generates predictions for upcoming matches.
It uses features from the upcoming_match_features table to make predictions.
"""

import argparse
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy import create_engine, text
import psycopg2
from psycopg2.extras import RealDictCursor

# Add project root to the path so we can import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Fix for external-api module
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external-api"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_database_connection():
    """Get a connection to the PostgreSQL database."""
    try:
        # Try to get DATABASE_URL from environment variables
        database_url = os.environ.get("DATABASE_URL")
        
        if database_url:
            logger.info(f"Connecting to database using DATABASE_URL")
            conn = psycopg2.connect(database_url)
        else:
            # Fallback to individual connection parameters
            dbname = os.environ.get("DB_NAME", "tennis_predictor")
            user = os.environ.get("DB_USER", "postgres")
            password = os.environ.get("DB_PASSWORD", "postgres")
            host = os.environ.get("DB_HOST", "localhost")
            port = os.environ.get("DB_PORT", "5432")
            
            logger.info(f"Connecting to database: {dbname} on {host}:{port} as {user}")
            conn = psycopg2.connect(
                dbname=dbname,
                user=user,
                password=password,
                host=host,
                port=port
            )
        
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise

def get_sqlalchemy_engine():
    """Get a SQLAlchemy engine for database operations."""
    try:
        # Try to get DATABASE_URL from environment variables
        database_url = os.environ.get("DATABASE_URL")
        
        if database_url:
            logger.info(f"Creating SQLAlchemy engine using DATABASE_URL")
            # For SQLAlchemy, ensure the URL starts with postgresql:// not postgres://
            if database_url.startswith("postgres://"):
                database_url = database_url.replace("postgres://", "postgresql://", 1)
            engine = create_engine(database_url)
        else:
            # Fallback to individual connection parameters
            dbname = os.environ.get("DB_NAME", "tennis_predictor")
            user = os.environ.get("DB_USER", "postgres")
            password = os.environ.get("DB_PASSWORD", "postgres")
            host = os.environ.get("DB_HOST", "localhost")
            port = os.environ.get("DB_PORT", "5432")
            
            logger.info(f"Creating SQLAlchemy engine for: {dbname} on {host}:{port} as {user}")
            engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")
        
        return engine
    except Exception as e:
        logger.error(f"SQLAlchemy engine creation error: {e}")
        raise

def load_model(model_path: str) -> Tuple[xgb.Booster, List[str]]:
    """Load the trained XGBoost model and feature list."""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        features = model_data['features']
        
        logger.info(f"Loaded model from {model_path} with {len(features)} features")
        return model, features
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def load_upcoming_match_features() -> pd.DataFrame:
    """Load features for upcoming matches from the database."""
    engine = get_sqlalchemy_engine()
    
    try:
        query = """
        SELECT 
            umf.*,
            um.match_id,
            um.tournament_name,
            um.player1_name,
            um.player2_name,
            um.player1_id,
            um.player2_id,
            um.surface,
            um.round,
            um.tournament_date,
            um.match_date
        FROM 
            upcoming_match_features umf
        JOIN 
            upcoming_matches um ON umf.match_id = um.match_id
        WHERE 
            um.status = 'scheduled'
        """
        
        df = pd.read_sql(query, engine)
        logger.info(f"Loaded features for {len(df)} upcoming matches")
        return df
    except Exception as e:
        logger.error(f"Error loading upcoming match features: {e}")
        return pd.DataFrame()

def prepare_features_for_prediction(df: pd.DataFrame, required_features: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Prepare features for prediction in the format expected by the model."""
    # Check if we have all required features
    missing_features = [f for f in required_features if f not in df.columns]
    if missing_features:
        logger.warning(f"Missing features: {missing_features}")
        # Add missing features with NaN values
        for feature in missing_features:
            df[feature] = np.nan
    
    # Select only required features in the correct order
    X = df[required_features].values
    match_ids = df['match_id'].tolist()
    
    return X, match_ids

def make_predictions(model: xgb.Booster, X: np.ndarray, match_ids: List[str], 
                    feature_names: List[str]) -> pd.DataFrame:
    """Make predictions for upcoming matches."""
    try:
        # Convert to DMatrix for prediction
        dmatrix = xgb.DMatrix(X, feature_names=feature_names)
        
        # Get probabilities
        probabilities = model.predict(dmatrix)
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'match_id': match_ids,
            'player1_win_probability': probabilities
        })
        
        # Calculate player2 probability
        results_df['player2_win_probability'] = 1 - results_df['player1_win_probability']
        
        # Determine predicted winner
        results_df['predicted_winner'] = np.where(
            results_df['player1_win_probability'] >= 0.5, 
            'player1', 
            'player2'
        )
        
        # Calculate prediction confidence
        results_df['prediction_confidence'] = results_df[['player1_win_probability', 'player2_win_probability']].max(axis=1)
        
        logger.info(f"Made predictions for {len(results_df)} matches")
        return results_df
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise

def save_predictions_to_db(predictions_df: pd.DataFrame, 
                          upcoming_matches_df: pd.DataFrame,
                          model_version: str,
                          features_used: List[str]):
    """Save predictions to the database."""
    if predictions_df.empty:
        logger.warning("No predictions to save")
        return
    
    # Join with upcoming_matches data to get player IDs
    df = predictions_df.merge(
        upcoming_matches_df[['match_id', 'player1_id', 'player2_id']], 
        on='match_id'
    )
    
    # Determine predicted winner ID
    df['predicted_winner_id'] = np.where(
        df['predicted_winner'] == 'player1',
        df['player1_id'],
        df['player2_id']
    )
    
    conn = get_database_connection()
    cursor = conn.cursor()
    
    try:
        # Insert each prediction
        for _, row in df.iterrows():
            query = """
            INSERT INTO match_predictions 
            (match_id, player1_win_probability, player2_win_probability, 
             predicted_winner_id, prediction_confidence, model_version, 
             created_at, features_used)
            VALUES (%s, %s, %s, %s, %s, %s, NOW(), %s)
            ON CONFLICT (match_id) 
            DO UPDATE SET
                player1_win_probability = EXCLUDED.player1_win_probability,
                player2_win_probability = EXCLUDED.player2_win_probability,
                predicted_winner_id = EXCLUDED.predicted_winner_id,
                prediction_confidence = EXCLUDED.prediction_confidence,
                model_version = EXCLUDED.model_version,
                created_at = NOW(),
                features_used = EXCLUDED.features_used
            """
            
            cursor.execute(
                query, 
                (
                    row['match_id'], 
                    float(row['player1_win_probability']), 
                    float(row['player2_win_probability']), 
                    int(row['predicted_winner_id']), 
                    float(row['prediction_confidence']),
                    model_version,
                    features_used
                )
            )
        
        conn.commit()
        logger.info(f"Saved {len(df)} predictions to database")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error saving predictions: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

def format_predictions_for_display(predictions_df: pd.DataFrame, 
                                 upcoming_matches_df: pd.DataFrame) -> pd.DataFrame:
    """Format predictions for display or reporting."""
    # Merge with match details
    display_df = predictions_df.merge(
        upcoming_matches_df[['match_id', 'tournament_name', 'player1_name', 'player2_name', 
                          'surface', 'round', 'tournament_date', 'match_date']],
        on='match_id'
    )
    
    # Format probabilities as percentages
    display_df['player1_win_prob'] = (display_df['player1_win_probability'] * 100).round(1).astype(str) + '%'
    display_df['player2_win_prob'] = (display_df['player2_win_probability'] * 100).round(1).astype(str) + '%'
    
    # Format predicted winner
    display_df['predicted_winner_name'] = np.where(
        display_df['predicted_winner'] == 'player1',
        display_df['player1_name'],
        display_df['player2_name']
    )
    
    # Format confidence
    display_df['confidence'] = (display_df['prediction_confidence'] * 100).round(1).astype(str) + '%'
    
    # Select and rename columns for display
    result = display_df[[
        'match_id', 'tournament_name', 'round', 'surface', 
        'tournament_date', 'match_date', 
        'player1_name', 'player1_win_prob',
        'player2_name', 'player2_win_prob',
        'predicted_winner_name', 'confidence'
    ]].copy()
    
    # Sort by match date and tournament
    result = result.sort_values(['match_date', 'tournament_name', 'round'])
    
    return result

def display_predictions(formatted_df: pd.DataFrame):
    """Display prediction results."""
    if formatted_df.empty:
        logger.info("No predictions to display")
        return
    
    print("\n=== TENNIS MATCH PREDICTIONS ===\n")
    
    # Group by tournament
    for tournament, tournament_df in formatted_df.groupby('tournament_name'):
        print(f"\n{tournament}")
        print("-" * len(tournament))
        
        # Format date
        if not pd.isna(tournament_df['tournament_date'].iloc[0]):
            date = pd.to_datetime(tournament_df['tournament_date'].iloc[0]).strftime('%Y-%m-%d')
            print(f"Date: {date}\n")
        
        # Print match predictions
        for _, match in tournament_df.iterrows():
            match_time = ""
            if not pd.isna(match['match_date']):
                match_time = pd.to_datetime(match['match_date']).strftime('%H:%M')
                if match_time != "00:00":
                    match_time = f" at {match_time}"
                else:
                    match_time = ""
            
            print(f"{match['round']}{match_time}: {match['player1_name']} vs {match['player2_name']}")
            print(f"  Prediction: {match['predicted_winner_name']} ({match['confidence']} confidence)")
            print(f"  Win probabilities: {match['player1_name']}: {match['player1_win_prob']}  |  {match['player2_name']}: {match['player2_win_prob']}")
            print()
    
    print(f"\nTotal predictions: {len(formatted_df)}")

def generate_prediction_report(formatted_df: pd.DataFrame, output_file: Optional[str] = None):
    """Generate a detailed prediction report as CSV."""
    if output_file:
        try:
            formatted_df.to_csv(output_file, index=False)
            logger.info(f"Prediction report saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving prediction report: {e}")

def main(model_path: str, output_file: Optional[str] = None, display: bool = True):
    """Main function to generate predictions for upcoming matches."""
    logger.info("Starting match prediction process")
    
    # Load model
    try:
        model, features = load_model(model_path)
        model_version = os.path.basename(model_path).replace('.pkl', '')
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Load upcoming match features
    upcoming_features_df = load_upcoming_match_features()
    if upcoming_features_df.empty:
        logger.warning("No upcoming matches with features found")
        return
    
    # Prepare features for prediction
    X, match_ids = prepare_features_for_prediction(upcoming_features_df, features)
    
    # Make predictions
    predictions_df = make_predictions(model, X, match_ids, features)
    
    # Save predictions to database
    save_predictions_to_db(
        predictions_df, 
        upcoming_features_df, 
        model_version,
        features
    )
    
    # Format predictions for display/reporting
    formatted_predictions = format_predictions_for_display(
        predictions_df, 
        upcoming_features_df
    )
    
    # Display predictions if requested
    if display:
        display_predictions(formatted_predictions)
    
    # Generate report if output file specified
    if output_file:
        generate_prediction_report(formatted_predictions, output_file)
    
    logger.info("Completed match prediction process")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict outcomes for upcoming tennis matches")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model file")
    parser.add_argument("--output", type=str, help="Path to save prediction report")
    parser.add_argument("--no-display", action="store_true", help="Suppress display of predictions")
    args = parser.parse_args()
    
    main(args.model, args.output, not args.no_display) 