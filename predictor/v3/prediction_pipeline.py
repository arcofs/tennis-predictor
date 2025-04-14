"""
Tennis Match Prediction Pipeline

This script runs the complete tennis match prediction pipeline:
1. Fetch upcoming matches from the API
2. Generate features for upcoming matches
3. Make predictions using the trained model
4. Update match results and prediction accuracy

It's designed to be run as a scheduled task to keep predictions up to date.
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Fix for external-api module
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external-api"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"prediction_pipeline_{datetime.now().strftime('%Y%m%d')}.log")
    ]
)
logger = logging.getLogger(__name__)

def run_fetch_upcoming_matches(days_ahead: int = 14) -> bool:
    """Run the script to fetch upcoming matches."""
    logger.info("Step 1: Fetching upcoming matches")
    
    try:
        from predictor.v3.fetch_upcoming_matches import main as fetch_main
        fetch_main(days_ahead)
        logger.info("Successfully fetched upcoming matches")
        return True
    except Exception as e:
        logger.error(f"Error fetching upcoming matches: {e}")
        return False

def run_generate_features() -> bool:
    """Run the script to generate features for upcoming matches."""
    logger.info("Step 2: Generating features for upcoming matches")
    
    try:
        from predictor.v3.generate_upcoming_features import main as generate_features_main
        generate_features_main()
        logger.info("Successfully generated features")
        return True
    except Exception as e:
        logger.error(f"Error generating features: {e}")
        return False

def run_predict_matches(model_path: str, output_file: Optional[str] = None) -> bool:
    """Run the script to predict match outcomes."""
    logger.info("Step 3: Predicting match outcomes")
    
    try:
        from predictor.v3.predict_matches import main as predict_main
        predict_main(model_path, output_file, display=True)
        logger.info("Successfully generated predictions")
        return True
    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        return False

def update_match_results() -> bool:
    """Update match results and prediction accuracy."""
    logger.info("Step 4: Updating match results and prediction accuracy")
    
    try:
        # This function is already called from fetch_upcoming_matches
        # but we call it again here to make sure it's done
        from predictor.v3.fetch_upcoming_matches import update_match_status
        update_match_status()
        logger.info("Successfully updated match results")
        return True
    except Exception as e:
        logger.error(f"Error updating match results: {e}")
        return False

def calculate_prediction_accuracy():
    """Calculate and log the current prediction accuracy."""
    import psycopg2
    from psycopg2.extras import RealDictCursor
    
    try:
        # Connect to the database
        conn = psycopg2.connect(
            dbname=os.environ.get("DB_NAME", "tennis_predictor"),
            user=os.environ.get("DB_USER", "postgres"),
            password=os.environ.get("DB_PASSWORD", "postgres"),
            host=os.environ.get("DB_HOST", "localhost"),
            port=os.environ.get("DB_PORT", "5432")
        )
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Calculate overall accuracy
        cursor.execute("""
        SELECT 
            COUNT(*) as total_predictions,
            SUM(CASE WHEN prediction_correct = true THEN 1 ELSE 0 END) as correct_predictions,
            CASE 
                WHEN COUNT(*) > 0 THEN 
                    ROUND(SUM(CASE WHEN prediction_correct = true THEN 1 ELSE 0 END)::numeric / COUNT(*)::numeric * 100, 2)
                ELSE 0
            END as accuracy_percentage
        FROM match_predictions mp
        JOIN upcoming_matches um ON mp.match_id = um.match_id
        WHERE um.status = 'completed' AND mp.prediction_correct IS NOT NULL
        """)
        
        overall = cursor.fetchone()
        
        # Calculate accuracy by surface
        cursor.execute("""
        SELECT 
            um.surface,
            COUNT(*) as total_predictions,
            SUM(CASE WHEN prediction_correct = true THEN 1 ELSE 0 END) as correct_predictions,
            CASE 
                WHEN COUNT(*) > 0 THEN 
                    ROUND(SUM(CASE WHEN prediction_correct = true THEN 1 ELSE 0 END)::numeric / COUNT(*)::numeric * 100, 2)
                ELSE 0
            END as accuracy_percentage
        FROM match_predictions mp
        JOIN upcoming_matches um ON mp.match_id = um.match_id
        WHERE um.status = 'completed' AND mp.prediction_correct IS NOT NULL
        GROUP BY um.surface
        ORDER BY COUNT(*) DESC
        """)
        
        by_surface = cursor.fetchall()
        
        # Calculate accuracy by confidence level
        cursor.execute("""
        SELECT 
            CASE 
                WHEN prediction_confidence < 0.55 THEN '50-55%'
                WHEN prediction_confidence < 0.6 THEN '55-60%'
                WHEN prediction_confidence < 0.65 THEN '60-65%'
                WHEN prediction_confidence < 0.7 THEN '65-70%'
                WHEN prediction_confidence < 0.75 THEN '70-75%'
                WHEN prediction_confidence < 0.8 THEN '75-80%'
                WHEN prediction_confidence < 0.85 THEN '80-85%'
                WHEN prediction_confidence < 0.9 THEN '85-90%'
                WHEN prediction_confidence < 0.95 THEN '90-95%'
                ELSE '95-100%'
            END as confidence_range,
            COUNT(*) as total_predictions,
            SUM(CASE WHEN prediction_correct = true THEN 1 ELSE 0 END) as correct_predictions,
            CASE 
                WHEN COUNT(*) > 0 THEN 
                    ROUND(SUM(CASE WHEN prediction_correct = true THEN 1 ELSE 0 END)::numeric / COUNT(*)::numeric * 100, 2)
                ELSE 0
            END as accuracy_percentage
        FROM match_predictions mp
        JOIN upcoming_matches um ON mp.match_id = um.match_id
        WHERE um.status = 'completed' AND mp.prediction_correct IS NOT NULL
        GROUP BY confidence_range
        ORDER BY MIN(prediction_confidence)
        """)
        
        by_confidence = cursor.fetchall()
        
        # Log the results
        logger.info(f"Prediction accuracy statistics:")
        logger.info(f"Overall: {overall['accuracy_percentage']}% ({overall['correct_predictions']}/{overall['total_predictions']})")
        
        if by_surface:
            logger.info("By surface:")
            for surface in by_surface:
                logger.info(f"  {surface['surface']}: {surface['accuracy_percentage']}% ({surface['correct_predictions']}/{surface['total_predictions']})")
        
        if by_confidence:
            logger.info("By confidence level:")
            for conf in by_confidence:
                logger.info(f"  {conf['confidence_range']}: {conf['accuracy_percentage']}% ({conf['correct_predictions']}/{conf['total_predictions']})")
        
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"Error calculating prediction accuracy: {e}")

def run_full_pipeline(model_path: str, days_ahead: int = 14, output_file: Optional[str] = None):
    """Run the full prediction pipeline."""
    logger.info("Starting full prediction pipeline")
    start_time = time.time()
    
    # Step 1: Fetch upcoming matches
    if not run_fetch_upcoming_matches(days_ahead):
        logger.error("Pipeline aborted at step 1")
        return
    
    # Step 2: Generate features
    if not run_generate_features():
        logger.error("Pipeline aborted at step 2")
        return
    
    # Step 3: Make predictions
    if not run_predict_matches(model_path, output_file):
        logger.error("Pipeline aborted at step 3")
        return
    
    # Step 4: Update results
    if not update_match_results():
        logger.warning("Failed to update match results")
    
    # Calculate prediction accuracy
    calculate_prediction_accuracy()
    
    elapsed_time = time.time() - start_time
    logger.info(f"Completed full prediction pipeline in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the tennis match prediction pipeline")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model file")
    parser.add_argument("--days", type=int, default=14, help="Number of days ahead to fetch tournaments for")
    parser.add_argument("--output", type=str, help="Path to save prediction report")
    args = parser.parse_args()
    
    run_full_pipeline(args.model, args.days, args.output) 