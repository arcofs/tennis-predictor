"""
Generate Features for Upcoming Matches

This script generates predictive features for upcoming tennis matches.
It builds on generate_features_v3.py but adapts it for upcoming matches with no result.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
import psycopg2
from psycopg2.extras import RealDictCursor

# Add project root to the path so we can import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Fix for external-api module
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external-api"))

# Import from the feature generation module
from predictor.v3.generate_features_v3 import (
    calculate_win_rates,
    calculate_serve_return_stats,
    calculate_serve_return_rolling_stats,
    process_match_features_batch
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProgressTracker:
    """Simple tracker for progress updates."""
    def __init__(self, total_steps: int, task_description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.task_description = task_description
        
    def update(self, step_description: str = "") -> None:
        """Update progress tracker."""
        self.current_step += 1
        percent = self.current_step / self.total_steps * 100
        logger.info(f"{self.task_description}: {percent:.1f}% complete - {step_description}")
    
    @property
    def percent_complete(self) -> float:
        """Get percent complete."""
        return self.current_step / self.total_steps * 100

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

def load_historical_matches() -> pd.DataFrame:
    """Load historical match data from the database."""
    engine = get_sqlalchemy_engine()
    
    logger.info("Loading historical matches from database")
    query = """
    SELECT 
        id as match_id,
        tournament_id,
        tournament_name,
        surface,
        tournament_level,
        tournament_date,
        winner_id as player1_id,
        winner_name as player1_name,
        winner_hand as player1_hand,
        winner_height_cm as player1_height_cm,
        winner_country_code as player1_country_code,
        winner_age as player1_age,
        winner_rank as player1_rank,
        winner_rank_points as player1_rank_points,
        loser_id as player2_id,
        loser_name as player2_name,
        loser_hand as player2_hand,
        loser_height_cm as player2_height_cm,
        loser_country_code as player2_country_code,
        loser_age as player2_age,
        loser_rank as player2_rank,
        loser_rank_points as player2_rank_points,
        score,
        best_of,
        round,
        minutes,
        winner_aces as player1_aces,
        winner_double_faults as player1_double_faults,
        winner_serve_points as player1_serve_points,
        winner_first_serves_in as player1_first_serves_in,
        winner_first_serve_points_won as player1_first_serve_points_won,
        winner_second_serve_points_won as player1_second_serve_points_won,
        winner_service_games as player1_service_games,
        winner_break_points_saved as player1_break_points_saved,
        winner_break_points_faced as player1_break_points_faced,
        loser_aces as player2_aces,
        loser_double_faults as player2_double_faults,
        loser_serve_points as player2_serve_points,
        loser_first_serves_in as player2_first_serves_in,
        loser_first_serve_points_won as player2_first_serve_points_won,
        loser_second_serve_points_won as player2_second_serve_points_won,
        loser_service_games as player2_service_games,
        loser_break_points_saved as player2_break_points_saved,
        loser_break_points_faced as player2_break_points_faced,
        winner_elo as player1_elo,
        loser_elo as player2_elo,
        winner_matches as player1_matches,
        loser_matches as player2_matches,
        match_type,
        1 as result  -- Winner (player1) always wins in historical data
    FROM matches
    ORDER BY tournament_date ASC
    """
    
    df = pd.read_sql(query, engine)
    logger.info(f"Loaded {len(df)} historical matches")
    return df

def load_upcoming_matches() -> pd.DataFrame:
    """Load upcoming match data from the database."""
    engine = get_sqlalchemy_engine()
    
    logger.info("Loading upcoming matches from database")
    query = """
    SELECT 
        match_id,
        tournament_id,
        tournament_name,
        surface,
        tournament_level,
        tournament_date,
        player1_id,
        player1_name,
        player1_hand,
        player1_height_cm,
        player1_country_code,
        player1_age,
        player1_rank,
        player1_rank_points,
        player2_id,
        player2_name,
        player2_hand,
        player2_height_cm,
        player2_country_code,
        player2_age,
        player2_rank,
        player2_rank_points,
        NULL as score,
        NULL as best_of,
        round,
        NULL as minutes,
        NULL as player1_aces,
        NULL as player1_double_faults,
        NULL as player1_serve_points,
        NULL as player1_first_serves_in,
        NULL as player1_first_serve_points_won,
        NULL as player1_second_serve_points_won,
        NULL as player1_service_games,
        NULL as player1_break_points_saved,
        NULL as player1_break_points_faced,
        NULL as player2_aces,
        NULL as player2_double_faults,
        NULL as player2_serve_points,
        NULL as player2_first_serves_in,
        NULL as player2_first_serve_points_won,
        NULL as player2_second_serve_points_won,
        NULL as player2_service_games,
        NULL as player2_break_points_saved,
        NULL as player2_break_points_faced,
        NULL as player1_elo,
        NULL as player2_elo,
        NULL as player1_matches,
        NULL as player2_matches,
        match_type,
        NULL as result  -- No result for upcoming matches
    FROM upcoming_matches
    WHERE status = 'scheduled'
    """
    
    df = pd.read_sql(query, engine)
    logger.info(f"Loaded {len(df)} upcoming matches")
    return df

def prepare_player_rankings(historical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare player ranking information from historical matches.
    
    This creates a dataframe with the latest ranking information for each player.
    """
    # Extract player1 (winner) data
    player1_df = historical_df[['tournament_date', 'player1_id', 'player1_name', 'player1_rank', 'player1_rank_points', 'player1_elo']].copy()
    player1_df.columns = ['date', 'player_id', 'player_name', 'rank', 'rank_points', 'elo']
    
    # Extract player2 (loser) data
    player2_df = historical_df[['tournament_date', 'player2_id', 'player2_name', 'player2_rank', 'player2_rank_points', 'player2_elo']].copy()
    player2_df.columns = ['date', 'player_id', 'player_name', 'rank', 'rank_points', 'elo']
    
    # Combine data
    players_df = pd.concat([player1_df, player2_df], ignore_index=True)
    
    # Sort by date (latest first) and drop duplicates to get most recent rank
    players_df = players_df.sort_values('date', ascending=False)
    players_df = players_df.drop_duplicates(subset=['player_id'])
    
    return players_df

def fill_missing_rankings(upcoming_df: pd.DataFrame, rankings_df: pd.DataFrame) -> pd.DataFrame:
    """Fill in missing ranking data for upcoming matches using historical data."""
    # Create a copy to avoid modifying the original
    df = upcoming_df.copy()
    
    # Map player rankings
    player_id_to_rank = dict(zip(rankings_df['player_id'], rankings_df['rank']))
    player_id_to_points = dict(zip(rankings_df['player_id'], rankings_df['rank_points']))
    player_id_to_elo = dict(zip(rankings_df['player_id'], rankings_df['elo']))
    
    # Fill missing values for player1
    mask = df['player1_rank'].isna()
    df.loc[mask, 'player1_rank'] = df.loc[mask, 'player1_id'].map(player_id_to_rank)
    
    mask = df['player1_rank_points'].isna()
    df.loc[mask, 'player1_rank_points'] = df.loc[mask, 'player1_id'].map(player_id_to_points)
    
    df['player1_elo'] = df['player1_id'].map(player_id_to_elo)
    
    # Fill missing values for player2
    mask = df['player2_rank'].isna()
    df.loc[mask, 'player2_rank'] = df.loc[mask, 'player2_id'].map(player_id_to_rank)
    
    mask = df['player2_rank_points'].isna()
    df.loc[mask, 'player2_rank_points'] = df.loc[mask, 'player2_id'].map(player_id_to_points)
    
    df['player2_elo'] = df['player2_id'].map(player_id_to_elo)
    
    return df

def merge_historical_and_upcoming(historical_df: pd.DataFrame, upcoming_df: pd.DataFrame) -> pd.DataFrame:
    """Merge historical and upcoming match data for feature generation."""
    # Ensure columns match
    common_columns = set(historical_df.columns) & set(upcoming_df.columns)
    
    # Merge dataframes
    merged_df = pd.concat([
        historical_df[common_columns],
        upcoming_df[common_columns]
    ], ignore_index=True)
    
    # Sort by date
    merged_df = merged_df.sort_values('tournament_date', ascending=True)
    
    return merged_df

def generate_features_for_upcoming_matches(
    historical_df: pd.DataFrame, 
    upcoming_df: pd.DataFrame,
    progress_tracker: Optional[ProgressTracker] = None
) -> pd.DataFrame:
    """Generate features for upcoming matches using historical data."""
    # Prepare player rankings and fill missing values
    rankings_df = prepare_player_rankings(historical_df)
    upcoming_df_filled = fill_missing_rankings(upcoming_df, rankings_df)
    
    if progress_tracker:
        progress_tracker.update("Filled missing ranking data")
    
    # Merge for feature calculation
    all_matches_df = merge_historical_and_upcoming(historical_df, upcoming_df_filled)
    
    if progress_tracker:
        progress_tracker.update("Merged historical and upcoming data")
        
    # Calculate win rates
    player_df = calculate_win_rates(all_matches_df)
    
    if progress_tracker:
        progress_tracker.update("Calculated win rates")
    
    # Calculate serve/return stats
    serve_return_df = calculate_serve_return_stats(all_matches_df)
    serve_return_rolling_df = calculate_serve_return_rolling_stats(serve_return_df)
    
    if progress_tracker:
        progress_tracker.update("Calculated serve/return stats")
    
    # Process features for upcoming matches
    surface_mapping = {
        'hard': 'hard',
        'clay': 'clay',
        'grass': 'grass',
        'carpet': 'carpet'
    }
    
    batch_size = 100
    num_batches = (len(upcoming_df) + batch_size - 1) // batch_size
    
    if progress_tracker:
        # Reset tracker for batch processing
        progress_tracker.total_steps = num_batches
        progress_tracker.current_step = 0
        progress_tracker.task_description = "Processing match batches"
    
    all_features = []
    upcoming_match_ids = upcoming_df['match_id'].tolist()
    
    for i in range(0, len(upcoming_df), batch_size):
        batch = upcoming_df.iloc[i:i+batch_size]
        # Convert to records for processing
        batch_data = batch.to_dict('records')
        
        # Process batch
        batch_features = process_match_features_batch(batch_data, player_df, serve_return_rolling_df, surface_mapping)
        all_features.extend(batch_features)
        
        if progress_tracker:
            progress_tracker.update(f"Processed batch {i//batch_size + 1}/{num_batches}")
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    
    # Ensure match_id is in the features
    features_df['match_id'] = upcoming_match_ids[:len(features_df)]
    
    return features_df

def save_features_to_db(features_df: pd.DataFrame):
    """Save generated features to the database."""
    if features_df.empty:
        logger.warning("No features to save")
        return
    
    engine = get_sqlalchemy_engine()
    conn = engine.connect()
    
    try:
        # Create temporary table
        temp_table_name = f"temp_upcoming_features_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        features_df.to_sql(temp_table_name, conn, if_exists='replace', index=False)
        
        # Create upcoming_match_features table if it doesn't exist
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS upcoming_match_features (
            id SERIAL PRIMARY KEY,
            match_id VARCHAR(50) UNIQUE REFERENCES upcoming_matches(match_id),
            player1_id INTEGER,
            player2_id INTEGER,
            surface VARCHAR(50),
            tournament_date DATE,
            tournament_level VARCHAR(50),
            player_elo_diff DOUBLE PRECISION,
            win_rate_5_diff DOUBLE PRECISION,
            win_streak_diff BIGINT,
            loss_streak_diff BIGINT,
            win_rate_hard_5_diff DOUBLE PRECISION,
            win_rate_clay_5_diff DOUBLE PRECISION,
            win_rate_grass_5_diff DOUBLE PRECISION,
            win_rate_carpet_5_diff DOUBLE PRECISION,
            win_rate_hard_overall_diff DOUBLE PRECISION,
            win_rate_clay_overall_diff DOUBLE PRECISION,
            win_rate_grass_overall_diff DOUBLE PRECISION,
            win_rate_carpet_overall_diff DOUBLE PRECISION,
            serve_efficiency_5_diff DOUBLE PRECISION,
            first_serve_pct_5_diff DOUBLE PRECISION,
            first_serve_win_pct_5_diff DOUBLE PRECISION,
            second_serve_win_pct_5_diff DOUBLE PRECISION,
            ace_pct_5_diff DOUBLE PRECISION,
            bp_saved_pct_5_diff DOUBLE PRECISION,
            return_efficiency_5_diff DOUBLE PRECISION,
            bp_conversion_pct_5_diff DOUBLE PRECISION,
            player1_win_rate_5 DOUBLE PRECISION,
            player2_win_rate_5 DOUBLE PRECISION,
            player1_win_streak BIGINT,
            player2_win_streak BIGINT,
            player1_loss_streak BIGINT,
            player2_loss_streak BIGINT,
            player1_win_rate_hard_5 DOUBLE PRECISION,
            player2_win_rate_hard_5 DOUBLE PRECISION,
            player1_win_rate_clay_5 DOUBLE PRECISION,
            player2_win_rate_clay_5 DOUBLE PRECISION,
            player1_win_rate_grass_5 DOUBLE PRECISION,
            player2_win_rate_grass_5 DOUBLE PRECISION,
            player1_win_rate_carpet_5 DOUBLE PRECISION,
            player2_win_rate_carpet_5 DOUBLE PRECISION,
            player1_win_rate_hard_overall DOUBLE PRECISION,
            player2_win_rate_hard_overall DOUBLE PRECISION,
            player1_win_rate_clay_overall DOUBLE PRECISION,
            player2_win_rate_clay_overall DOUBLE PRECISION,
            player1_win_rate_grass_overall DOUBLE PRECISION,
            player2_win_rate_grass_overall DOUBLE PRECISION,
            player1_win_rate_carpet_overall DOUBLE PRECISION,
            player2_win_rate_carpet_overall DOUBLE PRECISION,
            player1_serve_efficiency_5 DOUBLE PRECISION,
            player2_serve_efficiency_5 DOUBLE PRECISION,
            player1_first_serve_pct_5 DOUBLE PRECISION,
            player2_first_serve_pct_5 DOUBLE PRECISION,
            player1_first_serve_win_pct_5 DOUBLE PRECISION,
            player2_first_serve_win_pct_5 DOUBLE PRECISION,
            player1_second_serve_win_pct_5 DOUBLE PRECISION,
            player2_second_serve_win_pct_5 DOUBLE PRECISION,
            player1_ace_pct_5 DOUBLE PRECISION,
            player2_ace_pct_5 DOUBLE PRECISION,
            player1_bp_saved_pct_5 DOUBLE PRECISION,
            player2_bp_saved_pct_5 DOUBLE PRECISION,
            player1_return_efficiency_5 DOUBLE PRECISION,
            player2_return_efficiency_5 DOUBLE PRECISION,
            player1_bp_conversion_pct_5 DOUBLE PRECISION,
            player2_bp_conversion_pct_5 DOUBLE PRECISION,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """))
        
        # Insert from temp table
        key_columns = ['match_id', 'player1_id', 'player2_id', 'surface', 'tournament_date', 'tournament_level']
        feature_columns = [col for col in features_df.columns if col not in ['id', 'created_at', 'updated_at'] and col in features_df.columns]
        
        # Prepare column lists
        columns_str = ", ".join(feature_columns)
        source_columns_str = ", ".join([f"source.{col}" for col in feature_columns])
        excluded_columns_str = ", ".join([f"{col} = EXCLUDED.{col}" for col in feature_columns if col != 'match_id'])
        
        # Upsert query
        upsert_query = f"""
        INSERT INTO upcoming_match_features ({columns_str}, created_at, updated_at)
        SELECT {source_columns_str}, NOW(), NOW()
        FROM {temp_table_name} source
        ON CONFLICT (match_id) DO UPDATE SET
            {excluded_columns_str},
            updated_at = NOW()
        """
        
        conn.execute(text(upsert_query))
        
        # Drop temp table
        conn.execute(text(f"DROP TABLE IF EXISTS {temp_table_name}"))
        
        logger.info(f"Successfully saved features for {len(features_df)} upcoming matches")
    except Exception as e:
        logger.error(f"Error saving features to database: {e}")
        raise
    finally:
        conn.close()

def main():
    """Main function to generate features for upcoming matches."""
    logger.info("Starting feature generation for upcoming matches")
    
    # Create progress tracker
    progress_tracker = ProgressTracker(5, "Feature generation")
    
    # Load data
    historical_df = load_historical_matches()
    upcoming_df = load_upcoming_matches()
    
    if upcoming_df.empty:
        logger.info("No upcoming matches to process")
        return
    
    progress_tracker.update("Loaded data")
    
    # Generate features
    features_df = generate_features_for_upcoming_matches(historical_df, upcoming_df, progress_tracker)
    
    if features_df.empty:
        logger.warning("No features generated")
        return
    
    progress_tracker.update("Generated features")
    
    # Save features to database
    save_features_to_db(features_df)
    progress_tracker.update("Saved features to database")
    
    logger.info(f"Completed feature generation for {len(features_df)} upcoming matches")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate features for upcoming tennis matches")
    args = parser.parse_args()
    
    main() 