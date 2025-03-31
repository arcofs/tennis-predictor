import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import logging
from tqdm import tqdm
import time
import multiprocessing
from functools import partial
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values

# Multiprocessing settings
# Set to 0 to use all available cores, or specify a number to limit the cores used
NUM_CORES = 4  # Default: use a reasonable number of cores to prevent system overload

# If NUM_CORES is set to 0, use a reasonable number of cores (not all available)
if NUM_CORES <= 0:
    NUM_CORES = min(8, multiprocessing.cpu_count())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project directories
PROJECT_ROOT = Path(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
DATA_DIR = PROJECT_ROOT / "data"
CLEANED_DATA_DIR = DATA_DIR / "cleaned"
OUTPUT_DIR = DATA_DIR / "v3"
PRED_OUTPUT_DIR = PROJECT_ROOT / "predictor" / "output" / "v3"

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PRED_OUTPUT_DIR, exist_ok=True)

# File paths
INPUT_FILE = CLEANED_DATA_DIR / "cleaned_dataset_with_elo.csv"
OUTPUT_FILE = OUTPUT_DIR / "features_v3.csv"

# Constants
ROLLING_WINDOWS = [5, 10]

# Standard surface definitions
SURFACE_HARD = 'hard'
SURFACE_CLAY = 'clay'
SURFACE_GRASS = 'grass'
SURFACE_CARPET = 'carpet'
STANDARD_SURFACES = [SURFACE_HARD, SURFACE_CLAY, SURFACE_GRASS, SURFACE_CARPET]

# Updated column mappings for database
SERVE_COLS = {
    'winner': ['winner_aces', 'winner_double_faults', 'winner_serve_points', 'winner_first_serves_in', 
               'winner_first_serve_points_won', 'winner_second_serve_points_won', 'winner_service_games', 
               'winner_break_points_saved', 'winner_break_points_faced'],
    'loser': ['loser_aces', 'loser_double_faults', 'loser_serve_points', 'loser_first_serves_in', 
              'loser_first_serve_points_won', 'loser_second_serve_points_won', 'loser_service_games', 
              'loser_break_points_saved', 'loser_break_points_faced']
}

# Updated serve and return stats columns for database
SERVE_STATS_COLUMNS = SERVE_COLS['winner']
RETURN_STATS_COLUMNS = SERVE_COLS['loser']

# Database settings
DB_BATCH_SIZE = 10000
DB_PAGE_SIZE = 1000

def get_database_connection() -> create_engine:
    """Create database connection using environment variables"""
    load_dotenv()
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL not found in environment variables")
    
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    
    # Add connection timeout and command timeout settings
    return create_engine(
        database_url,
        connect_args={
            'connect_timeout': 10,  # Connection timeout in seconds
            'options': '-c statement_timeout=300000'  # Query timeout in milliseconds (5 minutes)
        }
    )

def get_psycopg2_connection():
    """Create a psycopg2 connection for efficient batch operations"""
    load_dotenv()
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL not found in environment variables")
    
    # Add connection timeout and statement timeout
    return psycopg2.connect(
        database_url,
        connect_timeout=10,  # Connection timeout in seconds
        options='-c statement_timeout=300000'  # Query timeout in milliseconds (5 minutes)
    )

def create_features_table(conn):
    """Create the match_features table if it doesn't exist and add any missing columns"""
    with conn.cursor() as cur:
        # First check if the table exists
        cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'match_features'
        );
        """)
        table_exists = cur.fetchone()[0]
        
        if not table_exists:
            logger.info("Creating match_features table...")
            # Create the table with all required columns, using lowercase for all column names
            cur.execute("""
            CREATE TABLE match_features (
                id SERIAL PRIMARY KEY,
                match_id BIGINT NOT NULL UNIQUE,
                player1_id BIGINT NOT NULL,
                player2_id BIGINT NOT NULL,
                surface VARCHAR(50),
                tournament_date DATE,
                result INTEGER,
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
            
            -- Add indexes for faster querying
            CREATE INDEX idx_match_features_match_id ON match_features(match_id);
            CREATE INDEX idx_match_features_player1_id ON match_features(player1_id);
            CREATE INDEX idx_match_features_player2_id ON match_features(player2_id);
            CREATE INDEX idx_match_features_tournament_date ON match_features(tournament_date);
            """)
            conn.commit()
            logger.info("Successfully created match_features table")
        else:
            # Check for missing columns and add them if needed
            logger.info("Table already exists. Checking for missing columns...")
            
            # Define all columns that should exist in the table with their types (all lowercase)
            expected_columns = {
                'match_id': 'BIGINT',
                'player1_id': 'BIGINT',
                'player2_id': 'BIGINT',
                'surface': 'VARCHAR(50)',
                'tournament_date': 'DATE',
                'result': 'INTEGER',
                'player_elo_diff': 'DOUBLE PRECISION',
                'win_rate_5_diff': 'DOUBLE PRECISION',
                'win_streak_diff': 'BIGINT',
                'loss_streak_diff': 'BIGINT',
                'win_rate_hard_5_diff': 'DOUBLE PRECISION',
                'win_rate_clay_5_diff': 'DOUBLE PRECISION',
                'win_rate_grass_5_diff': 'DOUBLE PRECISION',
                'win_rate_carpet_5_diff': 'DOUBLE PRECISION',
                'win_rate_hard_overall_diff': 'DOUBLE PRECISION',
                'win_rate_clay_overall_diff': 'DOUBLE PRECISION',
                'win_rate_grass_overall_diff': 'DOUBLE PRECISION',
                'win_rate_carpet_overall_diff': 'DOUBLE PRECISION',
                'serve_efficiency_5_diff': 'DOUBLE PRECISION',
                'first_serve_pct_5_diff': 'DOUBLE PRECISION',
                'first_serve_win_pct_5_diff': 'DOUBLE PRECISION',
                'second_serve_win_pct_5_diff': 'DOUBLE PRECISION',
                'ace_pct_5_diff': 'DOUBLE PRECISION',
                'bp_saved_pct_5_diff': 'DOUBLE PRECISION',
                'return_efficiency_5_diff': 'DOUBLE PRECISION',
                'bp_conversion_pct_5_diff': 'DOUBLE PRECISION',
                'player1_win_rate_5': 'DOUBLE PRECISION',
                'player2_win_rate_5': 'DOUBLE PRECISION',
                'player1_win_streak': 'BIGINT',
                'player2_win_streak': 'BIGINT',
                'player1_loss_streak': 'BIGINT',
                'player2_loss_streak': 'BIGINT',
                'player1_win_rate_hard_5': 'DOUBLE PRECISION',
                'player2_win_rate_hard_5': 'DOUBLE PRECISION',
                'player1_win_rate_clay_5': 'DOUBLE PRECISION',
                'player2_win_rate_clay_5': 'DOUBLE PRECISION',
                'player1_win_rate_grass_5': 'DOUBLE PRECISION',
                'player2_win_rate_grass_5': 'DOUBLE PRECISION',
                'player1_win_rate_carpet_5': 'DOUBLE PRECISION',
                'player2_win_rate_carpet_5': 'DOUBLE PRECISION',
                'player1_win_rate_hard_overall': 'DOUBLE PRECISION',
                'player2_win_rate_hard_overall': 'DOUBLE PRECISION',
                'player1_win_rate_clay_overall': 'DOUBLE PRECISION',
                'player2_win_rate_clay_overall': 'DOUBLE PRECISION',
                'player1_win_rate_grass_overall': 'DOUBLE PRECISION',
                'player2_win_rate_grass_overall': 'DOUBLE PRECISION',
                'player1_win_rate_carpet_overall': 'DOUBLE PRECISION',
                'player2_win_rate_carpet_overall': 'DOUBLE PRECISION',
                'player1_serve_efficiency_5': 'DOUBLE PRECISION',
                'player2_serve_efficiency_5': 'DOUBLE PRECISION',
                'player1_first_serve_pct_5': 'DOUBLE PRECISION',
                'player2_first_serve_pct_5': 'DOUBLE PRECISION',
                'player1_first_serve_win_pct_5': 'DOUBLE PRECISION',
                'player2_first_serve_win_pct_5': 'DOUBLE PRECISION',
                'player1_second_serve_win_pct_5': 'DOUBLE PRECISION',
                'player2_second_serve_win_pct_5': 'DOUBLE PRECISION',
                'player1_ace_pct_5': 'DOUBLE PRECISION',
                'player2_ace_pct_5': 'DOUBLE PRECISION',
                'player1_bp_saved_pct_5': 'DOUBLE PRECISION',
                'player2_bp_saved_pct_5': 'DOUBLE PRECISION',
                'player1_return_efficiency_5': 'DOUBLE PRECISION',
                'player2_return_efficiency_5': 'DOUBLE PRECISION',
                'player1_bp_conversion_pct_5': 'DOUBLE PRECISION',
                'player2_bp_conversion_pct_5': 'DOUBLE PRECISION'
            }
            
            # Get existing columns (convert to lowercase for case-insensitive comparison)
            cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'match_features'
            """)
            existing_columns = {row[0].lower() for row in cur.fetchall()}
            
            # Find missing columns
            missing_columns = {col: data_type for col, data_type in expected_columns.items() 
                              if col.lower() not in existing_columns}
            
            # Add missing columns
            for col_name, data_type in missing_columns.items():
                logger.info(f"Adding missing column: {col_name}")
                cur.execute(f"""
                ALTER TABLE match_features 
                ADD COLUMN {col_name} {data_type}
                """)
            
            if missing_columns:
                conn.commit()
                logger.info(f"Added {len(missing_columns)} missing columns")
            else:
                logger.info("No missing columns found")
            
            # Check if match_id has a UNIQUE constraint
            cur.execute("""
            SELECT COUNT(*) FROM information_schema.table_constraints 
            WHERE table_name = 'match_features'
            AND constraint_type = 'UNIQUE'
            AND constraint_name LIKE '%match_id%'
            """)
            
            unique_constraint_exists = cur.fetchone()[0] > 0
            
            if not unique_constraint_exists:
                logger.info("Adding UNIQUE constraint to match_id")
                try:
                    cur.execute("""
                    ALTER TABLE match_features 
                    ADD CONSTRAINT match_features_match_id_key UNIQUE (match_id)
                    """)
                    conn.commit()
                    logger.info("Added UNIQUE constraint to match_id")
                except Exception as e:
                    logger.warning(f"Could not add UNIQUE constraint: {str(e)}")
                    conn.rollback()
                    
            # Check for necessary indexes
            for idx_name, col_name in [
                ('idx_match_features_match_id', 'match_id'),
                ('idx_match_features_player1_id', 'player1_id'),
                ('idx_match_features_player2_id', 'player2_id'),
                ('idx_match_features_tournament_date', 'tournament_date')
            ]:
                cur.execute(f"""
                SELECT COUNT(*) FROM pg_indexes
                WHERE tablename = 'match_features'
                AND indexname = '{idx_name}'
                """)
                
                index_exists = cur.fetchone()[0] > 0
                
                if not index_exists:
                    logger.info(f"Adding index {idx_name} on {col_name}")
                    try:
                        cur.execute(f"""
                        CREATE INDEX {idx_name} ON match_features({col_name})
                        """)
                        conn.commit()
                        logger.info(f"Added index {idx_name}")
                    except Exception as e:
                        logger.warning(f"Could not add index {idx_name}: {str(e)}")
                        conn.rollback()

def load_data() -> pd.DataFrame:
    """
    Load the tennis match dataset from the database.
    
    Returns:
        DataFrame with tennis match data
    """
    logger.info("Connecting to database...")
    engine = get_database_connection()
    logger.info("Successfully connected to database")
    
    logger.info("Loading data from database...")
    
    query = """
        SELECT 
            id as match_id,
            tournament_date,
            tournament_id,
            tournament_name,
            surface,
            tournament_level,
            winner_id,
            winner_name,
            winner_hand,
            winner_height_cm,
            winner_country_code,
            winner_age,
            loser_id,
            loser_name,
            loser_hand,
            loser_height_cm,
            loser_country_code,
            loser_age,
            winner_aces,
            winner_double_faults,
            winner_serve_points,
            winner_first_serves_in,
            winner_first_serve_points_won,
            winner_second_serve_points_won,
            winner_service_games,
            winner_break_points_saved,
            winner_break_points_faced,
            loser_aces,
            loser_double_faults,
            loser_serve_points,
            loser_first_serves_in,
            loser_first_serve_points_won,
            loser_second_serve_points_won,
            loser_service_games,
            loser_break_points_saved,
            loser_break_points_faced,
            winner_elo,
            loser_elo,
            winner_matches,
            loser_matches
        FROM matches
        WHERE winner_id IS NOT NULL 
        AND loser_id IS NOT NULL
        ORDER BY tournament_date ASC
    """
    
    df = pd.read_sql(query, engine)
    
    # Convert date columns to datetime
    df['tournament_date'] = pd.to_datetime(df['tournament_date'])
    
    # Standardize surface types
    if 'surface' in df.columns:
        # First convert to lowercase
        df['surface'] = df['surface'].str.lower()
        
        # Map non-standard surfaces to standard ones
        surface_mapping = {
            'carpet': SURFACE_CARPET,
            'hard court': SURFACE_HARD,
            'clay court': SURFACE_CLAY,
            'grass court': SURFACE_GRASS
        }
        df['surface'] = df['surface'].map(lambda x: surface_mapping.get(x, x))
        
        # Ensure all surfaces are in the standard list
        df['surface'] = df['surface'].replace({
            'hard': SURFACE_HARD,
            'clay': SURFACE_CLAY,
            'grass': SURFACE_GRASS,
            'carpet': SURFACE_CARPET
        })
    
    logger.info(f"Successfully loaded {len(df)} matches from {len(df['tournament_id'].unique())} tournaments")
    
    return df

def calculate_win_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate win rates and streaks for players without introducing player position bias.
    
    Args:
        df: Match dataset
        
    Returns:
        DataFrame with win rate features
    """
    logger.info("Calculating win rates and streaks...")
    
    # Create a player-centric view of the data - one row per player per match
    matches = []
    
    # Process matches chronologically with progress bar
    for idx, row in tqdm(df.sort_values('tournament_date').iterrows(), 
                          total=len(df), 
                          desc="Processing matches for win rates", 
                          unit="match"):
        # Make sure surface is lowercase
        surface = row['surface'].lower() if isinstance(row['surface'], str) else row['surface']
        
        # Process winner
        winner_dict = {
            'match_id': row['match_id'],
            'player_id': row['winner_id'],
            'opponent_id': row['loser_id'],
            'tournament_date': row['tournament_date'],
            'surface': surface,
            'result': 1  # 1 means win
        }
        matches.append(winner_dict)
        
        # Process loser
        loser_dict = {
            'match_id': row['match_id'],
            'player_id': row['loser_id'],
            'opponent_id': row['winner_id'],
            'tournament_date': row['tournament_date'], 
            'surface': surface,
            'result': 0  # 0 means loss
        }
        matches.append(loser_dict)
    
    # Create player-centric dataframe
    player_df = pd.DataFrame(matches)
    
    # Sort by player and date
    player_df = player_df.sort_values(['player_id', 'tournament_date'])
    
    # Calculate overall win rates over different windows
    time_windows = [5]  # Focus on recent 5 matches as indicated by feature importance
    
    logger.info("Calculating rolling win rates...")
    # Overall win rates
    for window in time_windows:
        player_df[f'win_rate_{window}'] = (
            player_df.groupby('player_id')['result']
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
    
    # Surface-specific win rates (using lowercase surfaces)
    for surface in tqdm(STANDARD_SURFACES, desc="Processing surface-specific win rates", unit="surface"):
        # Create mask for this surface
        surface_mask = player_df['surface'] == surface
        
        for window in time_windows:
            # Initialize column with NaN values
            player_df[f'win_rate_{surface}_{window}'] = np.nan
            
            # Group by player and calculate win rate for this surface
            surface_rates = (
                player_df[surface_mask]
                .groupby('player_id')['result']
                .rolling(window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            
            # Update values for this surface
            player_df.loc[surface_mask, f'win_rate_{surface}_{window}'] = surface_rates
            
            # Forward fill the values - keep previous surface win rate until next match on this surface
            player_df[f'win_rate_{surface}_{window}'] = (
                player_df
                .groupby('player_id')[f'win_rate_{surface}_{window}']
                .transform(lambda x: x.ffill())
            )
            
    # Calculate overall win rate per surface (not just based on recent matches)
    logger.info("Calculating overall surface win rates...")
    for surface in STANDARD_SURFACES:
        # For each player, calculate their overall win rate on each surface
        surface_overall = (
            player_df[player_df['surface'] == surface]
            .groupby('player_id')['result']
            .expanding()
            .mean()
            .reset_index(level=0, drop=True)
        )
        player_df.loc[player_df['surface'] == surface, f'win_rate_{surface}_overall'] = surface_overall
        
        # Forward fill these values
        player_df[f'win_rate_{surface}_overall'] = (
            player_df
            .groupby('player_id')[f'win_rate_{surface}_overall']
            .transform(lambda x: x.ffill())
        )
    
    # Calculate win and loss streaks
    logger.info("Calculating win/loss streaks...")
    player_df['win_streak'] = 0
    player_df['loss_streak'] = 0
    
    # Group by player
    unique_players = player_df['player_id'].unique()
    for player_id in tqdm(unique_players, desc="Calculating player streaks", unit="player"):
        group = player_df[player_df['player_id'] == player_id]
        
        # Initialize streaks
        win_streak = 0
        loss_streak = 0
        win_streaks = []
        loss_streaks = []
        
        # Calculate streaks for each match
        for result in group['result']:
            if result == 1:  # Win
                win_streak += 1
                loss_streak = 0
            else:  # Loss
                loss_streak += 1
                win_streak = 0
            win_streaks.append(win_streak)
            loss_streaks.append(loss_streak)
        
        # Update the dataframe
        player_df.loc[group.index, 'win_streak'] = win_streaks
        player_df.loc[group.index, 'loss_streak'] = loss_streaks
    
    # Ensure we have no missing values in key columns
    for col in ['win_rate_5', 'win_streak', 'loss_streak']:
        player_df[col] = player_df[col].fillna(0)
    
    # For surface-specific win rates, keep NaN values if a player has never played on that surface
    # XGBoost will handle these NaN values appropriately
    
    return player_df

def calculate_serve_return_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate serve and return statistics for each player-match.
    
    Args:
        df: Original match dataset
        
    Returns:
        DataFrame with player-centric serve and return stats
    """
    logger.info("Calculating serve and return statistics...")
    
    # Create a player-centric view for serve and return stats
    matches = []
    
    # Process each match chronologically with progress bar
    for idx, row in tqdm(df.sort_values('tournament_date').iterrows(), 
                          total=len(df), 
                          desc="Processing matches for serve/return stats", 
                          unit="match"):
        
        # Check if serve stats are available for this match
        has_serve_stats = not pd.isna(row['winner_serve_points']) and row['winner_serve_points'] > 0
        
        # Process winner's serve and return stats
        winner_dict = {
            'match_id': row['match_id'],
            'player_id': row['winner_id'],
            'opponent_id': row['loser_id'],
            'tournament_date': row['tournament_date'],
            'surface': row['surface']
        }
        
        # Add serve stats for winner
        if has_serve_stats:
            # Basic serve stats directly from dataset
            winner_dict['aces'] = row['winner_aces']
            winner_dict['double_faults'] = row['winner_double_faults']
            winner_dict['serve_points'] = row['winner_serve_points']
            winner_dict['first_serves_in'] = row['winner_first_serves_in']
            winner_dict['first_serve_points_won'] = row['winner_first_serve_points_won']
            winner_dict['second_serve_points_won'] = row['winner_second_serve_points_won']
            winner_dict['service_games'] = row['winner_service_games']
            winner_dict['break_points_saved'] = row['winner_break_points_saved']
            winner_dict['break_points_faced'] = row['winner_break_points_faced']
            
            # Calculate derived serve metrics
            # 1. First serve percentage
            winner_dict['first_serve_pct'] = row['winner_first_serves_in'] / row['winner_serve_points'] if row['winner_serve_points'] > 0 else np.nan
            
            # 2. First serve win percentage
            winner_dict['first_serve_win_pct'] = row['winner_first_serve_points_won'] / row['winner_first_serves_in'] if row['winner_first_serves_in'] > 0 else np.nan
            
            # 3. Second serve win percentage
            second_serves = row['winner_serve_points'] - row['winner_first_serves_in'] if row['winner_serve_points'] > 0 else 0
            winner_dict['second_serve_win_pct'] = row['winner_second_serve_points_won'] / second_serves if second_serves > 0 else np.nan
            
            # 4. Service efficiency (overall serve points won percentage)
            serve_points_won = row['winner_first_serve_points_won'] + row['winner_second_serve_points_won']
            winner_dict['serve_efficiency'] = serve_points_won / row['winner_serve_points'] if row['winner_serve_points'] > 0 else np.nan
            
            # 5. Ace percentage
            winner_dict['ace_pct'] = row['winner_aces'] / row['winner_serve_points'] if row['winner_serve_points'] > 0 else np.nan
            
            # 6. Break points saved percentage
            winner_dict['bp_saved_pct'] = row['winner_break_points_saved'] / row['winner_break_points_faced'] if row['winner_break_points_faced'] > 0 else np.nan
            
            # Add return stats for winner (using loser's serve stats)
            winner_dict['return_points'] = row['loser_serve_points']
            winner_dict['return_points_won'] = row['loser_serve_points'] - (row['loser_first_serve_points_won'] + row['loser_second_serve_points_won']) if row['loser_serve_points'] > 0 else np.nan
            
            # 7. Return efficiency (percentage of opponent's serve points won)
            winner_dict['return_efficiency'] = winner_dict['return_points_won'] / row['loser_serve_points'] if row['loser_serve_points'] > 0 else np.nan
            
            # 8. Break points converted
            winner_dict['break_points_converted'] = row['loser_break_points_faced'] - row['loser_break_points_saved'] if row['loser_break_points_faced'] > 0 else 0
            
            # 9. Break point conversion percentage
            winner_dict['bp_conversion_pct'] = winner_dict['break_points_converted'] / row['loser_break_points_faced'] if row['loser_break_points_faced'] > 0 else np.nan
        
        matches.append(winner_dict)
        
        # Process loser's serve and return stats
        loser_dict = {
            'match_id': row['match_id'],
            'player_id': row['loser_id'],
            'opponent_id': row['winner_id'],
            'tournament_date': row['tournament_date'],
            'surface': row['surface']
        }
        
        # Add serve stats for loser
        if has_serve_stats:
            # Basic serve stats directly from dataset
            loser_dict['aces'] = row['loser_aces']
            loser_dict['double_faults'] = row['loser_double_faults']
            loser_dict['serve_points'] = row['loser_serve_points']
            loser_dict['first_serves_in'] = row['loser_first_serves_in']
            loser_dict['first_serve_points_won'] = row['loser_first_serve_points_won']
            loser_dict['second_serve_points_won'] = row['loser_second_serve_points_won']
            loser_dict['service_games'] = row['loser_service_games']
            loser_dict['break_points_saved'] = row['loser_break_points_saved']
            loser_dict['break_points_faced'] = row['loser_break_points_faced']
            
            # Calculate derived serve metrics
            # 1. First serve percentage
            loser_dict['first_serve_pct'] = row['loser_first_serves_in'] / row['loser_serve_points'] if row['loser_serve_points'] > 0 else np.nan
            
            # 2. First serve win percentage
            loser_dict['first_serve_win_pct'] = row['loser_first_serve_points_won'] / row['loser_first_serves_in'] if row['loser_first_serves_in'] > 0 else np.nan
            
            # 3. Second serve win percentage
            second_serves = row['loser_serve_points'] - row['loser_first_serves_in'] if row['loser_serve_points'] > 0 else 0
            loser_dict['second_serve_win_pct'] = row['loser_second_serve_points_won'] / second_serves if second_serves > 0 else np.nan
            
            # 4. Service efficiency (overall serve points won percentage)
            serve_points_won = row['loser_first_serve_points_won'] + row['loser_second_serve_points_won']
            loser_dict['serve_efficiency'] = serve_points_won / row['loser_serve_points'] if row['loser_serve_points'] > 0 else np.nan
            
            # 5. Ace percentage
            loser_dict['ace_pct'] = row['loser_aces'] / row['loser_serve_points'] if row['loser_serve_points'] > 0 else np.nan
            
            # 6. Break points saved percentage
            loser_dict['bp_saved_pct'] = row['loser_break_points_saved'] / row['loser_break_points_faced'] if row['loser_break_points_faced'] > 0 else np.nan
            
            # Add return stats for loser (using winner's serve stats)
            loser_dict['return_points'] = row['winner_serve_points']
            loser_dict['return_points_won'] = row['winner_serve_points'] - (row['winner_first_serve_points_won'] + row['winner_second_serve_points_won']) if row['winner_serve_points'] > 0 else np.nan
            
            # 7. Return efficiency (percentage of opponent's serve points won)
            loser_dict['return_efficiency'] = loser_dict['return_points_won'] / row['winner_serve_points'] if row['winner_serve_points'] > 0 else np.nan
            
            # 8. Break points converted
            loser_dict['break_points_converted'] = row['winner_break_points_faced'] - row['winner_break_points_saved'] if row['winner_break_points_faced'] > 0 else 0
            
            # 9. Break point conversion percentage
            loser_dict['bp_conversion_pct'] = loser_dict['break_points_converted'] / row['winner_break_points_faced'] if row['winner_break_points_faced'] > 0 else np.nan
        
        matches.append(loser_dict)
    
    # Create player stats dataframe
    player_stats_df = pd.DataFrame(matches)
    
    # Sort by player and date
    player_stats_df = player_stats_df.sort_values(['player_id', 'tournament_date'])
    
    return player_stats_df

def calculate_serve_return_rolling_stats(player_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate rolling averages for serve and return statistics.
    
    Args:
        player_df: Player-centric dataframe with serve and return stats
        
    Returns:
        DataFrame with rolling averages
    """
    logger.info("Calculating rolling serve and return stats...")
    
    # Create a copy of the dataframe
    df = player_df.copy()
    
    # Define the serve and return metrics for which we'll calculate rolling averages
    serve_metrics = [
        'serve_efficiency',
        'first_serve_pct',
        'first_serve_win_pct',
        'second_serve_win_pct',
        'ace_pct',
        'bp_saved_pct'
    ]
    
    return_metrics = [
        'return_efficiency',
        'bp_conversion_pct'
    ]
    
    # Define window sizes for rolling averages
    windows = [5, 10]
    
    # Calculate rolling averages for each metric
    with tqdm(total=len(serve_metrics + return_metrics) * len(windows), 
              desc="Computing rolling stats", 
              unit="metric") as pbar:
        
        # Serve metrics
        for metric in serve_metrics:
            # Skip if the metric is not in the dataframe
            if metric not in df.columns:
                continue
                
            for window in windows:
                # Calculate rolling average
                col_name = f'{metric}_{window}'
                df[col_name] = (
                    df.groupby('player_id')[metric]
                    .rolling(window, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )
                
                # Calculate surface-specific rolling average
                for surface in STANDARD_SURFACES:
                    # Create mask for this surface (using lowercase surface)
                    surface_mask = df['surface'] == surface
                    
                    # Initialize column with NaN values
                    surf_col_name = f'{metric}_{surface}_{window}'
                    df[surf_col_name] = np.nan
                    
                    # Group by player and calculate average for this surface
                    surface_avgs = (
                        df[surface_mask]
                        .groupby('player_id')[metric]
                        .rolling(window, min_periods=1)
                        .mean()
                        .reset_index(level=0, drop=True)
                    )
                    
                    # Update values for this surface
                    df.loc[surface_mask, surf_col_name] = surface_avgs
                    
                    # Forward fill the values
                    df[surf_col_name] = (
                        df
                        .groupby('player_id')[surf_col_name]
                        .transform(lambda x: x.ffill())
                    )
                
                pbar.update(1)
        
        # Return metrics
        for metric in return_metrics:
            # Skip if the metric is not in the dataframe
            if metric not in df.columns:
                continue
                
            for window in windows:
                # Calculate rolling average
                col_name = f'{metric}_{window}'
                df[col_name] = (
                    df.groupby('player_id')[metric]
                    .rolling(window, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )
                
                # Calculate surface-specific rolling average
                for surface in STANDARD_SURFACES:
                    # Create mask for this surface (using lowercase surface)
                    surface_mask = df['surface'] == surface
                    
                    # Initialize column with NaN values
                    surf_col_name = f'{metric}_{surface}_{window}'
                    df[surf_col_name] = np.nan
                    
                    # Group by player and calculate average for this surface
                    surface_avgs = (
                        df[surface_mask]
                        .groupby('player_id')[metric]
                        .rolling(window, min_periods=1)
                        .mean()
                        .reset_index(level=0, drop=True)
                    )
                    
                    # Update values for this surface
                    df.loc[surface_mask, surf_col_name] = surface_avgs
                    
                    # Forward fill the values
                    df[surf_col_name] = (
                        df
                        .groupby('player_id')[surf_col_name]
                        .transform(lambda x: x.ffill())
                    )
                
                pbar.update(1)
    
    return df

def process_match_features_batch(batch_data, player_df, serve_return_df, surface_mapping=None):
    """
    Process a batch of matches to extract features.
    
    Args:
        batch_data: Tuple containing (batch_idx, batch_df)
        player_df: Player-centric dataset with calculated features
        serve_return_df: Player-centric dataset with serve and return stats
        surface_mapping: Optional mapping for standardizing surfaces
        
    Returns:
        List of match features
    """
    try:
        batch_idx, batch_df = batch_data
        
        # Define serve and return metrics we'll use for feature differences
        serve_return_metrics = [
            'serve_efficiency_5',
            'first_serve_pct_5',
            'first_serve_win_pct_5',
            'second_serve_win_pct_5',
            'ace_pct_5',
            'bp_saved_pct_5',
            'return_efficiency_5',
            'bp_conversion_pct_5'
        ]
        
        features = []
        for idx, match in batch_df.iterrows():
            match_date = match['tournament_date']
            winner_id = match['winner_id']
            loser_id = match['loser_id']
            surface = match['surface'].lower()  # Ensure lowercase surface name
            
            # Get player win/loss stats just before this match (exclude the current match)
            winner_prev = player_df[(player_df['player_id'] == winner_id) & 
                                   (player_df['tournament_date'] < match_date)]
            
            loser_prev = player_df[(player_df['player_id'] == loser_id) & 
                                   (player_df['tournament_date'] < match_date)]
            
            # Get player serve/return stats just before this match
            winner_sr_prev = serve_return_df[(serve_return_df['player_id'] == winner_id) & 
                                           (serve_return_df['tournament_date'] < match_date)]
            
            loser_sr_prev = serve_return_df[(serve_return_df['player_id'] == loser_id) & 
                                           (serve_return_df['tournament_date'] < match_date)]
            
            # Initialize match features
            match_features = {
                'match_id': idx,
                'tournament_date': match_date,
                'surface': surface,
                'winner_id': winner_id,
                'loser_id': loser_id,
            }
            
            # Add Elo rating difference
            match_features['elo_diff'] = match['winner_elo'] - match['loser_elo']
            
            # If we have previous win/loss stats for both players
            if not winner_prev.empty and not loser_prev.empty:
                # Get most recent stats
                winner_stats = winner_prev.iloc[-1]
                loser_stats = loser_prev.iloc[-1]
                
                # Add win rate and streak features
                match_features.update({
                    # Win rate differences
                    'win_rate_5_diff': winner_stats.get('win_rate_5', 0) - loser_stats.get('win_rate_5', 0),
                    
                    # Win/loss streak differences
                    'win_streak_diff': winner_stats.get('win_streak', 0) - loser_stats.get('win_streak', 0),
                    'loss_streak_diff': winner_stats.get('loss_streak', 0) - loser_stats.get('loss_streak', 0),
                    
                    # Surface-specific win rates (use lowercase surface)
                    f'win_rate_{surface}_5_diff': 
                        winner_stats.get(f'win_rate_{surface}_5', np.nan) - 
                        loser_stats.get(f'win_rate_{surface}_5', np.nan),
                    
                    # Overall surface win rates
                    f'win_rate_{surface}_overall_diff': 
                        winner_stats.get(f'win_rate_{surface}_overall', np.nan) - 
                        loser_stats.get(f'win_rate_{surface}_overall', np.nan),
                    
                    # Raw player win/loss stats
                    'winner_win_rate_5': winner_stats.get('win_rate_5', 0),
                    'loser_win_rate_5': loser_stats.get('win_rate_5', 0),
                    'winner_win_streak': winner_stats.get('win_streak', 0),
                    'loser_win_streak': loser_stats.get('win_streak', 0),
                    'winner_loss_streak': winner_stats.get('loss_streak', 0),
                    'loser_loss_streak': loser_stats.get('loss_streak', 0),
                })
                
                # Add surface-specific win rates for all surfaces
                for surf in STANDARD_SURFACES:
                    match_features.update({
                        f'winner_win_rate_{surf}_5': winner_stats.get(f'win_rate_{surf}_5', np.nan),
                        f'loser_win_rate_{surf}_5': loser_stats.get(f'win_rate_{surf}_5', np.nan),
                        f'winner_win_rate_{surf}_overall': winner_stats.get(f'win_rate_{surf}_overall', np.nan),
                        f'loser_win_rate_{surf}_overall': loser_stats.get(f'win_rate_{surf}_overall', np.nan),
                    })
            
            # If we have previous serve/return stats for both players
            if not winner_sr_prev.empty and not loser_sr_prev.empty:
                # Get most recent stats
                winner_sr_stats = winner_sr_prev.iloc[-1]
                loser_sr_stats = loser_sr_prev.iloc[-1]
                
                # Add serve and return metric differences
                for metric in serve_return_metrics:
                    if metric in winner_sr_stats and metric in loser_sr_stats:
                        match_features[f'{metric}_diff'] = (
                            winner_sr_stats.get(metric, np.nan) - 
                            loser_sr_stats.get(metric, np.nan)
                        )
                        
                        # Surface-specific version
                        surf_metric = f'{metric}_{surface}'
                        if surf_metric in winner_sr_stats and surf_metric in loser_sr_stats:
                            match_features[f'{surf_metric}_diff'] = (
                                winner_sr_stats.get(surf_metric, np.nan) - 
                                loser_sr_stats.get(surf_metric, np.nan)
                            )
                
                # Add raw player serve/return stats
                for metric in serve_return_metrics:
                    if metric in winner_sr_stats and metric in loser_sr_stats:
                        match_features[f'winner_{metric}'] = winner_sr_stats.get(metric, np.nan)
                        match_features[f'loser_{metric}'] = loser_sr_stats.get(metric, np.nan)
            
            features.append(match_features)
    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        return None
    
    return features

def prepare_features_for_matches(df: pd.DataFrame, player_df: pd.DataFrame, serve_return_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for each match by joining player statistics using multiprocessing.
    
    Args:
        df: Original match dataset
        player_df: Player-centric dataset with calculated features
        serve_return_df: Player-centric dataset with serve and return stats
        
    Returns:
        DataFrame with match features
    """
    logger.info("Preparing match features using multiprocessing...")
    logger.info(f"Using {NUM_CORES} CPU cores")
    
    # Create a copy of the match dataframe
    match_df = df.copy()
    
    # Sort by date to ensure chronological processing
    match_df = match_df.sort_values('tournament_date').reset_index(drop=True)
    
    # Split the data into smaller chunks for parallel processing to reduce memory pressure
    chunk_size = max(1, min(1000, len(match_df) // (NUM_CORES * 4)))  # Smaller chunks for better load balancing and memory usage
    chunks = [(i, match_df.iloc[i:i+chunk_size]) for i in range(0, len(match_df), chunk_size)]
    
    # Create a pool of workers with maxtasksperchild to prevent memory leaks
    all_features = []
    
    try:
        with multiprocessing.Pool(processes=NUM_CORES, maxtasksperchild=2) as pool:
            # Process each chunk in parallel with error handling
            for chunk_result in tqdm(
                pool.imap(
                    partial(process_match_features_batch, player_df=player_df, serve_return_df=serve_return_df),
                    chunks
                ),
                total=len(chunks),
                desc="Preparing match features (parallel)",
                unit="chunk"
            ):
                if chunk_result:  # Check if result is not None
                    all_features.extend(chunk_result)
    except Exception as e:
        logger.error(f"Error in multiprocessing: {str(e)}")
        # Fallback to sequential processing if parallel processing fails
        logger.info("Falling back to sequential processing...")
        all_features = []
        for chunk_data in tqdm(chunks, desc="Processing sequentially"):
            try:
                result = process_match_features_batch(chunk_data, player_df, serve_return_df)
                all_features.extend(result)
            except Exception as e:
                logger.error(f"Error processing chunk: {str(e)}")
                continue
    
    # Create dataframe with all features
    features_df = pd.DataFrame(all_features)
    
    # Sort by date
    features_df = features_df.sort_values('tournament_date').reset_index(drop=True)
    
    return features_df

def process_symmetric_features_batch(batch_data, serve_return_metrics):
    """
    Process a batch of matches to create symmetric features.
    
    Args:
        batch_data: Tuple containing (batch_idx, batch_df)
        serve_return_metrics: List of serve and return metrics to include
        
    Returns:
        List of symmetric match features
    """
    try:
        batch_idx, batch_df = batch_data
        matches = []
        
        # First pass: p1 = winner, p2 = loser (actual match result)
        for idx, row in batch_df.iterrows():
            surface = row['surface'].lower() if isinstance(row['surface'], str) else row['surface']
            match_dict = {
                'match_id': row['match_id'],
                'tournament_date': row['tournament_date'],
                'surface': surface,
                'player1_id': row['winner_id'],
                'player2_id': row['loser_id'],
                'result': 1  # player1 won
            }
            
            # Add traditional features
            # Elo difference
            match_dict['player_elo_diff'] = row.get('elo_diff', 0)
            
            # Win rates
            match_dict['win_rate_5_diff'] = row.get('win_rate_5_diff', 0)
            
            # Streaks
            match_dict['win_streak_diff'] = row.get('win_streak_diff', 0)
            match_dict['loss_streak_diff'] = row.get('loss_streak_diff', 0)
            
            # Surface win rates (preserve NaN values for proper XGBoost handling)
            for surf in STANDARD_SURFACES:
                # Use lowercase for surface-related column names
                if f'win_rate_{surf}_5_diff' in row:
                    match_dict[f'win_rate_{surf}_5_diff'] = row[f'win_rate_{surf}_5_diff']
                
                if f'win_rate_{surf}_overall_diff' in row:
                    match_dict[f'win_rate_{surf}_overall_diff'] = row[f'win_rate_{surf}_overall_diff']
            
            # Add serve and return features
            for metric in serve_return_metrics:
                # General metric
                if f'{metric}_diff' in row:
                    match_dict[f'{metric}_diff'] = row[f'{metric}_diff']
                
                # Surface-specific metric
                if f'{metric}_{surface}_diff' in row:
                    match_dict[f'{metric}_{surface}_diff'] = row[f'{metric}_{surface}_diff']
            
            # Add raw player features to ensure consistent prediction regardless of player order
            if 'winner_win_rate_5' in row:
                match_dict['player1_win_rate_5'] = row['winner_win_rate_5']
                match_dict['player2_win_rate_5'] = row['loser_win_rate_5']
                match_dict['player1_win_streak'] = row['winner_win_streak']
                match_dict['player2_win_streak'] = row['loser_win_streak']
                match_dict['player1_loss_streak'] = row['winner_loss_streak']
                match_dict['player2_loss_streak'] = row['loser_loss_streak']
                
                # Surface-specific rates (use lowercase for surface names)
                for surf in STANDARD_SURFACES:
                    if f'winner_win_rate_{surf}_5' in row:
                        match_dict[f'player1_win_rate_{surf}_5'] = row[f'winner_win_rate_{surf}_5']
                        match_dict[f'player2_win_rate_{surf}_5'] = row[f'loser_win_rate_{surf}_5']
                        match_dict[f'player1_win_rate_{surf}_overall'] = row[f'winner_win_rate_{surf}_overall']
                        match_dict[f'player2_win_rate_{surf}_overall'] = row[f'loser_win_rate_{surf}_overall']
            
            # Add serve and return raw stats
            for metric in serve_return_metrics:
                if f'winner_{metric}' in row and f'loser_{metric}' in row:
                    match_dict[f'player1_{metric}'] = row[f'winner_{metric}']
                    match_dict[f'player2_{metric}'] = row[f'loser_{metric}']
            
            matches.append(match_dict)
        
        # Second pass: p1 = loser, p2 = winner (flipped match result)
        for idx, row in batch_df.iterrows():
            surface = row['surface'].lower() if isinstance(row['surface'], str) else row['surface']
            match_dict = {
                'match_id': row['match_id'],
                'tournament_date': row['tournament_date'],
                'surface': surface,
                'player1_id': row['loser_id'],
                'player2_id': row['winner_id'], 
                'result': 0  # player1 lost
            }
            
            # Add features with reversed signs
            match_dict['player_elo_diff'] = -row.get('elo_diff', 0)
            match_dict['win_rate_5_diff'] = -row.get('win_rate_5_diff', 0)
            match_dict['win_streak_diff'] = -row.get('win_streak_diff', 0)
            match_dict['loss_streak_diff'] = -row.get('loss_streak_diff', 0)
            
            # Surface win rates with reversed signs (use lowercase)
            for surf in STANDARD_SURFACES:
                if f'win_rate_{surf}_5_diff' in row:
                    match_dict[f'win_rate_{surf}_5_diff'] = -row[f'win_rate_{surf}_5_diff']
                
                if f'win_rate_{surf}_overall_diff' in row:
                    match_dict[f'win_rate_{surf}_overall_diff'] = -row[f'win_rate_{surf}_overall_diff']
            
            # Add serve and return features (with reversed signs)
            for metric in serve_return_metrics:
                # General metric
                if f'{metric}_diff' in row:
                    match_dict[f'{metric}_diff'] = -row[f'{metric}_diff']
                
                # Surface-specific metric
                if f'{metric}_{surface}_diff' in row:
                    match_dict[f'{metric}_{surface}_diff'] = -row[f'{metric}_{surface}_diff']
            
            # Add raw player features (swapped)
            if 'winner_win_rate_5' in row:
                match_dict['player1_win_rate_5'] = row['loser_win_rate_5']
                match_dict['player2_win_rate_5'] = row['winner_win_rate_5']
                match_dict['player1_win_streak'] = row['loser_win_streak']
                match_dict['player2_win_streak'] = row['winner_win_streak']
                match_dict['player1_loss_streak'] = row['loser_loss_streak']
                match_dict['player2_loss_streak'] = row['winner_loss_streak']
                
                # Surface-specific rates (use lowercase)
                for surf in STANDARD_SURFACES:
                    if f'winner_win_rate_{surf}_5' in row:
                        match_dict[f'player1_win_rate_{surf}_5'] = row[f'loser_win_rate_{surf}_5']
                        match_dict[f'player2_win_rate_{surf}_5'] = row[f'winner_win_rate_{surf}_5']
                        match_dict[f'player1_win_rate_{surf}_overall'] = row[f'loser_win_rate_{surf}_overall']
                        match_dict[f'player2_win_rate_{surf}_overall'] = row[f'winner_win_rate_{surf}_overall']
            
            # Add serve and return raw stats (swapped)
            for metric in serve_return_metrics:
                if f'winner_{metric}' in row and f'loser_{metric}' in row:
                    match_dict[f'player1_{metric}'] = row[f'loser_{metric}']
                    match_dict[f'player2_{metric}'] = row[f'winner_{metric}']
            
            matches.append(match_dict)
        
        return matches
    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        return None

def generate_player_symmetric_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate player-symmetric features to avoid player position bias using multiprocessing.
    
    Args:
        features_df: DataFrame with calculated features
        
    Returns:
        DataFrame with player-symmetric features
    """
    logger.info("Generating player-symmetric features using multiprocessing...")
    logger.info(f"Using {NUM_CORES} CPU cores")
    
    # Create a copy
    df = features_df.copy()
    
    # Define serve and return metrics to include in symmetric features
    serve_return_metrics = [
        'serve_efficiency_5',
        'first_serve_pct_5',
        'first_serve_win_pct_5',
        'second_serve_win_pct_5',
        'ace_pct_5',
        'bp_saved_pct_5',
        'return_efficiency_5',
        'bp_conversion_pct_5'
    ]
    
    # Split the data into smaller chunks for parallel processing
    chunk_size = max(1, min(1000, len(df) // (NUM_CORES * 4)))  # Smaller chunks for better memory management
    chunks = [(i, df.iloc[i:i+chunk_size]) for i in range(0, len(df), chunk_size)]
    
    all_matches = []
    
    try:
        # Create a pool of workers with maxtasksperchild to prevent memory leaks
        with multiprocessing.Pool(processes=NUM_CORES, maxtasksperchild=2) as pool:
            # Process each chunk in parallel with error handling
            for chunk_result in tqdm(
                pool.imap(
                    partial(process_symmetric_features_batch, serve_return_metrics=serve_return_metrics),
                    chunks
                ),
                total=len(chunks),
                desc="Generating symmetric features (parallel)",
                unit="chunk"
            ):
                if chunk_result:  # Check if result is not None
                    all_matches.extend(chunk_result)
    except Exception as e:
        logger.error(f"Error in multiprocessing: {str(e)}")
        # Fallback to sequential processing if parallel processing fails
        logger.info("Falling back to sequential processing...")
        all_matches = []
        for chunk_data in tqdm(chunks, desc="Processing sequentially"):
            try:
                result = process_symmetric_features_batch(chunk_data, serve_return_metrics)
                all_matches.extend(result)
            except Exception as e:
                logger.error(f"Error processing chunk: {str(e)}")
                continue
    
    # Convert to DataFrame
    symmetric_df = pd.DataFrame(all_matches)
    
    # Sort by date and match_id
    symmetric_df = symmetric_df.sort_values(['tournament_date', 'match_id']).reset_index(drop=True)
    
    return symmetric_df

def save_features_to_db(symmetric_df: pd.DataFrame):
    """
    Save the generated features to the database in batches.
    Only saves features for matches that don't already exist in the database.
    
    Args:
        symmetric_df: DataFrame with player-symmetric features
    """
    logger.info("Saving features to database (with duplicate prevention via ON CONFLICT DO NOTHING)...")
    
    # Get database connection
    conn = get_psycopg2_connection()
    
    try:
        # Create the features table if it doesn't exist
        create_features_table(conn)
        
        # Get existing match_ids from the database
        with conn.cursor() as cur:
            cur.execute("SELECT match_id FROM match_features")
            existing_match_ids = set(row[0] for row in cur.fetchall())
        
        # Filter out matches that already exist in the database
        new_df = symmetric_df[~symmetric_df['match_id'].isin(existing_match_ids)].copy()
        
        if len(new_df) == 0:
            logger.info("No new matches to process. All matches already exist in the database.")
            return
        
        logger.info(f"Found {len(new_df)} new matches to process out of {len(symmetric_df)} total matches")
        
        # Define all possible columns we expect to have (using lowercase for surface names)
        all_possible_columns = [
            'match_id', 'player1_id', 'player2_id', 'surface', 'tournament_date', 'result',
            'player_elo_diff', 'win_rate_5_diff', 'win_streak_diff', 'loss_streak_diff',
            'win_rate_hard_5_diff', 'win_rate_clay_5_diff', 'win_rate_grass_5_diff', 'win_rate_carpet_5_diff',
            'win_rate_hard_overall_diff', 'win_rate_clay_overall_diff', 'win_rate_grass_overall_diff', 'win_rate_carpet_overall_diff',
            'serve_efficiency_5_diff', 'first_serve_pct_5_diff', 'first_serve_win_pct_5_diff', 'second_serve_win_pct_5_diff',
            'ace_pct_5_diff', 'bp_saved_pct_5_diff', 'return_efficiency_5_diff', 'bp_conversion_pct_5_diff',
            'player1_win_rate_5', 'player2_win_rate_5', 'player1_win_streak', 'player2_win_streak',
            'player1_loss_streak', 'player2_loss_streak',
            'player1_win_rate_hard_5', 'player2_win_rate_hard_5',
            'player1_win_rate_clay_5', 'player2_win_rate_clay_5',
            'player1_win_rate_grass_5', 'player2_win_rate_grass_5',
            'player1_win_rate_carpet_5', 'player2_win_rate_carpet_5',
            'player1_win_rate_hard_overall', 'player2_win_rate_hard_overall',
            'player1_win_rate_clay_overall', 'player2_win_rate_clay_overall',
            'player1_win_rate_grass_overall', 'player2_win_rate_grass_overall',
            'player1_win_rate_carpet_overall', 'player2_win_rate_carpet_overall',
            'player1_serve_efficiency_5', 'player2_serve_efficiency_5',
            'player1_first_serve_pct_5', 'player2_first_serve_pct_5',
            'player1_first_serve_win_pct_5', 'player2_first_serve_win_pct_5',
            'player1_second_serve_win_pct_5', 'player2_second_serve_win_pct_5',
            'player1_ace_pct_5', 'player2_ace_pct_5',
            'player1_bp_saved_pct_5', 'player2_bp_saved_pct_5',
            'player1_return_efficiency_5', 'player2_return_efficiency_5',
            'player1_bp_conversion_pct_5', 'player2_bp_conversion_pct_5'
        ]
        
        # Convert any surface-related column names to lowercase
        for col in list(new_df.columns):
            if any(surface.capitalize() in col for surface in STANDARD_SURFACES):
                # Create lowercase version
                lowercase_col = col
                for surface in STANDARD_SURFACES:
                    if surface.capitalize() in col:
                        lowercase_col = col.replace(surface.capitalize(), surface)
                
                # If the column name changed, rename it
                if lowercase_col != col:
                    new_df.rename(columns={col: lowercase_col}, inplace=True)
        
        # Check which columns actually exist in the dataframe
        available_columns = [col for col in all_possible_columns if col in new_df.columns]
        
        # Log columns that are missing
        missing_columns = set(all_possible_columns) - set(available_columns)
        if missing_columns:
            logger.warning(f"The following columns are missing from the dataframe: {sorted(missing_columns)}")
        
        # Convert tournament_date to datetime if it's not already
        new_df['tournament_date'] = pd.to_datetime(new_df['tournament_date'])
        
        # Define required columns and check if they're available
        required_columns = ['match_id', 'player1_id', 'player2_id', 'surface', 'tournament_date', 'result']
        missing_required = set(required_columns) - set(available_columns)
        if missing_required:
            raise ValueError(f"Missing required columns: {missing_required}")
        
        # Define column types and fill values
        integer_columns = {
            'match_id': 0,
            'player1_id': 0,
            'player2_id': 0,
            'result': 0,
            'win_streak_diff': 0,
            'loss_streak_diff': 0,
            'player1_win_streak': 0,
            'player2_win_streak': 0,
            'player1_loss_streak': 0,
            'player2_loss_streak': 0
        }
        
        float_columns = [col for col in available_columns 
                        if col not in integer_columns.keys() 
                        and col not in ['surface', 'tournament_date']]
        
        # Handle integer columns - fill NaN with defaults and convert
        for col, default_value in integer_columns.items():
            if col in available_columns:
                new_df[col] = new_df[col].fillna(default_value).astype('int64')
        
        # Handle float columns - fill NaN with 0 and convert to float64
        for col in float_columns:
            if col in available_columns:
                new_df[col] = new_df[col].fillna(0.0).astype('float64')
        
        # Ensure surface is a string
        if 'surface' in available_columns:
            new_df['surface'] = new_df['surface'].fillna('Unknown')
        
        # Process in batches
        total_rows = len(new_df)
        with conn.cursor() as cur:
            for start_idx in tqdm(range(0, total_rows, DB_BATCH_SIZE), desc="Saving features to database"):
                end_idx = min(start_idx + DB_BATCH_SIZE, total_rows)
                batch_df = new_df.iloc[start_idx:end_idx]
                
                try:
                    # Create SQL query dynamically based on available columns
                    columns_str = ", ".join(available_columns)
                    placeholders = "%s"  # for execute_values
                    
                    # Convert DataFrame to list of tuples
                    values = [tuple(x) for x in batch_df[available_columns].values]
                    
                    # Insert batch using execute_values
                    execute_values(
                        cur,
                        f"""
                        INSERT INTO match_features (
                            {columns_str}
                        ) VALUES %s
                        ON CONFLICT (match_id) DO NOTHING
                        """,
                        values,
                        page_size=DB_PAGE_SIZE
                    )
                    
                    # Commit after each batch
                    conn.commit()
                    
                    logger.info(f"Processed {end_idx}/{total_rows} rows")
                    
                except Exception as e:
                    logger.error(f"Error in batch {start_idx}-{end_idx}: {str(e)}")
                    logger.error(f"First row of problematic batch: {batch_df.iloc[0].to_dict()}")
                    conn.rollback()
                    raise
        
        logger.info(f"Successfully saved {total_rows} new matches to database")
        
    except Exception as e:
        logger.error(f"Error saving features to database: {str(e)}")
        conn.rollback()
        raise
    
    finally:
        conn.close()

def main():
    """Generate features for tennis match prediction."""
    start_time = time.time()
    progress_tracker = ProgressTracker(total_steps=5)
    
    # Step 1: Load data
    logger.info("Step 1/5 (0%): Loading data...")
    df = load_data()
    logger.info(f"Loaded {len(df)} matches")
    
    # TEST MODE: Limit to first 5000 rows
    test_mode = True  # Set to False to process all data
    if test_mode:
        original_len = len(df)
        df = df.head(5000)
        logger.info(f"TEST MODE: Limiting to first 5000 rows out of {original_len}")
    
    progress_tracker.update()
    
    # Step 2: Calculate player win rates and streaks
    logger.info(f"Step 2/5 ({progress_tracker.percent_complete}%): Calculating win rates and streaks...")
    player_df = calculate_win_rates(df)
    logger.info(f"Calculated features for {len(player_df)} player-match combinations")
    progress_tracker.update()
    
    # Step 3: Calculate serve and return stats
    logger.info(f"Step 3/5 ({progress_tracker.percent_complete}%): Calculating serve and return statistics...")
    serve_return_df = calculate_serve_return_stats(df)
    serve_return_df = calculate_serve_return_rolling_stats(serve_return_df)
    logger.info(f"Calculated serve/return stats for {len(serve_return_df)} player-match combinations")
    progress_tracker.update()
    
    # Step 4: Prepare features for matches
    logger.info(f"Step 4/5 ({progress_tracker.percent_complete}%): Preparing match features...")
    features_df = prepare_features_for_matches(df, player_df, serve_return_df)
    logger.info(f"Prepared features for {len(features_df)} matches")
    progress_tracker.update()
    
    # Step 5: Generate player-symmetric features and save to database
    logger.info(f"Step 5/5 ({progress_tracker.percent_complete}%): Generating symmetric features and saving to database...")
    symmetric_df = generate_player_symmetric_features(features_df)
    logger.info(f"Generated {len(symmetric_df)} symmetric match features")
    
    # Save features to database
    save_features_to_db(symmetric_df)
    progress_tracker.update()
    
    # Print feature statistics
    logger.info(f"Total matches processed: {len(df)}")
    logger.info(f"Total features: {len(symmetric_df.columns) - 6}")  # Exclude match_id, tournament_date, player1_id, player2_id, surface, result
    
    # Print example features for a match
    if not symmetric_df.empty:
        example = symmetric_df.iloc[0].to_dict()
        feature_example = {k: v for k, v in example.items() 
                         if k not in ['match_id', 'tournament_date', 'player1_id', 'player2_id', 'surface', 'result']}
        logger.info(f"Example features: {feature_example}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Feature generation completed in {elapsed_time:.2f} seconds")


# Progress tracker class
class ProgressTracker:
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
    
    def update(self):
        self.current_step += 1
        elapsed = time.time() - self.start_time
        if self.current_step < self.total_steps:
            est_remaining = (elapsed / self.current_step) * (self.total_steps - self.current_step)
            logger.info(f"Progress: {self.percent_complete}% complete. Est. remaining time: {est_remaining:.1f}s")
        else:
            logger.info(f"Progress: 100% complete. Total time: {elapsed:.1f}s")
    
    @property
    def percent_complete(self):
        return int((self.current_step / self.total_steps) * 100)


if __name__ == "__main__":
    main() 