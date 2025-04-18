"""
Tennis Match Prediction - Incremental Historical Features Update (v4)

This script is an updated version of generate_features_v3.py that:
1. Only processes new matches that don't have features yet
2. Updates rolling statistics for matches affected by new data
3. Focuses on efficiency by limiting the match data processed

Key differences from v3:
- Identifies new matches not yet in match_features table
- Calculates the window of affected matches (based on players in new matches)
- Only regenerates features for the relevant matches
- Maintains proper is_future=FALSE flag for historical matches
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Set
from datetime import datetime, timedelta
import logging
from tqdm import tqdm
import time
import multiprocessing
from functools import partial
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values

# Add project root to path to allow imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Import functions from v3 script
from predictor.v3.generate_features_v3 import (
    calculate_win_rates,
    calculate_serve_return_stats,
    calculate_serve_return_rolling_stats,
    process_match_features_batch,
    prepare_features_for_matches,
    process_symmetric_features_batch,
    generate_player_symmetric_features,
    get_database_connection,
    get_psycopg2_connection,
    create_features_table,
    STANDARD_SURFACES
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{project_root}/predictor/v4/output/historical_features.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
DB_BATCH_SIZE = 10000
DB_PAGE_SIZE = 1000
MAX_LOOKBACK_DAYS = 365  # Increased max days to look back for affected matches
MAX_LOOKBACK_MATCHES = 20  # Ensure we have at least this many previous matches per player
ROLLING_WINDOWS = [5, 10]  # Match lookback windows used in feature generation

class HistoricalFeatureUpdater:
    def __init__(self):
        """Initialize the historical feature updater"""
        load_dotenv()
        
        # Database settings
        self.db_url = os.getenv("DATABASE_URL")
        if self.db_url and self.db_url.startswith('postgres://'):
            self.db_url = self.db_url.replace('postgres://', 'postgresql://', 1)
        
        # Batch processing settings
        self.batch_size = 1000  # Process this many new matches at a time
        self.max_batches = None  # Set to a number to limit batches, None for all
        
        logger.info("HistoricalFeatureUpdater initialized")
        print("HistoricalFeatureUpdater initialized")
    
    def get_total_unprocessed_matches(self) -> int:
        """
        Get the total count of unprocessed matches
        
        Returns:
            Count of matches without features
        """
        engine = get_database_connection()
        query = """
        SELECT COUNT(*)
        FROM matches m
        LEFT JOIN match_features f ON m.id = f.match_id
        WHERE f.id IS NULL
        AND m.winner_id IS NOT NULL 
        AND m.loser_id IS NOT NULL
        AND m.tournament_date IS NOT NULL
        """
        
        with engine.connect() as conn:
            result = conn.execute(text(query)).fetchone()
            return result[0] if result else 0
    
    def identify_new_matches(self, offset: int = 0) -> pd.DataFrame:
        """
        Identify matches that don't have features yet in match_features table
        
        Args:
            offset: Number of matches to skip (for batch processing)
            
        Returns:
            DataFrame with new matches that need features
        """
        print(f"Identifying new matches that need features (batch offset: {offset})...")
        
        # Connect to database
        engine = get_database_connection()
        
        # Get matches that aren't in match_features table
        query = f"""
        SELECT 
            m.id as match_id,
            m.tournament_date,
            m.tournament_id,
            m.tournament_name,
            m.surface,
            m.tournament_level,
            m.winner_id,
            m.winner_name,
            m.winner_hand,
            m.winner_height_cm,
            m.winner_country_code,
            m.winner_age,
            m.loser_id,
            m.loser_name,
            m.loser_hand,
            m.loser_height_cm,
            m.loser_country_code,
            m.loser_age,
            m.winner_aces,
            m.winner_double_faults,
            m.winner_serve_points,
            m.winner_first_serves_in,
            m.winner_first_serve_points_won,
            m.winner_second_serve_points_won,
            m.winner_service_games,
            m.winner_break_points_saved,
            m.winner_break_points_faced,
            m.loser_aces,
            m.loser_double_faults,
            m.loser_serve_points,
            m.loser_first_serves_in,
            m.loser_first_serve_points_won,
            m.loser_second_serve_points_won,
            m.loser_service_games,
            m.loser_break_points_saved,
            m.loser_break_points_faced,
            m.winner_elo,
            m.loser_elo
        FROM matches m
        LEFT JOIN match_features f ON m.id = f.match_id
        WHERE f.id IS NULL
        AND m.winner_id IS NOT NULL 
        AND m.loser_id IS NOT NULL
        AND m.tournament_date IS NOT NULL
        ORDER BY m.tournament_date DESC
        LIMIT {self.batch_size} OFFSET {offset}
        """
        
        df = pd.read_sql(query, engine)
        
        # Convert date columns to datetime
        df['tournament_date'] = pd.to_datetime(df['tournament_date'])
        
        # Make sure surface is lowercase
        df['surface'] = df['surface'].str.lower()
        
        print(f"Found {len(df)} new matches in this batch")
        return df
    
    def get_affected_players(self, new_matches_df: pd.DataFrame) -> Set[int]:
        """
        Identify players affected by new matches
        
        Args:
            new_matches_df: DataFrame with new matches
            
        Returns:
            Set of player IDs that are affected
        """
        affected_players = set()
        
        # Add both winners and losers from new matches
        affected_players.update(new_matches_df['winner_id'].unique())
        affected_players.update(new_matches_df['loser_id'].unique())
        
        print(f"Identified {len(affected_players)} players affected by new matches")
        return affected_players
    
    def load_affected_historical_matches(self, affected_players: Set[int], new_matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Load historical matches involving affected players, ensuring we have enough history
        for rolling window calculations.
        
        Args:
            affected_players: Set of player IDs affected by new matches
            new_matches_df: DataFrame with new matches
            
        Returns:
            DataFrame with historical matches involving affected players
        """
        print("Loading historical matches for affected players...")
        
        # Find earliest date in new matches to determine lookback period
        earliest_new_match_date = new_matches_df['tournament_date'].min()
        cutoff_date = earliest_new_match_date - timedelta(days=MAX_LOOKBACK_DAYS)
        
        # Format player IDs for SQL query
        player_ids_str = ','.join([str(p) for p in affected_players])
        
        # Connect to database
        engine = get_database_connection()
        
        # First query: Get matches within the time window
        time_window_query = f"""
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
            loser_elo
        FROM matches
        WHERE (winner_id IN ({player_ids_str}) OR loser_id IN ({player_ids_str}))
        AND tournament_date >= '{cutoff_date.strftime('%Y-%m-%d')}'
        AND tournament_date <= '{earliest_new_match_date.strftime('%Y-%m-%d')}'
        AND winner_id IS NOT NULL 
        AND loser_id IS NOT NULL
        ORDER BY tournament_date ASC
        """
        
        time_window_df = pd.read_sql(time_window_query, engine)
        
        # Check if we have enough matches per player
        match_counts = {}
        for player_id in affected_players:
            # Count matches where player is winner or loser
            winner_matches = len(time_window_df[time_window_df['winner_id'] == player_id])
            loser_matches = len(time_window_df[time_window_df['loser_id'] == player_id])
            match_counts[player_id] = winner_matches + loser_matches
        
        # Identify players needing more historical matches
        players_needing_more_matches = {
            player_id: MAX_LOOKBACK_MATCHES - count 
            for player_id, count in match_counts.items() 
            if count < MAX_LOOKBACK_MATCHES
        }
        
        # If we need more matches for any players
        if players_needing_more_matches:
            print(f"Fetching additional historical matches for {len(players_needing_more_matches)} players with insufficient history")
            
            # Get additional matches for these players
            extra_matches_list = []
            
            for player_id, matches_needed in players_needing_more_matches.items():
                # Query for additional matches for this player
                player_query = f"""
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
                    loser_elo
                FROM matches
                WHERE (winner_id = {player_id} OR loser_id = {player_id})
                AND tournament_date < '{cutoff_date.strftime('%Y-%m-%d')}'
                AND winner_id IS NOT NULL 
                AND loser_id IS NOT NULL
                ORDER BY tournament_date DESC
                LIMIT {matches_needed + 5}  -- Add a buffer
                """
                
                player_extra_df = pd.read_sql(player_query, engine)
                if not player_extra_df.empty:
                    extra_matches_list.append(player_extra_df)
            
            # Only concat if we have matches to add and ensure all DataFrames have same columns
            if extra_matches_list:
                # First, ensure all DataFrames have the same columns
                all_columns = set()
                for df in extra_matches_list:
                    all_columns.update(df.columns)
                
                # Add any missing columns with NaN values
                for i, df in enumerate(extra_matches_list):
                    for col in all_columns:
                        if col not in df.columns:
                            extra_matches_list[i][col] = np.nan
                
                extra_matches_df = pd.concat(extra_matches_list, axis=0, ignore_index=True)
                
                # Combine with time window matches, ensuring column consistency
                for col in extra_matches_df.columns:
                    if col not in time_window_df.columns:
                        time_window_df[col] = np.nan
                
                for col in time_window_df.columns:
                    if col not in extra_matches_df.columns:
                        extra_matches_df[col] = np.nan
                
                historical_df = pd.concat([time_window_df, extra_matches_df], axis=0, ignore_index=True)
            else:
                historical_df = time_window_df
        else:
            historical_df = time_window_df
        
        # Convert date columns to datetime
        historical_df['tournament_date'] = pd.to_datetime(historical_df['tournament_date'])
        
        # Make sure surface is lowercase
        historical_df['surface'] = historical_df['surface'].str.lower()
        
        # Combine with new matches to ensure we have the complete history
        # First, ensure column consistency between historical_df and new_matches_df
        all_columns = set(historical_df.columns) | set(new_matches_df.columns)
        
        for col in all_columns:
            if col not in historical_df.columns:
                historical_df[col] = np.nan
            if col not in new_matches_df.columns:
                new_matches_df[col] = np.nan
        
        combined_df = pd.concat([historical_df, new_matches_df], axis=0, ignore_index=True)
        
        # Remove duplicates (in case some matches were both in historical and new query)
        combined_df = combined_df.drop_duplicates(subset=['match_id'])
        
        # Sort by date to ensure proper chronological order
        combined_df = combined_df.sort_values('tournament_date').reset_index(drop=True)
        
        # Print some statistics
        player_match_counts = {}
        for player_id in affected_players:
            winner_matches = len(combined_df[combined_df['winner_id'] == player_id])
            loser_matches = len(combined_df[combined_df['loser_id'] == player_id])
            player_match_counts[player_id] = winner_matches + loser_matches
        
        min_matches = min(player_match_counts.values()) if player_match_counts else 0
        max_matches = max(player_match_counts.values()) if player_match_counts else 0
        avg_matches = sum(player_match_counts.values()) / len(player_match_counts) if player_match_counts else 0
        
        print(f"Loaded {len(combined_df)} total matches for feature calculation")
        print(f"Match history per player - Min: {min_matches}, Max: {max_matches}, Avg: {avg_matches:.1f}")
        
        return combined_df
    
    def save_features_to_db(self, symmetric_df: pd.DataFrame):
        """
        Save the generated features to the database in batches.
        
        Args:
            symmetric_df: DataFrame with player-symmetric features
        """
        logger.info("Saving features to database...")
        
        # Get database connection
        conn = get_psycopg2_connection()
        
        try:
            # Create the features table if it doesn't exist
            create_features_table(conn)
            
            # Get existing match_ids from the database
            with conn.cursor() as cur:
                cur.execute("SELECT match_id FROM match_features")
                existing_match_ids = set(row[0] for row in cur.fetchall())
            
            # Prepare dataframe and ensure surface names are lowercase
            df_to_save = symmetric_df.copy()
            
            # Remove invalid match_ids (0 or negative)
            invalid_ids = df_to_save[df_to_save['match_id'] <= 0].shape[0]
            if invalid_ids > 0:
                logger.warning(f"Removing {invalid_ids} records with invalid match_id (≤ 0)")
                df_to_save = df_to_save[df_to_save['match_id'] > 0]
            
            # Convert any surface-related column names to lowercase
            for col in list(df_to_save.columns):
                if any(surface.capitalize() in col for surface in STANDARD_SURFACES):
                    # Create lowercase version
                    lowercase_col = col
                    for surface in STANDARD_SURFACES:
                        if surface.capitalize() in col:
                            lowercase_col = col.replace(surface.capitalize(), surface)
                    
                    # If the column name changed, rename it
                    if lowercase_col != col:
                        df_to_save.rename(columns={col: lowercase_col}, inplace=True)
            
            # Handle data type issues - convert values to appropriate bounds for PostgreSQL
            # Get schema to understand column data types
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'match_features'
                """)
                column_types = {row[0]: row[1] for row in cur.fetchall()}
            
            # Log column types for debugging
            logger.info("Column types in match_features table:")
            for col, col_type in column_types.items():
                logger.info(f"  {col}: {col_type}")
            
            # Check for extremely large values before processing
            for col in df_to_save.columns:
                if df_to_save[col].dtype in [np.int64, np.float64]:
                    try:
                        # Skip NaN values when calculating min/max
                        non_nan_values = df_to_save[~df_to_save[col].isna()][col]
                        if not non_nan_values.empty:
                            max_val = non_nan_values.max()
                            min_val = non_nan_values.min()
                            
                            if abs(max_val) > 1e9 or abs(min_val) > 1e9:
                                logger.warning(f"Column {col} has extreme values: min={min_val}, max={max_val}")
                                
                                # Find rows with extreme values
                                extreme_rows = df_to_save[
                                    (~df_to_save[col].isna()) & 
                                    ((df_to_save[col] > 1e9) | (df_to_save[col] < -1e9))
                                ]
                                if not extreme_rows.empty:
                                    logger.warning(f"Sample of extreme values in column {col}:")
                                    for idx, row in extreme_rows.head(5).iterrows():
                                        logger.warning(f"  match_id={row['match_id']}, {col}={row[col]}")
                    except Exception as e:
                        logger.error(f"Error checking column {col}: {str(e)}")
            
            # Process each column based on its type
            for col in df_to_save.columns:
                if col in column_types:
                    col_type = column_types[col].upper()
                    
                    # Handle BIGINT columns - PostgreSQL bigint range is -2^63 to 2^63-1
                    if 'BIGINT' in col_type and col in df_to_save:
                        max_bigint = 9223372036854775807
                        min_bigint = -9223372036854775808
                        
                        # Only check non-NaN values
                        mask = ~df_to_save[col].isna()
                        if mask.any():
                            # Check for out-of-range values before replacing
                            out_of_range = df_to_save[
                                mask & 
                                ((df_to_save[col] > max_bigint) | (df_to_save[col] < min_bigint))
                            ]
                            
                            if not out_of_range.empty:
                                logger.warning(f"Found {len(out_of_range)} out-of-range values in BIGINT column {col}")
                                for idx, row in out_of_range.head(5).iterrows():
                                    logger.warning(f"  match_id={row['match_id']}, {col}={row[col]}")
                            
                            # Replace out-of-range values with boundary values
                            df_to_save.loc[mask & (df_to_save[col] > max_bigint), col] = max_bigint
                            df_to_save.loc[mask & (df_to_save[col] < min_bigint), col] = min_bigint
                    
                    # Handle INTEGER columns
                    elif 'INTEGER' in col_type and col in df_to_save:
                        max_int = 2147483647
                        min_int = -2147483648
                        
                        # Only check non-NaN values
                        mask = ~df_to_save[col].isna()
                        if mask.any():
                            # Check for out-of-range values before replacing
                            out_of_range = df_to_save[
                                mask & 
                                ((df_to_save[col] > max_int) | (df_to_save[col] < min_int))
                            ]
                            
                            if not out_of_range.empty:
                                logger.warning(f"Found {len(out_of_range)} out-of-range values in INTEGER column {col}")
                                for idx, row in out_of_range.head(5).iterrows():
                                    logger.warning(f"  match_id={row['match_id']}, {col}={row[col]}")
                            
                            # Replace out-of-range values with boundary values
                            df_to_save.loc[mask & (df_to_save[col] > max_int), col] = max_int
                            df_to_save.loc[mask & (df_to_save[col] < min_int), col] = min_int
                    
                    # Ensure strings are within appropriate length
                    elif 'VARCHAR' in col_type and col in df_to_save:
                        # Extract max length from type like VARCHAR(255)
                        max_length = int(col_type.split('(')[1].split(')')[0]) if '(' in col_type else 255
                        
                        # Only process non-NaN values
                        mask = ~df_to_save[col].isna()
                        if mask.any():
                            # Truncate strings that are too long
                            df_to_save.loc[mask, col] = df_to_save.loc[mask, col].apply(
                                lambda x: str(x)[:max_length]
                            )
            
            # Check for specific large values in any column
            for col in df_to_save.columns:
                if col != 'match_id' and df_to_save[col].dtype in [np.int64, np.float64]:
                    # Don't replace NaN values with defaults
                    non_nan_values = df_to_save[~df_to_save[col].isna()][col]
                    
                    if not non_nan_values.empty:
                        max_val = non_nan_values.max()
                        min_val = non_nan_values.min()
                        
                        if abs(max_val) > 1e9 or abs(min_val) > 1e9:
                            logger.warning(f"Column {col} has extreme values: min={min_val}, max={max_val}. Capping values.")
                            # Only clip non-NaN values, preserve NaN
                            mask = ~df_to_save[col].isna()
                            df_to_save.loc[mask, col] = df_to_save.loc[mask, col].clip(-1e9, 1e9)
            
            # Split dataframe into new matches and updates
            new_matches_df = df_to_save[~df_to_save['match_id'].isin(existing_match_ids)].copy()
            update_matches_df = df_to_save[df_to_save['match_id'].isin(existing_match_ids)].copy()
            
            # Set is_future flag for historical matches
            if 'is_future' not in new_matches_df.columns:
                new_matches_df['is_future'] = False
            
            # Process new matches first
            if len(new_matches_df) > 0:
                logger.info(f"Inserting {len(new_matches_df)} new matches")
                
                # Define all columns in the dataframe
                available_columns = list(new_matches_df.columns)
                
                # Convert tournament_date to datetime if needed
                new_matches_df['tournament_date'] = pd.to_datetime(new_matches_df['tournament_date'])
                
                # Process in batches
                total_rows = len(new_matches_df)
                with conn.cursor() as cur:
                    for start_idx in tqdm(range(0, total_rows, DB_BATCH_SIZE), desc="Saving new matches"):
                        end_idx = min(start_idx + DB_BATCH_SIZE, total_rows)
                        batch_df = new_matches_df.iloc[start_idx:end_idx]
                        
                        try:
                            # Create SQL query dynamically based on available columns
                            columns_str = ", ".join(available_columns)
                            
                            # Debug first row data
                            if start_idx == 0:
                                first_row = batch_df.iloc[0]
                                logger.info(f"Sample data for first row - match_id: {first_row['match_id']} (type: {type(first_row['match_id'])})")
                            
                            # Convert DataFrame to list of tuples, replacing NaN with None (SQL NULL)
                            values = []
                            for _, row in batch_df.iterrows():
                                row_values = []
                                for col in available_columns:
                                    val = row[col]
                                    # Convert NaN to None (SQL NULL) for database insertion
                                    if isinstance(val, (float, int)) and np.isnan(val):
                                        row_values.append(None)
                                    # Convert extremely large values to reasonable limits
                                    elif isinstance(val, (float, int)) and abs(val) > 1e15:
                                        if val > 0:
                                            row_values.append(1e9)
                                        else:
                                            row_values.append(-1e9)
                                    else:
                                        row_values.append(val)
                                values.append(tuple(row_values))
                            
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
                            
                        except Exception as e:
                            logger.error(f"Error in batch {start_idx}-{end_idx}: {str(e)}")
                            # Debug the first problematic row
                            if len(batch_df) > 0:
                                prob_row = batch_df.iloc[0]
                                logger.error(f"First row in failed batch - match_id: {prob_row['match_id']}, type: {type(prob_row['match_id'])}")
                                for col in ['win_streak_diff', 'loss_streak_diff', 'player1_win_streak', 'player2_win_streak']:
                                    if col in prob_row:
                                        logger.error(f"  {col}: {prob_row[col]} (type: {type(prob_row[col])})")
                            conn.rollback()
                            raise
                
                logger.info(f"Successfully inserted {total_rows} new matches")
            
            # Now process updates (existing matches with new features)
            if len(update_matches_df) > 0:
                logger.info(f"Updating features for {len(update_matches_df)} existing matches")
                
                # Define all columns in the dataframe
                update_columns = [col for col in update_matches_df.columns 
                               if col not in ['id', 'match_id', 'created_at', 'updated_at']]
                
                # Process in batches
                total_rows = len(update_matches_df)
                with conn.cursor() as cur:
                    for start_idx in tqdm(range(0, total_rows, DB_BATCH_SIZE), desc="Updating existing matches"):
                        end_idx = min(start_idx + DB_BATCH_SIZE, total_rows)
                        batch_df = update_matches_df.iloc[start_idx:end_idx]
                        
                        # Debug first row in batch
                        if start_idx == 0 and len(batch_df) > 0:
                            first_row = batch_df.iloc[0]
                            logger.info(f"First update row - match_id: {first_row['match_id']} (type: {type(first_row['match_id'])})")
                            
                            # Log first row values for debugging
                            logger.info("First row values for debugging:")
                            for col in update_columns:
                                val = first_row[col]
                                logger.info(f"  {col}: {val} (type: {type(val)})")
                        
                        # Use a different approach for UPDATE queries
                        # Build a single UPDATE query with multiple match_ids for better performance
                        try:
                            # Group by match parameters for bulk updates
                            processed_matches = 0
                            
                            # Process each row individually
                            for i, row in batch_df.iterrows():
                                # Skip any match_id that is 0 or negative
                                if row['match_id'] <= 0:
                                    logger.warning(f"Skipping update for invalid match_id: {row['match_id']}")
                                    continue
                                
                                # Create SET clause for SQL UPDATE
                                set_clauses = []
                                params = []
                                
                                for col in update_columns:
                                    # Get the value and properly handle NaN and large values
                                    val = row[col]
                                    
                                    # Convert NaN to None (SQL NULL)
                                    if isinstance(val, (float, int)) and np.isnan(val):
                                        val = None
                                    # Limit extremely large values
                                    elif isinstance(val, (float, int)) and abs(val) > 1e15:
                                        if val > 0:
                                            val = 1e9
                                        else:
                                            val = -1e9
                                    
                                    set_clauses.append(f"{col} = %s")
                                    params.append(val)
                                
                                # Add match_id parameter
                                params.append(row['match_id'])
                                
                                # Create and execute UPDATE query
                                update_query = f"""
                                UPDATE match_features
                                SET {', '.join(set_clauses)},
                                    updated_at = CURRENT_TIMESTAMP
                                WHERE match_id = %s
                                """
                                
                                try:
                                    cur.execute(update_query, params)
                                    processed_matches += 1
                                except Exception as e:
                                    logger.error(f"Error updating match {row['match_id']}: {str(e)}")
                                    # Log specific values for debugging
                                    for col, val in zip(update_columns, params[:-1]):
                                        logger.error(f"  {col}: {val} (type: {type(val)})")
                                    raise
                            
                            # Commit after processing all rows in this batch
                            conn.commit()
                            logger.info(f"Successfully updated {processed_matches} matches in batch {start_idx}-{end_idx}")
                            
                        except Exception as e:
                            logger.error(f"Error in batch {start_idx}-{end_idx}: {str(e)}")
                            conn.rollback()
            
            logger.info(f"Successfully updated {total_rows} existing matches")
            
        except Exception as e:
            logger.error(f"Error saving features to database: {str(e)}")
            conn.rollback()
            raise
        
        finally:
            conn.close()
    
    def update_features(self):
        """Main method to update features for new and affected matches"""
        start_time = time.time()
        
        print("\n" + "="*50)
        print("STARTING INCREMENTAL FEATURE UPDATE")
        print("="*50 + "\n")
        
        try:
            # Count total unprocessed matches
            total_matches = self.get_total_unprocessed_matches()
            print(f"Found {total_matches} total unprocessed matches")
            
            if total_matches == 0:
                print("No new matches to process. Exiting.")
                return
            
            # Calculate number of batches
            num_batches = (total_matches + self.batch_size - 1) // self.batch_size
            if self.max_batches is not None:
                num_batches = min(num_batches, self.max_batches)
                print(f"Will process {num_batches} batches of up to {self.batch_size} matches each (max: {self.max_batches} batches)")
            else:
                print(f"Will process {num_batches} batches of up to {self.batch_size} matches each")
            
            # Process in batches
            for batch_idx in range(num_batches):
                batch_start_time = time.time()
                offset = batch_idx * self.batch_size
                
                print("\n" + "="*50)
                print(f"PROCESSING BATCH {batch_idx+1}/{num_batches} (OFFSET: {offset})")
                print("="*50 + "\n")
                
                # Step 1: Identify matches that need features for this batch
                print("\n" + "-"*50)
                print(f"STEP 1: IDENTIFYING MATCHES NEEDING FEATURES (BATCH {batch_idx+1})")
                print("-"*50)
                
                new_matches_df = self.identify_new_matches(offset)
                
                if new_matches_df.empty:
                    print(f"No more matches to process in batch {batch_idx+1}. Moving to next batch.")
                    continue
                
                # Validate match_id values
                invalid_match_ids = new_matches_df[
                    (new_matches_df['match_id'].isna()) | 
                    (new_matches_df['match_id'] <= 0)
                ]
                
                if not invalid_match_ids.empty:
                    print(f"WARNING: Found {len(invalid_match_ids)} matches with invalid match_id. These will be excluded.")
                    # Filter out invalid match_ids
                    new_matches_df = new_matches_df[
                        (~new_matches_df['match_id'].isna()) & 
                        (new_matches_df['match_id'] > 0)
                    ]
                
                # Check if we still have matches to process after filtering
                if new_matches_df.empty:
                    print(f"No valid matches left to process in batch {batch_idx+1} after filtering. Moving to next batch.")
                    continue
                
                # Get players affected by new matches
                affected_players = self.get_affected_players(new_matches_df)
                
                # Get all matches involving affected players
                all_affected_matches_df = self.load_affected_historical_matches(affected_players, new_matches_df)
                
                # Ensure match_id is valid in the dataset
                invalid_match_ids = all_affected_matches_df[
                    (all_affected_matches_df['match_id'].isna()) | 
                    (all_affected_matches_df['match_id'] <= 0)
                ]
                
                if not invalid_match_ids.empty:
                    print(f"WARNING: Found {len(invalid_match_ids)} matches with invalid match_id in historical data. These will be excluded.")
                    # Filter out invalid match_ids
                    all_affected_matches_df = all_affected_matches_df[
                        (~all_affected_matches_df['match_id'].isna()) & 
                        (all_affected_matches_df['match_id'] > 0)
                    ]
                
                # Step 2: Calculate player-centric features
                print("\n" + "-"*50)
                print(f"STEP 2: CALCULATING PLAYER FEATURES (BATCH {batch_idx+1})")
                print("-"*50)
                
                player_df = calculate_win_rates(all_affected_matches_df)
                print(f"Calculated win rates for {len(player_df['player_id'].unique())} players")
                
                serve_return_df = calculate_serve_return_stats(all_affected_matches_df)
                serve_return_df = calculate_serve_return_rolling_stats(serve_return_df)
                print(f"Calculated serve/return stats for {len(serve_return_df['player_id'].unique())} players")
                
                # Step 3: Generate match features
                print("\n" + "-"*50)
                print(f"STEP 3: GENERATING MATCH FEATURES (BATCH {batch_idx+1})")
                print("-"*50)
                
                features_df = prepare_features_for_matches(all_affected_matches_df, player_df, serve_return_df)
                print(f"Generated features for {len(features_df)} matches")
                
                # Validate match_id in features
                if 'match_id' in features_df.columns:
                    invalid_features = features_df[
                        (features_df['match_id'].isna()) | 
                        (features_df['match_id'] <= 0)
                    ]
                    
                    if not invalid_features.empty:
                        print(f"WARNING: Found {len(invalid_features)} generated features with invalid match_id. These will be excluded.")
                        # Filter out invalid match_ids
                        features_df = features_df[
                            (~features_df['match_id'].isna()) & 
                            (features_df['match_id'] > 0)
                        ]
                
                # Step 4: Generate player-symmetric features
                print("\n" + "-"*50)
                print(f"STEP 4: GENERATING SYMMETRIC FEATURES (BATCH {batch_idx+1})")
                print("-"*50)
                
                symmetric_df = generate_player_symmetric_features(features_df)
                print(f"Generated symmetric features for {len(symmetric_df)} matches")
                
                # Final validation of symmetric features
                if 'match_id' in symmetric_df.columns:
                    invalid_symmetric = symmetric_df[
                        (symmetric_df['match_id'].isna()) | 
                        (symmetric_df['match_id'] <= 0)
                    ]
                    
                    if not invalid_symmetric.empty:
                        print(f"WARNING: Found {len(invalid_symmetric)} symmetric features with invalid match_id. These will be excluded.")
                        # Filter out invalid match_ids
                        symmetric_df = symmetric_df[
                            (~symmetric_df['match_id'].isna()) & 
                            (symmetric_df['match_id'] > 0)
                        ]
                
                # Step 5: Save to database
                print("\n" + "-"*50)
                print(f"STEP 5: SAVING FEATURES TO DATABASE (BATCH {batch_idx+1})")
                print("-"*50)
                
                self.save_features_to_db(symmetric_df)
                
                # Batch complete
                batch_elapsed_time = time.time() - batch_start_time
                print(f"\nBatch {batch_idx+1} completed in {batch_elapsed_time:.2f} seconds")
                
                # Clear memory after each batch
                del new_matches_df, all_affected_matches_df, player_df, serve_return_df, features_df, symmetric_df
                
                # Force garbage collection
                import gc
                gc.collect()
            
            # Done!
            elapsed_time = time.time() - start_time
            print("\n" + "="*50)
            print(f"ALL BATCHES COMPLETED IN {elapsed_time:.2f} SECONDS")
            print("="*50)
            
        except Exception as e:
            logger.error(f"Error updating features: {e}")
            print(f"\nERROR: Feature update failed: {e}")
            raise

def main():
    """Main execution function"""
    try:
        print("\nStarting incremental historical feature update...")
        updater = HistoricalFeatureUpdater()
        updater.update_features()
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        print(f"\nCRITICAL ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 