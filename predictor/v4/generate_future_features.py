"""
Tennis Match Prediction - Future Match Feature Generation (v4)

This script generates features for upcoming tennis matches by:
1. Loading unprocessed matches from scheduled_matches table
2. Calculating features using historical data from matches table
3. Storing features in match_features table with is_future flag

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
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values
import json

# Add project root to path to allow imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from predictor.v3.generate_features_v3 import (
    calculate_win_rates,
    calculate_serve_return_stats,
    calculate_serve_return_rolling_stats
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{project_root}/predictor/v4/output/feature_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FutureFeatureGenerator:
    def __init__(self):
        """Initialize the feature generator"""
        load_dotenv()
        
        # Database settings
        self.db_url = os.getenv("DATABASE_URL")
        if self.db_url and self.db_url.startswith('postgres://'):
            self.db_url = self.db_url.replace('postgres://', 'postgresql://', 1)
        
        logger.info("FutureFeatureGenerator initialized")
        print("FutureFeatureGenerator initialized")
    
    def get_db_connection(self):
        """Create a database connection"""
        print("Connecting to database...")
        return psycopg2.connect(self.db_url)
    
    def load_historical_matches(self) -> pd.DataFrame:
        """
        Load historical matches from the database
        
        Returns:
            DataFrame with historical match data
        """
        print("Loading historical matches from database...")
        query = """
            SELECT 
                id as match_id,  -- Note that for historical matches, match_id is the auto-increment PK from matches table
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
            WHERE winner_id IS NOT NULL 
            AND loser_id IS NOT NULL
            ORDER BY tournament_date ASC
        """
        
        with self.get_db_connection() as conn:
            df = pd.read_sql(query, conn)
        
        logger.info(f"Loaded {len(df)} historical matches")
        print(f"Loaded {len(df)} historical matches")
        
        # Print first few rows for debugging
        print("\nSample of historical data:")
        print(df.head(3).to_string())
        return df
    
    def load_scheduled_matches(self) -> pd.DataFrame:
        """
        Load unprocessed scheduled matches
        
        Returns:
            DataFrame with scheduled match data
        """
        print("Loading unprocessed scheduled matches...")
        query = """
            SELECT *
            FROM scheduled_matches
            WHERE is_processed = FALSE
            AND scheduled_date >= CURRENT_DATE
            AND scheduled_date <= CURRENT_DATE + INTERVAL '7 days'
            ORDER BY scheduled_date ASC
        """
        
        with self.get_db_connection() as conn:
            df = pd.read_sql(query, conn)
        
        logger.info(f"Loaded {len(df)} unprocessed scheduled matches")
        print(f"Loaded {len(df)} unprocessed scheduled matches")
        
        # Print sample of scheduled matches
        if not df.empty:
            print("\nSample of scheduled matches:")
            print(df.head(3).to_string())
        return df
    
    def prepare_player_features(self, historical_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate player-specific features from historical data
        
        Args:
            historical_df: DataFrame with historical matches
            
        Returns:
            Tuple of (player win rates DataFrame, player serve/return stats DataFrame)
        """
        print("Calculating player win rates and streaks...")
        # Calculate win rates and streaks
        player_stats_df = calculate_win_rates(historical_df)
        print(f"Generated win rate stats for {len(player_stats_df['player_id'].unique())} players")
        
        print("Calculating serve and return statistics...")
        # Calculate serve and return stats
        serve_return_df = calculate_serve_return_stats(historical_df)
        serve_return_df = calculate_serve_return_rolling_stats(serve_return_df)
        print(f"Generated serve/return stats for {len(serve_return_df['player_id'].unique())} players")
        
        # Print sample of player stats
        print("\nSample of player win rate stats:")
        print(player_stats_df.head(2).to_string())
        
        print("\nSample of serve/return stats:")
        print(serve_return_df.head(2).to_string())
        
        return player_stats_df, serve_return_df
    
    def generate_match_features(
        self,
        scheduled_match: pd.Series,
        player_stats_df: pd.DataFrame,
        serve_return_df: pd.DataFrame,
        match_date: datetime
    ) -> Dict[str, Any]:
        """
        Generate features for a scheduled match
        
        Args:
            scheduled_match: Series containing scheduled match data
            player_stats_df: DataFrame with player statistics
            serve_return_df: DataFrame with serve/return statistics
            match_date: Date of the scheduled match
            
        Returns:
            Dictionary of calculated features
        """
        # Get player IDs
        player1_id = scheduled_match['player1_id']
        player2_id = scheduled_match['player2_id']
        
        print(f"\nGenerating features for match: {player1_id} vs {player2_id}")
        
        # Get player stats just before match date
        player1_stats = player_stats_df[
            (player_stats_df['player_id'] == player1_id) &
            (player_stats_df['tournament_date'] < match_date)
        ].iloc[-1] if len(player_stats_df[player_stats_df['player_id'] == player1_id]) > 0 else None
        
        player2_stats = player_stats_df[
            (player_stats_df['player_id'] == player2_id) &
            (player_stats_df['tournament_date'] < match_date)
        ].iloc[-1] if len(player_stats_df[player_stats_df['player_id'] == player2_id]) > 0 else None
        
        if player1_stats is None:
            print(f"WARNING: No historical stats found for player1 (ID: {player1_id})")
        if player2_stats is None:
            print(f"WARNING: No historical stats found for player2 (ID: {player2_id})")
        
        # Get serve/return stats
        player1_sr = serve_return_df[
            (serve_return_df['player_id'] == player1_id) &
            (serve_return_df['tournament_date'] < match_date)
        ].iloc[-1] if len(serve_return_df[serve_return_df['player_id'] == player1_id]) > 0 else None
        
        player2_sr = serve_return_df[
            (serve_return_df['player_id'] == player2_id) &
            (serve_return_df['tournament_date'] < match_date)
        ].iloc[-1] if len(serve_return_df[serve_return_df['player_id'] == player2_id]) > 0 else None
        
        if player1_sr is None:
            print(f"WARNING: No serve/return stats found for player1 (ID: {player1_id})")
        if player2_sr is None:
            print(f"WARNING: No serve/return stats found for player2 (ID: {player2_id})")
        
        # Initialize features dictionary
        features = {
            'match_id': scheduled_match['match_id'],  # For future matches, we use the scheduled_matches.match_id
            'player1_id': player1_id,
            'player2_id': player2_id,
            'surface': scheduled_match['surface'].lower() if scheduled_match['surface'] else 'unknown',
            'tournament_date': match_date,
            'tournament_level': scheduled_match['tournament_level'],
            'is_future': True  # Mark as a future match for identification
        }
        
        # Add win rate features
        if player1_stats is not None and player2_stats is not None:
            features.update({
                'win_rate_5_diff': player1_stats.get('win_rate_5', 0) - player2_stats.get('win_rate_5', 0),
                'win_streak_diff': player1_stats.get('win_streak', 0) - player2_stats.get('win_streak', 0),
                'loss_streak_diff': player1_stats.get('loss_streak', 0) - player2_stats.get('loss_streak', 0),
                'player1_win_rate_5': player1_stats.get('win_rate_5', 0),
                'player2_win_rate_5': player2_stats.get('win_rate_5', 0),
                'player1_win_streak': player1_stats.get('win_streak', 0),
                'player2_win_streak': player2_stats.get('win_streak', 0),
                'player1_loss_streak': player1_stats.get('loss_streak', 0),
                'player2_loss_streak': player2_stats.get('loss_streak', 0)
            })
            
            # Add surface-specific win rates
            surface = features['surface']
            if surface in ['hard', 'clay', 'grass', 'carpet']:
                features.update({
                    f'win_rate_{surface}_5_diff': (
                        player1_stats.get(f'win_rate_{surface}_5', 0) -
                        player2_stats.get(f'win_rate_{surface}_5', 0)
                    ),
                    f'win_rate_{surface}_overall_diff': (
                        player1_stats.get(f'win_rate_{surface}_overall', 0) -
                        player2_stats.get(f'win_rate_{surface}_overall', 0)
                    ),
                    f'player1_win_rate_{surface}_5': player1_stats.get(f'win_rate_{surface}_5', 0),
                    f'player2_win_rate_{surface}_5': player2_stats.get(f'win_rate_{surface}_5', 0),
                    f'player1_win_rate_{surface}_overall': player1_stats.get(f'win_rate_{surface}_overall', 0),
                    f'player2_win_rate_{surface}_overall': player2_stats.get(f'win_rate_{surface}_overall', 0)
                })
        
        # Add serve/return features
        if player1_sr is not None and player2_sr is not None:
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
            
            for metric in serve_return_metrics:
                features[f'{metric}_diff'] = (
                    player1_sr.get(metric, 0) - player2_sr.get(metric, 0)
                )
                features[f'player1_{metric}'] = player1_sr.get(metric, 0)
                features[f'player2_{metric}'] = player2_sr.get(metric, 0)
        
        print(f"Generated {len(features)} features for match ID: {features['match_id']}")
        return features
    
    def store_features(self, features_list: List[Dict[str, Any]]):
        """
        Store generated features in the match_features table
        
        Args:
            features_list: List of feature dictionaries to store
        """
        if not features_list:
            logger.warning("No features to store")
            print("WARNING: No features to store")
            return
        
        print(f"Storing features for {len(features_list)} matches...")
        
        with self.get_db_connection() as conn:
            with conn.cursor() as cur:
                # Get column names from first feature dict
                columns = list(features_list[0].keys())
                
                # Prepare values list
                values = [[feature[col] for col in columns] for feature in features_list]
                
                # Create placeholders for SQL query
                placeholders = ','.join(['%s'] * len(columns))
                
                # Construct column string
                columns_str = ','.join(columns)
                
                # Insert features
                # Note: The match_id here will be the scheduled_matches.match_id for future matches
                execute_values(
                    cur,
                    f"""
                    INSERT INTO match_features (
                        {columns_str}
                    ) VALUES %s
                    ON CONFLICT (match_id) DO UPDATE SET
                        {','.join(f"{col} = EXCLUDED.{col}" for col in columns if col != 'match_id')}
                    """,
                    values
                )
                
                # Mark matches as processed
                match_ids = [f["match_id"] for f in features_list]
                cur.execute("""
                    UPDATE scheduled_matches
                    SET is_processed = TRUE
                    WHERE match_id = ANY(%s)
                """, (match_ids,))
                
                conn.commit()
                logger.info(f"Stored features for {len(features_list)} matches")
                print(f"Successfully stored features for {len(features_list)} matches and marked them as processed")
    
    def generate_features(self):
        """Main method to generate features for scheduled matches"""
        try:
            print("\n" + "="*50)
            print("STARTING FEATURE GENERATION FOR FUTURE MATCHES")
            print("="*50 + "\n")
            
            # Load historical data
            historical_df = self.load_historical_matches()
            
            # Calculate player features from historical data
            print("\n" + "-"*50)
            print("PREPARING PLAYER FEATURES FROM HISTORICAL DATA")
            print("-"*50)
            player_stats_df, serve_return_df = self.prepare_player_features(historical_df)
            
            # Load scheduled matches
            print("\n" + "-"*50)
            print("LOADING SCHEDULED MATCHES")
            print("-"*50)
            scheduled_df = self.load_scheduled_matches()
            
            if scheduled_df.empty:
                logger.info("No unprocessed scheduled matches found")
                print("No unprocessed scheduled matches found. Exiting.")
                return
            
            # Generate features for each scheduled match
            print("\n" + "-"*50)
            print("GENERATING FEATURES FOR SCHEDULED MATCHES")
            print("-"*50)
            features_list = []
            for _, match in tqdm(scheduled_df.iterrows(), total=len(scheduled_df), desc="Generating features"):
                match_date = pd.to_datetime(match['scheduled_date'])
                features = self.generate_match_features(
                    match,
                    player_stats_df,
                    serve_return_df,
                    match_date
                )
                features_list.append(features)
            
            # Store features
            print("\n" + "-"*50)
            print("STORING GENERATED FEATURES")
            print("-"*50)
            self.store_features(features_list)
            
            print("\n" + "="*50)
            print("FEATURE GENERATION COMPLETED SUCCESSFULLY")
            print("="*50)
            
        except Exception as e:
            logger.error(f"Error generating features: {e}")
            print(f"\nERROR: Feature generation failed: {e}")
            raise

def main():
    """Main execution function"""
    try:
        print("\nStarting tennis match feature generation script...")
        generator = FutureFeatureGenerator()
        generator.generate_features()
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        print(f"\nCRITICAL ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 