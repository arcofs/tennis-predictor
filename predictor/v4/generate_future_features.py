"""
Tennis Match Prediction - Future Match Feature Generation (v4)

This script generates features for upcoming tennis matches by:
1. Loading unprocessed matches from scheduled_matches table
2. Using pre-calculated historical features from match_features table
3. Generating features for upcoming matches
4. Storing features in match_features table with is_future flag

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{project_root}/predictor/v4/output/logs/feature_generation.log"),
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
    
    def load_player_features(self, cutoff_date: datetime) -> pd.DataFrame:
        """
        Load pre-calculated player features from match_features table
        
        Args:
            cutoff_date: Only load features from matches before this date
            
        Returns:
            DataFrame with player features
        """
        print("Loading pre-calculated player features...")
        query = """
            WITH RankedFeatures AS (
                SELECT 
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY 
                            CASE 
                                WHEN result = 1 THEN player1_id 
                                ELSE player2_id 
                            END 
                        ORDER BY tournament_date DESC
                    ) as rank
                FROM match_features
                WHERE tournament_date < %s
                AND is_future IS NOT TRUE
            )
            SELECT * FROM RankedFeatures
            WHERE rank = 1
        """
        
        with self.get_db_connection() as conn:
            df = pd.read_sql(query, conn, params=(cutoff_date,))
        
        logger.info(f"Loaded features for {len(df)} players")
        print(f"Loaded features for {len(df)} players")
        
        # Print sample of loaded features
        if not df.empty:
            print("\nSample of loaded player features:")
            print(df.head(2).to_string())
        return df
    
    def load_scheduled_matches(self) -> pd.DataFrame:
        """
        Load unplayed scheduled matches for the next 14 days.
        We use is_processed=FALSE to identify matches that haven't been played yet.
        
        Returns:
            DataFrame with scheduled match data
        """
        print("Loading unplayed scheduled matches...")
        query = """
            SELECT *
            FROM scheduled_matches
            WHERE scheduled_date >= CURRENT_DATE
            AND scheduled_date <= CURRENT_DATE + INTERVAL '14 days'
            AND is_processed = FALSE
            ORDER BY scheduled_date ASC
        """
        
        with self.get_db_connection() as conn:
            df = pd.read_sql(query, conn)
        
        logger.info(f"Loaded {len(df)} unplayed matches")
        print(f"Loaded {len(df)} unplayed matches")
        
        # Print sample of scheduled matches
        if not df.empty:
            print("\nSample of scheduled matches:")
            print(df.head(3).to_string())
        return df
    
    def generate_match_features(
        self,
        scheduled_match: pd.Series,
        player_features_df: pd.DataFrame,
        match_date: datetime
    ) -> Dict[str, Any]:
        """
        Generate features for a scheduled match using pre-calculated player features
        
        Args:
            scheduled_match: Series containing scheduled match data
            player_features_df: DataFrame with pre-calculated player features
            match_date: Date of the scheduled match
            
        Returns:
            Dictionary of calculated features
        """
        # Get player IDs
        player1_id = scheduled_match['player1_id']
        player2_id = scheduled_match['player2_id']
        
        print(f"\nGenerating features for match: {player1_id} vs {player2_id}")
        
        # Get most recent features for each player
        player1_features = player_features_df[
            ((player_features_df['player1_id'] == player1_id) & (player_features_df['result'] == 1)) |
            ((player_features_df['player2_id'] == player1_id) & (player_features_df['result'] == 0))
        ].iloc[0] if len(player_features_df[
            ((player_features_df['player1_id'] == player1_id) & (player_features_df['result'] == 1)) |
            ((player_features_df['player2_id'] == player1_id) & (player_features_df['result'] == 0))
        ]) > 0 else None
        
        player2_features = player_features_df[
            ((player_features_df['player1_id'] == player2_id) & (player_features_df['result'] == 1)) |
            ((player_features_df['player2_id'] == player2_id) & (player_features_df['result'] == 0))
        ].iloc[0] if len(player_features_df[
            ((player_features_df['player1_id'] == player2_id) & (player_features_df['result'] == 1)) |
            ((player_features_df['player2_id'] == player2_id) & (player_features_df['result'] == 0))
        ]) > 0 else None
        
        if player1_features is None:
            print(f"WARNING: No historical features found for player1 (ID: {player1_id})")
        if player2_features is None:
            print(f"WARNING: No historical features found for player2 (ID: {player2_id})")
        
        # Initialize features dictionary
        features = {
            'match_id': scheduled_match['match_id'],  # For future matches, we use the scheduled_matches.match_id
            'player1_id': player1_id,
            'player2_id': player2_id,
            'surface': scheduled_match['surface'].lower() if scheduled_match['surface'] else 'unknown',
            'tournament_level': scheduled_match['tournament_level'],  # Add tournament_level
            'tournament_date': match_date,
            'is_future': True  # Mark as a future match for identification
        }
        
        # If we have features for both players, calculate differences and copy individual stats
        if player1_features is not None and player2_features is not None:
            # Get the feature columns that represent differences
            diff_columns = [col for col in player1_features.index if col.endswith('_diff')]
            
            # Add difference features
            for col in diff_columns:
                p1_val = player1_features[col] if player1_features['result'] == 1 else -player1_features[col]
                p2_val = player2_features[col] if player2_features['result'] == 1 else -player2_features[col]
                features[col] = p1_val - p2_val
            
            # Get individual player stat columns (those starting with 'player1_' or 'player2_')
            player_stat_columns = [col for col in player1_features.index 
                                 if col.startswith(('player1_', 'player2_'))]
            
            # Add individual player stats
            for col in player_stat_columns:
                if col.startswith('player1_'):
                    # For player1 stats
                    if player1_features['result'] == 1:
                        features[col] = player1_features[col]
                    else:
                        features[col] = player1_features[col.replace('player1_', 'player2_')]
                elif col.startswith('player2_'):
                    # For player2 stats
                    if player2_features['result'] == 1:
                        features[col] = player2_features[col.replace('player2_', 'player1_')]
                    else:
                        features[col] = player2_features[col]
        
        print(f"Generated {len(features)} features for match ID: {features['match_id']}")
        return features
    
    def _convert_numpy_types(self, features_dict):
        """
        Convert numpy types to Python native types for database compatibility
        
        Args:
            features_dict: Dictionary containing feature values
            
        Returns:
            Dictionary with numpy types converted to Python native types
        """
        converted = {}
        for key, value in features_dict.items():
            if value is None:
                converted[key] = None
            elif isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
                converted[key] = int(value)
            elif isinstance(value, (np.float64, np.float32)):
                converted[key] = float(value)
            elif isinstance(value, (pd.Timestamp, np.datetime64)):
                converted[key] = value.to_pydatetime()
            else:
                converted[key] = value
        return converted
    
    def store_features(self, features_list: List[Dict[str, Any]]):
        """
        Store generated features in the match_features table.
        Note: We don't set is_processed here anymore as that's handled by update_completed_matches.py
        when the match is actually completed.
        
        Args:
            features_list: List of feature dictionaries to store
        """
        if not features_list:
            logger.warning("No features to store")
            print("WARNING: No features to store")
            return
        
        print(f"Storing/updating features for {len(features_list)} matches...")
        
        # Convert numpy types to Python native types
        converted_features = [self._convert_numpy_types(features) for features in features_list]
        
        with self.get_db_connection() as conn:
            with conn.cursor() as cur:
                # Get column names from first feature dict
                columns = list(converted_features[0].keys())
                
                # Prepare values list
                values = [[feature[col] for col in columns] for feature in converted_features]
                
                # Create placeholders for SQL query
                placeholders = ','.join(['%s'] * len(columns))
                
                # Construct column string
                columns_str = ','.join(columns)
                
                # Insert/update features
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
                
                conn.commit()
                logger.info(f"Stored/updated features for {len(features_list)} matches")
                print(f"Successfully stored/updated features for {len(features_list)} matches")
    
    def generate_features(self):
        """Main method to generate features for scheduled matches"""
        try:
            print("\n" + "="*50)
            print("STARTING FEATURE GENERATION FOR FUTURE MATCHES")
            print("="*50 + "\n")
            
            # Load scheduled matches
            print("\n" + "-"*50)
            print("LOADING SCHEDULED MATCHES")
            print("-"*50)
            scheduled_df = self.load_scheduled_matches()
            
            if scheduled_df.empty:
                logger.info("No scheduled matches found")
                print("No scheduled matches found. Exiting.")
                return
            
            # Load pre-calculated player features
            print("\n" + "-"*50)
            print("LOADING PRE-CALCULATED PLAYER FEATURES")
            print("-"*50)
            earliest_match_date = pd.to_datetime(scheduled_df['scheduled_date']).min()
            player_features_df = self.load_player_features(earliest_match_date)
            
            # Generate features for each scheduled match
            print("\n" + "-"*50)
            print("GENERATING FEATURES FOR SCHEDULED MATCHES")
            print("-"*50)
            features_list = []
            for _, match in tqdm(scheduled_df.iterrows(), total=len(scheduled_df), desc="Generating features"):
                match_date = pd.to_datetime(match['scheduled_date'])
                features = self.generate_match_features(
                    match,
                    player_features_df,
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