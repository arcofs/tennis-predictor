"""
Tennis Match Prediction - Future Match Prediction (v4)

This script makes predictions for upcoming tennis matches by:
1. Loading the latest trained model
2. Getting unprocessed scheduled matches
3. Generating features for these matches on-the-fly
4. Making predictions with confidence scores
5. Storing predictions in the match_predictions table

Important note about match IDs:
- matches table: 'id' is auto-incremented PK, 'match_num' is the external API match ID
- scheduled_matches table: 'match_id' is the external API match ID (as string)
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
import multiprocessing
from functools import partial

# Add project root to path to allow imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Multiprocessing settings
NUM_CORES = 0  # Set to 0 to use all available cores
CHUNK_MULTIPLIER = 8  # Controls chunk size for better load balancing
WORKER_BATCH_SIZE = 8000  # Maximum records to process in a worker
POOL_BATCH_SIZE = 16  # Number of chunks to process per pool creation

# If NUM_CORES is set to 0, use all available cores
if NUM_CORES <= 0:
    NUM_CORES = multiprocessing.cpu_count()

# Limit to a reasonable number to prevent system overload
NUM_CORES = min(NUM_CORES, multiprocessing.cpu_count())

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

# Add a process-safe logger for multiprocessing
mp_logger = multiprocessing.get_logger()
mp_logger.setLevel(logging.INFO)
mp_handler = logging.FileHandler(f"{project_root}/predictor/v4/output/logs/predictions_mp.log")
mp_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
mp_logger.addHandler(mp_handler)

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

        # Round mapping from API round_id to matches table format
        self.round_mapping = {
            # API round_id to standard format
            '1': 'Q1',     # Qualifier 1
            '2': 'Q2',     # Qualifier 2
            '3': 'Q3',     # Qualifier 3
            '4': 'R128',   # First round
            '5': 'R64',    # Second round
            '6': 'R32',    # Third round
            '7': 'R16',    # Fourth round
            '8': 'RR',     # Round Robin
            '9': 'QF',     # Quarter Final (1/4)
            '10': 'SF',    # Semi Final (1/2)
            '12': 'F',     # Final
            # Also support direct round names
            'Q1': 'Q1',
            'Q2': 'Q2',
            'Q3': 'Q3',
            'R128': 'R128',
            'R64': 'R64',
            'R32': 'R32',
            'R16': 'R16',
            'RR': 'RR',
            'QF': 'QF',
            'SF': 'SF',
            'F': 'F'
        }
    
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
        Get scheduled matches that need predictions
        
        Returns:
            DataFrame with match data
        """
        query = """
            WITH latest_player_features AS (
                SELECT 
                    player1_id as player_id,
                    player_elo_diff + 1500 as elo,  -- Convert diff back to absolute ELO
                    tournament_date,
                    ROW_NUMBER() OVER (PARTITION BY player1_id ORDER BY tournament_date DESC) as rn1
                FROM match_features
                UNION ALL
                SELECT 
                    player2_id as player_id,
                    1500 - player_elo_diff as elo,  -- Convert diff back to absolute ELO
                    tournament_date,
                    ROW_NUMBER() OVER (PARTITION BY player2_id ORDER BY tournament_date DESC) as rn1
                FROM match_features
            )
            SELECT 
                s.match_id,
                s.tournament_id,
                s.round,
                s.player1_id,
                s.player2_id,
                s.surface,
                s.scheduled_date,
                COALESCE(p1.elo, 1500) as player1_elo,
                COALESCE(p2.elo, 1500) as player2_elo
            FROM scheduled_matches s
            LEFT JOIN (
                SELECT player_id, elo
                FROM latest_player_features
                WHERE rn1 = 1
            ) p1 ON s.player1_id = p1.player_id
            LEFT JOIN (
                SELECT player_id, elo
                FROM latest_player_features
                WHERE rn1 = 1
            ) p2 ON s.player2_id = p2.player_id
            WHERE NOT EXISTS (
                SELECT 1 FROM match_predictions p
                WHERE p.match_id = s.match_id
                AND p.model_version = %s
            )
            AND s.scheduled_date >= CURRENT_DATE
            AND s.scheduled_date <= CURRENT_DATE + INTERVAL '7 days'
            AND s.is_processed = false
            ORDER BY s.scheduled_date ASC
        """
        
        with self.get_db_connection() as conn:
            df = pd.read_sql(query, conn, params=(self.model_version,))
        
        logger.info(f"Found {len(df)} matches needing predictions")
        return df
    
    def get_player_historical_stats(self, player_id: int, before_date: datetime) -> Dict[str, float]:
        """
        Get historical stats for a player before a given date
        
        Args:
            player_id: Player ID
            before_date: Get stats before this date
            
        Returns:
            Dictionary of player stats
        """
        query = """
            WITH player_matches AS (
                SELECT 
                    tournament_date,
                    CASE 
                        WHEN winner_id = %s THEN 1 
                        ELSE 0 
                    END as win,
                    surface
                FROM matches 
                WHERE (winner_id = %s OR loser_id = %s)
                AND tournament_date < %s
                ORDER BY tournament_date DESC
                LIMIT 20
            )
            SELECT 
                -- Overall stats
                AVG(win) as win_rate_5,
                SUM(CASE WHEN win = 1 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN win = 0 THEN 1 ELSE 0 END) as losses,
                -- Surface stats
                AVG(CASE WHEN surface = 'hard' THEN win END) as win_rate_hard,
                AVG(CASE WHEN surface = 'clay' THEN win END) as win_rate_clay,
                AVG(CASE WHEN surface = 'grass' THEN win END) as win_rate_grass,
                AVG(CASE WHEN surface = 'carpet' THEN win END) as win_rate_carpet
            FROM player_matches
        """
        
        with self.get_db_connection() as conn:
            stats = pd.read_sql(query, conn, params=(player_id, player_id, player_id, before_date))
            
            if stats.empty:
                return {}
            
            # Calculate streaks
            cursor = conn.cursor()
            streak_query = """
                WITH player_matches AS (
                    SELECT 
                        tournament_date,
                        CASE 
                            WHEN winner_id = %s THEN 1 
                            ELSE 0 
                        END as win
                    FROM matches 
                    WHERE (winner_id = %s OR loser_id = %s)
                    AND tournament_date < %s
                    ORDER BY tournament_date DESC
                    LIMIT 20
                )
                SELECT win FROM player_matches
            """
            cursor.execute(streak_query, (player_id, player_id, player_id, before_date))
            results = cursor.fetchall()
            
            # Calculate streaks
            win_streak = 0
            loss_streak = 0
            for (win,) in results:
                if win == 1:
                    win_streak += 1
                    loss_streak = 0
                else:
                    loss_streak += 1
                    win_streak = 0
            
            stats_dict = stats.iloc[0].to_dict()
            stats_dict.update({
                'win_streak': win_streak,
                'loss_streak': loss_streak
            })
            
            return stats_dict
    
    def generate_match_features(self, match: pd.Series) -> Dict[str, float]:
        """
        Generate features for a single match
        
        Args:
            match: Series with match data
            
        Returns:
            Dictionary of features
        """
        # Get historical stats for both players
        player1_stats = self.get_player_historical_stats(match['player1_id'], match['scheduled_date'])
        player2_stats = self.get_player_historical_stats(match['player2_id'], match['scheduled_date'])
        
        try:
            # Calculate feature differences
            features = {
                'player_elo_diff': match['player1_elo'] - match['player2_elo'],
                'win_rate_5_diff': player1_stats['win_rate_5'] - player2_stats['win_rate_5'],
                'win_streak_diff': player1_stats['win_streak'] - player2_stats['win_streak'],
                'loss_streak_diff': player1_stats['loss_streak'] - player2_stats['loss_streak']
            }
            
            # Add surface-specific features - handle None values by defaulting to 0
            # This matches the behavior in the training data where no history on a surface means 0
            for surface in ['hard', 'clay', 'grass', 'carpet']:
                p1_rate = player1_stats.get(f'win_rate_{surface}') or 0
                p2_rate = player2_stats.get(f'win_rate_{surface}') or 0
                features[f'win_rate_{surface}_5_diff'] = p1_rate - p2_rate
            
            # Add raw player stats
            for stat, value in player1_stats.items():
                features[f'player1_{stat}'] = value if value is not None else 0
            for stat, value in player2_stats.items():
                features[f'player2_{stat}'] = value if value is not None else 0
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating features for match {match['match_id']}: {str(e)}")
            raise
    
    def prepare_features(self, matches_df: pd.DataFrame) -> np.ndarray:
        """
        Prepare feature matrix for prediction using parallel processing
        
        Args:
            matches_df: DataFrame with match data
            
        Returns:
            numpy array of features in correct order
        """
        logger.info(f"Preparing features using {NUM_CORES} CPU cores")
        
        # Create smaller chunks for better load balancing
        chunk_size = max(1, len(matches_df) // (NUM_CORES * CHUNK_MULTIPLIER))
        chunks = [(i, matches_df.iloc[i:i+chunk_size]) for i in range(0, len(matches_df), chunk_size)]
        
        logger.info(f"Created {len(chunks)} chunks with approximately {chunk_size} matches each")
        
        all_features = []
        
        # Process in sequential chunks to avoid memory issues
        num_batches = (len(chunks) + POOL_BATCH_SIZE - 1) // POOL_BATCH_SIZE
        logger.info(f"Processing data in {num_batches} batches of {POOL_BATCH_SIZE} chunks each")
        
        for i in range(0, len(chunks), POOL_BATCH_SIZE):
            batch_chunks = chunks[i:i+POOL_BATCH_SIZE]
            batch_num = i // POOL_BATCH_SIZE + 1
            
            logger.info(f"Starting batch {batch_num}/{num_batches} with {len(batch_chunks)} chunks")
            
            try:
                # Create a pool of workers with maxtasksperchild to free resources
                with multiprocessing.Pool(processes=NUM_CORES, maxtasksperchild=2) as pool:
                    # Process each chunk in parallel
                    results = list(tqdm(
                        pool.imap(
                            partial(process_match_features_batch, db_url=self.db_url),
                            batch_chunks
                        ),
                        total=len(batch_chunks),
                        desc=f"Processing batch {batch_num}/{num_batches}",
                        unit="chunk"
                    ))
                
                # Count valid results
                valid_results = [r for r in results if r is not None]
                logger.info(f"Batch {batch_num}: processed {len(valid_results)}/{len(batch_chunks)} chunks successfully")
                
                # Combine results from this batch
                batch_features_count = 0
                for chunk_features in results:
                    if chunk_features is not None:
                        all_features.extend(chunk_features)
                        batch_features_count += len(chunk_features)
                
                logger.info(f"Batch {batch_num}: added {batch_features_count} features, total so far: {len(all_features)}")
                
                # Free memory
                del results
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {str(e)}")
                continue
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        
        # Ensure all required features are present
        missing_features = set(self.feature_columns) - set(features_df.columns)
        if missing_features:
            # Fill missing features with 0 or appropriate default
            for feature in missing_features:
                features_df[feature] = 0
        
        # Create feature matrix in correct order
        X = features_df[self.feature_columns].values
        
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
                        'match_id': row['match_id'],
                        'tournament_id': row['tournament_id'],
                        'round': row['round'],
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
                    logger.info(f"Stored predictions for {len(prediction_data)} matches")
    
    def update_prediction_accuracy(self):
        """Update accuracy for past predictions where results are now known"""
        query = """
            WITH round_mapping AS (
                SELECT unnest(ARRAY[
                    '1','2','3','4','5','6','7','8','9','10','12',
                    'Q1','Q2','Q3','R128','R64','R32','R16','RR','QF','SF','F'
                ]) as api_round,
                unnest(ARRAY[
                    'Q1','Q2','Q3','R128','R64','R32','R16','RR','QF','SF','F',
                    'Q1','Q2','Q3','R128','R64','R32','R16','RR','QF','SF','F'
                ]) as standard_round
            )
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
            WHERE p.tournament_id = m.tournament_id
            AND EXISTS (
                SELECT 1 FROM round_mapping rm 
                WHERE rm.api_round = p.round
                AND rm.standard_round = m.round
            )
            AND (
                (p.player1_id = m.winner_id AND p.player2_id = m.loser_id)
                OR 
                (p.player1_id = m.loser_id AND p.player2_id = m.winner_id)
            )
            AND p.actual_winner_id IS NULL
            AND m.winner_id IS NOT NULL
        """
        
        with self.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                updated_predictions = cur.rowcount
                conn.commit()
                
                if updated_predictions > 0:
                    logger.info(f"Updated accuracy for {updated_predictions} past predictions")
    
    def predict_matches(self):
        """Main method to generate predictions for upcoming matches"""
        try:
            # Load latest model
            self.model, self.model_version, self.feature_columns = self.load_latest_model()
            logger.info(f"Loaded model version: {self.model_version}")
            
            # Update accuracy for past predictions
            self.update_prediction_accuracy()
            
            # Get unprocessed matches
            matches_df = self.get_unprocessed_matches()
            
            if matches_df.empty:
                logger.info("No matches need predictions")
                return
            
            logger.info(f"Processing {len(matches_df)} matches...")
            
            # Prepare features
            X = self.prepare_features(matches_df)
            
            # Generate predictions
            predictions = self.make_predictions(X)
            
            # Store predictions
            self.store_predictions(matches_df, predictions)
            
            logger.info("Completed match predictions successfully")
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise

def process_match_features_batch(batch_data: Tuple[int, pd.DataFrame], db_url: str) -> List[Dict[str, float]]:
    """
    Process a batch of matches to generate features in parallel
    
    Args:
        batch_data: Tuple containing (batch_idx, batch_df)
        db_url: Database connection URL
        
    Returns:
        List of feature dictionaries
    """
    try:
        batch_idx, batch_df = batch_data
        
        # Create a connection for this worker
        conn = psycopg2.connect(db_url)
        
        features_list = []
        for _, match in batch_df.iterrows():
            # Get historical stats for both players
            player1_stats = get_player_historical_stats_worker(conn, match['player1_id'], match['scheduled_date'])
            player2_stats = get_player_historical_stats_worker(conn, match['player2_id'], match['scheduled_date'])
            
            try:
                # Calculate feature differences
                features = {
                    'player_elo_diff': match['player1_elo'] - match['player2_elo'],
                    'win_rate_5_diff': player1_stats['win_rate_5'] - player2_stats['win_rate_5'],
                    'win_streak_diff': player1_stats['win_streak'] - player2_stats['win_streak'],
                    'loss_streak_diff': player1_stats['loss_streak'] - player2_stats['loss_streak']
                }
                
                # Add surface-specific features
                for surface in ['hard', 'clay', 'grass', 'carpet']:
                    p1_rate = player1_stats.get(f'win_rate_{surface}') or 0
                    p2_rate = player2_stats.get(f'win_rate_{surface}') or 0
                    features[f'win_rate_{surface}_5_diff'] = p1_rate - p2_rate
                
                # Add raw player stats
                for stat, value in player1_stats.items():
                    features[f'player1_{stat}'] = value if value is not None else 0
                for stat, value in player2_stats.items():
                    features[f'player2_{stat}'] = value if value is not None else 0
                
                features_list.append(features)
                
            except Exception as e:
                mp_logger.error(f"Error calculating features for match {match['match_id']}: {str(e)}")
                continue
            
            # Limit batch size
            if len(features_list) >= WORKER_BATCH_SIZE:
                break
        
        conn.close()
        return features_list
        
    except Exception as e:
        mp_logger.error(f"Error processing batch {batch_idx}: {str(e)}")
        return None

def get_player_historical_stats_worker(conn: psycopg2.extensions.connection, player_id: int, before_date: datetime) -> Dict[str, float]:
    """Worker version of get_player_historical_stats that uses an existing connection"""
    query = """
        WITH player_matches AS (
            SELECT 
                tournament_date,
                CASE 
                    WHEN winner_id = %s THEN 1 
                    ELSE 0 
                END as win,
                surface
            FROM matches 
            WHERE (winner_id = %s OR loser_id = %s)
            AND tournament_date < %s
            ORDER BY tournament_date DESC
            LIMIT 20
        )
        SELECT 
            -- Overall stats
            AVG(win) as win_rate_5,
            SUM(CASE WHEN win = 1 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN win = 0 THEN 1 ELSE 0 END) as losses,
            -- Surface stats
            AVG(CASE WHEN surface = 'hard' THEN win END) as win_rate_hard,
            AVG(CASE WHEN surface = 'clay' THEN win END) as win_rate_clay,
            AVG(CASE WHEN surface = 'grass' THEN win END) as win_rate_grass,
            AVG(CASE WHEN surface = 'carpet' THEN win END) as win_rate_carpet
        FROM player_matches
    """
    
    stats = pd.read_sql(query, conn, params=(player_id, player_id, player_id, before_date))
    
    if stats.empty:
        return {}
    
    # Calculate streaks
    cursor = conn.cursor()
    streak_query = """
        WITH player_matches AS (
            SELECT 
                tournament_date,
                CASE 
                    WHEN winner_id = %s THEN 1 
                    ELSE 0 
                END as win
            FROM matches 
            WHERE (winner_id = %s OR loser_id = %s)
            AND tournament_date < %s
            ORDER BY tournament_date DESC
            LIMIT 20
        )
        SELECT win FROM player_matches
    """
    cursor.execute(streak_query, (player_id, player_id, player_id, before_date))
    results = cursor.fetchall()
    
    # Calculate streaks
    win_streak = 0
    loss_streak = 0
    for (win,) in results:
        if win == 1:
            win_streak += 1
            loss_streak = 0
        else:
            loss_streak += 1
            win_streak = 0
    
    stats_dict = stats.iloc[0].to_dict()
    stats_dict.update({
        'win_streak': win_streak,
        'loss_streak': loss_streak
    })
    
    return stats_dict

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