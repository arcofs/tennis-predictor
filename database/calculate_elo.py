import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from typing import Dict, Tuple, List
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
from pydantic import BaseModel
import multiprocessing as mp
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import psycopg2
import psycopg2.extras

# Configure number of CPU cores to use (set to -1 to use all cores)
N_CORES = 120  # Change this value to limit the number of cores used

# Configure batch sizes
PROCESSING_BATCH_SIZE = 10000  # Size of batches for parallel processing
DB_BATCH_SIZE = 10000  # Reduced batch size for database updates
DB_PAGE_SIZE = 1000    # Size of each page for execute_values

# Get actual number of cores to use
N_CORES = mp.cpu_count() if N_CORES == -1 else min(N_CORES, mp.cpu_count())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_database_url(url: str) -> str:
    """Convert postgres:// to postgresql:// if needed"""
    if url.startswith('postgres://'):
        url = url.replace('postgres://', 'postgresql://', 1)
    return url

class EloCalculator:
    def __init__(self, initial_rating: float = 1500.0):
        """
        Initialize the Elo calculator
        
        Args:
            initial_rating (float): The rating assigned to new players
        """
        self.initial_rating = initial_rating
        self.player_ratings: Dict[int, float] = {}
        self.player_matches: Dict[int, int] = {}
        self.player_last_match: Dict[int, datetime] = {}
        
        # Tournament level weights
        self.tournament_weights = {
            'G': 1.2,    # Grand Slams
            'M': 1.1,    # Masters
            'A': 1.0,    # ATP Tour 500
            'B': 0.9,    # ATP Tour 250
            'C': 0.8,    # Challengers
            'F': 0.7,    # Futures
            'D': 1.0     # Default weight
        }

    def get_k_factor(self, player_matches: int) -> float:
        """
        Calculate dynamic K-factor based on number of matches played
        
        Args:
            player_matches (int): Number of matches played by the player
            
        Returns:
            float: K-factor value
        """
        if player_matches < 30:
            return 32.0  # New players - higher K for faster rating adjustment
        elif player_matches < 100:
            return 24.0  # Developing players
        else:
            return 16.0  # Established players - more stable ratings

    def apply_time_decay(self, rating: float, last_match_date: datetime, current_match_date: datetime) -> float:
        """
        Apply rating decay based on inactivity period
        
        Args:
            rating (float): Current rating
            last_match_date (datetime): Date of player's last match
            current_match_date (datetime): Date of current match
            
        Returns:
            float: Adjusted rating after decay
        """
        if last_match_date is None:
            return rating
            
        # Calculate months of inactivity
        days_inactive = (current_match_date - last_match_date).days
        months_inactive = days_inactive / 30.0
        
        if months_inactive <= 3:
            return rating  # No decay for up to 3 months inactivity
        
        # Apply decay based on inactivity period
        decay_factor = 0.995 ** (months_inactive - 3)  # 0.5% decay per month after 3 months
        return self.initial_rating + (rating - self.initial_rating) * decay_factor

    def get_player_rating(self, player_id: int, match_date: datetime) -> float:
        """Get a player's current rating with time decay, or initial rating if new player"""
        if player_id not in self.player_ratings:
            return self.initial_rating
            
        current_rating = self.player_ratings[player_id]
        last_match_date = self.player_last_match.get(player_id)
        
        if last_match_date:
            current_rating = self.apply_time_decay(current_rating, last_match_date, match_date)
            
        return current_rating

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate expected score for player A when playing against player B
        Using the formula: E = 1 / (1 + 10^((RB - RA)/400))
        """
        return 1 / (1 + 10**((rating_b - rating_a) / 400))

    def update_rating(self, 
                     player_id: int, 
                     opponent_id: int, 
                     score: float, 
                     match_date: datetime,
                     tourney_level: str) -> float:
        """
        Update ratings after a match
        
        Args:
            player_id (int): ID of the player
            opponent_id (int): ID of the opponent
            score (float): Actual score (1 for win, 0 for loss)
            match_date (datetime): Date of the match
            tourney_level (str): Tournament level code
            
        Returns:
            float: The new rating
        """
        # Get current ratings with time decay
        rating = self.get_player_rating(player_id, match_date)
        opponent_rating = self.get_player_rating(opponent_id, match_date)

        # Calculate expected score
        expected = self.expected_score(rating, opponent_rating)

        # Get tournament weight
        tournament_weight = self.tournament_weights.get(tourney_level, self.tournament_weights['D'])

        # Get dynamic K-factor
        k_factor = self.get_k_factor(self.player_matches.get(player_id, 0))

        # Update ratings with tournament weight
        new_rating = rating + k_factor * tournament_weight * (score - expected)

        # Store new rating and update match date
        self.player_ratings[player_id] = new_rating
        self.player_last_match[player_id] = match_date
        self.player_matches[player_id] = self.player_matches.get(player_id, 0) + 1
        
        return new_rating

def add_elo_columns(engine: create_engine) -> None:
    """Add Elo rating columns to the matches table if they don't exist"""
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                ALTER TABLE matches 
                ADD COLUMN IF NOT EXISTS winner_elo FLOAT,
                ADD COLUMN IF NOT EXISTS loser_elo FLOAT,
                ADD COLUMN IF NOT EXISTS winner_matches INTEGER,
                ADD COLUMN IF NOT EXISTS loser_matches INTEGER;
            """))
            conn.commit()
        logger.info("Added Elo columns to matches table successfully")
    except Exception as e:
        logger.error(f"Error adding Elo columns: {str(e)}")
        raise

def process_match_batch(batch_data: List[dict], calculator: EloCalculator) -> List[dict]:
    """Process a batch of matches for parallel processing"""
    updates = []
    for match in batch_data:
        winner_rating = calculator.update_rating(
            match['winner_id'],
            match['loser_id'],
            1.0,
            match['tournament_date'],
            match['tournament_level']
        )
        loser_rating = calculator.update_rating(
            match['loser_id'],
            match['winner_id'],
            0.0,
            match['tournament_date'],
            match['tournament_level']
        )
        
        updates.append({
            'id': match['id'],
            'winner_elo': winner_rating,
            'loser_elo': loser_rating,
            'winner_matches': calculator.player_matches[match['winner_id']],
            'loser_matches': calculator.player_matches[match['loser_id']]
        })
    return updates

def update_batch(engine: create_engine, updates: list) -> None:
    """Update a batch of matches with their Elo ratings using psycopg2 fast executemany"""
    try:
        # Extract connection parameters from SQLAlchemy engine
        params = engine.url.translate_connect_args()
        database = params['database']
        user = params['username']
        password = params['password']
        host = params['host']
        port = params.get('port', 5432)

        # Connect using psycopg2 for faster bulk updates
        with psycopg2.connect(
            dbname=database,
            user=user,
            password=password,
            host=host,
            port=port
        ) as conn:
            with conn.cursor() as cur:
                # Create temporary table with index
                cur.execute("""
                    CREATE TEMP TABLE temp_updates (
                        id INTEGER PRIMARY KEY,
                        winner_elo FLOAT,
                        loser_elo FLOAT,
                        winner_matches INTEGER,
                        loser_matches INTEGER
                    );
                    CREATE INDEX temp_updates_id_idx ON temp_updates(id);
                """)

                # Prepare data for bulk insert
                data = [(
                    update['id'],
                    update['winner_elo'],
                    update['loser_elo'],
                    update['winner_matches'],
                    update['loser_matches']
                ) for update in updates]

                # Fast bulk insert using execute_values with smaller page size
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO temp_updates (id, winner_elo, loser_elo, winner_matches, loser_matches)
                    VALUES %s
                    """,
                    data,
                    page_size=DB_PAGE_SIZE
                )

                # Bulk update using JOIN with progress logging
                logger.info("Performing final update from temporary table...")
                cur.execute("""
                    UPDATE matches m
                    SET 
                        winner_elo = t.winner_elo,
                        loser_elo = t.loser_elo,
                        winner_matches = t.winner_matches,
                        loser_matches = t.loser_matches
                    FROM temp_updates t
                    WHERE m.id = t.id
                """)

                # Log the number of rows updated
                rows_updated = cur.rowcount
                logger.info(f"Updated {rows_updated} rows in matches table")

                # Clean up
                cur.execute("DROP TABLE temp_updates")
                conn.commit()

    except Exception as e:
        logger.error(f"Error updating batch: {str(e)}")
        raise

def calculate_and_update_elo_ratings() -> None:
    """Main function to calculate and update Elo ratings"""
    try:
        # Load environment variables
        load_dotenv()
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            raise ValueError("DATABASE_URL not found in environment variables")

        # Convert postgres:// to postgresql:// if needed
        database_url = validate_database_url(database_url)
        logger.info(f"Using database URL with dialect: {database_url.split('://')[0]}")
        logger.info(f"Using {N_CORES} CPU cores for processing")
        logger.info(f"Processing batch size: {PROCESSING_BATCH_SIZE}")
        logger.info(f"Database batch size: {DB_BATCH_SIZE}")

        # Create database engine
        engine = create_engine(database_url)

        # Add Elo columns
        add_elo_columns(engine)

        # Load matches in chunks to handle large datasets
        chunk_size = PROCESSING_BATCH_SIZE * N_CORES
        chunks = pd.read_sql(
            """
            SELECT 
                id,
                tournament_date,
                winner_id,
                loser_id,
                tournament_level
            FROM matches 
            WHERE winner_id IS NOT NULL 
            AND loser_id IS NOT NULL
            AND winner_elo IS NULL
            ORDER BY tournament_date ASC
            """,
            engine,
            chunksize=chunk_size
        )

        total_processed = 0
        all_updates = []
        calculator = EloCalculator(initial_rating=1500.0)

        for chunk_df in chunks:
            if len(chunk_df) == 0:
                continue

            chunk_df['tournament_date'] = pd.to_datetime(chunk_df['tournament_date'])
            matches = chunk_df.to_dict('records')
            
            # Process chunk in parallel
            batches = [matches[i:i + PROCESSING_BATCH_SIZE] for i in range(0, len(matches), PROCESSING_BATCH_SIZE)]
            
            chunk_updates = []
            with tqdm(total=len(matches), desc=f"Processing chunk {total_processed+1}-{total_processed+len(matches)}") as pbar:
                with ProcessPoolExecutor(max_workers=N_CORES) as executor:
                    future_to_batch = {
                        executor.submit(process_match_batch, batch, calculator): batch 
                        for batch in batches
                    }
                    
                    for future in as_completed(future_to_batch):
                        try:
                            batch_updates = future.result()
                            chunk_updates.extend(batch_updates)
                            pbar.update(len(future_to_batch[future]))
                        except Exception as e:
                            logger.error(f"Error processing batch: {str(e)}")
                            raise

            # Update database when we have enough updates
            all_updates.extend(chunk_updates)
            if len(all_updates) >= DB_BATCH_SIZE:
                logger.info(f"Updating database with {len(all_updates)} matches...")
                update_batch(engine, all_updates)
                total_processed += len(all_updates)
                all_updates = []

        # Update remaining matches
        if all_updates:
            logger.info(f"Updating database with remaining {len(all_updates)} matches...")
            update_batch(engine, all_updates)
            total_processed += len(all_updates)

        logger.info(f"Successfully updated {total_processed} matches with Elo ratings")

    except Exception as e:
        logger.error(f"Error in calculate_and_update_elo_ratings: {str(e)}")
        raise

if __name__ == "__main__":
    calculate_and_update_elo_ratings()
