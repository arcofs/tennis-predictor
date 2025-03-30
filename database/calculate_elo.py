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

# Configure number of CPU cores to use (set to -1 to use all cores)
N_CORES = 120  # Change this value to limit the number of cores used

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

        # Create database engine
        engine = create_engine(database_url)

        # Add Elo columns
        add_elo_columns(engine)

        # Load matches ordered by date
        query = """
            SELECT 
                id,
                tournament_date,
                winner_id,
                loser_id,
                tournament_level
            FROM matches 
            WHERE winner_id IS NOT NULL 
            AND loser_id IS NOT NULL
            ORDER BY tournament_date ASC
        """
        
        df = pd.read_sql(query, engine)
        df['tournament_date'] = pd.to_datetime(df['tournament_date'])
        total_matches = len(df)
        
        # Initialize Elo calculator
        calculator = EloCalculator(initial_rating=1500.0)
        
        # Convert DataFrame to list of dictionaries for parallel processing
        matches = df.to_dict('records')
        
        # Process matches in parallel with progress bar
        batch_size = max(1000, total_matches // (N_CORES * 10))  # Adjust batch size based on total matches
        batches = [matches[i:i + batch_size] for i in range(0, len(matches), batch_size)]
        
        all_updates = []
        with tqdm(total=total_matches, desc="Processing matches") as pbar:
            with ProcessPoolExecutor(max_workers=N_CORES) as executor:
                # Submit all batches to the process pool
                future_to_batch = {
                    executor.submit(process_match_batch, batch, calculator): batch 
                    for batch in batches
                }
                
                # Process completed batches and update progress
                for future in as_completed(future_to_batch):
                    batch = future_to_batch[future]
                    try:
                        batch_updates = future.result()
                        all_updates.extend(batch_updates)
                        pbar.update(len(batch))
                    except Exception as e:
                        logger.error(f"Error processing batch: {str(e)}")
                        raise
        
        # Update database with all processed matches
        logger.info("Updating database with processed matches...")
        with tqdm(total=len(all_updates), desc="Updating database") as pbar:
            for i in range(0, len(all_updates), batch_size):
                batch = all_updates[i:i + batch_size]
                update_batch(engine, batch)
                pbar.update(len(batch))
        
        logger.info(f"Successfully updated {total_matches} matches with Elo ratings")

    except Exception as e:
        logger.error(f"Error in calculate_and_update_elo_ratings: {str(e)}")
        raise

def update_batch(engine: create_engine, updates: list) -> None:
    """Update a batch of matches with their Elo ratings"""
    try:
        with engine.connect() as conn:
            for update in updates:
                conn.execute(
                    text("""
                        UPDATE matches 
                        SET 
                            winner_elo = :winner_elo,
                            loser_elo = :loser_elo,
                            winner_matches = :winner_matches,
                            loser_matches = :loser_matches
                        WHERE id = :id
                    """),
                    update
                )
            conn.commit()
    except Exception as e:
        logger.error(f"Error updating batch: {str(e)}")
        raise

if __name__ == "__main__":
    calculate_and_update_elo_ratings()
