"""
Tennis Match Prediction - Update Completed Matches (v4)

This script updates scheduled matches that have been completed by:
1. Finding scheduled matches that are past their date and not processed
2. Checking if the match exists in the matches table
3. Marking the scheduled match as processed if found

This connects the scheduled matches to historical matches,
ensuring prediction accuracy can be properly tracked.
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import psycopg2
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# Add project root to path to allow imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{project_root}/predictor/v4/output/logs/update_completed.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CompletedMatchUpdater:
    def __init__(self):
        """Initialize the completed match updater"""
        load_dotenv()
        
        # Database settings
        self.db_url = os.getenv("DATABASE_URL")
        if self.db_url and self.db_url.startswith('postgres://'):
            self.db_url = self.db_url.replace('postgres://', 'postgresql://', 1)
        
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
        
        logger.info("CompletedMatchUpdater initialized")
    
    def get_db_connection(self):
        """Create a database connection"""
        return psycopg2.connect(self.db_url)
    
    def get_completed_unprocessed_matches(self) -> pd.DataFrame:
        """
        Get scheduled matches that should be completed but aren't processed.
        This includes:
        1. Matches with dates in the past
        2. Matches with NULL dates that need to be checked
        
        Returns:
            DataFrame with match data including round and player IDs
        """
        query = """
            SELECT 
                match_id,
                tournament_id,
                round,
                player1_id,
                player2_id,
                scheduled_date,
                is_processed
            FROM scheduled_matches
            WHERE is_processed = FALSE
            AND (scheduled_date < CURRENT_DATE OR scheduled_date IS NULL)
            ORDER BY COALESCE(scheduled_date, '2099-12-31') DESC
            LIMIT 100
        """
        
        with self.get_db_connection() as conn:
            df = pd.read_sql(query, conn)
        
        null_date_count = df['scheduled_date'].isnull().sum()
        past_date_count = len(df) - null_date_count
        
        logger.info(f"Found {len(df)} unprocessed matches total:")
        logger.info(f"- {past_date_count} matches with past dates")
        logger.info(f"- {null_date_count} matches with NULL dates")
        
        return df
    
    def check_match_stats_exist(self, conn, match_id: str) -> bool:
        """
        Check if match exists in the matches table by matching tournament, round and players.
        
        Args:
            conn: Database connection
            match_id: Match ID to check
            
        Returns:
            bool: True if match exists in database, False otherwise
        """
        try:
            with conn.cursor() as cur:
                # First get the scheduled match details
                cur.execute("""
                    SELECT tournament_id, round, player1_id, player2_id
                    FROM scheduled_matches
                    WHERE match_id = %s
                """, (match_id,))
                
                scheduled_match = cur.fetchone()
                if not scheduled_match:
                    logger.info(f"Scheduled match {match_id} not found")
                    return False
                    
                tournament_id, round_name, player1_id, player2_id = scheduled_match
                
                # Map the round name
                mapped_round = self.round_mapping.get(round_name)
                if not mapped_round:
                    logger.warning(f"Unknown round mapping for {round_name}")
                    mapped_round = round_name  # Use original if no mapping exists
                
                # Check if match exists with these details
                cur.execute("""
                    SELECT COUNT(*) 
                    FROM matches 
                    WHERE tournament_id = %s 
                    AND round = %s
                    AND (
                        (winner_id = %s AND loser_id = %s)
                        OR 
                        (winner_id = %s AND loser_id = %s)
                    )
                """, (tournament_id, mapped_round, player1_id, player2_id, player2_id, player1_id))
                
                count = cur.fetchone()[0]
                
                exists = count > 0
                if exists:
                    logger.info(f"Match ID {match_id} found in matches table")
                else:
                    logger.info(f"Match ID {match_id} not found in matches table")
                    
                return exists
                
        except Exception as e:
            logger.error(f"Error checking match in database: {str(e)}")
            return False
    
    def mark_match_as_processed(self, match_id: str, processed: bool = True) -> bool:
        """
        Mark a scheduled match as processed.
        
        Args:
            match_id: Match ID
            processed: Whether to mark as processed (True) or unprocessed (False)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cur:
                    try:
                        # Update scheduled_matches table only
                        cur.execute(
                            "UPDATE scheduled_matches SET is_processed = %s, last_processed_at = CURRENT_TIMESTAMP WHERE match_id = %s",
                            (processed, match_id)
                        )
                        
                        # Commit the transaction
                        conn.commit()
                        logger.info(f"Successfully marked match {match_id} as processed={processed}")
                        
                        # Note: We deliberately don't update match_features here
                        # This allows generate_historical_features.py to handle all feature generation
                        # which maintains a cleaner separation of concerns
                        
                        return True
                        
                    except Exception as e:
                        # If update fails, roll back
                        conn.rollback()
                        logger.error(f"Error marking match as processed, rolling back: {str(e)}")
                        return False
                        
        except Exception as e:
            logger.error(f"Database connection error: {str(e)}")
            return False
    
    def update_completed_matches(self) -> tuple[int, int]:
        """
        Main method to update completed matches
        
        Returns:
            Tuple of (number of matches updated, number of matches skipped)
        """
        updated_count = 0
        skipped_count = 0
        
        try:
            # Get unprocessed matches
            matches_df = self.get_completed_unprocessed_matches()
            
            if matches_df.empty:
                logger.info("No completed unprocessed matches found")
                return 0, 0
            
            # Process each match
            with self.get_db_connection() as conn:
                for _, match in matches_df.iterrows():
                    match_id = match['match_id']
                    
                    # Check if match exists in matches table by tournament, round and players
                    if self.check_match_stats_exist(conn, match_id):
                        # Match found, mark as processed
                        if self.mark_match_as_processed(match_id, True):
                            updated_count += 1
                        else:
                            skipped_count += 1
                    else:
                        # Match not found, skip it
                        skipped_count += 1
            
            return updated_count, skipped_count
            
        except Exception as e:
            logger.error(f"Error in update_completed_matches: {str(e)}")
            return updated_count, skipped_count

def main():
    """Main execution function"""
    try:
        updater = CompletedMatchUpdater()
        updated, skipped = updater.update_completed_matches()
        logger.info(f"Updated {updated} matches, skipped {skipped} matches")
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 