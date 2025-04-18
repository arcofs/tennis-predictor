"""
Tennis Match Prediction - Update Completed Matches (v4)

This script updates scheduled matches that have been completed by:
1. Finding scheduled matches that are past their date and not processed
2. Fetching the match results from the external API
3. Adding the match data to the historical matches table
4. Marking the scheduled match as processed

This connects the scheduled matches to historical matches,
ensuring prediction accuracy can be properly tracked.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values
import requests
import time
import json

# Add project root to path to allow imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Import external API client
sys.path.append(str(project_root / "external-api"))
from get_data_from_external_api import (
    TennisAPIClient, 
    get_match_stats_for_db,
    get_tournament_info_for_db
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{project_root}/predictor/v4/output/update_completed.log"),
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
        
        # API settings
        self.api_key = os.getenv("TENNIS_API_KEY")
        self.api_client = TennisAPIClient(quiet_mode=True)
        
        logger.info("CompletedMatchUpdater initialized")
    
    def get_db_connection(self):
        """Create a database connection"""
        return psycopg2.connect(self.db_url)
    
    def get_completed_unprocessed_matches(self) -> pd.DataFrame:
        """
        Get scheduled matches that should be completed but aren't processed
        
        Returns:
            DataFrame with match data
        """
        query = """
            SELECT *
            FROM scheduled_matches
            WHERE is_processed = FALSE
            AND scheduled_date < CURRENT_DATE
            ORDER BY scheduled_date DESC
            LIMIT 100
        """
        
        with self.get_db_connection() as conn:
            df = pd.read_sql(query, conn)
        
        logger.info(f"Found {len(df)} completed but unprocessed matches")
        return df
    
    def check_match_exists_in_matches(self, match_id: str) -> bool:
        """
        Check if a match already exists in the historical matches table
        
        Args:
            match_id: The external API match ID
            
        Returns:
            True if match exists, False otherwise
        """
        query = """
            SELECT 1 FROM matches
            WHERE match_num = %s
            LIMIT 1
        """
        
        with self.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (int(match_id),))
                exists = cur.fetchone() is not None
        
        return exists
    
    def get_match_result(self, match_id: str, tournament_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch match result from external API
        
        Args:
            match_id: Match ID
            tournament_id: Tournament ID
            
        Returns:
            Match data dictionary or None if not found
        """
        try:
            # First get tournament information
            tournament_info = self.api_client.get_tournament_info(tournament_id)
            if not tournament_info:
                logger.error(f"Failed to get tournament info for tournament {tournament_id}")
                return None
            
            # Get tournament fixtures
            all_matches = self.api_client.get_tournament_results(tournament_id)
            if not all_matches or 'data' not in all_matches:
                logger.error(f"Failed to get match results for tournament {tournament_id}")
                return None
            
            # Find our specific match
            match_data = None
            for match in all_matches['data']:
                if str(match.get('id')) == match_id:
                    match_data = match
                    break
            
            if not match_data:
                logger.error(f"Match {match_id} not found in tournament {tournament_id}")
                return None
            
            # Check if match has finished
            if match_data.get('status') != 'finished':
                logger.info(f"Match {match_id} is not yet finished, status: {match_data.get('status')}")
                return None
            
            # Get match statistics if available
            match_stats = None
            player1_id = match_data.get('player1', {}).get('id')
            player2_id = match_data.get('player2', {}).get('id')
            
            if player1_id and player2_id:
                match_stats = self.api_client.get_match_stats(tournament_id, player1_id, player2_id)
            
            # Prepare match data for database
            processed_tournament = get_tournament_info_for_db(tournament_info)
            processed_match = get_match_stats_for_db(match_data, processed_tournament)
            
            # Add statistics if available
            if match_stats:
                stats_data = get_match_stats_for_db(match_stats, processed_tournament)
                processed_match.update(stats_data)
            
            return processed_match
            
        except Exception as e:
            logger.error(f"Error getting match result: {str(e)}")
            return None
    
    def store_match_in_matches_table(self, match_data: Dict[str, Any]) -> bool:
        """
        Store match result in the matches table
        
        Args:
            match_data: Processed match data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare columns and values
            columns = list(match_data.keys())
            values = [match_data[col] for col in columns]
            
            placeholders = ", ".join(["%s"] * len(columns))
            column_str = ", ".join(columns)
            
            with self.get_db_connection() as conn:
                with conn.cursor() as cur:
                    # Check if match exists by match_num
                    cur.execute(
                        "SELECT id FROM matches WHERE match_num = %s", 
                        (match_data.get('match_num'),)
                    )
                    match_exists = cur.fetchone()
                    
                    if match_exists:
                        logger.info(f"Match {match_data.get('match_num')} already exists in matches table")
                        return True
                    
                    # Insert match data
                    query = f"""
                        INSERT INTO matches ({column_str})
                        VALUES ({placeholders})
                        RETURNING id
                    """
                    
                    cur.execute(query, values)
                    inserted_id = cur.fetchone()[0]
                    conn.commit()
                    
                    logger.info(f"Stored match {match_data.get('match_num')} in matches table with ID {inserted_id}")
                    return True
                
        except Exception as e:
            logger.error(f"Error storing match in matches table: {str(e)}")
            return False
    
    def mark_match_as_processed(self, match_id: str, processed: bool = True) -> bool:
        """
        Mark a scheduled match as processed
        
        Args:
            match_id: Match ID
            processed: Whether to mark as processed (True) or unprocessed (False)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE scheduled_matches SET is_processed = %s WHERE match_id = %s",
                        (processed, match_id)
                    )
                    conn.commit()
                    
                    logger.info(f"Marked match {match_id} as processed={processed}")
                    return True
                    
        except Exception as e:
            logger.error(f"Error marking match as processed: {str(e)}")
            return False
    
    def update_completed_matches(self) -> Tuple[int, int]:
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
            for _, match in matches_df.iterrows():
                match_id = match['match_id']
                
                # Skip if match already exists in matches table
                if self.check_match_exists_in_matches(match_id):
                    logger.info(f"Match {match_id} already exists in matches table, marking as processed")
                    self.mark_match_as_processed(match_id, True)
                    skipped_count += 1
                    continue
                
                # Get match result
                match_data = self.get_match_result(match_id, match['tournament_id'])
                
                if not match_data:
                    logger.warning(f"Could not get result for match {match_id}, skipping")
                    skipped_count += 1
                    continue
                
                # Store match data in matches table
                if self.store_match_in_matches_table(match_data):
                    # Mark match as processed
                    self.mark_match_as_processed(match_id, True)
                    updated_count += 1
                else:
                    logger.error(f"Failed to store match {match_id}")
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