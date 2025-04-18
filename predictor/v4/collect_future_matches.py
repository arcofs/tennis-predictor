"""
Tennis Match Prediction - Future Match Collection Script (v4)

This script collects upcoming tennis matches for the next 7 days and stores them in the database.
It handles:
- Database table creation/verification
- Tournament calendar retrieval
- Match fixture collection
- Player information updates
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values
import requests
import time

# Add project root to path to allow imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{project_root}/predictor/v4/output/logs/collection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
API_BASE_URL = "https://tennis-api-atp-wta-itf.p.rapidapi.com/tennis/v2"
API_REQUESTS_PER_SECOND = 5  # Rate limiting
PAGE_SIZE = 500

class TennisDataCollector:
    def __init__(self):
        """Initialize the collector with API and database settings"""
        load_dotenv()
        
        # API settings
        self.api_key = os.getenv("TENNIS_API_KEY")
        self.headers = {
            'x-rapidapi-host': "tennis-api-atp-wta-itf.p.rapidapi.com",
            'x-rapidapi-key': self.api_key
        }
        
        # Database settings
        self.db_url = os.getenv("DATABASE_URL")
        if self.db_url and self.db_url.startswith('postgres://'):
            self.db_url = self.db_url.replace('postgres://', 'postgresql://', 1)
        
        # Rate limiting
        self.last_request_time = time.time()
        self.request_count = 0
    
    def _apply_rate_limiting(self):
        """Apply rate limiting to API requests"""
        self.request_count += 1
        if self.request_count >= API_REQUESTS_PER_SECOND:
            elapsed = time.time() - self.last_request_time
            if elapsed < 1.0:
                sleep_time = 1.0 - elapsed
                time.sleep(sleep_time)
            self.last_request_time = time.time()
            self.request_count = 0
    
    def get_db_connection(self):
        """Create a database connection"""
        return psycopg2.connect(self.db_url)
    
    def setup_database(self):
        """Create necessary database tables if they don't exist"""
        with self.get_db_connection() as conn:
            with conn.cursor() as cur:
                # Create scheduled_matches table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS scheduled_matches (
                        id SERIAL PRIMARY KEY,
                        match_id VARCHAR(50) UNIQUE,
                        tournament_id VARCHAR(50),
                        tournament_name VARCHAR(255),
                        surface VARCHAR(50),
                        tournament_level VARCHAR(50),
                        scheduled_date TIMESTAMP,
                        round VARCHAR(50),
                        player1_id INTEGER,
                        player1_name VARCHAR(255),
                        player2_id INTEGER,
                        player2_name VARCHAR(255),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        is_processed BOOLEAN DEFAULT FALSE
                    )
                """)
                
                # Create match_predictions table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS match_predictions (
                        id SERIAL PRIMARY KEY,
                        match_id VARCHAR(50),
                        player1_id INTEGER,
                        player2_id INTEGER,
                        player1_win_probability FLOAT,
                        prediction_date TIMESTAMP WITH TIME ZONE,
                        scheduled_date TIMESTAMP WITH TIME ZONE,
                        model_version VARCHAR(50),
                        features_used JSONB,
                        actual_winner_id INTEGER,
                        prediction_accuracy FLOAT,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (match_id) REFERENCES scheduled_matches(match_id)
                    )
                """)
                
                # Add indexes
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_scheduled_matches_date 
                    ON scheduled_matches(scheduled_date)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_match_predictions_date 
                    ON match_predictions(scheduled_date)
                """)
                
                conn.commit()
                logger.info("Database tables created successfully")
    
    def get_upcoming_tournaments(self, days_ahead: int = 7) -> List[Dict[str, Any]]:
        """
        Get tournaments scheduled in the next X days
        
        Args:
            days_ahead: Number of days to look ahead
            
        Returns:
            List of tournament dictionaries
        """
        self._apply_rate_limiting()
        
        # Calculate date range
        current_date = datetime.now()
        logger.info(f"Current date: {current_date}")
        end_date = current_date + timedelta(days=days_ahead)
        logger.info(f"End date: {end_date}")
        
        # Collect tournaments from both current year and last 30 days
        tournaments = []
        
        # Try current year first
        current_year = current_date.year
        logger.info(f"Fetching tournaments for current year: {current_year}")
        current_year_tournaments = self._fetch_tournaments_for_year(current_year)
        tournaments.extend(current_year_tournaments)

        # If we're in the first quarter of the year, also check previous year for ongoing tournaments
        if current_date.month <= 3:
            previous_year = current_year - 1
            logger.info(f"Fetching tournaments for previous year: {previous_year}")
            prev_year_tournaments = self._fetch_tournaments_for_year(previous_year)
            tournaments.extend(prev_year_tournaments)
            
        logger.info(f"Total tournaments fetched: {len(tournaments)}")
        
        # Filter tournaments: include those starting in our window OR 
        # those that might be in progress (started in the last 30 days)
        filtered_tournaments = []
        thirty_days_ago = current_date - timedelta(days=30)
        
        for tournament in tournaments:
            if 'date' not in tournament:
                continue
                
            try:
                tournament_date = datetime.strptime(
                    tournament['date'].split('T')[0], 
                    '%Y-%m-%d'
                )
                
                # Include tournament if:
                # 1. It starts within our window, OR
                # 2. It started within the last 30 days (might still have matches)
                if (current_date <= tournament_date <= end_date) or \
                   (thirty_days_ago <= tournament_date <= current_date):
                    filtered_tournaments.append(tournament)
                    logger.info(f"Including tournament: {tournament.get('name')} ({tournament_date.strftime('%Y-%m-%d')})")
            except (ValueError, TypeError) as e:
                logger.warning(f"Error parsing tournament date: {e}")
                continue
        
        logger.info(f"Found {len(filtered_tournaments)} tournaments in or near the specified date range")
        return filtered_tournaments
    
    def _fetch_tournaments_for_year(self, year: int) -> List[Dict[str, Any]]:
        """Helper method to fetch tournaments for a specific year"""
        url = f"{API_BASE_URL}/atp/tournament/calendar/{year}"
        params = {
            'pageNo': 1,
            'pageSize': PAGE_SIZE
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data or 'data' not in data:
                logger.warning(f"No tournament data received from API for year {year}")
                return []
                
            logger.info(f"Fetched {len(data['data'])} tournaments for year {year}")
            return data['data']
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching tournaments for year {year}: {e}")
            return []
    
    def get_tournament_fixtures(self, tournament_id: str) -> List[Dict[str, Any]]:
        """
        Get fixtures/matches for a specific tournament
        
        Args:
            tournament_id: Tournament ID to fetch fixtures for
            
        Returns:
            List of match dictionaries
        """
        self._apply_rate_limiting()
        
        url = f"{API_BASE_URL}/atp/fixtures/tournament/{tournament_id}"
        params = {
            'pageSize': PAGE_SIZE,
            'pageNo': 1,
            'filter': 'PlayerGroup:both;'
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data or 'data' not in data:
                logger.warning(f"No fixture data for tournament {tournament_id}")
                return []
            
            # Filter out doubles matches (where player names contain '/')
            singles_matches = [
                match for match in data['data']
                if not ('/' in match.get('player1', {}).get('name', '') or 
                       '/' in match.get('player2', {}).get('name', ''))
            ]
            
            logger.info(f"Found {len(singles_matches)} singles matches for tournament {tournament_id}")
            return singles_matches
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching fixtures for tournament {tournament_id}: {e}")
            return []
    
    def store_scheduled_match(self, conn, match: Dict[str, Any], tournament: Dict[str, Any]):
        """
        Store a scheduled match in the database
        
        Args:
            conn: Database connection
            match: Match data dictionary
            tournament: Tournament data dictionary
        """
        try:
            with conn.cursor() as cur:
                # Extract tournament info
                surface = tournament.get('court', {}).get('name', 'Unknown')
                tournament_level = tournament.get('round', {}).get('name', 'Unknown')
                
                # Prepare match data
                match_data = {
                    'match_id': str(match['id']),
                    'tournament_id': str(tournament['id']),
                    'tournament_name': tournament.get('name'),
                    'surface': surface,
                    'tournament_level': tournament_level,
                    'scheduled_date': match.get('date'),
                    'round': match.get('roundId'),
                    'player1_id': match.get('player1', {}).get('id'),
                    'player1_name': match.get('player1', {}).get('name'),
                    'player2_id': match.get('player2', {}).get('id'),
                    'player2_name': match.get('player2', {}).get('name'),
                    'is_processed': False
                }
                
                # Insert or update match
                cur.execute("""
                    INSERT INTO scheduled_matches (
                        match_id, tournament_id, tournament_name, surface,
                        tournament_level, scheduled_date, round,
                        player1_id, player1_name, player2_id, player2_name,
                        is_processed
                    ) VALUES (
                        %(match_id)s, %(tournament_id)s, %(tournament_name)s, %(surface)s,
                        %(tournament_level)s, %(scheduled_date)s, %(round)s,
                        %(player1_id)s, %(player1_name)s, %(player2_id)s, %(player2_name)s,
                        %(is_processed)s
                    )
                    ON CONFLICT (match_id) DO UPDATE SET
                        scheduled_date = EXCLUDED.scheduled_date,
                        round = EXCLUDED.round,
                        updated_at = CURRENT_TIMESTAMP
                """, match_data)
                
                conn.commit()
                logger.debug(f"Stored/updated match {match['id']}")
                
        except Exception as e:
            logger.error(f"Error storing match {match.get('id')}: {e}")
            conn.rollback()
    
    def cleanup_old_matches(self, conn):
        """Remove matches that are more than 7 days old"""
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM scheduled_matches
                    WHERE scheduled_date < NOW() - INTERVAL '7 days'
                """)
                deleted = cur.rowcount
                conn.commit()
                logger.info(f"Removed {deleted} old matches from scheduled_matches")
        except Exception as e:
            logger.error(f"Error cleaning up old matches: {e}")
            conn.rollback()
    
    def collect_matches(self, days_ahead: int = 7):
        """
        Main method to collect and store upcoming matches
        
        Args:
            days_ahead: Number of days to look ahead
        """
        try:
            # Setup database
            self.setup_database()
            
            # Get upcoming tournaments
            tournaments = self.get_upcoming_tournaments(days_ahead)
            
            with self.get_db_connection() as conn:
                # Clean up old matches
                self.cleanup_old_matches(conn)
                
                # Process each tournament
                for tournament in tournaments:
                    logger.info(f"Processing tournament: {tournament.get('name')}")
                    
                    # Get matches for tournament
                    matches = self.get_tournament_fixtures(tournament['id'])
                    
                    # Store each match
                    for match in matches:
                        self.store_scheduled_match(conn, match, tournament)
                
                logger.info("Match collection completed successfully")
                
        except Exception as e:
            logger.error(f"Error in match collection process: {e}")
            raise

def main():
    """Main execution function"""
    try:
        collector = TennisDataCollector()
        collector.collect_matches()
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 