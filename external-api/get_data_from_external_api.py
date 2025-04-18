"""
Tennis Data API Client for retrieving tournament calendars, results, match statistics, player profiles, and tournament information.

This module provides functionality to interact with the Tennis API to retrieve various data points needed for:
- Tournament calendar data and results
- Match statistics for specific players and tournaments
- Player profile information including physical attributes and playing style
- Detailed tournament information (level, surface, country, etc.)

The retrieved data is formatted for both human readability and database integration according to schema.py.
The script handles error cases, validates data, and provides detailed logging.

Usage:
    # Get tournament calendar data (default)
    python get_data_from_external_api.py
    
    # Get player profile
    python get_data_from_external_api.py player_profile <player_id>
    
    # Get match statistics
    python get_data_from_external_api.py match_stats <tournament_id> <player1_id> <player2_id>
    
    # Get tournament information
    python get_data_from_external_api.py tournament_info <tournament_id>

The output includes both human-readable formatted data and database-ready information that
can be stored according to the schema defined in schema.py.
"""

import os
import requests
import logging
import sys
import traceback
import json
from datetime import datetime, date, timedelta
from dotenv import load_dotenv
from typing import Dict, Any, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import argparse
import time
import http.client
import re
from tqdm import tqdm

# Date range for filtering tournaments (format: 'YYYY-MM-DD')
START_DATE = '2025-04-8'
END_DATE = '2025-04-20'

# Maximum allowed API requests per second across all API endpoints combined
API_REQUESTS_PER_SECOND = 7

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add file handler for warnings and errors
file_handler = logging.FileHandler('tennis_api_errors.log')
file_handler.setLevel(logging.WARNING)  # Only log WARNING and above (WARNING, ERROR, CRITICAL)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Get the root logger and add the file handler
root_logger = logging.getLogger()
root_logger.addHandler(file_handler)

# Load environment variables
load_dotenv()

def parse_arguments():
    """Parse command-line arguments for the script"""
    parser = argparse.ArgumentParser(description="Fetch tennis data from API and store in database")
    
    # Date range arguments
    parser.add_argument("--start-date", default=START_DATE, help=f"Start date for fetching tournaments (YYYY-MM-DD format). Default is {START_DATE}.")
    parser.add_argument("--end-date", default=END_DATE, help=f"End date for fetching tournaments (YYYY-MM-DD format). Default is {END_DATE}.")
    
    # Rate limiting
    parser.add_argument("--rate-limit", type=int, default=5, help="Maximum API requests per second (default: 5)")
    
    # Database options
    parser.add_argument("--db-url", help="Full database URL (overrides other DB parameters)")
    parser.add_argument("--db-host", default="localhost", help="Database host (default: localhost)")
    parser.add_argument("--db-port", default="5432", help="Database port (default: 5432)")
    parser.add_argument("--db-name", default="tennis", help="Database name (default: tennis)")
    parser.add_argument("--db-user", default="postgres", help="Database username (default: postgres)")
    parser.add_argument("--db-password", default="postgres", help="Database password (default: postgres)")
    
    # Other options
    parser.add_argument("--skip-db-write", action="store_true", help="Skip writing data to database")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed API response logging")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--page-size", type=int, default=1000, help="Number of results per page (default: 1000)")
    parser.add_argument("--save-responses", action="store_true", help="Save API responses to files")
    parser.add_argument("--no-progress", action="store_true", help="Hide progress bar (default: show)")
    
    return parser.parse_args()

class TennisAPIClient:
    def __init__(self, quiet_mode=False):
        """
        Initialize the TennisAPIClient with API settings.
        
        Args:
            quiet_mode: If True, reduces logging output
        """
        self.quiet_mode = quiet_mode
        self.base_url = "https://tennis-api-atp-wta-itf.p.rapidapi.com/tennis/v2"
        
        # Get API key from environment variable or fallback to default
        self.api_key = os.environ.get("TENNIS_API_KEY", "3d76672992msha995d7c3e1002b8p15659fjsn74f449524272")
        
        # Set headers for API requests
        self.headers = {
            'x-rapidapi-host': "tennis-api-atp-wta-itf.p.rapidapi.com",
            'x-rapidapi-key': self.api_key
        }
        
        # Rate limiting variables
        self.last_request_time = time.time()
        self.request_count = 0
        
        logging.info(f"TennisAPIClient initialized with API key: {self.api_key[:5]}...")
        
    def _apply_rate_limiting(self):
        """
        Apply rate limiting to ensure we don't exceed API_REQUESTS_PER_SECOND.
        """
        self.request_count += 1
        if self.request_count >= API_REQUESTS_PER_SECOND:
            elapsed = time.time() - self.last_request_time
            if elapsed < 1.0:
                sleep_time = 1.0 - elapsed
                if not self.quiet_mode:
                    logging.debug(f"Rate limiting: sleeping for {sleep_time:.3f} seconds")
                time.sleep(sleep_time)
            self.last_request_time = time.time()
            self.request_count = 0
        
        # If it's been more than 1 second since our last reset, reset the counter
        if time.time() - self.last_request_time > 1.0:
            self.last_request_time = time.time()
            self.request_count = 1

    def get_tournament_calendar(self, start_date: str, end_date: str, page: int = 1, page_size: int = 1000) -> Optional[Dict[str, Any]]:
        """
        Fetch tournament calendar data for a specific date range.
        
        Args:
            start_date: The start date to fetch data for (format: 'YYYY-MM-DD')
            end_date: The end date to fetch data for (format: 'YYYY-MM-DD')
            page: Page number for pagination
            page_size: Number of results per page
            
        Returns:
            Dictionary containing the API response data or None if request fails
        """
        try:
            # Apply rate limiting
            self._apply_rate_limiting()
            
            # Extract year from start_date - if provided
            year = datetime.now().year
            if start_date:
                year = start_date.split('-')[0]
                
            url = f"{self.base_url}/atp/tournament/calendar/{year}"
            params = {
                'pageNo': page,
                'pageSize': page_size
            }
            
            logging.info(f"Fetching tournament calendar data for year {year}")
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not self.quiet_mode:
                logging.debug(f"Calendar API Response: {json.dumps(data, indent=2)}")
            else:
                logging.info(f"Received calendar data successfully")
                
            return data
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching tournament data: {str(e)}")
            return None

    def get_tournament_results(self, tournament_id: str, page_size: int = 1000) -> Optional[Dict[str, Any]]:
        """
        Fetch results for a specific tournament.
        
        Args:
            tournament_id: The ID of the tournament to fetch results for
            page_size: Number of results per page
            
        Returns:
            Dictionary containing the tournament results or None if request fails
        """
        try:
            # Apply rate limiting
            self._apply_rate_limiting()
            
            url = f"{self.base_url}/atp/tournament/results/{tournament_id}"
            params = {'pageSize': page_size}
            
            logging.info(f"Fetching results for tournament ID: {tournament_id}")
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not self.quiet_mode:
                logging.debug(f"Tournament Results API Response: {json.dumps(data, indent=2)}")
            else:
                logging.info(f"Received tournament results successfully")
                
            return data
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching tournament results: {str(e)}")
            return None

    def get_match_stats(self, tournament_id: str, player1_id: str, player2_id: str) -> Optional[Dict[str, Any]]:
        """
        Get match statistics from the API.
        
        Args:
            tournament_id: Tournament ID
            player1_id: Player 1 ID
            player2_id: Player 2 ID
            
        Returns:
            Optional[Dict]: Match statistics if successful, None otherwise
        """
        logging.info(f"Fetching match statistics for tournament {tournament_id}, players {player1_id} vs {player2_id}")
        
        try:
            # Apply rate limiting
            self._apply_rate_limiting()
            
            url = f"{self.base_url}/atp/h2h/match-stats/{tournament_id}/{player1_id}/{player2_id}"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            match_stats = response.json()
            
            # Log just a brief message instead of the entire response
            if not self.quiet_mode:
                logging.debug(f"Match Stats API Response: Retrieved stats for tournament {tournament_id}, players {player1_id} vs {player2_id}")
            
            return match_stats
        except Exception as e:
            logging.error(f"Error getting match statistics: {str(e)}")
            return None

    def get_player_profile(self, player_id: str) -> Optional[Dict[str, Any]]:
        """
        Get player profile from the API.
        
        Args:
            player_id: Player ID to fetch
            
        Returns:
            Optional[Dict]: Player profile data if successful, None otherwise
        """
        logging.info(f"Fetching player profile for player ID: {player_id}")
        
        try:
            # Apply rate limiting
            self._apply_rate_limiting()
            
            url = f"{self.base_url}/atp/player/profile/{player_id}"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            player_profile = response.json()
            
            # Log just a brief message instead of the entire response
            if not self.quiet_mode:
                logging.debug(f"Player Profile API Response: Retrieved data for player ID {player_id}")
            
            return player_profile
        except Exception as e:
            logging.error(f"Error getting player profile: {str(e)}")
            return None

    def get_tournament_info(self, tournament_id: int) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific tournament by ID.
        
        Args:
            tournament_id (int): ID of the tournament to fetch
            
        Returns:
            dict: Tournament information data, or None if not found
        """
        try:
            # Apply rate limiting
            self._apply_rate_limiting()
            
            url = f"{self.base_url}/atp/tournament/info/{tournament_id}"
            
            logging.info(f"Fetching tournament info for tournament ID: {tournament_id}")
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            
            if not self.quiet_mode:
                logging.debug(f"Tournament Info API Response: {json.dumps(data, indent=2)}")
            else:
                logging.info(f"Received tournament info successfully")
                
            return data
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to get tournament info for ID: {tournament_id}: {str(e)}")
            return None

def print_tournament_info(tournament_info: dict) -> None:
    """
    Print tournament information in a readable format and show database-ready mapping.
    
    Args:
        tournament_info (dict): Tournament info API response
    """
    if not tournament_info or "data" not in tournament_info:
        print("No tournament information available")
        return
    
    logging.info(f"Tournament Info API Response structure: {tournament_info.keys()}")
    
    data = tournament_info.get("data", {})
    
    if not data:
        print("Tournament data is empty")
        return
    
    # Print formatted tournament information
    print("\nTournament Information:")
    print("-" * 100)
    
    print(f"Tournament ID       : {data.get('id')}")
    print(f"Tournament Name     : {data.get('name')}")
    print(f"Date                : {data.get('date')}")
    
    # Extract court information
    court_info = data.get("court", {})
    court_id = court_info.get("id") if court_info else None
    court_name = court_info.get("name") if court_info else None
    print(f"Court ID            : {court_id}")
    print(f"Court Name/Surface  : {court_name}")
    
    # Extract tournament level/round information
    round_info = data.get("round", {})
    round_id = round_info.get("id") if round_info else None
    round_name = round_info.get("name") if round_info else None
    print(f"Tournament Level ID : {round_id}")
    print(f"Tournament Level    : {round_name}")
    
    # Extract country information
    country_info = data.get("coutry", {})  # Note: API uses "coutry" not "country"
    country_code = country_info.get("acronym") if country_info else None
    country_name = country_info.get("name") if country_info else None
    print(f"Country Code        : {country_code}")
    print(f"Country Name        : {country_name}")
    
    # Extract rank information if available
    rank_id = data.get("rankId")
    print(f"Rank ID             : {rank_id}")
    
    # Map to database schema
    print("\nDatabase-Ready Tournament Information:")
    print("-" * 100)
    
    # Based on schema.py, map to tournaments table columns
    print(f"tournament_id       : {data.get('id')}")
    print(f"tournament_name     : {data.get('name')}")
    
    # Parse date to format needed
    tournament_date = data.get('date', '').split('T')[0] if data.get('date') else None
    print(f"tournament_date     : {tournament_date}")
    
    # Map surface to database format
    surface_mapping = {
        "Hard": "hard",
        "Clay": "clay", 
        "Grass": "grass",
        "Carpet": "carpet",
        "Indoor Hard": "indoor_hard"
    }
    surface = surface_mapping.get(court_name, "unknown") if court_name else "unknown"
    print(f"tournament_surface  : {surface}")
    
    # Map tournament level
    level_mapping = {
        "Grand Slam": "grand_slam",
        "Masters 1000": "atp_1000",
        "500": "atp_500",
        "250": "atp_250",
        "Challenger": "challenger"
    }
    tournament_level = level_mapping.get(round_name, "unknown") if round_name else "unknown"
    print(f"tournament_level    : {tournament_level}")
    
    # Other fields in schema not available directly from this endpoint
    print(f"tournament_country  : {country_code}")

def get_tournament_info_for_db(tournament_info: dict) -> dict:
    """
    Extract and format tournament information for database storage.
    
    Args:
        tournament_info (dict): Tournament info API response
        
    Returns:
        dict: Tournament information formatted for database
    """
    if not tournament_info or "data" not in tournament_info:
        return {}
    
    data = tournament_info.get("data", {})
    
    if not data:
        return {}
    
    # Extract court information
    court_info = data.get("court", {})
    court_name = court_info.get("name") if court_info else None
    
    # Extract tournament level/round information
    round_info = data.get("round", {})
    round_name = round_info.get("name") if round_info else None
    
    # Extract country information
    country_info = data.get("coutry", {})  # Note: API uses "coutry" not "country"
    country_code = country_info.get("acronym") if country_info else None
    
    # Parse date to format needed
    tournament_date = data.get('date', '').split('T')[0] if data.get('date') else None
    
    # Map surface to database format
    surface_mapping = {
        "Hard": "hard",
        "Clay": "clay", 
        "Grass": "grass",
        "Carpet": "carpet",
        "Indoor Hard": "indoor_hard"
    }
    surface = surface_mapping.get(court_name, "unknown") if court_name else "unknown"
    
    # Map tournament level
    level_mapping = {
        "Grand Slam": "grand_slam",
        "Masters 1000": "atp_1000",
        "500": "atp_500",
        "250": "atp_250",
        "Challenger": "challenger"
    }
    tournament_level = level_mapping.get(round_name, "unknown") if round_name else "unknown"
    
    # Create database-ready dict
    db_tournament = {
        "tournament_id": data.get('id'),
        "tournament_name": data.get('name'),
        "tournament_date": tournament_date,
        "tournament_surface": surface,
        "tournament_level": tournament_level,
        "tournament_country": country_code
    }
    
    return db_tournament

def print_match_stats(data: Dict[str, Any]) -> None:
    """Print the match statistics in a formatted way"""
    if not data:
        logging.warning("No match statistics available")
        return
    
    # Log response structure for debugging
    logging.info(f"Match Stats API Response structure: {data.keys() if isinstance(data, dict) else 'Not a dictionary'}")
    
    # Check if we have valid data
    if not isinstance(data, dict) or 'data' not in data:
        logging.warning("Invalid or empty match statistics data")
        return
        
    # Extract match statistics
    match_data = data.get('data', {})
    
    print("\nMatch Statistics:")
    print("-" * 100)
    
    # Print player statistics based on the actual API response structure
    player1_stats = match_data.get('player1Stats', {})
    player2_stats = match_data.get('player2Stats', {})
    
    if player1_stats and player2_stats:
        # Using player IDs since we don't have names in the response
        player1_id = player1_stats.get('player1Id', 'Player 1')
        player2_id = player2_stats.get('player2Id', 'Player 2')
        
        print(f"{'Statistic':<30} {'Player ' + str(player1_id):<20} {'Player ' + str(player2_id):<20}")
        print("-" * 100)
        
        # Define the statistics to display based on actual API response fields
        stat_mapping = [
            ('aces', 'Aces'),
            ('doubleFaults', 'Double Faults'),
            ('firstServe', 'First Serves In'),
            ('firstServeOf', 'First Serves Total'),
            ('winningOnFirstServe', 'First Serve Points Won'),
            ('winningOnFirstServeOf', 'First Serve Points Total'),
            ('winningOnSecondServe', 'Second Serve Points Won'),
            ('winningOnSecondServeOf', 'Second Serve Points Total'),
            ('breakPointFacedGm', 'Break Points Faced'),
            ('breakPointSavedGm', 'Break Points Saved'),
            ('breakPointChanceGm', 'Break Point Chances'),
            ('breakPointWonGm', 'Break Points Won')
        ]
        
        # Print each statistic
        for api_key, display_name in stat_mapping:
            p1_stat = player1_stats.get(api_key, 'N/A')
            p2_stat = player2_stats.get(api_key, 'N/A')
            print(f"{display_name:<30} {p1_stat:<20} {p2_stat:<20}")
        
        # Calculate and print additional derived statistics
        if isinstance(player1_stats.get('firstServe'), (int, float)) and isinstance(player1_stats.get('firstServeOf'), (int, float)) and player1_stats.get('firstServeOf') > 0:
            p1_first_serve_pct = round(player1_stats.get('firstServe') / player1_stats.get('firstServeOf') * 100, 1)
        else:
            p1_first_serve_pct = 'N/A'
            
        if isinstance(player2_stats.get('firstServe'), (int, float)) and isinstance(player2_stats.get('firstServeOf'), (int, float)) and player2_stats.get('firstServeOf') > 0:
            p2_first_serve_pct = round(player2_stats.get('firstServe') / player2_stats.get('firstServeOf') * 100, 1)
        else:
            p2_first_serve_pct = 'N/A'
            
        print(f"{'First Serve %':<30} {p1_first_serve_pct:<20} {p2_first_serve_pct:<20}")
        
        # Calculate first serve win percentage
        if isinstance(player1_stats.get('winningOnFirstServe'), (int, float)) and isinstance(player1_stats.get('firstServe'), (int, float)) and player1_stats.get('firstServe') > 0:
            p1_first_serve_win_pct = round(player1_stats.get('winningOnFirstServe') / player1_stats.get('firstServe') * 100, 1)
        else:
            p1_first_serve_win_pct = 'N/A'
            
        if isinstance(player2_stats.get('winningOnFirstServe'), (int, float)) and isinstance(player2_stats.get('firstServe'), (int, float)) and player2_stats.get('firstServe') > 0:
            p2_first_serve_win_pct = round(player2_stats.get('winningOnFirstServe') / player2_stats.get('firstServe') * 100, 1)
        else:
            p2_first_serve_win_pct = 'N/A'
            
        print(f"{'First Serve Win %':<30} {p1_first_serve_win_pct:<20} {p2_first_serve_win_pct:<20}")
        
        # Calculate second serve win percentage
        if isinstance(player1_stats.get('winningOnSecondServe'), (int, float)) and isinstance(player1_stats.get('winningOnSecondServeOf'), (int, float)) and player1_stats.get('winningOnSecondServeOf') > 0:
            p1_second_serve_win_pct = round(player1_stats.get('winningOnSecondServe') / player1_stats.get('winningOnSecondServeOf') * 100, 1)
        else:
            p1_second_serve_win_pct = 'N/A'
            
        if isinstance(player2_stats.get('winningOnSecondServe'), (int, float)) and isinstance(player2_stats.get('winningOnSecondServeOf'), (int, float)) and player2_stats.get('winningOnSecondServeOf') > 0:
            p2_second_serve_win_pct = round(player2_stats.get('winningOnSecondServe') / player2_stats.get('winningOnSecondServeOf') * 100, 1)
        else:
            p2_second_serve_win_pct = 'N/A'
            
        print(f"{'Second Serve Win %':<30} {p1_second_serve_win_pct:<20} {p2_second_serve_win_pct:<20}")
        
        # Calculate break point conversion percentage
        if isinstance(player1_stats.get('breakPointWonGm'), (int, float)) and isinstance(player1_stats.get('breakPointChanceGm'), (int, float)) and player1_stats.get('breakPointChanceGm') > 0:
            p1_bp_conversion = round(player1_stats.get('breakPointWonGm') / player1_stats.get('breakPointChanceGm') * 100, 1)
        else:
            p1_bp_conversion = 'N/A'
            
        if isinstance(player2_stats.get('breakPointWonGm'), (int, float)) and isinstance(player2_stats.get('breakPointChanceGm'), (int, float)) and player2_stats.get('breakPointChanceGm') > 0:
            p2_bp_conversion = round(player2_stats.get('breakPointWonGm') / player2_stats.get('breakPointChanceGm') * 100, 1)
        else:
            p2_bp_conversion = 'N/A'
            
        print(f"{'Break Point Conversion %':<30} {p1_bp_conversion:<20} {p2_bp_conversion:<20}")
        
        # Call function to calculate and print additional serve/return stats
        print_calculated_serve_return_stats(player1_stats, player2_stats, player1_id, player2_id)
    else:
        print("No detailed player statistics available")

def print_calculated_serve_return_stats(player1_stats: Dict[str, Any], player2_stats: Dict[str, Any], 
                                       player1_id: Any, player2_id: Any) -> None:
    """
    Calculate and print additional serve and return statistics derived from the match stats API data.
    These calculations match those needed for the feature generation pipeline.
    
    Args:
        player1_stats: Dictionary of player 1 statistics
        player2_stats: Dictionary of player 2 statistics
        player1_id: ID of player 1
        player2_id: ID of player 2
    """
    print("\nCalculated Serve/Return Statistics (for Feature Generation):")
    print("-" * 100)
    print(f"{'Statistic':<30} {'Player ' + str(player1_id):<20} {'Player ' + str(player2_id):<20}")
    print("-" * 100)
    
    # Calculate total serve points (if not directly available)
    p1_serve_points = player1_stats.get('firstServeOf', 0) or 0
    p2_serve_points = player2_stats.get('firstServeOf', 0) or 0
    
    # Calculate total points won on serve
    p1_first_serve_points_won = player1_stats.get('winningOnFirstServe', 0) or 0
    p2_first_serve_points_won = player2_stats.get('winningOnFirstServe', 0) or 0
    p1_second_serve_points_won = player1_stats.get('winningOnSecondServe', 0) or 0
    p2_second_serve_points_won = player2_stats.get('winningOnSecondServe', 0) or 0
    p1_serve_points_won = p1_first_serve_points_won + p1_second_serve_points_won
    p2_serve_points_won = p2_first_serve_points_won + p2_second_serve_points_won
    
    # 1. Serve Efficiency (% of service points won)
    if p1_serve_points > 0:
        p1_serve_efficiency = round(p1_serve_points_won / p1_serve_points * 100, 1)
    else:
        p1_serve_efficiency = 'N/A'
        
    if p2_serve_points > 0:
        p2_serve_efficiency = round(p2_serve_points_won / p2_serve_points * 100, 1)
    else:
        p2_serve_efficiency = 'N/A'
    
    print(f"{'Serve Efficiency %':<30} {p1_serve_efficiency:<20} {p2_serve_efficiency:<20}")
    
    # 2. Ace Percentage (aces per service point)
    if p1_serve_points > 0:
        p1_ace_pct = round(player1_stats.get('aces', 0) / p1_serve_points * 100, 1)
    else:
        p1_ace_pct = 'N/A'
        
    if p2_serve_points > 0:
        p2_ace_pct = round(player2_stats.get('aces', 0) / p2_serve_points * 100, 1)
    else:
        p2_ace_pct = 'N/A'
    
    print(f"{'Ace %':<30} {p1_ace_pct:<20} {p2_ace_pct:<20}")
    
    # 3. Double Fault Percentage (double faults per service point)
    if p1_serve_points > 0:
        p1_df_pct = round(player1_stats.get('doubleFaults', 0) / p1_serve_points * 100, 1)
    else:
        p1_df_pct = 'N/A'
        
    if p2_serve_points > 0:
        p2_df_pct = round(player2_stats.get('doubleFaults', 0) / p2_serve_points * 100, 1)
    else:
        p2_df_pct = 'N/A'
    
    print(f"{'Double Fault %':<30} {p1_df_pct:<20} {p2_df_pct:<20}")
    
    # 4. Break Points Saved Percentage
    p1_bp_faced = player1_stats.get('breakPointFacedGm', 0) or 0
    p2_bp_faced = player2_stats.get('breakPointFacedGm', 0) or 0
    
    if p1_bp_faced > 0:
        p1_bp_saved_pct = round(player1_stats.get('breakPointSavedGm', 0) / p1_bp_faced * 100, 1)
    else:
        p1_bp_saved_pct = 'N/A'
        
    if p2_bp_faced > 0:
        p2_bp_saved_pct = round(player2_stats.get('breakPointSavedGm', 0) / p2_bp_faced * 100, 1)
    else:
        p2_bp_saved_pct = 'N/A'
    
    print(f"{'Break Points Saved %':<30} {p1_bp_saved_pct:<20} {p2_bp_saved_pct:<20}")
    
    # 5. Return Points Won (calculated from opponent's serve points)
    p1_return_points = p2_serve_points
    p2_return_points = p1_serve_points
    
    # Player's return points won is equal to opponent's serve points minus opponent's serve points won
    p1_return_points_won = p1_return_points - p2_serve_points_won
    p2_return_points_won = p2_return_points - p1_serve_points_won
    
    # 6. Return Efficiency (% of return points won)
    if p1_return_points > 0:
        p1_return_efficiency = round(p1_return_points_won / p1_return_points * 100, 1)
    else:
        p1_return_efficiency = 'N/A'
        
    if p2_return_points > 0:
        p2_return_efficiency = round(p2_return_points_won / p2_return_points * 100, 1)
    else:
        p2_return_efficiency = 'N/A'
    
    print(f"{'Return Efficiency %':<30} {p1_return_efficiency:<20} {p2_return_efficiency:<20}")
    
    # 7. Service Games (estimate if not available)
    # Note: This is an approximation since we don't have actual service games data
    # In a typical tennis match, each player normally serves around every other game
    # This is a rough estimate based on total points
    estimated_games = (p1_serve_points + p2_serve_points) / 10  # Rough estimate: ~10 points per game on average
    p1_service_games = round(estimated_games / 2)  # Approximate half the games as service games
    p2_service_games = round(estimated_games / 2)
    
    print(f"{'Estimated Service Games':<30} {p1_service_games:<20} {p2_service_games:<20}")
    
    # Print a note about the calculations
    print("\nNote: These calculations match the formulas used in the feature generation pipeline.")
    print("      Return points and service games are estimated based on available data.")
    print("      These serve/return stats can be stored in the database for feature generation.")

def format_player_hand(hand: str) -> str:
    """
    Format player hand data from API to match database format (R/L/U).
    
    Args:
        hand: Hand data from API (e.g., "Right-Handed, Two-Handed Backhand")
        
    Returns:
        Formatted hand data (R, L, or None for unknown)
    """
    if not hand:
        # Return None instead of defaulting to R
        return None
        
    hand_lower = hand.lower() if isinstance(hand, str) else ''
    
    # Check for specific patterns in the 'plays' field
    if 'right-handed' in hand_lower or 'right handed' in hand_lower:
        return 'R'
    elif 'left-handed' in hand_lower or 'left handed' in hand_lower:
        return 'L'
    # Generic check for right/left
    elif 'right' in hand_lower:
        return 'R'
    elif 'left' in hand_lower:
        return 'L'
    
    # Return None instead of defaulting to right-handed
    return None

def format_tournament_level(level: str) -> str:
    """
    Convert tournament level to database format.
    
    Args:
        level: Tournament level string from API
        
    Returns:
        str: Single character level code
    """
    level_mapping = {
        "Grand Slam": "G",
        "Masters 1000": "M",
        "500": "F",
        "250": "T",
        "Challenger": "C"
    }
    return level_mapping.get(level, "C")  # Default to Challenger if unknown

def format_surface(surface: str) -> str:
    """
    Format surface to match database format.
    
    Args:
        surface: Surface string from API
        
    Returns:
        str: Formatted surface string
    """
    surface_mapping = {
        "Hard": "Hard",
        "Clay": "Clay",
        "Grass": "Grass",
        "Carpet": "Carpet",
        "Indoor Hard": "Indoor Hard"
    }
    return surface_mapping.get(surface, "Unknown")

def format_height(height_str: str) -> Optional[int]:
    """
    Format height data from API to centimeters value.
    
    Args:
        height_str: Height data from API
        
    Returns:
        Height in centimeters or None if not available
    """
    if not height_str:
        return None
        
    try:
        # Handle if height is already a number
        if isinstance(height_str, (int, float)):
            return int(height_str)
            
        # Check for cm format (e.g., "188 cm")
        if 'cm' in height_str.lower():
            cm_match = re.search(r'(\d+)\s*cm', height_str.lower())
            if cm_match:
                return int(cm_match.group(1))
        
        # Check for feet/inches format (e.g., "6'1\"")
        ft_in_match = re.search(r"(\d+)'(\d+)\"", height_str)
        if ft_in_match:
            feet = int(ft_in_match.group(1))
            inches = int(ft_in_match.group(2))
            return int((feet * 30.48) + (inches * 2.54))  # Convert to cm
            
        # Check for feet-inches format (e.g., "6-1")
        ft_in_dash = re.search(r"(\d+)-(\d+)", height_str)
        if ft_in_dash:
            feet = int(ft_in_dash.group(1))
            inches = int(ft_in_dash.group(2))
            return int((feet * 30.48) + (inches * 2.54))  # Convert to cm
            
        # Check for meters format (e.g., "1.88m")
        meters_match = re.search(r"(\d+\.\d+)\s*m", height_str.lower())
        if meters_match:
            meters = float(meters_match.group(1))
            return int(meters * 100)  # Convert to cm
            
        # Check if it's just a plain number (likely cm)
        if height_str.strip().isdigit():
            height_value = int(height_str.strip())
            # If it's a reasonable height in cm (130-220 cm is a normal human height range)
            if 130 <= height_value <= 220:
                return height_value
                
        # Log unrecognized format for debugging
        logging.debug(f"Unrecognized height format: {height_str}")
        return None
    except Exception as e:
        logging.debug(f"Error formatting height '{height_str}': {e}")
        return None

def calculate_player_age(birth_date: str, match_date: str) -> Optional[float]:
    """
    Calculate player age in decimal years at match date.
    
    Args:
        birth_date: Player birth date string
        match_date: Match date string
        
    Returns:
        Optional[float]: Player age in decimal years or None if invalid
    """
    if not birth_date or not match_date:
        return None
    try:
        birth = datetime.strptime(birth_date.split('T')[0], '%Y-%m-%d')
        match = datetime.strptime(match_date.split('T')[0], '%Y-%m-%d')
        
        # Calculate age in days
        age_days = (match - birth).days
        
        # Convert to decimal years
        age_years = age_days / 365.25  # Account for leap years
        
        return round(age_years, 1)  # Round to 1 decimal place
    except (ValueError, TypeError):
        return None

def get_calculated_serve_return_stats_for_db(player1_stats: Dict[str, Any], player2_stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate serve and return statistics derived from the match stats API data
    and return them in a format that can be used for database storage.
    Returns None/NULL for missing or empty statistics instead of zeros.
    
    Args:
        player1_stats: Dictionary of player 1 statistics (winner)
        player2_stats: Dictionary of player 2 statistics (loser)
        
    Returns:
        Dictionary with calculated stats for database storage, using None for missing values
    """
    # Ensure stats dictionaries are never None
    player1_stats = player1_stats or {}
    player2_stats = player2_stats or {}
    
    result = {}
    
    # Helper function to safely get numeric value or None
    def safe_get_number(stats: dict, key: str) -> Optional[float]:
        value = stats.get(key)
        # Return None for any falsy value (None, 0, empty string, etc)
        # except actual zero from API
        if value is None or value == '' or (isinstance(value, str) and value.strip() == ''):
            return None
        try:
            num_value = float(value)
            # If the value is explicitly 0 in the API response, keep it
            # Otherwise, treat 0 as None (missing data)
            if num_value == 0 and key not in stats:
                return None
            return num_value
        except (ValueError, TypeError):
            return None
    
    # Calculate basic stats for player 1 (winner)
    p1_serve_points = safe_get_number(player1_stats, 'firstServeOf')
    p1_first_serve_points_won = safe_get_number(player1_stats, 'winningOnFirstServe')
    p1_second_serve_points_won = safe_get_number(player1_stats, 'winningOnSecondServe')
    p1_bp_faced = safe_get_number(player1_stats, 'breakPointFacedGm')
    p1_bp_saved = safe_get_number(player1_stats, 'breakPointSavedGm')
    
    # Calculate basic stats for player 2 (loser)
    p2_serve_points = safe_get_number(player2_stats, 'firstServeOf')
    p2_first_serve_points_won = safe_get_number(player2_stats, 'winningOnFirstServe')
    p2_second_serve_points_won = safe_get_number(player2_stats, 'winningOnSecondServe')
    p2_bp_faced = safe_get_number(player2_stats, 'breakPointFacedGm')
    p2_bp_saved = safe_get_number(player2_stats, 'breakPointSavedGm')
    
    # Calculate serve points won only if both components are available
    p1_serve_points_won = None
    if p1_first_serve_points_won is not None and p1_second_serve_points_won is not None:
        p1_serve_points_won = p1_first_serve_points_won + p1_second_serve_points_won
    
    p2_serve_points_won = None
    if p2_first_serve_points_won is not None and p2_second_serve_points_won is not None:
        p2_serve_points_won = p2_first_serve_points_won + p2_second_serve_points_won
    
    # Estimate service games if we have serve points
    p1_service_games = None
    p2_service_games = None
    if p1_serve_points is not None and p2_serve_points is not None:
        estimated_games = (p1_serve_points + p2_serve_points) / 10  # Rough estimate: ~10 points per game on average
        if estimated_games > 0:
            p1_service_games = round(estimated_games / 2)  # Approximate half the games as service games
            p2_service_games = round(estimated_games / 2)
    
    # Map to database column names
    # Winner stats
    result['winner_aces'] = safe_get_number(player1_stats, 'aces')
    result['winner_double_faults'] = safe_get_number(player1_stats, 'doubleFaults')
    result['winner_serve_points'] = p1_serve_points
    result['winner_first_serves_in'] = safe_get_number(player1_stats, 'firstServe')
    result['winner_first_serve_points_won'] = p1_first_serve_points_won
    result['winner_second_serve_points_won'] = p1_second_serve_points_won
    result['winner_service_games'] = p1_service_games
    result['winner_break_points_saved'] = p1_bp_saved
    result['winner_break_points_faced'] = p1_bp_faced
    
    # Loser stats
    result['loser_aces'] = safe_get_number(player2_stats, 'aces')
    result['loser_double_faults'] = safe_get_number(player2_stats, 'doubleFaults')
    result['loser_serve_points'] = p2_serve_points
    result['loser_first_serves_in'] = safe_get_number(player2_stats, 'firstServe')
    result['loser_first_serve_points_won'] = p2_first_serve_points_won
    result['loser_second_serve_points_won'] = p2_second_serve_points_won
    result['loser_service_games'] = p2_service_games
    result['loser_break_points_saved'] = p2_bp_saved
    result['loser_break_points_faced'] = p2_bp_faced
    
    # Check if we have any valid statistics
    has_valid_stats = any(v is not None for v in result.values())
    
    # If no valid statistics found, return empty dict so no stats will be inserted
    return result if has_valid_stats else {}

def print_player_profile(data: Dict[str, Any]) -> None:
    """Print the player profile information in a formatted way"""
    if not data:
        logging.warning("No player profile data available")
        return
    
    # Log response structure for debugging
    logging.info(f"Player Profile API Response structure: {data.keys() if isinstance(data, dict) else 'Not a dictionary'}")
    
    # Check if we have valid data
    if not isinstance(data, dict) or 'data' not in data:
        logging.warning("Invalid or empty player profile data")
        return
        
    # Extract player profile data
    player_data = data.get('data', {})
    
    print("\nPlayer Profile:")
    print("-" * 100)
    
    # Extract basic information
    player_id = player_data.get('id', 'N/A')
    player_name = player_data.get('name', 'N/A')
    player_status = player_data.get('playerStatus', 'N/A')
    
    # Extract country information
    country_info = player_data.get('country', {})
    country_name = country_info.get('name', 'N/A')
    country_code = country_info.get('acronym', player_data.get('countryAcr', 'N/A'))
    
    # Extract detailed information - ensure information is never None
    information = player_data.get('information', {}) or {}
    turned_pro = information.get('turnedPro', 'N/A')
    weight = information.get('weight', 'N/A')
    height = information.get('height', 'N/A')
    birthplace = information.get('birthplace', 'N/A')
    residence = information.get('residence', 'N/A')
    plays = information.get('plays', 'N/A')
    coach = information.get('coach', 'N/A')
    
    # Print basic player information
    print(f"ID: {player_id}")
    print(f"Name: {player_name}")
    print(f"Status: {player_status}")
    print(f"Country: {country_code} - {country_name}")
    
    # Print detailed player information
    print("\nPhysical Information:")
    print(f"Height: {height}")
    print(f"Weight: {weight}")
    
    # Extract playing hand from the 'plays' field
    hand = "Right-Handed" if plays and "Right-Handed" in plays else "Left-Handed" if plays and "Left-Handed" in plays else "N/A"
    print(f"Plays: {plays}")
    print(f"Hand: {hand}")
    
    print("\nPersonal Information:")
    print(f"Birthplace: {birthplace}")
    print(f"Residence: {residence}")
    
    print("\nProfessional Information:")
    print(f"Turned Pro: {turned_pro}")
    print(f"Coach: {coach}")
    
    # Extract database-compatible values
    # Extract height in cm
    height_cm = None
    if height and isinstance(height, str):
        import re
        cm_match = re.search(r'(\d+)\s*cm', height)
        if cm_match:
            try:
                height_cm = int(cm_match.group(1))
            except ValueError:
                height_cm = None
    
    # Extract weight in kg
    weight_kg = None
    if weight and isinstance(weight, str):
        import re
        kg_match = re.search(r'(\d+)\s*kg', weight)
        if kg_match:
            try:
                weight_kg = int(kg_match.group(1))
            except ValueError:
                weight_kg = None
    
    # Print database-ready values
    print("\nDatabase-Compatible Values:")
    print(f"player_id: {player_id}")
    print(f"player_name: {player_name}")
    print(f"player_country_code: {country_code}")
    print(f"player_hand: {'R' if hand and 'Right' in hand else 'L' if hand and 'Left' in hand else 'N/A'}")
    print(f"player_height_cm: {height_cm}")
    print(f"player_weight_kg: {weight_kg}")
    
    print("-" * 100)

def print_tournaments(calendar_data: Dict[str, Any]) -> None:
    """
    Print tournament calendar data in a formatted way.
    
    Args:
        calendar_data (dict): Tournament calendar API response
    """
    if not calendar_data:
        logging.warning("No tournament calendar data available")
        return
    
    # Log response structure for debugging
    logging.info(f"Calendar API Response structure: {calendar_data.keys() if isinstance(calendar_data, dict) else 'Not a dictionary'}")
    
    # Check if we have valid data
    if not isinstance(calendar_data, dict) or 'data' not in calendar_data:
        logging.warning("Invalid or empty calendar data")
        return
        
    # Extract tournaments list
    data = calendar_data.get('data', [])
    
    # Handle case where data is directly a list of tournaments
    if isinstance(data, list):
        all_tournaments = data
    else:
        # Handle case where data is a dictionary with a tournaments key
        all_tournaments = data.get('tournaments', [])
    
    if not all_tournaments:
        print("No tournaments found in the specified date range")
        return
    
    # Filter tournaments by the specified date range
    filtered_tournaments = []
    start_date_obj = datetime.strptime(START_DATE, '%Y-%m-%d').date()
    end_date_obj = datetime.strptime(END_DATE, '%Y-%m-%d').date()

    for tournament in all_tournaments:
        tournament_date_str = tournament.get('date', '').split('T')[0] if tournament.get('date') else None
        
        if tournament_date_str:
            try:
                tournament_date_obj = datetime.strptime(tournament_date_str, '%Y-%m-%d').date()
                logging.debug(f"Tournament: {tournament.get('name')}, Date: {tournament_date_obj}")
                
                # Apply date filter
                if start_date_obj <= tournament_date_obj <= end_date_obj:
                    filtered_tournaments.append(tournament)
                    logging.debug(f"Tournament {tournament.get('name')} is within date range")
                else:
                    logging.debug(f"Tournament {tournament.get('name')} is outside date range, skipping")
            except (ValueError, TypeError) as e:
                logging.warning(f"Could not parse date for tournament {tournament.get('name')}: {str(e)}")
        else:
            logging.warning(f"Tournament {tournament.get('name')} has no date information")

    logging.info(f"Found {len(filtered_tournaments)} tournaments in the date range {START_DATE} to {END_DATE}")

    if len(filtered_tournaments) == 0:
        logging.warning("No tournaments found in the specified date range. Double-check your date format (YYYY-MM-DD).")
        return
    
    # Print tournament information in a table format
    print(f"{'Tournament ID':<15} {'Date':<15} {'Surface':<15} {'Level':<15} {'Name':<40} {'Country':<10}")
    print("-" * 100)
    
    for tournament in filtered_tournaments:
        tournament_id = tournament.get('id', 'N/A')
        tournament_name = tournament.get('name', 'N/A')
        
        # Parse date
        tournament_date_str = tournament.get('date', '').split('T')[0] if tournament.get('date') else 'N/A'
        
        # Extract court/surface information
        court_info = tournament.get('court', {})
        surface = court_info.get('name', 'N/A') if court_info else 'N/A'
        
        # Extract tournament level/round information
        round_info = tournament.get('round', {})
        level = round_info.get('name', 'N/A') if round_info else 'N/A'
        
        # Extract country information
        country_info = tournament.get('coutry', {})  # Note: API uses "coutry" not "country"
        country_code = country_info.get('acronym', 'N/A') if country_info else 'N/A'
        
        print(f"{tournament_id:<15} {tournament_date_str:<15} {surface:<15} {level:<15} {tournament_name:<40} {country_code:<10}")
    
    print("\nTo get detailed information about a tournament, use:")
    print("python get_data_from_external_api.py tournament_info <tournament_id>")

def get_player_profile_for_db(player_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format player profile data for database storage.
    
    Args:
        player_profile: Raw player profile data from API
        
    Returns:
        Dictionary with formatted player data for database storage
    """
    if not player_profile or "data" not in player_profile:
        return {}
    
    data = player_profile["data"]
    information = data.get("information", {}) or {}  # Ensure information is never None
    hand_data = information.get("plays", "")
    
    # Extract turned_pro as integer
    turned_pro_str = information.get("turnedPro", "")
    turned_pro = None
    if turned_pro_str and turned_pro_str.strip() and turned_pro_str.isdigit():
        try:
            turned_pro = int(turned_pro_str)
        except (ValueError, TypeError):
            turned_pro = None
    
    # Extract height - try multiple formats
    height_cm = None
    height_str = information.get("height", "")
    if height_str:
        # Try to extract height from formatted string
        height_cm = format_height(height_str)
        
        # If still None, try direct value
        if height_cm is None and isinstance(height_str, (int, float)):
            height_cm = int(height_str)
            
        # If still None but looks like a number in string format
        if height_cm is None and isinstance(height_str, str) and height_str.strip().isdigit():
            try:
                height_cm = int(height_str.strip())
            except (ValueError, TypeError):
                height_cm = None
    
    # Extract weight from format like "187 lbs (85 kg)"
    weight_kg = None
    weight_str = information.get("weight", "")
    if weight_str:
        # Try kg format
        kg_match = re.search(r'(\d+)\s*kg', weight_str, re.IGNORECASE)
        if kg_match:
            try:
                weight_kg = float(kg_match.group(1))
            except (ValueError, TypeError):
                weight_kg = None
        else:
            # Try lbs format and convert to kg
            lbs_match = re.search(r'(\d+)\s*lbs', weight_str, re.IGNORECASE)
            if lbs_match:
                try:
                    weight_lbs = float(lbs_match.group(1))
                    weight_kg = round(weight_lbs * 0.453592, 1)  # Convert lbs to kg
                except (ValueError, TypeError):
                    weight_kg = None
            # If it's just a number, assume kg
            elif weight_str.strip().isdigit():
                try:
                    weight_kg = float(weight_str.strip())
                except (ValueError, TypeError):
                    weight_kg = None
    
    # Log extracted physical data for debugging
    logging.debug(f"Player {data.get('name')}: Height str: '{height_str}' → {height_cm} cm, Weight str: '{weight_str}' → {weight_kg} kg")
    
    return {
        "id": data.get("id"),
        "name": data.get("name"),
        "hand": format_player_hand(hand_data),
        "height_cm": height_cm,
        "country_code": data.get("country", {}).get("acronym"),
        "birth_date": None,  # API doesn't provide birth date
        "turned_pro": turned_pro,
        "weight_kg": weight_kg,
        "birthplace": information.get("birthplace") if information else None,
        "residence": information.get("residence") if information else None,
        "coach": information.get("coach") if information else None,
        "player_status": data.get("playerStatus")
    }

def get_tournament_info_for_db(tournament_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format tournament information for database storage.
    
    Args:
        tournament_info: Raw tournament info data from API
        
    Returns:
        Dictionary with formatted tournament data for database storage
    """
    if not tournament_info or "data" not in tournament_info:
        return {}
    
    data = tournament_info["data"]
    
    return {
        "tournament_id": data.get("id"),
        "name": data.get("name"),
        "date": data.get("date"),
        "surface": format_surface(data.get("court", {}).get("name")),
        "tournament_level": format_tournament_level(data.get("round", {}).get("name")),
        "country_code": data.get("coutry", {}).get("acronym"),  # Note: API uses "coutry"
        "city": data.get("city"),
        "draw_size": data.get("drawSize"),
        "prize_money": data.get("prizeMoney")
    }

def get_match_stats_for_db(match_stats: Dict[str, Any], tournament_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format match statistics for database storage.
    
    Args:
        match_stats: Raw match stats data from API
        tournament_info: Tournament information for additional context
        
    Returns:
        Dictionary with formatted match data for database storage
    """
    if not match_stats or "data" not in match_stats:
        return {}
    
    data = match_stats["data"]
    tournament_data = tournament_info.get("data", {}) if tournament_info else {}
    
    # Get player profiles - ensure they are never None
    player1_profile = data.get("player1", {}) or {}
    player2_profile = data.get("player2", {}) or {}
    
    # Get match statistics
    p1_stats = data.get("player1Stats", {}) or {}
    p2_stats = data.get("player2Stats", {}) or {}
    
    # Check if match stats are available
    has_match_stats = p1_stats and p2_stats
    
    # Calculate serve/return stats
    serve_return_stats = get_calculated_serve_return_stats_for_db(p1_stats, p2_stats)
    
    # Get the match info
    match_info = data.get("match", {}) or {}
    
    # Format match data
    match_data = {
        "match_id": data.get("id"),
        "tournament_id": tournament_data.get("id"),
        "tournament_date": tournament_data.get("date"),
        "surface": format_surface(tournament_data.get("court", {}).get("name")),
        "winner_id": player1_profile.get("id"),
        "loser_id": player2_profile.get("id"),
        "winner_hand": format_player_hand(player1_profile.get("hand")),
        "winner_height_cm": format_height(player1_profile.get("height")),
        "winner_country_code": player1_profile.get("country", {}).get("acronym"),
        "winner_age": calculate_player_age(player1_profile.get("birthDate"), tournament_data.get("date")),
        "loser_hand": format_player_hand(player2_profile.get("hand")),
        "loser_height_cm": format_height(player2_profile.get("height")),
        "loser_country_code": player2_profile.get("country", {}).get("acronym"),
        "loser_age": calculate_player_age(player2_profile.get("birthDate"), tournament_data.get("date")),
        "score": match_info.get("result", ""),
        "best_of": 3,  # Default for most matches
        "round": match_info.get("round", {}).get("name"),
        "match_type": "singles" if '/' not in player1_profile.get("name", "") else "doubles"
    }
    
    # Add serve/return stats only if they are available
    if has_match_stats and serve_return_stats and len(serve_return_stats) > 0:
        match_data.update(serve_return_stats)
    
    return match_data

def create_players_table(engine: create_engine) -> None:
    """Create the players table if it doesn't exist"""
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS players (
                    id INTEGER PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    hand VARCHAR(1),
                    height_cm FLOAT,
                    country_code VARCHAR(3),
                    birth_date DATE,
                    turned_pro INTEGER,
                    weight_kg FLOAT,
                    birthplace VARCHAR(255),
                    residence VARCHAR(255),
                    coach VARCHAR(255),
                    player_status VARCHAR(50),
                    last_updated TIMESTAMP WITH TIME ZONE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.commit()
        logging.info("Players table created successfully")
    except SQLAlchemyError as e:
        logging.error(f"Error creating players table: {str(e)}")
        raise

def create_matches_table(engine: create_engine) -> None:
    """Create the matches table if it doesn't exist"""
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS matches (
                    id SERIAL PRIMARY KEY,
                    match_id VARCHAR(50) UNIQUE,
                    tournament_id VARCHAR(50),
                    tournament_name VARCHAR(255),
                    surface VARCHAR(50),
                    draw_size FLOAT,
                    tournament_level VARCHAR(50),
                    tournament_date DATE,
                    match_num BIGINT,
                    winner_id INTEGER,
                    winner_seed FLOAT,
                    winner_entry VARCHAR(50),
                    winner_name VARCHAR(255),
                    winner_hand VARCHAR(1),
                    winner_height_cm FLOAT,
                    winner_country_code VARCHAR(3),
                    winner_age FLOAT,
                    loser_id INTEGER,
                    loser_seed FLOAT,
                    loser_entry VARCHAR(50),
                    loser_name VARCHAR(255),
                    loser_hand VARCHAR(1),
                    loser_height_cm FLOAT,
                    loser_country_code VARCHAR(3),
                    loser_age FLOAT,
                    score VARCHAR(50),
                    best_of INTEGER,
                    round VARCHAR(50),
                    minutes FLOAT,
                    winner_aces FLOAT,
                    winner_double_faults FLOAT,
                    winner_serve_points FLOAT,
                    winner_first_serves_in FLOAT,
                    winner_first_serve_points_won FLOAT,
                    winner_second_serve_points_won FLOAT,
                    winner_service_games FLOAT,
                    winner_break_points_saved FLOAT,
                    winner_break_points_faced FLOAT,
                    loser_aces FLOAT,
                    loser_double_faults FLOAT,
                    loser_serve_points FLOAT,
                    loser_first_serves_in FLOAT,
                    loser_first_serve_points_won FLOAT,
                    loser_second_serve_points_won FLOAT,
                    loser_service_games FLOAT,
                    loser_break_points_saved FLOAT,
                    loser_break_points_faced FLOAT,
                    winner_rank FLOAT,
                    winner_rank_points FLOAT,
                    loser_rank FLOAT,
                    loser_rank_points FLOAT,
                    match_type VARCHAR(50),
                    winner_elo FLOAT,
                    loser_elo FLOAT,
                    winner_matches INTEGER,
                    loser_matches INTEGER,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.commit()
        logging.info("Matches table created successfully")
    except SQLAlchemyError as e:
        logging.error(f"Error creating matches table: {str(e)}")
        raise

def get_player_from_db(engine: create_engine, player_name: str) -> Optional[Dict[str, Any]]:
    """
    Get player from database by name.
    
    Args:
        engine: SQLAlchemy engine
        player_name: Player's name to search for
        
    Returns:
        Optional[Dict]: Player data if found, None otherwise
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT * FROM players 
                WHERE name = :name
            """), {"name": player_name})
            player = result.fetchone()
            
            if player:
                # Convert Row object to dictionary properly
                columns = result.keys()
                return {col: getattr(player, col) for col in columns}
            return None
    except SQLAlchemyError as e:
        logging.error(f"Error querying player from database: {str(e)}")
        return None

def check_tournament_exists(engine: create_engine, tournament_id: str) -> bool:
    """
    Check if a tournament ID already exists in the database.
    
    Args:
        engine: SQLAlchemy engine
        tournament_id: Tournament ID to check
        
    Returns:
        bool: True if tournament exists in database, False otherwise
    """
    try:
        # Convert tournament_id to string to match VARCHAR column type
        tournament_id_str = str(tournament_id)
        
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT COUNT(*) as count
                FROM matches 
                WHERE tournament_id = :tournament_id
                LIMIT 1
            """), {"tournament_id": tournament_id_str})
            
            row = result.fetchone()
            count = row.count if row else 0
            
            exists = count > 0
            if exists:
                logging.info(f"Tournament ID {tournament_id} already exists in database")
            else:
                logging.info(f"Tournament ID {tournament_id} not found in database")
                
            return exists
            
    except SQLAlchemyError as e:
        logging.error(f"Error checking tournament in database: {str(e)}")
        return False

def update_player_in_db(engine: create_engine, player_data: Dict[str, Any]) -> bool:
    """
    Update or insert player data in database.
    
    Args:
        engine: SQLAlchemy engine
        player_data: Player data to update/insert
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create a clean version of player_data with NULL for empty strings
        clean_data = {}
        for key, value in player_data.items():
            if value == '' or value is None or (isinstance(value, str) and value.strip() == ''):
                clean_data[key] = None
            else:
                clean_data[key] = value
        
        with engine.connect() as conn:
            # Check if player exists
            existing_player = get_player_from_db(engine, clean_data['name'])
            
            if existing_player:
                # Update existing player
                update_fields = []
                for key in clean_data:
                    if key not in ['id', 'name']:  # Don't update these fields
                        update_fields.append(f"{key} = :{key}")
                
                update_query = text(f"""
                    UPDATE players 
                    SET {', '.join(update_fields)},
                        last_updated = CURRENT_TIMESTAMP
                    WHERE name = :name
                """)
                
                conn.execute(update_query, clean_data)
            else:
                # Insert new player - ensure no empty strings
                insert_query = text("""
                    INSERT INTO players (
                        id, name, hand, height_cm, country_code, birth_date,
                        turned_pro, weight_kg, birthplace, residence, coach,
                        player_status, last_updated
                    ) VALUES (
                        :id, :name, :hand, :height_cm, :country_code, :birth_date,
                        :turned_pro, :weight_kg, :birthplace, :residence, :coach,
                        :player_status, CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute(insert_query, clean_data)
            
            conn.commit()
            return True
    except SQLAlchemyError as e:
        logging.error(f"Error updating player in database: {str(e)}")
        logging.error(f"Data that caused error: {clean_data}")
        return False

def get_or_create_player(api_client: TennisAPIClient, engine: create_engine, player_name: str, player_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Get player from database or create new entry from API.
    
    Args:
        api_client: TennisAPIClient instance
        engine: SQLAlchemy engine
        player_name: Player's name
        player_id: Optional player ID from API
        
    Returns:
        Optional[Dict]: Player data if successful, None otherwise
    """
    # First check database
    player = get_player_from_db(engine, player_name)
    if player:
        logging.info(f"Found player {player_name} in database")
        return player
    
    # If not in database and we have player_id, fetch from API
    if player_id:
        logging.info(f"Fetching player {player_name} from API")
        player_profile = api_client.get_player_profile(player_id)
        if player_profile and "data" in player_profile:
            # Use the get_player_profile_for_db function to format the data
            formatted_data = get_player_profile_for_db(player_profile)
            
            # Make sure we have at least player ID and name
            if not formatted_data.get("id") or not formatted_data.get("name"):
                # Set these fields from what we know
                if not formatted_data.get("id"):
                    formatted_data["id"] = player_id
                if not formatted_data.get("name"):
                    formatted_data["name"] = player_name
            
            # Update database
            if update_player_in_db(engine, formatted_data):
                logging.info(f"Player {player_name} added/updated in database")
                return formatted_data
            else:
                logging.error(f"Failed to update player {player_name} in database")
    
    # If we couldn't get player data from API, create minimal record
    logging.warning(f"Creating minimal player record for {player_name}")
    minimal_data = {
        "id": player_id,
        "name": player_name,
        "hand": None,  # Use None instead of defaulting to "R"
        "country_code": None,
        "height_cm": None,
        "weight_kg": None
    }
    
    if update_player_in_db(engine, minimal_data):
        return minimal_data
    
    return None

def check_match_stats_exist(engine: create_engine, match_id: str) -> bool:
    """
    Check if match statistics exist in the database for a given match ID.
    
    Args:
        engine: SQLAlchemy engine
        match_id: Match ID to check
        
    Returns:
        bool: True if match statistics exist in database, False otherwise
    """
    try:
        # Convert match_id to string as a precaution
        match_id_str = str(match_id)
        
        with engine.connect() as conn:
            # Check if any of the match statistics columns have non-null values
            result = conn.execute(text("""
                SELECT COUNT(*) as count
                FROM matches 
                WHERE match_num = :match_id
                AND (
                    winner_aces IS NOT NULL OR
                    winner_double_faults IS NOT NULL OR
                    winner_serve_points IS NOT NULL OR
                    winner_first_serves_in IS NOT NULL OR
                    winner_first_serve_points_won IS NOT NULL OR
                    winner_second_serve_points_won IS NOT NULL OR
                    winner_service_games IS NOT NULL OR
                    winner_break_points_saved IS NOT NULL OR
                    winner_break_points_faced IS NOT NULL OR
                    loser_aces IS NOT NULL OR
                    loser_double_faults IS NOT NULL OR
                    loser_serve_points IS NOT NULL OR
                    loser_first_serves_in IS NOT NULL OR
                    loser_first_serve_points_won IS NOT NULL OR
                    loser_second_serve_points_won IS NOT NULL OR
                    loser_service_games IS NOT NULL OR
                    loser_break_points_saved IS NOT NULL OR
                    loser_break_points_faced IS NOT NULL
                )
                LIMIT 1
            """), {"match_id": match_id_str})
            
            row = result.fetchone()
            count = row.count if row else 0
            
            exists = count > 0
            if exists:
                logging.info(f"Match ID {match_id} has statistics in database")
            else:
                logging.info(f"Match ID {match_id} has no statistics in database")
                
            return exists
            
    except SQLAlchemyError as e:
        logging.error(f"Error checking match statistics in database: {str(e)}")
        return False

def check_match_exists(engine: create_engine, match_id: str) -> bool:
    """
    Check if a match ID already exists in the database.
    
    Args:
        engine: SQLAlchemy engine
        match_id: Match ID to check
        
    Returns:
        bool: True if match exists in database, False otherwise
    """
    try:
        # Convert match_id to string as a precaution
        match_id_str = str(match_id)
        
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT COUNT(*) as count
                FROM matches 
                WHERE match_num = :match_id
                LIMIT 1
            """), {"match_id": match_id_str})
            
            row = result.fetchone()
            count = row.count if row else 0
            
            exists = count > 0
            if exists:
                logging.info(f"Match ID {match_id} exists in database")
            else:
                logging.info(f"Match ID {match_id} not found in database")
                
            return exists
            
    except SQLAlchemyError as e:
        logging.error(f"Error checking match in database: {str(e)}")
        return False

def insert_match_data_to_db(engine: create_engine, match_data: Dict[str, Any], tournament_info: Dict[str, Any], 
                           match_stats: Dict[str, Any], player1_data: Dict[str, Any], player2_data: Dict[str, Any]) -> bool:
    """
    Insert match data into the matches table.
    
    Args:
        engine: SQLAlchemy engine
        match_data: Match data from API
        tournament_info: Tournament info from API
        match_stats: Match statistics from API (can be empty if stats are not available)
        player1_data: Player 1 (winner) data
        player2_data: Player 2 (loser) data
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Extract tournament data
        tournament_data = tournament_info.get("data", {})
        tournament_date = tournament_data.get("date", "").split("T")[0] if tournament_data.get("date") else None
        
        # Extract court/surface information
        court_info = tournament_data.get("court", {})
        surface = format_surface(court_info.get("name")) if court_info else None
        
        # Extract tournament level
        round_info = tournament_data.get("round", {})
        tournament_level = format_tournament_level(round_info.get("name")) if round_info else None
        
        # Check if match stats are available
        has_match_stats = match_stats is not None and "data" in match_stats
        
        # Extract match statistics if available
        stats_data = {}
        player1_stats = {}
        player2_stats = {}
        serve_return_stats = {}
        
        if has_match_stats:
            stats_data = match_stats.get("data", {})
            player1_stats = stats_data.get("player1Stats", {})
            player2_stats = stats_data.get("player2Stats", {})
            
            # Get calculated serve/return stats
            serve_return_stats = get_calculated_serve_return_stats_for_db(player1_stats, player2_stats)
        
        # Extract match round information
        match_round = match_data.get("round", {}).get("name")
        
        # Handle roundId from getTournamentResults response format
        if not match_round and match_data.get("roundId"):
            # Map roundId values to match existing database format (R32, QF, R16, etc.)
            round_id_map = {
                1: "Q1",  # First Round Qualifying
                2: "Q2",  # Second Round Qualifying
                3: "Q3",  # Final Round Qualifying
                4: "R128",  # First Round (for large draws like Grand Slams)
                5: "R64",  # Second Round (for large draws)
                6: "R32",  # Third Round
                7: "R16",  # Fourth Round
                8: "QF",   # Quarter-Finals
                9: "SF",   # Semi-Finals
                10: "F",   # Finals
                11: "RR",  # Round Robin
                12: "F"    # Final
            }
            round_id = match_data.get("roundId")
            match_round = round_id_map.get(round_id)
            if not match_round:
                # For unmapped IDs, try to determine format based on common patterns
                if round_id < 4:
                    match_round = f"Q{round_id}"  # Qualifying rounds
                else:
                    match_round = f"R{2**(7-round_id)}" if round_id < 8 else f"R{round_id}"
        elif not match_round and isinstance(match_data.get("round"), int):
            # If round is available directly as an integer, use same mapping
            round_id = match_data.get("round")
            # Reuse the same mapping logic for consistency
            if round_id in round_id_map:
                match_round = round_id_map.get(round_id)
            elif round_id < 4:
                match_round = f"Q{round_id}"
            else:
                match_round = f"R{2**(7-round_id)}" if round_id < 8 else f"R{round_id}"
        
        if not match_round:
            # Set round to null if missing
            match_round = None
            logging.warning(f"No round information found for match {match_data.get('id')}, setting to null")
        
        # Extract match score/result
        score = match_data.get("result", "")
        
        # Determine match type (singles/doubles)
        match_type = "singles"
        player1_name = player1_data.get("name", "")
        if '/' in player1_name:
            match_type = "doubles"
        
        # Set best_of based on match type and tournament level
        best_of = 3  # Default for most matches
        if tournament_level == 'G' and match_type == 'singles':
            # Grand Slam singles matches are typically best of 5 for ATP
            best_of = 5
        
        # Prepare match data for database
        db_match_data = {
            "tournament_id": str(tournament_data.get("id")),  # Convert to string
            "tournament_name": tournament_data.get("name"),
            "tournament_date": tournament_date,
            "surface": surface or "Unknown",  # Provide default if missing
            "tournament_level": tournament_level or "C",  # Default to Challenger level if missing
            "match_num": match_data.get("id"),  # Use match ID as match_num
            "winner_id": player1_data.get("id"),
            "winner_name": player1_data.get("name"),
            "winner_hand": player1_data.get("hand"),
            "winner_height_cm": player1_data.get("height_cm"),
            "winner_country_code": player1_data.get("country_code"),
            "winner_age": calculate_player_age(player1_data.get("birth_date"), tournament_date),
            "loser_id": player2_data.get("id"),
            "loser_name": player2_data.get("name"),
            "loser_hand": player2_data.get("hand"),
            "loser_height_cm": player2_data.get("height_cm"),
            "loser_country_code": player2_data.get("country_code"),
            "loser_age": calculate_player_age(player2_data.get("birth_date"), tournament_date),
            "score": score,
            "best_of": best_of,
            "round": match_round,
            "match_type": match_type
        }
        
        # Add serve/return stats only if they are available
        if has_match_stats and serve_return_stats and len(serve_return_stats) > 0:
            db_match_data.update(serve_return_stats)
        
        # Clean up any empty strings to NULL
        clean_match_data = {}
        for key, value in db_match_data.items():
            if value == '' or value is None or (isinstance(value, str) and value.strip() == ''):
                # For NOT NULL fields, provide default values
                if key == "surface":
                    clean_match_data[key] = "Unknown"
                elif key == "tournament_level":
                    clean_match_data[key] = "C"
                else:
                    clean_match_data[key] = None
            else:
                clean_match_data[key] = value
        
        # Insert into database
        with engine.connect() as conn:
            # Check if match already exists using match_num and tournament_id
            result = conn.execute(text("""
                SELECT id FROM matches 
                WHERE match_num = :match_num AND tournament_id = :tournament_id
            """), {"match_num": clean_match_data["match_num"], "tournament_id": str(clean_match_data["tournament_id"])})
            existing_match = result.fetchone()
            
            if existing_match:
                # Update existing match
                logging.info(f"Updating existing match {clean_match_data['match_num']} for tournament {clean_match_data['tournament_id']}")
                
                # Build dynamic update query
                update_cols = []
                for key in clean_match_data:
                    if key not in ["match_num", "tournament_id"]:  # Don't update these keys
                        update_cols.append(f"{key} = :{key}")
                
                update_query = text(f"""
                    UPDATE matches 
                    SET {', '.join(update_cols)}
                    WHERE match_num = :match_num AND tournament_id = :tournament_id
                """)
                
                conn.execute(update_query, clean_match_data)
            else:
                # Insert new match
                logging.info(f"Inserting new match {clean_match_data['match_num']} for tournament {clean_match_data['tournament_id']}")
                
                # Build dynamic insert query
                columns = ', '.join(clean_match_data.keys())
                placeholders = ', '.join(f":{key}" for key in clean_match_data)
                
                insert_query = text(f"""
                    INSERT INTO matches (
                        {columns}
                    ) VALUES (
                        {placeholders}
                    )
                """)
                
                conn.execute(insert_query, clean_match_data)
            
            conn.commit()
            return True
            
    except SQLAlchemyError as e:
        logging.error(f"Error inserting match data into database: {str(e)}")
        return False

def main():
    """
    Main function for the API data collection process.
    """
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Configure logging
        log_level = logging.DEBUG if args.verbose else logging.INFO
        logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
        
        # Set rate limit (requests per second)
        rate_limit = API_REQUESTS_PER_SECOND
        
        # Initialize API client
        api_client = TennisAPIClient(quiet_mode=args.quiet)
        
        # Initialize database connection if not skipping
        engine = None
        if not args.skip_db_write:
            # Create SQLAlchemy engine
            # Setup database engine - prioritize environment variable if available
            database_url = os.environ.get('DATABASE_URL')
            if database_url and not args.db_url:
                # Check if we need to convert postgres:// to postgresql://
                if database_url.startswith('postgres://'):
                    database_url = database_url.replace('postgres://', 'postgresql://', 1)
                    logging.info(f"Modified database URL to use postgresql:// prefix")
                
                logging.info(f"Using database URL from environment: {database_url[:20]}...")
                db_url = database_url
            else:
                db_url = args.db_url if args.db_url else f"postgresql://{args.db_user}:{args.db_password}@{args.db_host}:{args.db_port}/{args.db_name}"
                logging.info(f"Using database URL from command-line arguments")
            
            logging.info(f"Connecting to database: {db_url}")
            engine = create_engine(db_url)
            
            # Create tables if they don't exist
            create_matches_table(engine)
            create_players_table(engine)
            
        # Dictionary to store API responses if save_responses is True
        api_responses = {}
        
        # Use the date range directly from arguments (defaults are set in parse_arguments)
        start_date = args.start_date
        end_date = args.end_date
        
        logging.info(f"Fetching tournaments between {start_date} and {end_date}")
        
        # 1. Get tournament calendar
        tournament_calendar = api_client.get_tournament_calendar(start_date, end_date, page_size=args.page_size)
        if not tournament_calendar or "data" not in tournament_calendar:
            logging.error("Failed to fetch tournament calendar data")
            return
            
        # Sort tournaments by date (process earliest first)
        tournaments = tournament_calendar.get('data', [])
        
        def get_tournament_date(tournament):
            date_str = tournament.get('date', '')
            if not date_str:
                return datetime.max  # If no date, sort to the end
            try:
                # Remove the time part and convert to datetime
                return datetime.strptime(date_str.split('T')[0], '%Y-%m-%d')
            except (ValueError, TypeError):
                return datetime.max  # If date parsing fails, sort to end
                
        tournaments.sort(key=get_tournament_date)
        logging.info(f"Found {len(tournaments)} tournaments, sorted by date")
        
        # Filter tournaments by date range
        filtered_tournaments = []
        start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        for tournament in tournaments:
            tournament_date_str = tournament.get('date', '')
            if not tournament_date_str:
                logging.warning(f"Tournament {tournament.get('name')} has no date information, skipping")
                continue
                
            try:
                # Parse the date string (format: "2025-01-01T00:00:00.000Z")
                tournament_date_obj = datetime.strptime(tournament_date_str.split('T')[0], '%Y-%m-%d').date()
                
                # Check if tournament date is within our specified range
                if start_date_obj <= tournament_date_obj <= end_date_obj:
                    filtered_tournaments.append(tournament)
                    logging.debug(f"Tournament {tournament.get('name')} is within date range")
                else:
                    logging.debug(f"Tournament {tournament.get('name')} is outside date range, skipping")
            except (ValueError, TypeError) as e:
                logging.warning(f"Could not parse date for tournament {tournament.get('name')}: {str(e)}")
        
        logging.info(f"Filtered to {len(filtered_tournaments)} tournaments within date range {start_date} to {end_date}")
        
        # If no tournaments found in the date range
        if len(filtered_tournaments) == 0:
            logging.warning(f"No tournaments found in the specified date range {start_date} to {end_date}")
            return
            
        # Use filtered tournaments instead of all tournaments
        tournaments = filtered_tournaments
        
        # Track rate limiting - no longer needed as it's handled by the TennisAPIClient
        # request_count = 0
        # last_request_time = time.time()

        # Set up progress tracking
        total_tournaments = len(tournaments)
        tournament_pbar = None if args.no_progress else tqdm(
            total=total_tournaments, 
            desc="Processing tournaments", 
            unit="tournament",
            bar_format="{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )
        
        # Track overall progress (tournaments + estimated matches)
        total_estimated_matches = 0  # Will be updated as we process tournaments
        total_expected_steps = total_tournaments  # Start with just tournaments, will add matches
        completed_steps = 0
        
        # Process each tournament
        for t_idx, tournament in enumerate(tournaments):
            tournament_id = tournament.get('id')
            tournament_name = tournament.get('name')
            
            if not tournament_id:
                logging.warning("Tournament found without ID, skipping")
                completed_steps += 1
                if tournament_pbar:
                    tournament_pbar.update(1)
                continue
                
            logging.info(f"Processing tournament {t_idx+1}/{total_tournaments}: {tournament_name} (ID: {tournament_id})")
            
            # We no longer skip tournaments that exist in the database
            # Instead we'll check individual matches
            
            # 2. Get tournament details
            # Rate limiting is now handled by the TennisAPIClient
            tournament_info = api_client.get_tournament_info(tournament_id)
            if not tournament_info or "data" not in tournament_info:
                logging.error(f"Failed to fetch info for tournament {tournament_id}")
                completed_steps += 1
                if tournament_pbar:
                    tournament_pbar.update(1)
                continue
                
            api_responses[f"tournament_{tournament_id}_info"] = tournament_info
            print_tournament_info(tournament_info)
            
            # 3. Get tournament results
            # Rate limiting is now handled by the TennisAPIClient
            tournament_results = api_client.get_tournament_results(tournament_id)
            if not tournament_results or "data" not in tournament_results:
                logging.error(f"Failed to fetch results for tournament {tournament_id}")
                completed_steps += 1
                if tournament_pbar:
                    tournament_pbar.update(1)
                continue
                
            api_responses[f"tournament_{tournament_id}_results"] = tournament_results
            
            # Process each match in the tournament
            results_data = tournament_results.get('data', {})
            
            # Extract matches from different categories (singles, doubles, qualifying)
            singles_matches = results_data.get('singles', [])
            qualifying_singles_matches = results_data.get('qualifying', [])
            
            # Skip doubles matches
            doubles_matches = []
            
            # Log the count of matches by type
            logging.info(f"Singles matches: {len(singles_matches)}, Qualifying singles matches: {len(qualifying_singles_matches)}")
            
            # Process only singles and qualifying singles matches
            all_matches = singles_matches + qualifying_singles_matches
            num_matches = len(all_matches)
            logging.info(f"Found {num_matches} singles matches for tournament {tournament_id}")
            
            # Update our total estimated steps
            total_estimated_matches += num_matches
            total_expected_steps = total_tournaments + total_estimated_matches
            
            # Create a progress bar for matches in this tournament
            match_pbar = None if args.no_progress else tqdm(
                total=num_matches, 
                desc=f"Matches for {tournament_name[:30]}", 
                unit="match",
                leave=False,
                bar_format="{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}"
            )
            
            # Sort matches by date to process oldest first
            all_matches.sort(key=lambda m: m.get('date', '0000-00-00'))
            
            # If no matches found, check if the tournament might be in the future
            if len(all_matches) == 0:
                logging.warning(f"No singles matches found for tournament {tournament_name}")
                logging.warning(f"API response might not contain match data for this tournament yet")
                completed_steps += 1
                if tournament_pbar:
                    tournament_pbar.update(1)
                continue
                
            for m_idx, match in enumerate(all_matches):
                # Check if it's a singles match by examining player names (extra safety check)
                player1_name = match.get('player1', {}).get('name', '')
                player2_name = match.get('player2', {}).get('name', '')
                
                # Update overall progress
                completed_steps += 1
                overall_progress = (completed_steps / total_expected_steps) * 100
                
                # Show overall progress (without tqdm)
                if not tournament_pbar and not args.no_progress:
                    print(f"Overall progress: {overall_progress:.1f}% (Step {completed_steps}/{total_expected_steps})")
                
                # Update the match progress bar
                if match_pbar:
                    match_pbar.update(1)
                
                # If player name contains '/' it's likely a doubles match, skip it
                if '/' in player1_name or '/' in player2_name:
                    logging.info(f"Skipping doubles match: {player1_name} vs {player2_name}")
                    continue
                
                match_id = match.get('id')
                player1_id = match.get('player1Id')
                player2_id = match.get('player2Id')
                
                if not match_id:
                    logging.warning(f"Match found without ID in tournament {tournament_id}, skipping")
                    continue
                
                if not player1_id or not player2_id:
                    logging.warning(f"Match {match_id} found with missing player ID(s) in tournament {tournament_id}, skipping")
                    continue
                
                # Check if match already exists in the database
                match_exists = not args.skip_db_write and check_match_exists(engine, match_id)
                match_stats_exist = not args.skip_db_write and check_match_stats_exist(engine, match_id)
                
                if match_exists and match_stats_exist:
                    logging.info(f"Match {match_id} and its statistics already exist in database, skipping")
                    continue
                
                logging.info(f"Processing match {m_idx+1}/{num_matches}: {player1_name} vs {player2_name}")
                
                # 4. Get match statistics if they don't exist in the database
                match_stats = None    
                if not match_stats_exist:
                    try:
                        match_stats = api_client.get_match_stats(tournament_id, player1_id, player2_id)
                        if match_stats and "data" in match_stats:
                            api_responses[f"match_{match_id}_stats"] = match_stats
                            logging.info(f"Successfully fetched statistics for match {match_id}")
                        else:
                            logging.warning(f"Failed to fetch statistics for match {match_id}, but will continue processing")
                    except Exception as e:
                        logging.warning(f"Error fetching match stats for match {match_id}: {str(e)}, but will continue processing")
                else:
                    logging.info(f"Match {match_id} statistics already exist in database, skipping API call")
                
                # Determine the winner and loser
                winner_id = match.get("match_winner")  
                loser_id = None
                
                # If match_winner is not available, determine it from player1Id/player2Id
                if not winner_id and match.get("result"):
                    # Default to player1 as winner if we can't determine
                    winner_id = player1_id
                    loser_id = player2_id
                    logging.warning(f"Match {match_id} has no explicit winner, defaulting to player1")
                elif winner_id:
                    # If we have a match_winner, set loser accordingly
                    if winner_id == player1_id:
                        loser_id = player2_id
                    else:
                        loser_id = player1_id
                else:
                    logging.warning(f"Match {match_id} has no winner information and no result, skipping")
                    continue
                
                # Get player names
                winner_name = None
                loser_name = None
                
                if winner_id == player1_id:
                    winner_name = player1_name
                    loser_name = player2_name
                else:
                    winner_name = player2_name
                    loser_name = player1_name
                
                if not winner_name or not loser_name:
                    logging.warning(f"Match {match_id} has missing player names, skipping")
                    continue
                
                logging.info(f"Match winner: {winner_name} (ID: {winner_id}), loser: {loser_name} (ID: {loser_id})")
                
                # Get or create player data
                # We'll try both by ID and by name to ensure we catch all player information
                winner_data = None
                loser_data = None
                
                # Process winner data
                if winner_id:
                    # Rate limiting is now handled by the TennisAPIClient
                    winner_data = get_or_create_player(api_client, engine, winner_name, winner_id)
                    if winner_data:
                        api_responses[f"player_{winner_id}_profile"] = winner_data
                        
                        # Update player ID in matches table if necessary and not skipping db write
                        if not args.skip_db_write:
                            with engine.connect() as conn:
                                # Check and update both winner_id and loser_id for this player's name
                                # First update any matches where this player was the winner
                                result = conn.execute(text("""
                                    SELECT COUNT(*) as count FROM matches 
                                    WHERE winner_name = :name AND winner_id != :id
                                """), {"name": winner_name, "id": winner_id})
                                winner_matches_count = result.fetchone()[0]
                                
                                if winner_matches_count > 0:
                                    logging.info(f"Updating winner_id for {winner_matches_count} matches with player {winner_name}")
                                    conn.execute(text("""
                                        UPDATE matches 
                                        SET winner_id = :new_id
                                        WHERE winner_name = :name AND winner_id != :new_id
                                    """), {"name": winner_name, "new_id": winner_id})
                                
                                # Also update any matches where this player was the loser
                                result = conn.execute(text("""
                                    SELECT COUNT(*) as count FROM matches 
                                    WHERE loser_name = :name AND loser_id != :id
                                """), {"name": winner_name, "id": winner_id})
                                loser_matches_count = result.fetchone()[0]
                                
                                if loser_matches_count > 0:
                                    logging.info(f"Updating loser_id for {loser_matches_count} matches with player {winner_name}")
                                    conn.execute(text("""
                                        UPDATE matches 
                                        SET loser_id = :new_id
                                        WHERE loser_name = :name AND loser_id != :new_id
                                    """), {"name": winner_name, "new_id": winner_id})
                                
                                conn.commit()
                
                # Process loser data
                if loser_id:
                    # Rate limiting is now handled by the TennisAPIClient
                    loser_data = get_or_create_player(api_client, engine, loser_name, loser_id)
                    if loser_data:
                        api_responses[f"player_{loser_id}_profile"] = loser_data
                        
                        # Update player ID in matches table if necessary and not skipping db write
                        if not args.skip_db_write:
                            with engine.connect() as conn:
                                # Check and update both winner_id and loser_id for this player's name
                                # First update any matches where this player was the loser
                                result = conn.execute(text("""
                                    SELECT COUNT(*) as count FROM matches 
                                    WHERE loser_name = :name AND loser_id != :id
                                """), {"name": loser_name, "id": loser_id})
                                loser_matches_count = result.fetchone()[0]
                                
                                if loser_matches_count > 0:
                                    logging.info(f"Updating loser_id for {loser_matches_count} matches with player {loser_name}")
                                    conn.execute(text("""
                                        UPDATE matches 
                                        SET loser_id = :new_id
                                        WHERE loser_name = :name AND loser_id != :new_id
                                    """), {"name": loser_name, "new_id": loser_id})
                                
                                # Also update any matches where this player was the winner
                                result = conn.execute(text("""
                                    SELECT COUNT(*) as count FROM matches 
                                    WHERE winner_name = :name AND winner_id != :id
                                """), {"name": loser_name, "id": loser_id})
                                winner_matches_count = result.fetchone()[0]
                                
                                if winner_matches_count > 0:
                                    logging.info(f"Updating winner_id for {winner_matches_count} matches with player {loser_name}")
                                    conn.execute(text("""
                                        UPDATE matches 
                                        SET winner_id = :new_id
                                        WHERE winner_name = :name AND winner_id != :new_id
                                    """), {"name": loser_name, "new_id": loser_id})
                                
                                conn.commit()
                
                # Insert match data into the database if not skipping
                if not args.skip_db_write:
                    if winner_data and loser_data:
                        # Insert match data regardless of whether we have match stats
                        if insert_match_data_to_db(engine, match, tournament_info, match_stats, winner_data, loser_data):
                            logging.info(f"Match {match_id} inserted into the database")
                    else:
                        if not winner_data:
                            logging.warning(f"Skipping match {match_id} insertion - missing winner data for player {winner_name} (ID: {winner_id})")
                        if not loser_data:
                            logging.warning(f"Skipping match {match_id} insertion - missing loser data for player {loser_name} (ID: {loser_id})")
                        logging.warning(f"Skipping match {match_id} insertion - missing player data")
                else:
                    logging.info(f"Skipping database write for match {match_id} (--skip-db-write flag set)")
            
            # Close the match progress bar
            if match_pbar:
                match_pbar.close()
            
            # Update tournament progress
            completed_steps += 1  # Count the tournament as completed
            if tournament_pbar:
                tournament_pbar.update(1)
        
        # Close tournament progress bar
        if tournament_pbar:
            tournament_pbar.close()
            
        # Final progress report    
        if not args.no_progress:
            print(f"Data collection completed: 100% ({completed_steps}/{total_expected_steps} steps processed)")
        
        # Save all API responses to a JSON file if requested
        if args.save_responses:
            with open("tennis_api_responses.json", "w") as f:
                json.dump(api_responses, f, indent=2)
            logging.info("API responses saved to tennis_api_responses.json")
        
        logging.info("API data collection completed successfully")
        
    except Exception as e:
        logging.error(f"Error in main function: {str(e)}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main() 