"""
Fetch Upcoming Tennis Matches

This script fetches upcoming tennis matches from the API and stores them in the upcoming_matches table.
It's designed to be run regularly to keep the upcoming matches database up to date.
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import pandas as pd
from sqlalchemy import create_engine, text
import psycopg2
from psycopg2.extras import RealDictCursor

# Add project root to the path so we can import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Fix for external-api module
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external-api"))

from get_data_from_external_api import (
    TennisAPIClient, 
    format_tournament_level,
    format_surface,
    format_player_hand,
    calculate_player_age
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_database_connection():
    """Get a connection to the PostgreSQL database."""
    try:
        # Try to get DATABASE_URL from environment variables
        database_url = os.environ.get("DATABASE_URL")
        
        if database_url:
            logger.info(f"Connecting to database using DATABASE_URL")
            conn = psycopg2.connect(database_url)
        else:
            # Fallback to individual connection parameters
            dbname = os.environ.get("DB_NAME", "tennis_predictor")
            user = os.environ.get("DB_USER", "postgres")
            password = os.environ.get("DB_PASSWORD", "postgres")
            host = os.environ.get("DB_HOST", "localhost")
            port = os.environ.get("DB_PORT", "5432")
            
            logger.info(f"Connecting to database: {dbname} on {host}:{port} as {user}")
            conn = psycopg2.connect(
                dbname=dbname,
                user=user,
                password=password,
                host=host,
                port=port
            )
        
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise

def get_sqlalchemy_engine():
    """Get a SQLAlchemy engine for database operations."""
    try:
        # Try to get DATABASE_URL from environment variables
        database_url = os.environ.get("DATABASE_URL")
        
        if database_url:
            logger.info(f"Creating SQLAlchemy engine using DATABASE_URL")
            # For SQLAlchemy, ensure the URL starts with postgresql:// not postgres://
            if database_url.startswith("postgres://"):
                database_url = database_url.replace("postgres://", "postgresql://", 1)
            engine = create_engine(database_url)
        else:
            # Fallback to individual connection parameters
            dbname = os.environ.get("DB_NAME", "tennis_predictor")
            user = os.environ.get("DB_USER", "postgres")
            password = os.environ.get("DB_PASSWORD", "postgres")
            host = os.environ.get("DB_HOST", "localhost")
            port = os.environ.get("DB_PORT", "5432")
            
            logger.info(f"Creating SQLAlchemy engine for: {dbname} on {host}:{port} as {user}")
            engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")
        
        return engine
    except Exception as e:
        logger.error(f"SQLAlchemy engine creation error: {e}")
        raise

def create_upcoming_matches_table():
    """Create the upcoming_matches table if it doesn't exist."""
    conn = get_database_connection()
    cursor = conn.cursor()
    
    try:
        # Create upcoming_matches table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS upcoming_matches (
            id SERIAL PRIMARY KEY,
            match_id VARCHAR(50) UNIQUE,
            tournament_id VARCHAR(50),
            tournament_name VARCHAR(255),
            surface VARCHAR(50),
            tournament_level VARCHAR(50),
            tournament_date DATE,
            match_date TIMESTAMP,
            round VARCHAR(50),
            player1_id INTEGER,
            player1_name VARCHAR(255),
            player1_seed FLOAT,
            player1_entry VARCHAR(50),
            player1_hand VARCHAR(1),
            player1_height_cm FLOAT,
            player1_country_code VARCHAR(3),
            player1_age FLOAT,
            player1_rank FLOAT,
            player1_rank_points FLOAT,
            player2_id INTEGER,
            player2_name VARCHAR(255),
            player2_seed FLOAT,
            player2_entry VARCHAR(50),
            player2_hand VARCHAR(1),
            player2_height_cm FLOAT,
            player2_country_code VARCHAR(3),
            player2_age FLOAT,
            player2_rank FLOAT,
            player2_rank_points FLOAT,
            status VARCHAR(20) DEFAULT 'scheduled',
            actual_winner_id INTEGER,
            score VARCHAR(50),
            match_type VARCHAR(50),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # Create match_predictions table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS match_predictions (
            id SERIAL PRIMARY KEY,
            match_id VARCHAR(50) REFERENCES upcoming_matches(match_id),
            player1_win_probability FLOAT,
            player2_win_probability FLOAT,
            predicted_winner_id INTEGER,
            prediction_confidence FLOAT,
            prediction_correct BOOLEAN,
            model_version VARCHAR(20),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            features_used TEXT[]
        );
        """)
        
        conn.commit()
        logger.info("Tables created or already exist")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error creating tables: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

def get_player_details(api_client: TennisAPIClient, player_id: str) -> Dict[str, Any]:
    """Fetch player details from the API or database."""
    conn = get_database_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Check if player exists in database
        cursor.execute("SELECT * FROM players WHERE id = %s", (player_id,))
        player = cursor.fetchone()
        
        if player:
            logger.info(f"Found player {player_id} in database")
            return dict(player)
        
        # Fetch from API if not in database
        player_data = api_client.get_player_profile(player_id)
        if not player_data:
            logger.warning(f"Could not get data for player {player_id}")
            return {}
        
        # Format player data
        player = {
            "id": int(player_id),
            "name": player_data.get("name", ""),
            "hand": format_player_hand(player_data.get("hand", "")),
            "height_cm": player_data.get("height_cm"),
            "country_code": player_data.get("country_code", ""),
            "birth_date": player_data.get("birth_date"),
            "turned_pro": player_data.get("turned_pro"),
            "weight_kg": player_data.get("weight_kg"),
            "birthplace": player_data.get("birthplace", ""),
            "residence": player_data.get("residence", ""),
            "coach": player_data.get("coach", ""),
            "player_status": player_data.get("status", "")
        }
        
        # Insert to database
        columns = ", ".join(player.keys())
        placeholders = ", ".join(["%s"] * len(player))
        
        query = f"""
        INSERT INTO players ({columns}, last_updated, created_at)
        VALUES ({placeholders}, NOW(), NOW())
        ON CONFLICT (id) DO UPDATE SET
            name = EXCLUDED.name,
            hand = EXCLUDED.hand,
            height_cm = EXCLUDED.height_cm,
            country_code = EXCLUDED.country_code,
            birth_date = EXCLUDED.birth_date,
            turned_pro = EXCLUDED.turned_pro,
            weight_kg = EXCLUDED.weight_kg,
            birthplace = EXCLUDED.birthplace,
            residence = EXCLUDED.residence,
            coach = EXCLUDED.coach,
            player_status = EXCLUDED.player_status,
            last_updated = NOW()
        RETURNING *;
        """
        
        cursor.execute(query, list(player.values()))
        conn.commit()
        player = cursor.fetchone()
        
        return dict(player)
    except Exception as e:
        conn.rollback()
        logger.error(f"Error handling player {player_id}: {e}")
        return {}
    finally:
        cursor.close()
        conn.close()

def fetch_upcoming_tournaments(api_client: TennisAPIClient, days_ahead: int = 14) -> List[Dict[str, Any]]:
    """Fetch upcoming tournaments from the API."""
    start_date = datetime.now().strftime("%Y-%m-%d")
    end_date = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    
    logger.info(f"Fetching tournaments from {start_date} to {end_date}")
    
    try:
        # Get tournament calendar
        calendar_data = api_client.get_tournament_calendar(start_date, end_date)
        if not calendar_data or "tournaments" not in calendar_data:
            logger.warning("No tournament data received from API")
            return []
        
        tournaments = calendar_data.get("tournaments", [])
        logger.info(f"Found {len(tournaments)} upcoming tournaments")
        
        # Filter out tournaments that haven't started
        upcoming_tournaments = []
        for tournament in tournaments:
            tournament_start = datetime.strptime(tournament.get("start_date", "1900-01-01"), "%Y-%m-%d")
            if tournament_start >= datetime.now():
                upcoming_tournaments.append(tournament)
        
        logger.info(f"Filtered to {len(upcoming_tournaments)} tournaments that haven't started yet")
        return upcoming_tournaments
    except Exception as e:
        logger.error(f"Error fetching upcoming tournaments: {e}")
        return []

def fetch_tournament_details(api_client: TennisAPIClient, tournament_id: str) -> Dict[str, Any]:
    """Fetch detailed information about a tournament from the API."""
    try:
        tournament_info = api_client.get_tournament_info(tournament_id)
        if not tournament_info:
            logger.warning(f"No data received for tournament {tournament_id}")
            return {}
        
        # Format tournament data
        tournament = {
            "tournament_id": tournament_id,
            "tournament_name": tournament_info.get("name", ""),
            "surface": format_surface(tournament_info.get("surface", "")),
            "tournament_level": format_tournament_level(tournament_info.get("level", "")),
            "tournament_date": tournament_info.get("start_date"),
            "draw_size": tournament_info.get("draw_size"),
            "match_type": tournament_info.get("type", "").lower()
        }
        
        return tournament
    except Exception as e:
        logger.error(f"Error fetching tournament details for {tournament_id}: {e}")
        return {}

def process_match_for_upcoming(match: Dict[str, Any], tournament_info: Dict[str, Any], api_client: TennisAPIClient) -> Dict[str, Any]:
    """Process match data for storage in upcoming_matches table."""
    try:
        # Generate unique match ID
        tournament_id = tournament_info.get("tournament_id", "")
        player1_id = match.get("player1_id", "")
        player2_id = match.get("player2_id", "")
        round_name = match.get("round", "")
        match_id = f"{tournament_id}_{player1_id}_{player2_id}_{round_name}"
        
        # Get player details
        player1 = get_player_details(api_client, player1_id) if player1_id else {}
        player2 = get_player_details(api_client, player2_id) if player2_id else {}
        
        # Calculate player ages if birth dates available
        player1_age = None
        if player1.get("birth_date") and tournament_info.get("tournament_date"):
            player1_age = calculate_player_age(player1.get("birth_date"), tournament_info.get("tournament_date"))
        
        player2_age = None
        if player2.get("birth_date") and tournament_info.get("tournament_date"):
            player2_age = calculate_player_age(player2.get("birth_date"), tournament_info.get("tournament_date"))
        
        # Format match data
        match_data = {
            "match_id": match_id,
            "tournament_id": tournament_id,
            "tournament_name": tournament_info.get("tournament_name", ""),
            "surface": tournament_info.get("surface", ""),
            "tournament_level": tournament_info.get("tournament_level", ""),
            "tournament_date": tournament_info.get("tournament_date"),
            "match_date": match.get("scheduled_time"),
            "round": match.get("round", ""),
            
            "player1_id": int(player1_id) if player1_id else None,
            "player1_name": player1.get("name", ""),
            "player1_seed": match.get("player1_seed"),
            "player1_entry": match.get("player1_entry", ""),
            "player1_hand": player1.get("hand", ""),
            "player1_height_cm": player1.get("height_cm"),
            "player1_country_code": player1.get("country_code", ""),
            "player1_age": player1_age,
            "player1_rank": match.get("player1_rank"),
            "player1_rank_points": match.get("player1_ranking_points"),
            
            "player2_id": int(player2_id) if player2_id else None,
            "player2_name": player2.get("name", ""),
            "player2_seed": match.get("player2_seed"),
            "player2_entry": match.get("player2_entry", ""),
            "player2_hand": player2.get("hand", ""),
            "player2_height_cm": player2.get("height_cm"),
            "player2_country_code": player2.get("country_code", ""),
            "player2_age": player2_age,
            "player2_rank": match.get("player2_rank"),
            "player2_rank_points": match.get("player2_ranking_points"),
            
            "status": "scheduled",
            "match_type": tournament_info.get("match_type", "")
        }
        
        return match_data
    except Exception as e:
        logger.error(f"Error processing match: {e}")
        return {}

def fetch_tournament_matches(api_client: TennisAPIClient, tournament_id: str, tournament_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fetch and process all matches for an upcoming tournament."""
    try:
        results_data = api_client.get_tournament_results(tournament_id)
        if not results_data or "matches" not in results_data:
            logger.warning(f"No match data received for tournament {tournament_id}")
            return []
        
        matches = results_data.get("matches", [])
        logger.info(f"Found {len(matches)} matches for tournament {tournament_id}")
        
        # Process each match
        upcoming_matches = []
        for match in matches:
            # Check if match has both players (might be TBD in some rounds)
            if not match.get("player1_id") or not match.get("player2_id"):
                continue
            
            # Process match data
            match_data = process_match_for_upcoming(match, tournament_info, api_client)
            if match_data:
                upcoming_matches.append(match_data)
        
        logger.info(f"Processed {len(upcoming_matches)} valid matches for tournament {tournament_id}")
        return upcoming_matches
    except Exception as e:
        logger.error(f"Error fetching matches for tournament {tournament_id}: {e}")
        return []

def save_upcoming_matches(matches: List[Dict[str, Any]]):
    """Save upcoming matches to the database."""
    if not matches:
        logger.info("No matches to save")
        return
    
    conn = get_database_connection()
    cursor = conn.cursor()
    
    try:
        # Prepare the query
        keys = matches[0].keys()
        columns = ", ".join(keys)
        placeholders = ", ".join(["%s"] * len(keys))
        
        # Prepare update clause for conflict resolution
        update_parts = [f"{key} = EXCLUDED.{key}" for key in keys if key != "match_id"]
        update_parts.append("updated_at = NOW()")
        update_clause = ", ".join(update_parts)
        
        query = f"""
        INSERT INTO upcoming_matches ({columns})
        VALUES ({placeholders})
        ON CONFLICT (match_id) DO UPDATE SET
        {update_clause};
        """
        
        # Execute for each match
        for match in matches:
            values = [match.get(key) for key in keys]
            cursor.execute(query, values)
        
        conn.commit()
        logger.info(f"Successfully saved {len(matches)} upcoming matches")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error saving upcoming matches: {e}")
    finally:
        cursor.close()
        conn.close()

def update_match_status():
    """Update the status of matches that have started or completed."""
    conn = get_database_connection()
    cursor = conn.cursor()
    
    try:
        # Find matches with match_date in the past
        cursor.execute("""
        UPDATE upcoming_matches
        SET status = 'in_progress'
        WHERE status = 'scheduled' AND match_date < NOW();
        """)
        
        # Check if any matches have results in the matches table
        cursor.execute("""
        UPDATE upcoming_matches um
        SET 
            status = 'completed',
            actual_winner_id = m.winner_id,
            score = m.score
        FROM matches m
        WHERE 
            um.tournament_id = m.tournament_id AND
            ((um.player1_id = m.winner_id AND um.player2_id = m.loser_id) OR
             (um.player1_id = m.loser_id AND um.player2_id = m.winner_id)) AND
            um.status != 'completed';
        """)
        
        updated_rows = cursor.rowcount
        conn.commit()
        logger.info(f"Updated status for {updated_rows} matches")
        
        # Update prediction correctness
        cursor.execute("""
        UPDATE match_predictions mp
        SET prediction_correct = (um.actual_winner_id = mp.predicted_winner_id)
        FROM upcoming_matches um
        WHERE 
            mp.match_id = um.match_id AND
            um.status = 'completed' AND
            mp.prediction_correct IS NULL;
        """)
        
        updated_predictions = cursor.rowcount
        conn.commit()
        logger.info(f"Updated prediction correctness for {updated_predictions} predictions")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error updating match status: {e}")
    finally:
        cursor.close()
        conn.close()

def main(days_ahead: int = 14):
    """Main function to fetch and store upcoming matches."""
    logger.info("Starting to fetch upcoming tennis matches")
    
    # Initialize API client
    api_client = TennisAPIClient()
    
    # Create tables if they don't exist
    create_upcoming_matches_table()
    
    # Fetch tournaments
    tournaments = fetch_upcoming_tournaments(api_client, days_ahead)
    
    # Process each tournament
    total_matches = 0
    for tournament in tournaments:
        tournament_id = tournament.get("id")
        if not tournament_id:
            continue
        
        # Get tournament details
        tournament_info = fetch_tournament_details(api_client, tournament_id)
        if not tournament_info:
            continue
        
        # Fetch matches for this tournament
        matches = fetch_tournament_matches(api_client, tournament_id, tournament_info)
        if matches:
            save_upcoming_matches(matches)
            total_matches += len(matches)
    
    logger.info(f"Successfully processed {total_matches} matches from {len(tournaments)} tournaments")
    
    # Update status of existing matches
    update_match_status()
    
    logger.info("Completed fetching upcoming tennis matches")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch upcoming tennis matches")
    parser.add_argument("--days", type=int, default=14, help="Number of days ahead to fetch tournaments for")
    args = parser.parse_args()
    
    main(args.days) 