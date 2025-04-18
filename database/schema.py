"""
Tennis Predictor Database Schema

This file documents the database schema used in the Tennis Predictor project.
It provides information about tables, their columns, and relationships.
"""

from typing import Dict, List

# Define matches table schema
MATCHES_TABLE = {
    "name": "matches",
    "description": "Historical tennis match data",
    "columns": [
        {"name": "id", "type": "INTEGER", "description": "Primary key, auto-incremented"},
        {"name": "tournament_id", "type": "VARCHAR(50)", "description": "Unique tournament identifier"},
        {"name": "tournament_name", "type": "VARCHAR(255)", "description": "Name of the tournament"},
        {"name": "surface", "type": "VARCHAR(50)", "description": "Playing surface (e.g., 'hard' , 'clay', 'grass')"},
        {"name": "draw_size", "type": "FLOAT", "description": "Tournament draw size"},
        {"name": "tournament_level", "type": "VARCHAR(50)", "description": "Tournament level code (G: Grand Slam, M: Masters, etc.)"},
        {"name": "tournament_date", "type": "TIMESTAMP", "description": "Date of the tournament"},
        {"name": "match_num", "type": "BIGINT", "description": "Match number within tournament"},
        {"name": "winner_id", "type": "INTEGER", "description": "ID of match winner"},
        {"name": "winner_seed", "type": "FLOAT", "description": "Seed of match winner"},
        {"name": "winner_entry", "type": "VARCHAR(50)", "description": "Entry status of winner (e.g., Q: Qualifier, WC: Wild Card)"},
        {"name": "winner_name", "type": "VARCHAR(255)", "description": "Name of match winner"},
        {"name": "winner_hand", "type": "VARCHAR(1)", "description": "Winner's playing hand (R: Right, L: Left)"},
        {"name": "winner_height_cm", "type": "FLOAT", "description": "Winner's height in cm"},
        {"name": "winner_country_code", "type": "VARCHAR(3)", "description": "Winner's country code"},
        {"name": "winner_age", "type": "FLOAT", "description": "Winner's age"},
        {"name": "loser_id", "type": "INTEGER", "description": "ID of match loser"},
        {"name": "loser_seed", "type": "FLOAT", "description": "Seed of match loser"},
        {"name": "loser_entry", "type": "VARCHAR(50)", "description": "Entry status of loser"},
        {"name": "loser_name", "type": "VARCHAR(255)", "description": "Name of match loser"},
        {"name": "loser_hand", "type": "VARCHAR(1)", "description": "Loser's playing hand"},
        {"name": "loser_height_cm", "type": "FLOAT", "description": "Loser's height in cm"},
        {"name": "loser_country_code", "type": "VARCHAR(3)", "description": "Loser's country code"},
        {"name": "loser_age", "type": "FLOAT", "description": "Loser's age"},
        {"name": "score", "type": "VARCHAR(50)", "description": "Match score"},
        {"name": "best_of", "type": "INTEGER", "description": "Best of X sets format (typically 3 or 5)"},
        {"name": "round", "type": "VARCHAR(50)", "description": "Tournament round"},
        {"name": "minutes", "type": "FLOAT", "description": "Match duration in minutes"},
        {"name": "winner_aces", "type": "FLOAT", "description": "Winner's aces count"},
        {"name": "winner_double_faults", "type": "FLOAT", "description": "Winner's double faults count"},
        {"name": "winner_serve_points", "type": "FLOAT", "description": "Winner's serve points"},
        {"name": "winner_first_serves_in", "type": "FLOAT", "description": "Winner's first serves in"},
        {"name": "winner_first_serve_points_won", "type": "FLOAT", "description": "Winner's first serve points won"},
        {"name": "winner_second_serve_points_won", "type": "FLOAT", "description": "Winner's second serve points won"},
        {"name": "winner_service_games", "type": "FLOAT", "description": "Winner's service games played"},
        {"name": "winner_break_points_saved", "type": "FLOAT", "description": "Winner's break points saved"},
        {"name": "winner_break_points_faced", "type": "FLOAT", "description": "Winner's break points faced"},
        {"name": "loser_aces", "type": "FLOAT", "description": "Loser's aces count"},
        {"name": "loser_double_faults", "type": "FLOAT", "description": "Loser's double faults count"},
        {"name": "loser_serve_points", "type": "FLOAT", "description": "Loser's serve points"},
        {"name": "loser_first_serves_in", "type": "FLOAT", "description": "Loser's first serves in"},
        {"name": "loser_first_serve_points_won", "type": "FLOAT", "description": "Loser's first serve points won"},
        {"name": "loser_second_serve_points_won", "type": "FLOAT", "description": "Loser's second serve points won"},
        {"name": "loser_service_games", "type": "FLOAT", "description": "Loser's service games played"},
        {"name": "loser_break_points_saved", "type": "FLOAT", "description": "Loser's break points saved"},
        {"name": "loser_break_points_faced", "type": "FLOAT", "description": "Loser's break points faced"},
        {"name": "winner_rank", "type": "FLOAT", "description": "Winner's ATP/WTA rank"},
        {"name": "winner_rank_points", "type": "FLOAT", "description": "Winner's ranking points"},
        {"name": "loser_rank", "type": "FLOAT", "description": "Loser's ATP/WTA rank"},
        {"name": "loser_rank_points", "type": "FLOAT", "description": "Loser's ranking points"},
        {"name": "match_type", "type": "VARCHAR(50)", "description": "Type of match (e.g., 'atp', 'wta')"},
        # Elo rating columns
        {"name": "winner_elo", "type": "FLOAT", "description": "Winner's Elo rating before the match"},
        {"name": "loser_elo", "type": "FLOAT", "description": "Loser's Elo rating before the match"},
        {"name": "winner_matches", "type": "INTEGER", "description": "Number of matches played by winner before this match"},
        {"name": "loser_matches", "type": "INTEGER", "description": "Number of matches played by loser before this match"}
    ]
}

# Define match_features table schema
MATCH_FEATURES_TABLE = {
    "name": "match_features",
    "description": "Engineered features for tennis match prediction",
    "columns": [
        {"name": "id", "type": "SERIAL", "description": "Primary key, auto-incremented"},
        {"name": "match_id", "type": "BIGINT", "description": "Foreign key referencing matches.id (unique)"},
        {"name": "player1_id", "type": "BIGINT", "description": "Player 1 ID"},
        {"name": "player2_id", "type": "BIGINT", "description": "Player 2 ID"},
        {"name": "surface", "type": "VARCHAR(50)", "description": "Playing surface"},
        {"name": "tournament_date", "type": "DATE", "description": "Tournament date"},
        {"name": "tournament_level", "type": "VARCHAR(50)", "description": "Tournament level code (G: Grand Slam, M: Masters, etc.)"},
        {"name": "result", "type": "INTEGER", "description": "Match result (1 if player1 won, 0 if player2 won)"},
        {"name": "player_elo_diff", "type": "DOUBLE PRECISION", "description": "Difference in Elo ratings between players"},
        {"name": "win_rate_5_diff", "type": "DOUBLE PRECISION", "description": "Difference in 5-match win rates"},
        {"name": "win_streak_diff", "type": "BIGINT", "description": "Difference in win streaks"},
        {"name": "loss_streak_diff", "type": "BIGINT", "description": "Difference in loss streaks"},
        {"name": "win_rate_hard_5_diff", "type": "DOUBLE PRECISION", "description": "Difference in hard court 5-match win rates"},
        {"name": "win_rate_clay_5_diff", "type": "DOUBLE PRECISION", "description": "Difference in clay court 5-match win rates"},
        {"name": "win_rate_grass_5_diff", "type": "DOUBLE PRECISION", "description": "Difference in grass court 5-match win rates"},
        {"name": "win_rate_carpet_5_diff", "type": "DOUBLE PRECISION", "description": "Difference in carpet court 5-match win rates"},
        {"name": "win_rate_hard_overall_diff", "type": "DOUBLE PRECISION", "description": "Difference in hard court overall win rates"},
        {"name": "win_rate_clay_overall_diff", "type": "DOUBLE PRECISION", "description": "Difference in clay court overall win rates"},
        {"name": "win_rate_grass_overall_diff", "type": "DOUBLE PRECISION", "description": "Difference in grass court overall win rates"},
        {"name": "win_rate_carpet_overall_diff", "type": "DOUBLE PRECISION", "description": "Difference in carpet court overall win rates"},
        {"name": "serve_efficiency_5_diff", "type": "DOUBLE PRECISION", "description": "Difference in 5-match serve efficiency"},
        {"name": "first_serve_pct_5_diff", "type": "DOUBLE PRECISION", "description": "Difference in 5-match first serve percentage"},
        {"name": "first_serve_win_pct_5_diff", "type": "DOUBLE PRECISION", "description": "Difference in 5-match first serve win percentage"},
        {"name": "second_serve_win_pct_5_diff", "type": "DOUBLE PRECISION", "description": "Difference in 5-match second serve win percentage"},
        {"name": "ace_pct_5_diff", "type": "DOUBLE PRECISION", "description": "Difference in 5-match ace percentage"},
        {"name": "bp_saved_pct_5_diff", "type": "DOUBLE PRECISION", "description": "Difference in 5-match break points saved percentage"},
        {"name": "return_efficiency_5_diff", "type": "DOUBLE PRECISION", "description": "Difference in 5-match return efficiency"},
        {"name": "bp_conversion_pct_5_diff", "type": "DOUBLE PRECISION", "description": "Difference in 5-match break point conversion percentage"},
        {"name": "player1_win_rate_5", "type": "DOUBLE PRECISION", "description": "Player 1's 5-match win rate"},
        {"name": "player2_win_rate_5", "type": "DOUBLE PRECISION", "description": "Player 2's 5-match win rate"},
        {"name": "player1_win_streak", "type": "BIGINT", "description": "Player 1's win streak"},
        {"name": "player2_win_streak", "type": "BIGINT", "description": "Player 2's win streak"},
        {"name": "player1_loss_streak", "type": "BIGINT", "description": "Player 1's loss streak"},
        {"name": "player2_loss_streak", "type": "BIGINT", "description": "Player 2's loss streak"},
        {"name": "player1_win_rate_hard_5", "type": "DOUBLE PRECISION", "description": "Player 1's 5-match win rate on hard courts"},
        {"name": "player2_win_rate_hard_5", "type": "DOUBLE PRECISION", "description": "Player 2's 5-match win rate on hard courts"},
        {"name": "player1_win_rate_clay_5", "type": "DOUBLE PRECISION", "description": "Player 1's 5-match win rate on clay courts"},
        {"name": "player2_win_rate_clay_5", "type": "DOUBLE PRECISION", "description": "Player 2's 5-match win rate on clay courts"},
        {"name": "player1_win_rate_grass_5", "type": "DOUBLE PRECISION", "description": "Player 1's 5-match win rate on grass courts"},
        {"name": "player2_win_rate_grass_5", "type": "DOUBLE PRECISION", "description": "Player 2's 5-match win rate on grass courts"},
        {"name": "player1_win_rate_carpet_5", "type": "DOUBLE PRECISION", "description": "Player 1's 5-match win rate on carpet courts"},
        {"name": "player2_win_rate_carpet_5", "type": "DOUBLE PRECISION", "description": "Player 2's 5-match win rate on carpet courts"},
        {"name": "player1_win_rate_hard_overall", "type": "DOUBLE PRECISION", "description": "Player 1's overall win rate on hard courts"},
        {"name": "player2_win_rate_hard_overall", "type": "DOUBLE PRECISION", "description": "Player 2's overall win rate on hard courts"},
        {"name": "player1_win_rate_clay_overall", "type": "DOUBLE PRECISION", "description": "Player 1's overall win rate on clay courts"},
        {"name": "player2_win_rate_clay_overall", "type": "DOUBLE PRECISION", "description": "Player 2's overall win rate on clay courts"},
        {"name": "player1_win_rate_grass_overall", "type": "DOUBLE PRECISION", "description": "Player 1's overall win rate on grass courts"},
        {"name": "player2_win_rate_grass_overall", "type": "DOUBLE PRECISION", "description": "Player 2's overall win rate on grass courts"},
        {"name": "player1_win_rate_carpet_overall", "type": "DOUBLE PRECISION", "description": "Player 1's overall win rate on carpet courts"},
        {"name": "player2_win_rate_carpet_overall", "type": "DOUBLE PRECISION", "description": "Player 2's overall win rate on carpet courts"},
        {"name": "player1_serve_efficiency_5", "type": "DOUBLE PRECISION", "description": "Player 1's 5-match serve efficiency"},
        {"name": "player2_serve_efficiency_5", "type": "DOUBLE PRECISION", "description": "Player 2's 5-match serve efficiency"},
        {"name": "player1_first_serve_pct_5", "type": "DOUBLE PRECISION", "description": "Player 1's 5-match first serve percentage"},
        {"name": "player2_first_serve_pct_5", "type": "DOUBLE PRECISION", "description": "Player 2's 5-match first serve percentage"},
        {"name": "player1_first_serve_win_pct_5", "type": "DOUBLE PRECISION", "description": "Player 1's 5-match first serve win percentage"},
        {"name": "player2_first_serve_win_pct_5", "type": "DOUBLE PRECISION", "description": "Player 2's 5-match first serve win percentage"},
        {"name": "player1_second_serve_win_pct_5", "type": "DOUBLE PRECISION", "description": "Player 1's 5-match second serve win percentage"},
        {"name": "player2_second_serve_win_pct_5", "type": "DOUBLE PRECISION", "description": "Player 2's 5-match second serve win percentage"},
        {"name": "player1_ace_pct_5", "type": "DOUBLE PRECISION", "description": "Player 1's 5-match ace percentage"},
        {"name": "player2_ace_pct_5", "type": "DOUBLE PRECISION", "description": "Player 2's 5-match ace percentage"},
        {"name": "player1_bp_saved_pct_5", "type": "DOUBLE PRECISION", "description": "Player 1's 5-match break points saved percentage"},
        {"name": "player2_bp_saved_pct_5", "type": "DOUBLE PRECISION", "description": "Player 2's 5-match break points saved percentage"},
        {"name": "player1_return_efficiency_5", "type": "DOUBLE PRECISION", "description": "Player 1's 5-match return efficiency"},
        {"name": "player2_return_efficiency_5", "type": "DOUBLE PRECISION", "description": "Player 2's 5-match return efficiency"},
        {"name": "player1_bp_conversion_pct_5", "type": "DOUBLE PRECISION", "description": "Player 1's 5-match break point conversion percentage"},
        {"name": "player2_bp_conversion_pct_5", "type": "DOUBLE PRECISION", "description": "Player 2's 5-match break point conversion percentage"},
        {"name": "created_at", "type": "TIMESTAMP WITH TIME ZONE", "description": "Record creation timestamp"},
        {"name": "updated_at", "type": "TIMESTAMP WITH TIME ZONE", "description": "Record last update timestamp"}
    ]
}

# Define players table schema
PLAYERS_TABLE = {
    "name": "players",
    "description": "Player information and attributes",
    "columns": [
        {"name": "id", "type": "INTEGER", "description": "Primary key, player ID from API"},
        {"name": "name", "type": "VARCHAR(255)", "description": "Player's full name"},
        {"name": "hand", "type": "VARCHAR(1)", "description": "Playing hand (R: Right, L: Left)"},
        {"name": "height_cm", "type": "FLOAT", "description": "Height in centimeters"},
        {"name": "country_code", "type": "VARCHAR(3)", "description": "Country code (e.g., USA, FRA)"},
        {"name": "birth_date", "type": "DATE", "description": "Player's birth date"},
        {"name": "turned_pro", "type": "INTEGER", "description": "Year player turned professional"},
        {"name": "weight_kg", "type": "FLOAT", "description": "Weight in kilograms"},
        {"name": "birthplace", "type": "VARCHAR(255)", "description": "Player's birthplace"},
        {"name": "residence", "type": "VARCHAR(255)", "description": "Player's current residence"},
        {"name": "coach", "type": "VARCHAR(255)", "description": "Current coach name"},
        {"name": "player_status", "type": "VARCHAR(50)", "description": "Current player status"},
        {"name": "last_updated", "type": "TIMESTAMP WITH TIME ZONE", "description": "Last time player data was updated"},
        {"name": "created_at", "type": "TIMESTAMP WITH TIME ZONE", "description": "Record creation timestamp"}
    ]
}

# Define relationships
RELATIONSHIPS = [
    {"from_table": "match_features", "from_column": "match_id", "to_table": "matches", "to_column": "id", "type": "Foreign Key"},
    {"from_table": "matches", "from_column": "winner_id", "to_table": "players", "to_column": "id", "type": "Foreign Key"},
    {"from_table": "matches", "from_column": "loser_id", "to_table": "players", "to_column": "id", "type": "Foreign Key"}
]

# Combine all schema information
SCHEMA = {
    "tables": [MATCHES_TABLE, MATCH_FEATURES_TABLE, PLAYERS_TABLE],
    "relationships": RELATIONSHIPS
}

def get_schema_info() -> Dict:
    """
    Returns a dictionary containing database schema information
    for easy programmatic access
    """
    return SCHEMA

def print_schema_summary() -> None:
    """
    Prints a readable summary of the database schema
    """
    schema = get_schema_info()
    
    print("Tennis Predictor Database Schema Summary")
    print("=" * 50)
    
    for table in schema["tables"]:
        print(f"\nTable: {table['name']}")
        print(f"Description: {table['description']}")
        print("-" * 50)
        
        for column in table["columns"]:
            print(f"  {column['name']}: {column['type']}")
            print(f"    {column['description']}")
    
    print("\nRelationships:")
    print("-" * 50)
    for rel in schema["relationships"]:
        print(f"  - {rel['from_table']}.{rel['from_column']} → {rel['to_table']}.{rel['to_column']} ({rel['type']})")

if __name__ == "__main__":
    print_schema_summary() 