import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CLEANED_DATA_DIR = DATA_DIR / "cleaned"
OUTPUT_DIR = PROJECT_ROOT / "predictor" / "output"

# Create necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Input and output files
INPUT_FILE = CLEANED_DATA_DIR / "cleaned_dataset_with_elo.csv"
OUTPUT_FILE = DATA_DIR / "features_v2.csv"

# Standard surface definitions
SURFACE_HARD = 'Hard'
SURFACE_CLAY = 'Clay'
SURFACE_GRASS = 'Grass'
SURFACE_CARPET = 'Carpet'
STANDARD_SURFACES = [SURFACE_HARD, SURFACE_CLAY, SURFACE_GRASS, SURFACE_CARPET]

def load_data() -> pd.DataFrame:
    """
    Load the tennis match dataset.
    
    Returns:
        pd.DataFrame: The loaded dataset
    """
    logger.info(f"Loading data from {INPUT_FILE}...")
    
    # Read the CSV
    df = pd.read_csv(INPUT_FILE)
    
    # Convert date column to datetime
    df['tourney_date'] = pd.to_datetime(df['tourney_date'])
    
    # Standardize surface names
    df['surface'] = df['surface'].str.lower()
    df['surface'] = df['surface'].replace({
        'hard': SURFACE_HARD,
        'clay': SURFACE_CLAY,
        'grass': SURFACE_GRASS,
        'carpet': SURFACE_CARPET
    })
    
    # Sort by date to ensure chronological order
    df = df.sort_values('tourney_date').reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} matches spanning from {df['tourney_date'].min().date()} to {df['tourney_date'].max().date()}")
    return df

def calculate_win_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate win rates and streaks for players without introducing player position bias.
    
    Args:
        df: Match dataset
        
    Returns:
        DataFrame with win rate features
    """
    logger.info("Calculating win rates and streaks...")
    
    # Create a player-centric view of the data - one row per player per match
    matches = []
    
    # Process matches chronologically with progress bar
    for idx, row in tqdm(df.sort_values('tourney_date').iterrows(), 
                          total=len(df), 
                          desc="Processing matches for win rates", 
                          unit="match"):
        # Process winner
        winner_dict = {
            'match_id': idx,
            'player_id': row['winner_id'],
            'opponent_id': row['loser_id'],
            'tourney_date': row['tourney_date'],
            'surface': row['surface'],
            'result': 1  # 1 means win
        }
        matches.append(winner_dict)
        
        # Process loser
        loser_dict = {
            'match_id': idx,
            'player_id': row['loser_id'],
            'opponent_id': row['winner_id'],
            'tourney_date': row['tourney_date'], 
            'surface': row['surface'],
            'result': 0  # 0 means loss
        }
        matches.append(loser_dict)
    
    # Create player-centric dataframe
    player_df = pd.DataFrame(matches)
    
    # Sort by player and date
    player_df = player_df.sort_values(['player_id', 'tourney_date'])
    
    # Calculate overall win rates over different windows
    time_windows = [5]  # Focus on recent 5 matches as indicated by feature importance
    
    logger.info("Calculating rolling win rates...")
    # Overall win rates
    for window in time_windows:
        player_df[f'win_rate_{window}'] = (
            player_df.groupby('player_id')['result']
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
    
    # Surface-specific win rates
    for surface in tqdm(STANDARD_SURFACES, desc="Processing surface-specific win rates", unit="surface"):
        # Create mask for this surface
        surface_mask = player_df['surface'] == surface
        
        for window in time_windows:
            # Initialize column with NaN values
            player_df[f'win_rate_{surface}_{window}'] = np.nan
            
            # Group by player and calculate win rate for this surface
            surface_rates = (
                player_df[surface_mask]
                .groupby('player_id')['result']
                .rolling(window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            
            # Update values for this surface
            player_df.loc[surface_mask, f'win_rate_{surface}_{window}'] = surface_rates
            
            # Forward fill the values - keep previous surface win rate until next match on this surface
            player_df[f'win_rate_{surface}_{window}'] = (
                player_df
                .groupby('player_id')[f'win_rate_{surface}_{window}']
                .transform(lambda x: x.ffill())
            )
            
    # Calculate overall win rate per surface (not just based on recent matches)
    logger.info("Calculating overall surface win rates...")
    for surface in STANDARD_SURFACES:
        # For each player, calculate their overall win rate on each surface
        surface_overall = (
            player_df[player_df['surface'] == surface]
            .groupby('player_id')['result']
            .expanding()
            .mean()
            .reset_index(level=0, drop=True)
        )
        player_df.loc[player_df['surface'] == surface, f'win_rate_{surface}_overall'] = surface_overall
        
        # Forward fill these values
        player_df[f'win_rate_{surface}_overall'] = (
            player_df
            .groupby('player_id')[f'win_rate_{surface}_overall']
            .transform(lambda x: x.ffill())
        )
    
    # Calculate win and loss streaks
    logger.info("Calculating win/loss streaks...")
    player_df['win_streak'] = 0
    player_df['loss_streak'] = 0
    
    # Group by player
    unique_players = player_df['player_id'].unique()
    for player_id in tqdm(unique_players, desc="Calculating player streaks", unit="player"):
        group = player_df[player_df['player_id'] == player_id]
        
        # Initialize streaks
        win_streak = 0
        loss_streak = 0
        win_streaks = []
        loss_streaks = []
        
        # Calculate streaks for each match
        for result in group['result']:
            if result == 1:  # Win
                win_streak += 1
                loss_streak = 0
            else:  # Loss
                loss_streak += 1
                win_streak = 0
            win_streaks.append(win_streak)
            loss_streaks.append(loss_streak)
        
        # Update the dataframe
        player_df.loc[group.index, 'win_streak'] = win_streaks
        player_df.loc[group.index, 'loss_streak'] = loss_streaks
    
    # Ensure we have no missing values in key columns
    for col in ['win_rate_5', 'win_streak', 'loss_streak']:
        player_df[col] = player_df[col].fillna(0)
    
    # For surface-specific win rates, keep NaN values if a player has never played on that surface
    # XGBoost will handle these NaN values appropriately
    
    return player_df

def prepare_features_for_matches(df: pd.DataFrame, player_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for each match by joining player statistics.
    
    Args:
        df: Original match dataset
        player_df: Player-centric dataset with calculated features
        
    Returns:
        DataFrame with match features
    """
    logger.info("Preparing match features...")
    
    # Create a copy of the match dataframe
    match_df = df.copy()
    
    # Get the features for each player before each match
    features = []
    
    # Process each match with progress bar
    for idx, match in tqdm(match_df.iterrows(), total=len(match_df), desc="Preparing match features", unit="match"):
        match_date = match['tourney_date']
        winner_id = match['winner_id']
        loser_id = match['loser_id']
        
        # Get player stats just before this match (exclude the current match)
        winner_prev = player_df[(player_df['player_id'] == winner_id) & 
                              (player_df['tourney_date'] < match_date)]
        
        loser_prev = player_df[(player_df['player_id'] == loser_id) & 
                             (player_df['tourney_date'] < match_date)]
        
        # If we have previous matches for these players
        if not winner_prev.empty and not loser_prev.empty:
            # Get most recent stats
            winner_stats = winner_prev.iloc[-1]
            loser_stats = loser_prev.iloc[-1]
            
            # Calculate key feature differences (most important according to our analysis)
            match_features = {
                'match_id': idx,
                'tourney_date': match_date,
                'surface': match['surface'],
                'winner_id': winner_id,
                'loser_id': loser_id,
                
                # Elo difference (from the original dataset)
                'elo_diff': match['winner_elo'] - match['loser_elo'],
                
                # Win rate differences
                'win_rate_5_diff': winner_stats.get('win_rate_5', 0) - loser_stats.get('win_rate_5', 0),
                
                # Win/loss streak differences
                'win_streak_diff': winner_stats.get('win_streak', 0) - loser_stats.get('win_streak', 0),
                'loss_streak_diff': winner_stats.get('loss_streak', 0) - loser_stats.get('loss_streak', 0),
                
                # Surface-specific win rates
                f'win_rate_{match["surface"]}_5_diff': 
                    winner_stats.get(f'win_rate_{match["surface"]}_5', np.nan) - 
                    loser_stats.get(f'win_rate_{match["surface"]}_5', np.nan),
                
                # Overall surface win rates
                f'win_rate_{match["surface"]}_overall_diff': 
                    winner_stats.get(f'win_rate_{match["surface"]}_overall', np.nan) - 
                    loser_stats.get(f'win_rate_{match["surface"]}_overall', np.nan),
                
                # Also store raw values for both players to avoid player position bias during prediction
                'winner_win_rate_5': winner_stats.get('win_rate_5', 0),
                'loser_win_rate_5': loser_stats.get('win_rate_5', 0),
                'winner_win_streak': winner_stats.get('win_streak', 0),
                'loser_win_streak': loser_stats.get('win_streak', 0),
                'winner_loss_streak': winner_stats.get('loss_streak', 0),
                'loser_loss_streak': loser_stats.get('loss_streak', 0)
            }
            
            # Add surface-specific win rates for all surfaces
            for surface in STANDARD_SURFACES:
                match_features[f'winner_win_rate_{surface}_5'] = winner_stats.get(f'win_rate_{surface}_5', np.nan)
                match_features[f'loser_win_rate_{surface}_5'] = loser_stats.get(f'win_rate_{surface}_5', np.nan)
                match_features[f'winner_win_rate_{surface}_overall'] = winner_stats.get(f'win_rate_{surface}_overall', np.nan)
                match_features[f'loser_win_rate_{surface}_overall'] = loser_stats.get(f'win_rate_{surface}_overall', np.nan)
            
            features.append(match_features)
        else:
            # For matches without prior history, we only include basic information
            match_features = {
                'match_id': idx,
                'tourney_date': match_date,
                'surface': match['surface'],
                'winner_id': winner_id,
                'loser_id': loser_id,
                'elo_diff': match['winner_elo'] - match['loser_elo'],
            }
            features.append(match_features)
    
    # Create dataframe with all features
    features_df = pd.DataFrame(features)
    
    # Sort by date
    features_df = features_df.sort_values('tourney_date').reset_index(drop=True)
    
    return features_df

def generate_player_symmetric_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate player-symmetric features to avoid player position bias.
    
    Args:
        features_df: DataFrame with calculated features
        
    Returns:
        DataFrame with player-symmetric features
    """
    logger.info("Generating player-symmetric features...")
    
    # Create a copy
    df = features_df.copy()
    
    # Create player-symmetric features required for prediction
    # Create two versions of each match: p1 = winner, p2 = loser and p1 = loser, p2 = winner
    matches = []
    
    # First pass: p1 = winner, p2 = loser (actual match result)
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating winner perspective features", unit="match"):
        match_dict = {
            'match_id': row['match_id'],
            'tourney_date': row['tourney_date'],
            'surface': row['surface'],
            'player1_id': row['winner_id'],
            'player2_id': row['loser_id'],
            'result': 1  # player1 won
        }
        
        # Add features
        # Elo difference
        match_dict['player_elo_diff'] = row.get('elo_diff', 0)
        
        # Win rates
        match_dict['win_rate_5_diff'] = row.get('win_rate_5_diff', 0)
        
        # Streaks
        match_dict['win_streak_diff'] = row.get('win_streak_diff', 0)
        match_dict['loss_streak_diff'] = row.get('loss_streak_diff', 0)
        
        # Surface win rates (preserve NaN values for proper XGBoost handling)
        for surface in STANDARD_SURFACES:
            if f'win_rate_{surface}_5_diff' in row:
                match_dict[f'win_rate_{surface}_5_diff'] = row[f'win_rate_{surface}_5_diff']
            
            if f'win_rate_{surface}_overall_diff' in row:
                match_dict[f'win_rate_{surface}_overall_diff'] = row[f'win_rate_{surface}_overall_diff']
        
        # Add raw player features to ensure consistent prediction regardless of player order
        if 'winner_win_rate_5' in row:
            match_dict['player1_win_rate_5'] = row['winner_win_rate_5']
            match_dict['player2_win_rate_5'] = row['loser_win_rate_5']
            match_dict['player1_win_streak'] = row['winner_win_streak']
            match_dict['player2_win_streak'] = row['loser_win_streak']
            match_dict['player1_loss_streak'] = row['winner_loss_streak']
            match_dict['player2_loss_streak'] = row['loser_loss_streak']
            
            # Surface-specific rates
            for surface in STANDARD_SURFACES:
                if f'winner_win_rate_{surface}_5' in row:
                    match_dict[f'player1_win_rate_{surface}_5'] = row[f'winner_win_rate_{surface}_5']
                    match_dict[f'player2_win_rate_{surface}_5'] = row[f'loser_win_rate_{surface}_5']
                    match_dict[f'player1_win_rate_{surface}_overall'] = row[f'winner_win_rate_{surface}_overall']
                    match_dict[f'player2_win_rate_{surface}_overall'] = row[f'loser_win_rate_{surface}_overall']
        
        matches.append(match_dict)
    
    # Second pass: p1 = loser, p2 = winner (flipped match result)
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating loser perspective features", unit="match"):
        match_dict = {
            'match_id': row['match_id'],
            'tourney_date': row['tourney_date'],
            'surface': row['surface'],
            'player1_id': row['loser_id'],
            'player2_id': row['winner_id'], 
            'result': 0  # player1 lost
        }
        
        # Add features with reversed signs
        match_dict['player_elo_diff'] = -row.get('elo_diff', 0)
        match_dict['win_rate_5_diff'] = -row.get('win_rate_5_diff', 0)
        match_dict['win_streak_diff'] = -row.get('win_streak_diff', 0)
        match_dict['loss_streak_diff'] = -row.get('loss_streak_diff', 0)
        
        # Surface win rates with reversed signs
        for surface in STANDARD_SURFACES:
            if f'win_rate_{surface}_5_diff' in row:
                match_dict[f'win_rate_{surface}_5_diff'] = -row[f'win_rate_{surface}_5_diff']
            
            if f'win_rate_{surface}_overall_diff' in row:
                match_dict[f'win_rate_{surface}_overall_diff'] = -row[f'win_rate_{surface}_overall_diff']
        
        # Add raw player features (swapped)
        if 'winner_win_rate_5' in row:
            match_dict['player1_win_rate_5'] = row['loser_win_rate_5']
            match_dict['player2_win_rate_5'] = row['winner_win_rate_5']
            match_dict['player1_win_streak'] = row['loser_win_streak']
            match_dict['player2_win_streak'] = row['winner_win_streak']
            match_dict['player1_loss_streak'] = row['loser_loss_streak']
            match_dict['player2_loss_streak'] = row['winner_loss_streak']
            
            # Surface-specific rates
            for surface in STANDARD_SURFACES:
                if f'winner_win_rate_{surface}_5' in row:
                    match_dict[f'player1_win_rate_{surface}_5'] = row[f'loser_win_rate_{surface}_5']
                    match_dict[f'player2_win_rate_{surface}_5'] = row[f'winner_win_rate_{surface}_5']
                    match_dict[f'player1_win_rate_{surface}_overall'] = row[f'loser_win_rate_{surface}_overall']
                    match_dict[f'player2_win_rate_{surface}_overall'] = row[f'winner_win_rate_{surface}_overall']
        
        matches.append(match_dict)
    
    # Convert to DataFrame
    symmetric_df = pd.DataFrame(matches)
    
    # Sort by date and match_id
    symmetric_df = symmetric_df.sort_values(['tourney_date', 'match_id']).reset_index(drop=True)
    
    return symmetric_df

def main():
    """Main function to generate features for the tennis prediction model."""
    start_time = datetime.now()
    logger.info(f"Starting feature generation at {start_time}")
    
    # Define main steps and create a progress meter
    steps = ["Loading data", "Calculating win rates", "Preparing match features", 
             "Generating symmetric features", "Saving results"]
    total_steps = len(steps)
    
    # Display overall progress
    logger.info(f"Feature generation process: 0% complete - Starting step 1/{total_steps}: {steps[0]}")
    
    # 1. Load data
    df = load_data()
    logger.info(f"Feature generation process: {100*1/total_steps:.1f}% complete - Starting step 2/{total_steps}: {steps[1]}")
    
    # 2. Calculate win rates and streaks
    player_df = calculate_win_rates(df)
    logger.info(f"Feature generation process: {100*2/total_steps:.1f}% complete - Starting step 3/{total_steps}: {steps[2]}")
    
    # 3. Prepare features for matches
    features_df = prepare_features_for_matches(df, player_df)
    logger.info(f"Feature generation process: {100*3/total_steps:.1f}% complete - Starting step 4/{total_steps}: {steps[3]}")
    
    # 4. Generate player-symmetric features
    symmetric_df = generate_player_symmetric_features(features_df)
    logger.info(f"Feature generation process: {100*4/total_steps:.1f}% complete - Starting step 5/{total_steps}: {steps[4]}")
    
    # 5. Save to output file
    logger.info(f"Saving features to {OUTPUT_FILE}...")
    symmetric_df.to_csv(OUTPUT_FILE, index=False)
    
    # Log completion
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Feature generation process: 100% complete - All steps finished")
    logger.info(f"Feature generation completed in {duration}")
    logger.info(f"Generated features for {len(features_df)} matches")
    logger.info(f"Created {len(symmetric_df)} rows in the symmetric dataset")
    logger.info(f"Output file saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main() 