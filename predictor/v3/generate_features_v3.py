import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import logging
from tqdm import tqdm
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project directories
PROJECT_ROOT = Path(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
DATA_DIR = PROJECT_ROOT / "data"
CLEANED_DATA_DIR = DATA_DIR / "cleaned"
OUTPUT_DIR = DATA_DIR / "v3"
PRED_OUTPUT_DIR = PROJECT_ROOT / "predictor" / "output" / "v3"

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PRED_OUTPUT_DIR, exist_ok=True)

# File paths
INPUT_FILE = CLEANED_DATA_DIR / "cleaned_dataset_with_elo.csv"
OUTPUT_FILE = OUTPUT_DIR / "features_v3.csv"

# Constants
STANDARD_SURFACES = ['hard', 'clay', 'grass']

# Columns for serve and return stats
SERVE_COLS = {
    'winner': ['w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced'],
    'loser': ['l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced']
}

# Rolling window sizes
ROLLING_WINDOWS = [5, 10]

# Standard surface definitions
SURFACE_HARD = 'Hard'
SURFACE_CLAY = 'Clay'
SURFACE_GRASS = 'Grass'
SURFACE_CARPET = 'Carpet'
STANDARD_SURFACES = [SURFACE_HARD, SURFACE_CLAY, SURFACE_GRASS, SURFACE_CARPET]

# Serve and return stats columns
SERVE_STATS_COLUMNS = ['w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced']
RETURN_STATS_COLUMNS = ['l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced']

def load_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load the tennis match dataset.
    
    Args:
        file_path: Path to the input CSV file
        
    Returns:
        DataFrame with tennis match data
    """
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    
    # Convert date columns to datetime
    if 'tourney_date' in df.columns:
        df['tourney_date'] = pd.to_datetime(df['tourney_date'])
    
    # Sort by date
    df = df.sort_values(by='tourney_date').reset_index(drop=True)
    
    # Create a unique match identifier
    df['match_id'] = df.index
    
    # Standardize surface types
    if 'surface' in df.columns:
        df['surface'] = df['surface'].str.lower()
        # Map non-standard surfaces to standard ones
        surface_mapping = {
            'carpet': 'hard',  # Carpet is most similar to hard
            'hard court': 'hard',
            'clay court': 'clay',
            'grass court': 'grass'
        }
        df['surface'] = df['surface'].map(lambda x: surface_mapping.get(x, x))
    
    # Make sure necessary serve stats columns exist
    for player_type in ['winner', 'loser']:
        for col in SERVE_COLS[player_type]:
            if col not in df.columns:
                df[col] = np.nan
    
    # Make sure data types are correct for serve stats
    serve_stats_cols = SERVE_COLS['winner'] + SERVE_COLS['loser']
    for col in serve_stats_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    logger.info(f"Loaded {len(df)} matches from {len(df['tourney_id'].unique())} tournaments")
    
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

def calculate_serve_return_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate serve and return statistics for each player-match.
    
    Args:
        df: Original match dataset
        
    Returns:
        DataFrame with player-centric serve and return stats
    """
    logger.info("Calculating serve and return statistics...")
    
    # Create a player-centric view for serve and return stats
    matches = []
    
    # Process each match chronologically with progress bar
    for idx, row in tqdm(df.sort_values('tourney_date').iterrows(), 
                          total=len(df), 
                          desc="Processing matches for serve/return stats", 
                          unit="match"):
        
        # Check if serve stats are available for this match
        has_serve_stats = not pd.isna(row['w_svpt']) and row['w_svpt'] > 0
        
        # Process winner's serve and return stats
        winner_dict = {
            'match_id': idx,
            'player_id': row['winner_id'],
            'opponent_id': row['loser_id'],
            'tourney_date': row['tourney_date'],
            'surface': row['surface']
        }
        
        # Add serve stats for winner
        if has_serve_stats:
            # Basic serve stats directly from dataset
            winner_dict['aces'] = row['w_ace']
            winner_dict['double_faults'] = row['w_df']
            winner_dict['serve_points'] = row['w_svpt']
            winner_dict['first_serves_in'] = row['w_1stIn']
            winner_dict['first_serve_points_won'] = row['w_1stWon']
            winner_dict['second_serve_points_won'] = row['w_2ndWon']
            winner_dict['service_games'] = row['w_SvGms']
            winner_dict['break_points_saved'] = row['w_bpSaved']
            winner_dict['break_points_faced'] = row['w_bpFaced']
            
            # Calculate derived serve metrics
            # 1. First serve percentage
            winner_dict['first_serve_pct'] = row['w_1stIn'] / row['w_svpt'] if row['w_svpt'] > 0 else np.nan
            
            # 2. First serve win percentage
            winner_dict['first_serve_win_pct'] = row['w_1stWon'] / row['w_1stIn'] if row['w_1stIn'] > 0 else np.nan
            
            # 3. Second serve win percentage
            second_serves = row['w_svpt'] - row['w_1stIn'] if row['w_svpt'] > 0 else 0
            winner_dict['second_serve_win_pct'] = row['w_2ndWon'] / second_serves if second_serves > 0 else np.nan
            
            # 4. Service efficiency (overall serve points won percentage)
            serve_points_won = row['w_1stWon'] + row['w_2ndWon']
            winner_dict['serve_efficiency'] = serve_points_won / row['w_svpt'] if row['w_svpt'] > 0 else np.nan
            
            # 5. Ace percentage
            winner_dict['ace_pct'] = row['w_ace'] / row['w_svpt'] if row['w_svpt'] > 0 else np.nan
            
            # 6. Break points saved percentage
            winner_dict['bp_saved_pct'] = row['w_bpSaved'] / row['w_bpFaced'] if row['w_bpFaced'] > 0 else np.nan
            
            # Add return stats for winner (using loser's serve stats)
            winner_dict['return_points'] = row['l_svpt']
            winner_dict['return_points_won'] = row['l_svpt'] - (row['l_1stWon'] + row['l_2ndWon']) if row['l_svpt'] > 0 else np.nan
            
            # 7. Return efficiency (percentage of opponent's serve points won)
            winner_dict['return_efficiency'] = winner_dict['return_points_won'] / row['l_svpt'] if row['l_svpt'] > 0 else np.nan
            
            # 8. Break points converted
            winner_dict['break_points_converted'] = row['l_bpFaced'] - row['l_bpSaved'] if row['l_bpFaced'] > 0 else 0
            
            # 9. Break point conversion percentage
            winner_dict['bp_conversion_pct'] = winner_dict['break_points_converted'] / row['l_bpFaced'] if row['l_bpFaced'] > 0 else np.nan
        
        matches.append(winner_dict)
        
        # Process loser's serve and return stats
        loser_dict = {
            'match_id': idx,
            'player_id': row['loser_id'],
            'opponent_id': row['winner_id'],
            'tourney_date': row['tourney_date'],
            'surface': row['surface']
        }
        
        # Add serve stats for loser
        if has_serve_stats:
            # Basic serve stats directly from dataset
            loser_dict['aces'] = row['l_ace']
            loser_dict['double_faults'] = row['l_df']
            loser_dict['serve_points'] = row['l_svpt']
            loser_dict['first_serves_in'] = row['l_1stIn']
            loser_dict['first_serve_points_won'] = row['l_1stWon']
            loser_dict['second_serve_points_won'] = row['l_2ndWon']
            loser_dict['service_games'] = row['l_SvGms']
            loser_dict['break_points_saved'] = row['l_bpSaved']
            loser_dict['break_points_faced'] = row['l_bpFaced']
            
            # Calculate derived serve metrics
            # 1. First serve percentage
            loser_dict['first_serve_pct'] = row['l_1stIn'] / row['l_svpt'] if row['l_svpt'] > 0 else np.nan
            
            # 2. First serve win percentage
            loser_dict['first_serve_win_pct'] = row['l_1stWon'] / row['l_1stIn'] if row['l_1stIn'] > 0 else np.nan
            
            # 3. Second serve win percentage
            second_serves = row['l_svpt'] - row['l_1stIn'] if row['l_svpt'] > 0 else 0
            loser_dict['second_serve_win_pct'] = row['l_2ndWon'] / second_serves if second_serves > 0 else np.nan
            
            # 4. Service efficiency (overall serve points won percentage)
            serve_points_won = row['l_1stWon'] + row['l_2ndWon']
            loser_dict['serve_efficiency'] = serve_points_won / row['l_svpt'] if row['l_svpt'] > 0 else np.nan
            
            # 5. Ace percentage
            loser_dict['ace_pct'] = row['l_ace'] / row['l_svpt'] if row['l_svpt'] > 0 else np.nan
            
            # 6. Break points saved percentage
            loser_dict['bp_saved_pct'] = row['l_bpSaved'] / row['l_bpFaced'] if row['l_bpFaced'] > 0 else np.nan
            
            # Add return stats for loser (using winner's serve stats)
            loser_dict['return_points'] = row['w_svpt']
            loser_dict['return_points_won'] = row['w_svpt'] - (row['w_1stWon'] + row['w_2ndWon']) if row['w_svpt'] > 0 else np.nan
            
            # 7. Return efficiency (percentage of opponent's serve points won)
            loser_dict['return_efficiency'] = loser_dict['return_points_won'] / row['w_svpt'] if row['w_svpt'] > 0 else np.nan
            
            # 8. Break points converted
            loser_dict['break_points_converted'] = row['w_bpFaced'] - row['w_bpSaved'] if row['w_bpFaced'] > 0 else 0
            
            # 9. Break point conversion percentage
            loser_dict['bp_conversion_pct'] = loser_dict['break_points_converted'] / row['w_bpFaced'] if row['w_bpFaced'] > 0 else np.nan
        
        matches.append(loser_dict)
    
    # Create player stats dataframe
    player_stats_df = pd.DataFrame(matches)
    
    # Sort by player and date
    player_stats_df = player_stats_df.sort_values(['player_id', 'tourney_date'])
    
    return player_stats_df

def calculate_serve_return_rolling_stats(player_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate rolling averages for serve and return statistics.
    
    Args:
        player_df: Player-centric dataframe with serve and return stats
        
    Returns:
        DataFrame with rolling averages
    """
    logger.info("Calculating rolling serve and return stats...")
    
    # Create a copy of the dataframe
    df = player_df.copy()
    
    # Define the serve and return metrics for which we'll calculate rolling averages
    serve_metrics = [
        'serve_efficiency',
        'first_serve_pct',
        'first_serve_win_pct',
        'second_serve_win_pct',
        'ace_pct',
        'bp_saved_pct'
    ]
    
    return_metrics = [
        'return_efficiency',
        'bp_conversion_pct'
    ]
    
    # Define window sizes for rolling averages
    windows = [5, 10]
    
    # Calculate rolling averages for each metric
    with tqdm(total=len(serve_metrics + return_metrics) * len(windows), 
              desc="Computing rolling stats", 
              unit="metric") as pbar:
        
        # Serve metrics
        for metric in serve_metrics:
            # Skip if the metric is not in the dataframe
            if metric not in df.columns:
                continue
                
            for window in windows:
                # Calculate rolling average
                col_name = f'{metric}_{window}'
                df[col_name] = (
                    df.groupby('player_id')[metric]
                    .rolling(window, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )
                
                # Calculate surface-specific rolling average
                for surface in STANDARD_SURFACES:
                    # Create mask for this surface
                    surface_mask = df['surface'] == surface
                    
                    # Initialize column with NaN values
                    surf_col_name = f'{metric}_{surface}_{window}'
                    df[surf_col_name] = np.nan
                    
                    # Group by player and calculate average for this surface
                    surface_avgs = (
                        df[surface_mask]
                        .groupby('player_id')[metric]
                        .rolling(window, min_periods=1)
                        .mean()
                        .reset_index(level=0, drop=True)
                    )
                    
                    # Update values for this surface
                    df.loc[surface_mask, surf_col_name] = surface_avgs
                    
                    # Forward fill the values
                    df[surf_col_name] = (
                        df
                        .groupby('player_id')[surf_col_name]
                        .transform(lambda x: x.ffill())
                    )
                
                pbar.update(1)
        
        # Return metrics
        for metric in return_metrics:
            # Skip if the metric is not in the dataframe
            if metric not in df.columns:
                continue
                
            for window in windows:
                # Calculate rolling average
                col_name = f'{metric}_{window}'
                df[col_name] = (
                    df.groupby('player_id')[metric]
                    .rolling(window, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )
                
                # Calculate surface-specific rolling average
                for surface in STANDARD_SURFACES:
                    # Create mask for this surface
                    surface_mask = df['surface'] == surface
                    
                    # Initialize column with NaN values
                    surf_col_name = f'{metric}_{surface}_{window}'
                    df[surf_col_name] = np.nan
                    
                    # Group by player and calculate average for this surface
                    surface_avgs = (
                        df[surface_mask]
                        .groupby('player_id')[metric]
                        .rolling(window, min_periods=1)
                        .mean()
                        .reset_index(level=0, drop=True)
                    )
                    
                    # Update values for this surface
                    df.loc[surface_mask, surf_col_name] = surface_avgs
                    
                    # Forward fill the values
                    df[surf_col_name] = (
                        df
                        .groupby('player_id')[surf_col_name]
                        .transform(lambda x: x.ffill())
                    )
                
                pbar.update(1)
    
    return df

def prepare_features_for_matches(df: pd.DataFrame, player_df: pd.DataFrame, serve_return_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for each match by joining player statistics.
    
    Args:
        df: Original match dataset
        player_df: Player-centric dataset with calculated features
        serve_return_df: Player-centric dataset with serve and return stats
        
    Returns:
        DataFrame with match features
    """
    logger.info("Preparing match features...")
    
    # Create a copy of the match dataframe
    match_df = df.copy()
    
    # Get the features for each player before each match
    features = []
    
    # Define serve and return metrics we'll use for feature differences
    serve_return_metrics = [
        'serve_efficiency_5',
        'first_serve_pct_5',
        'first_serve_win_pct_5',
        'second_serve_win_pct_5',
        'ace_pct_5',
        'bp_saved_pct_5',
        'return_efficiency_5',
        'bp_conversion_pct_5'
    ]
    
    # Process each match with progress bar
    for idx, match in tqdm(match_df.iterrows(), total=len(match_df), desc="Preparing match features", unit="match"):
        match_date = match['tourney_date']
        winner_id = match['winner_id']
        loser_id = match['loser_id']
        surface = match['surface']
        
        # Get player win/loss stats just before this match (exclude the current match)
        winner_prev = player_df[(player_df['player_id'] == winner_id) & 
                              (player_df['tourney_date'] < match_date)]
        
        loser_prev = player_df[(player_df['player_id'] == loser_id) & 
                             (player_df['tourney_date'] < match_date)]
        
        # Get player serve/return stats just before this match
        winner_sr_prev = serve_return_df[(serve_return_df['player_id'] == winner_id) & 
                                      (serve_return_df['tourney_date'] < match_date)]
        
        loser_sr_prev = serve_return_df[(serve_return_df['player_id'] == loser_id) & 
                                     (serve_return_df['tourney_date'] < match_date)]
        
        # Initialize match features
        match_features = {
            'match_id': idx,
            'tourney_date': match_date,
            'surface': surface,
            'winner_id': winner_id,
            'loser_id': loser_id,
        }
        
        # Add Elo rating difference
        match_features['elo_diff'] = match['winner_elo'] - match['loser_elo']
        
        # If we have previous win/loss stats for both players
        if not winner_prev.empty and not loser_prev.empty:
            # Get most recent stats
            winner_stats = winner_prev.iloc[-1]
            loser_stats = loser_prev.iloc[-1]
            
            # Add win rate and streak features
            match_features.update({
                # Win rate differences
                'win_rate_5_diff': winner_stats.get('win_rate_5', 0) - loser_stats.get('win_rate_5', 0),
                
                # Win/loss streak differences
                'win_streak_diff': winner_stats.get('win_streak', 0) - loser_stats.get('win_streak', 0),
                'loss_streak_diff': winner_stats.get('loss_streak', 0) - loser_stats.get('loss_streak', 0),
                
                # Surface-specific win rates
                f'win_rate_{surface}_5_diff': 
                    winner_stats.get(f'win_rate_{surface}_5', np.nan) - 
                    loser_stats.get(f'win_rate_{surface}_5', np.nan),
                
                # Overall surface win rates
                f'win_rate_{surface}_overall_diff': 
                    winner_stats.get(f'win_rate_{surface}_overall', np.nan) - 
                    loser_stats.get(f'win_rate_{surface}_overall', np.nan),
                
                # Raw player win/loss stats
                'winner_win_rate_5': winner_stats.get('win_rate_5', 0),
                'loser_win_rate_5': loser_stats.get('win_rate_5', 0),
                'winner_win_streak': winner_stats.get('win_streak', 0),
                'loser_win_streak': loser_stats.get('win_streak', 0),
                'winner_loss_streak': winner_stats.get('loss_streak', 0),
                'loser_loss_streak': loser_stats.get('loss_streak', 0),
            })
            
            # Add surface-specific win rates for all surfaces
            for surf in STANDARD_SURFACES:
                match_features.update({
                    f'winner_win_rate_{surf}_5': winner_stats.get(f'win_rate_{surf}_5', np.nan),
                    f'loser_win_rate_{surf}_5': loser_stats.get(f'win_rate_{surf}_5', np.nan),
                    f'winner_win_rate_{surf}_overall': winner_stats.get(f'win_rate_{surf}_overall', np.nan),
                    f'loser_win_rate_{surf}_overall': loser_stats.get(f'win_rate_{surf}_overall', np.nan),
                })
        
        # If we have previous serve/return stats for both players
        if not winner_sr_prev.empty and not loser_sr_prev.empty:
            # Get most recent stats
            winner_sr_stats = winner_sr_prev.iloc[-1]
            loser_sr_stats = loser_sr_prev.iloc[-1]
            
            # Add serve and return metric differences
            for metric in serve_return_metrics:
                if metric in winner_sr_stats and metric in loser_sr_stats:
                    match_features[f'{metric}_diff'] = (
                        winner_sr_stats.get(metric, np.nan) - 
                        loser_sr_stats.get(metric, np.nan)
                    )
                    
                    # Surface-specific version
                    surf_metric = f'{metric}_{surface}'
                    if surf_metric in winner_sr_stats and surf_metric in loser_sr_stats:
                        match_features[f'{surf_metric}_diff'] = (
                            winner_sr_stats.get(surf_metric, np.nan) - 
                            loser_sr_stats.get(surf_metric, np.nan)
                        )
            
            # Add raw player serve/return stats
            for metric in serve_return_metrics:
                if metric in winner_sr_stats and metric in loser_sr_stats:
                    match_features[f'winner_{metric}'] = winner_sr_stats.get(metric, np.nan)
                    match_features[f'loser_{metric}'] = loser_sr_stats.get(metric, np.nan)
        
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
    
    # Define serve and return metrics to include in symmetric features
    serve_return_metrics = [
        'serve_efficiency_5',
        'first_serve_pct_5',
        'first_serve_win_pct_5',
        'second_serve_win_pct_5',
        'ace_pct_5',
        'bp_saved_pct_5',
        'return_efficiency_5',
        'bp_conversion_pct_5'
    ]
    
    # Create player-symmetric features required for prediction
    # Create two versions of each match: p1 = winner, p2 = loser and p1 = loser, p2 = winner
    matches = []
    
    # First pass: p1 = winner, p2 = loser (actual match result)
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating winner perspective features", unit="match"):
        surface = row['surface']
        match_dict = {
            'match_id': row['match_id'],
            'tourney_date': row['tourney_date'],
            'surface': surface,
            'player1_id': row['winner_id'],
            'player2_id': row['loser_id'],
            'result': 1  # player1 won
        }
        
        # Add traditional features
        # Elo difference
        match_dict['player_elo_diff'] = row.get('elo_diff', 0)
        
        # Win rates
        match_dict['win_rate_5_diff'] = row.get('win_rate_5_diff', 0)
        
        # Streaks
        match_dict['win_streak_diff'] = row.get('win_streak_diff', 0)
        match_dict['loss_streak_diff'] = row.get('loss_streak_diff', 0)
        
        # Surface win rates (preserve NaN values for proper XGBoost handling)
        for surf in STANDARD_SURFACES:
            if f'win_rate_{surf}_5_diff' in row:
                match_dict[f'win_rate_{surf}_5_diff'] = row[f'win_rate_{surf}_5_diff']
            
            if f'win_rate_{surf}_overall_diff' in row:
                match_dict[f'win_rate_{surf}_overall_diff'] = row[f'win_rate_{surf}_overall_diff']
        
        # Add serve and return features
        for metric in serve_return_metrics:
            # General metric
            if f'{metric}_diff' in row:
                match_dict[f'{metric}_diff'] = row[f'{metric}_diff']
            
            # Surface-specific metric
            if f'{metric}_{surface}_diff' in row:
                match_dict[f'{metric}_{surface}_diff'] = row[f'{metric}_{surface}_diff']
        
        # Add raw player features to ensure consistent prediction regardless of player order
        if 'winner_win_rate_5' in row:
            match_dict['player1_win_rate_5'] = row['winner_win_rate_5']
            match_dict['player2_win_rate_5'] = row['loser_win_rate_5']
            match_dict['player1_win_streak'] = row['winner_win_streak']
            match_dict['player2_win_streak'] = row['loser_win_streak']
            match_dict['player1_loss_streak'] = row['winner_loss_streak']
            match_dict['player2_loss_streak'] = row['loser_loss_streak']
            
            # Surface-specific rates
            for surf in STANDARD_SURFACES:
                if f'winner_win_rate_{surf}_5' in row:
                    match_dict[f'player1_win_rate_{surf}_5'] = row[f'winner_win_rate_{surf}_5']
                    match_dict[f'player2_win_rate_{surf}_5'] = row[f'loser_win_rate_{surf}_5']
                    match_dict[f'player1_win_rate_{surf}_overall'] = row[f'winner_win_rate_{surf}_overall']
                    match_dict[f'player2_win_rate_{surf}_overall'] = row[f'loser_win_rate_{surf}_overall']
        
        # Add serve and return raw stats
        for metric in serve_return_metrics:
            if f'winner_{metric}' in row and f'loser_{metric}' in row:
                match_dict[f'player1_{metric}'] = row[f'winner_{metric}']
                match_dict[f'player2_{metric}'] = row[f'loser_{metric}']
        
        matches.append(match_dict)
    
    # Second pass: p1 = loser, p2 = winner (flipped match result)
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating loser perspective features", unit="match"):
        surface = row['surface']
        match_dict = {
            'match_id': row['match_id'],
            'tourney_date': row['tourney_date'],
            'surface': surface,
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
        for surf in STANDARD_SURFACES:
            if f'win_rate_{surf}_5_diff' in row:
                match_dict[f'win_rate_{surf}_5_diff'] = -row[f'win_rate_{surf}_5_diff']
            
            if f'win_rate_{surf}_overall_diff' in row:
                match_dict[f'win_rate_{surf}_overall_diff'] = -row[f'win_rate_{surf}_overall_diff']
        
        # Add serve and return features (with reversed signs)
        for metric in serve_return_metrics:
            # General metric
            if f'{metric}_diff' in row:
                match_dict[f'{metric}_diff'] = -row[f'{metric}_diff']
            
            # Surface-specific metric
            if f'{metric}_{surface}_diff' in row:
                match_dict[f'{metric}_{surface}_diff'] = -row[f'{metric}_{surface}_diff']
        
        # Add raw player features (swapped)
        if 'winner_win_rate_5' in row:
            match_dict['player1_win_rate_5'] = row['loser_win_rate_5']
            match_dict['player2_win_rate_5'] = row['winner_win_rate_5']
            match_dict['player1_win_streak'] = row['loser_win_streak']
            match_dict['player2_win_streak'] = row['winner_win_streak']
            match_dict['player1_loss_streak'] = row['loser_loss_streak']
            match_dict['player2_loss_streak'] = row['winner_loss_streak']
            
            # Surface-specific rates
            for surf in STANDARD_SURFACES:
                if f'winner_win_rate_{surf}_5' in row:
                    match_dict[f'player1_win_rate_{surf}_5'] = row[f'loser_win_rate_{surf}_5']
                    match_dict[f'player2_win_rate_{surf}_5'] = row[f'winner_win_rate_{surf}_5']
                    match_dict[f'player1_win_rate_{surf}_overall'] = row[f'loser_win_rate_{surf}_overall']
                    match_dict[f'player2_win_rate_{surf}_overall'] = row[f'winner_win_rate_{surf}_overall']
        
        # Add serve and return raw stats (swapped)
        for metric in serve_return_metrics:
            if f'winner_{metric}' in row and f'loser_{metric}' in row:
                match_dict[f'player1_{metric}'] = row[f'loser_{metric}']
                match_dict[f'player2_{metric}'] = row[f'winner_{metric}']
        
        matches.append(match_dict)
    
    # Convert to DataFrame
    symmetric_df = pd.DataFrame(matches)
    
    # Sort by date and match_id
    symmetric_df = symmetric_df.sort_values(['tourney_date', 'match_id']).reset_index(drop=True)
    
    return symmetric_df

def main():
    """Generate features for tennis match prediction."""
    start_time = time.time()
    progress_tracker = ProgressTracker(total_steps=5)
    
    # Step 1: Load data
    logger.info("Step 1/5 (0%): Loading data...")
    df = load_data(INPUT_FILE)
    logger.info(f"Loaded {len(df)} matches")
    progress_tracker.update()
    
    # Step 2: Calculate player win rates and streaks
    logger.info(f"Step 2/5 ({progress_tracker.percent_complete}%): Calculating win rates and streaks...")
    player_df = calculate_win_rates(df)
    logger.info(f"Calculated features for {len(player_df)} player-match combinations")
    progress_tracker.update()
    
    # Step 3: Calculate serve and return stats
    logger.info(f"Step 3/5 ({progress_tracker.percent_complete}%): Calculating serve and return statistics...")
    serve_return_df = calculate_serve_return_stats(df)
    serve_return_df = calculate_serve_return_rolling_stats(serve_return_df)
    logger.info(f"Calculated serve/return stats for {len(serve_return_df)} player-match combinations")
    progress_tracker.update()
    
    # Step 4: Prepare features for matches
    logger.info(f"Step 4/5 ({progress_tracker.percent_complete}%): Preparing match features...")
    features_df = prepare_features_for_matches(df, player_df, serve_return_df)
    logger.info(f"Prepared features for {len(features_df)} matches")
    progress_tracker.update()
    
    # Step 5: Generate player-symmetric features
    logger.info(f"Step 5/5 ({progress_tracker.percent_complete}%): Generating symmetric features...")
    symmetric_df = generate_player_symmetric_features(features_df)
    logger.info(f"Generated {len(symmetric_df)} symmetric match features")
    progress_tracker.update()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Save results
    symmetric_df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Saved results to {OUTPUT_FILE}")
    
    # Print feature statistics
    logger.info(f"Total matches: {len(df)}")
    logger.info(f"Total features: {len(symmetric_df.columns) - 6}")  # Exclude match_id, tourney_date, player1_id, player2_id, surface, result
    
    # Print example features for a match
    if not symmetric_df.empty:
        example = symmetric_df.iloc[0].to_dict()
        feature_example = {k: v for k, v in example.items() 
                         if k not in ['match_id', 'tourney_date', 'player1_id', 'player2_id', 'surface', 'result']}
        logger.info(f"Example features: {feature_example}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Feature generation completed in {elapsed_time:.2f} seconds")


# Progress tracker class
class ProgressTracker:
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
    
    def update(self):
        self.current_step += 1
        elapsed = time.time() - self.start_time
        if self.current_step < self.total_steps:
            est_remaining = (elapsed / self.current_step) * (self.total_steps - self.current_step)
            logger.info(f"Progress: {self.percent_complete}% complete. Est. remaining time: {est_remaining:.1f}s")
        else:
            logger.info(f"Progress: 100% complete. Total time: {elapsed:.1f}s")
    
    @property
    def percent_complete(self):
        return int((self.current_step / self.total_steps) * 100)


if __name__ == "__main__":
    main() 