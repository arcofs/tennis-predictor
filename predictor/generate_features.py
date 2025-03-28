import os
import sys
import time

# Check if running in Google Colab
def is_colab() -> bool:
    """Check if the code is running in Google Colab."""
    return 'google.colab' in sys.modules

# Install required packages if needed
if is_colab():
    try:
        import pandas as pd
        import numpy as np
        from tqdm.auto import tqdm
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pandas", "numpy", "tqdm"])

    # Try to install GPU acceleration packages
    try:
        import cudf
        import cupy
    except ImportError:
        print("Installing GPU acceleration packages...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "cudf-cu11", "cupy-cuda11x"])
        except Exception as e:
            print(f"Warning: Failed to install GPU packages: {e}")
            print("Continuing without GPU acceleration")

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Set, Any
from tqdm.auto import tqdm
from pydantic import BaseModel, Field

# Print pandas version for debugging
print(f"Using pandas version: {pd.__version__}")

# Try to import GPU acceleration libraries, fall back to CPU if not available
USE_GPU = False
try:
    import cudf
    import cupy as cp
    # Test that GPU is actually working
    try:
        # Create a small test dataframe to verify GPU functionality
        test_df = cudf.DataFrame({'test': [1, 2, 3]})
        test_arr = cp.array([1, 2, 3])
        del test_df, test_arr  # Clean up test objects
        
        USE_GPU = True
        print("GPU acceleration enabled - using cuDF and cuPy")
    except Exception as e:
        print(f"GPU libraries imported but failed initialization test: {e}")
        print("Falling back to CPU-only mode")
        # Unload GPU libraries to prevent partial usage
        import sys
        if 'cudf' in sys.modules:
            del sys.modules['cudf']
        if 'cupy' in sys.modules:
            del sys.modules['cupy']
except ImportError:
    print("GPU libraries not available - falling back to CPU-only mode")

# Define a dummy cp namespace to avoid conditional code
if not USE_GPU:
    class DummyCP:
        def __getattr__(self, name: str) -> Any:
            return getattr(np, name)
    cp = DummyCP()

# Mount Google Drive if in Colab and not already mounted
if is_colab():
    from google.colab import drive
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')
        print("Google Drive mounted")
    else:
        print("Google Drive already mounted")

    # Google Drive paths for Colab
    BASE_DIR = Path('/content/drive/MyDrive/Colab Notebooks/tennis-predictor')
else:
    # Local paths if not running in Colab
    BASE_DIR = Path(__file__).parent.parent

DATA_DIR = BASE_DIR / "data"
CLEANED_DATA_DIR = DATA_DIR / "cleaned"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "predictor" / "output"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Input and output files
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INPUT_FILE = DATA_DIR / "cleaned" / "cleaned_dataset_with_elo.csv"
OUTPUT_FILE = DATA_DIR / "enhanced_features_v2.csv"  # Direct in data directory

# Define data types for memory optimization
DTYPE_DICT = {
    'tourney_date': 'str',
    'winner_id': 'int32',
    'loser_id': 'int32',
    'winner_elo': 'float32',
    'loser_elo': 'float32',
    'surface': 'category',
    'tourney_level': 'category',
    'winner_ht': 'float32',
    'loser_ht': 'float32',
    'w_ace': 'float32',
    'l_ace': 'float32',
    'w_svpt': 'float32',
    'l_svpt': 'float32',
    'w_1stWon': 'float32',
    'l_1stWon': 'float32',
    'w_1stIn': 'float32',
    'l_1stIn': 'float32',
    'w_bpSaved': 'float32',
    'l_bpSaved': 'float32',
    'w_bpFaced': 'float32',
    'l_bpFaced': 'float32'
}

# Pydantic models for type checking
class MatchData(BaseModel):
    tourney_date: str
    winner_id: int
    loser_id: int
    winner_elo: float
    loser_elo: float
    surface: str
    tourney_level: str
    winner_ht: Optional[float] = None
    loser_ht: Optional[float] = None
    w_ace: Optional[float] = 0.0
    l_ace: Optional[float] = 0.0
    w_svpt: Optional[float] = 0.0
    l_svpt: Optional[float] = 0.0
    w_1stWon: Optional[float] = 0.0
    l_1stWon: Optional[float] = 0.0
    w_1stIn: Optional[float] = 0.0
    l_1stIn: Optional[float] = 0.0
    w_bpSaved: Optional[float] = 0.0
    l_bpSaved: Optional[float] = 0.0
    w_bpFaced: Optional[float] = 0.0
    l_bpFaced: Optional[float] = 0.0
    
    class Config:
        validate_assignment = True
        extra = "ignore"

class PlayerStats(BaseModel):
    elo: float = Field(1500.0)
    surface_elo: Dict[str, float] = Field(default_factory=dict)
    recent_matches: List[int] = Field(default_factory=list)
    surface_matches: Dict[str, List[int]] = Field(default_factory=dict)
    h2h_stats: Dict[int, Dict[str, int]] = Field(default_factory=dict)
    
    class Config:
        validate_assignment = True
        extra = "ignore"

def load_data() -> pd.DataFrame:
    """
    Load the tennis match dataset with optimized dtypes.
    
    Returns:
        pd.DataFrame: The loaded dataset with proper data types
    """
    # Check if we're using a test file (filename starts with TEST_)
    input_file_path = Path(INPUT_FILE)
    is_test_file = input_file_path.name.startswith("TEST_")
    
    if is_test_file:
        print(f"Detected test file: {input_file_path.name}")
    
    print(f"Loading data from {INPUT_FILE}...")
    
    # Define columns to read - only the ones we need for feature generation
    needed_columns = [
        'tourney_date', 'winner_id', 'loser_id', 'surface', 'tourney_level',
        'winner_ht', 'loser_ht', 'w_ace', 'l_ace', 'w_svpt', 'l_svpt',
        'w_1stWon', 'l_1stWon', 'w_1stIn', 'l_1stIn', 'w_bpSaved', 'l_bpSaved',
        'w_bpFaced', 'l_bpFaced', 'winner_elo', 'loser_elo'
    ]
    
    try:
        # Read the CSV without parsing dates first
        try:
            df = pd.read_csv(
                INPUT_FILE,
                usecols=needed_columns,
                dtype={col: DTYPE_DICT.get(col, 'object') for col in needed_columns if col != 'tourney_date'}
            )
        except ValueError as e:
            # If we can't find all the needed columns, read without specifying columns
            print(f"Warning: Couldn't read all specified columns: {e}")
            print("Attempting to read all columns from the file...")
            df = pd.read_csv(
                INPUT_FILE,
                dtype={col: DTYPE_DICT.get(col, 'object') for col in DTYPE_DICT if col != 'tourney_date'}
            )
            
            # Add any missing columns with default values
            for col in needed_columns:
                if col not in df.columns:
                    print(f"Adding missing column: {col}")
                    if col in ['winner_id', 'loser_id']:
                        df[col] = np.arange(len(df)) + 1  # Generate unique IDs
                    elif col in ['winner_elo', 'loser_elo']:
                        df[col] = 1500.0  # Default Elo rating
                    elif col in ['winner_ht', 'loser_ht']:
                        df[col] = 180.0  # Default height
                    elif col == 'surface':
                        df[col] = 'Hard'  # Default surface
                    elif col == 'tourney_level':
                        df[col] = 'ATP'  # Default level
                    elif col.startswith('w_') or col.startswith('l_'):
                        df[col] = 0.0  # Default stats
                    elif col == 'tourney_date':
                        df[col] = '2000-01-01'  # Default date
        
        # Convert date column manually with explicit format
        print("Converting date column...")
        if 'tourney_date' in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df['tourney_date']):
                print("Date column is already in datetime format")
            else:
                df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y-%m-%d', errors='coerce')
        else:
            print("Adding missing tourney_date column")
            df['tourney_date'] = pd.to_datetime('2000-01-01')
        
    except Exception as e:
        print(f"Error reading CSV: {e}")
        raise
    
    # Check for invalid dates and handle them
    invalid_dates = df['tourney_date'].isna()
    if invalid_dates.any():
        print(f"Warning: Found {invalid_dates.sum()} invalid dates. Removing these rows.")
        df = df[~invalid_dates]
    
    # Sort by tournament date
    df = df.sort_values('tourney_date').reset_index(drop=True)
    
    # Try to convert numeric columns to appropriate types
    for col in df.columns:
        if col != 'tourney_date' and col in DTYPE_DICT:
            try:
                df[col] = df[col].astype(DTYPE_DICT[col])
            except (ValueError, TypeError) as e:
                print(f"Warning: Couldn't convert column {col} to {DTYPE_DICT[col]}: {e}")
    
    print(f"Loaded {len(df)} matches spanning from {df['tourney_date'].min().date()} to {df['tourney_date'].max().date()}")
    
    # For test data, add additional debug information
    if is_test_file:
        print("\nTEST DATA INFO:")
        print(f"Columns in the dataset: {df.columns.tolist()}")
        print(f"Sample of player IDs - Winners: {df['winner_id'].head(3).tolist()}, Losers: {df['loser_id'].head(3).tolist()}")
        print(f"Data types: {df.dtypes}")
    
    return df

def calculate_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Elo-based features including momentum and surface-specific ratings.
    
    Args:
        df: DataFrame with match data
        
    Returns:
        DataFrame with additional Elo features
    """
    print("Calculating Elo features...")
    
    # Sort by date to ensure chronological order
    df = df.sort_values('tourney_date')
    
    # Initialize Elo ratings
    player_elos = {}
    surface_elos = {
        'Hard': {},
        'Clay': {},
        'Grass': {}
    }
    
    # Default Elo values
    DEFAULT_ELO = 1500.0
    K_FACTOR = 32.0
    
    def update_elo(winner_elo: float, loser_elo: float, surface: str) -> Tuple[float, float]:
        """Update Elo ratings for both players."""
        expected_winner = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
        expected_loser = 1 - expected_winner
        
        winner_new = winner_elo + K_FACTOR * (1 - expected_winner)
        loser_new = loser_elo + K_FACTOR * (0 - expected_loser)
        
        return winner_new, loser_new
    
    # Calculate Elo changes and surface-specific ratings
    elo_changes = []
    surface_elo_changes = {
        'Hard': [],
        'Clay': [],
        'Grass': []
    }
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Calculating Elo features"):
        winner_id = row['winner_id']
        loser_id = row['loser_id']
        surface = row['surface']
        
        # Initialize Elo ratings if not exists
        if winner_id not in player_elos:
            player_elos[winner_id] = DEFAULT_ELO
        if loser_id not in player_elos:
            player_elos[loser_id] = DEFAULT_ELO
            
        # Initialize surface-specific Elo ratings
        for s in surface_elos:
            if winner_id not in surface_elos[s]:
                surface_elos[s][winner_id] = DEFAULT_ELO
            if loser_id not in surface_elos[s]:
                surface_elos[s][loser_id] = DEFAULT_ELO
        
        # Store current Elo ratings
        winner_elo = player_elos[winner_id]
        loser_elo = player_elos[loser_id]
        winner_surface_elo = surface_elos[surface][winner_id]
        loser_surface_elo = surface_elos[surface][loser_id]
        
        # Update overall Elo ratings
        winner_new, loser_new = update_elo(winner_elo, loser_elo, surface)
        player_elos[winner_id] = winner_new
        player_elos[loser_id] = loser_new
        
        # Update surface-specific Elo ratings
        winner_surface_new, loser_surface_new = update_elo(winner_surface_elo, loser_surface_elo, surface)
        surface_elos[surface][winner_id] = winner_surface_new
        surface_elos[surface][loser_id] = loser_surface_new
        
        # Store Elo changes
        elo_changes.append({
            'match_id': idx,
            'winner_id': winner_id,
            'loser_id': loser_id,
            'winner_elo_change': winner_new - winner_elo,
            'loser_elo_change': loser_new - loser_elo
        })
        
        # Store surface-specific Elo changes
        surface_elo_changes[surface].append({
            'match_id': idx,
            'winner_id': winner_id,
            'loser_id': loser_id,
            'winner_elo_change': winner_surface_new - winner_surface_elo,
            'loser_elo_change': loser_surface_new - loser_surface_elo
        })
    
    # Convert Elo changes to DataFrame
    elo_changes_df = pd.DataFrame(elo_changes)
    
    # Calculate rolling Elo changes and volatility
    for player_type in ['winner', 'loser']:
        # Calculate rolling Elo changes
        df[f'{player_type}_elo_change_5'] = elo_changes_df.groupby(f'{player_type}_id')['winner_elo_change'].transform(
            lambda x: x.rolling(window=5, min_periods=1).sum()
        )
        df[f'{player_type}_elo_change_10'] = elo_changes_df.groupby(f'{player_type}_id')['winner_elo_change'].transform(
            lambda x: x.rolling(window=10, min_periods=1).sum()
        )
        df[f'{player_type}_elo_change_20'] = elo_changes_df.groupby(f'{player_type}_id')['winner_elo_change'].transform(
            lambda x: x.rolling(window=20, min_periods=1).sum()
        )
        
        # Calculate Elo volatility (standard deviation of changes)
        df[f'{player_type}_elo_volatility'] = elo_changes_df.groupby(f'{player_type}_id')['winner_elo_change'].transform(
            lambda x: x.rolling(window=20, min_periods=1).std()
        )
    
    # Calculate surface-specific Elo ratings and differences
    for surface in surface_elos:
        surface_elo_changes_df = pd.DataFrame(surface_elo_changes[surface])
        
        for player_type in ['winner', 'loser']:
            # Calculate surface-specific Elo rating
            df[f'{player_type}_elo_{surface.lower()}'] = df[f'{player_type}_id'].map(
                surface_elos[surface]
            )
            
            # Calculate difference between surface Elo and overall Elo
            df[f'{player_type}_surface_elo_diff'] = (
                df[f'{player_type}_elo_{surface.lower()}'] - 
                df[f'{player_type}_elo']
            )
    
    return df

def calculate_consecutive_true(series: List[int]) -> int:
    """
    Calculate the maximum number of consecutive True values in a series.
    
    Args:
        series: List of binary values (1 for win, 0 for loss)
        
    Returns:
        int: Maximum number of consecutive True values
    """
    max_consecutive = 0
    current_consecutive = 0
    
    for val in series:
        if val == 1:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
            
    return max_consecutive

def preprocess_for_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataframe to have player-centric rows instead of match-centric.
    
    Args:
        df: Match-level DataFrame
        
    Returns:
        DataFrame: Long format with player-level stats
    """
    print("Preprocessing data for feature generation...")
    
    # Create a long format dataframe with one row per player per match
    winners = df.copy()
    losers = df.copy()
    
    # Add result column (1 for win, 0 for loss)
    winners['result'] = 1
    losers['result'] = 0
    
    # Rename columns to be player-centric
    winner_cols = {
        'winner_id': 'player_id',
        'loser_id': 'opponent_id',
        'winner_elo': 'player_elo',
        'loser_elo': 'opponent_elo',
        'winner_ht': 'player_ht',
        'loser_ht': 'opponent_ht',
        'w_ace': 'player_ace',
        'l_ace': 'opponent_ace',
        'w_svpt': 'player_svpt',
        'l_svpt': 'opponent_svpt',
        'w_1stWon': 'player_1stWon',
        'l_1stWon': 'opponent_1stWon',
        'w_1stIn': 'player_1stIn',
        'l_1stIn': 'opponent_1stIn',
        'w_bpSaved': 'player_bpSaved',
        'l_bpSaved': 'opponent_bpSaved',
        'w_bpFaced': 'player_bpFaced',
        'l_bpFaced': 'opponent_bpFaced'
    }
    
    loser_cols = {
        'loser_id': 'player_id',
        'winner_id': 'opponent_id',
        'loser_elo': 'player_elo',
        'winner_elo': 'opponent_elo',
        'loser_ht': 'player_ht',
        'winner_ht': 'opponent_ht',
        'l_ace': 'player_ace',
        'w_ace': 'opponent_ace',
        'l_svpt': 'player_svpt',
        'w_svpt': 'opponent_svpt',
        'l_1stWon': 'player_1stWon',
        'w_1stWon': 'opponent_1stWon',
        'l_1stIn': 'player_1stIn',
        'w_1stIn': 'opponent_1stIn',
        'l_bpSaved': 'player_bpSaved',
        'w_bpSaved': 'opponent_bpSaved',
        'l_bpFaced': 'player_bpFaced',
        'w_bpFaced': 'opponent_bpFaced'
    }
    
    winners = winners.rename(columns=winner_cols)
    losers = losers.rename(columns=loser_cols)
    
    # Keep only the columns we need
    cols_to_keep = [
        'tourney_date', 'player_id', 'opponent_id', 'surface', 'tourney_level',
        'player_elo', 'opponent_elo', 'player_ht', 'opponent_ht',
        'player_ace', 'opponent_ace', 'player_svpt', 'opponent_svpt',
        'player_1stWon', 'opponent_1stWon', 'player_1stIn', 'opponent_1stIn',
        'player_bpSaved', 'opponent_bpSaved', 'player_bpFaced', 'opponent_bpFaced',
        'result'
    ]
    
    winners = winners[cols_to_keep]
    losers = losers[cols_to_keep]
    
    # Combine and sort by date
    player_df = pd.concat([winners, losers], ignore_index=True)
    player_df = player_df.sort_values(['tourney_date', 'player_id']).reset_index(drop=True)
    
    return player_df

def optimize_for_gpu(df: pd.DataFrame) -> Union[pd.DataFrame, 'cudf.DataFrame']:
    """
    Convert pandas DataFrame to cuDF DataFrame if GPU is available.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        cudf.DataFrame if GPU is available, otherwise original pandas DataFrame
    """
    if USE_GPU:
        try:
            # Convert to cuDF for GPU acceleration
            return cudf.DataFrame.from_pandas(df)
        except Exception as e:
            print(f"Warning: Failed to convert to cuDF: {e}")
            return df
    return df

def convert_from_gpu(df: Union[pd.DataFrame, 'cudf.DataFrame']) -> pd.DataFrame:
    """
    Convert cuDF DataFrame back to pandas DataFrame if necessary.
    
    Args:
        df: cuDF DataFrame or pandas DataFrame
        
    Returns:
        pandas DataFrame
    """
    if USE_GPU and isinstance(df, cudf.DataFrame):
        return df.to_pandas()
    return df

def generate_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate rolling window features for each player with multiple time windows.
    Focus only on surface, Elo, and win rate features.
    
    Args:
        df: Player-level DataFrame
        
    Returns:
        DataFrame with added rolling features
    """
    print("Generating rolling features with multiple time windows...")
    
    # Make sure player_df does not have player_id as an index
    if 'player_id' in df.index.names:
        df = df.reset_index()
    
    # Group by player
    grouped = df.groupby('player_id')
    
    # Define multiple time windows for win rate calculation
    time_windows = [5, 10, 20, 30, 50]
    
    # Calculate win rates for each time window
    for window in time_windows:
        print(f"  Calculating {window}-match win rate...")
        df[f'win_rate_{window}'] = grouped['result'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        ).astype('float32')
    
    # Surface-specific win rates with multiple time windows
    print("  Calculating surface-specific win rates...")
    for surface in df['surface'].unique():
        # Skip NaN surfaces
        if pd.isna(surface):
            continue
            
        print(f"  Processing surface: {surface}")
        surface_mask = df['surface'] == surface
        
        for window in time_windows:
            df[f'win_rate_{surface}_{window}'] = np.nan
            
            for player_id in tqdm(df['player_id'].unique(), desc=f"Surface: {surface}, Window: {window}"):
                player_surface_mask = (df['player_id'] == player_id) & surface_mask
                if player_surface_mask.sum() > 0:
                    player_indices = np.where(player_surface_mask)[0]
                    
                    for i, idx in enumerate(player_indices):
                        if i == 0:
                            df.loc[idx, f'win_rate_{surface}_{window}'] = df.loc[idx, 'result']
                        else:
                            # Get up to window previous matches on this surface
                            start_idx = max(0, i - window)
                            prev_results = df.loc[player_indices[start_idx:i], 'result'].values
                            df.loc[idx, f'win_rate_{surface}_{window}'] = prev_results.mean()
    
    # Recent form indicators (win/loss streaks)
    print("  Calculating win/loss streaks...")
    
    player_ids = df['player_id'].unique()
    consecutive_wins = np.zeros(len(df), dtype=np.int8)
    consecutive_losses = np.zeros(len(df), dtype=np.int8)
    
    with tqdm(total=len(player_ids), desc="Processing streaks") as pbar:
        for player_id in player_ids:
            player_mask = df['player_id'] == player_id
            player_indices = np.where(player_mask)[0]
            
            current_win_streak = 0
            current_loss_streak = 0
            
            for i, idx in enumerate(player_indices):
                result = df.loc[idx, 'result']
                
                if result == 1:  # Win
                    current_win_streak += 1
                    current_loss_streak = 0
                else:  # Loss
                    current_loss_streak += 1
                    current_win_streak = 0
                
                consecutive_wins[idx] = current_win_streak
                consecutive_losses[idx] = current_loss_streak
            
            pbar.update(1)
    
    df['current_win_streak'] = consecutive_wins
    df['current_loss_streak'] = consecutive_losses
    
    # Ensure player_id is a column and not an index
    if 'player_id' in df.index.names:
        df = df.reset_index()
    
    # Fill NaN values with default values
    float_cols = df.select_dtypes(include=['float32', 'float64']).columns
    df[float_cols] = df[float_cols].fillna(0.5)  # Use 0.5 as default for win rates
    
    return df

def calculate_head_to_head_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate head-to-head features between players without data leakage.
    Only use historical H2H data (up to but NOT including the current match).
    
    Args:
        df: Match-level DataFrame (original format, not player-centric)
        
    Returns:
        DataFrame with added H2H features
    """
    print("Calculating non-leaky head-to-head features...")
    
    # Add date column for sorting if not present
    if 'date' not in df.columns and 'tourney_date' in df.columns:
        df['date'] = pd.to_datetime(df['tourney_date'])
    
    # Ensure df is sorted by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Initialize H2H feature columns
    h2h_columns = [
        'h2h_wins_winner', 'h2h_wins_loser', 'h2h_matches', 
        'h2h_win_pct_winner', 'h2h_win_pct_loser',
        'h2h_hard_win_pct_winner', 'h2h_clay_win_pct_winner', 'h2h_grass_win_pct_winner'
    ]
    
    for col in h2h_columns:
        df[col] = 0.0
    
    # Store all previous matches
    player_matches = {}  # Dict to store all matches between each pair of players
    
    # Process matches chronologically to avoid leakage
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing H2H data"):
        p1 = row['winner_id']
        p2 = row['loser_id']
        
        # Create a unique key for this player pair (always put smaller ID first for consistency)
        pair_key = (min(p1, p2), max(p1, p2))
        
        # Get previous matches between these players
        prev_matches = player_matches.get(pair_key, [])
        
        # Calculate H2H stats based on previous matches only
        if prev_matches:
            p1_wins = sum(1 for m in prev_matches if m['winner'] == p1)
            p2_wins = len(prev_matches) - p1_wins
            
            # Count wins by surface
            p1_hard_matches = [m for m in prev_matches if m['surface'] == 'Hard']
            p1_clay_matches = [m for m in prev_matches if m['surface'] == 'Clay']
            p1_grass_matches = [m for m in prev_matches if m['surface'] == 'Grass']
            
            p1_hard_wins = sum(1 for m in p1_hard_matches if m['winner'] == p1)
            p1_clay_wins = sum(1 for m in p1_clay_matches if m['winner'] == p1)
            p1_grass_wins = sum(1 for m in p1_grass_matches if m['winner'] == p1)
            
            # Set H2H features
            df.loc[idx, 'h2h_wins_winner'] = p1_wins
            df.loc[idx, 'h2h_wins_loser'] = p2_wins
            df.loc[idx, 'h2h_matches'] = len(prev_matches)
            
            # Win percentages (avoid division by zero)
            df.loc[idx, 'h2h_win_pct_winner'] = p1_wins / len(prev_matches) if len(prev_matches) > 0 else 0.5
            df.loc[idx, 'h2h_win_pct_loser'] = p2_wins / len(prev_matches) if len(prev_matches) > 0 else 0.5
            
            # Surface-specific win percentages
            df.loc[idx, 'h2h_hard_win_pct_winner'] = (
                p1_hard_wins / len(p1_hard_matches) if len(p1_hard_matches) > 0 else 0.5
            )
            df.loc[idx, 'h2h_clay_win_pct_winner'] = (
                p1_clay_wins / len(p1_clay_matches) if len(p1_clay_matches) > 0 else 0.5
            )
            df.loc[idx, 'h2h_grass_win_pct_winner'] = (
                p1_grass_wins / len(p1_grass_matches) if len(p1_grass_matches) > 0 else 0.5
            )
        
        # Add current match to history (for future matches)
        player_matches.setdefault(pair_key, []).append({
            'winner': p1,
            'loser': p2,
            'surface': row['surface']
        })
    
    return df

def calculate_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate physical matchup features.
    
    Args:
        df: Match-level DataFrame
        
    Returns:
        DataFrame with added physical features
    """
    print("Calculating physical features...")
    
    # Make sure height columns are numeric
    df['winner_ht'] = pd.to_numeric(df['winner_ht'], errors='coerce')
    df['loser_ht'] = pd.to_numeric(df['loser_ht'], errors='coerce')
    
    # Fill NaN values with median height
    winner_median = df['winner_ht'].median()
    loser_median = df['loser_ht'].median()
    df['winner_ht'] = df['winner_ht'].fillna(winner_median)
    df['loser_ht'] = df['loser_ht'].fillna(loser_median)
    
    # Height advantage
    df['height_diff'] = df['winner_ht'] - df['loser_ht']
    
    return df

def convert_to_match_prediction_format(player_df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert player-level features back to match-level format for prediction.
    Keep only surface, Elo, and win rate features.
    
    Args:
        player_df: DataFrame with player-level features
        original_df: Original match-level DataFrame
        
    Returns:
        DataFrame with features in match prediction format
    """
    print("Converting to match prediction format...")
    
    # Make sure player_df does not have player_id as an index
    if 'player_id' in player_df.index.names:
        player_df = player_df.reset_index()
    
    # Create a new dataframe with match information
    try:
        # Verify the key columns exist in original_df
        required_cols = ['tourney_date', 'winner_id', 'loser_id', 'surface']
        missing_cols = [col for col in required_cols if col not in original_df.columns]
        
        if missing_cols:
            print(f"WARNING: Missing columns in original_df: {missing_cols}")
            # If columns are missing, add them with default values
            for col in missing_cols:
                if col == 'tourney_date':
                    original_df[col] = pd.to_datetime('2000-01-01')
                elif col in ['winner_id', 'loser_id']:
                    original_df[col] = 0
                else:
                    original_df[col] = 'unknown'
        
        match_df = original_df[required_cols].copy()
    except Exception as e:
        print(f"Error creating match_df: {e}")
        print("Creating minimal match_df with required columns")
        
        # Create a minimal DataFrame with the necessary columns
        match_df = pd.DataFrame({
            'tourney_date': original_df['tourney_date'].copy() if 'tourney_date' in original_df.columns else pd.to_datetime('2000-01-01'),
            'winner_id': original_df['winner_id'].copy() if 'winner_id' in original_df.columns else 0,
            'loser_id': original_df['loser_id'].copy() if 'loser_id' in original_df.columns else 0,
            'surface': original_df['surface'].copy() if 'surface' in original_df.columns else 'unknown'
        })
    
    # Convert categorical columns to string to avoid category issues
    for col in match_df.select_dtypes(include=['category']).columns:
        match_df[col] = match_df[col].astype(str)
    
    # Define core features and time windows
    time_windows = [5, 10, 20, 30, 50]
    
    # Generate feature columns - only include surface, Elo, and win rate features
    feature_cols = []
    
    # Win rate with different time windows
    for window in time_windows:
        feature_cols.append(f'win_rate_{window}')
    
    # Surface-specific win rates with different time windows
    for surface in player_df['surface'].unique():
        if pd.isna(surface):
            continue
        for window in time_windows:
            feature_cols.append(f'win_rate_{surface}_{window}')
    
    # Add Elo and streak features
    feature_cols.extend(['current_win_streak', 'current_loss_streak', 'player_elo'])
    
    # H2H features
    h2h_features = [
        'h2h_win_pct', 'h2h_hard_win_pct', 'h2h_clay_win_pct', 'h2h_grass_win_pct'
    ]
    
    # Only add H2H features if they exist in player_df
    for feature in h2h_features:
        if feature in player_df.columns:
            feature_cols.append(feature)
    
    # Filter to features that exist in the dataframe
    feature_cols = [col for col in feature_cols if col in player_df.columns]
    
    print(f"Using {len(feature_cols)} features: {feature_cols[:5]}..." if len(feature_cols) > 5 else f"Using {len(feature_cols)} features: {feature_cols}")
    
    total_matches = len(match_df)
    
    # Initialize all feature columns with 0.0 to avoid insertion issues
    for col in feature_cols:
        match_df[f'winner_{col}'] = 0.0
        match_df[f'loser_{col}'] = 0.0
    
    # For each match, get features for both players
    for idx, row in tqdm(match_df.iterrows(), total=total_matches, desc="Creating prediction features"):
        # Get features for winner
        winner_mask = (player_df['player_id'] == row['winner_id']) & (player_df['tourney_date'] <= row['tourney_date'])
        if winner_mask.any():
            winner_features = player_df.loc[winner_mask].iloc[-1]
            
            for col in feature_cols:
                if col in winner_features:
                    match_df.loc[idx, f'winner_{col}'] = winner_features[col]
        
        # Get features for loser
        loser_mask = (player_df['player_id'] == row['loser_id']) & (player_df['tourney_date'] <= row['tourney_date'])
        if loser_mask.any():
            loser_features = player_df.loc[loser_mask].iloc[-1]
            
            for col in feature_cols:
                if col in loser_features:
                    match_df.loc[idx, f'loser_{col}'] = loser_features[col]
    
    # Calculate feature differences (winner - loser)
    for col in feature_cols:
        match_df[f'{col}_diff'] = match_df[f'winner_{col}'] - match_df[f'loser_{col}']
    
    # Add Elo difference directly
    if 'winner_elo' in original_df.columns and 'loser_elo' in original_df.columns:
        match_df['elo_diff'] = original_df['winner_elo'] - original_df['loser_elo']
    
    # Fill NaN values with 0.5 (for win rates)
    match_df = match_df.fillna(0.5)
    
    # Ensure required columns are present and properly typed
    print(f"Final match_df shape: {match_df.shape}")
    print(f"Final match_df columns: {match_df.columns.tolist()[:5]}..." if len(match_df.columns) > 5 else f"Final match_df columns: {match_df.columns.tolist()}")
    
    return match_df

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by using appropriate dtypes.
    
    Args:
        df: DataFrame to optimize
        
    Returns:
        Optimized DataFrame
    """
    print("Optimizing data types for memory efficiency...")
    
    # Convert float64 to float32
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    
    # Convert int64 to smaller int types where possible
    for col in df.select_dtypes(include=['int64']).columns:
        col_min, col_max = df[col].min(), df[col].max()
        
        if col_min >= -128 and col_max <= 127:
            df[col] = df[col].astype('int8')
        elif col_min >= -32768 and col_max <= 32767:
            df[col] = df[col].astype('int16')
        elif col_min >= -2147483648 and col_max <= 2147483647:
            df[col] = df[col].astype('int32')
    
    # Convert object columns to categories where appropriate
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() < len(df) * 0.5:  # Only if cardinality is low enough
            df[col] = df[col].astype('category')
    
    return df

def debug_dataframe_info(df: pd.DataFrame, df_name: str) -> None:
    """
    Print debugging information about a DataFrame.
    
    Args:
        df: DataFrame to debug
        df_name: Name of the DataFrame for display
    """
    print(f"\nDEBUG INFO for {df_name}:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Index type: {type(df.index)}")
    print(f"First few rows preview:")
    print(df.head(2))
    print("-" * 40)

def main() -> None:
    """
    Main function to generate all features and save to output file.
    """
    start_time = time.time()
    
    # Define the steps for processing
    steps = [
        "Loading data",
        "Calculating Elo features",
        "Calculating physical features",
        "Calculating H2H features",
        "Preprocessing for player-level features",
        "Generating rolling features",
        "Converting to match prediction format",
        "Combining all features",
        "Optimizing memory usage",
        "Saving to output file"
    ]
    total_steps = len(steps)
    
    print(f"Starting focused feature generation process with {total_steps} steps")
    print(f"Keeping only surface, Elo ratings, and win rate features")
    print(f"GPU acceleration: {'Enabled' if USE_GPU else 'Disabled'}")
    print("=" * 80)
    
    # 1. Load data
    print(f"[Step 1/{total_steps}] {steps[0]} - 0% complete")
    df = load_data()
    
    # Check if dataset is empty, if so exit gracefully
    if len(df) == 0:
        print("WARNING: Dataset is empty. No features to generate.")
        print("=" * 80)
        print("Feature generation completed with 0 matches.")
        return
        
    print(f"Completed step 1/{total_steps} - 10% complete")
    print("-" * 80)
    
    # 2. Calculate Elo features
    print(f"[Step 2/{total_steps}] {steps[1]} - 10% complete")
    df = calculate_elo_features(df)
    print(f"Completed step 2/{total_steps} - 20% complete")
    print("-" * 80)
    
    # 3. Calculate physical features
    print(f"[Step 3/{total_steps}] {steps[2]} - 20% complete")
    df = calculate_physical_features(df)
    print(f"Completed step 3/{total_steps} - 30% complete")
    print("-" * 80)
    
    # 4. Calculate H2H features
    print(f"[Step 4/{total_steps}] {steps[3]} - 30% complete")
    df = calculate_head_to_head_features(df)
    print(f"Completed step 4/{total_steps} - 40% complete")
    print("-" * 80)
    
    # 5. Preprocess for player-level features
    print(f"[Step 5/{total_steps}] {steps[4]} - 40% complete")
    player_df = preprocess_for_features(df)
    print(f"Completed step 5/{total_steps} - 50% complete")
    print("-" * 80)
    
    # 6. Generate rolling features - only surface and win rate features
    print(f"[Step 6/{total_steps}] {steps[5]} - 50% complete")
    player_df = generate_rolling_features(player_df)
    print(f"Completed step 6/{total_steps} - 70% complete")
    print("-" * 80)
    
    # 7. Convert back to match prediction format - only surface, Elo, and win rate features
    print(f"[Step 7/{total_steps}] {steps[6]} - 70% complete")
    match_df = convert_to_match_prediction_format(player_df, df)
    print(f"Completed step 7/{total_steps} - 80% complete")
    print("-" * 80)
    
    # 8. Combine all features
    print(f"[Step 8/{total_steps}] {steps[7]} - 80% complete")
    print("Combining all features...")
    
    # Make sure df.index and match_df.index don't clash
    if df.index.name is not None:
        df = df.reset_index()
    if match_df.index.name is not None:
        match_df = match_df.reset_index()
    
    # Ensure indices are aligned
    if len(df) == len(match_df):
        match_df.index = df.index
    
    # Final dataframe is match_df
    final_df = match_df
    
    # 9. Optimize memory usage
    print(f"[Step 9/{total_steps}] {steps[8]} - 90% complete")
    final_df = optimize_dtypes(final_df)
    print(f"Completed step 9/{total_steps} - 95% complete")
    print("-" * 80)
    
    # 10. Save to output file
    print(f"[Step 10/{total_steps}] {steps[9]} - 95% complete")
    print(f"Saving focused features to {OUTPUT_FILE}...")
    # Make sure directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Completed step 10/{total_steps} - 100% complete")
    print("=" * 80)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    
    print(f"Feature generation completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"Generated {len(final_df)} matches with {len(final_df.columns)} features")
    print(f"DataFrame memory usage: {final_df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    print(f"Output file saved to: {OUTPUT_FILE}")
    print("=" * 80)

if __name__ == "__main__":
    main() 