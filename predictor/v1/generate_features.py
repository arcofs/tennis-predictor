import os
import sys
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*swapaxes.*")
warnings.filterwarnings("ignore", message=".*Downcasting object dtype.*")
warnings.filterwarnings("ignore", message=".*Series.fillna.*")

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
from typing import Dict, List, Tuple, Optional, Union, Set, Any, Callable
from tqdm.auto import tqdm
from pydantic import BaseModel, Field
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime, timedelta

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

# Create all necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CLEANED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Input and output files
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

# Add after line 70 (after the DTYPE_DICT definition)
# Define standard surface names as constants
SURFACE_HARD = 'Hard'
SURFACE_CLAY = 'Clay'
SURFACE_GRASS = 'Grass'
SURFACE_CARPET = 'Carpet' 
STANDARD_SURFACES = [SURFACE_HARD, SURFACE_CLAY, SURFACE_GRASS, SURFACE_CARPET]

# Add a function to verify and correct surface names
def verify_surface_name(surface: str) -> str:
    """
    Verify and correct surface name to ensure consistency.
    
    Args:
        surface: Surface name to verify
        
    Returns:
        Standardized surface name
    """
    if pd.isna(surface):
        return None
    
    surface_str = str(surface).lower()
    surface_mapping = {
        'hard': SURFACE_HARD,
        'h': SURFACE_HARD,
        'clay': SURFACE_CLAY,
        'cl': SURFACE_CLAY,
        'grass': SURFACE_GRASS,
        'gr': SURFACE_GRASS,
        'carpet': SURFACE_CARPET,
        'cpt': SURFACE_CARPET,
        'indoor': SURFACE_HARD,  # Map indoor to hard
        'outdoor': SURFACE_HARD,  # Map outdoor to hard by default
    }
    
    return surface_mapping.get(surface_str, surface)

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

# Add parallel processing configuration
NUM_CORES = multiprocessing.cpu_count()
NUM_THREADS = max(1, NUM_CORES - 1)  # Leave one core free for system
CHUNK_SIZE = 10000  # Size of chunks for parallel processing

# Add automatic scaling for very high-core systems
MAX_USABLE_CORES = min(NUM_CORES, 120)  # Cap at 64 cores to prevent overhead on very large systems
print(f"System has {NUM_CORES} cores, using {MAX_USABLE_CORES} for processing")

def parallel_process_chunk(chunk_df: pd.DataFrame, func: callable) -> pd.DataFrame:
    """
    Process a chunk of data in parallel.
    
    Args:
        chunk_df: DataFrame chunk to process
        func: Function to apply to the chunk
        
    Returns:
        Processed DataFrame chunk
    """
    return func(chunk_df)

def parallel_process_dataframe(df: pd.DataFrame, func: callable, desc: str) -> pd.DataFrame:
    """
    Process DataFrame in parallel chunks.
    
    Args:
        df: DataFrame to process
        func: Function to apply to each chunk
        desc: Description for progress bar
        
    Returns:
        Processed DataFrame
    """
    # Split DataFrame into chunks
    chunks = np.array_split(df, MAX_USABLE_CORES)
    
    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=MAX_USABLE_CORES) as executor:
        process_chunk = partial(parallel_process_chunk, func=func)
        results = list(tqdm(
            executor.map(process_chunk, chunks),
            total=len(chunks),
            desc=desc
        ))
    
    # Combine results
    return pd.concat(results, ignore_index=True)

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
                        df[col] = SURFACE_HARD  # Default surface
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
    
    # Standardize surface names
    print("Standardizing surface names...")
    if 'surface' in df.columns:
        # Apply surface name standardization
        df['surface'] = df['surface'].apply(verify_surface_name)
        
        # Count occurrences of each surface
        surface_counts = df['surface'].value_counts()
        print(f"Surface distribution after standardization: {surface_counts.to_dict()}")
    
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

def process_chunk(chunk_df: pd.DataFrame, player_elos: Dict[int, float], surface_elos: Dict[str, Dict[int, float]], DEFAULT_ELO: float, K_FACTOR: float) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    """
    Process a chunk of matches to calculate Elo ratings.
    
    Args:
        chunk_df: DataFrame chunk to process
        player_elos: Dictionary of player Elo ratings
        surface_elos: Dictionary of surface-specific Elo ratings
        DEFAULT_ELO: Default Elo rating for new players
        K_FACTOR: K-factor for Elo calculations
        
    Returns:
        Tuple of (list of Elo changes, dict of surface-specific changes)
    """
    chunk_changes = []
    chunk_surface_changes = {surface: [] for surface in surface_elos}
    
    for idx, row in chunk_df.iterrows():
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
        expected_winner = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
        expected_loser = 1 - expected_winner
        
        winner_new = winner_elo + K_FACTOR * (1 - expected_winner)
        loser_new = loser_elo + K_FACTOR * (0 - expected_loser)
        
        player_elos[winner_id] = winner_new
        player_elos[loser_id] = loser_new
        
        # Update surface-specific Elo ratings
        expected_winner_surface = 1 / (1 + 10 ** ((loser_surface_elo - winner_surface_elo) / 400))
        expected_loser_surface = 1 - expected_winner_surface
        
        winner_surface_new = winner_surface_elo + K_FACTOR * (1 - expected_winner_surface)
        loser_surface_new = loser_surface_elo + K_FACTOR * (0 - expected_loser_surface)
        
        surface_elos[surface][winner_id] = winner_surface_new
        surface_elos[surface][loser_id] = loser_surface_new
        
        # Store changes
        chunk_changes.append({
            'match_id': idx,
            'winner_id': winner_id,
            'loser_id': loser_id,
            'winner_elo_change': winner_new - winner_elo,
            'loser_elo_change': loser_new - loser_elo
        })
        
        chunk_surface_changes[surface].append({
            'match_id': idx,
            'winner_id': winner_id,
            'loser_id': loser_id,
            'winner_elo_change': winner_surface_new - winner_surface_elo,
            'loser_elo_change': loser_surface_new - loser_surface_elo
        })
    
    return chunk_changes, chunk_surface_changes

def calculate_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Elo-based features including momentum and surface-specific ratings.
    Now with parallel processing and GPU acceleration.
    """
    print("Calculating Elo features...")
    
    # Sort by date to ensure chronological order
    df = df.sort_values('tourney_date')
    
    # Initialize Elo ratings
    player_elos = {}
    
    # Get unique surfaces from the data
    unique_surfaces = df['surface'].unique()
    unique_surfaces = [s for s in unique_surfaces if pd.notna(s)]  # Remove any NaN values
    
    # Initialize surface_elos with all unique surfaces
    surface_elos = {surface: {} for surface in unique_surfaces}
    
    # Default Elo values
    DEFAULT_ELO = 1500.0
    K_FACTOR = 32.0
    
    print(f"Processing matches with surfaces: {unique_surfaces}")
    
    # Process matches in parallel chunks
    chunks = np.array_split(df, MAX_USABLE_CORES)
    all_changes = []
    all_surface_changes = {surface: [] for surface in surface_elos}
    
    with ProcessPoolExecutor(max_workers=MAX_USABLE_CORES) as executor:
        process_chunk_partial = partial(
            process_chunk,
            player_elos=player_elos,
            surface_elos=surface_elos,
            DEFAULT_ELO=DEFAULT_ELO,
            K_FACTOR=K_FACTOR
        )
        results = list(tqdm(
            executor.map(process_chunk_partial, chunks),
            total=len(chunks),
            desc="Processing Elo chunks"
        ))
        
        # Combine results
        for chunk_changes, chunk_surface_changes in results:
            all_changes.extend(chunk_changes)
            for surface in surface_elos:
                all_surface_changes[surface].extend(chunk_surface_changes[surface])
    
    # Convert to DataFrame and calculate features using GPU if available
    elo_changes_df = pd.DataFrame(all_changes)
    
    if USE_GPU:
        # Use cuDF for GPU-accelerated calculations
        elo_changes_gdf = cudf.DataFrame.from_pandas(elo_changes_df)
        
        # Calculate rolling features on GPU
        for player_type in ['winner', 'loser']:
            # Rolling Elo changes
            for window in [5, 10, 20]:
                elo_changes_gdf[f'{player_type}_elo_change_{window}'] = elo_changes_gdf.groupby(f'{player_type}_id')['winner_elo_change'].transform(
                    lambda x: x.rolling(window, min_periods=1).sum()
                ).to_pandas()
            
            # Elo volatility
            elo_changes_gdf[f'{player_type}_elo_volatility'] = elo_changes_gdf.groupby(f'{player_type}_id')['winner_elo_change'].transform(
                lambda x: x.rolling(20, min_periods=1).std()
            ).to_pandas()
    else:
        # CPU calculations with parallel processing
        for player_type in ['winner', 'loser']:
            # Rolling Elo changes
            for window in [5, 10, 20]:
                elo_changes_df[f'{player_type}_elo_change_{window}'] = elo_changes_df.groupby(f'{player_type}_id')['winner_elo_change'].transform(
                    lambda x: x.rolling(window, min_periods=1).sum()
                )
            
            # Elo volatility
            elo_changes_df[f'{player_type}_elo_volatility'] = elo_changes_df.groupby(f'{player_type}_id')['winner_elo_change'].transform(
                lambda x: x.rolling(20, min_periods=1).std()
            )
    
    # Calculate surface-specific Elo ratings and differences
    for surface in surface_elos:
        surface_elo_changes_df = pd.DataFrame(all_surface_changes[surface])
        
        if USE_GPU:
            # Use GPU for surface-specific calculations
            surface_elo_gdf = cudf.DataFrame.from_pandas(surface_elo_changes_df)
            
            for player_type in ['winner', 'loser']:
                df[f'{player_type}_elo_{surface.lower()}'] = df[f'{player_type}_id'].map(
                    surface_elos[surface]
                )
                df[f'{player_type}_surface_elo_diff'] = (
                    df[f'{player_type}_elo_{surface.lower()}'] - 
                    df[f'{player_type}_elo']
                )
        else:
            # CPU calculations with parallel processing
            for player_type in ['winner', 'loser']:
                df[f'{player_type}_elo_{surface.lower()}'] = df[f'{player_type}_id'].map(
                    surface_elos[surface]
                )
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

def process_player_chunk(chunk_df: pd.DataFrame, time_windows: List[int]) -> pd.DataFrame:
    """
    Process a chunk of player data to calculate rolling features.
    
    Args:
        chunk_df: DataFrame chunk to process
        time_windows: List of window sizes for rolling calculations
        
    Returns:
        DataFrame with calculated rolling features
    """
    # Calculate win rates
    for window in time_windows:
        chunk_df[f'win_rate_{window}'] = chunk_df.groupby('player_id')['result'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        
        # Add feature flag to indicate insufficient data
        chunk_df[f'win_rate_{window}_reliable'] = chunk_df.groupby('player_id')['result'].transform(
            lambda x: x.rolling(window, min_periods=1).count() >= window/2
        )
    
    # Surface-specific win rates
    for surface in chunk_df['surface'].unique():
        if pd.isna(surface):
            continue
        surface_mask = chunk_df['surface'] == surface
        
        for window in time_windows:
            # Calculate surface-specific win rates only for matches on that surface
            surface_win_rates = chunk_df[surface_mask].groupby('player_id')['result'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            
            # Create the column in the full dataframe
            chunk_df[f'win_rate_{surface}_{window}'] = None
            
            # Assign the calculated values only to rows with that surface
            chunk_df.loc[surface_mask, f'win_rate_{surface}_{window}'] = surface_win_rates
            
            # Forward-fill the values to later matches (regardless of surface)
            chunk_df[f'win_rate_{surface}_{window}'] = chunk_df.groupby('player_id')[f'win_rate_{surface}_{window}'].transform(
                lambda x: x.ffill().infer_objects(copy=False)
            )
            
            # Add reliability indicator
            chunk_df[f'win_rate_{surface}_{window}_reliable'] = chunk_df.groupby('player_id')[f'win_rate_{surface}_{window}'].transform(
                lambda x: x.notna()
            )
    
    return chunk_df

def generate_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate rolling window features with parallel processing and GPU acceleration.
    """
    print("Generating rolling features with multiple time windows...")
    
    if USE_GPU:
        # Convert to cuDF for GPU acceleration
        df_gpu = cudf.DataFrame.from_pandas(df)
        
        # Calculate win rates on GPU
        time_windows = [5, 10, 20, 30, 50]
        for window in time_windows:
            df_gpu[f'win_rate_{window}'] = df_gpu.groupby('player_id')['result'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            
            # Add feature flag to indicate insufficient data
            df_gpu[f'win_rate_{window}_reliable'] = df_gpu.groupby('player_id')['result'].transform(
                lambda x: x.rolling(window, min_periods=1).count() >= window/2
            )
        
        # Surface-specific win rates on GPU
        for surface in df['surface'].unique():
            if pd.isna(surface):
                continue
            surface_mask = df_gpu['surface'] == surface
            
            for window in time_windows:
                # Calculate surface-specific win rates only for matches on that surface
                surface_win_rates = df_gpu[surface_mask].groupby('player_id')['result'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                
                # Create the column in the full dataframe
                df_gpu[f'win_rate_{surface}_{window}'] = None
                
                # Assign the calculated values only to rows with that surface
                df_gpu.loc[surface_mask, f'win_rate_{surface}_{window}'] = surface_win_rates
                
                # Forward-fill the values to later matches (regardless of surface)
                df_gpu[f'win_rate_{surface}_{window}'] = df_gpu.groupby('player_id')[f'win_rate_{surface}_{window}'].transform(
                    lambda x: x.ffill().infer_objects(copy=False)
                )
                
                # Add reliability indicator
                df_gpu[f'win_rate_{surface}_{window}_reliable'] = df_gpu.groupby('player_id')[f'win_rate_{surface}_{window}'].transform(
                    lambda x: x.notna()
                )
        
        # Calculate streaks for GPU version
        # Convert to CPU for streak calculations
        print("Calculating win/loss streaks with parallel processing...")
        df_cpu = df_gpu.to_pandas()
        
        # Calculate streak features with parallel processing
        # Create shift column to check consecutive results
        df_cpu['prev_result'] = df_cpu.groupby('player_id')['result'].shift(1).fillna(0)
        
        # Initialize streak columns
        df_cpu['current_win_streak'] = 0
        df_cpu['current_loss_streak'] = 0
        
        # Get unique player IDs for parallelization
        player_ids = df_cpu['player_id'].unique()
        print(f"Processing streaks for {len(player_ids)} players using {MAX_USABLE_CORES} cores...")
        
        # Split players into chunks for parallel processing
        player_chunks = np.array_split(player_ids, MAX_USABLE_CORES)
        
        # Process player streaks in parallel
        with ProcessPoolExecutor(max_workers=MAX_USABLE_CORES) as executor:
            process_func = partial(process_player_streaks_gpu, data=df_cpu)
            all_results = list(tqdm(
                executor.map(process_func, player_chunks),
                total=len(player_chunks),
                desc="Calculating player streaks (GPU version)"
            ))
        
        # Combine results and update the dataframe
        print("Combining streak results...")
        for chunk_result in all_results:
            for player_id, player_data in chunk_result.items():
                indices = player_data['indices']
                if indices:  # Skip empty results
                    df_cpu.loc[indices, 'current_win_streak'] = player_data['win_streaks']
                    df_cpu.loc[indices, 'current_loss_streak'] = player_data['loss_streaks']
        
        # Transfer streak features back to GPU
        df_gpu['current_win_streak'] = df_cpu['current_win_streak']
        df_gpu['current_loss_streak'] = df_cpu['current_loss_streak']
        
        # Clean up intermediate columns
        df_cpu = df_cpu.drop(['prev_result'], axis=1, errors='ignore')
        
        # Convert back to pandas
        df = df_gpu.to_pandas()
    else:
        # CPU parallel processing
        time_windows = [5, 10, 20, 30, 50]
        process_chunk_partial = partial(process_player_chunk, time_windows=time_windows)
        df = parallel_process_dataframe(df, process_chunk_partial, "Generating rolling features")
        
        # CPU streak calculations with parallel processing
        print("Calculating win/loss streaks...")
        
        # Create shift column to check consecutive results
        df['prev_result'] = df.groupby('player_id')['result'].shift(1).fillna(0)
        
        # Initialize streak columns
        df['current_win_streak'] = 0
        df['current_loss_streak'] = 0
        
        # Get unique player IDs for parallelization
        player_ids = df['player_id'].unique()
        print(f"Processing streaks for {len(player_ids)} players using {MAX_USABLE_CORES} cores...")
        
        # Split players into chunks for parallel processing
        player_chunks = np.array_split(player_ids, MAX_USABLE_CORES)
        
        # Process player streaks in parallel
        with ProcessPoolExecutor(max_workers=MAX_USABLE_CORES) as executor:
            process_func = partial(process_player_streaks, data=df)
            all_results = list(tqdm(
                executor.map(process_func, player_chunks),
                total=len(player_chunks),
                desc="Calculating player streaks"
            ))
        
        # Combine results and update the dataframe
        print("Combining streak results...")
        for chunk_result in all_results:
            for player_id, player_data in chunk_result.items():
                indices = player_data['indices']
                if indices:  # Skip empty results
                    df.loc[indices, 'current_win_streak'] = player_data['win_streaks']
                    df.loc[indices, 'current_loss_streak'] = player_data['loss_streaks']
        
        # Clean up intermediate columns
        df = df.drop(['prev_result'], axis=1, errors='ignore')
    
    return df

def process_h2h_chunk(chunk_df: pd.DataFrame, all_matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process a chunk of matches to calculate head-to-head features.
    
    Args:
        chunk_df: DataFrame chunk to process
        all_matches_df: Complete DataFrame with all matches (sorted by date)
        
    Returns:
        DataFrame chunk with added H2H features
    """
    result_chunk = chunk_df.copy()
    
    # Initialize H2H feature columns - use p1/p2 naming instead of winner/loser
    h2h_columns = [
        'h2h_wins_p1', 'h2h_wins_p2', 'h2h_matches', 
        'h2h_win_pct_p1', 'h2h_win_pct_p2',
        'h2h_hard_win_pct_p1', 'h2h_clay_win_pct_p1', 'h2h_grass_win_pct_p1'
    ]
    
    for col in h2h_columns:
        result_chunk[col] = 0.0
    
    # Process matches in the chunk
    for idx, row in result_chunk.iterrows():
        p1 = row['winner_id']  # During prediction, this would be player1_id
        p2 = row['loser_id']   # During prediction, this would be player2_id
        match_date = row['date'] if 'date' in row else row['tourney_date']
        
        # Get all previous matches between these players
        # (from the complete dataset, not just this chunk)
        pair_mask = ((all_matches_df['winner_id'] == p1) & (all_matches_df['loser_id'] == p2)) | \
                    ((all_matches_df['winner_id'] == p2) & (all_matches_df['loser_id'] == p1))
        date_mask = all_matches_df['tourney_date'] < match_date
        prev_matches = all_matches_df[pair_mask & date_mask]
        
        if len(prev_matches) > 0:
            # Count wins for each player
            p1_wins = len(prev_matches[(prev_matches['winner_id'] == p1)])
            p2_wins = len(prev_matches[(prev_matches['winner_id'] == p2)])
            total_matches = len(prev_matches)
            
            # Count wins by surface
            p1_hard_matches = prev_matches[prev_matches['surface'] == 'Hard']
            p1_clay_matches = prev_matches[prev_matches['surface'] == 'Clay']
            p1_grass_matches = prev_matches[prev_matches['surface'] == 'Grass']
            
            p1_hard_wins = len(p1_hard_matches[p1_hard_matches['winner_id'] == p1])
            p1_clay_wins = len(p1_clay_matches[p1_clay_matches['winner_id'] == p1])
            p1_grass_wins = len(p1_grass_matches[p1_grass_matches['winner_id'] == p1])
            
            # Set H2H features
            result_chunk.loc[idx, 'h2h_wins_p1'] = p1_wins
            result_chunk.loc[idx, 'h2h_wins_p2'] = p2_wins
            result_chunk.loc[idx, 'h2h_matches'] = total_matches
            
            # Win percentages (avoid division by zero)
            result_chunk.loc[idx, 'h2h_win_pct_p1'] = p1_wins / total_matches if total_matches > 0 else np.nan
            result_chunk.loc[idx, 'h2h_win_pct_p2'] = p2_wins / total_matches if total_matches > 0 else np.nan
            
            # Surface-specific win percentages - use NaN to indicate no data
            result_chunk.loc[idx, 'h2h_hard_win_pct_p1'] = (
                p1_hard_wins / len(p1_hard_matches) if len(p1_hard_matches) > 0 else np.nan
            )
            result_chunk.loc[idx, 'h2h_clay_win_pct_p1'] = (
                p1_clay_wins / len(p1_clay_matches) if len(p1_clay_matches) > 0 else np.nan
            )
            result_chunk.loc[idx, 'h2h_grass_win_pct_p1'] = (
                p1_grass_wins / len(p1_grass_matches) if len(p1_grass_matches) > 0 else np.nan
            )
    
    # Map these back to winner/loser during historical feature generation
    # but use p1/p2 terminology for prediction
    result_chunk['h2h_wins_winner'] = result_chunk['h2h_wins_p1']
    result_chunk['h2h_wins_loser'] = result_chunk['h2h_wins_p2']
    result_chunk['h2h_win_pct_winner'] = result_chunk['h2h_win_pct_p1']
    result_chunk['h2h_win_pct_loser'] = result_chunk['h2h_win_pct_p2']
    result_chunk['h2h_hard_win_pct_winner'] = result_chunk['h2h_hard_win_pct_p1']
    result_chunk['h2h_clay_win_pct_winner'] = result_chunk['h2h_clay_win_pct_p1']
    result_chunk['h2h_grass_win_pct_winner'] = result_chunk['h2h_grass_win_pct_p1']
    
    return result_chunk

def calculate_head_to_head_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate head-to-head features between players without data leakage.
    Only use historical H2H data (up to but NOT including the current match).
    Parallelized to use all available cores.
    
    Args:
        df: Match-level DataFrame (original format, not player-centric)
        
    Returns:
        DataFrame with added H2H features
    """
    print("Calculating non-leaky head-to-head features in parallel...")
    
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
    
    # Split into chunks for parallel processing
    chunks = np.array_split(df, MAX_USABLE_CORES)
    
    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=MAX_USABLE_CORES) as executor:
        process_func = partial(process_h2h_chunk, all_matches_df=df)
        results = list(tqdm(
            executor.map(process_func, chunks),
            total=len(chunks),
            desc="Processing H2H features in parallel"
        ))
    
    # Combine results
    result_df = pd.concat(results, ignore_index=True)
    
    # Ensure the result is in the same order as the input
    result_df = result_df.sort_values('date').reset_index(drop=True)
    
    # Verify we have all rows
    if len(result_df) != len(df):
        print(f"WARNING: Row count mismatch after H2H processing. Original: {len(df)}, Result: {len(result_df)}")
    
    return result_df

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

def process_matches_parallel(df_chunk: pd.DataFrame, player_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Process a chunk of matches to extract player features in parallel.
    
    Args:
        df_chunk: DataFrame chunk with matches to process
        player_df: DataFrame with player features
        feature_cols: List of feature columns to extract
        
    Returns:
        DataFrame with features for matches in chunk
    """
    result_chunk = df_chunk.copy()
    
    # Create dictionaries to store all the columns we'll add, avoiding fragmentation
    winner_features_dict = {f'winner_{col}': np.full(len(result_chunk), None) for col in feature_cols}
    loser_features_dict = {f'loser_{col}': np.full(len(result_chunk), None) for col in feature_cols}
    winner_imputed_dict = {f'winner_{col}_imputed': np.full(len(result_chunk), False) for col in feature_cols}
    loser_imputed_dict = {f'loser_{col}_imputed': np.full(len(result_chunk), False) for col in feature_cols}
    
    # Process each match in the chunk
    for idx, row in result_chunk.iterrows():
        # Get features for winner
        winner_mask = (player_df['player_id'] == row['winner_id']) & (player_df['tourney_date'] <= row['tourney_date'])
        if winner_mask.any():
            winner_features = player_df.loc[winner_mask].iloc[-1]
            
            for col in feature_cols:
                if col in winner_features:
                    winner_features_dict[f'winner_{col}'][idx - result_chunk.index[0]] = winner_features[col]
        
        # Get features for loser
        loser_mask = (player_df['player_id'] == row['loser_id']) & (player_df['tourney_date'] <= row['tourney_date'])
        if loser_mask.any():
            loser_features = player_df.loc[loser_mask].iloc[-1]
            
            for col in feature_cols:
                if col in loser_features:
                    loser_features_dict[f'loser_{col}'][idx - result_chunk.index[0]] = loser_features[col]
    
    # Combine all feature dictionaries
    all_features = {**winner_features_dict, **loser_features_dict, **winner_imputed_dict, **loser_imputed_dict}
    
    # Add all columns at once to avoid fragmentation
    feature_df = pd.DataFrame(all_features, index=result_chunk.index)
    
    # Concatenate with the original chunk
    result_chunk = pd.concat([result_chunk, feature_df], axis=1)
    
    return result_chunk

def get_related_surface_rates(player_df: pd.DataFrame, player_id: int, target_surface: str, match_date, window: int) -> Optional[float]:
    """
    Get a player's win rate on related surfaces.
    
    Args:
        player_df: DataFrame with player data
        player_id: Player ID
        target_surface: Target surface
        match_date: Match date
        window: Window size
        
    Returns:
        Related surface win rate or None
    """
    player_matches = player_df[(player_df['player_id'] == player_id) & 
                               (player_df['tourney_date'] <= match_date)]
    
    if player_matches.empty:
        return None
    
    # Map of related surfaces (by similarity)
    surface_map = {
        'Hard': ['Carpet', 'Grass', 'Clay'],  # Most similar to least
        'Clay': ['Carpet', 'Hard', 'Grass'],
        'Grass': ['Carpet', 'Hard', 'Clay'],
        'Carpet': ['Hard', 'Grass', 'Clay']
    }
    
    # If target surface not in map, return None
    if target_surface not in surface_map:
        return None
    
    # Check if player has win rates for related surfaces
    related_surfaces = surface_map[target_surface]
    
    for surface in related_surfaces:
        surface_col = f'win_rate_{surface}_{window}'
        if surface_col in player_matches.columns and not player_matches[surface_col].isna().all():
            # Return the most recent non-NaN value
            for idx in range(len(player_matches)-1, -1, -1):
                value = player_matches.iloc[idx][surface_col]
                if not pd.isna(value):
                    return value
    
    return None

def get_player_tier(player_df: pd.DataFrame, elo_quantiles: List[float], player_id: int, match_date) -> Optional[int]:
    """
    Find player's ranking tier based on Elo.
    
    Args:
        player_df: DataFrame with player data
        elo_quantiles: List of Elo quantiles
        player_id: Player ID
        match_date: Match date
        
    Returns:
        Player tier (0-4) or None
    """
    player_matches = player_df[(player_df['player_id'] == player_id) & 
                               (player_df['tourney_date'] <= match_date)]
    
    if player_matches.empty or 'player_elo' not in player_matches.columns:
        return None
    
    player_elo = player_matches.iloc[-1]['player_elo']
    
    # Determine tier based on Elo
    for i, (lower, upper) in enumerate(zip([0] + elo_quantiles, elo_quantiles + [float('inf')])):
        if lower < player_elo <= upper:
            return i
    
    return None

def impute_chunk_optimized(df_chunk: pd.DataFrame, 
                        player_type: str, 
                        features_by_type: Tuple, 
                        player_df: pd.DataFrame, 
                        elo_quantiles: List[float],
                        tier_win_rates: Dict, 
                        surface_tour_win_rates: Dict, 
                        tour_win_rates: Dict) -> pd.DataFrame:
    """
    Process a chunk for imputation with optimized hybrid approach.
    Combines vectorized operations with targeted row-wise processing for complex cases.
    
    Args:
        df_chunk: DataFrame chunk
        player_type: Type of player (winner/loser)
        features_by_type: Features grouped by type
        player_df: Player dataframe
        elo_quantiles: Elo quantiles for tiers
        tier_win_rates: Win rates by tier
        surface_tour_win_rates: Surface-specific tour averages
        tour_win_rates: Overall tour averages
        
    Returns:
        Imputed DataFrame chunk
    """
    win_rate_features, elo_features, streak_features, h2h_features = features_by_type
    result_chunk = df_chunk.copy()
    
    # 1. Simple imputations (vectorized operations)
    
    # Elo features: impute with starting Elo (1500.0) - fully vectorized
    for col in elo_features:
        col_name = f'{player_type}_{col}'
        if col_name in result_chunk.columns:
            mask = result_chunk[col_name].isna()
            if mask.any():
                result_chunk.loc[mask, col_name] = 1500.0
                result_chunk.loc[mask, f'{col_name}_imputed'] = True
    
    # Streak features: impute with 0 - fully vectorized
    for col in streak_features:
        col_name = f'{player_type}_{col}'
        if col_name in result_chunk.columns:
            mask = result_chunk[col_name].isna()
            if mask.any():
                result_chunk.loc[mask, col_name] = 0
                result_chunk.loc[mask, f'{col_name}_imputed'] = True
    
    # H2H features: use more advanced imputation strategy instead of 0.5
    for col in h2h_features:
        col_name = f'{player_type}_{col}'
        if col_name in result_chunk.columns:
            mask = result_chunk[col_name].isna()
            if mask.any():
                # For h2h features, we keep track of the fact they were imputed
                result_chunk.loc[mask, f'{col_name}_imputed'] = True
                
                # Instead of defaulting to 0.5, we'll use a smarter approach:
                # 1. For win percentage features, use player's overall win rate as a proxy
                if 'win_pct' in col:
                    # Look up player's relevant win rate for each missing value
                    for idx in result_chunk[mask].index:
                        player_id = result_chunk.loc[idx, f'{player_type}_id']
                        match_date = result_chunk.loc[idx, 'tourney_date']
                        
                        # Get player's overall win rate
                        win_rate_col = 'win_rate_10'  # Use 10-match window as default
                        
                        # For surface-specific h2h features, use surface-specific win rates
                        if 'hard' in col.lower():
                            win_rate_col = 'win_rate_Hard_10'
                        elif 'clay' in col.lower():
                            win_rate_col = 'win_rate_Clay_10'
                        elif 'grass' in col.lower():
                            win_rate_col = 'win_rate_Grass_10'
                        
                        # Get player's historical data up to match date
                        player_history = player_df[(player_df['player_id'] == player_id) & 
                                                  (player_df['tourney_date'] < match_date)]
                        
                        if not player_history.empty and win_rate_col in player_history.columns:
                            # Use most recent win rate as proxy for h2h
                            latest_win_rate = player_history.iloc[-1][win_rate_col]
                            if not pd.isna(latest_win_rate):
                                result_chunk.loc[idx, col_name] = latest_win_rate
                                continue
                        
                        # If we can't get a player-specific rate, use tour average
                        if win_rate_col in tour_win_rates:
                            result_chunk.loc[idx, col_name] = tour_win_rates[win_rate_col]
                        else:
                            # As a last resort, use a calculated average (not 0.5)
                            # Calculate the mean of the non-missing values for this feature
                            mean_val = result_chunk[col_name].dropna().mean()
                            if not pd.isna(mean_val):
                                result_chunk.loc[idx, col_name] = mean_val
                            else:
                                # Only if all else fails, use 0.5
                                result_chunk.loc[idx, col_name] = 0.5
                else:
                    # For h2h non-win percentage features (like match count), use 0
                    result_chunk.loc[mask, col_name] = 0
    
    # 2. Win rate features - hybrid approach with proper handling of missing values
    
    # First, identify which features need complex imputation
    complex_imputation_needed = {}
    
    for col in win_rate_features:
        col_name = f'{player_type}_{col}'
        if col_name in result_chunk.columns:
            # Mark all missing values as imputed
            mask = result_chunk[col_name].isna()
            if not mask.any():
                continue
                
            result_chunk.loc[mask, f'{col_name}_imputed'] = True
            
            # Simple win rate features - use tour average directly (vectorized)
            if col in tour_win_rates:
                # Store indices that need complex imputation (about 20% of cases)
                # This helps us focus row-by-row processing only where needed
                if '_' in col and not col.startswith('win_rate_'):  # Surface-specific features
                    # For surface-specific win rates, we need more complex imputation for some cases
                    complex_imputation_needed[col] = mask.copy()
                else:
                    # For general win rates, apply tour average directly (fully vectorized)
                    result_chunk.loc[mask, col_name] = tour_win_rates[col]
            else:
                # Don't default to 0.5, instead use average of non-missing values
                mean_val = result_chunk[col_name].dropna().mean()
                if pd.isna(mean_val):
                    # Use tour-wide average for this category if available
                    result_chunk.loc[mask, col_name] = tour_win_rates.get(col, np.nan)
    
    # 3. Process complex imputations only where needed
    for col, mask in complex_imputation_needed.items():
        col_name = f'{player_type}_{col}'
        
        # Skip if no complex imputation needed
        if not mask.any():
            continue
        
        # Extract parts for surface-specific rates
        parts = col.split('_')
        if len(parts) >= 3:
            surface = parts[1]
            window = int(parts[2])
            
            # For each missing value requiring complex imputation
            for idx in result_chunk[mask].index:
                player_id = result_chunk.loc[idx, f'{player_type}_id']
                match_date = result_chunk.loc[idx, 'tourney_date']
                
                # Try strategies in order of complexity (most accurate to least)
                
                # STRATEGY 1: Related surfaces
                related_rate = get_related_surface_rates(player_df, player_id, surface, match_date, window)
                if related_rate is not None:
                    result_chunk.loc[idx, col_name] = related_rate
                    continue
                
                # STRATEGY 2: Elo tier-based rates
                player_tier = get_player_tier(player_df, elo_quantiles, player_id, match_date)
                if player_tier is not None and (col, player_tier) in tier_win_rates:
                    result_chunk.loc[idx, col_name] = tier_win_rates[(col, player_tier)]
                    continue
                
                # STRATEGY 3: Surface-specific tour average
                if col in surface_tour_win_rates:
                    result_chunk.loc[idx, col_name] = surface_tour_win_rates[col]
                    continue
                
                # FALLBACK: Use 0.5
                result_chunk.loc[idx, col_name] = 0.5
    
    return result_chunk

def calculate_diffs(chunk_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Calculate feature differences between winner and loser.
    
    Args:
        chunk_df: DataFrame chunk
        feature_cols: Feature columns to process
        
    Returns:
        DataFrame with calculated differences
    """
    for col in feature_cols:
        winner_col = f'winner_{col}'
        loser_col = f'loser_{col}'
        diff_col = f'{col}_diff'
        
        # Calculate diff only when both columns exist
        if winner_col in chunk_df.columns and loser_col in chunk_df.columns:
            chunk_df[diff_col] = chunk_df[winner_col] - chunk_df[loser_col]
            
            # Add imputation flag for diff column - True if either player value was imputed
            chunk_df[f'{diff_col}_imputed'] = chunk_df[f'winner_{col}_imputed'] | chunk_df[f'loser_{col}_imputed']
    
    return chunk_df

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
    
    # Improved parallel processing for match features
    print(f"Processing {len(match_df)} matches with parallel processing using {MAX_USABLE_CORES} cores")
    
    # Split the dataframe into chunks for parallel processing
    chunks = np.array_split(match_df, MAX_USABLE_CORES)
    results = []
    
    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=MAX_USABLE_CORES) as executor:
        process_func = partial(process_matches_parallel, player_df=player_df, feature_cols=feature_cols)
        results = list(tqdm(
            executor.map(process_func, chunks),
            total=len(chunks),
            desc="Processing match features in parallel"
        ))
    
    # Combine the processed chunks
    match_df = pd.concat(results, ignore_index=True)
    
    # Calculate tour averages for different surfaces and time windows for better imputation
    print("Calculating tour averages for improved imputation...")
    
    # Overall win rates by window - will be close to 0.5 by definition but for completeness
    tour_win_rates = {}
    for window in time_windows:
        win_rate_col = f'win_rate_{window}'
        if win_rate_col in player_df.columns:
            tour_win_rates[win_rate_col] = player_df[win_rate_col].mean()
    
    # Surface-specific tour averages
    surface_tour_win_rates = {}
    for surface in player_df['surface'].unique():
        if pd.isna(surface):
            continue
        
        surface_mask = player_df['surface'] == surface
        for window in time_windows:
            surface_win_rate_col = f'win_rate_{surface}_{window}'
            if surface_win_rate_col in player_df.columns:
                surface_data = player_df.loc[surface_mask, surface_win_rate_col]
                if not surface_data.empty:
                    surface_tour_win_rates[surface_win_rate_col] = surface_data.mean()
    
    # Calculate player ranking tiers for rank-based imputation
    print("Calculating player ranking tiers for rank-based imputation...")
    
    # Use Elo as a proxy for ranking
    elo_quantiles = []
    tier_win_rates = {}
    
    if 'player_elo' in player_df.columns:
        # Create 5 ranking tiers based on Elo
        elo_quantiles = player_df['player_elo'].quantile([0.2, 0.4, 0.6, 0.8]).tolist()
        
        # Tier win rates for different windows
        for i, (lower, upper) in enumerate(zip([0] + elo_quantiles, elo_quantiles + [float('inf')])):
            tier_mask = (player_df['player_elo'] > lower) & (player_df['player_elo'] <= upper)
            
            for window in time_windows:
                win_rate_col = f'win_rate_{window}'
                if win_rate_col in player_df.columns:
                    tier_data = player_df.loc[tier_mask, win_rate_col]
                    if not tier_data.empty:
                        tier_win_rates[(win_rate_col, i)] = tier_data.mean()
            
            # Surface-specific tier win rates
            for surface in player_df['surface'].unique():
                if pd.isna(surface):
                    continue
                
                for window in time_windows:
                    surface_win_rate_col = f'win_rate_{surface}_{window}'
                    if surface_win_rate_col in player_df.columns:
                        surface_tier_mask = tier_mask & (player_df['surface'] == surface)
                        surface_tier_data = player_df.loc[surface_tier_mask, surface_win_rate_col]
                        if not surface_tier_data.empty and len(surface_tier_data) > 10:  # Only if we have enough data
                            tier_win_rates[(surface_win_rate_col, i)] = surface_tier_data.mean()
    
    # Mark imputed values and apply smart imputation strategies with optimized hybrid approach
    print("Applying optimized imputation for missing values...")
    
    # Group features by type for appropriate imputation strategies
    win_rate_features = [col for col in feature_cols if 'win_rate' in col]
    elo_features = [col for col in feature_cols if 'elo' in col or 'rating' in col]
    streak_features = [col for col in feature_cols if 'streak' in col]
    h2h_features = [col for col in feature_cols if 'h2h' in col]
    
    # Pre-sort player_df by tourney_date to ensure chronological integrity
    player_df = player_df.sort_values('tourney_date')
    
    # Split into chunks for parallel imputation
    chunks = np.array_split(match_df, MAX_USABLE_CORES)
    features_by_type = (win_rate_features, elo_features, streak_features, h2h_features)
    
    # Process winner features in parallel with optimized imputation
    print("Imputing winner features with optimized approach...")
    with ProcessPoolExecutor(max_workers=MAX_USABLE_CORES) as executor:
        impute_winner = partial(
            impute_chunk_optimized, 
            player_type='winner', 
            features_by_type=features_by_type,
            player_df=player_df,
            elo_quantiles=elo_quantiles,
            tier_win_rates=tier_win_rates,
            surface_tour_win_rates=surface_tour_win_rates,
            tour_win_rates=tour_win_rates
        )
        winner_results = list(tqdm(
            executor.map(impute_winner, chunks),
            total=len(chunks),
            desc="Imputing winner features"
        ))
    
    # Combine winner results
    match_df = pd.concat(winner_results, ignore_index=True)
    
    # Process loser features in parallel with optimized imputation
    print("Imputing loser features with optimized approach...")
    chunks = np.array_split(match_df, MAX_USABLE_CORES)
    with ProcessPoolExecutor(max_workers=MAX_USABLE_CORES) as executor:
        impute_loser = partial(
            impute_chunk_optimized, 
            player_type='loser', 
            features_by_type=features_by_type,
            player_df=player_df,
            elo_quantiles=elo_quantiles,
            tier_win_rates=tier_win_rates,
            surface_tour_win_rates=surface_tour_win_rates,
            tour_win_rates=tour_win_rates
        )
        loser_results = list(tqdm(
            executor.map(impute_loser, chunks),
            total=len(chunks),
            desc="Imputing loser features"
        ))
    
    # Combine loser results
    match_df = pd.concat(loser_results, ignore_index=True)
    
    # Calculate feature differences (winner - loser)
    print("Calculating feature differences...")
    
    # Calculate differences in parallel
    chunks = np.array_split(match_df, MAX_USABLE_CORES)
    with ProcessPoolExecutor(max_workers=MAX_USABLE_CORES) as executor:
        calc_diffs_partial = partial(calculate_diffs, feature_cols=feature_cols)
        diff_results = list(tqdm(
            executor.map(calc_diffs_partial, chunks),
            total=len(chunks),
            desc="Calculating feature differences"
        ))
    
    # Combine diff results
    match_df = pd.concat(diff_results, ignore_index=True)
    
    # Add Elo difference directly - safely check for columns first
    if 'winner_elo' in original_df.columns and 'loser_elo' in original_df.columns:
        # First ensure these columns exist in match_df before using them
        if 'winner_elo' not in match_df.columns:
            match_df['winner_elo'] = original_df['winner_elo'].values
            print("Added missing winner_elo column to match_df")
            
        if 'loser_elo' not in match_df.columns:
            match_df['loser_elo'] = original_df['loser_elo'].values
            print("Added missing loser_elo column to match_df")
        
        # Calculate the difference
        match_df['elo_diff'] = match_df['winner_elo'] - match_df['loser_elo']
        
        # Add imputation flag for elo_diff (safely check for these columns first)
        if 'winner_elo' in match_df.columns and 'loser_elo' in match_df.columns:
            winner_elo_imputed = match_df['winner_elo'].isna().fillna(False) 
            loser_elo_imputed = match_df['loser_elo'].isna().fillna(False)
            match_df['elo_diff_imputed'] = winner_elo_imputed | loser_elo_imputed
        else:
            # If we can't check for NA values, assume no imputation
            match_df['elo_diff_imputed'] = False
            print("WARNING: Could not check for NA values in Elo columns")
    else:
        # If original_df doesn't have these columns, add a placeholder elo_diff
        print("WARNING: winner_elo or loser_elo not found in original dataframe. Adding placeholder elo_diff.")
        match_df['elo_diff'] = 0.0
        match_df['elo_diff_imputed'] = True
    
    # Print imputation statistics
    imputed_cols = [col for col in match_df.columns if col.endswith('_imputed')]
    for col in imputed_cols:
        base_col = col.replace('_imputed', '')
        if base_col in match_df.columns:
            pct_imputed = match_df[col].mean() * 100
            if pct_imputed > 0:
                print(f"Imputed {pct_imputed:.2f}% of values for {base_col}")
    
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

def apply_lag(df: pd.DataFrame, group_col: str, sort_col: str, lag_col: str, lag: int) -> pd.Series:
    """Apply a lag to a column within groups."""
    return df.sort_values(sort_col).groupby(group_col)[lag_col].shift(lag).transform(
        lambda x: x.ffill().infer_objects(copy=False)
    )

def main() -> None:
    """
    Main function to generate all features with parallel processing and GPU acceleration.
    """
    start_time = time.time()
    
    print(f"Starting focused feature generation process")
    print(f"System has {NUM_CORES} cores, using {MAX_USABLE_CORES} for parallel processing")
    print(f"GPU acceleration: {'Enabled' if USE_GPU else 'Disabled'}")
    print("=" * 80)
    
    # Print paths being used
    print(f"Running in {'Google Colab' if is_colab() else 'local environment'}")
    print(f"Base directory: {BASE_DIR}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Input file: {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print("=" * 80)
    
    # 1. Load data
    print(f"[Step 1/{MAX_USABLE_CORES}] Loading data - 0% complete")
    df = load_data()
    
    # Check if dataset is empty, if so exit gracefully
    if len(df) == 0:
        print("WARNING: Dataset is empty. No features to generate.")
        print("=" * 80)
        print("Feature generation completed with 0 matches.")
        return
        
    print(f"Completed step 1/{MAX_USABLE_CORES} - 10% complete")
    print("-" * 80)
    
    # 2. Calculate Elo features
    print(f"[Step 2/{MAX_USABLE_CORES}] Calculating Elo features - 10% complete")
    df = calculate_elo_features(df)
    print(f"Completed step 2/{MAX_USABLE_CORES} - 20% complete")
    print("-" * 80)
    
    # 3. Calculate physical features
    print(f"[Step 3/{MAX_USABLE_CORES}] Calculating physical features - 20% complete")
    df = calculate_physical_features(df)
    print(f"Completed step 3/{MAX_USABLE_CORES} - 30% complete")
    print("-" * 80)
    
    # 4. Calculate H2H features
    print(f"[Step 4/{MAX_USABLE_CORES}] Calculating H2H features - 30% complete")
    df = calculate_head_to_head_features(df)
    print(f"Completed step 4/{MAX_USABLE_CORES} - 40% complete")
    print("-" * 80)
    
    # 5. Preprocess for player-level features
    print(f"[Step 5/{MAX_USABLE_CORES}] Preprocessing for player-level features - 40% complete")
    player_df = preprocess_for_features(df)
    print(f"Completed step 5/{MAX_USABLE_CORES} - 50% complete")
    print("-" * 80)
    
    # 6. Generate rolling features - only surface and win rate features
    print(f"[Step 6/{MAX_USABLE_CORES}] Generating rolling features - 50% complete")
    player_df = generate_rolling_features(player_df)
    print(f"Completed step 6/{MAX_USABLE_CORES} - 70% complete")
    print("-" * 80)
    
    # 7. Convert back to match prediction format - only surface, Elo, and win rate features
    print(f"[Step 7/{MAX_USABLE_CORES}] Converting to match prediction format - 70% complete")
    match_df = convert_to_match_prediction_format(player_df, df)
    print(f"Completed step 7/{MAX_USABLE_CORES} - 80% complete")
    print("-" * 80)
    
    # 8. Combine all features
    print(f"[Step 8/{MAX_USABLE_CORES}] Combining all features - 80% complete")
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
    
    # XGBoost preparation
    print(f"[Step 9/{MAX_USABLE_CORES}] Preparing features for XGBoost - 85% complete")
    
    # 9. Calculate feature reliability weights and add XGBoost metadata
    final_df = prepare_features_for_xgboost(final_df)
    print(f"Completed step 9/{MAX_USABLE_CORES} - 90% complete")
    print("-" * 80)
    
    # 10. Optimize memory usage
    print(f"[Step 10/{MAX_USABLE_CORES}] Optimizing memory usage - 90% complete")
    final_df = optimize_dtypes(final_df)
    print(f"Completed step 10/{MAX_USABLE_CORES} - 95% complete")
    print("-" * 80)
    
    # 11. Save to output file
    print(f"[Step 11/{MAX_USABLE_CORES}] Saving to output file - 95% complete")
    print(f"Saving focused features to {OUTPUT_FILE}...")
    
    # Make sure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Save the dataframe to CSV
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Completed step 11/{MAX_USABLE_CORES} - 100% complete")
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

def prepare_features_for_xgboost(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for optimal XGBoost performance:
    1. Add feature reliability weights
    2. Handle categorical variables
    3. Add feature metadata
    4. Check for and eliminate potential data leakage
    
    Args:
        df: DataFrame with features
        
    Returns:
        DataFrame optimized for XGBoost
    """
    print("Preparing features for XGBoost...")
    
    # Check for columns with too many missing values that might cause problems for XGBoost
    missing_pct = df.isna().mean()
    high_missing_cols = missing_pct[missing_pct > 0.5].index.tolist()
    if high_missing_cols:
        print(f"Warning: Features with >50% missing values may reduce model quality: {high_missing_cols}")
        # Don't remove these columns - XGBoost can handle missing values effectively
        print("These columns will be kept as XGBoost has special handling for missing values")
    
    # 1. Calculate feature reliability weights
    # These can be used in sample_weight parameter in XGBoost to give less weight to samples with imputed values
    reliability_cols = [col for col in df.columns if col.endswith('_imputed')]
    
    # Map imputed columns to their base feature
    imputed_feature_map = {}
    for col in reliability_cols:
        base_feature = col.replace('_imputed', '')
        if base_feature in df.columns:
            imputed_feature_map[base_feature] = col
    
    # Calculate overall reliability score for each row (1 = fully reliable, lower = less reliable)
    # Focus on the most important features for tennis prediction
    key_features = [
        'win_rate_10_diff', 'win_rate_Hard_10_diff', 'win_rate_Clay_10_diff', 
        'elo_diff', 'player_elo_diff', 'h2h_win_pct_diff'
    ]
    
    # Filter to only key features that actually exist in the dataframe
    key_features = [f for f in key_features if f in df.columns]
    
    # Initialize reliability weight column
    df['reliability_weight'] = 1.0
    
    if key_features:
        print(f"Calculating reliability weights based on {len(key_features)} key features...")
        
        # Get imputation flags for key features
        key_imputed_flags = []
        for feature in key_features:
            if f"{feature}_imputed" in df.columns:
                key_imputed_flags.append(f"{feature}_imputed")
        
        if key_imputed_flags:
            # Calculate what percentage of key features are imputed (0 = none, 1 = all)
            df['imputed_key_pct'] = df[key_imputed_flags].mean(axis=1)
            
            # Reliability weight formula: 1.0 for no imputation, decreasing as more features are imputed
            # We use a sigmoid function to ensure weights remain positive but decrease with more imputation
            df['reliability_weight'] = 1.0 - (df['imputed_key_pct'] * 0.5)
            
            print(f"Reliability weights range: {df['reliability_weight'].min():.4f} to {df['reliability_weight'].max():.4f}")
    
    # 2. Handle categorical features (surface, tourney_level)
    # XGBoost handles categorical features automatically with enable_categorical=True
    # but we need to ensure they're properly encoded
    
    for cat_col in ['surface', 'tourney_level']:
        if cat_col in df.columns:
            print(f"Preparing categorical feature: {cat_col}")
            
            # Ensure categorical columns are properly encoded as category dtype
            if not pd.api.types.is_categorical_dtype(df[cat_col]):
                df[cat_col] = df[cat_col].astype('category')
    
    # 3. Handle missing values properly
    missing_summary = df.isna().sum().to_dict()
    print(f"Missing values summary: {', '.join([f'{k}: {v}' for k, v in missing_summary.items() if v > 0])}")
    
    # DO NOT fill missing values - XGBoost handles them natively
    print("Keeping missing values as NaN - XGBoost has specialized handling for missing values")
    
    # 4. Check for data leakage risk with improved approach
    print("Checking for potential data leakage with enhanced method...")
    
    # 4.1 Ensure strict time-based separation
    if 'tourney_date' in df.columns:
        df = df.sort_values('tourney_date').reset_index(drop=True)
        print("DataFrame sorted by tourney_date to enforce temporal order")
    
    # 4.2 Improved check for post-match statistics that might leak results
    leakage_keyword_patterns = [
        'result', 'winner', 'loser', 'win', 'loss', 'score', 'games_won',
        'sets_won', 'match_time', 'retirement', 'walkover'
    ]
    
    # 1. First pass: keyword-based detection
    potential_leakage_cols = []
    for pattern in leakage_keyword_patterns:
        matches = [col for col in df.columns if pattern in col.lower() and not col.endswith('_imputed')]
        potential_leakage_cols.extend(matches)
    
    # 2. Second pass: correlation-based detection (if target variable exists)
    correlated_cols = []
    if 'result' in df.columns:
        try:
            # Calculate correlations with the target
            correlations = df.corr()['result'].abs()
            # Identify suspiciously high correlations (above 0.8)
            suspiciously_high_corr = correlations[correlations > 0.8].index.tolist()
            if suspiciously_high_corr:
                print(f"Found {len(suspiciously_high_corr)} features with suspiciously high correlation to result")
                correlated_cols.extend(suspiciously_high_corr)
        except Exception as e:
            print(f"Warning: Could not calculate correlations due to: {e}")
    
    # Add correlation-based findings to potential leakage
    potential_leakage_cols.extend(correlated_cols)
    
    # Define safe feature patterns that are known to be correctly calculated
    safe_feature_patterns = [
        'win_rate_', 
        'h2h_win_pct_',
        'current_win_streak',
        'current_loss_streak',
        'elo_diff',
        'player_elo',
        'height_diff',
        'reliability_weight'
    ]
    
    # A feature is safe if it matches any safe pattern
    def is_safe_feature(feature_name):
        return any(pattern in feature_name for pattern in safe_feature_patterns)
    
    # Remove duplicates and filter out safe features
    potential_leakage_cols = list(set([col for col in potential_leakage_cols 
                                      if col in df.columns and not is_safe_feature(col)]))
    
    if potential_leakage_cols:
        print(f"Warning: Potential data leakage in {len(potential_leakage_cols)} columns: {potential_leakage_cols[:10]}..." 
              if len(potential_leakage_cols) > 10 else potential_leakage_cols)
        
        # Optionally, remove these columns from the dataframe
        # df = df.drop(columns=potential_leakage_cols)
        # print(f"Removed {len(potential_leakage_cols)} potentially leaky features")
    else:
        print("No potential data leakage detected in features")
    
    # 5. Add feature metadata - create a feature definitions JSON file
    feature_metadata = {}
    
    for col in df.columns:
        if col.endswith('_diff') and not col.endswith('_imputed'):
            # This is a prediction feature
            has_missing = bool(df[col].isna().any())  # Convert numpy.bool_ to Python bool
            missing_pct = float(df[col].isna().mean() * 100)  # Convert to float to ensure JSON serialization
            
            # Create metadata object with properly serializable types
            feature_metadata[col] = {
                'is_prediction_feature': True,
                'derived_from': [col.replace('_diff', '')],
                'imputation_flag': f"{col}_imputed" if f"{col}_imputed" in df.columns else None,
                'has_missing_values': has_missing,
                'missing_pct': missing_pct,
                'description': f"Difference between winner and loser {col.replace('_diff', '')}"
            }
    
    # Save feature metadata to JSON file
    import json
    metadata_path = OUTPUT_DIR / "feature_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(feature_metadata, f, indent=2)
    
    print(f"Feature metadata saved to {metadata_path}")
    
    # 6. Create XGBoost-friendly final feature list
    # Extract the difference features that will be used for prediction
    # These are features that represent the difference between player stats, excluding metadata columns
    
    prediction_features = [
        col for col in df.columns 
        if col.endswith('_diff') and not col.endswith('_imputed')
    ]
    
    # Add metadata column with the list of prediction features
    df.attrs['prediction_features'] = prediction_features
    
    print(f"Identified {len(prediction_features)} XGBoost-ready prediction features")
    print(f"Top prediction features: {prediction_features[:5]}...")
    
    return df

def process_player_streaks(player_ids_chunk: List[int], data: pd.DataFrame) -> Dict[int, Dict[str, List]]:
    """
    Process win/loss streaks for a chunk of players.
    
    Args:
        player_ids_chunk: List of player IDs to process
        data: DataFrame with player data
        
    Returns:
        Dictionary of player streak data
    """
    results = {}
    
    for i, player_id in enumerate(player_ids_chunk):
        player_mask = data['player_id'] == player_id
        if not player_mask.any():
            continue
            
        # Get player's matches in chronological order
        player_df = data.loc[player_mask].sort_values('tourney_date')
        
        # Initialize streak counters
        current_win_streak = 0
        current_loss_streak = 0
        
        # Store results for this player
        win_streaks = []
        loss_streaks = []
        indices = []
        
        # Calculate streaks
        for idx, row in player_df.iterrows():
            if row['result'] == 1:  # Win
                current_win_streak += 1
                current_loss_streak = 0
            else:  # Loss
                current_loss_streak += 1
                current_win_streak = 0
            
            # Store values
            indices.append(idx)
            win_streaks.append(current_win_streak)
            loss_streaks.append(current_loss_streak)
        
        # Save player results
        results[player_id] = {
            'indices': indices,
            'win_streaks': win_streaks,
            'loss_streaks': loss_streaks
        }
    
    return results


def process_player_streaks_gpu(player_ids_chunk: List[int], data: pd.DataFrame) -> Dict[int, Dict[str, List]]:
    """
    Process win/loss streaks for a chunk of players (GPU version).
    
    Args:
        player_ids_chunk: List of player IDs to process
        data: DataFrame with player data
        
    Returns:
        Dictionary of player streak data
    """
    results = {}
    
    for i, player_id in enumerate(player_ids_chunk):
        player_mask = data['player_id'] == player_id
        if not player_mask.any():
            continue
            
        # Get player's matches in chronological order
        player_df = data.loc[player_mask].sort_values('tourney_date')
        
        # Initialize streak counters
        current_win_streak = 0
        current_loss_streak = 0
        
        # Store results for this player
        win_streaks = []
        loss_streaks = []
        indices = []
        
        # Calculate streaks
        for idx, row in player_df.iterrows():
            if row['result'] == 1:  # Win
                current_win_streak += 1
                current_loss_streak = 0
            else:  # Loss
                current_loss_streak += 1
                current_win_streak = 0
            
            # Store values
            indices.append(idx)
            win_streaks.append(current_win_streak)
            loss_streaks.append(current_loss_streak)
        
        # Save player results
        results[player_id] = {
            'indices': indices,
            'win_streaks': win_streaks,
            'loss_streaks': loss_streaks
        }
    
    return results

if __name__ == "__main__":
    main() 