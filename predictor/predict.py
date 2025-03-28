import os
import sys
import time
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from pydantic import BaseModel, Field

# Check if running in Google Colab
def is_colab() -> bool:
    """Check if the code is running in Google Colab."""
    return 'google.colab' in sys.modules

# Install required packages if needed
if is_colab():
    try:
        import pandas as pd
        import numpy as np
        import xgboost
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                              "pandas", "numpy", "xgboost", "pydantic"])

    # Try to mount Google Drive if in Colab
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

# File paths
DATA_DIR = BASE_DIR / "data"
CLEANED_DATA_DIR = DATA_DIR / "cleaned"
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "figures"
OUTPUT_DIR = BASE_DIR / "predictor" / "output"

# Input and output files
MODEL_FILE = MODELS_DIR / "xgboost_model.json"
SCALER_FILE = MODELS_DIR / "feature_scaler.pkl"
FEATURES_FILE = CLEANED_DATA_DIR / "enhanced_features.csv"

class MatchPredictionInput(BaseModel):
    """Pydantic model for match prediction input."""
    player1_id: int = Field(..., description="ID of player 1")
    player2_id: int = Field(..., description="ID of player 2")
    surface: str = Field(..., description="Match surface (Hard, Clay, Grass, Carpet)")
    tourney_level: str = Field(..., description="Tournament level (ATP, Challenger, Grand Slam, etc.)")
    tournament_date: Optional[str] = Field(None, description="Tournament date (YYYY-MM-DD)")
    
    class Config:
        validate_assignment = True
        extra = "ignore"

class MatchPredictionResult(BaseModel):
    """Pydantic model for match prediction result."""
    player1_id: int = Field(..., description="ID of player 1")
    player2_id: int = Field(..., description="ID of player 2")
    player1_name: Optional[str] = Field(None, description="Name of player 1")
    player2_name: Optional[str] = Field(None, description="Name of player 2")
    predicted_winner_id: int = Field(..., description="ID of predicted winner")
    predicted_winner_name: Optional[str] = Field(None, description="Name of predicted winner")
    win_probability: float = Field(..., description="Probability of predicted winner winning")
    
    class Config:
        validate_assignment = True
        extra = "ignore"

class PlayerData(BaseModel):
    player_id: int
    name: str = Field("Unknown Player")
    elo: float = Field(1500.0)
    height_cm: Optional[float] = None
    win_rate: float = Field(0.5)
    win_rate_surface: Dict[str, float] = Field(default_factory=dict)
    win_rate_level: Dict[str, float] = Field(default_factory=dict)
    
    class Config:
        validate_assignment = True

class MatchPrediction(BaseModel):
    player1: PlayerData
    player2: PlayerData
    surface: str
    tournament_level: str
    win_probability_player1: float
    
    class Config:
        validate_assignment = True

def load_model() -> Tuple[xgb.Booster, Any, List[str]]:
    """
    Load the trained model, scaler, and feature list.
    
    Returns:
        Tuple of (model, scaler, feature_list)
    """
    try:
        model_path, scaler_path, feature_list_path = get_latest_model_files()
        
        print(f"Loading model from {model_path}")
        model = xgb.Booster()
        model.load_model(model_path)
        
        print(f"Loading scaler from {scaler_path}")
        scaler = joblib.load(scaler_path)
        
        print(f"Loading feature list from {feature_list_path}")
        with open(feature_list_path, 'r') as f:
            feature_list = [line.strip() for line in f.readlines()]
        
        return model, scaler, feature_list
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def get_latest_model_files() -> Tuple[str, str, str]:
    """
    Get the paths to the latest model, scaler, and feature list files.
    
    Returns:
        Tuple of (model_path, scaler_path, feature_list_path)
    """
    models = sorted(glob.glob(str(MODELS_DIR / "xgb_model_*.json")))
    scalers = sorted(glob.glob(str(MODELS_DIR / "scaler_*.joblib")))
    feature_lists = sorted(glob.glob(str(MODELS_DIR / "feature_list_*.txt")))
    
    if not models or not scalers or not feature_lists:
        raise FileNotFoundError("Model files not found. Please train a model first.")
    
    return models[-1], scalers[-1], feature_lists[-1]

def create_feature_vector(
    player1: PlayerData, 
    player2: PlayerData, 
    surface: str, 
    tournament_level: str,
    feature_list: List[str],
    scaler: Any
) -> np.ndarray:
    """
    Create a feature vector for the match.
    
    Args:
        player1: First player data
        player2: Second player data
        surface: Match surface (e.g., 'Hard', 'Clay', 'Grass')
        tournament_level: Tournament level (e.g., 'ATP', 'GSL')
        feature_list: List of features used by the model
        scaler: Trained scaler for feature normalization
        
    Returns:
        Normalized feature vector
    """
    # Create a dictionary to store features
    features = {}
    
    # Player Elo ratings
    features['winner_elo'] = player1.elo
    features['loser_elo'] = player2.elo
    features['elo_diff'] = player1.elo - player2.elo
    
    # Player heights if available
    if player1.height_cm and player2.height_cm:
        features['winner_ht'] = player1.height_cm
        features['loser_ht'] = player2.height_cm
        features['height_diff'] = player1.height_cm - player2.height_cm
    
    # Win rates
    features['winner_win_rate_20'] = player1.win_rate
    features['loser_win_rate_20'] = player2.win_rate
    features['win_rate_20_diff'] = player1.win_rate - player2.win_rate
    
    # Surface-specific win rates
    for s in ['Hard', 'Clay', 'Grass']:
        p1_rate = player1.win_rate_surface.get(s, player1.win_rate)
        p2_rate = player2.win_rate_surface.get(s, player2.win_rate)
        features[f'winner_win_rate_{s}_20'] = p1_rate
        features[f'loser_win_rate_{s}_20'] = p2_rate
        features[f'win_rate_{s}_20_diff'] = p1_rate - p2_rate
    
    # Tournament level win rates
    for level in ['ATP', 'GSL', 'CH', 'F']:
        p1_rate = player1.win_rate_level.get(level, player1.win_rate)
        p2_rate = player2.win_rate_level.get(level, player2.win_rate)
        features[f'winner_win_rate_{level}_50'] = p1_rate
        features[f'loser_win_rate_{level}_50'] = p2_rate
        features[f'win_rate_{level}_50_diff'] = p1_rate - p2_rate
    
    # Create a feature vector with zeros for all features
    feature_vector = np.zeros(len(feature_list))
    
    # Fill in the feature vector with available values
    for i, feature_name in enumerate(feature_list):
        if feature_name in features:
            feature_vector[i] = features[feature_name]
    
    # Normalize the feature vector
    normalized_vector = scaler.transform(feature_vector.reshape(1, -1))
    
    return normalized_vector

def predict_match(
    player1: PlayerData, 
    player2: PlayerData, 
    surface: str, 
    tournament_level: str
) -> MatchPrediction:
    """
    Predict the outcome of a tennis match.
    
    Args:
        player1: First player data
        player2: Second player data
        surface: Match surface
        tournament_level: Tournament level
        
    Returns:
        Match prediction with win probability for player1
    """
    # Load the model
    model, scaler, feature_list = load_model()
    
    # Create feature vector
    features = create_feature_vector(
        player1, player2, surface, tournament_level, feature_list, scaler
    )
    
    # Create DMatrix
    dtest = xgb.DMatrix(features, feature_names=feature_list)
    
    # Make prediction
    win_probability = float(model.predict(dtest)[0])
    
    # Create prediction object
    prediction = MatchPrediction(
        player1=player1,
        player2=player2,
        surface=surface,
        tournament_level=tournament_level,
        win_probability_player1=win_probability
    )
    
    return prediction

def load_features_data() -> pd.DataFrame:
    """
    Load the features dataset.
    
    Returns:
        DataFrame with match features
    """
    print(f"Loading features data from {FEATURES_FILE}...")
    
    # Check if features file exists
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(f"Features file not found: {FEATURES_FILE}")
    
    # Load the features data
    features_df = pd.read_csv(FEATURES_FILE)
    
    # Convert date column
    if 'tourney_date' in features_df.columns:
        features_df['tourney_date'] = pd.to_datetime(features_df['tourney_date'])
    
    print(f"Loaded {len(features_df)} matches")
    
    return features_df

def get_feature_columns(features_df: pd.DataFrame) -> List[str]:
    """
    Get the list of feature columns to use for prediction.
    
    Args:
        features_df: DataFrame with match features
        
    Returns:
        List of feature column names
    """
    # Define columns to exclude
    exclude_cols = ['tourney_date', 'winner_id', 'loser_id']
    
    # Get all numeric columns
    numeric_cols = features_df.select_dtypes(include=['float32', 'float64', 'int32', 'int64']).columns
    
    # Filter out excluded columns
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    return feature_cols

def predict_match_from_input(input_data: MatchPredictionInput) -> MatchPredictionResult:
    """
    Predict the winner of a tennis match from input data.
    
    Args:
        input_data: Match prediction input
        
    Returns:
        Match prediction result
    """
    print(f"Processing match prediction between players {input_data.player1_id} and {input_data.player2_id}...")
    
    # Load features data
    features_df = load_features_data()
    
    # Get feature columns
    feature_cols = get_feature_columns(features_df)
    
    # Prepare player data
    player1 = PlayerData(
        player_id=input_data.player1_id,
        name=features_df[features_df['winner_id'] == input_data.player1_id]['winner_name'].iloc[0],
        elo=features_df[features_df['winner_id'] == input_data.player1_id]['winner_elo'].iloc[0],
        height_cm=features_df[features_df['winner_id'] == input_data.player1_id]['winner_ht'].iloc[0],
        win_rate=features_df[features_df['winner_id'] == input_data.player1_id]['winner_win_rate_20'].iloc[0],
        win_rate_surface=features_df[features_df['winner_id'] == input_data.player1_id].set_index('surface')['winner_win_rate_20'].to_dict(),
        win_rate_level=features_df[features_df['winner_id'] == input_data.player1_id].set_index('tourney_level')['winner_win_rate_20'].to_dict()
    )
    
    player2 = PlayerData(
        player_id=input_data.player2_id,
        name=features_df[features_df['loser_id'] == input_data.player2_id]['loser_name'].iloc[0],
        elo=features_df[features_df['loser_id'] == input_data.player2_id]['loser_elo'].iloc[0],
        height_cm=features_df[features_df['loser_id'] == input_data.player2_id]['loser_ht'].iloc[0],
        win_rate=features_df[features_df['loser_id'] == input_data.player2_id]['loser_win_rate_20'].iloc[0],
        win_rate_surface=features_df[features_df['loser_id'] == input_data.player2_id].set_index('surface')['loser_win_rate_20'].to_dict(),
        win_rate_level=features_df[features_df['loser_id'] == input_data.player2_id].set_index('tourney_level')['loser_win_rate_20'].to_dict()
    )
    
    # Predict the match
    prediction = predict_match(player1, player2, input_data.surface, input_data.tourney_level)
    
    # Format the result
    result = MatchPredictionResult(
        player1_id=input_data.player1_id,
        player2_id=input_data.player2_id,
        player1_name=player1.name,
        player2_name=player2.name,
        predicted_winner_id=prediction.player1.player_id if prediction.win_probability_player1 > 0.5 else prediction.player2.player_id,
        predicted_winner_name=prediction.player1.name if prediction.win_probability_player1 > 0.5 else prediction.player2.name,
        win_probability=prediction.win_probability_player1
    )
    
    return result

def predict_from_command_line() -> None:
    """
    Predict a tennis match winner from command line arguments.
    """
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Predict the winner of a tennis match")
    parser.add_argument("--player1", type=int, required=True, help="ID of player 1")
    parser.add_argument("--player2", type=int, required=True, help="ID of player 2")
    parser.add_argument("--surface", type=str, required=True, choices=["Hard", "Clay", "Grass", "Carpet"], 
                        help="Match surface")
    parser.add_argument("--level", type=str, required=True, 
                        choices=["ATP", "Challenger", "Grand Slam", "Masters", "Davis Cup", "Tour Final", "Other"],
                        help="Tournament level")
    parser.add_argument("--date", type=str, help="Tournament date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    # Create input data
    input_data = MatchPredictionInput(
        player1_id=args.player1,
        player2_id=args.player2,
        surface=args.surface,
        tourney_level=args.level,
        tournament_date=args.date
    )
    
    # Predict the match
    result = predict_match_from_input(input_data)
    
    # Print the result
    print("\nPREDICTION RESULT:")
    print("=" * 50)
    print(f"Player 1 (ID: {result.player1_id}) vs. Player 2 (ID: {result.player2_id})")
    print(f"Surface: {input_data.surface}, Tournament level: {input_data.tourney_level}")
    print("\nPredicted winner: Player", "1" if result.predicted_winner_id == result.player1_id else "2", 
          f"(ID: {result.predicted_winner_id})")
    print(f"Win probability: {result.win_probability:.2%}")
    print("=" * 50)

def predict_interactive() -> None:
    """
    Predict a tennis match winner interactively.
    """
    print("\nTENNIS MATCH WINNER PREDICTION")
    print("=" * 50)
    
    # Get player IDs
    player1_id = int(input("Enter Player 1 ID: "))
    player2_id = int(input("Enter Player 2 ID: "))
    
    # Get match details
    print("\nMatch surface options: Hard, Clay, Grass, Carpet")
    surface = input("Enter match surface: ")
    
    print("\nTournament level options: ATP, Challenger, Grand Slam, Masters, Davis Cup, Tour Final, Other")
    tourney_level = input("Enter tournament level: ")
    
    # Get optional date
    date = input("\nEnter tournament date (YYYY-MM-DD, optional): ")
    if date.strip() == "":
        date = None
    
    # Create input data
    input_data = MatchPredictionInput(
        player1_id=player1_id,
        player2_id=player2_id,
        surface=surface,
        tourney_level=tourney_level,
        tournament_date=date
    )
    
    # Predict the match
    result = predict_match_from_input(input_data)
    
    # Print the result
    print("\nPREDICTION RESULT:")
    print("=" * 50)
    print(f"Player 1 (ID: {result.player1_id}) vs. Player 2 (ID: {result.player2_id})")
    print(f"Surface: {input_data.surface}, Tournament level: {input_data.tourney_level}")
    print("\nPredicted winner: Player", "1" if result.predicted_winner_id == result.player1_id else "2", 
          f"(ID: {result.predicted_winner_id})")
    print(f"Win probability: {result.win_probability:.2%}")
    print("=" * 50)

def main() -> None:
    """
    Main function to demonstrate prediction using the trained model.
    """
    # Load model, scaler and feature list once
    print("Loading model and related files...")
    model, scaler, feature_list = load_model()
    
    # Example player data
    player1 = PlayerData(
        player_id=1,
        name="Novak Djokovic",
        elo=2100.0,
        height_cm=188.0,
        win_rate=0.83,
        win_rate_surface={"Hard": 0.85, "Clay": 0.80, "Grass": 0.84},
        win_rate_level={"GSL": 0.87, "ATP": 0.82}
    )
    
    player2 = PlayerData(
        player_id=2,
        name="Rafael Nadal",
        elo=2050.0,
        height_cm=185.0,
        win_rate=0.82,
        win_rate_surface={"Hard": 0.77, "Clay": 0.92, "Grass": 0.78},
        win_rate_level={"GSL": 0.85, "ATP": 0.81}
    )
    
    # Create a modified prediction function that uses the already loaded model
    def predict_with_loaded_model(p1, p2, surface, tournament_level):
        features = create_feature_vector(p1, p2, surface, tournament_level, feature_list, scaler)
        dtest = xgb.DMatrix(features, feature_names=feature_list)
        win_probability = float(model.predict(dtest)[0])
        return MatchPrediction(
            player1=p1,
            player2=p2,
            surface=surface,
            tournament_level=tournament_level,
            win_probability_player1=win_probability
        )
    
    # Predict outcomes on different surfaces
    surfaces = ["Hard", "Clay", "Grass"]
    
    print("\nMatch Predictions:")
    print("=" * 80)
    
    for surface in surfaces:
        prediction = predict_with_loaded_model(player1, player2, surface, "GSL")
        
        print(f"\nSurface: {surface}")
        print(f"{player1.name} vs {player2.name}")
        print(f"Win probability for {player1.name}: {prediction.win_probability_player1:.2%}")
        print(f"Win probability for {player2.name}: {(1 - prediction.win_probability_player1):.2%}")
        
        winner = player1.name if prediction.win_probability_player1 > 0.5 else player2.name
        print(f"Predicted winner: {winner}")
        print("-" * 50)
    
    # Now try with reversed players on clay
    reversed_prediction = predict_with_loaded_model(player2, player1, "Clay", "GSL")
    
    print("\nReversed match on clay:")
    print(f"{player2.name} vs {player1.name}")
    print(f"Win probability for {player2.name}: {reversed_prediction.win_probability_player1:.2%}")
    print(f"Win probability for {player1.name}: {(1 - reversed_prediction.win_probability_player1):.2%}")
    
    # Check if the predictions are consistent (p1 vs p2 should be ~1 - p2 vs p1)
    p1_vs_p2 = predict_with_loaded_model(player1, player2, "Hard", "GSL").win_probability_player1
    p2_vs_p1 = predict_with_loaded_model(player2, player1, "Hard", "GSL").win_probability_player1
    
    print("\nConsistency check:")
    print(f"P({player1.name} beats {player2.name}) = {p1_vs_p2:.2%}")
    print(f"P({player2.name} beats {player1.name}) = {p2_vs_p1:.2%}")
    print(f"Sum of probabilities: {p1_vs_p2 + p2_vs_p1:.2%} (should be close to 100%)")
    
    if abs(p1_vs_p2 + p2_vs_p1 - 1.0) < 0.1:
        print("✓ Model predictions are consistent")
    else:
        print("✗ Model predictions are inconsistent - this suggests a potential issue")
        
    # Create a few more example players with different strengths
    player3 = PlayerData(
        player_id=3,
        name="Roger Federer",
        elo=2000.0,
        height_cm=185.0,
        win_rate=0.80,
        win_rate_surface={"Hard": 0.82, "Clay": 0.70, "Grass": 0.89},
        win_rate_level={"GSL": 0.84, "ATP": 0.80}
    )
    
    player4 = PlayerData(
        player_id=4,
        name="Andy Murray",
        elo=1950.0,
        height_cm=190.0,
        win_rate=0.75,
        win_rate_surface={"Hard": 0.78, "Clay": 0.68, "Grass": 0.82},
        win_rate_level={"GSL": 0.76, "ATP": 0.75}
    )
    
    print("\nAdditional Matchups:")
    print("=" * 80)
    
    # Test different matchups
    matchups = [
        (player1, player3, "Hard"),
        (player2, player3, "Clay"),
        (player3, player4, "Grass"),
        (player4, player1, "Hard")
    ]
    
    for p1, p2, surface in matchups:
        prediction = predict_with_loaded_model(p1, p2, surface, "GSL")
        print(f"\n{p1.name} vs {p2.name} on {surface}:")
        print(f"Win probability for {p1.name}: {prediction.win_probability_player1:.2%}")
        winner = p1.name if prediction.win_probability_player1 > 0.5 else p2.name
        print(f"Predicted winner: {winner}")
        print("-" * 40)

if __name__ == "__main__":
    main() 