import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from typing import Optional
from datetime import datetime
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field, PostgresDsn, validator
import logging
from pathlib import Path
import re
import numpy as np
from typing import Union, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MatchData(BaseModel):
    """Pydantic model for match data validation"""
    tournament_id: str
    tournament_name: str
    surface: str
    draw_size: Optional[float] = None
    tournament_level: str
    tournament_date: datetime
    match_num: Optional[int] = None
    winner_id: Optional[int] = None
    winner_seed: Optional[float] = None
    winner_entry: Optional[str] = None
    winner_name: str
    winner_hand: Optional[str] = None
    winner_height_cm: Optional[float] = None
    winner_country_code: Optional[str] = None
    winner_age: Optional[float] = None
    loser_id: Optional[int] = None
    loser_seed: Optional[float] = None
    loser_entry: Optional[str] = None
    loser_name: str
    loser_hand: Optional[str] = None
    loser_height_cm: Optional[float] = None
    loser_country_code: Optional[str] = None
    loser_age: Optional[float] = None
    score: Optional[str] = None
    best_of: Optional[int] = None
    round: str
    minutes: Optional[float] = None
    winner_aces: Optional[float] = None
    winner_double_faults: Optional[float] = None
    winner_serve_points: Optional[float] = None
    winner_first_serves_in: Optional[float] = None
    winner_first_serve_points_won: Optional[float] = None
    winner_second_serve_points_won: Optional[float] = None
    winner_service_games: Optional[float] = None
    winner_break_points_saved: Optional[float] = None
    winner_break_points_faced: Optional[float] = None
    loser_aces: Optional[float] = None
    loser_double_faults: Optional[float] = None
    loser_serve_points: Optional[float] = None
    loser_first_serves_in: Optional[float] = None
    loser_first_serve_points_won: Optional[float] = None
    loser_second_serve_points_won: Optional[float] = None
    loser_service_games: Optional[float] = None
    loser_break_points_saved: Optional[float] = None
    loser_break_points_faced: Optional[float] = None
    winner_rank: Optional[float] = None
    winner_rank_points: Optional[float] = None
    loser_rank: Optional[float] = None
    loser_rank_points: Optional[float] = None
    match_type: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
        alias_generator = lambda x: {
            'tournament_id': 'tourney_id',
            'tournament_name': 'tourney_name',
            'tournament_level': 'tourney_level',
            'tournament_date': 'tourney_date',
            'winner_aces': 'w_ace',
            'winner_double_faults': 'w_df',
            'winner_serve_points': 'w_svpt',
            'winner_first_serves_in': 'w_1stIn',
            'winner_first_serve_points_won': 'w_1stWon',
            'winner_second_serve_points_won': 'w_2ndWon',
            'winner_service_games': 'w_SvGms',
            'winner_break_points_saved': 'w_bpSaved',
            'winner_break_points_faced': 'w_bpFaced',
            'loser_aces': 'l_ace',
            'loser_double_faults': 'l_df',
            'loser_serve_points': 'l_svpt',
            'loser_first_serves_in': 'l_1stIn',
            'loser_first_serve_points_won': 'l_1stWon',
            'loser_second_serve_points_won': 'l_2ndWon',
            'loser_service_games': 'l_SvGms',
            'loser_break_points_saved': 'l_bpSaved',
            'loser_break_points_faced': 'l_bpFaced',
            'winner_height_cm': 'winner_ht',
            'winner_country_code': 'winner_ioc',
            'loser_height_cm': 'loser_ht',
            'loser_country_code': 'loser_ioc'
        }.get(x, x)

    @validator('*', pre=True)
    def handle_nan(cls, v):
        if pd.isna(v):
            return None
        return v

    @validator('tourney_date', pre=True)
    def parse_date(cls, v):
        if pd.isna(v):
            return None
        try:
            # First try parsing with time
            return pd.to_datetime(str(v))
        except Exception as e:
            try:
                # If that fails, try parsing just the date
                return datetime.strptime(str(v), '%Y-%m-%d')
            except ValueError as e:
                logger.warning(f"Invalid date format: {v}. Error: {str(e)}")
                return None

def validate_database_url(url: str) -> str:
    """Validate and format database URL to ensure correct dialect"""
    if url.startswith('postgres://'):
        url = url.replace('postgres://', 'postgresql://', 1)
    return url

def create_matches_table(engine: create_engine) -> None:
    """Create the matches table if it doesn't exist"""
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS matches (
                    id SERIAL PRIMARY KEY,
                    tournament_id VARCHAR(50) NOT NULL,
                    tournament_name VARCHAR(255) NOT NULL,
                    surface VARCHAR(50) NOT NULL,
                    draw_size FLOAT,
                    tournament_level VARCHAR(50) NOT NULL,
                    tournament_date TIMESTAMP NOT NULL,
                    match_num BIGINT,
                    winner_id INTEGER,
                    winner_seed FLOAT,
                    winner_entry VARCHAR(50),
                    winner_name VARCHAR(255) NOT NULL,
                    winner_hand VARCHAR(1),
                    winner_height_cm FLOAT,
                    winner_country_code VARCHAR(3),
                    winner_age FLOAT,
                    loser_id INTEGER,
                    loser_seed FLOAT,
                    loser_entry VARCHAR(50),
                    loser_name VARCHAR(255) NOT NULL,
                    loser_hand VARCHAR(1),
                    loser_height_cm FLOAT,
                    loser_country_code VARCHAR(3),
                    loser_age FLOAT,
                    score VARCHAR(50),
                    best_of INTEGER,
                    round VARCHAR(50) NOT NULL,
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
                    match_type VARCHAR(50)
                )
            """))
            conn.commit()
        logger.info("Matches table created successfully")
    except SQLAlchemyError as e:
        logger.error(f"Error creating matches table: {str(e)}")
        raise

def load_and_validate_data(file_path: Path) -> pd.DataFrame:
    """Load and validate the CSV data"""
    try:
        df: pd.DataFrame = pd.read_csv(file_path)
        
        # Replace NaN values with None for proper validation
        df = df.replace({np.nan: None})
        
        # Convert tourney_date to datetime using pandas
        df['tourney_date'] = pd.to_datetime(df['tourney_date'])
        
        # Validate each row using Pydantic model
        validated_data: list = []
        total_rows = len(df)
        valid_rows = 0
        
        for idx, row in df.iterrows():
            try:
                validated_row = MatchData(**row.to_dict())
                validated_data.append(validated_row.dict())
                valid_rows += 1
                
                # Log progress every 1000 rows
                if idx % 1000 == 0:
                    logger.info(f"Processed {idx}/{total_rows} rows...")
                    
            except Exception as e:
                logger.warning(f"Invalid row {idx}: {str(e)}")
        
        logger.info(f"Validation complete. {valid_rows}/{total_rows} rows valid.")
        return pd.DataFrame(validated_data)
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def main() -> None:
    """Main function to load data into database"""
    try:
        # Load environment variables
        load_dotenv()
        database_url: str = os.getenv('DATABASE_URL')
        if not database_url:
            raise ValueError("DATABASE_URL not found in environment variables")

        # Validate and format database URL
        database_url = validate_database_url(database_url)
        logger.info(f"Using database URL: {re.sub(r'://[^:]+:[^@]+@', '://*****:*****@', database_url)}")  # Log URL with hidden credentials

        # Create database engine
        engine = create_engine(database_url)

        # Create matches table
        create_matches_table(engine)

        # Load and validate data
        file_path = Path('data/raw/combined_matches_1990_2024.csv')
        df = load_and_validate_data(file_path)

        # Insert data into database
        df.to_sql('matches', engine, if_exists='append', index=False)
        logger.info(f"Successfully inserted {len(df)} rows into matches table")

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    main()
