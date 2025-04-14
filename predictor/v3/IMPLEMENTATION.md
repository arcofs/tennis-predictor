# Tennis Match Prediction Implementation Guide

This document provides implementation details for setting up and running the tennis match prediction pipeline.

## Architecture

The prediction pipeline is structured as follows:

```
Tennis Match Prediction Pipeline
├── Database Tables
│   ├── matches (historical matches)
│   ├── match_features (historical features)
│   ├── players (player information)
│   ├── upcoming_matches (future matches)
│   ├── upcoming_match_features (features for prediction)
│   └── match_predictions (prediction results)
│
├── Process Flow
│   ├── 1. Fetch upcoming matches from API
│   ├── 2. Generate features for upcoming matches
│   ├── 3. Make predictions using trained model
│   └── 4. Update results and accuracy metrics
```

## Database Schema Changes

We've added the following tables to the existing schema:

1. **upcoming_matches**: Stores data about future matches
2. **upcoming_match_features**: Contains calculated features for upcoming matches
3. **match_predictions**: Stores prediction results and accuracy metrics

## Implementation Steps

### 1. Set Up Database Tables

The new tables are created automatically by the pipeline scripts, but you can also create them manually:

```sql
-- Create upcoming_matches table
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

-- Create upcoming_match_features table
CREATE TABLE IF NOT EXISTS upcoming_match_features (
    id SERIAL PRIMARY KEY,
    match_id VARCHAR(50) UNIQUE REFERENCES upcoming_matches(match_id),
    player1_id INTEGER,
    player2_id INTEGER,
    surface VARCHAR(50),
    tournament_date DATE,
    tournament_level VARCHAR(50),
    player_elo_diff DOUBLE PRECISION,
    win_rate_5_diff DOUBLE PRECISION,
    win_streak_diff BIGINT,
    -- Additional feature columns omitted for brevity
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create match_predictions table
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
```

### 2. Train a Model

Before making predictions, you need a trained model:

```bash
python -m predictor.v3.train_model_v3 --output models/tennis_prediction_v3.pkl
```

### 3. Run the Full Prediction Pipeline

```bash
python -m predictor.v3.prediction_pipeline --model models/tennis_prediction_v3.pkl --days 14 --output predictions.csv
```

### 4. Run Individual Pipeline Components

If you want to run specific steps separately:

```bash
# Fetch upcoming matches
python -m predictor.v3.fetch_upcoming_matches --days 14

# Generate features
python -m predictor.v3.generate_upcoming_features

# Make predictions
python -m predictor.v3.predict_matches --model models/tennis_prediction_v3.pkl --output predictions.csv
```

## Handling Historical and Future Data

The design separates future matches (`upcoming_matches`) from historical matches (`matches`):

- Historical data is used to train the model and calculate features
- Future matches are stored separately until they're completed
- When a match is completed, its status is updated in the upcoming_matches table
- You could optionally move completed matches to the historical table

## Updating Predictions

As new data becomes available:

1. Run `fetch_upcoming_matches.py` to get the latest match schedule
2. Run `generate_upcoming_features.py` to calculate updated features
3. Run `predict_matches.py` to generate new predictions

The full pipeline (`prediction_pipeline.py`) handles all these steps for you.

## Tracking Prediction Accuracy

The pipeline automatically tracks prediction accuracy:

- Match results are updated as they become available
- Prediction accuracy is calculated and logged
- You can query the database to analyze performance

Example query for prediction accuracy:

```sql
SELECT 
    COUNT(*) as total_predictions,
    SUM(CASE WHEN prediction_correct = true THEN 1 ELSE 0 END) as correct_predictions,
    ROUND(SUM(CASE WHEN prediction_correct = true THEN 1 ELSE 0 END)::numeric / COUNT(*)::numeric * 100, 2) as accuracy_percentage
FROM match_predictions mp
JOIN upcoming_matches um ON mp.match_id = um.match_id
WHERE um.status = 'completed' AND mp.prediction_correct IS NOT NULL;
``` 