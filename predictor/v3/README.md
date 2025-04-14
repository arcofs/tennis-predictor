# Tennis Match Prediction Pipeline

This directory contains the modules for the tennis match prediction pipeline, which can predict outcomes of upcoming tennis matches using machine learning.

## Overview

The prediction pipeline consists of several components:

1. **Fetch Upcoming Matches**: Retrieves information about upcoming tennis matches from the external API
2. **Generate Features**: Calculates predictive features for upcoming matches based on historical data
3. **Predict Outcomes**: Uses a trained XGBoost model to predict match outcomes
4. **Update Results**: Updates match statuses and prediction accuracy as matches are played

## Database Structure

The pipeline uses the following database tables:

- `matches`: Historical match data (existing)
- `match_features`: Features for historical matches (existing)
- `players`: Player information (existing)
- `upcoming_matches`: Data for scheduled future matches
- `upcoming_match_features`: Features calculated for upcoming matches
- `match_predictions`: Prediction results for upcoming matches

## Modules

### `fetch_upcoming_matches.py`

Fetches upcoming tennis matches from the API and stores them in the database.

```
python fetch_upcoming_matches.py --days 14
```

- `--days`: Number of days ahead to fetch tournaments for (default: 14)

### `generate_upcoming_features.py`

Generates features for upcoming matches based on historical data.

```
python generate_upcoming_features.py
```

### `predict_matches.py`

Makes predictions for upcoming matches using a trained model.

```
python predict_matches.py --model /path/to/model.pkl --output predictions.csv
```

- `--model`: Path to the trained model file (required)
- `--output`: Path to save prediction report (optional)
- `--no-display`: Suppress display of predictions (optional)

### `prediction_pipeline.py`

Runs the complete prediction pipeline in sequence.

```
python prediction_pipeline.py --model /path/to/model.pkl --days 14 --output predictions.csv
```

- `--model`: Path to the trained model file (required)
- `--days`: Number of days ahead to fetch tournaments for (default: 14)
- `--output`: Path to save prediction report (optional)

## Workflow

1. Train a model using `train_model_v3.py`
2. Run the prediction pipeline using `prediction_pipeline.py`
3. View predictions in the console or in the output CSV file
4. As matches are completed, the pipeline updates results and calculates prediction accuracy

## Setting Up as a Scheduled Job

To keep predictions up to date, you can set up the pipeline to run automatically:

### Using cron (Linux/Mac):

```bash
# Run daily at 1 AM
0 1 * * * cd /path/to/project && python -m predictor.v3.prediction_pipeline --model /path/to/model.pkl --output /path/to/reports/predictions_$(date +\%Y\%m\%d).csv
```

### Using Task Scheduler (Windows):

Create a batch file (run_predictions.bat):
```batch
cd /path/to/project
python -m predictor.v3.prediction_pipeline --model /path/to/model.pkl --output /path/to/reports/predictions_%date:~-4,4%%date:~-7,2%%date:~-10,2%.csv
```

Then schedule it to run daily using Windows Task Scheduler.

## Accuracy Tracking

The pipeline automatically tracks prediction accuracy:

- Overall accuracy
- Accuracy by surface type (hard, clay, grass, carpet)
- Accuracy by confidence level

You can view this information in the log file or query the database directly. 