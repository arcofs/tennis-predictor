# Tennis Match Predictor

A machine learning system that predicts tennis match outcomes based on player statistics and match conditions.

## Overview

This project uses XGBoost to predict tennis match outcomes based on player statistics, surface type, and tournament level information. The model is trained on historical tennis data with features such as player Elo ratings, win rates, height, and surface-specific performance.

## Directory Structure

- `data/`: Contains raw tennis match data
- `predictor/`: Python scripts for feature generation, model training, and prediction
- `models/`: Saved models, scalers, and feature lists

## Key Components

### Feature Generation (`generate_features.py`)

Processes raw tennis match data to create derived features like:
- Player Elo ratings
- Surface-specific win rates
- Tournament level performance
- Physical matchup features (height difference)
- Head-to-head statistics

### Model Training (`train_model.py`)

Trains an XGBoost model with safeguards against data leakage:
- Removes post-match statistics that could cause leakage
- Uses proper time-based validation
- Creates balanced training data by flipping winner/loser pairs
- Implements rigorous cross-validation
- Performs checks for potential data leakage

### Prediction (`predict.py`)

Makes predictions for tennis match outcomes:
- Loads trained model, scaler, and feature list
- Creates feature vectors for player matchups
- Calculates win probabilities
- Supports prediction on different surfaces

## Addressing Data Leakage

Data leakage was a major concern in the initial model which achieved unrealistic 100% accuracy. The following measures were implemented to prevent leakage:

1. **Feature Selection**: Removed all post-match statistics and features that directly reveal match outcomes, including:
   - Match performance statistics (aces, service points, etc.)
   - Head-to-head statistics calculated after matches
   - Any feature derived directly from match outcomes

2. **Temporal Validation**: 
   - Implemented time-series cross-validation
   - Ensured test data comes chronologically after training data
   - Created stratified folds that maintain proper time order

3. **Balanced Dataset Creation**:
   - Created synthetic negative examples by flipping winner/loser pairs
   - Ensured both classes are represented in train and test sets
   - Maintained proper feature relationships when flipping pairs

4. **Leakage Detection**:
   - Added automatic checks for suspiciously high accuracy
   - Analyzed feature importance for potential leaky features
   - Validated consistent predictions when flipping player positions

## Usage

### Training the Model

```bash
source .venv/bin/activate
python predictor/train_model.py
```

This will:
1. Load the enhanced features dataset
2. Preprocess the data to prevent leakage
3. Create a time-based train/test split
4. Train an XGBoost model
5. Perform cross-validation
6. Save the model, scaler, and feature list

### Making Predictions

```bash
source .venv/bin/activate
python predictor/predict.py
```

This will:
1. Load the trained model
2. Make predictions for example player matchups
3. Show win probabilities on different surfaces
4. Validate prediction consistency

## Model Performance

The model achieves around 72% accuracy on the test set and 88% in cross-validation, which is realistic for tennis match prediction. The most important features are:
- Elo rating difference
- Surface-specific win rates
- Tournament level performance

## Requirements

See `requirements.txt` for dependencies. Key libraries include:
- pandas >= 2.0.0
- xgboost >= 2.0.0
- scikit-learn >= 1.3.0
- numpy >= 1.24.0
- pydantic >= 2.0.0

## Project Structure

```
tennis-predictor/
├── data/
│   ├── raw/                  # Raw CSV files from tennis_atp repository
│   └── cleaned/              # Cleaned and processed datasets
├── models/                   # Trained models and scalers
├── figures/                  # Visualization outputs
├── predictor/
│   ├── generate_features.py  # Feature engineering script
│   ├── train_model.py        # Model training script
│   ├── predict.py            # Match prediction script
│   ├── predict_tournament.py # Tournament simulation script
│   └── output/               # Prediction outputs
└── README.md                 # This file
```

## Setup

### Requirements

- Python 3.7+
- Packages: pandas, numpy, xgboost, scikit-learn, pydantic, matplotlib, seaborn

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/tennis-predictor.git
   cd tennis-predictor
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### 1. Feature Engineering

Run the feature engineering script to generate player and match features:

```python
python predictor/generate_features.py
```

This will read the cleaned dataset with Elo ratings and generate enhanced features for training.

### 2. Model Training

Train the XGBoost model:

```python
python predictor/train_model.py
```

This will:
- Load the enhanced features dataset
- Preprocess the data
- Train an XGBoost model with optimized hyperparameters
- Evaluate model performance
- Save the trained model and metrics

### 3. Match Prediction

Predict the outcome of a specific match:

```python
python predictor/predict.py
```

You can use the interactive mode to enter player IDs and match details, or provide command-line arguments:

```python
python predictor/predict.py --player1 104122 --player2 103285 --surface Hard --level "Grand Slam" --date "2023-01-15"
```

### 4. Tournament Simulation

Simulate an entire tournament and predict the winner:

```python
python predictor/predict_tournament.py
```

Edit the `main()` function in the script to customize the tournament format, participants, and other details.

## Using in Google Colab

All scripts are designed to work in Google Colab with minimal setup:

1. Upload your data to Google Drive
2. Mount your Google Drive in Colab
3. Run the scripts as cells in your notebook

Example Colab cell:
```python
!python predictor/train_model.py
```

## Model Features

The prediction model uses various features including:

- Player Elo ratings
- Surface-specific performance
- Recent form (win rates, streaks)
- Head-to-head statistics
- Tournament level performance
- Service statistics

## Performance

The model achieves strong predictive accuracy on historical tennis match data, correctly predicting match outcomes in approximately 70-80% of cases (depending on the test set and features used).

## Customization

To customize the model:

1. Adjust hyperparameters in `train_model.py`
2. Add new features in `generate_features.py`
3. Modify prediction thresholds in `predict.py`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- ATP match data from [Jeff Sackmann's tennis_atp repository](https://github.com/JeffSackmann/tennis_atp)
- Elo rating system adapted from [FiveThirtyEight's tennis Elo methodology](https://fivethirtyeight.com/features/how-were-forecasting-the-2018-tennis-grand-slams/) 