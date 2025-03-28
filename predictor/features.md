# Tennis Match Prediction Features

## Core Features Implementation Guide

### 1. Elo Rating Features
- **Base Elo Difference**
  - Implementation: `winner_elo - loser_elo`
  - Type: `float32`
  - Importance: Primary predictor

- **Surface-Specific Elo**
  - Implementation: Calculate separate Elo ratings for each surface type
  - Rolling window: Last 2 years of matches
  - Update formula: `new_elo = old_elo + K * (actual_result - expected_result)`
  - Type: `float32`

### 2. Recent Form Features (20-match rolling window)
- **Overall Win Rate**
  ```python
  wins_last_20 = matches_last_20.apply(lambda x: sum(x == 1)) / 20
  ```
  - Type: `float32`
  - Window: Last 20 matches

- **Surface Win Rate**
  ```python
  surface_wins = matches_surface_last_20.apply(lambda x: sum(x == 1)) / len(x)
  ```
  - Type: `float32`
  - Window: Last 20 matches on specific surface

- **Momentum Indicators**
  ```python
  # Last 5 matches only
  consecutive_wins = matches_last_5.apply(lambda x: max(consecutive_true(x == 1)))
  consecutive_losses = matches_last_5.apply(lambda x: max(consecutive_true(x == 0)))
  ```
  - Type: `int8`
  - Window: Last 5 matches

### 3. Serve Performance Features (10-match rolling window)
- **Ace Rate**
  ```python
  ace_rate = rolling_sum(aces) / rolling_sum(serve_points)
  ```
  - Type: `float32`

- **First Serve Win Rate**
  ```python
  first_serve_win_rate = rolling_sum(first_serve_points_won) / rolling_sum(first_serve_points_total)
  ```
  - Type: `float32`

- **Break Point Defense**
  ```python
  bp_save_rate = rolling_sum(break_points_saved) / rolling_sum(break_points_faced)
  ```
  - Type: `float32`

### 4. Physical Matchup Features
- **Height Advantage**
  ```python
  height_diff = player1_height - player2_height
  ```
  - Type: `int16`

### 5. Tournament Level Performance (50-match rolling window)
- **Tournament Level Win Rate**
  ```python
  tourney_win_rate = matches_at_level.apply(lambda x: sum(x == 1) / len(x))
  ```
  - Type: `float32`
  - Grouped by tournament level (Grand Slam, Masters, etc.)

### 7. Head-to-Head Features
- **Historical H2H Statistics**
  - Features:
    - `h2h_wins_player1`: Total wins of player1 vs player2
    - `h2h_wins_player2`: Total wins of player2 vs player1
    - `h2h_win_rate_player1`: Win rate of player1 vs player2
    - `h2h_total_matches`: Total matches between the players
  - Types:
    - Use `int16` for count features
    - Use `float32` for rate features
  - Optimization Requirements:
    - Pre-calculate H2H statistics using player ID pairs as keys
    - Store in memory-efficient lookup table
    - Sort player IDs in pair to ensure consistent lookups
    - Update incrementally as new matches occur

- **Recent H2H Performance**
  - Features:
    - `recent_h2h_wins`: Wins in last 2 years
    - `recent_h2h_win_rate`: Win rate in last 2 years
  - Window: Last 2 years of matches
  - Types:
    - Use `int16` for count features
    - Use `float32` for rate features
  - Default Values:
    - Use 0 for win counts when no matches
    - Use 0.5 for win rates when no matches
  - Optimization Requirements:
    - Filter matches by date first
    - Use vectorized operations for calculations
    - Cache results for frequently accessed pairs

## Implementation Notes

### Performance Optimization
1. Use efficient data types:
   ```python
   dtype_dict = {
       'elo_diff': 'float32',
       'win_rate': 'float32',
       'consecutive_wins': 'int8',
       'height_diff': 'int16'
   }
   ```

2. Vectorized Operations:
   ```python
   # Instead of loops:
   df['rolling_wins'] = df.groupby('player_id')['result'].rolling(20).mean()
   ```

3. Pre-calculation Strategy:
   ```python
   # Calculate once per tournament update
   surface_elo_ratings = calculate_surface_elo(historical_matches)
   
   # Calculate daily
   recent_form_features = calculate_recent_form(last_50_matches)
   ```

### Feature Generation Order
1. Calculate static features first (height_diff)
2. Generate rolling window features in batch operations
3. Calculate Elo ratings chronologically
4. Compute final feature differences between players

### Memory Management
- Process in chunks of 50,000 matches
- Use `float32` instead of `float64` where possible
- Drop intermediate columns after feature calculation
- Use sparse matrices for one-hot encoded features

### Required Columns from Dataset
- `tourney_date`
- `winner_id`, `loser_id`
- `winner_elo`, `loser_elo`
- `surface`
- `tourney_level`
- `winner_ht`, `loser_ht`
- `w_ace`, `w_svpt`, `l_ace`, `l_svpt`
- `w_1stWon`, `w_1stIn`, `l_1stWon`, `l_1stIn`
- `w_bpSaved`, `w_bpFaced`, `l_bpSaved`, `l_bpFaced`

### Feature Scaling
- Elo ratings: No scaling needed (already normalized)
- Win rates: Already 0-1 scaled
- Physical features: StandardScaler