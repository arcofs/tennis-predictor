# Player Statistics Needed (for both players)
player_stats = {
    'basic_stats': {
        'elo_rating': float,  # Current Elo rating
        'win_streak': int,    # Current winning streak
        'loss_streak': int,   # Current losing streak
    },
    
    'recent_performance': {
        'win_rate_5': float,  # Win rate in last 5 matches
        'win_rate_10': float, # Win rate in last 10 matches
    },
    
    'surface_stats': {
        'Hard': {
            'win_rate_5': float,     # Recent win rate on hard courts
            'win_rate_overall': float # Overall win rate on hard courts
        },
        'Clay': {
            'win_rate_5': float,
            'win_rate_overall': float
        },
        'Grass': {
            'win_rate_5': float,
            'win_rate_overall': float
        }
    },
    
    'serve_stats': {
        'serve_efficiency_5': float,      # Recent serve efficiency
        'first_serve_pct_5': float,       # First serve percentage
        'first_serve_win_pct_5': float,   # First serve win percentage
        'second_serve_win_pct_5': float,  # Second serve win percentage
        'ace_pct_5': float,               # Ace percentage
        'bp_saved_pct_5': float           # Break points saved percentage
    },
    
    'return_stats': {
        'return_efficiency_5': float,    # Return efficiency
        'bp_conversion_pct_5': float     # Break point conversion rate
    }
}

# Match Information Needed
match_info = {
    'player1_id': str,
    'player2_id': str,
    'surface': str,  # One of: ['Hard', 'Clay', 'Grass', 'Carpet']
    'tournament_date': datetime
}