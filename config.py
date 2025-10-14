"""
Configuration file for Stryktipset Predictor
Store API keys, paths, and model parameters
"""

import os
from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API-Football Configuration
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")
API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"
API_RATE_LIMIT_CALLS = 10000  
API_RATE_LIMIT_PERIOD = 86400  

# Find LEAGUES section and change to:
LEAGUES = {
    "premier_league": 39,
    "championship": 40,
    "league_one": 41,
    "league_two": 42
}

# Team name variations
TEAM_NAME_MAPPING = {
    "Man United": "Manchester United",
    "Mboro": "Middlesbrough",
    "Charlotn": "Charlton",
    "Huddersf.": "Huddersfield",
    
    # Add these three:
    "Wolverhampton": "Wolves",
    "West Bromwich": "West Brom",
    "Sheffield United": "Sheffield Utd",
}

# Find SEASONS section and change to:
SEASONS = [2020, 2021, 2022, 2023, 2024, 2025]

# Feature Engineering Parameters
FORM_WINDOW = 5  # Last N matches for form calculation
RECENT_H2H_MATCHES = 5  # Head-to-head matches to consider
MIN_MATCHES_FOR_PREDICTION = 5  # Minimum matches played before predicting

# Rule-Based Predictor Weights
RULE_WEIGHTS = {
    "form": 0.40,
    "league_position": 0.20,
    "home_advantage": 0.15,
    "head_to_head": 0.15,
    "goal_difference": 0.10,
}

# Rule-Based Thresholds
HOME_WIN_THRESHOLD = 0.55  # Confidence threshold for predicting home win
AWAY_WIN_THRESHOLD = 0.55
DRAW_LOWER_THRESHOLD = 0.30  # If both teams between these values, predict draw
DRAW_UPPER_THRESHOLD = 0.50

# Machine Learning Parameters
ML_TEST_SIZE = 0.2
ML_RANDOM_STATE = 42
ML_CV_FOLDS = 5

ML_FEATURES_ENHANCED = [
    # === BASIC FORM (Home/Away specific) ===
    "home_form_points",
    "home_goals_scored_avg",
    "home_goals_conceded_avg",
    "home_win_rate",
    "home_clean_sheet_rate",
    "home_failed_to_score_rate",
    
    "away_form_points",
    "away_goals_scored_avg",
    "away_goals_conceded_avg",
    "away_win_rate",
    "away_clean_sheet_rate",
    "away_failed_to_score_rate",
    
    # === OVERALL FORM ===
    "home_overall_form_points",
    "away_overall_form_points",
    
    # === FORM DIFFERENTIALS ===
    "form_points_diff",
    "goals_scored_diff",
    "goals_conceded_diff",
    
    # === MOMENTUM (NEW!) ===
    "home_momentum",
    "away_momentum",
    "momentum_diff",
    
    # === BTTS & OVER/UNDER (NEW!) ===
    "home_btts_rate",
    "away_btts_rate",
    "home_over25_rate",
    "away_over25_rate",
    "home_avg_total_goals",
    "away_avg_total_goals",
    "combined_btts_likelihood",
    "combined_over25_likelihood",
    
    # === SCORING CONSISTENCY (NEW!) ===
    "home_scoring_std",
    "away_scoring_std",
    "home_scoring_range",
    "away_scoring_range",
    
    # === LEAGUE POSITION ===
    "home_position",
    "away_position",
    "position_diff",
    "points_diff",
    "goal_diff_diff",
    
    # === HEAD-TO-HEAD ===
    "h2h_home_wins",
    "h2h_draws",
    "h2h_away_wins",
    "h2h_home_goals_avg",
    "h2h_away_goals_avg",
    "h2h_total_matches",
    
    # === REST DAYS ===
    "days_since_last_match_home",
    "days_since_last_match_away",
    "rest_days_diff",
    
    # === TEMPORAL FEATURES (NEW!) ===
    "month",
    "day_of_week",
    "is_weekend",
    "is_midweek",
    "is_december",
    "is_holiday_period",
    "week_of_season",
    
    # === FIXTURE DIFFICULTY (NEW!) ===
    "home_recent_opponent_strength",
    "away_recent_opponent_strength",
    
    # === DERBY (NEW!) ===
    "is_derby",
    
    # === INJURIES (NEW!) ===
    "home_missing_players",
    "away_missing_players",
    "home_questionable_players",
    "away_questionable_players",
    "injury_impact_diff",
    
    # === API TEAM STATISTICS (NEW!) ===
    "home_season_goals_avg",
    "away_season_goals_avg",
    "home_season_conceded_avg",
    "away_season_conceded_avg",
    "season_goals_diff",
    "season_defense_diff",
    "home_clean_sheet_pct",
    "away_clean_sheet_pct",
    "home_failed_to_score_pct",
    "away_failed_to_score_pct",
    "home_season_win_pct",
    "away_season_win_pct",
    "home_season_draw_pct",
    "away_season_draw_pct",
    "home_penalty_conversion",
    "away_penalty_conversion",
]

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Cache settings
CACHE_EXPIRY_HOURS = 24  # How long to keep cached API responses

# Prediction output
PREDICTION_CONFIDENCE_THRESHOLD = 0.4  # Only show predictions above this confidence