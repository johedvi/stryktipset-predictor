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
API_FOOTBALL_KEY = "100ef076dfd4356ae5902647b44e3255"
API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"
API_RATE_LIMIT_CALLS = 10000  # Pro plan daily limit (check your dashboard)
API_RATE_LIMIT_PERIOD = 86400  # 24 hours

# Find LEAGUES section and change to:
LEAGUES = {
    "premier_league": 39,
    "championship": 40,
    "league_one": 41,
    "league_two": 42
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

# Features to use for ML model
ML_FEATURES = [
    "home_form_points",
    "away_form_points",
    "home_goals_scored_avg",
    "away_goals_scored_avg",
    "home_goals_conceded_avg",
    "away_goals_conceded_avg",
    # Add these:
    "home_overall_form_points",      # Overall form
    "away_overall_form_points",
    "form_points_diff",              # Form differential
    "goals_scored_diff",
    "goals_conceded_diff",
    "home_win_rate",                 # Win rates
    "away_win_rate",
    "home_clean_sheet_rate",         # Defense metrics
    "away_clean_sheet_rate",
    "h2h_home_wins",
    "h2h_draws",
    "h2h_away_wins",
    "h2h_home_goals_avg",           # H2H goals
    "h2h_away_goals_avg",
    "days_since_last_match_home",
    "days_since_last_match_away",
    "rest_days_diff",               # Rest advantage
]

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Cache settings
CACHE_EXPIRY_HOURS = 24  # How long to keep cached API responses

# Prediction output
PREDICTION_CONFIDENCE_THRESHOLD = 0.4  # Only show predictions above this confidence