"""
Utility functions for the Stryktipset Predictor
"""

import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any
import hashlib

from config import LOG_LEVEL, LOG_FORMAT, LOGS_DIR


def setup_logging(name: str) -> logging.Logger:
    """
    Set up logging for a module
    
    Args:
        name: Name of the logger (usually __name__)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    formatter = logging.Formatter(LOG_FORMAT)
    console_handler.setFormatter(formatter)
    
    # File handler
    log_file = LOGS_DIR / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def calculate_points_from_result(home_goals: int, away_goals: int, is_home: bool) -> int:
    """
    Calculate points earned from a match result
    
    Args:
        home_goals: Goals scored by home team
        away_goals: Goals scored by away team
        is_home: Whether calculating for home team (True) or away team (False)
    
    Returns:
        Points earned (3 for win, 1 for draw, 0 for loss)
    """
    if home_goals == away_goals:
        return 1
    
    if is_home:
        return 3 if home_goals > away_goals else 0
    else:
        return 3 if away_goals > home_goals else 0


def calculate_form(recent_results: List[Dict[str, Any]], team_id: int) -> Dict[str, float]:
    """
    Calculate team form metrics from recent matches
    
    Args:
        recent_results: List of recent match dictionaries
        team_id: ID of the team to calculate form for
    
    Returns:
        Dictionary with form metrics (points, goals_scored, goals_conceded, etc.)
    """
    if not recent_results:
        return {
            "points": 0,
            "points_per_game": 0,
            "goals_scored": 0,
            "goals_conceded": 0,
            "goal_difference": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
        }
    
    total_points = 0
    goals_scored = 0
    goals_conceded = 0
    wins = 0
    draws = 0
    losses = 0
    
    for match in recent_results:
        is_home = match["home_team_id"] == team_id
        home_goals = match["home_goals"]
        away_goals = match["away_goals"]
        
        points = calculate_points_from_result(home_goals, away_goals, is_home)
        total_points += points
        
        if is_home:
            goals_scored += home_goals
            goals_conceded += away_goals
        else:
            goals_scored += away_goals
            goals_conceded += home_goals
        
        if points == 3:
            wins += 1
        elif points == 1:
            draws += 1
        else:
            losses += 1
    
    num_matches = len(recent_results)
    
    return {
        "points": total_points,
        "points_per_game": total_points / num_matches,
        "goals_scored": goals_scored,
        "goals_scored_per_game": goals_scored / num_matches,
        "goals_conceded": goals_conceded,
        "goals_conceded_per_game": goals_conceded / num_matches,
        "goal_difference": goals_scored - goals_conceded,
        "wins": wins,
        "draws": draws,
        "losses": losses,
    }


def date_to_string(date: datetime) -> str:
    """Convert datetime to API-friendly string format"""
    return date.strftime("%Y-%m-%d")


def string_to_date(date_str: str) -> datetime:
    """Convert string to datetime object"""
    return datetime.strptime(date_str, "%Y-%m-%d")


def days_between(date1: str, date2: str) -> int:
    """
    Calculate days between two date strings
    
    Args:
        date1: First date string (YYYY-MM-DD)
        date2: Second date string (YYYY-MM-DD)
    
    Returns:
        Number of days between dates
    """
    d1 = string_to_date(date1)
    d2 = string_to_date(date2)
    return abs((d2 - d1).days)


def generate_cache_key(endpoint: str, params: Dict[str, Any]) -> str:
    """
    Generate a cache key from endpoint and parameters
    
    Args:
        endpoint: API endpoint
        params: Request parameters
    
    Returns:
        MD5 hash to use as cache key
    """
    cache_string = f"{endpoint}_{json.dumps(params, sort_keys=True)}"
    return hashlib.md5(cache_string.encode()).hexdigest()


def calculate_prediction_accuracy(predictions: List[str], actual_results: List[str]) -> Dict[str, float]:
    """
    Calculate accuracy metrics for predictions
    
    Args:
        predictions: List of predictions ('H', 'D', 'A')
        actual_results: List of actual results ('H', 'D', 'A')
    
    Returns:
        Dictionary with accuracy metrics
    """
    if len(predictions) != len(actual_results):
        raise ValueError("Predictions and actual results must have same length")
    
    if not predictions:
        return {"accuracy": 0.0, "total": 0, "correct": 0}
    
    correct = sum(1 for pred, actual in zip(predictions, actual_results) if pred == actual)
    total = len(predictions)
    
    # Calculate accuracy by outcome type
    outcomes = ['H', 'D', 'A']
    accuracy_by_outcome = {}
    
    for outcome in outcomes:
        outcome_predictions = [(p, a) for p, a in zip(predictions, actual_results) if a == outcome]
        if outcome_predictions:
            outcome_correct = sum(1 for p, a in outcome_predictions if p == a)
            accuracy_by_outcome[outcome] = outcome_correct / len(outcome_predictions)
        else:
            accuracy_by_outcome[outcome] = 0.0
    
    return {
        "accuracy": correct / total,
        "total": total,
        "correct": correct,
        "home_accuracy": accuracy_by_outcome['H'],
        "draw_accuracy": accuracy_by_outcome['D'],
        "away_accuracy": accuracy_by_outcome['A'],
    }


def get_match_result(home_goals: int, away_goals: int) -> str:
    """
    Determine match result
    
    Args:
        home_goals: Goals scored by home team
        away_goals: Goals scored by away team
    
    Returns:
        'H' for home win, 'D' for draw, 'A' for away win
    """
    if home_goals > away_goals:
        return 'H'
    elif home_goals < away_goals:
        return 'A'
    else:
        return 'D'