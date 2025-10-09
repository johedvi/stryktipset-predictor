"""
Rule-based predictor using domain knowledge and heuristics
"""

import json
import pandas as pd
from typing import Dict, List, Any, Tuple
from datetime import datetime

from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    RULE_WEIGHTS,
    HOME_WIN_THRESHOLD,
    AWAY_WIN_THRESHOLD,
    DRAW_LOWER_THRESHOLD,
    DRAW_UPPER_THRESHOLD,
    FORM_WINDOW,
    RECENT_H2H_MATCHES,
    MIN_MATCHES_FOR_PREDICTION,
)
from utils import setup_logging, calculate_form, get_match_result

logger = setup_logging(__name__)


class RuleBasedPredictor:
    """
    Predict match outcomes using rule-based heuristics
    """
    
    def __init__(self):
        """Initialize the rule-based predictor"""
        self.weights = RULE_WEIGHTS
        self.fixtures_cache = {}
        self.standings_cache = {}
        logger.info("RuleBasedPredictor initialized")
    
    def load_league_data(self, league_name: str, season: int) -> Dict[str, Any]:
        """
        Load league data from file
        
        Args:
            league_name: Name of the league
            season: Season year
        
        Returns:
            League data dictionary
        """
        filename = RAW_DATA_DIR / f"{league_name}_{season}.json"
        
        if not filename.exists():
            logger.warning(f"File not found: {filename}")
            return None
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Cache the data
        cache_key = f"{league_name}_{season}"
        self.fixtures_cache[cache_key] = data['fixtures']
        self.standings_cache[cache_key] = data.get('standings', [])
        
        logger.info(f"Loaded {league_name} {season}: {len(data['fixtures'])} fixtures")
        return data
    
    def get_team_form(self, team_id: int, fixtures: List[Dict], as_of_date: str, 
                      home_only: bool = False, away_only: bool = False) -> Dict[str, float]:
        """
        Calculate team's recent form
        
        Args:
            team_id: Team ID
            fixtures: List of all fixtures
            as_of_date: Calculate form as of this date (YYYY-MM-DD)
            home_only: Only consider home matches
            away_only: Only consider away matches
        
        Returns:
            Form metrics dictionary
        """
        # Get matches before the specified date
        team_matches = []
        
        for fixture in fixtures:
            # Skip if not finished
            if fixture['fixture']['status']['short'] != 'FT':
                continue
            
            fixture_date = fixture['fixture']['date'][:10]
            
            # Only include matches before as_of_date
            if fixture_date >= as_of_date:
                continue
            
            home_id = fixture['teams']['home']['id']
            away_id = fixture['teams']['away']['id']
            
            # Check if team is involved
            if team_id not in [home_id, away_id]:
                continue
            
            # Apply home/away filter
            is_home = (team_id == home_id)
            if home_only and not is_home:
                continue
            if away_only and is_home:
                continue
            
            team_matches.append({
                'date': fixture_date,
                'home_team_id': home_id,
                'away_team_id': away_id,
                'home_goals': fixture['goals']['home'],
                'away_goals': fixture['goals']['away'],
            })
        
        # Sort by date and take most recent
        team_matches.sort(key=lambda x: x['date'], reverse=True)
        recent_matches = team_matches[:FORM_WINDOW]
        
        return calculate_form(recent_matches, team_id)
    
    def get_team_position(self, team_id: int, standings: List[Dict]) -> int:
        """
        Get team's league position
        
        Args:
            team_id: Team ID
            standings: League standings
        
        Returns:
            League position (1-based)
        """
        for standing in standings:
            if standing['team']['id'] == team_id:
                return standing['rank']
        return 999  # Unknown position
    
    def get_head_to_head_record(self, home_id: int, away_id: int, 
                                 fixtures: List[Dict], as_of_date: str) -> Dict[str, int]:
        """
        Get head-to-head record between two teams
        
        Args:
            home_id: Home team ID
            away_id: Away team ID
            fixtures: List of all fixtures
            as_of_date: Calculate as of this date
        
        Returns:
            Dictionary with h2h statistics
        """
        h2h_matches = []
        
        for fixture in fixtures:
            if fixture['fixture']['status']['short'] != 'FT':
                continue
            
            fixture_date = fixture['fixture']['date'][:10]
            if fixture_date >= as_of_date:
                continue
            
            fhome_id = fixture['teams']['home']['id']
            faway_id = fixture['teams']['away']['id']
            
            # Check if these two teams played
            if not ((fhome_id == home_id and faway_id == away_id) or 
                    (fhome_id == away_id and faway_id == home_id)):
                continue
            
            h2h_matches.append({
                'date': fixture_date,
                'home_id': fhome_id,
                'away_id': faway_id,
                'home_goals': fixture['goals']['home'],
                'away_goals': fixture['goals']['away'],
                'result': get_match_result(fixture['goals']['home'], fixture['goals']['away']),
            })
        
        # Sort and take recent matches
        h2h_matches.sort(key=lambda x: x['date'], reverse=True)
        h2h_matches = h2h_matches[:RECENT_H2H_MATCHES]
        
        # Count results from home team's perspective
        home_wins = 0
        draws = 0
        away_wins = 0
        
        for match in h2h_matches:
            if match['home_id'] == home_id:
                # Current home team was home in this match
                if match['result'] == 'H':
                    home_wins += 1
                elif match['result'] == 'D':
                    draws += 1
                else:
                    away_wins += 1
            else:
                # Current home team was away in this match
                if match['result'] == 'A':
                    home_wins += 1
                elif match['result'] == 'D':
                    draws += 1
                else:
                    away_wins += 1
        
        return {
            'total': len(h2h_matches),
            'home_wins': home_wins,
            'draws': draws,
            'away_wins': away_wins,
        }
    
    def calculate_prediction_scores(self, home_id: int, away_id: int, 
                                     league_name: str, season: int,
                                     match_date: str) -> Dict[str, float]:
        """
        Calculate prediction scores for a match
        
        Args:
            home_id: Home team ID
            away_id: Away team ID
            league_name: League name
            season: Season year
            match_date: Date of the match (YYYY-MM-DD)
        
        Returns:
            Dictionary with scores for home, draw, away
        """
        cache_key = f"{league_name}_{season}"
        
        if cache_key not in self.fixtures_cache:
            logger.error(f"Data not loaded for {league_name} {season}")
            return None
        
        fixtures = self.fixtures_cache[cache_key]
        standings = self.standings_cache.get(cache_key, [])
        
        # Get team forms
        home_form = self.get_team_form(home_id, fixtures, match_date, home_only=True)
        away_form = self.get_team_form(away_id, fixtures, match_date, away_only=True)
        home_overall_form = self.get_team_form(home_id, fixtures, match_date)
        away_overall_form = self.get_team_form(away_id, fixtures, match_date)
        
        # Check if enough matches played
        if (len([f for f in fixtures if home_id in [f['teams']['home']['id'], f['teams']['away']['id']] 
                 and f['fixture']['status']['short'] == 'FT' and f['fixture']['date'][:10] < match_date]) < MIN_MATCHES_FOR_PREDICTION or
            len([f for f in fixtures if away_id in [f['teams']['home']['id'], f['teams']['away']['id']] 
                 and f['fixture']['status']['short'] == 'FT' and f['fixture']['date'][:10] < match_date]) < MIN_MATCHES_FOR_PREDICTION):
            logger.warning(f"Not enough matches for prediction")
            return None
        
        # Get league positions
        home_position = self.get_team_position(home_id, standings)
        away_position = self.get_team_position(away_id, standings)
        
        # Get head-to-head
        h2h = self.get_head_to_head_record(home_id, away_id, fixtures, match_date)
        
        # Calculate individual factor scores (0-1 scale, higher favors home)
        
        # 1. Form score
        home_ppg = home_form.get('points_per_game', 0)
        away_ppg = away_form.get('points_per_game', 0)
        form_score = 0.5  # Neutral
        if home_ppg + away_ppg > 0:
            form_score = home_ppg / (home_ppg + away_ppg)
        
        # 2. League position score (lower position = better)
        position_score = 0.5
        if home_position < 900 and away_position < 900:  # Valid positions
            # Invert so better position gets higher score
            total = (1/home_position) + (1/away_position)
            if total > 0:
                position_score = (1/home_position) / total
        
        # 3. Home advantage (baseline advantage for home team)
        home_advantage_score = 0.60  # Home teams win ~40-45% historically
        
        # 4. Head-to-head score
        h2h_score = 0.5
        if h2h['total'] > 0:
            h2h_score = (h2h['home_wins'] + 0.5 * h2h['draws']) / h2h['total']
        
        # 5. Goal difference score
        home_gd = home_overall_form.get('goal_difference', 0)
        away_gd = away_overall_form.get('goal_difference', 0)
        gd_score = 0.5
        gd_diff = home_gd - away_gd
        if abs(gd_diff) > 0:
            # Scale to 0-1, with extreme differences capped
            gd_score = 0.5 + (gd_diff / (abs(gd_diff) + 10))
        
        # Weighted combination
        home_score = (
            self.weights['form'] * form_score +
            self.weights['league_position'] * position_score +
            self.weights['home_advantage'] * home_advantage_score +
            self.weights['head_to_head'] * h2h_score +
            self.weights['goal_difference'] * gd_score
        )
        
        # Away score is inverse (simplified model)
        away_score = 1 - home_score
        
        # Draw score based on how close home and away scores are
        score_diff = abs(home_score - away_score)
        draw_score = 0.3 * (1 - score_diff)  # Max 0.3 for perfect balance
        
        # Normalize so scores sum to ~1
        total = home_score + draw_score + away_score
        
        return {
            'home_score': home_score / total,
            'draw_score': draw_score / total,
            'away_score': away_score / total,
            'confidence': max(home_score, draw_score, away_score) / total,
            'details': {
                'form_score': form_score,
                'position_score': position_score,
                'h2h_score': h2h_score,
                'gd_score': gd_score,
                'home_form_ppg': home_ppg,
                'away_form_ppg': away_ppg,
                'home_position': home_position,
                'away_position': away_position,
            }
        }
    
    def predict_match(self, home_id: int, away_id: int, home_name: str, away_name: str,
                     league_name: str, season: int, match_date: str) -> Dict[str, Any]:
        """
        Predict the outcome of a match
        
        Args:
            home_id: Home team ID
            away_id: Away team ID
            home_name: Home team name
            away_name: Away team name
            league_name: League name
            season: Season year
            match_date: Match date (YYYY-MM-DD)
        
        Returns:
            Prediction dictionary
        """
        scores = self.calculate_prediction_scores(home_id, away_id, league_name, season, match_date)
        
        if scores is None:
            return None
        
        # Determine prediction
        if scores['home_score'] >= HOME_WIN_THRESHOLD:
            prediction = 'H'
            prediction_text = 'Home Win'
        elif scores['away_score'] >= AWAY_WIN_THRESHOLD:
            prediction = 'A'
            prediction_text = 'Away Win'
        elif scores['home_score'] >= DRAW_LOWER_THRESHOLD and scores['home_score'] <= DRAW_UPPER_THRESHOLD:
            prediction = 'D'
            prediction_text = 'Draw'
        else:
            # Default to highest score
            if scores['home_score'] > scores['away_score']:
                prediction = 'H'
                prediction_text = 'Home Win'
            else:
                prediction = 'A'
                prediction_text = 'Away Win'
        
        return {
            'home_team': home_name,
            'away_team': away_name,
            'prediction': prediction,
            'prediction_text': prediction_text,
            'confidence': scores['confidence'],
            'home_score': scores['home_score'],
            'draw_score': scores['draw_score'],
            'away_score': scores['away_score'],
            'details': scores['details'],
        }


def main():
    """
    Example usage: predict upcoming matches
    """
    predictor = RuleBasedPredictor()
    
    # Load data
    predictor.load_league_data('premier_league', 2024)
    
    # Example: Predict a specific match
    # You would get these IDs and names from the API
    print("\n" + "="*70)
    print("RULE-BASED MATCH PREDICTION")
    print("="*70 + "\n")
    
    # This is just an example - you'd need real team IDs and upcoming match dates
    print("Load real match data and call predictor.predict_match() with actual team IDs")
    print("\nPredictor is ready to use!")
    
    logger.info("Rule-based predictor demonstration complete")


if __name__ == "__main__":
    main()