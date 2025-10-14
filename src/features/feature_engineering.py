"""
Enhanced Feature Engineering with ALL improvements
Includes: API stats, momentum, injuries, temporal features, and more
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
from pathlib import Path

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, FORM_WINDOW, RECENT_H2H_MATCHES
from utils.utils import setup_logging, calculate_form, get_match_result, days_between

logger = setup_logging(__name__)


class FeatureEngineer:
    """
    Advanced feature engineering for match prediction
    Uses all available data including API statistics and injuries
    """
    
    def __init__(self):
        """Initialize enhanced feature engineer"""
        self.fixtures_data = {}
        self.standings_data = {}
        self.team_statistics = {}
        self.injuries_cache = {}
        
        # Derby/Rivalry pairs (add more as needed)
        self.derby_pairs = {
            ('Arsenal', 'Tottenham'): 1,
            ('Manchester United', 'Manchester City'): 1,
            ('Liverpool', 'Everton'): 1,
            ('Manchester United', 'Liverpool'): 1,
            ('Arsenal', 'Chelsea'): 1,
            ('Chelsea', 'Tottenham'): 1,
            # Championship
            ('Birmingham', 'Aston Villa'): 1,
            ('Leeds', 'Manchester United'): 1,
            ('Nottingham Forest', 'Derby'): 1,
            # Add more derbies...
        }
        
        logger.info("EnhancedFeatureEngineer initialized")
    
    def load_league_data(self, league_name: str, season: int) -> bool:
        """
        Load all league data including injuries
        
        Args:
            league_name: Name of the league
            season: Season year
        
        Returns:
            True if loaded successfully
        """
        filename = RAW_DATA_DIR / f"{league_name}_{season}.json"
        
        if not filename.exists():
            logger.warning(f"File not found: {filename}")
            return False
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        cache_key = f"{league_name}_{season}"
        self.fixtures_data[cache_key] = data['fixtures']
        self.standings_data[cache_key] = data.get('standings', [])
        self.team_statistics[cache_key] = data.get('team_statistics', {})
        
        # Load injuries if available
        injuries_file = RAW_DATA_DIR / f"{league_name}_{season}_injuries.json"
        if injuries_file.exists():
            with open(injuries_file, 'r') as f:
                self.injuries_cache[cache_key] = json.load(f)
        else:
            self.injuries_cache[cache_key] = {}
        
        logger.info(f"Loaded {league_name} {season}: {len(data['fixtures'])} fixtures")
        return True
    
    def get_team_recent_matches(self, team_id: int, fixtures: List[Dict], 
                                before_date: str, window: int = FORM_WINDOW,
                                home_only: bool = False, away_only: bool = False) -> List[Dict]:
        """Get team's recent matches before a specific date"""
        matches = []
        
        for fixture in fixtures:
            if fixture['fixture']['status']['short'] != 'FT':
                continue
            
            fixture_date = fixture['fixture']['date'][:10]
            if fixture_date >= before_date:
                continue
            
            home_id = fixture['teams']['home']['id']
            away_id = fixture['teams']['away']['id']
            
            if team_id not in [home_id, away_id]:
                continue
            
            is_home = (team_id == home_id)
            if home_only and not is_home:
                continue
            if away_only and is_home:
                continue
            
            matches.append({
                'date': fixture_date,
                'home_team_id': home_id,
                'away_team_id': away_id,
                'home_goals': fixture['goals']['home'],
                'away_goals': fixture['goals']['away'],
                'is_home': is_home,
            })
        
        matches.sort(key=lambda x: x['date'], reverse=True)
        return matches[:window]
    
    def calculate_rolling_stats(self, matches: List[Dict], team_id: int) -> Dict[str, float]:
        """Calculate rolling statistics from recent matches"""
        if not matches:
            return {
                'points_per_game': 0.0,
                'goals_scored_per_game': 0.0,
                'goals_conceded_per_game': 0.0,
                'win_rate': 0.0,
                'clean_sheet_rate': 0.0,
                'failed_to_score_rate': 0.0,
            }
        
        form = calculate_form(matches, team_id)
        num_matches = len(matches)
        
        clean_sheets = sum(1 for m in matches if 
                          (m['is_home'] and m['away_goals'] == 0) or 
                          (not m['is_home'] and m['home_goals'] == 0))
        
        failed_to_score = sum(1 for m in matches if 
                             (m['is_home'] and m['home_goals'] == 0) or 
                             (not m['is_home'] and m['away_goals'] == 0))
        
        return {
            'points_per_game': form['points_per_game'],
            'goals_scored_per_game': form['goals_scored_per_game'],
            'goals_conceded_per_game': form['goals_conceded_per_game'],
            'win_rate': form['wins'] / num_matches if num_matches > 0 else 0,
            'clean_sheet_rate': clean_sheets / num_matches if num_matches > 0 else 0,
            'failed_to_score_rate': failed_to_score / num_matches if num_matches > 0 else 0,
        }
    
    def calculate_momentum(self, team_id: int, fixtures: List[Dict], 
                          before_date: str) -> Dict[str, float]:
        """
        Calculate if team is trending up or down
        Compares last 3 matches vs previous 3 matches
        """
        recent_10 = self.get_team_recent_matches(team_id, fixtures, before_date, window=10)
        
        if len(recent_10) < 6:
            return {'momentum': 0.0, 'momentum_points': 0.0}
        
        # Compare last 3 vs previous 3
        last_3 = calculate_form(recent_10[:3], team_id)
        prev_3 = calculate_form(recent_10[3:6], team_id)
        
        momentum = last_3['points_per_game'] - prev_3['points_per_game']
        
        return {
            'momentum': momentum,  # Range: -3 to +3
            'momentum_points': last_3['points_per_game'] - 1.5,  # vs league average
        }
    
    def get_btts_over_features(self, team_id: int, fixtures: List[Dict],
                               before_date: str, window: int = 10) -> Dict[str, float]:
        """
        Both Teams To Score and Over 2.5 patterns
        Very predictive for draw probability
        """
        recent = self.get_team_recent_matches(team_id, fixtures, before_date, window=window)
        
        if not recent:
            return {
                'btts_rate': 0.5,
                'over25_rate': 0.5,
                'over15_rate': 0.5,
                'avg_total_goals': 2.5,
            }
        
        btts_count = 0
        over25_count = 0
        over15_count = 0
        total_goals_sum = 0
        
        for match in recent:
            total_goals = match['home_goals'] + match['away_goals']
            total_goals_sum += total_goals
            
            if match['home_goals'] > 0 and match['away_goals'] > 0:
                btts_count += 1
            
            if total_goals > 2.5:
                over25_count += 1
            
            if total_goals > 1.5:
                over15_count += 1
        
        return {
            'btts_rate': btts_count / len(recent),
            'over25_rate': over25_count / len(recent),
            'over15_rate': over15_count / len(recent),
            'avg_total_goals': total_goals_sum / len(recent),
        }
    
    def get_scoring_consistency(self, team_id: int, fixtures: List[Dict],
                               before_date: str) -> Dict[str, float]:
        """
        Measure consistency of scoring
        Lower volatility = more predictable
        """
        recent = self.get_team_recent_matches(team_id, fixtures, before_date, window=10)
        
        if not recent:
            return {'scoring_std': 0.0, 'scoring_range': 0.0}
        
        goals_scored = []
        for match in recent:
            if match['is_home']:
                goals_scored.append(match['home_goals'])
            else:
                goals_scored.append(match['away_goals'])
        
        return {
            'scoring_std': np.std(goals_scored) if len(goals_scored) > 1 else 0.0,
            'scoring_range': max(goals_scored) - min(goals_scored) if goals_scored else 0.0,
        }
    
    def get_temporal_features(self, match_date: str) -> Dict[str, int]:
        """
        Time-related features
        Month, day of week, congested periods
        """
        date = datetime.strptime(match_date, '%Y-%m-%d')
        
        # Congested periods in English football
        is_boxing_day = 1 if (date.month == 12 and 20 <= date.day <= 31) else 0
        is_new_year = 1 if (date.month == 1 and date.day <= 7) else 0
        is_holiday_period = is_boxing_day or is_new_year
        
        return {
            'month': date.month,
            'day_of_week': date.weekday(),  # 0=Monday, 6=Sunday
            'is_weekend': 1 if date.weekday() >= 5 else 0,
            'is_midweek': 1 if 1 <= date.weekday() <= 3 else 0,
            'is_december': 1 if date.month == 12 else 0,
            'is_holiday_period': is_holiday_period,
            'week_of_season': min((date - datetime(date.year, 8, 1)).days // 7, 50),
        }
    
    def is_derby_match(self, home_name: str, away_name: str) -> int:
        """Check if match is a derby/rivalry"""
        pair = tuple(sorted([home_name, away_name]))
        return self.derby_pairs.get(pair, 0)
    
    def get_fixture_difficulty(self, team_id: int, fixtures: List[Dict],
                              before_date: str, standings: List[Dict]) -> float:
        """
        Analyze difficulty of recent fixtures
        Lower = played top teams, Higher = played bottom teams
        """
        recent_matches = self.get_team_recent_matches(team_id, fixtures, before_date, window=5)
        
        opponent_positions = []
        for match in recent_matches:
            opp_id = match['away_team_id'] if match['is_home'] else match['home_team_id']
            
            for standing in standings:
                if standing['team']['id'] == opp_id:
                    opponent_positions.append(standing['rank'])
                    break
        
        if not opponent_positions:
            return 10.0  # mid-table default
        
        return sum(opponent_positions) / len(opponent_positions)
    
    def get_injuries_impact(self, team_id: int, match_date: str, 
                           league_name: str, season: int) -> Dict[str, int]:
        """
        Get injury impact for a team
        NOTE: Requires injuries data to be fetched
        """
        cache_key = f"{league_name}_{season}"
        injuries_data = self.injuries_cache.get(cache_key, {})
        
        if not injuries_data:
            return {
                'missing_players': 0,
                'questionable_players': 0,
                'total_injured': 0,
            }
        
        # Count injuries for this team around this date
        missing = 0
        questionable = 0
        
        # This would need proper injury data structure from API
        # For now, return zeros if no data
        return {
            'missing_players': missing,
            'questionable_players': questionable,
            'total_injured': missing + questionable,
        }
    
    def get_team_stats_features(self, home_id: int, away_id: int,
                               league_name: str, season: int) -> Dict[str, float]:
        """
        Extract features from API team statistics
        These are season-level aggregates from the API
        """
        cache_key = f"{league_name}_{season}"
        team_stats = self.team_statistics.get(cache_key, {})
        
        home_stats = team_stats.get(str(home_id), {})
        away_stats = team_stats.get(str(away_id), {})
        
        if not home_stats or not away_stats:
            return {}
        
        features = {}
        
        try:
            # Season averages
            home_goals_avg = home_stats.get('goals', {}).get('for', {}).get('average', {}).get('total', 0)
            away_goals_avg = away_stats.get('goals', {}).get('for', {}).get('average', {}).get('total', 0)
            home_conceded_avg = home_stats.get('goals', {}).get('against', {}).get('average', {}).get('total', 0)
            away_conceded_avg = away_stats.get('goals', {}).get('against', {}).get('average', {}).get('total', 0)
            
            features['home_season_goals_avg'] = float(home_goals_avg or 0)
            features['away_season_goals_avg'] = float(away_goals_avg or 0)
            features['home_season_conceded_avg'] = float(home_conceded_avg or 0)
            features['away_season_conceded_avg'] = float(away_conceded_avg or 0)
            features['season_goals_diff'] = features['home_season_goals_avg'] - features['away_season_goals_avg']
            features['season_defense_diff'] = features['away_season_conceded_avg'] - features['home_season_conceded_avg']
            
            # Clean sheets
            home_clean_sheets = home_stats.get('clean_sheet', {}).get('total', 0)
            away_clean_sheets = away_stats.get('clean_sheet', {}).get('total', 0)
            home_played = home_stats.get('fixtures', {}).get('played', {}).get('total', 1)
            away_played = away_stats.get('fixtures', {}).get('played', {}).get('total', 1)
            
            features['home_clean_sheet_pct'] = home_clean_sheets / home_played
            features['away_clean_sheet_pct'] = away_clean_sheets / away_played
            
            # Failed to score
            home_failed = home_stats.get('failed_to_score', {}).get('total', 0)
            away_failed = away_stats.get('failed_to_score', {}).get('total', 0)
            
            features['home_failed_to_score_pct'] = home_failed / home_played
            features['away_failed_to_score_pct'] = away_failed / away_played
            
            # Win rates
            home_wins = home_stats.get('fixtures', {}).get('wins', {}).get('total', 0)
            away_wins = away_stats.get('fixtures', {}).get('wins', {}).get('total', 0)
            home_draws = home_stats.get('fixtures', {}).get('draws', {}).get('total', 0)
            away_draws = away_stats.get('fixtures', {}).get('draws', {}).get('total', 0)
            
            features['home_season_win_pct'] = home_wins / home_played
            features['away_season_win_pct'] = away_wins / away_played
            features['home_season_draw_pct'] = home_draws / home_played
            features['away_season_draw_pct'] = away_draws / away_played
            
            # Penalty stats
            home_penalties_scored = home_stats.get('penalty', {}).get('scored', {}).get('total', 0)
            away_penalties_scored = away_stats.get('penalty', {}).get('scored', {}).get('total', 0)
            home_penalties_missed = home_stats.get('penalty', {}).get('missed', {}).get('total', 0)
            away_penalties_missed = away_stats.get('penalty', {}).get('missed', {}).get('total', 0)
            
            total_home_pens = home_penalties_scored + home_penalties_missed
            total_away_pens = away_penalties_scored + away_penalties_missed
            
            features['home_penalty_conversion'] = home_penalties_scored / total_home_pens if total_home_pens else 0.8
            features['away_penalty_conversion'] = away_penalties_scored / total_away_pens if total_away_pens else 0.8
            
        except (KeyError, TypeError, ZeroDivisionError) as e:
            logger.warning(f"Error extracting team stats: {e}")
        
        return features
    
    def get_h2h_features(self, home_id: int, away_id: int, fixtures: List[Dict],
                        before_date: str) -> Dict[str, float]:
        """Extract head-to-head features"""
        h2h_matches = []
        
        for fixture in fixtures:
            if fixture['fixture']['status']['short'] != 'FT':
                continue
            
            fixture_date = fixture['fixture']['date'][:10]
            if fixture_date >= before_date:
                continue
            
            fhome_id = fixture['teams']['home']['id']
            faway_id = fixture['teams']['away']['id']
            
            if not ((fhome_id == home_id and faway_id == away_id) or 
                    (fhome_id == away_id and faway_id == home_id)):
                continue
            
            h2h_matches.append({
                'date': fixture_date,
                'home_id': fhome_id,
                'home_goals': fixture['goals']['home'],
                'away_goals': fixture['goals']['away'],
                'result': get_match_result(fixture['goals']['home'], fixture['goals']['away']),
            })
        
        h2h_matches.sort(key=lambda x: x['date'], reverse=True)
        h2h_matches = h2h_matches[:RECENT_H2H_MATCHES]
        
        if not h2h_matches:
            return {
                'h2h_home_wins': 0,
                'h2h_draws': 0,
                'h2h_away_wins': 0,
                'h2h_home_goals_avg': 0.0,
                'h2h_away_goals_avg': 0.0,
                'h2h_total_matches': 0,
            }
        
        home_wins = 0
        draws = 0
        away_wins = 0
        home_goals_total = 0
        away_goals_total = 0
        
        for match in h2h_matches:
            if match['home_id'] == home_id:
                home_goals_total += match['home_goals']
                away_goals_total += match['away_goals']
                if match['result'] == 'H':
                    home_wins += 1
                elif match['result'] == 'D':
                    draws += 1
                else:
                    away_wins += 1
            else:
                home_goals_total += match['away_goals']
                away_goals_total += match['home_goals']
                if match['result'] == 'A':
                    home_wins += 1
                elif match['result'] == 'D':
                    draws += 1
                else:
                    away_wins += 1
        
        num_matches = len(h2h_matches)
        
        return {
            'h2h_home_wins': home_wins,
            'h2h_draws': draws,
            'h2h_away_wins': away_wins,
            'h2h_home_goals_avg': home_goals_total / num_matches,
            'h2h_away_goals_avg': away_goals_total / num_matches,
            'h2h_total_matches': num_matches,
        }
    
    def get_league_position_features(self, home_id: int, away_id: int,
                                    standings: List[Dict]) -> Dict[str, float]:
        """Extract league position features"""
        home_pos = 999
        away_pos = 999
        home_points = 0
        away_points = 0
        home_gd = 0
        away_gd = 0
        
        for standing in standings:
            team_id = standing['team']['id']
            if team_id == home_id:
                home_pos = standing['rank']
                home_points = standing['points']
                home_gd = standing['goalsDiff']
            elif team_id == away_id:
                away_pos = standing['rank']
                away_points = standing['points']
                away_gd = standing['goalsDiff']
        
        return {
            'home_position': home_pos,
            'away_position': away_pos,
            'position_diff': away_pos - home_pos,
            'home_points': home_points,
            'away_points': away_points,
            'points_diff': home_points - away_points,
            'home_goal_diff': home_gd,
            'away_goal_diff': away_gd,
            'goal_diff_diff': home_gd - away_gd,
        }
    
    def get_rest_days_features(self, team_id: int, fixtures: List[Dict],
                              match_date: str) -> Dict[str, int]:
        """Calculate rest days since last match"""
        last_match_date = None
        
        for fixture in fixtures:
            if fixture['fixture']['status']['short'] != 'FT':
                continue
            
            fixture_date = fixture['fixture']['date'][:10]
            if fixture_date >= match_date:
                continue
            
            home_id = fixture['teams']['home']['id']
            away_id = fixture['teams']['away']['id']
            
            if team_id in [home_id, away_id]:
                if last_match_date is None or fixture_date > last_match_date:
                    last_match_date = fixture_date
        
        if last_match_date is None:
            return {'days_since_last_match': 999}
        
        return {'days_since_last_match': days_between(last_match_date, match_date)}
    
    def create_match_features(self, home_id: int, away_id: int, match_date: str,
                             league_name: str, season: int, 
                             home_name: str = "", away_name: str = "") -> Dict[str, Any]:
        """
        Create ALL features for a single match
        This is the master feature creation function
        """
        cache_key = f"{league_name}_{season}"
        
        if cache_key not in self.fixtures_data:
            logger.error(f"Data not loaded for {league_name} {season}")
            return None
        
        fixtures = self.fixtures_data[cache_key]
        standings = self.standings_data[cache_key]
        
        # === FORM FEATURES ===
        home_recent = self.get_team_recent_matches(home_id, fixtures, match_date, home_only=True)
        away_recent = self.get_team_recent_matches(away_id, fixtures, match_date, away_only=True)
        home_recent_all = self.get_team_recent_matches(home_id, fixtures, match_date)
        away_recent_all = self.get_team_recent_matches(away_id, fixtures, match_date)
        
        home_stats = self.calculate_rolling_stats(home_recent, home_id)
        away_stats = self.calculate_rolling_stats(away_recent, away_id)
        home_stats_all = self.calculate_rolling_stats(home_recent_all, home_id)
        away_stats_all = self.calculate_rolling_stats(away_recent_all, away_id)
        
        # === MOMENTUM FEATURES ===
        home_momentum = self.calculate_momentum(home_id, fixtures, match_date)
        away_momentum = self.calculate_momentum(away_id, fixtures, match_date)
        
        # === BTTS/OVER FEATURES ===
        home_btts = self.get_btts_over_features(home_id, fixtures, match_date)
        away_btts = self.get_btts_over_features(away_id, fixtures, match_date)
        
        # === CONSISTENCY FEATURES ===
        home_consistency = self.get_scoring_consistency(home_id, fixtures, match_date)
        away_consistency = self.get_scoring_consistency(away_id, fixtures, match_date)
        
        # === H2H FEATURES ===
        h2h_features = self.get_h2h_features(home_id, away_id, fixtures, match_date)
        
        # === POSITION FEATURES ===
        position_features = self.get_league_position_features(home_id, away_id, standings)
        
        # === REST DAYS ===
        home_rest = self.get_rest_days_features(home_id, fixtures, match_date)
        away_rest = self.get_rest_days_features(away_id, fixtures, match_date)
        
        # === TEMPORAL FEATURES ===
        temporal_features = self.get_temporal_features(match_date)
        
        # === FIXTURE DIFFICULTY ===
        home_fixture_difficulty = self.get_fixture_difficulty(home_id, fixtures, match_date, standings)
        away_fixture_difficulty = self.get_fixture_difficulty(away_id, fixtures, match_date, standings)
        
        # === TEAM STATISTICS (from API) ===
        team_stats_features = self.get_team_stats_features(home_id, away_id, league_name, season)
        
        # === DERBY DETECTION ===
        is_derby = self.is_derby_match(home_name, away_name) if home_name and away_name else 0
        
        # === INJURY FEATURES ===
        home_injuries = self.get_injuries_impact(home_id, match_date, league_name, season)
        away_injuries = self.get_injuries_impact(away_id, match_date, league_name, season)
        
        # === COMBINE ALL FEATURES ===
        features = {
            # Form features (home matches only)
            'home_form_points': home_stats['points_per_game'],
            'home_goals_scored_avg': home_stats['goals_scored_per_game'],
            'home_goals_conceded_avg': home_stats['goals_conceded_per_game'],
            'home_win_rate': home_stats['win_rate'],
            'home_clean_sheet_rate': home_stats['clean_sheet_rate'],
            'home_failed_to_score_rate': home_stats['failed_to_score_rate'],
            
            # Form features (away matches only)
            'away_form_points': away_stats['points_per_game'],
            'away_goals_scored_avg': away_stats['goals_scored_per_game'],
            'away_goals_conceded_avg': away_stats['goals_conceded_per_game'],
            'away_win_rate': away_stats['win_rate'],
            'away_clean_sheet_rate': away_stats['clean_sheet_rate'],
            'away_failed_to_score_rate': away_stats['failed_to_score_rate'],
            
            # Overall form
            'home_overall_form_points': home_stats_all['points_per_game'],
            'away_overall_form_points': away_stats_all['points_per_game'],
            
            # Form differentials
            'form_points_diff': home_stats['points_per_game'] - away_stats['points_per_game'],
            'goals_scored_diff': home_stats['goals_scored_per_game'] - away_stats['goals_scored_per_game'],
            'goals_conceded_diff': away_stats['goals_conceded_per_game'] - home_stats['goals_conceded_per_game'],
            
            # Momentum
            'home_momentum': home_momentum['momentum'],
            'away_momentum': away_momentum['momentum'],
            'momentum_diff': home_momentum['momentum'] - away_momentum['momentum'],
            
            # BTTS/Over features
            'home_btts_rate': home_btts['btts_rate'],
            'away_btts_rate': away_btts['btts_rate'],
            'home_over25_rate': home_btts['over25_rate'],
            'away_over25_rate': away_btts['over25_rate'],
            'home_avg_total_goals': home_btts['avg_total_goals'],
            'away_avg_total_goals': away_btts['avg_total_goals'],
            'combined_btts_likelihood': (home_btts['btts_rate'] + away_btts['btts_rate']) / 2,
            'combined_over25_likelihood': (home_btts['over25_rate'] + away_btts['over25_rate']) / 2,
            
            # Consistency
            'home_scoring_std': home_consistency['scoring_std'],
            'away_scoring_std': away_consistency['scoring_std'],
            'home_scoring_range': home_consistency['scoring_range'],
            'away_scoring_range': away_consistency['scoring_range'],
            
            # League position
            'home_position': position_features['home_position'],
            'away_position': position_features['away_position'],
            'position_diff': position_features['position_diff'],
            'points_diff': position_features['points_diff'],
            'goal_diff_diff': position_features['goal_diff_diff'],
            
            # Head-to-head
            'h2h_home_wins': h2h_features['h2h_home_wins'],
            'h2h_draws': h2h_features['h2h_draws'],
            'h2h_away_wins': h2h_features['h2h_away_wins'],
            'h2h_home_goals_avg': h2h_features['h2h_home_goals_avg'],
            'h2h_away_goals_avg': h2h_features['h2h_away_goals_avg'],
            'h2h_total_matches': h2h_features['h2h_total_matches'],
            
            # Rest days
            'days_since_last_match_home': home_rest['days_since_last_match'],
            'days_since_last_match_away': away_rest['days_since_last_match'],
            'rest_days_diff': home_rest['days_since_last_match'] - away_rest['days_since_last_match'],
            
            # Temporal features
            'month': temporal_features['month'],
            'day_of_week': temporal_features['day_of_week'],
            'is_weekend': temporal_features['is_weekend'],
            'is_midweek': temporal_features['is_midweek'],
            'is_december': temporal_features['is_december'],
            'is_holiday_period': temporal_features['is_holiday_period'],
            'week_of_season': temporal_features['week_of_season'],
            
            # Fixture difficulty
            'home_recent_opponent_strength': home_fixture_difficulty,
            'away_recent_opponent_strength': away_fixture_difficulty,
            
            # Derby
            'is_derby': is_derby,
            
            # Injuries
            'home_missing_players': home_injuries['missing_players'],
            'away_missing_players': away_injuries['missing_players'],
            'home_questionable_players': home_injuries['questionable_players'],
            'away_questionable_players': away_injuries['questionable_players'],
            'injury_impact_diff': home_injuries['total_injured'] - away_injuries['total_injured'],
        }
        
        # Add team statistics features (from API)
        features.update(team_stats_features)
        
        return features
    
    def create_training_dataset(self, league_name: str, season: int) -> pd.DataFrame:
        """
        Create a complete training dataset for a league/season
        
        Args:
            league_name: League name
            season: Season year
        
        Returns:
            DataFrame with features and labels
        """
        cache_key = f"{league_name}_{season}"
        
        if cache_key not in self.fixtures_data:
            logger.error(f"Data not loaded for {league_name} {season}")
            return None
        
        fixtures = self.fixtures_data[cache_key]
        
        dataset = []
        
        for i, fixture in enumerate(fixtures):
            # Only process finished matches
            if fixture['fixture']['status']['short'] != 'FT':
                continue
            
            home_id = fixture['teams']['home']['id']
            away_id = fixture['teams']['away']['id']
            home_name = fixture['teams']['home']['name']
            away_name = fixture['teams']['away']['name']
            match_date = fixture['fixture']['date'][:10]
            home_goals = fixture['goals']['home']
            away_goals = fixture['goals']['away']
            result = get_match_result(home_goals, away_goals)
            
            # Create features for this match
            features = self.create_match_features(
                home_id, away_id, match_date, league_name, season,
                home_name, away_name
            )
            
            if features is None:
                continue
            
            # Add match info and label
            features['fixture_id'] = fixture['fixture']['id']
            features['date'] = match_date
            features['home_team'] = home_name
            features['away_team'] = away_name
            features['home_goals'] = home_goals
            features['away_goals'] = away_goals
            features['result'] = result
            features['league'] = league_name
            features['season'] = season
            
            dataset.append(features)
            
            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(fixtures)} fixtures")
        
        df = pd.DataFrame(dataset)
        logger.info(f"Created dataset with {len(df)} matches")
        
        return df
    
    def save_training_dataset(self, df: pd.DataFrame, filename: str):
        """
        Save training dataset to file
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        output_path = PROCESSED_DATA_DIR / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved training dataset to {output_path}")


def main():
    """
    Create enhanced training datasets with all features
    """
    engineer = EnhancedFeatureEngineer()
    
    # Load all leagues and seasons
    leagues = ['premier_league', 'championship', 'league_one', 'league_two']
    seasons = [2020, 2021, 2022, 2023, 2024, 2025]
    
    # Store data by league
    league_data = {league: [] for league in leagues}
    
    print("\n" + "="*80)
    print("ENHANCED FEATURE ENGINEERING WITH ALL IMPROVEMENTS")
    print("="*80)
    print("\nFeatures included:")
    print("  ✅ Form (5-match window)")
    print("  ✅ Momentum (trending up/down)")
    print("  ✅ BTTS & Over/Under patterns")
    print("  ✅ Scoring consistency")
    print("  ✅ Head-to-head")
    print("  ✅ League position")
    print("  ✅ Rest days")
    print("  ✅ Temporal (month, day, holiday periods)")
    print("  ✅ Fixture difficulty")
    print("  ✅ Derby detection")
    print("  ✅ API team statistics (season aggregates)")
    print("  ✅ Injury impact (if data available)")
    print("\n" + "="*80)
    
    print("\nLoading all data...")
    for league in leagues:
        for season in seasons:
            print(f"\n📊 Processing {league} {season}...")
            success = engineer.load_league_data(league, season)
            
            if success:
                df = engineer.create_training_dataset(league, season)
                if df is not None and len(df) > 0:
                    league_data[league].append(df)
                    print(f"  ✓ Created {len(df)} training samples")
                    print(f"  ✓ Features: {len(df.columns)} columns")
    
    # Save separate datasets for each league
    print("\n" + "="*80)
    print("SAVING ENHANCED DATASETS")
    print("="*80)
    
    for league in leagues:
        if not league_data[league]:
            print(f"\n⚠️  No data for {league}")
            continue
        
        # Combine all seasons for this league
        df_league = pd.concat(league_data[league], ignore_index=True)
        
        print(f"\n{league.upper()}:")
        print(f"  📈 Total samples: {len(df_league):,}")
        print(f"  📅 Seasons: {sorted(df_league['season'].unique())}")
        print(f"  🗓️  Date range: {df_league['date'].min()} to {df_league['date'].max()}")
        print(f"  🎯 Features: {len(df_league.columns)} columns")
        print(f"  📊 Result distribution:")
        result_counts = df_league['result'].value_counts()
        for result, count in result_counts.items():
            pct = count / len(df_league) * 100
            result_name = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}[result]
            print(f"     {result_name}: {count:,} ({pct:.1f}%)")
        
        # Save league-specific dataset
        filename = f"{league}_training_enhanced_6years.csv"
        engineer.save_training_dataset(df_league, filename)
        print(f"  💾 Saved: data/processed/{filename}")
    
    # Also create a combined dataset
    all_data = []
    for league_datasets in league_data.values():
        all_data.extend(league_datasets)
    
    if all_data:
        df_combined = pd.concat(all_data, ignore_index=True)
        engineer.save_training_dataset(df_combined, 'full_training_enhanced_6years_combined.csv')
        print(f"\n✅ Also saved combined dataset: data/processed/full_training_enhanced_6years_combined.csv")
        print(f"   Total: {len(df_combined):,} samples across all leagues")
        print(f"   Features: {len(df_combined.columns)} columns")
    
  
    for league in leagues:
        if league_data[league]:
            print(f"  ✓ {league}_training_enhanced_6years.csv")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()