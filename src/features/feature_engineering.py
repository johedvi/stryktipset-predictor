"""
Feature engineering for machine learning models
Transform raw match data into ML-ready features
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, FORM_WINDOW, RECENT_H2H_MATCHES
from utils import setup_logging, calculate_form, get_match_result, days_between

logger = setup_logging(__name__)


class FeatureEngineer:
    """
    Create features from raw football data for machine learning
    """
    
    def __init__(self):
        """Initialize feature engineer"""
        self.fixtures_data = {}
        self.standings_data = {}
        logger.info("FeatureEngineer initialized")
    
    def load_league_data(self, league_name: str, season: int) -> bool:
        """
        Load league data from file
        
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
        
        logger.info(f"Loaded {league_name} {season}: {len(data['fixtures'])} fixtures")
        return True
    
    def get_team_recent_matches(self, team_id: int, fixtures: List[Dict], 
                                before_date: str, window: int = FORM_WINDOW,
                                home_only: bool = False, away_only: bool = False) -> List[Dict]:
        """
        Get team's recent matches before a specific date
        
        Args:
            team_id: Team ID
            fixtures: All fixtures
            before_date: Get matches before this date
            window: Number of recent matches
            home_only: Only home matches
            away_only: Only away matches
        
        Returns:
            List of recent matches
        """
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
        """
        Calculate rolling statistics from recent matches
        
        Args:
            matches: List of recent matches
            team_id: Team ID
        
        Returns:
            Dictionary of rolling statistics
        """
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
    
    def get_h2h_features(self, home_id: int, away_id: int, fixtures: List[Dict], 
                        before_date: str) -> Dict[str, float]:
        """
        Extract head-to-head features
        
        Args:
            home_id: Home team ID
            away_id: Away team ID
            fixtures: All fixtures
            before_date: Get H2H before this date
        
        Returns:
            H2H features
        """
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
        """
        Extract league position features
        
        Args:
            home_id: Home team ID
            away_id: Away team ID
            standings: League standings
        
        Returns:
            Position features
        """
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
            'position_diff': away_pos - home_pos,  # Positive means home is higher
            'home_points': home_points,
            'away_points': away_points,
            'points_diff': home_points - away_points,
            'home_goal_diff': home_gd,
            'away_goal_diff': away_gd,
            'goal_diff_diff': home_gd - away_gd,
        }
    
    def get_rest_days_features(self, team_id: int, fixtures: List[Dict], 
                               match_date: str) -> Dict[str, int]:
        """
        Calculate rest days since last match
        
        Args:
            team_id: Team ID
            fixtures: All fixtures
            match_date: Current match date
        
        Returns:
            Rest days features
        """
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
                             league_name: str, season: int) -> Dict[str, Any]:
        """
        Create all features for a single match
        
        Args:
            home_id: Home team ID
            away_id: Away team ID
            match_date: Match date (YYYY-MM-DD)
            league_name: League name
            season: Season year
        
        Returns:
            Dictionary of features
        """
        cache_key = f"{league_name}_{season}"
        
        if cache_key not in self.fixtures_data:
            logger.error(f"Data not loaded for {league_name} {season}")
            return None
        
        fixtures = self.fixtures_data[cache_key]
        standings = self.standings_data[cache_key]
        
        # Get recent matches
        home_recent = self.get_team_recent_matches(home_id, fixtures, match_date, home_only=True)
        away_recent = self.get_team_recent_matches(away_id, fixtures, match_date, away_only=True)
        home_recent_all = self.get_team_recent_matches(home_id, fixtures, match_date)
        away_recent_all = self.get_team_recent_matches(away_id, fixtures, match_date)
        
        # Calculate rolling stats
        home_stats = self.calculate_rolling_stats(home_recent, home_id)
        away_stats = self.calculate_rolling_stats(away_recent, away_id)
        home_stats_all = self.calculate_rolling_stats(home_recent_all, home_id)
        away_stats_all = self.calculate_rolling_stats(away_recent_all, away_id)
        
        # Get H2H features
        h2h_features = self.get_h2h_features(home_id, away_id, fixtures, match_date)
        
        # Get position features
        position_features = self.get_league_position_features(home_id, away_id, standings)
        
        # Get rest days
        home_rest = self.get_rest_days_features(home_id, fixtures, match_date)
        away_rest = self.get_rest_days_features(away_id, fixtures, match_date)
        
        # Combine all features
        features = {
            # Home team form (home matches only)
            'home_form_points': home_stats['points_per_game'],
            'home_goals_scored_avg': home_stats['goals_scored_per_game'],
            'home_goals_conceded_avg': home_stats['goals_conceded_per_game'],
            'home_win_rate': home_stats['win_rate'],
            'home_clean_sheet_rate': home_stats['clean_sheet_rate'],
            
            # Away team form (away matches only)
            'away_form_points': away_stats['points_per_game'],
            'away_goals_scored_avg': away_stats['goals_scored_per_game'],
            'away_goals_conceded_avg': away_stats['goals_conceded_per_game'],
            'away_win_rate': away_stats['win_rate'],
            'away_clean_sheet_rate': away_stats['clean_sheet_rate'],
            
            # Overall form (all matches)
            'home_overall_form_points': home_stats_all['points_per_game'],
            'away_overall_form_points': away_stats_all['points_per_game'],
            
            # Form differentials
            'form_points_diff': home_stats['points_per_game'] - away_stats['points_per_game'],
            'goals_scored_diff': home_stats['goals_scored_per_game'] - away_stats['goals_scored_per_game'],
            'goals_conceded_diff': away_stats['goals_conceded_per_game'] - home_stats['goals_conceded_per_game'],
            
            # League position features
            'home_position': position_features['home_position'],
            'away_position': position_features['away_position'],
            'position_diff': position_features['position_diff'],
            'points_diff': position_features['points_diff'],
            'goal_diff_diff': position_features['goal_diff_diff'],
            
            # Head-to-head features
            'h2h_home_wins': h2h_features['h2h_home_wins'],
            'h2h_draws': h2h_features['h2h_draws'],
            'h2h_away_wins': h2h_features['h2h_away_wins'],
            'h2h_home_goals_avg': h2h_features['h2h_home_goals_avg'],
            'h2h_away_goals_avg': h2h_features['h2h_away_goals_avg'],
            
            # Rest days
            'days_since_last_match_home': home_rest['days_since_last_match'],
            'days_since_last_match_away': away_rest['days_since_last_match'],
            'rest_days_diff': home_rest['days_since_last_match'] - away_rest['days_since_last_match'],
        }
        
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
            features = self.create_match_features(home_id, away_id, match_date, league_name, season)
            
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
    Create training datasets from all available data
    """
    engineer = FeatureEngineer()
    
    # Load all leagues and seasons
    leagues = ['premier_league', 'championship', 'league_one']
    seasons = [2020, 2021, 2022, 2023, 2024]
    
    all_data = []
    
    print("Loading all data...")
    for league in leagues:
        for season in seasons:
            print(f"\nProcessing {league} {season}...")
            success = engineer.load_league_data(league, season)
            
            if success:
                df = engineer.create_training_dataset(league, season)
                if df is not None and len(df) > 0:
                    all_data.append(df)
                    print(f"  ✓ Created {len(df)} training samples")
    
    if not all_data:
        print("\n❌ No data loaded!")
        return
    
    # Combine all datasets
    print("\n" + "="*80)
    print("Combining all datasets...")
   
    df_combined = pd.concat(all_data, ignore_index=True)
    
    print(f"Total training samples: {len(df_combined)}")
    print(f"Leagues: {df_combined['league'].unique()}")
    print(f"Seasons: {sorted(df_combined['season'].unique())}")
    
    # Save
    engineer.save_training_dataset(df_combined, 'full_training_5years.csv')
    
    print("\n✓ Feature engineering complete!")
    print(f"Dataset saved: data/processed/full_training_5years.csv")


if __name__ == "__main__":
    main()