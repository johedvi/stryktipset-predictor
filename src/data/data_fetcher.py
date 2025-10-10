"""
Data fetcher for API-Football
Handles all API requests with caching and rate limiting
"""

import requests
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from config import (
    API_FOOTBALL_KEY,
    API_FOOTBALL_BASE_URL,
    API_RATE_LIMIT_CALLS,
    API_RATE_LIMIT_PERIOD,
    RAW_DATA_DIR,
    CACHE_EXPIRY_HOURS,
    LEAGUES,
    SEASONS,
)
from utils.utils import setup_logging, generate_cache_key, date_to_string


logger = setup_logging(__name__)


class APIFootballFetcher:
    """
    Fetches data from API-Football with caching and rate limiting
    """
    
    def __init__(self, api_key: str = API_FOOTBALL_KEY):
        """
        Initialize the API fetcher
        
        Args:
            api_key: API-Football API key
        """
        self.api_key = api_key
        self.base_url = API_FOOTBALL_BASE_URL
        self.headers = {
            "x-apisports-key": self.api_key,
        }
        self.cache_dir = RAW_DATA_DIR / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Rate limiting
        self.request_times = []
        self.max_requests = API_RATE_LIMIT_CALLS
        self.time_window = API_RATE_LIMIT_PERIOD
        
        logger.info("APIFootballFetcher initialized")
    
    def _check_rate_limit(self):
   
        now = time.time()
    
    # Pro Plan: 300 requests per minute = 0.2 seconds between requests
        if self.request_times:
            time_since_last = now - self.request_times[-1]
            if time_since_last < 0.2:  # Changed from 6 to 0.2
                wait_time = 0.2 - time_since_last
                time.sleep(wait_time)
    
        self.request_times.append(time.time())
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key"""
        return self.cache_dir / f"{cache_key}.json"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cached data is still valid"""
        if not cache_path.exists():
            return False
        
        # Check if cache has expired
        cache_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        expiry_time = cache_time + timedelta(hours=CACHE_EXPIRY_HOURS)
        
        return datetime.now() < expiry_time
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load data from cache if valid"""
        cache_path = self._get_cache_path(cache_key)
        
        if self._is_cache_valid(cache_path):
            logger.info(f"Loading from cache: {cache_key}")
            with open(cache_path, 'r') as f:
                return json.load(f)
        
        return None
    
    def _save_to_cache(self, cache_key: str, data: Dict[str, Any]):
        """Save data to cache"""
        cache_path = self._get_cache_path(cache_key)
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved to cache: {cache_key}")
    
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make an API request with caching
        
        Args:
            endpoint: API endpoint (e.g., '/fixtures')
            params: Query parameters
        
        Returns:
            API response data
        """
        cache_key = generate_cache_key(endpoint, params)
        
        # Try to load from cache first
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        # Make the actual API request
        self._check_rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        logger.info(f"Making API request: {endpoint} with params {params}")
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if data.get('errors'):
                logger.error(f"API returned errors: {data['errors']}")
                return None
            
            # Save to cache
            self._save_to_cache(cache_key, data)
            
            return data
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None
    
    def get_fixtures(self, league_id: int, season: int) -> List[Dict[str, Any]]:
        """
        Get all fixtures for a league and season
        
        Args:
            league_id: League ID
            season: Season year
        
        Returns:
            List of fixture dictionaries
        """
        logger.info(f"Fetching fixtures for league {league_id}, season {season}")
        
        data = self._make_request('/fixtures', {
            'league': league_id,
            'season': season,
        })
        
        if data and data.get('response'):
            return data['response']
        return []
    
    def get_team_statistics(self, team_id: int, league_id: int, season: int) -> Dict[str, Any]:
        """
        Get statistics for a team in a specific league/season
        
        Args:
            team_id: Team ID
            league_id: League ID
            season: Season year
        
        Returns:
            Team statistics dictionary
        """
        logger.info(f"Fetching statistics for team {team_id}")
        
        data = self._make_request('/teams/statistics', {
            'team': team_id,
            'league': league_id,
            'season': season,
        })
        
        if data and data.get('response'):
            return data['response']
        return {}
    
    def get_head_to_head(self, team1_id: int, team2_id: int, last_n: int = 10) -> List[Dict[str, Any]]:
        """
        Get head-to-head matches between two teams
        
        Args:
            team1_id: First team ID
            team2_id: Second team ID
            last_n: Number of recent matches to fetch
        
        Returns:
            List of head-to-head matches
        """
        logger.info(f"Fetching H2H for teams {team1_id} vs {team2_id}")
        
        data = self._make_request('/fixtures/headtohead', {
            'h2h': f"{team1_id}-{team2_id}",
            'last': last_n,
        })
        
        if data and data.get('response'):
            return data['response']
        return []
    
    def get_standings(self, league_id: int, season: int) -> List[Dict[str, Any]]:
        """
        Get league standings
        
        Args:
            league_id: League ID
            season: Season year
        
        Returns:
            List of team standings
        """
        logger.info(f"Fetching standings for league {league_id}, season {season}")
        
        data = self._make_request('/standings', {
            'league': league_id,
            'season': season,
        })
        
        if data and data.get('response') and len(data['response']) > 0:
            # API returns nested structure
            return data['response'][0]['league']['standings'][0]
        return []
    
    def fetch_all_league_data(self, league_id: int, season: int) -> Dict[str, Any]:
        """
        Fetch comprehensive data for a league/season
        
        Args:
            league_id: League ID
            season: Season year
        
        Returns:
            Dictionary containing fixtures, standings, and team stats
        """
        logger.info(f"Fetching all data for league {league_id}, season {season}")
        
        fixtures = self.get_fixtures(league_id, season)
        standings = self.get_standings(league_id, season)
        
        # Extract unique team IDs from fixtures
        team_ids = set()
        for fixture in fixtures:
            team_ids.add(fixture['teams']['home']['id'])
            team_ids.add(fixture['teams']['away']['id'])
        
        # Get statistics for each team
        team_stats = {}
        for team_id in team_ids:
            stats = self.get_team_statistics(team_id, league_id, season)
            if stats:
                team_stats[team_id] = stats
        
        return {
            'league_id': league_id,
            'season': season,
            'fixtures': fixtures,
            'standings': standings,
            'team_statistics': team_stats,
        }
    
    def save_league_data(self, league_name: str, season: int, data: Dict[str, Any]):
        """
        Save league data to file
        
        Args:
            league_name: Name of the league
            season: Season year
            data: Data to save
        """
        filename = RAW_DATA_DIR / f"{league_name}_{season}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved league data to {filename}")


def main():
    """
    Example usage: fetch data for all configured leagues
    """
    fetcher = APIFootballFetcher()
    
    # Use all configured leagues now - ADDED LEAGUE TWO
    leagues_to_fetch = {
        'premier_league': 39,
        'championship': 40,
        'league_one': 41,
        'league_two': 42,  # ← ADDED THIS LINE
    }
    
    seasons_to_fetch = [2020, 2021, 2022, 2023, 2024, 2025]  # Last 5 years
    
    for league_name, league_id in leagues_to_fetch.items():
        for season in seasons_to_fetch:
            logger.info(f"\n{'='*60}")
            logger.info(f"Fetching {league_name} - Season {season}")
            logger.info(f"{'='*60}\n")
            
            data = fetcher.fetch_all_league_data(league_id, season)
            fetcher.save_league_data(league_name, season, data)
            
            logger.info(f"✓ Completed {league_name} {season}")
            logger.info(f"  Fixtures: {len(data['fixtures'])}")
            logger.info(f"  Teams: {len(data['team_statistics'])}")


if __name__ == "__main__":
    main()