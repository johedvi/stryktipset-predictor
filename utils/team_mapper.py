"""
Team ID mapper - dynamically loads from your fetched data
"""

import json
from pathlib import Path
from typing import Dict

def get_all_teams() -> Dict[str, Dict[str, int]]:
    """
    Load all teams from fetched data
    Returns dict: {team_name: {league_id: int, team_id: int}}
    """
    teams = {}
    data_dir = Path("data/raw")
    
    # League mapping
    league_names = {
        39: "Premier League",
        40: "Championship",
        41: "League One",
        42: "League Two"
    }
    
    # Look at both 2024 and 2025 files to capture promotions/relegations
    for json_file in data_dir.glob("*_202[45].json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            league_id = data.get('league_id')
                
            if league_id not in league_names:
                continue
                
            # Extract teams from fixtures
            for fixture in data.get('fixtures', []):
                home_team = fixture['teams']['home']
                away_team = fixture['teams']['away']
                
                # Only update if team not already in dict, or if this is 2025 data (more recent)
                is_2025 = '2025' in str(json_file)
                
                for team in [home_team, away_team]:
                    if team['name'] not in teams or is_2025:
                        teams[team['name']] = {
                            'id': team['id'],
                            'league': league_names[league_id],
                            'league_id': league_id
                        }
                
        except Exception as e:
            continue
    
    return teams

def get_team_id(team_name: str) -> int:
    """Get team ID by name"""
    teams = get_all_teams()
    team = teams.get(team_name)
    return team['id'] if team else None

def get_teams_by_league(league_name: str) -> Dict[str, int]:
    """Get all teams in a specific league"""
    all_teams = get_all_teams()
    return {
        name: info['id'] 
        for name, info in all_teams.items() 
        if info['league'] == league_name
    }