"""
Make predictions with Stryktipset signing strategy
"""

from main import StryktipsetPredictor
from stryktipset_strategy import create_stryktipset_coupon, display_coupon, calculate_expected_combinations
import json
from pathlib import Path

# Your 13 Stryktipset matches
MATCHES = [
    {"home": "Chelsea", "away": "Liverpool"},
    {"home": "Arsenal", "away": "West Ham"},
    {"home": "Manchester United", "away": "Sunderland"},
    {"home": "Bristol City", "away": "QPR"},
    {"home": "Derby", "away": "Southampton"},
    {"home": "Millwall", "away": "West Brom"},
    {"home": "Portsmouth", "away": "Middlesbrough"},
    {"home": "Preston", "away": "Charlton"},
    {"home": "Swansea", "away": "Leicester"},
    {"home": "Watford", "away": "Oxford United"},
    {"home": "Huddersfield", "away": "Stockport"},
    {"home": "Plymouth", "away": "Wigan"},
    {"home": "Stevenage", "away": "Luton"},
]

# Team name variations
TEAM_NAME_MAPPING = {
    "Man United": "Manchester United",
    "Mboro": "Middlesbrough",
    "Charlotn": "Charlton",
    "Huddersf.": "Huddersfield",
    "QPR": "Queens Park Rangers",
    "West Brom": "West Bromwich Albion",
}


def find_team_id(team_name, all_fixtures):
    """
    Find team ID by name from fixtures data
    
    Args:
        team_name: Team name to search for
        all_fixtures: List of all fixtures
    
    Returns:
        Tuple of (team_id, official_name) or (None, None)
    """
    # Normalize team name
    team_name = TEAM_NAME_MAPPING.get(team_name, team_name)
    team_name_lower = team_name.lower()
    
    for fixture in all_fixtures:
        home_team = fixture['teams']['home']
        away_team = fixture['teams']['away']
        
        if team_name_lower in home_team['name'].lower():
            return home_team['id'], home_team['name']
        if team_name_lower in away_team['name'].lower():
            return away_team['id'], away_team['name']
    
    return None, None


def load_all_fixtures():
    """Load all available fixtures from data directory"""
    all_fixtures = []
    
    data_dir = Path('data/raw')
    for json_file in data_dir.glob('*.json'):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                if 'fixtures' in data:
                    all_fixtures.extend(data['fixtures'])
        except:
            pass
    
    return all_fixtures


def main():
    # Initialize predictor
    print("Loading predictor...")
    predictor = StryktipsetPredictor(use_ml=True)
    
    # Load ALL seasons including 2024 (need current season data for form)
    predictor.load_historical_data('premier_league', [2020, 2021, 2022, 2023, 2024])
    predictor.load_historical_data('championship', [2020, 2021, 2022, 2023, 2024])
    predictor.load_historical_data('league_one', [2020, 2021, 2022, 2023, 2024])
    
    # Load fixtures
    print("Loading fixtures...")
    all_fixtures = load_all_fixtures()
    print(f"Loaded {len(all_fixtures)} fixtures from database\n")
    
    # Make predictions
    predictions = []
    
    for i, match in enumerate(MATCHES, 1):
        home_name = match['home']
        away_name = match['away']
        
        print(f"Predicting {i}/13: {home_name} vs {away_name}...", end=" ")
        
        home_id, home_official = find_team_id(home_name, all_fixtures)
        away_id, away_official = find_team_id(away_name, all_fixtures)
        
        if not home_id or not away_id:
            print("❌ Not found in database")
            predictions.append({
                'home_team': home_name,
                'away_team': away_name,
                'final_prediction': None
            })
            continue
        
        # Determine league
        league_name = 'premier_league'
        for fixture in all_fixtures:
            if fixture['teams']['home']['id'] == home_id:
                league_id = fixture['league']['id']
                if league_id == 40:
                    league_name = 'championship'
                elif league_id == 41:
                    league_name = 'league_one'
                break
        
        pred = predictor.predict_match(
            home_id=home_id,
            away_id=away_id,
            home_name=home_official,
            away_name=away_official,
            league_name=league_name,
            season=2024,
            match_date='2024-10-09'  # Current date
        )
        
        if pred and pred.get('final_prediction'):
            print(f"✓ Predicted")
        else:
            print("⚠️ Insufficient data")
        
        predictions.append(pred)
    
    # Create coupon with different strategies
    print("\n" + "="*100)
    print("COMPARING STRATEGIES")
    print("="*100)
    
    for strategy in ['aggressive', 'balanced', 'safe']:
        print(f"\n{'='*100}")
        print(f"{strategy.upper()} STRATEGY")
        print(f"{'='*100}")
        
        coupon = create_stryktipset_coupon(predictions, strategy=strategy)
        display_coupon(coupon, show_reasoning=True)
        
        stats = calculate_expected_combinations(coupon)
        print(f"\n💰 Cost: {stats['cost_sek']} SEK")
        print(f"📊 Combinations: {stats['total_combinations']}")
        print(f"🎯 Single signs: {stats['single_signs']} (high confidence)")
        print(f"🎲 Double signs: {stats['double_signs']} (medium confidence)")
        print(f"🔄 Triple signs: {stats['triple_signs']} (low confidence)")


if __name__ == "__main__":
    main()