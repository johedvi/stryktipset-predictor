"""
Make predictions with Stryktipset signing strategy
Now uses league-specific models for better accuracy
"""

from src.prediction.league_specific_predictor import LeagueSpecificPredictor
from src.features.feature_engineering import FeatureEngineer
from stryktipset_strategy import create_stryktipset_coupon, display_coupon, calculate_expected_combinations
import json
from pathlib import Path
from datetime import datetime

# Your 13 Stryktipset matches
MATCHES = [
    {"home": "AFC Wimbledon", "away": "Port Vale"},
    {"home": "Burton", "away": "Bolton"},
    {"home": "Exeter", "away": "Reading"},
    {"home": "Leyton Orient", "away": "Doncaster"},
    {"home": "Stockport", "away": "Blackpool"},
    {"home": "Wigan", "away": "Wycombe"},
    {"home": "Bristol Rovers", "away": "Milton Keynes Dons"},
    {"home": "Chesterfield", "away": "Salford City"},
    {"home": "Crawley Town", "away": "Walsall"},
    {"home": "Fleetwood", "away": "Harrogate Town"},
    {"home": "Grimsby", "away": "Colchester"},
    {"home": "Shrewsbury", "away": "Cambridge United"},
    {"home": "Tranmere", "away": "Barnet"},
]

# Team name variations
TEAM_NAME_MAPPING = {
    "Man United": "Manchester United",
    "Mboro": "Middlesbrough",
    "Charlotn": "Charlton",
    "Huddersf.": "Huddersfield",
    # QPR and West Brom are already correct in the database
}

# League ID mapping
LEAGUE_ID_MAP = {
    39: "premier_league",
    40: "championship",
    41: "league_one",
    42: "league_two"
}


def find_team_id(team_name, all_fixtures):
    """
    Find team ID by name from fixtures data
    
    Args:
        team_name: Team name to search for
        all_fixtures: List of all fixtures
    
    Returns:
        Tuple of (team_id, official_name, league_name) or (None, None, None)
    """
    # Normalize team name
    team_name = TEAM_NAME_MAPPING.get(team_name, team_name)
    team_name_lower = team_name.lower()
    
    for fixture in all_fixtures:
        home_team = fixture['teams']['home']
        away_team = fixture['teams']['away']
        league_id = fixture['league']['id']
        league_name = LEAGUE_ID_MAP.get(league_id, 'premier_league')
        
        if team_name_lower in home_team['name'].lower():
            return home_team['id'], home_team['name'], league_name
        if team_name_lower in away_team['name'].lower():
            return away_team['id'], away_team['name'], league_name
    
    return None, None, None


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
    print("="*100)
    print("STRYKTIPSET PREDICTOR - LEAGUE-SPECIFIC MODELS")
    print("="*100)
    
    # Initialize league-specific predictor
    print("\n🔄 Loading league-specific models...")
    predictor = LeagueSpecificPredictor()
    
    # Show which models are loaded
    info = predictor.model_info()
    print(f"✓ Loaded {info['total_models']} models: {', '.join(info['leagues'])}")
    
    # Initialize feature engineer for all leagues
    print("\n🔄 Loading historical data for feature engineering...")
    engineer = FeatureEngineer()
    
    # Load data for all leagues (need for feature calculation)
    seasons = [2020, 2021, 2022, 2023, 2024, 2025]
    for league in ['premier_league', 'championship', 'league_one', 'league_two']:
        for season in seasons:
            engineer.load_league_data(league, season)
    print("✓ Historical data loaded")
    
    # Load fixtures
    print("\n🔄 Loading fixtures database...")
    all_fixtures = load_all_fixtures()
    print(f"✓ Loaded {len(all_fixtures)} fixtures\n")
    
    # Make predictions
    print("="*100)
    print("PREDICTING 13 MATCHES")
    print("="*100)
    
    predictions = []
    match_date = datetime.now().strftime('%Y-%m-%d')
    
    for i, match in enumerate(MATCHES, 1):
        home_name = match['home']
        away_name = match['away']
        
        print(f"\n{i}/13: {home_name} vs {away_name}")
        print("-" * 60)
        
        # Find team IDs and league
        home_id, home_official, home_league = find_team_id(home_name, all_fixtures)
        away_id, away_official, away_league = find_team_id(away_name, all_fixtures)
        
        if not home_id or not away_id:
            print("❌ Teams not found in database")
            predictions.append({
                'home_team': home_name,
                'away_team': away_name,
                'final_prediction': None,
                'probabilities': {'H': 0.33, 'D': 0.34, 'A': 0.33}
            })
            continue
        
        # Determine the league (use home team's league)
        league_name = home_league
        
        # Check if we have a model for this league
        if league_name not in predictor.models:
            print(f"⚠️  No model available for {league_name}")
            predictions.append({
                'home_team': home_official,
                'away_team': away_official,
                'final_prediction': None,
                'probabilities': {'H': 0.33, 'D': 0.34, 'A': 0.33}
            })
            continue
        
        print(f"League: {league_name.replace('_', ' ').title()}")
        print(f"Model: {league_name}_model.pkl")
        
        # Create features for this match
        features = engineer.create_match_features(
            home_id=home_id,
            away_id=away_id,
            match_date=match_date,
            league_name=league_name,
            season=2024
        )
        
        if features is None:
            print("⚠️  Could not create features (insufficient data)")
            predictions.append({
                'home_team': home_official,
                'away_team': away_official,
                'final_prediction': None,
                'probabilities': {'H': 0.33, 'D': 0.34, 'A': 0.33}
            })
            continue
        
        # Predict using league-specific model
        result = predictor.predict_match(
            home_id=home_id,
            away_id=away_id,
            home_name=home_official,
            away_name=away_official,
            league_name=league_name,
            features=features
        )
        
        # Display prediction
        print(f"Prediction: {result['final_prediction']}")
        print(f"Probabilities:")
        print(f"  Home (1): {result['probabilities']['H']:.1%}")
        print(f"  Draw (X): {result['probabilities']['D']:.1%}")
        print(f"  Away (2): {result['probabilities']['A']:.1%}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"✓ Predicted using {result['model_used']} model")
        
        predictions.append(result)
    
    # Create optimal 192 SEK coupon
    print("\n" + "="*100)
    print("CREATING OPTIMAL 192 SEK COUPON")
    print("="*100)
    
    from smart_budget_coupon import create_optimal_192_coupon
    
    coupon = create_optimal_192_coupon(predictions)
    display_coupon(coupon, show_reasoning=True)
    
    stats = calculate_expected_combinations(coupon)
    print(f"\n💰 Final Cost: {stats['cost_sek']} SEK")
    print(f"📊 Combinations: {stats['total_combinations']}")
    print(f"🎯 Singles: {stats['single_signs']}")
    print(f"🎲 Doubles: {stats['double_signs']}")
    print(f"🔄 Triples: {stats['triple_signs']}")
    
    # Summary
    print("\n" + "="*100)
    print("PREDICTION SUMMARY")
    print("="*100)
    
    leagues_used = {}
    for pred in predictions:
        if pred.get('model_used'):
            league = pred['model_used']
            leagues_used[league] = leagues_used.get(league, 0) + 1
    
    print(f"\nMatches by league:")
    for league, count in leagues_used.items():
        print(f"  {league.replace('_', ' ').title()}: {count} matches")
    
    print(f"\nTotal matches: {len(predictions)}")
    print(f"Successful predictions: {sum(1 for p in predictions if p.get('final_prediction') is not None)}")
    print(f"Failed predictions: {sum(1 for p in predictions if p.get('final_prediction') is None)}")
    
    print("\n✓ Prediction complete!")
    print("="*100)


if __name__ == "__main__":
    main()