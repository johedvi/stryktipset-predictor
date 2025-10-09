"""
Predict Stryktipset matches
"""

import json
from main import StryktipsetPredictor
from pathlib import Path

# Stryktipset matches to predict
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

# Team name variations (API might use different names)
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
    print("\n" + "="*80)
    print("STRYKTIPSET PREDICTIONS")
    print("="*80 + "\n")
    
    # Initialize predictor
    print("Initializing predictor...")
    predictor = StryktipsetPredictor(use_ml=True)
    
    # Load historical data for all available leagues and seasons
    print("Loading historical data...")
    predictor.load_historical_data('premier_league', [2022, 2023])
    predictor.load_historical_data('championship', [2022, 2023])
    
    # Load all fixtures to find team IDs
    print("Loading fixture data to find teams...")
    all_fixtures = load_all_fixtures()
    
    if not all_fixtures:
        print("\n⚠️  No fixture data found!")
        print("Make sure you've run: python data_fetcher.py")
        return
    
    print(f"Loaded {len(all_fixtures)} fixtures\n")
    
    # Make predictions
    results = []
    not_found = []
    
    for i, match in enumerate(MATCHES, 1):
        home_name = match['home']
        away_name = match['away']
        
        # Find team IDs
        home_id, home_official = find_team_id(home_name, all_fixtures)
        away_id, away_official = find_team_id(away_name, all_fixtures)
        
        if not home_id or not away_id:
            not_found.append(f"{i}. {home_name} vs {away_name}")
            results.append({
                'match_num': i,
                'home': home_name,
                'away': away_name,
                'prediction': 'N/A',
                'confidence': 0,
                'reason': 'Team not found in data'
            })
            continue
        
        # Determine league (simplified - check which league the teams are in)
        league_name = 'premier_league'  # Default
        for fixture in all_fixtures:
            if fixture['teams']['home']['id'] == home_id:
                league_id = fixture['league']['id']
                if league_id == 40:  # Championship
                    league_name = 'championship'
                break
        
        # Make prediction (use mid-season date)
        try:
            prediction = predictor.predict_match(
                home_id=home_id,
                away_id=away_id,
                home_name=home_official,
                away_name=away_official,
                league_name=league_name,
                season=2023,
                match_date='2023-02-15'  # Mid-season date
            )
            
            if prediction and prediction.get('final_prediction'):
                pred_map = {'H': '1', 'D': 'X', 'A': '2'}
                results.append({
                    'match_num': i,
                    'home': home_official,
                    'away': away_official,
                    'prediction': pred_map[prediction['final_prediction']],
                    'confidence': prediction['final_confidence'],
                    'probs': prediction.get('ensemble', {}).get('probabilities', {})
                })
            else:
                results.append({
                    'match_num': i,
                    'home': home_official,
                    'away': away_official,
                    'prediction': 'N/A',
                    'confidence': 0,
                    'reason': 'Insufficient data'
                })
        
        except Exception as e:
            results.append({
                'match_num': i,
                'home': home_official,
                'away': away_official,
                'prediction': 'ERROR',
                'confidence': 0,
                'reason': str(e)[:50]
            })
    
    # Display results
    print("\n" + "="*80)
    print("PREDICTIONS")
    print("="*80 + "\n")
    
    for result in results:
        print(f"{result['match_num']:2d}. {result['home']:20s} vs {result['away']:20s}", end="")
        
        if result['prediction'] in ['1', 'X', '2']:
            print(f" → {result['prediction']} ({result['confidence']*100:.0f}%)")
            
            if 'probs' in result and result['probs']:
                probs = result['probs']
                print(f"     Probabilities: 1={probs.get('H', 0)*100:.0f}% | "
                      f"X={probs.get('D', 0)*100:.0f}% | "
                      f"2={probs.get('A', 0)*100:.0f}%")
        else:
            print(f" → {result['prediction']}")
            if 'reason' in result:
                print(f"     Reason: {result['reason']}")
        print()
    
    # Summary
    print("="*80)
    valid_predictions = len([r for r in results if r['prediction'] in ['1', 'X', '2']])
    print(f"Valid predictions: {valid_predictions}/{len(MATCHES)}")
    
    if not_found:
        print(f"\n⚠️  Teams not found in database:")
        for match in not_found:
            print(f"   {match}")
        print("\nThese teams might be from leagues not in your data (2023 only).")
        print("You need 2024/2025 data for current season matches (requires paid API).")
    
    print("\n" + "="*80)
    
    # Save to file
    with open('stryktipset_predictions.txt', 'w') as f:
        f.write("STRYKTIPSET PREDICTIONS\n")
        f.write("="*80 + "\n\n")
        
        for result in results:
            f.write(f"{result['match_num']:2d}. {result['home']:20s} vs {result['away']:20s} → {result['prediction']}\n")
    
    print("✓ Predictions saved to: stryktipset_predictions.txt\n")


if __name__ == "__main__":
    main()