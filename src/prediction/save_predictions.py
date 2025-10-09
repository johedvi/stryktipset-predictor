"""
Save predictions to a readable text file
"""

from src.prediction.predict_with_strategy import *
from main import StryktipsetPredictor
from src.prediction.predict_with_strategy import load_all_fixtures

def save_predictions_to_file(predictions, filename='stryktipset_predictions.txt'):
    """
    Save predictions to a readable text file
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("STRYKTIPSET PREDICTIONS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated: 2024-10-09\n")
        f.write(f"Model: Random Forest (5 years data)\n")
        f.write(f"Leagues: Premier League, Championship, League One\n\n")
        f.write("="*80 + "\n\n")
        
        for i, pred in enumerate(predictions, 1):
            home = pred.get('home_team', 'Unknown')
            away = pred.get('away_team', 'Unknown')
            
            f.write(f"Match {i}: {home} vs {away}\n")
            f.write("-" * 60 + "\n")
            
            if not pred.get('final_prediction'):
                f.write("❌ NO PREDICTION AVAILABLE (team not in database)\n")
                f.write("Suggested sign: 1X2 (all outcomes)\n\n")
                continue
            
            # Get probabilities
            probs = pred.get('ensemble', {}).get('probabilities', 
                           pred.get('ml', {}).get('probabilities', {}))
            
            if probs:
                # Find most likely outcome
                max_prob = max(probs.values())
                winner = max(probs, key=probs.get)
                
                winner_text = {
                    'H': f'HOME WIN ({home})',
                    'D': 'DRAW',
                    'A': f'AWAY WIN ({away})'
                }[winner]
                
                f.write(f"🎯 MODEL PREDICTION: {winner_text}\n")
                f.write(f"   Confidence: {max_prob*100:.1f}%\n\n")
                
                f.write("Probabilities:\n")
                f.write(f"   Home Win (1): {probs['H']*100:5.1f}%\n")
                f.write(f"   Draw (X):     {probs['D']*100:5.1f}%\n")
                f.write(f"   Away Win (2): {probs['A']*100:5.1f}%\n\n")
                
                # Determine sign recommendation
                if max_prob >= 0.52:
                    sign = {'H': '1', 'D': 'X', 'A': '2'}[winner]
                    f.write(f"✅ RECOMMENDED SIGN: {sign} (single sign)\n")
                elif max_prob >= 0.40:
                    # Double sign
                    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                    top_two = [{'H': '1', 'D': 'X', 'A': '2'}[p[0]] for p in sorted_probs[:2]]
                    sign = ''.join(sorted(top_two))
                    f.write(f"⚠️  RECOMMENDED SIGN: {sign} (double sign - medium confidence)\n")
                else:
                    f.write(f"❓ RECOMMENDED SIGN: 1X2 (all outcomes - low confidence)\n")
                
                # Confidence assessment
                if max_prob >= 0.55:
                    f.write("   Risk level: LOW ✅\n")
                elif max_prob >= 0.45:
                    f.write("   Risk level: MEDIUM ⚠️\n")
                else:
                    f.write("   Risk level: HIGH ❌\n")
            else:
                f.write("⚠️  Insufficient data for detailed prediction\n")
            
            f.write("\n" + "="*80 + "\n\n")
        
        # Summary section
        f.write("\n" + "="*80 + "\n")
        f.write("BETTING STRATEGY SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        confident_picks = []
        medium_picks = []
        uncertain_picks = []
        
        for i, pred in enumerate(predictions, 1):
            probs = pred.get('ensemble', {}).get('probabilities', {})
            if probs:
                max_prob = max(probs.values())
                match = f"{i}. {pred['home_team']} vs {pred['away_team']}"
                
                if max_prob >= 0.52:
                    confident_picks.append(match)
                elif max_prob >= 0.40:
                    medium_picks.append(match)
                else:
                    uncertain_picks.append(match)
        
        f.write(f"CONFIDENT PICKS (single sign recommended):\n")
        if confident_picks:
            for pick in confident_picks:
                f.write(f"  ✅ {pick}\n")
        else:
            f.write("  None - all matches have low confidence\n")
        
        f.write(f"\nMEDIUM CONFIDENCE (double sign recommended):\n")
        if medium_picks:
            for pick in medium_picks:
                f.write(f"  ⚠️  {pick}\n")
        else:
            f.write("  None\n")
        
        f.write(f"\nUNCERTAIN (1X2 recommended):\n")
        if uncertain_picks:
            for pick in uncertain_picks:
                f.write(f"  ❌ {pick}\n")
        else:
            f.write("  None\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("ESTIMATED COSTS:\n")
        f.write("="*80 + "\n")
        f.write(f"If all confident: ~{2**len(confident_picks)} SEK\n")
        f.write(f"If confident + medium: ~{2**len(confident_picks) * 2**len(medium_picks)} SEK\n")
        f.write(f"If all matches: ~1,594,323 SEK (NOT RECOMMENDED!)\n\n")
        


def main():
    # Initialize predictor (same as predict_with_strategy.py)
    print("Loading predictor...")
    predictor = StryktipsetPredictor(use_ml=True)
    
    predictor.load_historical_data('premier_league', [2020, 2021, 2022, 2023, 2024])
    predictor.load_historical_data('championship', [2020, 2021, 2022, 2023, 2024])
    predictor.load_historical_data('league_one', [2020, 2021, 2022, 2023, 2024])
    
    print("Loading fixtures...")
    all_fixtures = load_all_fixtures()
    
    # Make predictions
    predictions = []
    
    for match in MATCHES:
        home_id, home_official = find_team_id(match['home'], all_fixtures)
        away_id, away_official = find_team_id(match['away'], all_fixtures)
        
        if not home_id or not away_id:
            predictions.append({
                'home_team': match['home'],
                'away_team': match['away'],
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
            home_id, away_id, home_official, away_official,
            league_name, 2024, '2024-10-09'
        )
        
        predictions.append(pred)
    
    # Save to file
    print("\nSaving predictions to file...")
    save_predictions_to_file(predictions, 'stryktipset_predictions.txt')
    
    print("✅ Predictions saved to: stryktipset_predictions.txt")
    print("\nOpen the file to see:")
    print("  - Model's winner prediction for each match")
    print("  - Confidence levels")
    print("  - Recommended signs (1, X, 2, 1X, 12, X2, or 1X2)")
    print("  - Risk assessment")
    print("  - Betting strategy summary")


if __name__ == "__main__":
    main()