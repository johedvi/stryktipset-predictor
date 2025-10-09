"""
Main orchestrator for Stryktipset predictions
Combines rule-based and ML predictions
"""

import argparse
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd

from data_fetcher import APIFootballFetcher
from rule_based_predictor import RuleBasedPredictor
from ml_predictor import MLPredictor
from feature_engineering import FeatureEngineer
from utils import setup_logging
from config import LEAGUES, PREDICTION_CONFIDENCE_THRESHOLD

logger = setup_logging(__name__)


class StryktipsetPredictor:
    """
    Main predictor combining multiple approaches
    """
    
    def __init__(self, use_ml: bool = True):
        """
        Initialize the predictor
        
        Args:
            use_ml: Whether to use ML model (requires trained model)
        """
        self.fetcher = APIFootballFetcher()
        self.rule_predictor = RuleBasedPredictor()
        self.ml_predictor = MLPredictor() if use_ml else None
        self.feature_engineer = FeatureEngineer()
        self.use_ml = use_ml
        
        # Try to load ML model if using ML
        if self.use_ml and self.ml_predictor:
            success = self.ml_predictor.load_model('RF_boosting_5years.pkl')
            if not success:
                logger.warning("ML model not found. Will use rule-based only.")
                self.use_ml = False
        
        logger.info(f"StryktipsetPredictor initialized (ML: {self.use_ml})")
    
    def load_historical_data(self, league_name: str, seasons: List[int]):
        """
        Load historical data for predictions
        
        Args:
            league_name: League name
            seasons: List of seasons to load
        """
        logger.info(f"Loading historical data for {league_name}")
        
        for season in seasons:
            self.rule_predictor.load_league_data(league_name, season)
            self.feature_engineer.load_league_data(league_name, season)
    
    def predict_match(self, home_id: int, away_id: int, home_name: str, 
                     away_name: str, league_name: str, season: int, 
                     match_date: str) -> Dict[str, Any]:
        """
        Predict a single match using ensemble approach
        
        Args:
            home_id: Home team ID
            away_id: Away team ID
            home_name: Home team name
            away_name: Away team name
            league_name: League name
            season: Season year
            match_date: Match date (YYYY-MM-DD)
        
        Returns:
            Combined prediction dictionary
        """
        predictions = {
            'home_team': home_name,
            'away_team': away_name,
            'match_date': match_date,
            'league': league_name,
        }
        
        # Get rule-based prediction
        rule_pred = self.rule_predictor.predict_match(
            home_id, away_id, home_name, away_name, 
            league_name, season, match_date
        )
        
        if rule_pred:
            predictions['rule_based'] = {
                'prediction': rule_pred['prediction'],
                'confidence': rule_pred['confidence'],
                'probabilities': {
                    'H': rule_pred['home_score'],
                    'D': rule_pred['draw_score'],
                    'A': rule_pred['away_score'],
                }
            }
        
        # Get ML prediction if available
        if self.use_ml:
            features = self.feature_engineer.create_match_features(
                home_id, away_id, match_date, league_name, season
            )
            
            if features:
                ml_pred = self.ml_predictor.predict(features)
                
                if ml_pred:
                    predictions['ml'] = {
                        'prediction': ml_pred['prediction'],
                        'confidence': ml_pred['confidence'],
                        'probabilities': ml_pred['probabilities']
                    }
        
        # Ensemble prediction (average probabilities)
        if 'rule_based' in predictions and 'ml' in predictions:
            ensemble_probs = {
                'H': (predictions['rule_based']['probabilities']['H'] + 
                      predictions['ml']['probabilities']['H']) / 2,
                'D': (predictions['rule_based']['probabilities']['D'] + 
                      predictions['ml']['probabilities']['D']) / 2,
                'A': (predictions['rule_based']['probabilities']['A'] + 
                      predictions['ml']['probabilities']['A']) / 2,
            }
            
            ensemble_prediction = max(ensemble_probs, key=ensemble_probs.get)
            ensemble_confidence = ensemble_probs[ensemble_prediction]
            
            predictions['ensemble'] = {
                'prediction': ensemble_prediction,
                'confidence': ensemble_confidence,
                'probabilities': ensemble_probs
            }
            
            # Use ensemble as final prediction
            predictions['final_prediction'] = ensemble_prediction
            predictions['final_confidence'] = ensemble_confidence
            predictions['method'] = 'ensemble'
        
        elif 'ml' in predictions:
            predictions['final_prediction'] = predictions['ml']['prediction']
            predictions['final_confidence'] = predictions['ml']['confidence']
            predictions['method'] = 'ml_only'
        
        elif 'rule_based' in predictions:
            predictions['final_prediction'] = predictions['rule_based']['prediction']
            predictions['final_confidence'] = predictions['rule_based']['confidence']
            predictions['method'] = 'rule_based_only'
        
        else:
            predictions['final_prediction'] = None
            predictions['final_confidence'] = 0
            predictions['method'] = 'none'
        
        return predictions
    
    def predict_upcoming_matches(self, league_name: str, season: int = 2025) -> List[Dict[str, Any]]:
        """
        Predict all upcoming matches in a league
        
        Args:
            league_name: League name
            season: Season year
        
        Returns:
            List of predictions
        """
        logger.info(f"Fetching upcoming matches for {league_name} {season}")
        
        # Get league ID
        league_id = LEAGUES.get(league_name)
        if not league_id:
            logger.error(f"Unknown league: {league_name}")
            return []
        
        # Fetch fixtures
        fixtures = self.fetcher.get_fixtures(league_id, season)
        
        # Filter for upcoming matches (not yet played)
        upcoming = [f for f in fixtures if f['fixture']['status']['short'] in ['NS', 'TBD']]
        
        logger.info(f"Found {len(upcoming)} upcoming matches")
        
        predictions = []
        
        for fixture in upcoming[:10]:  # Limit to next 10 matches
            home_id = fixture['teams']['home']['id']
            away_id = fixture['teams']['away']['id']
            home_name = fixture['teams']['home']['name']
            away_name = fixture['teams']['away']['name']
            match_date = fixture['fixture']['date'][:10]
            
            pred = self.predict_match(
                home_id, away_id, home_name, away_name,
                league_name, season, match_date
            )
            
            predictions.append(pred)
        
        return predictions
    
    def display_predictions(self, predictions: List[Dict[str, Any]]):
        """
        Display predictions in a readable format
        
        Args:
            predictions: List of prediction dictionaries
        """
        print("\n" + "="*80)
        print("STRYKTIPSET PREDICTIONS")
        print("="*80 + "\n")
        
        for i, pred in enumerate(predictions, 1):
            if pred.get('final_prediction') is None:
                continue
            
            # Only show predictions above confidence threshold
            if pred['final_confidence'] < PREDICTION_CONFIDENCE_THRESHOLD:
                continue
            
            prediction_text = {
                'H': '1 (Home Win)',
                'D': 'X (Draw)',
                'A': '2 (Away Win)'
            }.get(pred['final_prediction'], 'Unknown')
            
            print(f"{i}. {pred['home_team']} vs {pred['away_team']}")
            print(f"   Date: {pred['match_date']}")
            print(f"   Prediction: {prediction_text}")
            print(f"   Confidence: {pred['final_confidence']:.1%}")
            print(f"   Method: {pred['method']}")
            
            # Show probabilities if ensemble
            if 'ensemble' in pred:
                probs = pred['ensemble']['probabilities']
                print(f"   Probabilities: 1={probs['H']:.1%} | X={probs['D']:.1%} | 2={probs['A']:.1%}")
            
            print()
    
    def export_predictions_to_csv(self, predictions: List[Dict[str, Any]], filename: str):
        """
        Export predictions to CSV
        
        Args:
            predictions: List of predictions
            filename: Output filename
        """
        rows = []
        
        for pred in predictions:
            if pred.get('final_prediction') is None:
                continue
            
            row = {
                'date': pred['match_date'],
                'home_team': pred['home_team'],
                'away_team': pred['away_team'],
                'prediction': pred['final_prediction'],
                'confidence': pred['final_confidence'],
                'method': pred['method'],
            }
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        logger.info(f"Exported predictions to {filename}")


def main():
    """
    Main entry point
    """
    parser = argparse.ArgumentParser(description='Stryktipset Match Predictor')
    parser.add_argument('--league', type=str, default='premier_league',
                       help='League name (e.g., premier_league, la_liga)')
    parser.add_argument('--season', type=int, default=2025,
                       help='Season year')
    parser.add_argument('--no-ml', action='store_true',
                       help='Disable ML predictions, use rule-based only')
    parser.add_argument('--export', type=str, default=None,
                       help='Export predictions to CSV file')
    
    args = parser.parse_args()
    
    # Initialize predictor
    use_ml = not args.no_ml
    predictor = StryktipsetPredictor(use_ml=use_ml)
    
    # Load historical data (last 2 seasons)
    print(f"\nLoading historical data for {args.league}...")
    predictor.load_historical_data(args.league, [args.season - 2, args.season - 1, args.season])
    
    # Get predictions for upcoming matches
    print(f"\nPredicting upcoming matches...")
    predictions = predictor.predict_upcoming_matches(args.league, args.season)
    
    # Display predictions
    predictor.display_predictions(predictions)
    
    # Export if requested
    if args.export:
        predictor.export_predictions_to_csv(predictions, args.export)
    
    print("\n✓ Prediction complete!")
    print(f"\nTotal predictions: {len([p for p in predictions if p.get('final_prediction')])}")
    print(f"High confidence predictions (>{PREDICTION_CONFIDENCE_THRESHOLD:.0%}): "
          f"{len([p for p in predictions if p.get('final_confidence', 0) >= PREDICTION_CONFIDENCE_THRESHOLD])}")


if __name__ == "__main__":
    main()