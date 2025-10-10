"""
Predictor that automatically selects the correct league-specific model
"""

import joblib
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any

from config import MODELS_DIR
from utils.utils import setup_logging

logger = setup_logging(__name__)


class LeagueSpecificPredictor:
    """
    Uses league-specific models for predictions
    """
    
    def __init__(self):
        """Initialize predictor with all league models"""
        self.models = {}
        self.feature_columns = None
        self.load_all_models()
    
    def load_all_models(self):
        """
        Load all league-specific models
        """
        leagues = ['premier_league', 'championship', 'league_one', 'league_two']
        
        for league in leagues:
            model_path = MODELS_DIR / f"{league}_model.pkl"
            if model_path.exists():
                self.models[league] = joblib.load(model_path)
                logger.info(f"Loaded {league} model")
            else:
                logger.warning(f"Model not found for {league}: {model_path}")
        
        # Load feature columns
        feature_path = MODELS_DIR / "feature_columns.pkl"
        if feature_path.exists():
            with open(feature_path, 'rb') as f:
                self.feature_columns = pickle.load(f)
            logger.info(f"Loaded {len(self.feature_columns)} feature columns")
        else:
            logger.error("Feature columns file not found!")
        
        logger.info(f"Loaded {len(self.models)} league-specific models")
    
    def predict(self, features: Dict[str, Any], league_name: str) -> Dict[str, Any]:
        """
        Predict match outcome using the appropriate league model
        
        Args:
            features: Dictionary of match features
            league_name: Name of the league (e.g., 'premier_league')
        
        Returns:
            Dictionary with prediction and probabilities
        """
        # Check if model exists for this league
        if league_name not in self.models:
            logger.error(f"No model available for {league_name}")
            logger.info(f"Available leagues: {list(self.models.keys())}")
            return {
                'prediction': None,
                'probabilities': {'H': 0.33, 'D': 0.34, 'A': 0.33},
                'confidence': 0.0,
                'model_used': 'none',
                'error': f'No model for {league_name}'
            }
        
        # Get the league-specific model
        model = self.models[league_name]
        
        # Prepare features in correct order
        feature_values = []
        for col in self.feature_columns:
            feature_values.append(features.get(col, 0))
        
        X = np.array([feature_values])
        
        # Make prediction
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        # Map to H/D/A
        classes = model.classes_
        prob_dict = {
            'H': probabilities[list(classes).index('H')] if 'H' in classes else 0.0,
            'D': probabilities[list(classes).index('D')] if 'D' in classes else 0.0,
            'A': probabilities[list(classes).index('A')] if 'A' in classes else 0.0,
        }
        
        confidence = max(probabilities)
        
        return {
            'prediction': prediction,
            'probabilities': prob_dict,
            'confidence': confidence,
            'model_used': league_name,
            'league': league_name
        }
    
    def predict_match(self, home_id: int, away_id: int, home_name: str, 
                     away_name: str, league_name: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict a specific match with full details
        
        Args:
            home_id: Home team ID
            away_id: Away team ID
            home_name: Home team name
            away_name: Away team name
            league_name: League name
            features: Match features
        
        Returns:
            Full prediction dictionary
        """
        prediction = self.predict(features, league_name)
        
        return {
            'home_team': home_name,
            'away_team': away_name,
            'home_id': home_id,
            'away_id': away_id,
            'league': league_name,
            'final_prediction': prediction['prediction'],
            'probabilities': prediction['probabilities'],
            'confidence': prediction['confidence'],
            'model_used': prediction['model_used']
        }
    
    def get_available_leagues(self) -> list:
        """
        Get list of leagues with trained models
        
        Returns:
            List of league names
        """
        return list(self.models.keys())
    
    def model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models
        
        Returns:
            Dictionary with model info
        """
        info = {
            'total_models': len(self.models),
            'leagues': list(self.models.keys()),
            'feature_count': len(self.feature_columns) if self.feature_columns else 0
        }
        
        for league, model in self.models.items():
            info[f'{league}_features'] = model.n_features_in_
            info[f'{league}_classes'] = list(model.classes_)
        
        return info


def main():
    """
    Test the league-specific predictor
    """
    print("="*80)
    print("LEAGUE-SPECIFIC PREDICTOR")
    print("="*80)
    
    predictor = LeagueSpecificPredictor()
    
    # Show model info
    info = predictor.model_info()
    print(f"\n✓ Loaded {info['total_models']} models")
    print(f"Available leagues: {', '.join(info['leagues'])}")
    print(f"Features per model: {info['feature_count']}")
    
    print("\nModel Details:")
    for league in info['leagues']:
        print(f"  {league}:")
        print(f"    - Features: {info[f'{league}_features']}")
        print(f"    - Classes: {info[f'{league}_classes']}")
    
    # Example prediction (with dummy features)
    print("\n" + "="*80)
    print("EXAMPLE PREDICTION")
    print("="*80)
    
    # Create dummy features (in real use, these come from FeatureEngineer)
    dummy_features = {col: 0.0 for col in predictor.feature_columns}
    dummy_features.update({
        'home_form_points': 2.1,
        'away_form_points': 1.5,
        'position_diff': 5,
        'home_goals_scored_avg': 1.8,
        'away_goals_scored_avg': 1.2,
    })
    
    # Predict using Premier League model
    result = predictor.predict(dummy_features, 'premier_league')
    
    print(f"\nLeague: Premier League")
    print(f"Prediction: {result['prediction']}")
    print(f"Probabilities:")
    print(f"  Home: {result['probabilities']['H']:.2%}")
    print(f"  Draw: {result['probabilities']['D']:.2%}")
    print(f"  Away: {result['probabilities']['A']:.2%}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Model used: {result['model_used']}")
    
    print("\n" + "="*80)
    print("✓ Predictor ready for use!")
    print("="*80)
    print("\nKey benefits:")
    print("  • Each league uses its own specialized model")
    print("  • Premier League predictions based on PL data only")
    print("  • Championship predictions based on Championship data only")
    print("  • Automatically selects correct model based on league")


if __name__ == "__main__":
    main()