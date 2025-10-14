"""
Train league-specific machine learning models
Each league gets its own model trained on league-specific data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

from config import PROCESSED_DATA_DIR, MODELS_DIR
from utils.utils import setup_logging

logger = setup_logging(__name__)


class LeagueSpecificTrainer:
    """
    Train separate models for each league
    """
    
    def __init__(self):
        """Initialize the trainer"""
        self.models = {}
        self.feature_columns = None
        self.leagues = ['premier_league', 'championship', 'league_one', 'league_two']
        logger.info("LeagueSpecificTrainer initialized")
    
    def load_league_data(self, league_name: str) -> pd.DataFrame:
        """
        Load training data for a specific league
        
        Args:
            league_name: Name of the league
        
        Returns:
            DataFrame with training data
        """
        filename = PROCESSED_DATA_DIR / f"{league_name}_training_6years.csv"
        
        if not filename.exists():
            logger.error(f"Training data not found: {filename}")
            return None
        
        df = pd.read_csv(filename)
        logger.info(f"Loaded {league_name} data: {len(df)} samples")
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """
        Prepare features and labels for training
        
        Args:
            df: DataFrame with training data
        
        Returns:
            X (features), y (labels)
        """
        # Define feature columns (exclude metadata and target)
        exclude_cols = ['fixture_id', 'date', 'home_team', 'away_team', 
                       'home_goals', 'away_goals', 'result', 'league', 'season']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Store feature columns for later use
        if self.feature_columns is None:
            self.feature_columns = feature_cols
        
        X = df[feature_cols].fillna(0)
        y = df['result']
        
        logger.info(f"Features: {len(feature_cols)}")
        logger.info(f"Samples: {len(X)}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def train_model(self, league_name: str, test_size: float = 0.2, 
                   random_state: int = 42) -> dict:
        """
        Train a model for a specific league
        
        Args:
            league_name: Name of the league
            test_size: Proportion of data for testing
            random_state: Random seed
        
        Returns:
            Dictionary with training results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Training model for: {league_name.upper()}")
        logger.info(f"{'='*60}")
        
        # Load data
        df = self.load_league_data(league_name)
        if df is None or len(df) == 0:
            logger.error(f"No data available for {league_name}")
            return None
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Train Random Forest
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        logger.info("Training Random Forest...")
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"\n✓ Training complete!")
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        
        # Detailed metrics
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Home Win', 'Draw', 'Away Win']))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        logger.info(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features for {league_name}:")
        print(feature_importance.head(10).to_string(index=False))
        
        # Store model
        self.models[league_name] = model
        
        return {
            'model': model,
            'accuracy': accuracy,
            'cv_scores': cv_scores,
            'feature_importance': feature_importance,
            'samples': len(df)
        }
    
    def train_all_leagues(self) -> dict:
        """
        Train models for all leagues
        
        Returns:
            Dictionary with results for all leagues
        """
        results = {}
        
        print("\n" + "="*80)
        print("TRAINING LEAGUE-SPECIFIC MODELS")
        print("="*80)
        
        for league in self.leagues:
            result = self.train_model(league)
            if result:
                results[league] = result
        
        # Summary
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        
        for league, result in results.items():
            print(f"\n{league.upper()}:")
            print(f"  Samples: {result['samples']}")
            print(f"  Test Accuracy: {result['accuracy']:.4f}")
            print(f"  CV Accuracy: {result['cv_scores'].mean():.4f} (+/- {result['cv_scores'].std() * 2:.4f})")
        
        return results
    
    def save_models(self):
        """
        Save all trained models to disk
        """
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        for league_name, model in self.models.items():
            model_path = MODELS_DIR / f"{league_name}_model.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved {league_name} model to {model_path}")
        
        # Save feature columns
        feature_path = MODELS_DIR / "feature_columns.pkl"
        with open(feature_path, 'wb') as f:
            pickle.dump(self.feature_columns, f)
        logger.info(f"Saved feature columns to {feature_path}")
        
        print("\n✓ All models saved successfully!")
    
    def load_model(self, league_name: str):
        """
        Load a trained model for a specific league
        
        Args:
            league_name: Name of the league
        
        Returns:
            Trained model
        """
        model_path = MODELS_DIR / f"{league_name}_model.pkl"
        
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return None
        
        model = joblib.load(model_path)
        logger.info(f"Loaded {league_name} model from {model_path}")
        return model


def main():
    """
    Train league-specific models
    """
    trainer = LeagueSpecificTrainer()
    
    # Train all leagues
    results = trainer.train_all_leagues()
    
    if not results:
        print("\n❌ No models trained!")
        return
    
    # Save models
    trainer.save_models()
    
    print("\n" + "="*80)
    print("✓ COMPLETE!")
    print("="*80)
    print("\nLeague-specific models saved:")
    for league in results.keys():
        print(f"  - models/{league}_model.pkl")
    print("\nYou can now use these models to predict matches!")
    print("Each league will use its own specialized model.")


if __name__ == "__main__":
    main()