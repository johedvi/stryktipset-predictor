"""
Machine Learning predictor for match outcomes
Uses trained models to predict 1/X/2 results
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    ML_TEST_SIZE,
    ML_RANDOM_STATE,
    ML_CV_FOLDS,
    ML_FEATURES,
)
from utils import setup_logging, calculate_prediction_accuracy

logger = setup_logging(__name__)


class MLPredictor:
    """
    Machine learning predictor for football matches
    """
    
    def __init__(self):
        """Initialize ML predictor"""
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.label_mapping = {'H': 0, 'D': 1, 'A': 2}
        self.inverse_label_mapping = {0: 'H', 1: 'D', 2: 'A'}
        logger.info("MLPredictor initialized")
    
    def load_training_data(self, filename: str) -> pd.DataFrame:
        """
        Load training data from CSV
        
        Args:
            filename: CSV filename in processed data directory
        
        Returns:
            DataFrame with training data
        """
        filepath = PROCESSED_DATA_DIR / filename
        
        if not filepath.exists():
            logger.error(f"Training data not found: {filepath}")
            return None
        
        df = pd.read_csv(filepath)
        logger.info(f"Loaded training data: {len(df)} matches from {filepath}")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and labels for training
        
        Args:
            df: Raw DataFrame
        
        Returns:
            Tuple of (features DataFrame, labels Series)
        """
        # Use only the specified features
        available_features = [f for f in ML_FEATURES if f in df.columns]
        
        if len(available_features) < len(ML_FEATURES):
            missing = set(ML_FEATURES) - set(available_features)
            logger.warning(f"Missing features: {missing}")
        
        X = df[available_features].copy()
        y = df['result'].map(self.label_mapping)
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        
        self.feature_columns = available_features
        
        logger.info(f"Prepared {len(X)} samples with {len(available_features)} features")
        
        return X, y
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                   model_type: str = 'random_forest'):
        """
        Train the machine learning model
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_type: Type of model ('random_forest' or 'gradient_boosting')
        """
        logger.info(f"Training {model_type} model...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Initialize model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=ML_RANDOM_STATE,
                n_jobs=-1,
                class_weight='balanced',
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.1,
                random_state=ML_RANDOM_STATE,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, 
            cv=ML_CV_FOLDS, scoring='accuracy'
        )
        
        logger.info(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(feature_importance.head(10).to_string(index=False))
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
        """
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\n" + "="*70)
        print("MODEL EVALUATION")
        print("="*70)
        print(f"\nOverall Accuracy: {accuracy:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=['Home Win', 'Draw', 'Away Win']
        ))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Home Win', 'Draw', 'Away Win'],
                   yticklabels=['Home Win', 'Draw', 'Away Win'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(MODELS_DIR / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        logger.info("Saved confusion matrix plot")
        plt.show()
        
        return accuracy
    
    def predict_proba(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Predict match outcome probabilities
        
        Args:
            features: Dictionary of feature values
        
        Returns:
            Dictionary with probabilities for H, D, A
        """
        if self.model is None:
            logger.error("Model not trained yet")
            return None
        
        # Create feature vector
        feature_vector = []
        for feat in self.feature_columns:
            feature_vector.append(features.get(feat, 0))
        
        # Reshape and scale
        X = np.array(feature_vector).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Get probabilities
        proba = self.model.predict_proba(X_scaled)[0]
        
        return {
            'H': proba[0],
            'D': proba[1],
            'A': proba[2],
        }
    
    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict match outcome
        
        Args:
            features: Dictionary of feature values
        
        Returns:
            Prediction dictionary
        """
        probas = self.predict_proba(features)
        
        if probas is None:
            return None
        
        # Get prediction (highest probability)
        prediction = max(probas, key=probas.get)
        confidence = probas[prediction]
        
        prediction_text = {
            'H': 'Home Win',
            'D': 'Draw',
            'A': 'Away Win'
        }[prediction]
        
        return {
            'prediction': prediction,
            'prediction_text': prediction_text,
            'confidence': confidence,
            'probabilities': probas,
        }
    
    def save_model(self, filename: str = 'ml_model.pkl'):
        """
        Save trained model to file
        
        Args:
            filename: Output filename
        """
        if self.model is None:
            logger.error("No model to save")
            return
        
        model_path = MODELS_DIR / filename
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'label_mapping': self.label_mapping,
            'inverse_label_mapping': self.inverse_label_mapping,
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Saved model to {model_path}")
    
    def load_model(self, filename: str = 'ml_model.pkl'):
        """
        Load trained model from file
        
        Args:
            filename: Model filename
        """
        model_path = MODELS_DIR / filename
        
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return False
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.label_mapping = model_data['label_mapping']
        self.inverse_label_mapping = model_data['inverse_label_mapping']
        
        logger.info(f"Loaded model from {model_path}")
        return True
    
    def plot_feature_importance(self):
        """Plot feature importance"""
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            logger.warning("Model doesn't have feature importances")
            return
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(feature_importance)), feature_importance['importance'])
        plt.yticks(range(len(feature_importance)), feature_importance['feature'])
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(MODELS_DIR / 'feature_importance.png', dpi=300, bbox_inches='tight')
        logger.info("Saved feature importance plot")
        plt.show()


def main():
    """
    Train model on 5 years of data
    """
    predictor = MLPredictor()
    
    # Load the big dataset
    print("Loading training data...")
    df = predictor.load_training_data('full_training_5years.csv')
    
    if df is None:
        print("\n⚠️  No training data found!")
        print("Run: python feature_engineering.py")
        return
    
    print(f"Total matches: {len(df)}")
    print(f"Leagues: {df['league'].unique()}")
    print(f"Seasons: {sorted(df['season'].unique())}")
    
    # Prepare features
    print("\nPreparing features...")
    X, y = predictor.prepare_features(df)
    
    # Temporal split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"\nTraining set: {len(X_train)} matches")
    print(f"Test set: {len(X_test)} matches")
    
    # Train Gradient Boosting
    print("\n" + "="*80)
    print("TRAINING RF BOOSTING MODEL")
    print("="*80 + "\n")
    
    predictor.train_model(X_train, y_train, model_type='random_forest')
    
    # Evaluate
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)
    
    accuracy = predictor.evaluate_model(X_test, y_test)
    
    # Feature importance
    predictor.plot_feature_importance()
    
    # Save
    print("\nSaving model...")
    predictor.save_model('RF_boosting_5years.pkl')
    
    print(f"\n✓ Training complete!")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Model: models/RF_boosting_5years.pkl")


if __name__ == "__main__":
    main()