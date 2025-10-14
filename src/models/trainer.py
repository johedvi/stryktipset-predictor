"""
Enhanced League-Specific Model Trainer
- Trains separate models for each league
- Automatic feature importance visualization
- Hyperparameter tuning
- Cross-validation with multiple metrics
- Saves performance reports
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from config import PROCESSED_DATA_DIR, MODELS_DIR
from utils.utils import setup_logging

logger = setup_logging(__name__)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


class EnhancedLeagueSpecificTrainer:
    """
    Enhanced trainer with visualization and hyperparameter tuning
    """
    
    def __init__(self):
        """Initialize the trainer"""
        self.models = {}
        self.feature_columns = None
        self.leagues = ['premier_league', 'championship', 'league_one', 'league_two']
        self.training_results = {}
        logger.info("EnhancedLeagueSpecificTrainer initialized")
    
    def load_league_data(self, league_name: str, enhanced: bool = True) -> pd.DataFrame:
        """
        Load training data for a specific league
        
        Args:
            league_name: Name of the league
            enhanced: Whether to load enhanced features dataset
        
        Returns:
            DataFrame with training data
        """
        if enhanced:
            filename = PROCESSED_DATA_DIR / f"{league_name}_training_enhanced_6years.csv"
        else:
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
    
    def plot_feature_importance(self, model, league_name: str, top_n: int = 30):
        """
        Create and save feature importance visualization
        
        Args:
            model: Trained model
            league_name: Name of the league
            top_n: Number of top features to show
        """
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model doesn't have feature importances")
            return
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
        fig.suptitle(f'{league_name.replace("_", " ").title()} - Feature Importance Analysis', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        # === PLOT 1: Top Features Bar Chart ===
        top_features = importance_df.head(top_n)
        colors = sns.color_palette("viridis", len(top_features))
        
        ax1.barh(range(len(top_features)), top_features['importance'], color=colors)
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features['feature'], fontsize=9)
        ax1.invert_yaxis()
        ax1.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax1.set_title(f'Top {top_n} Most Important Features', fontsize=13, fontweight='bold', pad=15)
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (idx, row) in enumerate(top_features.iterrows()):
            ax1.text(row['importance'] + 0.0005, i, f"{row['importance']:.4f}", 
                    va='center', fontsize=8)
        
        # === PLOT 2: Cumulative Importance ===
        cumulative_importance = np.cumsum(importance_df['importance'].values)
        
        ax2.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 
                linewidth=3, color='#2E86AB', label='Cumulative Importance')
        ax2.fill_between(range(1, len(cumulative_importance) + 1), cumulative_importance,
                        alpha=0.3, color='#2E86AB')
        
        # Add reference lines
        ax2.axhline(y=0.95, color='red', linestyle='--', linewidth=2, 
                   label='95% Importance', alpha=0.7)
        ax2.axhline(y=0.90, color='orange', linestyle='--', linewidth=2, 
                   label='90% Importance', alpha=0.7)
        
        # Find number of features for 95% importance
        features_95 = np.argmax(cumulative_importance >= 0.95) + 1
        ax2.axvline(x=features_95, color='red', linestyle=':', alpha=0.5)
        ax2.text(features_95, 0.5, f'{features_95} features\nfor 95%', 
                ha='left', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax2.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Cumulative Importance', fontsize=12, fontweight='bold')
        ax2.set_title('Cumulative Feature Importance', fontsize=13, fontweight='bold', pad=15)
        ax2.legend(loc='lower right', fontsize=10)
        ax2.grid(alpha=0.3)
        ax2.set_xlim(0, len(cumulative_importance))
        ax2.set_ylim(0, 1.05)
        
        # Save plot
        plt.tight_layout()
        plot_path = MODELS_DIR / f'{league_name}_feature_importance.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {plot_path}")
        plt.close()
        
        # Save importance data to CSV
        csv_path = MODELS_DIR / f'{league_name}_feature_importance.csv'
        importance_df.to_csv(csv_path, index=False)
        logger.info(f"Saved feature importance data to {csv_path}")
    
    def plot_confusion_matrix(self, y_true, y_pred, league_name: str):
        """
        Create and save confusion matrix visualization
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            league_name: Name of the league
        """
        cm = confusion_matrix(y_true, y_pred, labels=['H', 'D', 'A'])
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'{league_name.replace("_", " ").title()} - Confusion Matrix', 
                     fontsize=14, fontweight='bold')
        
        # Plot 1: Counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['Home Win', 'Draw', 'Away Win'],
                   yticklabels=['Home Win', 'Draw', 'Away Win'],
                   cbar_kws={'label': 'Count'})
        ax1.set_title('Prediction Counts', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Actual', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Predicted', fontsize=11, fontweight='bold')
        
        # Plot 2: Percentages
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Greens', ax=ax2,
                   xticklabels=['Home Win', 'Draw', 'Away Win'],
                   yticklabels=['Home Win', 'Draw', 'Away Win'],
                   cbar_kws={'label': 'Percentage'})
        ax2.set_title('Prediction Accuracy by Class (%)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Actual', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Predicted', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plot_path = MODELS_DIR / f'{league_name}_confusion_matrix.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {plot_path}")
        plt.close()
    
    def tune_hyperparameters(self, X_train, y_train, model_type: str = 'random_forest'):
        """
        Tune hyperparameters using GridSearchCV
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_type: Type of model to tune
        
        Returns:
            Best model
        """
        logger.info(f"Tuning hyperparameters for {model_type}...")
        
        if model_type == 'random_forest':
            param_grid = {
                'n_estimators': [150, 200, 250],
                'max_depth': [12, 15, 18],
                'min_samples_split': [8, 10, 12],
                'min_samples_leaf': [3, 5, 7],
                'class_weight': ['balanced', 'balanced_subsample']
            }
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        else:  # gradient_boosting
            param_grid = {
                'n_estimators': [150, 200, 250],
                'max_depth': [5, 7, 9],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9, 1.0]
            }
            base_model = GradientBoostingClassifier(random_state=42)
        
        # Use smaller param grid for speed
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=3,  # 3-fold CV for speed
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def train_model(self, league_name: str, test_size: float = 0.2, 
                   random_state: int = 42, tune_hyperparameters: bool = False,
                   model_type: str = 'random_forest') -> dict:
        """
        Train a model for a specific league
        
        Args:
            league_name: Name of the league
            test_size: Proportion of data for testing
            random_state: Random seed
            tune_hyperparameters: Whether to tune hyperparameters
            model_type: Type of model
        
        Returns:
            Dictionary with training results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Training model for: {league_name.upper()}")
        logger.info(f"{'='*60}")
        
        # Load data
        df = self.load_league_data(league_name, enhanced=True)
        if df is None or len(df) == 0:
            logger.error(f"No data available for {league_name}")
            return None
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Split data (temporal split - most recent 20% as test)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Train model
        if tune_hyperparameters:
            model = self.tune_hyperparameters(X_train, y_train, model_type)
        else:
            # Use good default parameters
            if model_type == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=random_state,
                    n_jobs=-1,
                    class_weight='balanced'
                )
            else:  # gradient_boosting
                model = GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=7,
                    learning_rate=0.1,
                    random_state=random_state
                )
        
        logger.info("Training model...")
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        logger.info(f"\n✓ Training complete!")
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Weighted F1: {f1:.4f}")
        
        # Detailed metrics
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Home Win', 'Draw', 'Away Win']))
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        logger.info(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Create visualizations
        self.plot_feature_importance(model, league_name)
        self.plot_confusion_matrix(y_test, y_pred, league_name)
        
        # Feature importance summary
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 15 Most Important Features for {league_name}:")
        print(feature_importance.head(15).to_string(index=False))
        
        # Store model
        self.models[league_name] = model
        
        return {
            'model': model,
            'accuracy': accuracy,
            'f1_score': f1,
            'cv_scores': cv_scores,
            'feature_importance': feature_importance,
            'samples': len(df),
            'test_samples': len(X_test)
        }
    
    def train_all_leagues(self, tune: bool = False) -> dict:
        """
        Train models for all leagues
        
        Args:
            tune: Whether to tune hyperparameters
        
        Returns:
            Dictionary with results for all leagues
        """
        results = {}
        
        print("\n" + "="*80)
        print("TRAINING LEAGUE-SPECIFIC MODELS WITH ENHANCED FEATURES")
        print("="*80)
        
        for league in self.leagues:
            result = self.train_model(league, tune_hyperparameters=tune)
            if result:
                results[league] = result
                self.training_results[league] = {
                    'accuracy': result['accuracy'],
                    'f1_score': result['f1_score'],
                    'cv_mean': result['cv_scores'].mean(),
                    'cv_std': result['cv_scores'].std()
                }
        
        # Summary
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        
        summary_df = pd.DataFrame(self.training_results).T
        summary_df = summary_df.round(4)
        print("\n" + summary_df.to_string())
        
        # Overall statistics
        print(f"\n{'='*80}")
        print(f"Average Accuracy: {summary_df['accuracy'].mean():.4f}")
        print(f"Average F1 Score: {summary_df['f1_score'].mean():.4f}")
        print(f"Best League: {summary_df['accuracy'].idxmax()} ({summary_df['accuracy'].max():.4f})")
        print(f"{'='*80}")
        
        # Save summary
        summary_path = MODELS_DIR / f'training_summary_{datetime.now().strftime("%Y%m%d_%H%M")}.csv'
        summary_df.to_csv(summary_path)
        logger.info(f"Saved training summary to {summary_path}")
        
        return results
    
    def save_models(self):
        """Save all trained models to disk"""
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


def main():
    """
    Train enhanced league-specific models
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Train enhanced league-specific models')
    parser.add_argument('--tune', action='store_true', help='Tune hyperparameters (slower)')
    parser.add_argument('--league', type=str, help='Train specific league only')
    args = parser.parse_args()
    
    trainer = EnhancedLeagueSpecificTrainer()
    
    if args.league:
        # Train single league
        result = trainer.train_model(args.league, tune_hyperparameters=args.tune)
        if result:
            trainer.save_models()
    else:
        # Train all leagues
        results = trainer.train_all_leagues(tune=args.tune)
        
        if results:
            trainer.save_models()
            
            print("\n" + "="*80)
            print("✓ TRAINING COMPLETE!")
            print("="*80)
            print("\nGenerated files:")
            print("  Models:")
            for league in results.keys():
                print(f"    ✓ models/{league}_model.pkl")
            print("  Visualizations:")
            for league in results.keys():
                print(f"    ✓ models/{league}_feature_importance.png")
                print(f"    ✓ models/{league}_confusion_matrix.png")
            print("  Data:")
            for league in results.keys():
                print(f"    ✓ models/{league}_feature_importance.csv")
            print(f"    ✓ models/training_summary_[timestamp].csv")
            print("="*80 + "\n")


if __name__ == "__main__":
    main()