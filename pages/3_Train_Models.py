"""
Model Training Page - Train and compare ML models
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(str(Path(__file__).parent.parent))

from src.features.feature_engineering import FeatureEngineer
from src.models.ml_predictor import MLPredictor
from config import ML_FEATURES

st.set_page_config(page_title="Train Models", page_icon="🤖", layout="wide")

st.title("🤖 Model Training")
st.markdown("### Train and Evaluate Machine Learning Models")

# Check if data exists
data_dir = Path("data/raw")
data_files = list(data_dir.glob("*.json"))

if not data_files:
    st.error("⚠️ No data found! Please fetch data first.")
    if st.button("Go to Data Manager"):
        st.switch_page("pages/2_📊_Data_Manager.py")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["🎓 Train New Model", "📊 Compare Models", "⚙️ Advanced Settings"])

# ============================================================
# TAB 1: TRAIN NEW MODEL
# ============================================================
with tab1:
    st.subheader("🎓 Train New Model")
    
    # Parse available data
    available_data = {}
    for file in data_files:
        parts = file.stem.split('_')
        league = '_'.join(parts[:-1])
        season = int(parts[-1])
        
        if league not in available_data:
            available_data[league] = []
        available_data[league].append(season)
    
    # Model configuration
    st.markdown("### Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_name = st.text_input(
            "Model Name",
            value=f"model_{datetime.now().strftime('%Y%m%d_%H%M')}",
            help="Give your model a unique name"
        )
        
        model_type = st.selectbox(
            "Model Type",
            options=["random_forest", "gradient_boosting"],
            format_func=lambda x: "Random Forest" if x == "random_forest" else "Gradient Boosting",
            help="Random Forest is generally more stable, Gradient Boosting can be more accurate"
        )
    
    with col2:
        test_size = st.slider(
            "Test Set Size (%)",
            min_value=10,
            max_value=40,
            value=20,
            help="Percentage of data to use for testing"
        )
        
        cv_folds = st.slider(
            "Cross-Validation Folds",
            min_value=3,
            max_value=10,
            value=5,
            help="Number of folds for cross-validation"
        )
    
    # Training data selection
    st.markdown("### Training Data Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_leagues = st.multiselect(
            "Leagues",
            options=sorted(available_data.keys()),
            default=list(available_data.keys())[:2],
            format_func=lambda x: x.replace('_', ' ').title()
        )
    
    with col2:
        # Get all available seasons for selected leagues
        all_seasons = set()
        for league in selected_leagues:
            all_seasons.update(available_data[league])
        
        selected_seasons = st.multiselect(
            "Seasons",
            options=sorted(all_seasons, reverse=True),
            default=list(sorted(all_seasons, reverse=True))[:3]
        )
    
    # Estimate training samples
    if selected_leagues and selected_seasons:
        estimated_samples = 0
        for league in selected_leagues:
            for season in selected_seasons:
                if season in available_data[league]:
                    file_path = data_dir / f"{league}_{season}.json"
                    try:
                        import json
                        with open(file_path) as f:
                            data = json.load(f)
                            estimated_samples += len([f for f in data['fixtures'] if f['fixture']['status']['short'] == 'FT'])
                    except:
                        pass
        
        st.info(f"""
        **Estimated Training Samples:** ~{estimated_samples:,} matches
        
        **Training Time:** ~{estimated_samples * 0.001:.1f} minutes
        """)
    
    # Hyperparameters (expandable)
    with st.expander("🔧 Hyperparameters (Advanced)"):
        if model_type == "random_forest":
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_estimators = st.number_input("Number of Trees", min_value=50, max_value=500, value=200, step=50)
            with col2:
                max_depth = st.number_input("Max Depth", min_value=5, max_value=30, value=15, step=5)
            with col3:
                min_samples_split = st.number_input("Min Samples Split", min_value=2, max_value=20, value=10, step=2)
            
            hyperparams = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'random_state': 42,
                'n_jobs': -1,
                'class_weight': 'balanced'
            }
        else:  # gradient_boosting
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_estimators = st.number_input("Number of Estimators", min_value=50, max_value=500, value=200, step=50)
            with col2:
                max_depth = st.number_input("Max Depth", min_value=3, max_value=15, value=7, step=2)
            with col3:
                learning_rate = st.number_input("Learning Rate", min_value=0.01, max_value=0.3, value=0.05, step=0.01, format="%.2f")
            
            hyperparams = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'random_state': 42
            }
    
    st.markdown("---")
    
    # Train button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        train_button = st.button(
            "🚀 Start Training",
            type="primary",
            use_container_width=True,
            disabled=not (selected_leagues and selected_seasons)
        )
    
    if train_button:
        with st.spinner("🔄 Preparing training data..."):
            try:
                # Feature engineering
                engineer = FeatureEngineer()
                all_data = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total = len(selected_leagues) * len(selected_seasons)
                current = 0
                
                for league in selected_leagues:
                    for season in selected_seasons:
                        if season not in available_data[league]:
                            continue
                        
                        current += 1
                        status_text.text(f"Loading {league} {season}... ({current}/{total})")
                        
                        engineer.load_league_data(league, season)
                        df = engineer.create_training_dataset(league, season)
                        
                        if df is not None and len(df) > 0:
                            all_data.append(df)
                        
                        progress_bar.progress(current / total)
                
                if not all_data:
                    st.error("❌ No training data could be loaded!")
                    st.stop()
                
                # Combine data
                status_text.text("Combining datasets...")
                combined_df = pd.concat(all_data, ignore_index=True)
                
                st.success(f"✅ Loaded {len(combined_df):,} training samples")
                
                # Train model
                status_text.text("🤖 Training model...")
                
                predictor = MLPredictor()
                X, y = predictor.prepare_features(combined_df)
                
                # Split
                split_idx = int(len(X) * (1 - test_size/100))
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                
                st.info(f"""
                **Training set:** {len(X_train):,} matches
                **Test set:** {len(X_test):,} matches
                """)
                
                # Train
                predictor.train_model(X_train, y_train, model_type=model_type)
                
                # Evaluate
                status_text.text("📊 Evaluating model...")
                accuracy = predictor.evaluate_model(X_test, y_test)
                
                # Save model
                model_filename = f"{model_name}.pkl"
                predictor.save_model(model_filename)
                
                status_text.empty()
                progress_bar.empty()
                
                # Results
                st.markdown("---")
                st.success("✅ Training Complete!")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Test Accuracy", f"{accuracy*100:.2f}%")
                col2.metric("Model Type", model_type.replace('_', ' ').title())
                col3.metric("Training Samples", f"{len(X_train):,}")
                
                # Feature importance
                if hasattr(predictor.model, 'feature_importances_'):
                    st.markdown("---")
                    st.subheader("📊 Feature Importance")
                    
                    feature_importance = pd.DataFrame({
                        'feature': predictor.feature_columns,
                        'importance': predictor.model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    fig = px.bar(
                        feature_importance.head(15),
                        x='importance',
                        y='feature',
                        orientation='h',
                        title='Top 15 Most Important Features',
                        labels={'importance': 'Importance', 'feature': 'Feature'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                st.balloons()
            
            except Exception as e:
                st.error(f"❌ Training failed: {e}")
                st.exception(e)

# ============================================================
# TAB 2: COMPARE MODELS
# ============================================================
with tab2:
    st.subheader("📊 Compare Models")
    
    # Get all trained models
    model_dir = Path("models")
    model_files = list(model_dir.glob("*.pkl"))
    
    if not model_files:
        st.warning("⚠️ No trained models found. Train a model first!")
    else:
        st.success(f"Found {len(model_files)} trained models")
        
        # Model list
        model_info = []
        for model_file in model_files:
            size_mb = model_file.stat().st_size / (1024 * 1024)
            mod_time = datetime.fromtimestamp(model_file.stat().st_mtime)
            
            model_info.append({
                'Name': model_file.stem,
                'Size (MB)': f"{size_mb:.2f}",
                'Created': mod_time.strftime('%Y-%m-%d %H:%M'),
                'Path': str(model_file)
            })
        
        df_models = pd.DataFrame(model_info)
        
        st.dataframe(
            df_models[['Name', 'Size (MB)', 'Created']],
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Compare two models
        st.subheader("⚖️ Compare Two Models")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model1_name = st.selectbox(
                "Model 1",
                options=[m['Name'] for m in model_info],
                key="model1"
            )
        
        with col2:
            model2_name = st.selectbox(
                "Model 2",
                options=[m['Name'] for m in model_info],
                key="model2"
            )
        
        if st.button("🔍 Compare Models", use_container_width=True):
            if model1_name == model2_name:
                st.error("Please select two different models")
            else:
                with st.spinner("Loading and comparing models..."):
                    try:
                        # Load test data
                        engineer = FeatureEngineer()
                        
                        # Use most recent data for testing
                        test_league = list(available_data.keys())[0]
                        test_season = max(available_data[test_league])
                        
                        engineer.load_league_data(test_league, test_season)
                        df_test = engineer.create_training_dataset(test_league, test_season)
                        
                        if df_test is None or len(df_test) == 0:
                            st.error("No test data available")
                            st.stop()
                        
                        # Take last 20% as test
                        test_df = df_test.tail(int(len(df_test) * 0.2))
                        
                        # Load models
                        predictor1 = MLPredictor()
                        predictor1.load_model(f"{model1_name}.pkl")
                        
                        predictor2 = MLPredictor()
                        predictor2.load_model(f"{model2_name}.pkl")
                        
                        # Prepare features
                        X_test, y_test = predictor1.prepare_features(test_df)
                        
                        # Evaluate both
                        from sklearn.metrics import accuracy_score, classification_report
                        
                        # Model 1
                        X_test_scaled1 = predictor1.scaler.transform(X_test)
                        y_pred1 = predictor1.model.predict(X_test_scaled1)
                        acc1 = accuracy_score(y_test, y_pred1)
                        
                        # Model 2
                        X_test_scaled2 = predictor2.scaler.transform(X_test)
                        y_pred2 = predictor2.model.predict(X_test_scaled2)
                        acc2 = accuracy_score(y_test, y_pred2)
                        
                        # Display comparison
                        st.markdown("---")
                        st.subheader("📊 Comparison Results")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        col1.metric(
                            f"{model1_name}",
                            f"{acc1*100:.2f}%",
                            delta=f"{(acc1-acc2)*100:+.2f}%" if acc1 > acc2 else None
                        )
                        
                        col2.metric("Test Samples", len(test_df))
                        
                        col3.metric(
                            f"{model2_name}",
                            f"{acc2*100:.2f}%",
                            delta=f"{(acc2-acc1)*100:+.2f}%" if acc2 > acc1 else None
                        )
                        
                        # Winner
                        if acc1 > acc2:
                            st.success(f"🏆 **Winner:** {model1_name} (+{(acc1-acc2)*100:.2f}%)")
                        elif acc2 > acc1:
                            st.success(f"🏆 **Winner:** {model2_name} (+{(acc2-acc1)*100:.2f}%)")
                        else:
                            st.info("🤝 **Tie!** Both models perform equally")
                        
                        # Detailed metrics
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"### {model1_name}")
                            report1 = classification_report(
                                y_test, y_pred1,
                                target_names=['Home Win', 'Draw', 'Away Win'],
                                output_dict=True
                            )
                            st.dataframe(
                                pd.DataFrame(report1).transpose()[['precision', 'recall', 'f1-score']],
                                use_container_width=True
                            )
                        
                        with col2:
                            st.markdown(f"### {model2_name}")
                            report2 = classification_report(
                                y_test, y_pred2,
                                target_names=['Home Win', 'Draw', 'Away Win'],
                                output_dict=True
                            )
                            st.dataframe(
                                pd.DataFrame(report2).transpose()[['precision', 'recall', 'f1-score']],
                                use_container_width=True
                            )
                        
                        # Visualization
                        st.markdown("---")
                        st.subheader("📈 Performance Comparison")
                        
                        metrics_df = pd.DataFrame({
                            'Model': [model1_name, model2_name],
                            'Accuracy': [acc1, acc2],
                            'Home Win Recall': [
                                report1['Home Win']['recall'],
                                report2['Home Win']['recall']
                            ],
                            'Draw Recall': [
                                report1['Draw']['recall'],
                                report2['Draw']['recall']
                            ],
                            'Away Win Recall': [
                                report1['Away Win']['recall'],
                                report2['Away Win']['recall']
                            ]
                        })
                        
                        fig = go.Figure()
                        
                        for col in ['Accuracy', 'Home Win Recall', 'Draw Recall', 'Away Win Recall']:
                            fig.add_trace(go.Bar(
                                name=col,
                                x=metrics_df['Model'],
                                y=metrics_df[col],
                                text=[f"{v*100:.1f}%" for v in metrics_df[col]],
                                textposition='auto',
                            ))
                        
                        fig.update_layout(
                            title='Model Performance Metrics',
                            yaxis_title='Score',
                            barmode='group'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"Error comparing models: {e}")
                        st.exception(e)
        
        # Delete models
        st.markdown("---")
        st.subheader("🗑️ Delete Models")
        
        models_to_delete = st.multiselect(
            "Select models to delete",
            options=[m['Name'] for m in model_info]
        )
        
        if models_to_delete:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🗑️ Delete Selected Models", type="primary", use_container_width=True):
                    for model_name in models_to_delete:
                        try:
                            model_path = model_dir / f"{model_name}.pkl"
                            model_path.unlink()
                            st.success(f"✅ Deleted: {model_name}")
                        except Exception as e:
                            st.error(f"❌ Error deleting {model_name}: {e}")
                    
                    st.rerun()

# ============================================================
# TAB 3: ADVANCED SETTINGS
# ============================================================
with tab3:
    st.subheader("⚙️ Advanced Settings")
    
    # Feature selection
    st.markdown("### Feature Configuration")
    
    st.info(f"""
    **Current Features:** {len(ML_FEATURES)}
    
    Features used for training models. Modify in `config/config.py`.
    """)
    
    with st.expander("📋 View Current Features"):
        feature_df = pd.DataFrame({
            'Feature': ML_FEATURES,
            'Category': [
                'Form' if 'form' in f else
                'Goals' if 'goals' in f else
                'Position' if 'position' in f else
                'H2H' if 'h2h' in f else
                'Rest' if 'days' in f else
                'Other'
                for f in ML_FEATURES
            ]
        })
        st.dataframe(feature_df, use_container_width=True)
    
    st.markdown("---")
    
    # Model recommendations
    st.markdown("### 💡 Model Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **Random Forest**
        
        ✅ Generally more stable
        ✅ Less prone to overfitting
        ✅ Good for imbalanced data
        ✅ Faster training
        
        **Best for:** Beginners, balanced accuracy
        """)
    
    with col2:
        st.info("""
        **Gradient Boosting**
        
        ✅ Can achieve higher accuracy
        ✅ Better with complex patterns
        ⚠️ More sensitive to hyperparameters
        ⚠️ Slower training
        
        **Best for:** Advanced users, maximum accuracy
        """)
    
    st.markdown("---")
    
    # Training tips
    st.markdown("### 📚 Training Tips")
    
    st.markdown("""
    1. **Data Quality Over Quantity**
       - 3 years of recent data is better than 10 years of old data
       - Ensure data is complete (no missing fixtures)
    
    2. **Feature Engineering**
       - Form-based features (last 5 matches) are most important
       - Head-to-head history helps for derbies
       - Rest days can indicate fatigue
    
    3. **Hyperparameter Tuning**
       - Start with defaults, then optimize
       - Higher `max_depth` = more complex model (risk overfitting)
       - More `n_estimators` = better but slower
    
    4. **Model Evaluation**
       - Don't trust accuracy alone
       - Check precision/recall for each outcome
       - Test on recent season data
    
    5. **Realistic Expectations**
       - 48-52% accuracy is competitive
       - Draws are very hard to predict (20-30%)
       - No model is perfect - football is unpredictable!
    """)
    
    st.markdown("---")
    
    # Export training log
    st.markdown("### 📝 Training Logs")
    
    log_dir = Path("logs")
    log_files = list(log_dir.glob("ml_predictor_*.log"))
    
    if log_files:
        latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_log, 'r') as f:
            log_content = f.read()
        
        st.download_button(
            label="💾 Download Latest Training Log",
            data=log_content,
            file_name=latest_log.name,
            mime="text/plain",
            use_container_width=True
        )
    else:
        st.info("No training logs available yet")