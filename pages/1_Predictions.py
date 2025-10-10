"""
Match Predictions Page
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from main import StryktipsetPredictor
from utils.team_mapper import get_all_teams, get_teams_by_league
from src.prediction.predict_with_strategy import create_stryktipset_coupon, calculate_expected_combinations

st.set_page_config(page_title="Predictions", page_icon="🎯", layout="wide")

# Team IDs database
# Replace the TEAMS dict with:
ALL_TEAMS = get_all_teams()
TEAMS = {name: info['id'] for name, info in ALL_TEAMS.items()}

# For team selection by league:
def get_team_options_by_league():
    pl_teams = get_teams_by_league("Premier League")
    ch_teams = get_teams_by_league("Championship")
    l1_teams = get_teams_by_league("League One")
    l2_teams = get_teams_by_league("League Two")
    
    return {
        "Premier League": pl_teams,
        "Championship": ch_teams,
        "League One": l1_teams,
        "League Two": l2_teams
    }

# Then in the UI, you can filter:
league_filter = st.selectbox("Filter by League", ["All", "Premier League", "Championship", "League One", "League Two"])

if league_filter == "All":
    available_teams = sorted(TEAMS.keys())
else:
    available_teams = sorted(get_teams_by_league(league_filter).keys())

LEAGUE_MAP = {
    "Premier League": "premier_league",
    "Championship": "championship",
    "League One": "league_one",
    "League Two": "league_two"
}

st.title("🎯 Match Predictions")
st.markdown("### Generate Your Stryktipset Coupon")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Check if models exist
model_files = list(Path("models").glob("*.pkl"))
if not model_files:
    st.error("⚠️ No trained models found! Please train a model first.")
    if st.button("Go to Model Training"):
        st.switch_page("pages/3_🤖_Train_Models.py")
    st.stop()

# Model selection
model_name = st.sidebar.selectbox(
    "Select Model",
    [m.stem for m in model_files]
)

model_type_display = st.sidebar.selectbox(
    "Prediction Method",
    ["Ensemble (ML + Rules)", "Machine Learning Only", "Rule-Based Only"]
)

use_ml = model_type_display != "Rule-Based Only"

# Data to load
leagues_to_load = st.sidebar.multiselect(
    "Leagues for Historical Context",
    ["Premier League", "Championship", "League One", "League Two"],
    default=["Premier League", "Championship"]
)

seasons_to_load = st.sidebar.multiselect(
    "Seasons to Load",
    [2020, 2021, 2022, 2023, 2024, 2025],
    default=[2023, 2024]
)

strategy = st.sidebar.selectbox(
    "Betting Strategy",
    ["aggressive", "balanced", "safe"],
    index=1
)

st.sidebar.markdown("---")
st.sidebar.info(f"""
**Strategy: {strategy.title()}**

{
    "Lower thresholds, more single signs, higher risk" if strategy == "aggressive" else
    "Balanced approach, recommended for most users" if strategy == "balanced" else
    "Higher thresholds, more coverage, lower risk"
}
""")

# Main content
tab1, tab2 = st.tabs(["📝 Build Coupon", "💡 Tips"])

with tab1:
    # Number of matches
    num_matches = st.number_input(
        "Number of matches in coupon",
        min_value=1,
        max_value=13,
        value=13,
        help="Standard Stryktipset has 13 matches"
    )
    
    # Initialize session state for matches
    if 'matches' not in st.session_state:
        st.session_state.matches = []
    
    # Match inputs
    st.subheader("Select Teams")
    
    matches = []
    for i in range(num_matches):
        with st.expander(f"⚽ Match {i+1}", expanded=i < 3):
            col1, col2, col3 = st.columns([5, 5, 2])
            
            with col1:
                home_team = st.selectbox(
                    "Home Team",
                    options=[""] + sorted(TEAMS.keys()),
                    key=f"home_{i}",
                    label_visibility="collapsed"
                )
            
            with col2:
                away_team = st.selectbox(
                    "Away Team",
                    options=[""] + sorted(TEAMS.keys()),
                    key=f"away_{i}",
                    label_visibility="collapsed"
                )
            
            with col3:
                if home_team and away_team:
                    st.success("✓")
                else:
                    st.error("✗")
            
            if home_team and away_team and home_team != away_team:
                matches.append({
                    "home": home_team,
                    "away": away_team,
                    "home_id": TEAMS[home_team],
                    "away_id": TEAMS[away_team]
                })
    
    st.markdown("---")
    
    # Predict button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button(
            "🎯 Generate Predictions",
            type="primary",
            use_container_width=True,
            disabled=len(matches) == 0
        )
    
    if predict_button:
        if len(matches) < num_matches:
            st.warning(f"⚠️ Please select teams for all {num_matches} matches")
        else:
            # Initialize predictor
            with st.spinner("🔄 Loading predictor..."):
                try:
                    predictor = StryktipsetPredictor(use_ml=use_ml)
                    
                    # Load historical data
                    for league in leagues_to_load:
                        league_key = LEAGUE_MAP[league]
                        predictor.load_historical_data(league_key, seasons_to_load)
                    
                    st.success("✅ Predictor loaded!")
                except Exception as e:
                    st.error(f"❌ Error loading predictor: {e}")
                    st.stop()
            
            # Make predictions
            predictions = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, match in enumerate(matches):
                status_text.text(f"Predicting {match['home']} vs {match['away']}...")
                
                try:
                    # Determine league
                    home_team = match['home']
                    if home_team in list(TEAMS.keys())[:20]:  # First 20 are PL
                        league_name = "premier_league"
                    elif home_team in list(TEAMS.keys())[20:44]:  # Next 24 are Championship
                        league_name = "championship"
                    else:
                        league_name = "league_one"
                    
                    pred = predictor.predict_match(
                        home_id=match['home_id'],
                        away_id=match['away_id'],
                        home_name=match['home'],
                        away_name=match['away'],
                        league_name=league_name,
                        season=2024,
                        match_date=datetime.now().strftime('%Y-%m-%d')
                    )
                    
                    predictions.append(pred)
                except Exception as e:
                    st.warning(f"⚠️ Could not predict {match['home']} vs {match['away']}: {e}")
                    predictions.append({
                        'home_team': match['home'],
                        'away_team': match['away'],
                        'final_prediction': None
                    })
                
                progress_bar.progress((idx + 1) / len(matches))
            
            status_text.empty()
            progress_bar.empty()
            
            # Create coupon
            coupon = create_stryktipset_coupon(predictions, strategy=strategy)
            stats = calculate_expected_combinations(coupon)
            
            # Display results
            st.success("✅ Predictions Complete!")
            st.markdown("---")
            
            # Summary metrics
            st.subheader("📊 Coupon Summary")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Single Signs (1/X/2)", stats['single_signs'], help="High confidence picks")
            col2.metric("Double Signs (1X/12/X2)", stats['double_signs'], help="Medium confidence")
            col3.metric("Triple Signs (1X2)", stats['triple_signs'], help="Low confidence")
            col4.metric("💰 Total Cost", f"{stats['cost_sek']:,} SEK")
            
            # Warning if cost is high
            if stats['cost_sek'] > 10000:
                st.warning(f"⚠️ High cost! Consider using 'aggressive' strategy or picking only confident matches.")
            
            st.markdown("---")
            
            # Detailed predictions
            st.subheader("📋 Your Stryktipset Coupon")
            
            for entry in coupon:
                with st.container():
                    col1, col2, col3, col4 = st.columns([4, 2, 2, 2])
                    
                    with col1:
                        st.markdown(f"**{entry['match_num']}. {entry['match']}**")
                    
                    with col2:
                        sign = entry['signs']
                        if len(sign) == 1:
                            st.success(f"**Sign: {sign}**")
                        elif len(sign) == 2:
                            st.warning(f"**Sign: {sign}**")
                        else:
                            st.error(f"**Sign: {sign}**")
                    
                    with col3:
                        conf = entry.get('confidence', 0)
                        if conf > 0:
                            st.metric("Confidence", f"{conf*100:.0f}%")
                    
                    with col4:
                        strat = entry.get('strategy', 'unknown')
                        emoji = "🎯" if strat == "high_confidence" else "⚠️" if strat == "medium_confidence" else "❓"
                        st.caption(f"{emoji} {strat.replace('_', ' ').title()}")
                    
                    # Show probabilities
                    if 'probabilities' in entry and entry['probabilities']:
                        probs = entry['probabilities']
                        prob_col1, prob_col2, prob_col3 = st.columns(3)
                        prob_col1.progress(probs.get('H', 0), text=f"1: {probs.get('H', 0)*100:.0f}%")
                        prob_col2.progress(probs.get('D', 0), text=f"X: {probs.get('D', 0)*100:.0f}%")
                        prob_col3.progress(probs.get('A', 0), text=f"2: {probs.get('A', 0)*100:.0f}%")
                    
                    st.divider()
            
            # Export
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                # Create export text
                export_text = "STRYKTIPSET COUPON\n"
                export_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                export_text += f"Strategy: {strategy.title()}\n"
                export_text += "="*60 + "\n\n"
                
                for entry in coupon:
                    export_text += f"{entry['match_num']:2d}. {entry['match']:40s} [{entry['signs']:3s}]\n"
                    if 'probabilities' in entry:
                        probs = entry['probabilities']
                        export_text += f"     1:{probs.get('H', 0)*100:3.0f}% | X:{probs.get('D', 0)*100:3.0f}% | 2:{probs.get('A', 0)*100:3.0f}%\n"
                    export_text += "\n"
                
                export_text += f"\n{'='*60}\n"
                export_text += f"Total Cost: {stats['cost_sek']:,} SEK\n"
                export_text += f"Combinations: {stats['total_combinations']:,}\n"
                
                st.download_button(
                    label="💾 Download as Text",
                    data=export_text,
                    file_name=f"stryktipset_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col2:
                # Create CSV export
                df_data = []
                for entry in coupon:
                    df_data.append({
                        'Match': entry['match_num'],
                        'Home': entry['match'].split(' vs ')[0],
                        'Away': entry['match'].split(' vs ')[1],
                        'Sign': entry['signs'],
                        'Confidence': f"{entry.get('confidence', 0)*100:.1f}%"
                    })
                
                df = pd.DataFrame(df_data)
                csv = df.to_csv(index=False)
                
                st.download_button(
                    label="📊 Download as CSV",
                    data=csv,
                    file_name=f"stryktipset_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

with tab2:
    st.subheader("💡 Tips for Better Predictions")
    
    st.markdown("""
    ### Understanding Signs
    - **1** = Home win (high confidence in home team)
    - **X** = Draw (rarely used, draws are hard to predict)
    - **2** = Away win (high confidence in away team)
    - **1X** = Home win OR draw (medium confidence)
    - **12** = Home OR away win (confident no draw)
    - **X2** = Draw OR away win (medium confidence)
    - **1X2** = Any outcome (very low confidence - avoid if possible)
    
    ### Strategy Guide
    
    **Aggressive** 🔥
    - More single signs
    - Lower cost (50-500 SEK)
    - Higher risk, higher reward
    - Best for: Experienced bettors
    
    **Balanced** ⚖️
    - Mix of single and double signs
    - Medium cost (500-5,000 SEK)
    - Best for: Most users
    - Recommended default
    
    **Safe** 🛡️
    - More double/triple signs
    - Higher cost (5,000-50,000 SEK)
    - Lower risk, more coverage
    - Best for: Conservative approach
    
    ### Best Practices
    1. ✅ Only bet on matches with >45% confidence
    2. ✅ Check team news and injuries before finalizing
    3. ✅ Compare with bookmaker odds for validation
    4. ✅ Don't chase losses - set a budget
    5. ✅ Use historical data to validate predictions
    
    ### Common Mistakes to Avoid
    - ❌ Using 1X2 on too many matches (cost explodes)
    - ❌ Ignoring low confidence warnings
    - ❌ Betting more than you can afford to lose
    - ❌ Not considering recent form changes
    - ❌ Trusting predictions blindly without research
    """)