"""
Analytics Dashboard - Advanced data visualization
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import json

sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_explorer import DataExplorer

st.set_page_config(page_title="Analytics", page_icon="📈", layout="wide")

st.title("📈 Analytics Dashboard")
st.markdown("### Advanced Data Visualization and Insights")

# Check data
data_dir = Path("data/raw")
data_files = list(data_dir.glob("*.json"))

if not data_files:
    st.error("⚠️ No data available. Fetch data first!")
    if st.button("Go to Data Manager"):
        st.switch_page("pages/2_📊_Data_Manager.py")
    st.stop()

# Parse available data
available_data = {}
for file in data_files:
    parts = file.stem.split('_')
    league = '_'.join(parts[:-1])
    season = int(parts[-1])
    
    if league not in available_data:
        available_data[league] = []
    available_data[league].append(season)

# Sidebar filters
st.sidebar.header("🔍 Filters")

selected_leagues = st.sidebar.multiselect(
    "Leagues",
    options=sorted(available_data.keys()),
    default=list(available_data.keys())[:2],
    format_func=lambda x: x.replace('_', ' ').title()
)

if selected_leagues:
    all_seasons = set()
    for league in selected_leagues:
        all_seasons.update(available_data[league])
    
    selected_seasons = st.sidebar.multiselect(
        "Seasons",
        options=sorted(all_seasons, reverse=True),
        default=[max(all_seasons)]
    )
else:
    selected_seasons = []

# Main content
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Home Advantage",
    "🎯 Form Analysis",
    "⚔️ Head-to-Head",
    "📊 League Comparison"
])

if not selected_leagues or not selected_seasons:
    st.warning("⚠️ Please select leagues and seasons from the sidebar")
    st.stop()

# Load data
@st.cache_data
def load_explorer_data(leagues, seasons):
    explorer = DataExplorer()
    explorer.load_all_data(leagues=leagues, seasons=seasons)
    return explorer

with st.spinner("Loading data..."):
    explorer = load_explorer_data(selected_leagues, selected_seasons)

# ============================================================
# TAB 1: HOME ADVANTAGE
# ============================================================
with tab1:
    st.subheader("🏠 Home Advantage Analysis")
    
    for league in selected_leagues:
        for season in selected_seasons:
            if league not in explorer.data or season not in explorer.data[league]:
                continue
            
            st.markdown(f"### {league.replace('_', ' ').title()} - {season}")
            
            data = explorer.data[league][season]
            df = explorer.fixtures_to_dataframe(data['fixtures'])
            
            if df.empty:
                st.warning("No finished matches")
                continue
            
            home_adv = explorer.analyze_home_advantage(df)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Home Wins", f"{home_adv['home_win_pct']:.1f}%")
            col2.metric("Draws", f"{home_adv['draw_pct']:.1f}%")
            col3.metric("Away Wins", f"{home_adv['away_win_pct']:.1f}%")
            col4.metric("Matches", home_adv['total_matches'])
            
            # Visualization
            fig = go.Figure(data=[
                go.Bar(name='Home Goals', x=['Average'], y=[home_adv['home_goals_avg']], marker_color='#2ecc71'),
                go.Bar(name='Away Goals', x=['Average'], y=[home_adv['away_goals_avg']], marker_color='#e74c3c')
            ])
            fig.update_layout(title='Goals per Match', barmode='group')
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()

# ============================================================
# TAB 2: FORM ANALYSIS
# ============================================================
with tab2:
    st.subheader("🎯 Team Form Analysis")
    
    # Select league and season
    if len(selected_leagues) > 0:
        league = selected_leagues[0]
        season = selected_seasons[0] if selected_seasons else None
        
        if season and league in explorer.data and season in explorer.data[league]:
            data = explorer.data[league][season]
            df = explorer.fixtures_to_dataframe(data['fixtures'])
            
            # Get all teams
            teams = sorted(df['home_team'].unique())
            
            selected_team = st.selectbox("Select Team", teams)
            
            if selected_team:
                # Team matches
                team_matches = df[
                    (df['home_team'] == selected_team) | 
                    (df['away_team'] == selected_team)
                ].copy()
                
                team_matches['is_home'] = team_matches['home_team'] == selected_team
                team_matches['team_goals'] = team_matches.apply(
                    lambda x: x['home_goals'] if x['is_home'] else x['away_goals'], axis=1
                )
                team_matches['opponent_goals'] = team_matches.apply(
                    lambda x: x['away_goals'] if x['is_home'] else x['home_goals'], axis=1
                )
                team_matches['points'] = team_matches.apply(
                    lambda x: 3 if x['team_goals'] > x['opponent_goals'] else
                             1 if x['team_goals'] == x['opponent_goals'] else 0,
                    axis=1
                )
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Matches Played", len(team_matches))
                col2.metric("Points", team_matches['points'].sum())
                col3.metric("Goals Scored", team_matches['team_goals'].sum())
                col4.metric("Goals Conceded", team_matches['opponent_goals'].sum())
                
                # Form timeline
                team_matches['rolling_points'] = team_matches['points'].rolling(window=5, min_periods=1).mean() * 3
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=team_matches.index,
                    y=team_matches['rolling_points'],
                    mode='lines+markers',
                    name='Form (5-match rolling avg)',
                    line=dict(color='#3498db', width=3)
                ))
                fig.add_hline(y=1.5, line_dash="dash", line_color="gray", annotation_text="Average")
                fig.update_layout(title=f'{selected_team} Form Over Season', yaxis_title='Points per Game')
                st.plotly_chart(fig, use_container_width=True)
                
                # Recent results
                st.markdown("### Recent Results (Last 10)")
                recent = team_matches.tail(10)[['date', 'home_team', 'home_goals', 'away_goals', 'away_team', 'points']]
                st.dataframe(recent, use_container_width=True)

# ============================================================
# TAB 3: HEAD TO HEAD
# ============================================================
with tab3:
    st.subheader("⚔️ Head-to-Head Analysis")
    
    if selected_leagues:
        league = selected_leagues[0]
        season = selected_seasons[0] if selected_seasons else None
        
        if season and league in explorer.data and season in explorer.data[league]:
            data = explorer.data[league][season]
            df = explorer.fixtures_to_dataframe(data['fixtures'])
            
            teams = sorted(df['home_team'].unique())
            
            col1, col2 = st.columns(2)
            
            with col1:
                team1 = st.selectbox("Team 1", teams, key="h2h_team1")
            with col2:
                team2 = st.selectbox("Team 2", [t for t in teams if t != team1], key="h2h_team2")
            
            if team1 and team2:
                # Get H2H matches
                h2h = df[
                    ((df['home_team'] == team1) & (df['away_team'] == team2)) |
                    ((df['home_team'] == team2) & (df['away_team'] == team1))
                ]
                
                if len(h2h) == 0:
                    st.warning("No head-to-head matches found this season")
                else:
                    # Stats
                    team1_wins = len(h2h[
                        ((h2h['home_team'] == team1) & (h2h['result'] == 'H')) |
                        ((h2h['away_team'] == team1) & (h2h['result'] == 'A'))
                    ])
                    draws = len(h2h[h2h['result'] == 'D'])
                    team2_wins = len(h2h) - team1_wins - draws
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric(f"{team1} Wins", team1_wins)
                    col2.metric("Draws", draws)
                    col3.metric(f"{team2} Wins", team2_wins)
                    
                    # Matches
                    st.markdown("### Matches")
                    st.dataframe(
                        h2h[['date', 'home_team', 'home_goals', 'away_goals', 'away_team']],
                        use_container_width=True
                    )

# ============================================================
# TAB 4: LEAGUE COMPARISON
# ============================================================
with tab4:
    st.subheader("📊 League Comparison")
    
    if len(selected_seasons) > 0:
        season = selected_seasons[0]
        
        comparison_data = []
        
        for league in selected_leagues:
            if league not in explorer.data or season not in explorer.data[league]:
                continue
            
            data = explorer.data[league][season]
            df = explorer.fixtures_to_dataframe(data['fixtures'])
            
            if df.empty:
                continue
            
            home_adv = explorer.analyze_home_advantage(df)
            total_goals = df['home_goals'] + df['away_goals']
            
            comparison_data.append({
                'League': league.replace('_', ' ').title(),
                'Matches': home_adv['total_matches'],
                'Home Win %': home_adv['home_win_pct'],
                'Draw %': home_adv['draw_pct'],
                'Away Win %': home_adv['away_win_pct'],
                'Goals/Match': total_goals.mean()
            })
        
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            
            # Table
            st.dataframe(comp_df, use_container_width=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Bar(name='Home Win %', x=comp_df['League'], y=comp_df['Home Win %'], marker_color='#2ecc71'))
                fig.add_trace(go.Bar(name='Draw %', x=comp_df['League'], y=comp_df['Draw %'], marker_color='#f39c12'))
                fig.add_trace(go.Bar(name='Away Win %', x=comp_df['League'], y=comp_df['Away Win %'], marker_color='#e74c3c'))
                fig.update_layout(title='Result Distribution by League', barmode='group')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(comp_df, x='League', y='Goals/Match', title='Average Goals per Match')
                st.plotly_chart(fig, use_container_width=True)