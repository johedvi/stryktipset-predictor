"""
Data Manager - Fetch, view, and manage football data
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import sys
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_fetcher import APIFootballFetcher
from src.data.data_explorer import DataExplorer
from config import API_FOOTBALL_KEY, LEAGUES, SEASONS

st.set_page_config(page_title="Data Manager", page_icon="📊", layout="wide")

st.title("📊 Data Manager")
st.markdown("### Fetch, View, and Analyze Football Data")

# Check API key
if API_FOOTBALL_KEY == "your_api_key_here":
    st.error("⚠️ API key not configured!")
    st.info("Set your API key in `.env` or `config/config.py`")
    st.stop()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔄 Fetch Data", "📁 View Data", "📈 Analytics", "🗑️ Manage"])

# ============================================================
# TAB 1: FETCH DATA
# ============================================================
with tab1:
    st.subheader("🔄 Fetch New Data from API-Football")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **API Status:** ✅ Configured
        
        **Available Leagues:**
        - Premier League (ID: 39)
        - Championship (ID: 40)
        - League One (ID: 41)
        """)
    
    with col2:
        st.warning("""
        **Rate Limits:**
        - Pro Plan: 300 requests/minute
        - ~2-3 minutes for all data
        
        **Cost:** ~500 API calls for 3 leagues × 5 seasons
        """)
    
    st.markdown("---")
    
    # Selection
    st.subheader("Select Data to Fetch")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_leagues = st.multiselect(
            "Leagues",
            options=list(LEAGUES.keys()),
            default=list(LEAGUES.keys()),
            help="Select which leagues to fetch"
        )
    
    with col2:
        selected_seasons = st.multiselect(
            "Seasons",
            options=SEASONS,
            default=[2024],
            help="Select which seasons to fetch"
        )
    
    # Estimate
    estimated_requests = len(selected_leagues) * len(selected_seasons) * 30
    estimated_time = estimated_requests * 0.2 / 60  # seconds to minutes
    
    st.info(f"""
    **Estimated:**
    - API Requests: ~{estimated_requests}
    - Time: ~{estimated_time:.1f} minutes
    """)
    
    # Fetch button
    if st.button("🚀 Start Fetching Data", type="primary", use_container_width=True):
        if not selected_leagues or not selected_seasons:
            st.error("Please select at least one league and season")
        else:
            fetcher = APIFootballFetcher()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_combinations = len(selected_leagues) * len(selected_seasons)
            current = 0
            
            results = []
            
            for league_name in selected_leagues:
                league_id = LEAGUES[league_name]
                
                for season in selected_seasons:
                    current += 1
                    status_text.text(f"Fetching {league_name} {season}... ({current}/{total_combinations})")
                    
                    try:
                        data = fetcher.fetch_all_league_data(league_id, season)
                        fetcher.save_league_data(league_name, season, data)
                        
                        results.append({
                            'league': league_name,
                            'season': season,
                            'status': '✅ Success',
                            'fixtures': len(data['fixtures']),
                            'teams': len(data['team_statistics'])
                        })
                        
                        st.success(f"✅ {league_name} {season}: {len(data['fixtures'])} fixtures")
                    
                    except Exception as e:
                        results.append({
                            'league': league_name,
                            'season': season,
                            'status': f'❌ Error: {str(e)[:50]}',
                            'fixtures': 0,
                            'teams': 0
                        })
                        st.error(f"❌ Failed: {league_name} {season}")
                    
                    progress_bar.progress(current / total_combinations)
            
            status_text.empty()
            progress_bar.empty()
            
            # Summary
            st.markdown("---")
            st.subheader("📋 Fetch Summary")
            
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
            
            success_count = len(df[df['status'] == '✅ Success'])
            st.success(f"✅ Successfully fetched {success_count}/{len(results)} league-seasons")

# ============================================================
# TAB 2: VIEW DATA
# ============================================================
with tab2:
    st.subheader("📁 Available Data")
    
    # Scan data directory
    data_dir = Path("data/raw")
    data_files = list(data_dir.glob("*.json"))
    
    if not data_files:
        st.warning("⚠️ No data files found. Fetch data first!")
    else:
        st.success(f"Found {len(data_files)} data files")
        
        # Parse file info
        file_info = []
        for file in data_files:
            parts = file.stem.split('_')
            league = '_'.join(parts[:-1])
            season = parts[-1]
            
            # Get file size
            size_mb = file.stat().st_size / (1024 * 1024)
            
            # Get modification time
            mod_time = datetime.fromtimestamp(file.stat().st_mtime)
            
            # Try to get match count
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    match_count = len([f for f in data['fixtures'] if f['fixture']['status']['short'] == 'FT'])
            except:
                match_count = 0
            
            file_info.append({
                'League': league.replace('_', ' ').title(),
                'Season': season,
                'Matches': match_count,
                'Size (MB)': f"{size_mb:.2f}",
                'Last Updated': mod_time.strftime('%Y-%m-%d %H:%M'),
                'File': file.name
            })
        
        df = pd.DataFrame(file_info)
        
        # Display with selection
        st.dataframe(
            df[['League', 'Season', 'Matches', 'Size (MB)', 'Last Updated']],
            use_container_width=True
        )
        
        # View details
        st.markdown("---")
        st.subheader("🔍 View File Details")
        
        selected_file = st.selectbox(
            "Select file to view",
            options=[f"{row['League']} - {row['Season']}" for _, row in df.iterrows()]
        )
        
        if selected_file:
            league, season = selected_file.split(' - ')
            league_key = league.lower().replace(' ', '_')
            filename = f"{league_key}_{season}.json"
            
            with open(data_dir / filename, 'r') as f:
                data = json.load(f)
            
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("Total Fixtures", len(data['fixtures']))
            col2.metric("Finished Matches", len([f for f in data['fixtures'] if f['fixture']['status']['short'] == 'FT']))
            col3.metric("Teams", len(data.get('team_statistics', {})))
            col4.metric("League ID", data.get('league_id', 'N/A'))
            
            # Sample fixtures
            with st.expander("📋 Sample Fixtures (First 10)"):
                sample_fixtures = []
                for fixture in data['fixtures'][:10]:
                    if fixture['fixture']['status']['short'] == 'FT':
                        sample_fixtures.append({
                            'Date': fixture['fixture']['date'][:10],
                            'Home': fixture['teams']['home']['name'],
                            'Score': f"{fixture['goals']['home']}-{fixture['goals']['away']}",
                            'Away': fixture['teams']['away']['name']
                        })
                
                if sample_fixtures:
                    st.dataframe(pd.DataFrame(sample_fixtures), use_container_width=True)

# ============================================================
# TAB 3: ANALYTICS
# ============================================================
with tab3:
    st.subheader("📈 Data Analytics")
    
    # Check if data exists
    data_dir = Path("data/raw")
    data_files = list(data_dir.glob("*.json"))
    
    if not data_files:
        st.warning("⚠️ No data available for analysis. Fetch data first!")
    else:
        # League and season selection
        col1, col2 = st.columns(2)
        
        # Parse available leagues and seasons
        available_data = {}
        for file in data_files:
            parts = file.stem.split('_')
            league = '_'.join(parts[:-1])
            season = int(parts[-1])
            
            if league not in available_data:
                available_data[league] = []
            available_data[league].append(season)
        
        with col1:
            selected_league_analytics = st.selectbox(
                "Select League",
                options=sorted(available_data.keys()),
                format_func=lambda x: x.replace('_', ' ').title()
            )
        
        with col2:
            selected_season_analytics = st.selectbox(
                "Select Season",
                options=sorted(available_data[selected_league_analytics], reverse=True)
            )
        
        if st.button("📊 Generate Analytics", use_container_width=True):
            with st.spinner("Analyzing data..."):
                try:
                    explorer = DataExplorer()
                    explorer.load_all_data(
                        leagues=[selected_league_analytics],
                        seasons=[selected_season_analytics]
                    )
                    
                    # Get data
                    data = explorer.data[selected_league_analytics][selected_season_analytics]
                    df = explorer.fixtures_to_dataframe(data['fixtures'])
                    
                    if df.empty:
                        st.warning("No finished matches found")
                    else:
                        # Home advantage analysis
                        home_adv = explorer.analyze_home_advantage(df)
                        
                        st.markdown("---")
                        st.subheader("📊 Match Statistics")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total Matches", home_adv['total_matches'])
                        col2.metric("Home Wins", f"{home_adv['home_win_pct']:.1f}%")
                        col3.metric("Draws", f"{home_adv['draw_pct']:.1f}%")
                        col4.metric("Away Wins", f"{home_adv['away_win_pct']:.1f}%")
                        
                        # Goals
                        col1, col2, col3 = st.columns(3)
                        total_goals = df['home_goals'] + df['away_goals']
                        col1.metric("Avg Home Goals", f"{home_adv['home_goals_avg']:.2f}")
                        col2.metric("Avg Away Goals", f"{home_adv['away_goals_avg']:.2f}")
                        col3.metric("Avg Total Goals", f"{total_goals.mean():.2f}")
                        
                        st.markdown("---")
                        
                        # Visualizations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Result distribution pie chart
                            result_counts = df['result'].value_counts()
                            fig = px.pie(
                                values=result_counts.values,
                                names=['Home Win', 'Draw', 'Away Win'],
                                title='Match Result Distribution',
                                color_discrete_sequence=['#2ecc71', '#f39c12', '#e74c3c']
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Goals distribution histogram
                            fig = px.histogram(
                                total_goals,
                                nbins=range(0, total_goals.max() + 2),
                                title='Goals per Match Distribution',
                                labels={'value': 'Total Goals', 'count': 'Number of Matches'}
                            )
                            fig.update_traces(marker_color='#3498db')
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Top scoring teams
                        st.markdown("---")
                        st.subheader("🏆 Top Scoring Teams")
                        
                        home_goals_by_team = df.groupby('home_team')['home_goals'].sum()
                        away_goals_by_team = df.groupby('away_team')['away_goals'].sum()
                        total_goals_by_team = home_goals_by_team.add(away_goals_by_team, fill_value=0).sort_values(ascending=False)
                        
                        fig = px.bar(
                            x=total_goals_by_team.head(10).index,
                            y=total_goals_by_team.head(10).values,
                            title='Top 10 Scoring Teams',
                            labels={'x': 'Team', 'y': 'Total Goals'},
                            color=total_goals_by_team.head(10).values,
                            color_continuous_scale='Blues'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Timeline
                        st.markdown("---")
                        st.subheader("📅 Results Over Time")
                        
                        df['date'] = pd.to_datetime(df['date'])
                        timeline_data = df.groupby([df['date'].dt.to_period('M'), 'result']).size().unstack(fill_value=0)
                        timeline_data.index = timeline_data.index.to_timestamp()
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=timeline_data.index, y=timeline_data['H'], mode='lines', name='Home Wins', line=dict(color='#2ecc71')))
                        fig.add_trace(go.Scatter(x=timeline_data.index, y=timeline_data['D'], mode='lines', name='Draws', line=dict(color='#f39c12')))
                        fig.add_trace(go.Scatter(x=timeline_data.index, y=timeline_data['A'], mode='lines', name='Away Wins', line=dict(color='#e74c3c')))
                        fig.update_layout(title='Match Results by Month', xaxis_title='Month', yaxis_title='Number of Matches')
                        st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error analyzing data: {e}")

# ============================================================
# TAB 4: MANAGE
# ============================================================
with tab4:
    st.subheader("🗑️ Manage Data")
    
    data_dir = Path("data/raw")
    data_files = list(data_dir.glob("*.json"))
    
    if not data_files:
        st.info("No data files to manage")
    else:
        st.warning("⚠️ Be careful! Deleted files cannot be recovered.")
        
        # Parse files
        file_list = []
        for file in data_files:
            parts = file.stem.split('_')
            league = '_'.join(parts[:-1])
            season = parts[-1]
            size_mb = file.stat().st_size / (1024 * 1024)
            
            file_list.append({
                'League': league.replace('_', ' ').title(),
                'Season': season,
                'Size (MB)': f"{size_mb:.2f}",
                'Path': str(file)
            })
        
        df = pd.DataFrame(file_list)
        
        # Selection
        selected_indices = st.multiselect(
            "Select files to delete",
            options=range(len(df)),
            format_func=lambda i: f"{df.iloc[i]['League']} - {df.iloc[i]['Season']} ({df.iloc[i]['Size (MB)']} MB)"
        )
        
        if selected_indices:
            st.dataframe(
                df.iloc[selected_indices][['League', 'Season', 'Size (MB)']],
                use_container_width=True
            )
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🗑️ Delete Selected Files", type="primary", use_container_width=True):
                    for idx in selected_indices:
                        file_path = Path(df.iloc[idx]['Path'])
                        try:
                            file_path.unlink()
                            st.success(f"✅ Deleted: {df.iloc[idx]['League']} - {df.iloc[idx]['Season']}")
                        except Exception as e:
                            st.error(f"❌ Error deleting {file_path.name}: {e}")
                    
                    st.rerun()
        
        st.markdown("---")
        st.subheader("🧹 Cleanup Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Cache", use_container_width=True):
                cache_dir = Path("data/cache")
                cache_files = list(cache_dir.glob("*.json"))
                
                for file in cache_files:
                    file.unlink()
                
                st.success(f"✅ Cleared {len(cache_files)} cache files")
        
        with col2:
            if st.button("🗑️ Clear Processed Data", use_container_width=True):
                processed_dir = Path("data/processed")
                processed_files = list(processed_dir.glob("*.csv")) + list(processed_dir.glob("*.png"))
                
                for file in processed_files:
                    file.unlink()
                
                st.success(f"✅ Cleared {len(processed_files)} processed files")
        
        st.markdown("---")
        st.subheader("📊 Storage Summary")
        
        # Calculate storage
        raw_size = sum(f.stat().st_size for f in data_dir.glob("*.json")) / (1024 * 1024)
        cache_size = sum(f.stat().st_size for f in Path("data/cache").glob("*.json")) / (1024 * 1024) if Path("data/cache").exists() else 0
        processed_size = sum(f.stat().st_size for f in Path("data/processed").glob("*")) / (1024 * 1024) if Path("data/processed").exists() else 0
        model_size = sum(f.stat().st_size for f in Path("models").glob("*.pkl")) / (1024 * 1024) if Path("models").exists() else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Raw Data", f"{raw_size:.1f} MB")
        col2.metric("Cache", f"{cache_size:.1f} MB")
        col3.metric("Processed", f"{processed_size:.1f} MB")
        col4.metric("Models", f"{model_size:.1f} MB")
        
        st.metric("Total Storage", f"{raw_size + cache_size + processed_size + model_size:.1f} MB")