"""
Data Explorer for analyzing fetched football data
Provides insights and visualizations
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, LEAGUES
from utils import setup_logging, get_match_result

logger = setup_logging(__name__)

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class DataExplorer:
    """
    Explore and analyze football data
    """
    
    def __init__(self):
        """Initialize the data explorer"""
        self.data = {}
        logger.info("DataExplorer initialized")
    
    def load_league_data(self, league_name: str, season: int) -> Dict[str, Any]:
        """
        Load league data from file
        
        Args:
            league_name: Name of the league
            season: Season year
        
        Returns:
            League data dictionary
        """
        filename = RAW_DATA_DIR / f"{league_name}_{season}.json"
        
        if not filename.exists():
            logger.warning(f"File not found: {filename}")
            return None
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {league_name} {season}")
        return data
    
    def load_all_data(self, leagues: List[str] = None, seasons: List[int] = None):
        """
        Load all available data
        
        Args:
            leagues: List of league names (default: all)
            seasons: List of seasons (default: all)
        """
        if leagues is None:
            leagues = list(LEAGUES.keys())
        if seasons is None:
            seasons = [2022, 2023, 2024]
        
        for league in leagues:
            self.data[league] = {}
            for season in seasons:
                data = self.load_league_data(league, season)
                if data:
                    self.data[league][season] = data
        
        logger.info(f"Loaded data for {len(self.data)} leagues")
    
    def fixtures_to_dataframe(self, fixtures: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert fixtures to pandas DataFrame
        
        Args:
            fixtures: List of fixture dictionaries
        
        Returns:
            DataFrame with fixture data
        """
        records = []
        
        for fixture in fixtures:
            # Only include finished matches
            if fixture['fixture']['status']['short'] != 'FT':
                continue
            
            record = {
                'fixture_id': fixture['fixture']['id'],
                'date': fixture['fixture']['date'][:10],
                'home_team': fixture['teams']['home']['name'],
                'home_team_id': fixture['teams']['home']['id'],
                'away_team': fixture['teams']['away']['name'],
                'away_team_id': fixture['teams']['away']['id'],
                'home_goals': fixture['goals']['home'],
                'away_goals': fixture['goals']['away'],
                'result': get_match_result(fixture['goals']['home'], fixture['goals']['away']),
            }
            records.append(record)
        
        return pd.DataFrame(records)
    
    def analyze_home_advantage(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze home advantage statistics
        
        Args:
            df: DataFrame with match results
        
        Returns:
            Dictionary with home advantage metrics
        """
        total_matches = len(df)
        home_wins = len(df[df['result'] == 'H'])
        draws = len(df[df['result'] == 'D'])
        away_wins = len(df[df['result'] == 'A'])
        
        return {
            'total_matches': total_matches,
            'home_win_pct': (home_wins / total_matches) * 100,
            'draw_pct': (draws / total_matches) * 100,
            'away_win_pct': (away_wins / total_matches) * 100,
            'home_goals_avg': df['home_goals'].mean(),
            'away_goals_avg': df['away_goals'].mean(),
        }
    
    def analyze_league(self, league_name: str, season: int):
        """
        Comprehensive analysis of a league season
        
        Args:
            league_name: Name of the league
            season: Season year
        """
        if league_name not in self.data or season not in self.data[league_name]:
            logger.error(f"Data not loaded for {league_name} {season}")
            return
        
        data = self.data[league_name][season]
        fixtures = data['fixtures']
        
        df = self.fixtures_to_dataframe(fixtures)
        
        if df.empty:
            logger.warning(f"No finished matches found for {league_name} {season}")
            return
        
        print(f"\n{'='*70}")
        print(f"Analysis: {league_name.upper()} - Season {season}")
        print(f"{'='*70}\n")
        
        # Basic statistics
        print(f"Total matches: {len(df)}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Unique teams: {df['home_team'].nunique()}")
        
        # Home advantage analysis
        home_adv = self.analyze_home_advantage(df)
        print(f"\n--- Home Advantage ---")
        print(f"Home wins: {home_adv['home_win_pct']:.1f}%")
        print(f"Draws: {home_adv['draw_pct']:.1f}%")
        print(f"Away wins: {home_adv['away_win_pct']:.1f}%")
        print(f"Avg home goals: {home_adv['home_goals_avg']:.2f}")
        print(f"Avg away goals: {home_adv['away_goals_avg']:.2f}")
        
        # Goal distribution
        print(f"\n--- Goal Distribution ---")
        total_goals = df['home_goals'] + df['away_goals']
        print(f"Avg goals per match: {total_goals.mean():.2f}")
        print(f"Highest scoring: {df['home_team'].iloc[total_goals.argmax()]} {df['home_goals'].iloc[total_goals.argmax()]}-{df['away_goals'].iloc[total_goals.argmax()]} {df['away_team'].iloc[total_goals.argmax()]}")
        
        # Top scoring teams
        home_goals_by_team = df.groupby('home_team')['home_goals'].sum()
        away_goals_by_team = df.groupby('away_team')['away_goals'].sum()
        total_goals_by_team = home_goals_by_team.add(away_goals_by_team, fill_value=0).sort_values(ascending=False)
        
        print(f"\n--- Top 5 Scoring Teams ---")
        for i, (team, goals) in enumerate(total_goals_by_team.head(5).items(), 1):
            print(f"{i}. {team}: {int(goals)} goals")
        
        return df
    
    def plot_result_distribution(self, league_name: str, season: int):
        """
        Plot the distribution of match results
        
        Args:
            league_name: Name of the league
            season: Season year
        """
        if league_name not in self.data or season not in self.data[league_name]:
            logger.error(f"Data not loaded for {league_name} {season}")
            return
        
        data = self.data[league_name][season]
        df = self.fixtures_to_dataframe(data['fixtures'])
        
        if df.empty:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Result distribution
        result_counts = df['result'].value_counts()
        colors = {'H': '#2ecc71', 'D': '#f39c12', 'A': '#e74c3c'}
        result_colors = [colors[r] for r in result_counts.index]
        
        axes[0].bar(['Home Win', 'Draw', 'Away Win'], 
                    [result_counts.get('H', 0), result_counts.get('D', 0), result_counts.get('A', 0)],
                    color=result_colors)
        axes[0].set_title(f'{league_name.title()} {season} - Match Results')
        axes[0].set_ylabel('Number of Matches')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Goal distribution
        total_goals = df['home_goals'] + df['away_goals']
        axes[1].hist(total_goals, bins=range(0, total_goals.max() + 2), 
                     color='#3498db', edgecolor='black', alpha=0.7)
        axes[1].set_title('Goals per Match Distribution')
        axes[1].set_xlabel('Total Goals')
        axes[1].set_ylabel('Number of Matches')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(PROCESSED_DATA_DIR / f'{league_name}_{season}_distribution.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {league_name}_{season}_distribution.png")
        plt.show()
    
    def plot_home_away_comparison(self, league_name: str, season: int):
        """
        Compare home vs away performance
        
        Args:
            league_name: Name of the league
            season: Season year
        """
        if league_name not in self.data or season not in self.data[league_name]:
            logger.error(f"Data not loaded for {league_name} {season}")
            return
        
        data = self.data[league_name][season]
        df = self.fixtures_to_dataframe(data['fixtures'])
        
        if df.empty:
            return
        
        # Calculate home and away statistics
        teams = pd.concat([df['home_team'], df['away_team']]).unique()
        
        team_stats = []
        for team in teams:
            home_matches = df[df['home_team'] == team]
            away_matches = df[df['away_team'] == team]
            
            if len(home_matches) > 0 and len(away_matches) > 0:
                team_stats.append({
                    'team': team,
                    'home_goals_avg': home_matches['home_goals'].mean(),
                    'away_goals_avg': away_matches['away_goals'].mean(),
                    'home_conceded_avg': home_matches['away_goals'].mean(),
                    'away_conceded_avg': away_matches['home_goals'].mean(),
                })
        
        stats_df = pd.DataFrame(team_stats)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Goals scored comparison
        axes[0].scatter(stats_df['home_goals_avg'], stats_df['away_goals_avg'], 
                       alpha=0.6, s=100, color='#3498db')
        axes[0].plot([0, stats_df[['home_goals_avg', 'away_goals_avg']].max().max()],
                     [0, stats_df[['home_goals_avg', 'away_goals_avg']].max().max()],
                     'r--', alpha=0.5, label='Equal performance')
        axes[0].set_xlabel('Home Goals per Game')
        axes[0].set_ylabel('Away Goals per Game')
        axes[0].set_title('Goals Scored: Home vs Away')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Goals conceded comparison
        axes[1].scatter(stats_df['home_conceded_avg'], stats_df['away_conceded_avg'],
                       alpha=0.6, s=100, color='#e74c3c')
        axes[1].plot([0, stats_df[['home_conceded_avg', 'away_conceded_avg']].max().max()],
                     [0, stats_df[['home_conceded_avg', 'away_conceded_avg']].max().max()],
                     'r--', alpha=0.5, label='Equal performance')
        axes[1].set_xlabel('Home Goals Conceded per Game')
        axes[1].set_ylabel('Away Goals Conceded per Game')
        axes[1].set_title('Goals Conceded: Home vs Away')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(PROCESSED_DATA_DIR / f'{league_name}_{season}_home_away.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {league_name}_{season}_home_away.png")
        plt.show()
    
    def analyze_draw_patterns(self, league_name: str, season: int):
        """
        Analyze when draws are most likely
        
        Args:
            league_name: Name of the league
            season: Season year
        """
        if league_name not in self.data or season not in self.data[league_name]:
            logger.error(f"Data not loaded for {league_name} {season}")
            return
        
        data = self.data[league_name][season]
        df = self.fixtures_to_dataframe(data['fixtures'])
        
        if df.empty:
            return
        
        draws = df[df['result'] == 'D']
        
        print(f"\n--- Draw Analysis for {league_name} {season} ---")
        print(f"Total draws: {len(draws)} ({len(draws)/len(df)*100:.1f}%)")
        
        # Score distribution in draws
        draw_scores = draws.groupby(['home_goals', 'away_goals']).size().sort_values(ascending=False)
        print(f"\nMost common draw scores:")
        for (h_goals, a_goals), count in draw_scores.head(5).items():
            print(f"  {int(h_goals)}-{int(a_goals)}: {count} times ({count/len(draws)*100:.1f}%)")
    
    def compare_leagues(self, leagues: List[str], season: int):
        """
        Compare statistics across multiple leagues
        
        Args:
            leagues: List of league names
            season: Season year
        """
        league_stats = []
        
        for league in leagues:
            if league not in self.data or season not in self.data[league]:
                logger.warning(f"Skipping {league} {season} - data not loaded")
                continue
            
            data = self.data[league][season]
            df = self.fixtures_to_dataframe(data['fixtures'])
            
            if df.empty:
                continue
            
            home_adv = self.analyze_home_advantage(df)
            total_goals = df['home_goals'] + df['away_goals']
            
            league_stats.append({
                'league': league,
                'matches': len(df),
                'home_win_pct': home_adv['home_win_pct'],
                'draw_pct': home_adv['draw_pct'],
                'away_win_pct': home_adv['away_win_pct'],
                'goals_per_game': total_goals.mean(),
            })
        
        stats_df = pd.DataFrame(league_stats)
        
        print(f"\n{'='*80}")
        print(f"League Comparison - Season {season}")
        print(f"{'='*80}\n")
        print(stats_df.to_string(index=False))
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Result distribution by league
        x = range(len(stats_df))
        width = 0.25
        
        axes[0].bar([i - width for i in x], stats_df['home_win_pct'], 
                    width, label='Home Win %', color='#2ecc71')
        axes[0].bar(x, stats_df['draw_pct'], 
                    width, label='Draw %', color='#f39c12')
        axes[0].bar([i + width for i in x], stats_df['away_win_pct'], 
                    width, label='Away Win %', color='#e74c3c')
        
        axes[0].set_xlabel('League')
        axes[0].set_ylabel('Percentage')
        axes[0].set_title('Match Result Distribution by League')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(stats_df['league'], rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # Goals per game
        axes[1].bar(stats_df['league'], stats_df['goals_per_game'], color='#3498db')
        axes[1].set_xlabel('League')
        axes[1].set_ylabel('Goals per Game')
        axes[1].set_title('Average Goals per Game by League')
        axes[1].set_xticklabels(stats_df['league'], rotation=45, ha='right')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(PROCESSED_DATA_DIR / f'league_comparison_{season}.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved comparison plot")
        plt.show()
        
        return stats_df
    
    def export_processed_data(self, league_name: str, season: int):
        """
        Export processed data to CSV for ML training
        
        Args:
            league_name: Name of the league
            season: Season year
        """
        if league_name not in self.data or season not in self.data[league_name]:
            logger.error(f"Data not loaded for {league_name} {season}")
            return
        
        data = self.data[league_name][season]
        df = self.fixtures_to_dataframe(data['fixtures'])
        
        if df.empty:
            return
        
        output_file = PROCESSED_DATA_DIR / f'{league_name}_{season}_processed.csv'
        df.to_csv(output_file, index=False)
        logger.info(f"Exported processed data to {output_file}")


def main():
    """
    Example usage of DataExplorer
    """
    explorer = DataExplorer()
    
    # Load ALL your data (3 leagues × 5 seasons)
    print("Loading data...")
    explorer.load_all_data(
        leagues=['premier_league', 'championship', 'league_one'],
        seasons=[2020, 2021, 2022, 2023, 2024]
    )
    
    # Analyze each league
    for league in ['premier_league', 'championship', 'league_one']:
        for season in [2020, 2021, 2022, 2023, 2024]:
            print("\n" + "="*70)
            print(f"ANALYZING {league.upper()} {season}")
            print("="*70)
            explorer.analyze_league(league, season)
    
    # Compare all leagues for 2024
    print("\n" + "="*70)
    print("COMPARING ALL LEAGUES - 2024")
    print("="*70)
    explorer.compare_leagues(['premier_league', 'championship', 'league_one'], 2024)
    
    print("\n✓ Data exploration complete!")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()