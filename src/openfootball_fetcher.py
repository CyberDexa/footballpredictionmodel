"""
OpenFootball Data Fetcher
Uses openfootball/football.json for free, up-to-date match data
No API key required!
"""

import pandas as pd
import requests
import os
import time
from typing import List, Dict, Optional
from datetime import datetime


class OpenFootballFetcher:
    """
    Fetches football data from OpenFootball (openfootball.github.io)
    
    Free, open public domain football data - no API key required!
    Data includes: EPL, Bundesliga, La Liga, Serie A, Ligue 1, and more.
    """
    
    # Base URL for raw JSON data
    BASE_URL = "https://raw.githubusercontent.com/openfootball/football.json/master"
    
    # League file mappings
    LEAGUE_FILES = {
        # England
        'EPL': 'en.1.json',              # English Premier League
        'CHAMPIONSHIP': 'en.2.json',      # English Championship
        'LEAGUE_ONE': 'en.3.json',        # English League One
        'LEAGUE_TWO': 'en.4.json',        # English League Two
        # Scotland
        'SCOTTISH_PREM': 'sco.1.json',    # Scottish Premiership
        # Spain
        'LA_LIGA': 'es.1.json',           # Spanish La Liga
        'LA_LIGA_2': 'es.2.json',         # Spanish Segunda División
        # Italy
        'SERIE_A': 'it.1.json',           # Italian Serie A
        'SERIE_B': 'it.2.json',           # Italian Serie B
        # Germany
        'BUNDESLIGA': 'de.1.json',        # German Bundesliga
        'BUNDESLIGA_2': 'de.2.json',      # German 2. Bundesliga
        # France
        'LIGUE_1': 'fr.1.json',           # French Ligue 1
        'LIGUE_2': 'fr.2.json',           # French Ligue 2
        # Other top leagues
        'EREDIVISIE': 'nl.1.json',        # Dutch Eredivisie
        'PRIMEIRA_LIGA': 'pt.1.json',     # Portuguese Primeira Liga
        'BELGIAN_PRO': 'be.1.json',       # Belgian Pro League
        'AUSTRIAN_BL': 'at.1.json',       # Austrian Bundesliga
        'SUPER_LIG': 'tr.1.json',         # Turkish Süper Lig
        'GREEK_SL': 'gr.1.json',          # Greek Super League
    }
    
    # Available seasons (most recent first)
    SEASONS = [
        '2025-26', '2024-25', '2023-24', '2022-23', '2021-22',
        '2020-21', '2019-20', '2018-19', '2017-18', '2016-17',
        '2015-16', '2014-15', '2013-14', '2012-13', '2011-12', '2010-11'
    ]
    
    # League metadata
    LEAGUES = {
        # England
        'EPL': {
            'name': 'English Premier League',
            'country': 'England',
            'flag': '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
            'file': 'en.1.json',
            'tier': 1
        },
        'CHAMPIONSHIP': {
            'name': 'English Championship',
            'country': 'England', 
            'flag': '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
            'file': 'en.2.json',
            'tier': 2
        },
        'LEAGUE_ONE': {
            'name': 'English League One',
            'country': 'England',
            'flag': '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
            'file': 'en.3.json',
            'tier': 3
        },
        'LEAGUE_TWO': {
            'name': 'English League Two',
            'country': 'England',
            'flag': '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
            'file': 'en.4.json',
            'tier': 4
        },
        # Scotland
        'SCOTTISH_PREM': {
            'name': 'Scottish Premiership',
            'country': 'Scotland',
            'flag': '🏴󠁧󠁢󠁳󠁣󠁴󠁿',
            'file': 'sco.1.json',
            'tier': 1
        },
        # Spain
        'LA_LIGA': {
            'name': 'La Liga',
            'country': 'Spain',
            'flag': '🇪🇸',
            'file': 'es.1.json',
            'tier': 1
        },
        'LA_LIGA_2': {
            'name': 'La Liga 2',
            'country': 'Spain',
            'flag': '🇪🇸',
            'file': 'es.2.json',
            'tier': 2
        },
        # Italy
        'SERIE_A': {
            'name': 'Serie A',
            'country': 'Italy',
            'flag': '🇮🇹',
            'file': 'it.1.json',
            'tier': 1
        },
        'SERIE_B': {
            'name': 'Serie B',
            'country': 'Italy',
            'flag': '🇮🇹',
            'file': 'it.2.json',
            'tier': 2
        },
        # Germany
        'BUNDESLIGA': {
            'name': 'Bundesliga',
            'country': 'Germany',
            'flag': '🇩🇪',
            'file': 'de.1.json',
            'tier': 1
        },
        'BUNDESLIGA_2': {
            'name': '2. Bundesliga',
            'country': 'Germany',
            'flag': '🇩🇪',
            'file': 'de.2.json',
            'tier': 2
        },
        # France
        'LIGUE_1': {
            'name': 'Ligue 1',
            'country': 'France',
            'flag': '🇫🇷',
            'file': 'fr.1.json',
            'tier': 1
        },
        'LIGUE_2': {
            'name': 'Ligue 2',
            'country': 'France',
            'flag': '🇫🇷',
            'file': 'fr.2.json',
            'tier': 2
        },
        # Netherlands
        'EREDIVISIE': {
            'name': 'Eredivisie',
            'country': 'Netherlands',
            'flag': '🇳🇱',
            'file': 'nl.1.json',
            'tier': 1
        },
        # Portugal
        'PRIMEIRA_LIGA': {
            'name': 'Primeira Liga',
            'country': 'Portugal',
            'flag': '🇵🇹',
            'file': 'pt.1.json',
            'tier': 1
        },
        # Belgium
        'BELGIAN_PRO': {
            'name': 'Belgian Pro League',
            'country': 'Belgium',
            'flag': '🇧🇪',
            'file': 'be.1.json',
            'tier': 1
        },
        # Austria
        'AUSTRIAN_BL': {
            'name': 'Austrian Bundesliga',
            'country': 'Austria',
            'flag': '🇦🇹',
            'file': 'at.1.json',
            'tier': 1
        },
        # Turkey
        'SUPER_LIG': {
            'name': 'Süper Lig',
            'country': 'Turkey',
            'flag': '🇹🇷',
            'file': 'tr.1.json',
            'tier': 1
        },
        # Greece
        'GREEK_SL': {
            'name': 'Super League',
            'country': 'Greece',
            'flag': '🇬🇷',
            'file': 'gr.1.json',
            'tier': 1
        },
    }
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the fetcher."""
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
    
    def fetch_season(self, league: str, season: str) -> pd.DataFrame:
        """
        Fetch data for a specific league and season.
        
        Args:
            league: League code (EPL, LA_LIGA, etc.)
            season: Season string (e.g., '2025-26')
            
        Returns:
            DataFrame with match data
        """
        if league not in self.LEAGUES:
            raise ValueError(f"Unknown league: {league}. Available: {list(self.LEAGUES.keys())}")
        
        league_file = self.LEAGUES[league]['file']
        url = f"{self.BASE_URL}/{season}/{league_file}"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return pd.DataFrame()  # Season not available
            raise
        except Exception as e:
            print(f"  ✗ Error fetching {league} {season}: {e}")
            return pd.DataFrame()
        
        matches = []
        for match in data.get('matches', []):
            # Only include completed matches (with scores)
            score = match.get('score')
            if not score:
                continue
            
            ft_score = score.get('ft')
            if not ft_score or len(ft_score) != 2:
                continue
            
            home_goals = ft_score[0]
            away_goals = ft_score[1]
            
            # Determine result
            if home_goals > away_goals:
                result = 'H'
            elif away_goals > home_goals:
                result = 'A'
            else:
                result = 'D'
            
            # Get half-time score if available
            ht_score = score.get('ht', [0, 0])
            if ht_score and len(ht_score) == 2:
                ht_home = ht_score[0]
                ht_away = ht_score[1]
            else:
                ht_home = 0
                ht_away = 0
            
            if ht_home > ht_away:
                ht_result = 'H'
            elif ht_away > ht_home:
                ht_result = 'A'
            else:
                ht_result = 'D'
            
            # Clean team names (remove FC, etc. for consistency)
            home_team = self._clean_team_name(match.get('team1', ''))
            away_team = self._clean_team_name(match.get('team2', ''))
            
            matches.append({
                'Date': match.get('date', ''),
                'HomeTeam': home_team,
                'AwayTeam': away_team,
                'FTHG': home_goals,
                'FTAG': away_goals,
                'FTR': result,
                'HTHG': ht_home,
                'HTAG': ht_away,
                'HTR': ht_result,
                'Season': season,
                'League': league,
                'Round': match.get('round', '')
            })
        
        return pd.DataFrame(matches)
    
    def _clean_team_name(self, name: str) -> str:
        """Clean team name for consistency."""
        # Remove common suffixes
        suffixes = [' FC', ' AFC', ' CF', ' SC', ' AC']
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        return name.strip()
    
    def fetch_league(self, league: str, seasons: List[str] = None, 
                     min_matches: int = 300) -> pd.DataFrame:
        """
        Fetch multiple seasons for a league.
        
        Args:
            league: League code
            seasons: List of seasons to fetch (defaults to most recent)
            min_matches: Minimum total matches to fetch
            
        Returns:
            Combined DataFrame with all seasons
        """
        if seasons is None:
            seasons = self.SEASONS.copy()
        
        print(f"Fetching {self.LEAGUES[league]['name']} data...")
        
        all_data = []
        total_matches = 0
        
        for season in seasons:
            print(f"  Fetching {season}...")
            
            df = self.fetch_season(league, season)
            
            if len(df) > 0:
                all_data.append(df)
                total_matches += len(df)
                print(f"    ✓ {len(df)} matches")
            else:
                print(f"    ✗ No data available")
            
            # Stop if we have enough matches
            if total_matches >= min_matches:
                break
            
            time.sleep(0.2)  # Be nice to GitHub
        
        if not all_data:
            raise ValueError(f"No data available for {league}")
        
        combined = pd.concat(all_data, ignore_index=True)
        print(f"✓ Total: {len(combined)} matches for {league}")
        
        return combined
    
    def fetch_all_leagues(self, seasons: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Fetch data for all available leagues."""
        results = {}
        
        for league in self.LEAGUES.keys():
            try:
                df = self.fetch_league(league, seasons)
                if len(df) > 0:
                    results[league] = df
                    self.save_league_data(df, league)
            except Exception as e:
                print(f"  ✗ Error fetching {league}: {e}")
        
        return results
    
    def save_league_data(self, df: pd.DataFrame, league: str):
        """Save league data to CSV."""
        filepath = os.path.join(self.data_dir, f"{league.lower()}_data.csv")
        df.to_csv(filepath, index=False)
        print(f"✓ Saved {len(df)} matches to {filepath}")
    
    def load_league_data(self, league: str) -> pd.DataFrame:
        """Load cached league data from CSV."""
        filepath = os.path.join(self.data_dir, f"{league.lower()}_data.csv")
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        raise FileNotFoundError(f"No cached data for {league}. Run fetch first.")
    
    def get_or_fetch_league_data(self, league: str, force_refresh: bool = False, 
                                  max_age_days: int = 7) -> pd.DataFrame:
        """
        Get league data from cache or fetch if needed.
        
        Args:
            league: League code
            force_refresh: Force fetch even if cache exists
            max_age_days: Auto-refresh if data is older than this many days
        """
        filepath = os.path.join(self.data_dir, f"{league.lower()}_data.csv")
        
        if not force_refresh and os.path.exists(filepath):
            # Check if data needs auto-refresh
            file_age = self.get_data_age_days(league)
            
            if file_age is not None and file_age <= max_age_days:
                print(f"Loading cached {league} data (updated {file_age} days ago)...")
                df = pd.read_csv(filepath)
                print(f"  ✓ Loaded {len(df)} matches")
                return df
            elif file_age is not None:
                print(f"Data is {file_age} days old (max: {max_age_days}). Auto-refreshing...")
        
        # Fetch fresh data
        df = self.fetch_league(league)
        self.save_league_data(df, league)
        return df
    
    def get_data_age_days(self, league: str) -> Optional[int]:
        """Get how many days old the cached data is."""
        try:
            df = self.load_league_data(league)
            df['Date'] = pd.to_datetime(df['Date'])
            latest_date = df['Date'].max()
            days_old = (datetime.now() - latest_date.to_pydatetime()).days
            return days_old
        except:
            return None
    
    def get_file_modified_days(self, league: str) -> Optional[int]:
        """Get how many days since the data file was last modified."""
        filepath = os.path.join(self.data_dir, f"{league.lower()}_data.csv")
        if os.path.exists(filepath):
            modified_time = os.path.getmtime(filepath)
            modified_date = datetime.fromtimestamp(modified_time)
            return (datetime.now() - modified_date).days
        return None
    
    def needs_refresh(self, league: str, max_age_days: int = 7) -> bool:
        """Check if league data needs to be refreshed."""
        age = self.get_data_age_days(league)
        return age is None or age > max_age_days
    
    def refresh_all_if_needed(self, max_age_days: int = 7) -> Dict[str, bool]:
        """Refresh all leagues that need updating."""
        results = {}
        for league in self.LEAGUES.keys():
            if self.needs_refresh(league, max_age_days):
                try:
                    self.fetch_league(league)
                    results[league] = True
                except Exception as e:
                    print(f"  ✗ Error refreshing {league}: {e}")
                    results[league] = False
            else:
                results[league] = None  # Already up to date
        return results
    
    def get_available_leagues(self) -> Dict:
        """Return available leagues with metadata."""
        return self.LEAGUES
    
    def get_league_teams(self, league: str) -> List[str]:
        """Get list of teams from cached data."""
        try:
            df = self.load_league_data(league)
            teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
            return sorted(list(teams))
        except FileNotFoundError:
            return []
    
    def get_data_info(self, league: str) -> Dict:
        """Get info about cached data for a league."""
        try:
            df = self.load_league_data(league)
            df['Date'] = pd.to_datetime(df['Date'])
            
            return {
                'total_matches': len(df),
                'earliest_date': df['Date'].min().strftime('%Y-%m-%d'),
                'latest_date': df['Date'].max().strftime('%Y-%m-%d'),
                'seasons': df['Season'].unique().tolist(),
                'teams': len(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))
            }
        except:
            return None


# Test the fetcher
if __name__ == "__main__":
    fetcher = OpenFootballFetcher()
    
    # Test fetching current season EPL
    print("\n" + "="*50)
    print("Testing OpenFootball Data Fetcher")
    print("="*50 + "\n")
    
    # Fetch EPL data (multiple seasons)
    df = fetcher.fetch_league('EPL', seasons=['2025-26', '2024-25', '2023-24'])
    
    print(f"\nTotal EPL matches: {len(df)}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Show last 5 matches
    print("\nLast 5 matches:")
    df_sorted = df.sort_values('Date', ascending=False)
    for _, row in df_sorted.head(5).iterrows():
        print(f"  {row['Date']}: {row['HomeTeam']} {row['FTHG']}-{row['FTAG']} {row['AwayTeam']}")
    
    # Get teams
    teams = fetcher.get_league_teams('EPL')
    print(f"\nTeams ({len(teams)}): {', '.join(teams[:10])}...")
