"""
Real Football Data Fetcher
Uses API-Football and StatsBomb for real match data
"""

import pandas as pd
import requests
from io import StringIO
import os
import time
import json
from typing import List, Optional, Dict
from datetime import datetime, timedelta


class RealFootballDataFetcher:
    """
    Fetches real football data from multiple sources:
    - API-Football (requires free API key from https://www.api-football.com/)
    - StatsBomb Open Data (free, no API key required)
    """
    
    # API-Football endpoints
    API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
    
    # StatsBomb Open Data (GitHub)
    STATSBOMB_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
    
    # League IDs for API-Football
    LEAGUE_IDS = {
        'EPL': 39,           # Premier League
        'LA_LIGA': 140,      # La Liga
        'SERIE_A': 135,      # Serie A
        'BUNDESLIGA': 78,    # Bundesliga
        'LIGUE_1': 61,       # Ligue 1
        'EREDIVISIE': 88,    # Eredivisie
        'PRIMEIRA_LIGA': 94, # Primeira Liga
        'CHAMPIONS_LEAGUE': 2,
    }
    
    # League metadata
    LEAGUES = {
        'EPL': {'name': 'English Premier League', 'country': 'England', 'flag': '🏴󠁧󠁢󠁥󠁮󠁧󠁿'},
        'LA_LIGA': {'name': 'La Liga', 'country': 'Spain', 'flag': '🇪🇸'},
        'SERIE_A': {'name': 'Serie A', 'country': 'Italy', 'flag': '🇮🇹'},
        'BUNDESLIGA': {'name': 'Bundesliga', 'country': 'Germany', 'flag': '🇩🇪'},
        'LIGUE_1': {'name': 'Ligue 1', 'country': 'France', 'flag': '🇫🇷'},
        'EREDIVISIE': {'name': 'Eredivisie', 'country': 'Netherlands', 'flag': '🇳🇱'},
        'PRIMEIRA_LIGA': {'name': 'Primeira Liga', 'country': 'Portugal', 'flag': '🇵🇹'},
    }
    
    def __init__(self, data_dir: str = "data", api_key: Optional[str] = None):
        """
        Initialize the fetcher.
        
        Args:
            data_dir: Directory to store cached data
            api_key: API-Football API key (get free key at https://www.api-football.com/)
        """
        self.data_dir = data_dir
        self.api_key = api_key or os.environ.get('API_FOOTBALL_KEY')
        os.makedirs(data_dir, exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
    
    def set_api_key(self, api_key: str):
        """Set the API-Football API key"""
        self.api_key = api_key
    
    # ==================== API-FOOTBALL METHODS ====================
    
    def _api_football_request(self, endpoint: str, params: dict = None) -> dict:
        """Make a request to API-Football"""
        if not self.api_key:
            raise ValueError(
                "API-Football key not set. Get a free key at https://www.api-football.com/\n"
                "Then set it using: fetcher.set_api_key('your-key') or\n"
                "Set environment variable: export API_FOOTBALL_KEY='your-key'"
            )
        
        headers = {
            'x-apisports-key': self.api_key,
            'x-rapidapi-host': 'v3.football.api-sports.io'
        }
        
        url = f"{self.API_FOOTBALL_BASE}/{endpoint}"
        response = self.session.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Check for API errors
        if data.get('errors'):
            raise ValueError(f"API Error: {data['errors']}")
        
        return data
    
    def fetch_fixtures_api_football(self, league: str, season: int) -> pd.DataFrame:
        """
        Fetch fixtures/results from API-Football
        
        Args:
            league: League code (EPL, LA_LIGA, etc.)
            season: Season year (e.g., 2024 for 2024-25 season)
        """
        if league not in self.LEAGUE_IDS:
            raise ValueError(f"Unknown league: {league}")
        
        league_id = self.LEAGUE_IDS[league]
        
        print(f"Fetching {league} {season}-{season+1} from API-Football...")
        
        data = self._api_football_request('fixtures', {
            'league': league_id,
            'season': season
        })
        
        matches = []
        for fixture in data.get('response', []):
            fixture_data = fixture.get('fixture', {})
            teams = fixture.get('teams', {})
            goals = fixture.get('goals', {})
            score = fixture.get('score', {})
            
            # Only include finished matches
            if fixture_data.get('status', {}).get('short') not in ['FT', 'AET', 'PEN']:
                continue
            
            home_goals = goals.get('home')
            away_goals = goals.get('away')
            
            if home_goals is None or away_goals is None:
                continue
            
            # Determine result
            if home_goals > away_goals:
                result = 'H'
            elif away_goals > home_goals:
                result = 'A'
            else:
                result = 'D'
            
            # Half-time score
            ht = score.get('halftime', {})
            ht_home = ht.get('home', 0) or 0
            ht_away = ht.get('away', 0) or 0
            
            if ht_home > ht_away:
                ht_result = 'H'
            elif ht_away > ht_home:
                ht_result = 'A'
            else:
                ht_result = 'D'
            
            matches.append({
                'Date': fixture_data.get('date', '')[:10],
                'HomeTeam': teams.get('home', {}).get('name', ''),
                'AwayTeam': teams.get('away', {}).get('name', ''),
                'FTHG': home_goals,
                'FTAG': away_goals,
                'FTR': result,
                'HTHG': ht_home,
                'HTAG': ht_away,
                'HTR': ht_result,
                'Season': f"{season}-{str(season+1)[2:]}",
                'League': league
            })
        
        df = pd.DataFrame(matches)
        print(f"  ✓ Fetched {len(df)} matches")
        return df
    
    def fetch_league_api_football(self, league: str, seasons: List[int] = None) -> pd.DataFrame:
        """Fetch multiple seasons from API-Football"""
        if seasons is None:
            seasons = [2024, 2023, 2022, 2021, 2020]
        
        all_data = []
        for season in seasons:
            try:
                df = self.fetch_fixtures_api_football(league, season)
                if len(df) > 0:
                    all_data.append(df)
                time.sleep(1)  # Rate limiting
            except Exception as e:
                print(f"  ✗ Error fetching {season}: {e}")
        
        if not all_data:
            raise ValueError(f"No data fetched for {league}")
        
        return pd.concat(all_data, ignore_index=True)
    
    # ==================== STATSBOMB METHODS ====================
    
    def fetch_statsbomb_competitions(self) -> pd.DataFrame:
        """Fetch list of available competitions from StatsBomb open data"""
        url = f"{self.STATSBOMB_BASE}/competitions.json"
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        
        return pd.DataFrame(response.json())
    
    def fetch_statsbomb_matches(self, competition_id: int, season_id: int) -> pd.DataFrame:
        """Fetch matches for a competition/season from StatsBomb"""
        url = f"{self.STATSBOMB_BASE}/matches/{competition_id}/{season_id}.json"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return pd.DataFrame(response.json())
        except:
            return pd.DataFrame()
    
    def fetch_statsbomb_all_available(self) -> pd.DataFrame:
        """Fetch all available data from StatsBomb open data"""
        print("Fetching StatsBomb open data competitions...")
        
        competitions = self.fetch_statsbomb_competitions()
        print(f"Found {len(competitions)} competition/season combinations")
        
        all_matches = []
        
        for _, comp in competitions.iterrows():
            comp_id = comp['competition_id']
            season_id = comp['season_id']
            comp_name = comp['competition_name']
            season_name = comp['season_name']
            
            print(f"  Fetching {comp_name} {season_name}...")
            
            matches = self.fetch_statsbomb_matches(comp_id, season_id)
            
            if len(matches) > 0:
                # Process matches
                for _, match in matches.iterrows():
                    home_team = match.get('home_team', {})
                    away_team = match.get('away_team', {})
                    
                    home_score = match.get('home_score')
                    away_score = match.get('away_score')
                    
                    if home_score is None or away_score is None:
                        continue
                    
                    if home_score > away_score:
                        result = 'H'
                    elif away_score > home_score:
                        result = 'A'
                    else:
                        result = 'D'
                    
                    all_matches.append({
                        'Date': match.get('match_date', ''),
                        'HomeTeam': home_team.get('home_team_name', '') if isinstance(home_team, dict) else str(home_team),
                        'AwayTeam': away_team.get('away_team_name', '') if isinstance(away_team, dict) else str(away_team),
                        'FTHG': home_score,
                        'FTAG': away_score,
                        'FTR': result,
                        'Competition': comp_name,
                        'Season': season_name,
                        'League': self._map_competition_to_league(comp_name)
                    })
            
            time.sleep(0.2)  # Be nice to the API
        
        df = pd.DataFrame(all_matches)
        print(f"\n✓ Total matches from StatsBomb: {len(df)}")
        return df
    
    def _map_competition_to_league(self, competition_name: str) -> str:
        """Map StatsBomb competition name to our league codes"""
        mapping = {
            'Premier League': 'EPL',
            'La Liga': 'LA_LIGA',
            'Serie A': 'SERIE_A',
            'Bundesliga': 'BUNDESLIGA',
            'Ligue 1': 'LIGUE_1',
            '1. Bundesliga': 'BUNDESLIGA',
        }
        
        for key, value in mapping.items():
            if key.lower() in competition_name.lower():
                return value
        
        return competition_name
    
    # ==================== COMBINED METHODS ====================
    
    def get_available_leagues(self) -> Dict:
        """Return available leagues"""
        return self.LEAGUES
    
    def get_league_teams(self, league: str) -> List[str]:
        """Get teams for a league from cached data"""
        filepath = os.path.join(self.data_dir, f"{league.lower()}_data.csv")
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
            return sorted(list(teams))
        return []
    
    def save_league_data(self, df: pd.DataFrame, league: str):
        """Save league data to CSV"""
        filepath = os.path.join(self.data_dir, f"{league.lower()}_data.csv")
        df.to_csv(filepath, index=False)
        print(f"✓ Saved {len(df)} matches to {filepath}")
    
    def load_league_data(self, league: str) -> pd.DataFrame:
        """Load cached league data"""
        filepath = os.path.join(self.data_dir, f"{league.lower()}_data.csv")
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        raise FileNotFoundError(f"No cached data for {league}")
    
    def get_or_fetch_league_data(self, league: str, force_refresh: bool = False) -> pd.DataFrame:
        """Get league data from cache or fetch from API"""
        filepath = os.path.join(self.data_dir, f"{league.lower()}_data.csv")
        
        if not force_refresh and os.path.exists(filepath):
            print(f"Loading cached {league} data...")
            df = pd.read_csv(filepath)
            print(f"  ✓ Loaded {len(df)} matches")
            return df
        
        # Try API-Football first
        if self.api_key:
            try:
                df = self.fetch_league_api_football(league)
                self.save_league_data(df, league)
                return df
            except Exception as e:
                print(f"  API-Football error: {e}")
        
        # Try StatsBomb
        try:
            print("Trying StatsBomb open data...")
            all_data = self.fetch_statsbomb_all_available()
            league_data = all_data[all_data['League'] == league]
            
            if len(league_data) > 0:
                self.save_league_data(league_data, league)
                return league_data
        except Exception as e:
            print(f"  StatsBomb error: {e}")
        
        raise ValueError(
            f"Could not fetch data for {league}.\n"
            "Please set up API-Football key:\n"
            "  1. Get free API key at https://www.api-football.com/\n"
            "  2. Set it: fetcher.set_api_key('your-key')\n"
            "  Or set environment variable: export API_FOOTBALL_KEY='your-key'"
        )


def setup_api_key_instructions():
    """Print instructions for setting up API key"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║              🔑 API-FOOTBALL SETUP INSTRUCTIONS                   ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  To get REAL football data, you need a free API key:             ║
║                                                                   ║
║  1. Go to: https://www.api-football.com/                         ║
║  2. Click "Get API Key" and create a free account                ║
║  3. Free tier includes:                                          ║
║     • 100 requests/day                                           ║
║     • All leagues & competitions                                 ║
║     • Live scores & fixtures                                     ║
║                                                                   ║
║  Once you have your key, set it:                                 ║
║                                                                   ║
║  Option 1 - Environment variable (recommended):                  ║
║    export API_FOOTBALL_KEY='your-api-key-here'                   ║
║                                                                   ║
║  Option 2 - In the app:                                          ║
║    Enter your API key in the sidebar                             ║
║                                                                   ║
║  Option 3 - Create a .env file:                                  ║
║    echo "API_FOOTBALL_KEY=your-key" > .env                       ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    setup_api_key_instructions()
    
    # Test StatsBomb data
    fetcher = RealFootballDataFetcher(data_dir="../data")
    
    print("\nTesting StatsBomb open data...")
    try:
        competitions = fetcher.fetch_statsbomb_competitions()
        print(f"\nAvailable competitions:")
        print(competitions[['competition_name', 'season_name', 'country_name']].head(20))
    except Exception as e:
        print(f"Error: {e}")
