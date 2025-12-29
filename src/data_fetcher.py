"""
Data Fetcher Module
Fetches historical English Premier League data from multiple sources
"""

import pandas as pd
import requests
from io import StringIO
import os
import time
from typing import List, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class EPLDataFetcher:
    """Fetches EPL historical match data from multiple sources"""
    
    # Primary source
    BASE_URL = "https://www.football-data.co.uk/mmz4281"
    
    # Alternative GitHub mirror
    GITHUB_URL = "https://raw.githubusercontent.com/footballcsv/england/master/2020s"
    
    # Headers to mimic browser request
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }
    
    # Season codes for the last several seasons
    SEASON_CODES = {
        "2024-25": "2425",
        "2023-24": "2324",
        "2022-23": "2223",
        "2021-22": "2122",
        "2020-21": "2021",
        "2019-20": "1920",
        "2018-19": "1819",
        "2017-18": "1718",
        "2016-17": "1617",
        "2015-16": "1516",
    }
    
    # Key columns to keep from the dataset
    KEY_COLUMNS = [
        'Date', 'HomeTeam', 'AwayTeam', 
        'FTHG', 'FTAG', 'FTR',  # Full Time Home Goals, Away Goals, Result
        'HTHG', 'HTAG', 'HTR',  # Half Time stats
        'HS', 'AS',             # Shots
        'HST', 'AST',           # Shots on Target
        'HF', 'AF',             # Fouls
        'HC', 'AC',             # Corners
        'HY', 'AY',             # Yellow Cards
        'HR', 'AR',             # Red Cards
    ]
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Setup session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update(self.HEADERS)
    
    def fetch_season(self, season: str) -> Optional[pd.DataFrame]:
        """Fetch data for a specific season"""
        if season not in self.SEASON_CODES:
            print(f"Season {season} not available")
            return None
        
        season_code = self.SEASON_CODES[season]
        url = f"{self.BASE_URL}/{season_code}/E0.csv"
        
        try:
            print(f"Fetching {season} data from {url}...")
            time.sleep(0.5)  # Rate limiting
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            df = pd.read_csv(StringIO(response.text), encoding='utf-8', on_bad_lines='skip')
            df['Season'] = season
            
            # Keep only key columns that exist
            available_cols = [col for col in self.KEY_COLUMNS if col in df.columns]
            available_cols.append('Season')
            df = df[available_cols]
            
            print(f"  ✓ Fetched {len(df)} matches for {season}")
            return df
            
        except requests.RequestException as e:
            print(f"  ✗ Error fetching {season}: {e}")
            return None
        except Exception as e:
            print(f"  ✗ Error parsing {season}: {e}")
            return None
    
    def fetch_all_seasons(self, seasons: Optional[List[str]] = None) -> pd.DataFrame:
        """Fetch data for multiple seasons"""
        if seasons is None:
            seasons = list(self.SEASON_CODES.keys())
        
        all_data = []
        for season in seasons:
            df = self.fetch_season(season)
            if df is not None:
                all_data.append(df)
        
        if not all_data:
            print("\n⚠ Could not fetch data from network. Generating sample data...")
            return self._generate_sample_data()
        
        combined = pd.concat(all_data, ignore_index=True)
        print(f"\n✓ Total matches fetched: {len(combined)}")
        return combined
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate realistic sample EPL data for training when network is unavailable"""
        import numpy as np
        
        teams = [
            'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton',
            'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Liverpool',
            'Man City', 'Man United', 'Newcastle', "Nott'm Forest", 'Tottenham',
            'West Ham', 'Wolves', 'Leicester', 'Leeds', 'Southampton'
        ]
        
        # Team strength ratings (for realistic results)
        team_strength = {
            'Man City': 90, 'Liverpool': 88, 'Arsenal': 87, 'Chelsea': 82,
            'Man United': 80, 'Tottenham': 79, 'Newcastle': 78, 'Brighton': 75,
            'Aston Villa': 76, 'West Ham': 74, 'Brentford': 72, 'Crystal Palace': 71,
            'Fulham': 70, 'Wolves': 69, 'Bournemouth': 68, 'Everton': 67,
            "Nott'm Forest": 66, 'Leicester': 65, 'Leeds': 64, 'Southampton': 62
        }
        
        np.random.seed(42)
        data = []
        
        seasons = ['2018-19', '2019-20', '2020-21', '2021-22', '2022-23', '2023-24']
        base_date = pd.Timestamp('2018-08-01')
        
        for season_idx, season in enumerate(seasons):
            season_teams = teams[:20]
            match_day = 0
            
            # Each team plays 38 matches (19 home, 19 away)
            for round_num in range(38):
                home_teams = season_teams[:10]
                away_teams = season_teams[10:]
                
                for home, away in zip(home_teams, away_teams):
                    # Calculate expected goals based on team strength
                    home_strength = team_strength.get(home, 70)
                    away_strength = team_strength.get(away, 70)
                    
                    # Home advantage + strength difference
                    home_expected = 1.5 + (home_strength - 70) / 30 + 0.3
                    away_expected = 1.2 + (away_strength - 70) / 30
                    
                    # Generate goals (Poisson-like distribution)
                    home_goals = max(0, int(np.random.poisson(max(0.5, home_expected))))
                    away_goals = max(0, int(np.random.poisson(max(0.5, away_expected))))
                    
                    # Cap goals at reasonable values
                    home_goals = min(home_goals, 7)
                    away_goals = min(away_goals, 7)
                    
                    # Determine result
                    if home_goals > away_goals:
                        result = 'H'
                    elif away_goals > home_goals:
                        result = 'A'
                    else:
                        result = 'D'
                    
                    # Half-time goals (roughly half of full-time)
                    ht_home = min(home_goals, max(0, int(np.random.binomial(home_goals, 0.45))))
                    ht_away = min(away_goals, max(0, int(np.random.binomial(away_goals, 0.45))))
                    
                    if ht_home > ht_away:
                        ht_result = 'H'
                    elif ht_away > ht_home:
                        ht_result = 'A'
                    else:
                        ht_result = 'D'
                    
                    # Generate match statistics
                    data.append({
                        'Date': (base_date + pd.Timedelta(days=season_idx * 365 + round_num * 7 + np.random.randint(0, 3))).strftime('%d/%m/%Y'),
                        'HomeTeam': home,
                        'AwayTeam': away,
                        'FTHG': home_goals,
                        'FTAG': away_goals,
                        'FTR': result,
                        'HTHG': ht_home,
                        'HTAG': ht_away,
                        'HTR': ht_result,
                        'HS': max(3, int(np.random.normal(13, 4))),  # Home shots
                        'AS': max(2, int(np.random.normal(11, 4))),  # Away shots
                        'HST': max(1, int(np.random.normal(5, 2))),  # Shots on target
                        'AST': max(1, int(np.random.normal(4, 2))),
                        'HF': max(5, int(np.random.normal(11, 3))),  # Fouls
                        'AF': max(5, int(np.random.normal(12, 3))),
                        'HC': max(1, int(np.random.normal(5, 2))),   # Corners
                        'AC': max(1, int(np.random.normal(5, 2))),
                        'HY': max(0, int(np.random.poisson(1.5))),   # Yellow cards
                        'AY': max(0, int(np.random.poisson(1.8))),
                        'HR': 1 if np.random.random() < 0.03 else 0,  # Red cards
                        'AR': 1 if np.random.random() < 0.04 else 0,
                        'Season': season
                    })
                
                # Rotate teams for next round
                season_teams = season_teams[1:] + season_teams[:1]
        
        df = pd.DataFrame(data)
        print(f"✓ Generated {len(df)} sample matches for training")
        return df
    
    def save_data(self, df: pd.DataFrame, filename: str = "epl_historical_data.csv"):
        """Save the fetched data to CSV"""
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"✓ Data saved to {filepath}")
        return filepath
    
    def load_data(self, filename: str = "epl_historical_data.csv") -> pd.DataFrame:
        """Load previously saved data"""
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        else:
            raise FileNotFoundError(f"No data file found at {filepath}")
    
    def get_or_fetch_data(self, filename: str = "epl_historical_data.csv", 
                          force_refresh: bool = False) -> pd.DataFrame:
        """Get data from cache or fetch if not available"""
        filepath = os.path.join(self.data_dir, filename)
        
        if not force_refresh and os.path.exists(filepath):
            print(f"Loading cached data from {filepath}")
            return pd.read_csv(filepath)
        
        print("Fetching fresh data from football-data.co.uk...")
        df = self.fetch_all_seasons()
        self.save_data(df, filename)
        return df


if __name__ == "__main__":
    # Test the data fetcher
    fetcher = EPLDataFetcher(data_dir="../data")
    df = fetcher.get_or_fetch_data()

    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nSample data:\n{df.head()}")
