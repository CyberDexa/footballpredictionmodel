"""
Multi-League Data Fetcher Module
Fetches historical football data from multiple European leagues
"""

import pandas as pd
import requests
from io import StringIO
import os
import time
from typing import List, Optional, Dict
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np


class MultiLeagueDataFetcher:
    """Fetches historical match data from multiple European football leagues"""
    
    # Primary source: football-data.co.uk
    BASE_URL = "https://www.football-data.co.uk/mmz4281"
    
    # Headers to mimic browser request
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }
    
    # League codes and their details
    LEAGUES = {
        'EPL': {
            'name': 'English Premier League',
            'country': 'England',
            'code': 'E0',
            'flag': '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
            'teams': [
                'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton',
                'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Liverpool',
                'Man City', 'Man United', 'Newcastle', "Nott'm Forest", 'Tottenham',
                'West Ham', 'Wolves', 'Leicester', 'Ipswich', 'Southampton'
            ],
            'strength': {
                'Man City': 92, 'Liverpool': 90, 'Arsenal': 89, 'Chelsea': 84,
                'Man United': 82, 'Tottenham': 81, 'Newcastle': 80, 'Brighton': 77,
                'Aston Villa': 78, 'West Ham': 75, 'Brentford': 73, 'Crystal Palace': 72,
                'Fulham': 71, 'Wolves': 70, 'Bournemouth': 69, 'Everton': 68,
                "Nott'm Forest": 67, 'Leicester': 66, 'Ipswich': 62, 'Southampton': 63
            }
        },
        'LA_LIGA': {
            'name': 'La Liga',
            'country': 'Spain',
            'code': 'SP1',
            'flag': '🇪🇸',
            'teams': [
                'Real Madrid', 'Barcelona', 'Ath Madrid', 'Sevilla', 'Real Sociedad',
                'Villarreal', 'Ath Bilbao', 'Valencia', 'Betis', 'Celta',
                'Osasuna', 'Getafe', 'Mallorca', 'Girona', 'Rayo Vallecano',
                'Alaves', 'Las Palmas', 'Valladolid', 'Espanol', 'Leganes'
            ],
            'strength': {
                'Real Madrid': 93, 'Barcelona': 91, 'Ath Madrid': 86, 'Girona': 79,
                'Ath Bilbao': 78, 'Real Sociedad': 77, 'Villarreal': 76, 'Betis': 75,
                'Sevilla': 74, 'Valencia': 73, 'Celta': 71, 'Osasuna': 70,
                'Getafe': 69, 'Mallorca': 68, 'Rayo Vallecano': 67, 'Alaves': 65,
                'Las Palmas': 64, 'Valladolid': 63, 'Espanol': 62, 'Leganes': 61
            }
        },
        'SERIE_A': {
            'name': 'Serie A',
            'country': 'Italy',
            'code': 'I1',
            'flag': '🇮🇹',
            'teams': [
                'Inter', 'Juventus', 'AC Milan', 'Napoli', 'Roma',
                'Lazio', 'Atalanta', 'Fiorentina', 'Bologna', 'Torino',
                'Monza', 'Udinese', 'Sassuolo', 'Empoli', 'Lecce',
                'Verona', 'Cagliari', 'Genoa', 'Salernitana', 'Frosinone'
            ],
            'strength': {
                'Inter': 89, 'Juventus': 86, 'AC Milan': 85, 'Napoli': 84,
                'Atalanta': 82, 'Roma': 80, 'Lazio': 79, 'Fiorentina': 76,
                'Bologna': 75, 'Torino': 73, 'Monza': 70, 'Udinese': 69,
                'Sassuolo': 68, 'Empoli': 67, 'Lecce': 66, 'Verona': 65,
                'Cagliari': 64, 'Genoa': 63, 'Salernitana': 61, 'Frosinone': 60
            }
        },
        'BUNDESLIGA': {
            'name': 'Bundesliga',
            'country': 'Germany',
            'code': 'D1',
            'flag': '🇩🇪',
            'teams': [
                'Bayern Munich', 'Dortmund', 'RB Leipzig', 'Leverkusen', 'Frankfurt',
                'Wolfsburg', 'Freiburg', 'Union Berlin', 'Hoffenheim', 'Mainz',
                'Gladbach', 'Werder Bremen', 'Augsburg', 'Stuttgart', 'Bochum',
                'Koln', 'Heidenheim', 'Darmstadt', 'Holstein Kiel', 'St Pauli'
            ],
            'strength': {
                'Bayern Munich': 91, 'Leverkusen': 88, 'Dortmund': 85, 'RB Leipzig': 84,
                'Stuttgart': 80, 'Frankfurt': 78, 'Freiburg': 76, 'Hoffenheim': 74,
                'Wolfsburg': 73, 'Gladbach': 72, 'Werder Bremen': 71, 'Union Berlin': 70,
                'Mainz': 69, 'Augsburg': 67, 'Bochum': 65, 'Koln': 64,
                'Heidenheim': 63, 'Darmstadt': 61, 'Holstein Kiel': 60, 'St Pauli': 62
            }
        },
        'LIGUE_1': {
            'name': 'Ligue 1',
            'country': 'France',
            'code': 'F1',
            'flag': '🇫🇷',
            'teams': [
                'Paris SG', 'Monaco', 'Marseille', 'Lyon', 'Lille',
                'Nice', 'Lens', 'Rennes', 'Montpellier', 'Nantes',
                'Strasbourg', 'Toulouse', 'Brest', 'Reims', 'Lorient',
                'Le Havre', 'Metz', 'Clermont', 'Angers', 'Auxerre'
            ],
            'strength': {
                'Paris SG': 92, 'Monaco': 82, 'Marseille': 80, 'Lyon': 78,
                'Lille': 77, 'Lens': 76, 'Nice': 75, 'Rennes': 74,
                'Brest': 72, 'Toulouse': 71, 'Montpellier': 70, 'Nantes': 69,
                'Strasbourg': 68, 'Reims': 67, 'Lorient': 65, 'Le Havre': 64,
                'Metz': 63, 'Clermont': 62, 'Angers': 61, 'Auxerre': 60
            }
        },
        'EREDIVISIE': {
            'name': 'Eredivisie',
            'country': 'Netherlands',
            'code': 'N1',
            'flag': '🇳🇱',
            'teams': [
                'Ajax', 'PSV', 'Feyenoord', 'AZ Alkmaar', 'Twente',
                'Utrecht', 'Vitesse', 'Heerenveen', 'Groningen', 'Sparta Rotterdam',
                'NEC Nijmegen', 'Go Ahead Eagles', 'Fortuna Sittard', 'Heracles',
                'Volendam', 'Waalwijk', 'Excelsior', 'Emmen', 'Zwolle', 'Almere City'
            ],
            'strength': {
                'PSV': 85, 'Ajax': 83, 'Feyenoord': 82, 'AZ Alkmaar': 76,
                'Twente': 74, 'Utrecht': 72, 'Vitesse': 70, 'Heerenveen': 68,
                'Groningen': 67, 'Sparta Rotterdam': 66, 'NEC Nijmegen': 65, 
                'Go Ahead Eagles': 64, 'Fortuna Sittard': 63, 'Heracles': 62,
                'Volendam': 61, 'Waalwijk': 60, 'Excelsior': 59, 'Emmen': 58,
                'Zwolle': 57, 'Almere City': 56
            }
        },
        'PRIMEIRA_LIGA': {
            'name': 'Primeira Liga',
            'country': 'Portugal',
            'code': 'P1',
            'flag': '🇵🇹',
            'teams': [
                'Benfica', 'Porto', 'Sporting CP', 'Braga', 'Vitoria Guimaraes',
                'Rio Ave', 'Famalicao', 'Gil Vicente', 'Boavista', 'Santa Clara',
                'Estoril', 'Arouca', 'Vizela', 'Portimonense', 'Chaves',
                'Casa Pia', 'Maritimo', 'Estrela', 'Moreirense', 'Farense'
            ],
            'strength': {
                'Benfica': 84, 'Porto': 83, 'Sporting CP': 82, 'Braga': 76,
                'Vitoria Guimaraes': 72, 'Rio Ave': 68, 'Famalicao': 67, 
                'Gil Vicente': 66, 'Boavista': 65, 'Santa Clara': 64,
                'Estoril': 63, 'Arouca': 62, 'Vizela': 61, 'Portimonense': 60,
                'Chaves': 59, 'Casa Pia': 58, 'Maritimo': 57, 'Estrela': 56,
                'Moreirense': 55, 'Farense': 54
            }
        }
    }
    
    # Season codes - Updated for 2025-26 season
    SEASON_CODES = {
        "2025-26": "2526",
        "2024-25": "2425",
        "2023-24": "2324",
        "2022-23": "2223",
        "2021-22": "2122",
        "2020-21": "2021",
        "2019-20": "1920",
        "2018-19": "1819",
        "2017-18": "1718",
    }
    
    # Key columns to keep
    KEY_COLUMNS = [
        'Date', 'HomeTeam', 'AwayTeam', 
        'FTHG', 'FTAG', 'FTR',
        'HTHG', 'HTAG', 'HTR',
        'HS', 'AS', 'HST', 'AST',
        'HF', 'AF', 'HC', 'AC',
        'HY', 'AY', 'HR', 'AR',
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
    
    def get_available_leagues(self) -> Dict[str, Dict]:
        """Return all available leagues with their details"""
        return self.LEAGUES
    
    def fetch_league_season(self, league: str, season: str) -> Optional[pd.DataFrame]:
        """Fetch data for a specific league and season"""
        if league not in self.LEAGUES:
            print(f"League {league} not available")
            return None
        
        if season not in self.SEASON_CODES:
            print(f"Season {season} not available")
            return None
        
        league_code = self.LEAGUES[league]['code']
        season_code = self.SEASON_CODES[season]
        
        # Try multiple data sources
        urls = [
            f"{self.BASE_URL}/{season_code}/{league_code}.csv",
            f"https://raw.githubusercontent.com/openfootball/football.csv/master/{season_code}/{league_code}.csv",
        ]
        
        for url in urls:
            try:
                print(f"Fetching {league} {season}...")
                time.sleep(0.3)
                response = self.session.get(url, timeout=15)
                response.raise_for_status()
                
                df = pd.read_csv(StringIO(response.text), encoding='utf-8', on_bad_lines='skip')
                df['Season'] = season
                df['League'] = league
                
                available_cols = [col for col in self.KEY_COLUMNS if col in df.columns]
                available_cols.extend(['Season', 'League'])
                df = df[available_cols]
                
                print(f"  ✓ Fetched {len(df)} matches")
                return df
                
            except Exception as e:
                continue
        
        print(f"  ✗ Could not fetch {league} {season}")
        return None
    
    def fetch_league_data(self, league: str, seasons: Optional[List[str]] = None) -> pd.DataFrame:
        """Fetch all data for a specific league"""
        if seasons is None:
            seasons = list(self.SEASON_CODES.keys())
        
        all_data = []
        for season in seasons:
            df = self.fetch_league_season(league, season)
            if df is not None:
                all_data.append(df)
        
        if not all_data:
            print(f"⚠ Could not fetch data for {league}. Generating sample data...")
            return self._generate_sample_data(league)
        
        combined = pd.concat(all_data, ignore_index=True)
        print(f"✓ Total {league} matches: {len(combined)}")
        return combined
    
    def fetch_all_leagues(self, leagues: Optional[List[str]] = None) -> pd.DataFrame:
        """Fetch data for multiple leagues"""
        if leagues is None:
            leagues = list(self.LEAGUES.keys())
        
        all_data = []
        for league in leagues:
            df = self.fetch_league_data(league)
            if df is not None and len(df) > 0:
                all_data.append(df)
        
        if not all_data:
            raise ValueError("No data could be fetched!")
        
        combined = pd.concat(all_data, ignore_index=True)
        return combined
    
    def _generate_sample_data(self, league: str) -> pd.DataFrame:
        """Generate realistic sample data for a league including current 2025-26 season"""
        if league not in self.LEAGUES:
            return pd.DataFrame()
        
        league_info = self.LEAGUES[league]
        teams = league_info['teams'][:20]
        team_strength = league_info.get('strength', {})
        
        np.random.seed(42 + hash(league) % 1000)
        data = []
        
        # Include seasons up to current 2025-26
        seasons = ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25', '2025-26']
        base_date = pd.Timestamp('2019-08-01')
        
        for season_idx, season in enumerate(seasons):
            season_teams = teams.copy()
            
            # For 2025-26, only generate matches up to December 29, 2025 (matchday ~18)
            max_rounds = 18 if season == '2025-26' else 38
            
            for round_num in range(max_rounds):
                home_teams = season_teams[:10]
                away_teams = season_teams[10:]
                
                for home, away in zip(home_teams, away_teams):
                    home_str = team_strength.get(home, 70)
                    away_str = team_strength.get(away, 70)
                    
                    home_expected = 1.5 + (home_str - 70) / 30 + 0.3
                    away_expected = 1.2 + (away_str - 70) / 30
                    
                    home_goals = min(7, max(0, int(np.random.poisson(max(0.5, home_expected)))))
                    away_goals = min(7, max(0, int(np.random.poisson(max(0.5, away_expected)))))
                    
                    if home_goals > away_goals:
                        result = 'H'
                    elif away_goals > home_goals:
                        result = 'A'
                    else:
                        result = 'D'
                    
                    ht_home = min(home_goals, max(0, int(np.random.binomial(home_goals, 0.45))))
                    ht_away = min(away_goals, max(0, int(np.random.binomial(away_goals, 0.45))))
                    
                    if ht_home > ht_away:
                        ht_result = 'H'
                    elif ht_away > ht_home:
                        ht_result = 'A'
                    else:
                        ht_result = 'D'
                    
                    # Calculate proper date based on season
                    if season == '2025-26':
                        match_date = pd.Timestamp('2025-08-15') + pd.Timedelta(days=round_num * 7 + np.random.randint(0, 3))
                    elif season == '2024-25':
                        match_date = pd.Timestamp('2024-08-15') + pd.Timedelta(days=round_num * 7 + np.random.randint(0, 3))
                    else:
                        match_date = base_date + pd.Timedelta(days=season_idx * 365 + round_num * 7 + np.random.randint(0, 3))
                    
                    data.append({
                        'Date': match_date.strftime('%d/%m/%Y'),
                        'HomeTeam': home,
                        'AwayTeam': away,
                        'FTHG': home_goals,
                        'FTAG': away_goals,
                        'FTR': result,
                        'HTHG': ht_home,
                        'HTAG': ht_away,
                        'HTR': ht_result,
                        'HS': max(3, int(np.random.normal(13, 4))),
                        'AS': max(2, int(np.random.normal(11, 4))),
                        'HST': max(1, int(np.random.normal(5, 2))),
                        'AST': max(1, int(np.random.normal(4, 2))),
                        'HF': max(5, int(np.random.normal(11, 3))),
                        'AF': max(5, int(np.random.normal(12, 3))),
                        'HC': max(1, int(np.random.normal(5, 2))),
                        'AC': max(1, int(np.random.normal(5, 2))),
                        'HY': max(0, int(np.random.poisson(1.5))),
                        'AY': max(0, int(np.random.poisson(1.8))),
                        'HR': 1 if np.random.random() < 0.03 else 0,
                        'AR': 1 if np.random.random() < 0.04 else 0,
                        'Season': season,
                        'League': league
                    })
                
                season_teams = season_teams[1:] + season_teams[:1]
        
        df = pd.DataFrame(data)
        print(f"✓ Generated {len(df)} sample matches for {league} (up to Dec 2025)")
        return df
    
    def save_league_data(self, df: pd.DataFrame, league: str):
        """Save league data to CSV"""
        filepath = os.path.join(self.data_dir, f"{league.lower()}_data.csv")
        df.to_csv(filepath, index=False)
        print(f"✓ Saved to {filepath}")
        return filepath
    
    def load_league_data(self, league: str) -> pd.DataFrame:
        """Load league data from CSV"""
        filepath = os.path.join(self.data_dir, f"{league.lower()}_data.csv")
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        raise FileNotFoundError(f"No data file found at {filepath}")
    
    def get_or_fetch_league_data(self, league: str, force_refresh: bool = False) -> pd.DataFrame:
        """Get data from cache or fetch if not available"""
        filepath = os.path.join(self.data_dir, f"{league.lower()}_data.csv")
        
        if not force_refresh and os.path.exists(filepath):
            print(f"Loading cached {league} data...")
            return pd.read_csv(filepath)
        
        print(f"Fetching {league} data...")
        df = self.fetch_league_data(league)
        self.save_league_data(df, league)
        return df
    
    def get_league_teams(self, league: str) -> List[str]:
        """Get list of teams for a league"""
        if league in self.LEAGUES:
            return self.LEAGUES[league]['teams']
        return []


if __name__ == "__main__":
    fetcher = MultiLeagueDataFetcher(data_dir="../data")
    
    # Test fetching EPL
    print("\nAvailable Leagues:")
    for code, info in fetcher.get_available_leagues().items():
        print(f"  {info['flag']} {code}: {info['name']}")
    
    # Fetch EPL data
    df = fetcher.get_or_fetch_league_data('EPL')
    print(f"\nDataset shape: {df.shape}")
