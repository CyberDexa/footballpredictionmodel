"""
Football-Data.co.uk Data Fetcher
Fetches historical match data with betting odds from football-data.co.uk

This is a high-quality data source with:
- 31 seasons of results (1993-2026)
- 26 seasons of betting odds (2000-2026)
- Match statistics (shots, corners, fouls, cards)
- Odds from multiple bookmakers
"""

import os
import pandas as pd
import requests
from datetime import datetime
from typing import Optional, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FootballDataFetcher:
    """Fetches data from football-data.co.uk"""
    
    BASE_URL = "https://www.football-data.co.uk"
    
    # League codes and their divisions
    # Main leagues with full data (odds, stats)
    LEAGUES = {
        'EPL': {
            'name': 'English Premier League',
            'flag': '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
            'country': 'England',
            'code': 'E0',
            'folder': 'mmz4281',
            'seasons': range(1993, 2027)
        },
        'EFL_CHAMPIONSHIP': {
            'name': 'English Championship',
            'flag': '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
            'country': 'England',
            'code': 'E1',
            'folder': 'mmz4281',
            'seasons': range(1993, 2027)
        },
        'EFL_LEAGUE1': {
            'name': 'English League One',
            'flag': '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
            'country': 'England',
            'code': 'E2',
            'folder': 'mmz4281',
            'seasons': range(1993, 2027)
        },
        'EFL_LEAGUE2': {
            'name': 'English League Two',
            'flag': '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
            'country': 'England',
            'code': 'E3',
            'folder': 'mmz4281',
            'seasons': range(1993, 2027)
        },
        'LA_LIGA': {
            'name': 'Spanish La Liga',
            'flag': '🇪🇸',
            'country': 'Spain',
            'code': 'SP1',
            'folder': 'mmz4281',
            'seasons': range(1993, 2027)
        },
        'LA_LIGA2': {
            'name': 'Spanish Segunda División',
            'flag': '🇪🇸',
            'country': 'Spain',
            'code': 'SP2',
            'folder': 'mmz4281',
            'seasons': range(1993, 2027)
        },
        'SERIE_A': {
            'name': 'Italian Serie A',
            'flag': '🇮🇹',
            'country': 'Italy',
            'code': 'I1',
            'folder': 'mmz4281',
            'seasons': range(1993, 2027)
        },
        'SERIE_B': {
            'name': 'Italian Serie B',
            'flag': '🇮🇹',
            'country': 'Italy',
            'code': 'I2',
            'folder': 'mmz4281',
            'seasons': range(1993, 2027)
        },
        'BUNDESLIGA': {
            'name': 'German Bundesliga',
            'flag': '🇩🇪',
            'country': 'Germany',
            'code': 'D1',
            'folder': 'mmz4281',
            'seasons': range(1993, 2027)
        },
        'BUNDESLIGA2': {
            'name': 'German 2. Bundesliga',
            'flag': '🇩🇪',
            'country': 'Germany',
            'code': 'D2',
            'folder': 'mmz4281',
            'seasons': range(1993, 2027)
        },
        'LIGUE_1': {
            'name': 'French Ligue 1',
            'flag': '🇫🇷',
            'country': 'France',
            'code': 'F1',
            'folder': 'mmz4281',
            'seasons': range(1993, 2027)
        },
        'LIGUE_2': {
            'name': 'French Ligue 2',
            'flag': '🇫🇷',
            'country': 'France',
            'code': 'F2',
            'folder': 'mmz4281',
            'seasons': range(1993, 2027)
        },
        'EREDIVISIE': {
            'name': 'Dutch Eredivisie',
            'flag': '🇳🇱',
            'country': 'Netherlands',
            'code': 'N1',
            'folder': 'mmz4281',
            'seasons': range(1993, 2027)
        },
        'JUPILER': {
            'name': 'Belgian Jupiler League',
            'flag': '🇧🇪',
            'country': 'Belgium',
            'code': 'B1',
            'folder': 'mmz4281',
            'seasons': range(1995, 2027)
        },
        'PRIMEIRA': {
            'name': 'Portuguese Primeira Liga',
            'flag': '🇵🇹',
            'country': 'Portugal',
            'code': 'P1',
            'folder': 'mmz4281',
            'seasons': range(1994, 2027)
        },
        'SUPER_LIG': {
            'name': 'Turkish Süper Lig',
            'flag': '🇹🇷',
            'country': 'Turkey',
            'code': 'T1',
            'folder': 'mmz4281',
            'seasons': range(1994, 2027)
        },
        'SUPER_LEAGUE_GR': {
            'name': 'Greek Super League',
            'flag': '🇬🇷',
            'country': 'Greece',
            'code': 'G1',
            'folder': 'mmz4281',
            'seasons': range(1994, 2027)
        },
        'SCOTTISH_PREM': {
            'name': 'Scottish Premiership',
            'flag': '🏴󠁧󠁢󠁳󠁣󠁴󠁿',
            'country': 'Scotland',
            'code': 'SC0',
            'folder': 'mmz4281',
            'seasons': range(1994, 2027)
        },
    }
    
    # Extra leagues (results and basic odds only)
    EXTRA_LEAGUES = {
        'MLS': {
            'name': 'MLS',
            'flag': '🇺🇸',
            'country': 'USA',
            'code': 'USA',
            'folder': 'new',
            'seasons': range(2012, 2027)
        },
        'SUPERLIGA_ARG': {
            'name': 'Argentine Primera División',
            'flag': '🇦🇷',
            'country': 'Argentina',
            'code': 'ARG',
            'folder': 'new',
            'seasons': range(2012, 2027)
        },
        'SERIE_A_BR': {
            'name': 'Brazilian Série A',
            'flag': '🇧🇷',
            'country': 'Brazil',
            'code': 'BRA',
            'folder': 'new',
            'seasons': range(2012, 2027)
        },
        'J_LEAGUE': {
            'name': 'Japanese J-League',
            'flag': '🇯🇵',
            'country': 'Japan',
            'code': 'JPN',
            'folder': 'new',
            'seasons': range(2012, 2027)
        },
    }
    
    def __init__(self, data_dir: str = "data/footballdata"):
        """Initialize the fetcher"""
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Combine all leagues
        self.all_leagues = {**self.LEAGUES, **self.EXTRA_LEAGUES}
    
    def _season_code(self, year: int) -> str:
        """Convert year to season code (e.g., 2025 -> '2526')"""
        year_short = year % 100
        next_year = (year + 1) % 100
        return f"{year_short:02d}{next_year:02d}"
    
    def _get_csv_url(self, league_code: str, season_year: int) -> str:
        """Get the CSV download URL for a league and season"""
        league = self.all_leagues.get(league_code)
        if not league:
            raise ValueError(f"Unknown league: {league_code}")
        
        season = self._season_code(season_year)
        folder = league['folder']
        code = league['code']
        
        if folder == 'mmz4281':
            return f"{self.BASE_URL}/{folder}/{season}/{code}.csv"
        else:
            # Extra leagues use different URL structure
            return f"{self.BASE_URL}/{folder}/{code}.csv"
    
    def fetch_season(self, league_code: str, season_year: int, 
                     force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Fetch data for a specific league and season
        
        Args:
            league_code: League code (e.g., 'EPL', 'LA_LIGA')
            season_year: Starting year of season (e.g., 2025 for 2025/26)
            force_refresh: Force download even if cached
        
        Returns:
            DataFrame with match data or None if not available
        """
        league = self.all_leagues.get(league_code)
        if not league:
            logger.error(f"Unknown league: {league_code}")
            return None
        
        # Check season availability
        if season_year not in league['seasons']:
            logger.warning(f"Season {season_year}/{season_year+1} not available for {league_code}")
            return None
        
        # Cache file path
        cache_file = os.path.join(
            self.data_dir, 
            f"{league_code}_{self._season_code(season_year)}.csv"
        )
        
        # Check cache
        if not force_refresh and os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file)
                logger.info(f"Loaded {len(df)} matches from cache: {cache_file}")
                return self._standardize_columns(df, league_code)
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")
        
        # Download fresh data
        url = self._get_csv_url(league_code, season_year)
        logger.info(f"Downloading: {url}")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save to cache
            with open(cache_file, 'wb') as f:
                f.write(response.content)
            
            # Parse CSV
            df = pd.read_csv(cache_file, encoding='utf-8', on_bad_lines='skip')
            logger.info(f"Downloaded {len(df)} matches for {league_code} {season_year}/{season_year+1}")
            
            return self._standardize_columns(df, league_code)
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Data not yet available for {league_code} {season_year}/{season_year+1}")
            else:
                logger.error(f"HTTP error: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            return None
    
    def _standardize_columns(self, df: pd.DataFrame, league_code: str) -> pd.DataFrame:
        """Standardize column names for consistency"""
        # Standard column mapping
        column_map = {
            'Div': 'Division',
            'HomeTeam': 'HomeTeam',
            'AwayTeam': 'AwayTeam',
            'FTHG': 'FTHG',  # Full Time Home Goals
            'FTAG': 'FTAG',  # Full Time Away Goals
            'FTR': 'FTR',    # Full Time Result (H/D/A)
            'HTHG': 'HTHG',  # Half Time Home Goals
            'HTAG': 'HTAG',  # Half Time Away Goals
            'HTR': 'HTR',    # Half Time Result
            'HS': 'HomeShots',
            'AS': 'AwayShots',
            'HST': 'HomeShotsTarget',
            'AST': 'AwayShotsTarget',
            'HF': 'HomeFouls',
            'AF': 'AwayFouls',
            'HC': 'HomeCorners',
            'AC': 'AwayCorners',
            'HY': 'HomeYellow',
            'AY': 'AwayYellow',
            'HR': 'HomeRed',
            'AR': 'AwayRed',
            # Betting odds columns (various bookmakers)
            'B365H': 'B365_Home',
            'B365D': 'B365_Draw',
            'B365A': 'B365_Away',
            'BWH': 'BW_Home',
            'BWD': 'BW_Draw',
            'BWA': 'BW_Away',
            'IWH': 'IW_Home',
            'IWD': 'IW_Draw',
            'IWA': 'IW_Away',
            'PSH': 'PS_Home',
            'PSD': 'PS_Draw',
            'PSA': 'PS_Away',
            'WHH': 'WH_Home',
            'WHD': 'WH_Draw',
            'WHA': 'WH_Away',
            'VCH': 'VC_Home',
            'VCD': 'VC_Draw',
            'VCA': 'VC_Away',
            # Over/Under odds
            'BbAv>2.5': 'AvgOver25',
            'BbAv<2.5': 'AvgUnder25',
            # Average odds
            'BbAvH': 'AvgHome',
            'BbAvD': 'AvgDraw',
            'BbAvA': 'AvgAway',
            # Maximum odds
            'BbMxH': 'MaxHome',
            'BbMxD': 'MaxDraw',
            'BbMxA': 'MaxAway',
        }
        
        # Rename columns that exist
        rename_dict = {k: v for k, v in column_map.items() if k in df.columns}
        df = df.rename(columns=rename_dict)
        
        # Add league code
        df['League'] = league_code
        
        # Parse date
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        
        return df
    
    def load_league_data(self, league_code: str, 
                         seasons: Optional[List[int]] = None,
                         force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Load data for a league across multiple seasons
        
        Args:
            league_code: League code
            seasons: List of season years, defaults to last 5 seasons
            force_refresh: Force re-download
        
        Returns:
            Combined DataFrame
        """
        league = self.all_leagues.get(league_code)
        if not league:
            return None
        
        if seasons is None:
            # Default to current and last 4 seasons
            current_year = datetime.now().year
            if datetime.now().month < 8:
                current_year -= 1  # Season hasn't started yet
            seasons = list(range(current_year - 4, current_year + 1))
        
        all_data = []
        for year in seasons:
            if year in league['seasons']:
                df = self.fetch_season(league_code, year, force_refresh)
                if df is not None and len(df) > 0:
                    df['Season'] = f"{year}/{year+1}"
                    all_data.append(df)
        
        if not all_data:
            return None
        
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.sort_values('Date', ascending=False).reset_index(drop=True)
        
        logger.info(f"Loaded {len(combined)} total matches for {league_code}")
        return combined
    
    def get_available_leagues(self) -> Dict:
        """Get dictionary of available leagues"""
        return {
            code: {
                'name': info['name'],
                'flag': info['flag'],
                'country': info['country']
            }
            for code, info in self.all_leagues.items()
        }
    
    def get_best_odds(self, df: pd.DataFrame, match_idx: int) -> Dict:
        """
        Get the best available odds for a match
        
        Args:
            df: DataFrame with match data
            match_idx: Index of the match
        
        Returns:
            Dict with best home, draw, away odds
        """
        row = df.iloc[match_idx]
        
        bookmakers = ['B365', 'BW', 'IW', 'PS', 'WH', 'VC']
        
        best_home = 0
        best_draw = 0
        best_away = 0
        
        for bm in bookmakers:
            home_col = f"{bm}_Home"
            draw_col = f"{bm}_Draw"
            away_col = f"{bm}_Away"
            
            if home_col in row and pd.notna(row[home_col]):
                best_home = max(best_home, float(row[home_col]))
            if draw_col in row and pd.notna(row[draw_col]):
                best_draw = max(best_draw, float(row[draw_col]))
            if away_col in row and pd.notna(row[away_col]):
                best_away = max(best_away, float(row[away_col]))
        
        # Fall back to average if no individual bookmaker odds
        if best_home == 0 and 'AvgHome' in row:
            best_home = float(row['AvgHome']) if pd.notna(row['AvgHome']) else 0
        if best_draw == 0 and 'AvgDraw' in row:
            best_draw = float(row['AvgDraw']) if pd.notna(row['AvgDraw']) else 0
        if best_away == 0 and 'AvgAway' in row:
            best_away = float(row['AvgAway']) if pd.notna(row['AvgAway']) else 0
        
        return {
            'home': best_home,
            'draw': best_draw,
            'away': best_away
        }
    
    def get_match_stats(self, df: pd.DataFrame, match_idx: int) -> Dict:
        """
        Get match statistics for a specific match
        
        Args:
            df: DataFrame with match data
            match_idx: Index of the match
        
        Returns:
            Dict with match statistics
        """
        row = df.iloc[match_idx]
        
        stats = {
            'home_team': row.get('HomeTeam', ''),
            'away_team': row.get('AwayTeam', ''),
            'date': row.get('Date', ''),
            'score': f"{int(row.get('FTHG', 0))}-{int(row.get('FTAG', 0))}",
            'ht_score': f"{int(row.get('HTHG', 0))}-{int(row.get('HTAG', 0))}",
            'result': row.get('FTR', ''),
        }
        
        # Add match statistics if available
        stat_cols = [
            ('HomeShots', 'AwayShots', 'shots'),
            ('HomeShotsTarget', 'AwayShotsTarget', 'shots_on_target'),
            ('HomeCorners', 'AwayCorners', 'corners'),
            ('HomeFouls', 'AwayFouls', 'fouls'),
            ('HomeYellow', 'AwayYellow', 'yellow_cards'),
            ('HomeRed', 'AwayRed', 'red_cards'),
        ]
        
        for home_col, away_col, name in stat_cols:
            if home_col in row and pd.notna(row[home_col]):
                stats[f'home_{name}'] = int(row[home_col])
                stats[f'away_{name}'] = int(row[away_col]) if pd.notna(row[away_col]) else 0
        
        return stats


# Singleton instance
_fetcher = None

def get_footballdata_fetcher() -> FootballDataFetcher:
    """Get the singleton FootballDataFetcher instance"""
    global _fetcher
    if _fetcher is None:
        _fetcher = FootballDataFetcher()
    return _fetcher


if __name__ == "__main__":
    # Test the fetcher
    fetcher = get_footballdata_fetcher()
    
    print("Available leagues:")
    for code, info in fetcher.get_available_leagues().items():
        print(f"  {info['flag']} {code}: {info['name']}")
    
    print("\nFetching EPL 2024/25...")
    df = fetcher.fetch_season('EPL', 2024)
    
    if df is not None:
        print(f"\nLoaded {len(df)} matches")
        print(f"Columns: {list(df.columns)}")
        print(f"\nRecent matches:")
        print(df[['Date', 'HomeTeam', 'FTHG', 'FTAG', 'AwayTeam', 'FTR']].head(10))
        
        if 'B365_Home' in df.columns:
            print(f"\nOdds available: Yes")
            print(df[['HomeTeam', 'AwayTeam', 'B365_Home', 'B365_Draw', 'B365_Away']].head(5))
