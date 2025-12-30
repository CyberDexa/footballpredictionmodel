"""
Football API Integration (API-Football via RapidAPI)
Free tier: 100 requests/day
https://www.api-football.com/

Provides:
- Player statistics (top scorers, assists, cards)
- Injuries and suspensions
- Team lineups
- Live scores
"""

import requests
import os
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json

class FootballAPI:
    """Client for API-Football - player stats and injuries"""
    
    # Use direct API-Football endpoint (not RapidAPI)
    BASE_URL = "https://v3.football.api-sports.io"
    
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
        'EUROPA_LEAGUE': 3,
        'SCOTTISH_PREM': 179,
        'CHAMPIONSHIP': 40,
    }
    
    # Free plan only supports seasons 2021-2023
    # Use 2023 as default for current data
    SEASONS = {
        2025: '2023',  # Free plan: use 2023 data
        2024: '2023',
        2023: '2022',
    }
    DEFAULT_SEASON = '2023'
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with API-Football key (direct, not RapidAPI)"""
        self.api_key = api_key or os.getenv('FOOTBALL_API_KEY')
        self.cache = {}
        self.cache_expiry = {}
        self.requests_today = 0
        self.last_request_date = None
    
    def is_configured(self) -> bool:
        """Check if API key is configured"""
        return bool(self.api_key)
    
    def _get_headers(self) -> Dict:
        """Get API request headers for direct API-Football"""
        return {
            'x-apisports-key': self.api_key
        }
    
    def _make_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Make API request with error handling"""
        if not self.is_configured():
            return None
        
        # Track daily usage
        today = datetime.now().date()
        if self.last_request_date != today:
            self.requests_today = 0
            self.last_request_date = today
        
        if self.requests_today >= 100:
            print("Daily API limit reached (100 requests)")
            return None
        
        try:
            response = requests.get(
                f"{self.BASE_URL}/{endpoint}",
                headers=self._get_headers(),
                params=params
            )
            self.requests_today += 1
            
            if response.status_code == 200:
                data = response.json()
                if data.get('errors'):
                    print(f"API Error: {data['errors']}")
                    return None
                return data
            elif response.status_code == 429:
                print("Rate limit exceeded")
            elif response.status_code == 403:
                print("Invalid or expired API key")
            return None
        except Exception as e:
            print(f"API request error: {e}")
            return None
    
    def get_top_scorers(self, league: str, season: str = '2023', limit: int = 10) -> List[Dict]:
        """
        Get top scorers for a league
        
        Args:
            league: League code (EPL, LA_LIGA, etc.)
            season: Season year (e.g., '2024' for 2024-25)
            limit: Number of players to return
        
        Returns:
            List of top scorers with goals and stats
        """
        league_id = self.LEAGUE_IDS.get(league)
        if not league_id:
            return []
        
        cache_key = f"scorers_{league}_{season}"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key][:limit]
        
        data = self._make_request('players/topscorers', {
            'league': league_id,
            'season': season
        })
        
        if not data:
            return []
        
        scorers = []
        for player_data in data.get('response', []):
            player = player_data.get('player', {})
            stats = player_data.get('statistics', [{}])[0]
            
            scorers.append({
                'name': player.get('name'),
                'photo': player.get('photo'),
                'team': stats.get('team', {}).get('name'),
                'team_logo': stats.get('team', {}).get('logo'),
                'goals': stats.get('goals', {}).get('total', 0),
                'assists': stats.get('goals', {}).get('assists', 0),
                'games': stats.get('games', {}).get('appearences', 0),
                'minutes': stats.get('games', {}).get('minutes', 0),
                'rating': stats.get('games', {}).get('rating'),
                'penalties': stats.get('penalty', {}).get('scored', 0),
            })
        
        self._cache_response(cache_key, scorers, hours=6)
        return scorers[:limit]
    
    def get_top_assists(self, league: str, season: str = '2023', limit: int = 10) -> List[Dict]:
        """Get top assist providers for a league"""
        league_id = self.LEAGUE_IDS.get(league)
        if not league_id:
            return []
        
        cache_key = f"assists_{league}_{season}"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key][:limit]
        
        data = self._make_request('players/topassists', {
            'league': league_id,
            'season': season
        })
        
        if not data:
            return []
        
        assists = []
        for player_data in data.get('response', []):
            player = player_data.get('player', {})
            stats = player_data.get('statistics', [{}])[0]
            
            assists.append({
                'name': player.get('name'),
                'photo': player.get('photo'),
                'team': stats.get('team', {}).get('name'),
                'team_logo': stats.get('team', {}).get('logo'),
                'assists': stats.get('goals', {}).get('assists', 0),
                'goals': stats.get('goals', {}).get('total', 0),
                'games': stats.get('games', {}).get('appearences', 0),
                'key_passes': stats.get('passes', {}).get('key', 0),
            })
        
        self._cache_response(cache_key, assists, hours=6)
        return assists[:limit]
    
    def get_team_injuries(self, team_name: str, league: str) -> List[Dict]:
        """
        Get current injuries for a team
        
        Args:
            team_name: Team name to search
            league: League code
        
        Returns:
            List of injured players with details
        """
        # First, find team ID
        team_id = self._get_team_id(team_name, league)
        if not team_id:
            return []
        
        cache_key = f"injuries_{team_id}"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        data = self._make_request('injuries', {
            'team': team_id,
            'season': '2024'
        })
        
        if not data:
            return []
        
        injuries = []
        for injury_data in data.get('response', []):
            player = injury_data.get('player', {})
            
            injuries.append({
                'name': player.get('name'),
                'photo': player.get('photo'),
                'type': player.get('type'),  # 'Injury', 'Suspension'
                'reason': player.get('reason'),
                'date': injury_data.get('fixture', {}).get('date'),
            })
        
        self._cache_response(cache_key, injuries, hours=12)
        return injuries
    
    def get_team_squad(self, team_name: str, league: str) -> List[Dict]:
        """Get team squad with player details"""
        team_id = self._get_team_id(team_name, league)
        if not team_id:
            return []
        
        cache_key = f"squad_{team_id}"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        data = self._make_request('players/squads', {
            'team': team_id
        })
        
        if not data:
            return []
        
        players = []
        for team_data in data.get('response', []):
            for player in team_data.get('players', []):
                players.append({
                    'name': player.get('name'),
                    'age': player.get('age'),
                    'number': player.get('number'),
                    'position': player.get('position'),
                    'photo': player.get('photo'),
                })
        
        self._cache_response(cache_key, players, hours=24)
        return players
    
    def get_standings(self, league: str, season: str = '2023') -> List[Dict]:
        """Get league standings/table"""
        league_id = self.LEAGUE_IDS.get(league)
        if not league_id:
            return []
        
        cache_key = f"standings_{league}_{season}"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        data = self._make_request('standings', {
            'league': league_id,
            'season': season
        })
        
        if not data:
            return []
        
        standings = []
        response = data.get('response', [])
        if response:
            league_data = response[0].get('league', {})
            for standing in league_data.get('standings', [[]])[0]:
                standings.append({
                    'rank': standing.get('rank'),
                    'team': standing.get('team', {}).get('name'),
                    'logo': standing.get('team', {}).get('logo'),
                    'points': standing.get('points'),
                    'played': standing.get('all', {}).get('played', 0),
                    'wins': standing.get('all', {}).get('win', 0),
                    'draws': standing.get('all', {}).get('draw', 0),
                    'losses': standing.get('all', {}).get('lose', 0),
                    'goals_for': standing.get('all', {}).get('goals', {}).get('for', 0),
                    'goals_against': standing.get('all', {}).get('goals', {}).get('against', 0),
                    'goal_diff': standing.get('goalsDiff'),
                    'form': standing.get('form'),
                })
        
        self._cache_response(cache_key, standings, hours=1)
        return standings
    
    def _get_team_id(self, team_name: str, league: str) -> Optional[int]:
        """Get team ID from team name"""
        league_id = self.LEAGUE_IDS.get(league)
        if not league_id:
            return None
        
        cache_key = f"team_id_{team_name}_{league}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        data = self._make_request('teams', {
            'league': league_id,
            'season': '2024',
            'search': team_name
        })
        
        if not data or not data.get('response'):
            return None
        
        # Find best match
        team_lower = team_name.lower()
        for team_data in data.get('response', []):
            team = team_data.get('team', {})
            if team_lower in team.get('name', '').lower():
                team_id = team.get('id')
                self.cache[cache_key] = team_id
                return team_id
        
        # Return first result if no exact match
        first_team = data['response'][0].get('team', {})
        team_id = first_team.get('id')
        self.cache[cache_key] = team_id
        return team_id
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if key not in self.cache:
            return False
        expiry = self.cache_expiry.get(key)
        if not expiry:
            return False
        return datetime.now() < expiry
    
    def _cache_response(self, key: str, data: any, hours: int = 1):
        """Cache response for specified hours"""
        self.cache[key] = data
        self.cache_expiry[key] = datetime.now() + timedelta(hours=hours)
    
    def get_usage_stats(self) -> Dict:
        """Get API usage statistics"""
        return {
            'requests_today': self.requests_today,
            'requests_remaining': 100 - self.requests_today,
            'limit': 100,  # Free tier daily limit
            'last_request_date': str(self.last_request_date) if self.last_request_date else None
        }


# Singleton instance
_football_api = None

def get_football_api(api_key: Optional[str] = None) -> FootballAPI:
    """Get singleton instance of FootballAPI"""
    global _football_api
    if _football_api is None:
        _football_api = FootballAPI(api_key)
    elif api_key and api_key != _football_api.api_key:
        _football_api = FootballAPI(api_key)
    return _football_api
