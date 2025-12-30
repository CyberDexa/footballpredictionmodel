"""
TheSportsDB API Integration
FREE - No API key required!
https://www.thesportsdb.com/

Provides:
- Team information and logos
- Player photos and bios
- Recent match results
- Upcoming fixtures
- Stadium information
"""

import requests
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json


class TheSportsDBAPI:
    """Client for TheSportsDB - team/player info and photos"""
    
    # Free tier API key (public)
    BASE_URL = "https://www.thesportsdb.com/api/v1/json/3"
    
    # League IDs for TheSportsDB
    LEAGUE_IDS = {
        'EPL': 4328,              # English Premier League
        'CHAMPIONSHIP': 4329,      # English Championship
        'LEAGUE_ONE': 4396,        # English League 1
        'LEAGUE_TWO': 4397,        # English League 2
        'LA_LIGA': 4335,           # Spanish La Liga
        'SERIE_A': 4332,           # Italian Serie A
        'BUNDESLIGA': 4331,        # German Bundesliga
        'LIGUE_1': 4334,           # French Ligue 1
        'EREDIVISIE': 4337,        # Dutch Eredivisie
        'PRIMEIRA_LIGA': 4344,     # Portuguese Primeira Liga
        'SCOTTISH_PREM': 4330,     # Scottish Premiership
        'CHAMPIONS_LEAGUE': 4480,  # UEFA Champions League
        'EUROPA_LEAGUE': 4481,     # UEFA Europa League
        'FA_CUP': 4482,            # FA Cup
        'EFL_CUP': 4570,           # EFL Cup
    }
    
    def __init__(self):
        """Initialize - no API key needed!"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'FootballPredictionModel/1.0'
        })
        self.cache = {}
        self.cache_expiry = {}
    
    def _is_cache_valid(self, key: str, hours: int = 24) -> bool:
        """Check if cached data is still valid"""
        if key not in self.cache:
            return False
        expiry = self.cache_expiry.get(key)
        if not expiry:
            return False
        return datetime.now() < expiry
    
    def _cache_data(self, key: str, data, hours: int = 24):
        """Cache data with expiry"""
        self.cache[key] = data
        self.cache_expiry[key] = datetime.now() + timedelta(hours=hours)
    
    def _make_request(self, endpoint: str) -> Optional[Dict]:
        """Make API request"""
        try:
            url = f"{self.BASE_URL}/{endpoint}"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f"TheSportsDB request error: {e}")
            return None
    
    def get_team_info(self, team_name: str) -> Optional[Dict]:
        """
        Get team information including logo and stadium
        
        Returns:
            Dict with team info, logo, stadium, etc.
        """
        cache_key = f"team_{team_name}"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        data = self._make_request(f"searchteams.php?t={team_name}")
        if not data or not data.get('teams'):
            return None
        
        team = data['teams'][0]
        result = {
            'id': team.get('idTeam'),
            'name': team.get('strTeam'),
            'short_name': team.get('strTeamShort'),
            'alternate_names': team.get('strTeamAlternate'),
            'logo': team.get('strBadge'),  # Team badge/logo URL
            'jersey': team.get('strEquipment'),  # Jersey image
            'stadium': team.get('strStadium'),
            'stadium_capacity': team.get('intStadiumCapacity'),
            'stadium_image': team.get('strStadiumThumb'),
            'location': team.get('strLocation'),
            'country': team.get('strCountry'),
            'league': team.get('strLeague'),
            'formed_year': team.get('intFormedYear'),
            'website': team.get('strWebsite'),
            'facebook': team.get('strFacebook'),
            'twitter': team.get('strTwitter'),
            'instagram': team.get('strInstagram'),
            'description': team.get('strDescriptionEN'),
            'banner': team.get('strBanner'),  # Wide banner image
        }
        
        self._cache_data(cache_key, result)
        return result
    
    def get_player_info(self, player_name: str) -> Optional[Dict]:
        """
        Get player information including photo
        
        Returns:
            Dict with player info, photo, nationality, etc.
        """
        cache_key = f"player_{player_name}"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        data = self._make_request(f"searchplayers.php?p={player_name}")
        if not data or not data.get('player'):
            return None
        
        player = data['player'][0]
        result = {
            'id': player.get('idPlayer'),
            'name': player.get('strPlayer'),
            'team': player.get('strTeam'),
            'team_id': player.get('idTeam'),
            'nationality': player.get('strNationality'),
            'position': player.get('strPosition'),
            'photo': player.get('strThumb'),  # Player photo URL
            'cutout': player.get('strCutout'),  # Cutout image (no background)
            'render': player.get('strRender'),  # Action render
            'date_of_birth': player.get('dateBorn'),
            'height': player.get('strHeight'),
            'weight': player.get('strWeight'),
            'description': player.get('strDescriptionEN'),
            'signing_date': player.get('dateSigned'),
            'wage': player.get('strWage'),
            'birth_location': player.get('strBirthLocation'),
            'facebook': player.get('strFacebook'),
            'twitter': player.get('strTwitter'),
            'instagram': player.get('strInstagram'),
        }
        
        self._cache_data(cache_key, result)
        return result
    
    def get_team_players(self, team_name: str) -> List[Dict]:
        """
        Get all players for a team
        
        Returns:
            List of player dicts with photos
        """
        # First get team ID
        team = self.get_team_info(team_name)
        if not team:
            return []
        
        cache_key = f"team_players_{team['id']}"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        data = self._make_request(f"lookup_all_players.php?id={team['id']}")
        if not data or not data.get('player'):
            return []
        
        players = []
        for p in data['player']:
            players.append({
                'id': p.get('idPlayer'),
                'name': p.get('strPlayer'),
                'position': p.get('strPosition'),
                'nationality': p.get('strNationality'),
                'photo': p.get('strThumb'),
                'cutout': p.get('strCutout'),
                'date_of_birth': p.get('dateBorn'),
                'number': p.get('strNumber'),
            })
        
        self._cache_data(cache_key, players)
        return players
    
    def get_recent_results(self, league: str, limit: int = 15) -> List[Dict]:
        """
        Get recent match results for a league
        Note: Free tier limited to ~15 most recent
        
        Returns:
            List of recent match results
        """
        league_id = self.LEAGUE_IDS.get(league)
        if not league_id:
            return []
        
        cache_key = f"recent_{league}"
        if self._is_cache_valid(cache_key, hours=1):
            return self.cache[cache_key][:limit]
        
        data = self._make_request(f"eventspastleague.php?id={league_id}")
        if not data or not data.get('events'):
            return []
        
        results = []
        for e in data['events']:
            results.append({
                'id': e.get('idEvent'),
                'date': e.get('dateEvent'),
                'time': e.get('strTime'),
                'home_team': e.get('strHomeTeam'),
                'away_team': e.get('strAwayTeam'),
                'home_score': e.get('intHomeScore'),
                'away_score': e.get('intAwayScore'),
                'home_badge': e.get('strHomeTeamBadge'),
                'away_badge': e.get('strAwayTeamBadge'),
                'venue': e.get('strVenue'),
                'status': e.get('strStatus'),
                'video': e.get('strVideo'),  # Highlights video URL
            })
        
        self._cache_data(cache_key, results, hours=1)
        return results[:limit]
    
    def get_upcoming_fixtures(self, league: str, limit: int = 15) -> List[Dict]:
        """
        Get upcoming fixtures for a league
        Note: Free tier limited to ~15 upcoming
        
        Returns:
            List of upcoming fixtures
        """
        league_id = self.LEAGUE_IDS.get(league)
        if not league_id:
            return []
        
        cache_key = f"upcoming_{league}"
        if self._is_cache_valid(cache_key, hours=1):
            return self.cache[cache_key][:limit]
        
        data = self._make_request(f"eventsnextleague.php?id={league_id}")
        if not data or not data.get('events'):
            return []
        
        fixtures = []
        for e in data['events']:
            fixtures.append({
                'id': e.get('idEvent'),
                'date': e.get('dateEvent'),
                'time': e.get('strTime'),
                'home_team': e.get('strHomeTeam'),
                'away_team': e.get('strAwayTeam'),
                'home_badge': e.get('strHomeTeamBadge'),
                'away_badge': e.get('strAwayTeamBadge'),
                'venue': e.get('strVenue'),
            })
        
        self._cache_data(cache_key, fixtures, hours=1)
        return fixtures[:limit]
    
    def get_team_next_match(self, team_name: str) -> Optional[Dict]:
        """
        Get next scheduled match for a team
        
        Returns:
            Dict with next match info
        """
        team = self.get_team_info(team_name)
        if not team:
            return None
        
        data = self._make_request(f"eventsnext.php?id={team['id']}")
        if not data or not data.get('events'):
            return None
        
        e = data['events'][0]
        return {
            'date': e.get('dateEvent'),
            'time': e.get('strTime'),
            'home_team': e.get('strHomeTeam'),
            'away_team': e.get('strAwayTeam'),
            'venue': e.get('strVenue'),
            'league': e.get('strLeague'),
        }
    
    def get_team_last_matches(self, team_name: str, limit: int = 5) -> List[Dict]:
        """
        Get last matches for a team
        
        Returns:
            List of recent match results
        """
        team = self.get_team_info(team_name)
        if not team:
            return []
        
        data = self._make_request(f"eventslast.php?id={team['id']}")
        if not data or not data.get('results'):
            return []
        
        results = []
        for e in data['results'][:limit]:
            results.append({
                'date': e.get('dateEvent'),
                'home_team': e.get('strHomeTeam'),
                'away_team': e.get('strAwayTeam'),
                'home_score': e.get('intHomeScore'),
                'away_score': e.get('intAwayScore'),
                'league': e.get('strLeague'),
            })
        
        return results
    
    def get_league_teams(self, league: str) -> List[Dict]:
        """
        Get all teams in a league with logos
        
        Returns:
            List of teams with badges
        """
        league_id = self.LEAGUE_IDS.get(league)
        if not league_id:
            return []
        
        cache_key = f"league_teams_{league}"
        if self._is_cache_valid(cache_key, hours=24):
            return self.cache[cache_key]
        
        data = self._make_request(f"lookup_all_teams.php?id={league_id}")
        if not data or not data.get('teams'):
            return []
        
        teams = []
        for t in data['teams']:
            teams.append({
                'id': t.get('idTeam'),
                'name': t.get('strTeam'),
                'short_name': t.get('strTeamShort'),
                'logo': t.get('strBadge'),
                'stadium': t.get('strStadium'),
                'location': t.get('strLocation'),
            })
        
        self._cache_data(cache_key, teams, hours=24)
        return teams
    
    def search_all(self, query: str) -> Dict:
        """
        Search for teams and players matching query
        
        Returns:
            Dict with 'teams' and 'players' lists
        """
        result = {'teams': [], 'players': []}
        
        # Search teams
        team_data = self._make_request(f"searchteams.php?t={query}")
        if team_data and team_data.get('teams'):
            for t in team_data['teams'][:5]:
                result['teams'].append({
                    'name': t.get('strTeam'),
                    'logo': t.get('strBadge'),
                    'league': t.get('strLeague'),
                })
        
        # Search players
        player_data = self._make_request(f"searchplayers.php?p={query}")
        if player_data and player_data.get('player'):
            for p in player_data['player'][:5]:
                result['players'].append({
                    'name': p.get('strPlayer'),
                    'photo': p.get('strThumb'),
                    'team': p.get('strTeam'),
                    'position': p.get('strPosition'),
                })
        
        return result


# Convenience function
def get_sportsdb_api() -> TheSportsDBAPI:
    """Get TheSportsDB API client"""
    return TheSportsDBAPI()


# Test if run directly
if __name__ == "__main__":
    api = TheSportsDBAPI()
    
    print("=== TheSportsDB API Test ===\n")
    
    # Test team info
    print("1. Team Info (Arsenal):")
    team = api.get_team_info("Arsenal")
    if team:
        print(f"   Name: {team['name']}")
        print(f"   Stadium: {team['stadium']} ({team['stadium_capacity']} capacity)")
        print(f"   Logo: {team['logo'][:50]}...")
    
    # Test player info
    print("\n2. Player Info (Salah):")
    player = api.get_player_info("Salah")
    if player:
        print(f"   Name: {player['name']}")
        print(f"   Team: {player['team']}")
        print(f"   Position: {player['position']}")
        print(f"   Photo: {player['photo'][:50]}..." if player['photo'] else "   No photo")
    
    # Test league teams
    print("\n3. EPL Teams with Logos:")
    teams = api.get_league_teams("EPL")
    for t in teams[:5]:
        print(f"   {t['name']}: {t['logo'][:40]}..." if t['logo'] else f"   {t['name']}: No logo")
    
    print("\n✅ TheSportsDB integration ready!")
