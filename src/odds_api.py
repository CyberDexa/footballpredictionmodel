"""
Odds API Integration
Free tier: 500 requests/month
https://the-odds-api.com/
"""

import requests
import os
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class OddsAPI:
    """Client for The Odds API - get live bookmaker odds"""
    
    BASE_URL = "https://api.the-odds-api.com/v4"
    
    # Sport keys for football leagues (with alternate names)
    SPORT_KEYS = {
        # English leagues
        'EPL': 'soccer_epl',
        'CHAMPIONSHIP': 'soccer_efl_champ',
        'EFL_CHAMPIONSHIP': 'soccer_efl_champ',
        'LEAGUE_ONE': 'soccer_england_league1',
        'EFL_LEAGUE1': 'soccer_england_league1',
        'LEAGUE_TWO': 'soccer_england_league2',
        'EFL_LEAGUE2': 'soccer_england_league2',
        'SCOTTISH_PREM': 'soccer_spl',
        'FA_CUP': 'soccer_fa_cup',
        'EFL_CUP': 'soccer_england_efl_cup',
        # Spanish leagues
        'LA_LIGA': 'soccer_spain_la_liga',
        'LA_LIGA_2': 'soccer_spain_segunda_division',
        'LA_LIGA2': 'soccer_spain_segunda_division',
        # Italian leagues
        'SERIE_A': 'soccer_italy_serie_a',
        # German leagues
        'BUNDESLIGA': 'soccer_germany_bundesliga',
        'BUNDESLIGA_2': 'soccer_germany_bundesliga2',
        'BUNDESLIGA2': 'soccer_germany_bundesliga2',
        # French leagues
        'LIGUE_1': 'soccer_france_ligue_one',
        'LIGUE_2': 'soccer_france_ligue_two',
        # Other European
        'EREDIVISIE': 'soccer_netherlands_eredivisie',
        'PRIMEIRA_LIGA': 'soccer_portugal_primeira_liga',
        'PRIMEIRA': 'soccer_portugal_primeira_liga',
        'SUPER_LIG': 'soccer_turkey_super_league',
        'GREEK_SL': 'soccer_greece_super_league',
        'SUPER_LEAGUE_GR': 'soccer_greece_super_league',
        # USA
        'MLS': 'soccer_usa_mls',
        # European competitions
        'CHAMPIONS_LEAGUE': 'soccer_uefa_champs_league',
        'EUROPA_LEAGUE': 'soccer_uefa_europa_league',
    }
    
    # Market types
    MARKETS = {
        'h2h': 'Match Result (1X2)',
        'spreads': 'Asian Handicap',
        'totals': 'Over/Under Goals',
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with API key from env or parameter"""
        self.api_key = api_key or os.getenv('ODDS_API_KEY')
        self.cache = {}
        self.cache_expiry = {}
        self.requests_used = 0
        self.requests_remaining = 500  # Default free tier
    
    def is_configured(self) -> bool:
        """Check if API key is configured"""
        return bool(self.api_key)
    
    def get_sports(self) -> List[Dict]:
        """Get list of available sports"""
        if not self.is_configured():
            return []
        
        try:
            response = requests.get(
                f"{self.BASE_URL}/sports",
                params={'apiKey': self.api_key}
            )
            self._update_usage(response)
            
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            print(f"Error fetching sports: {e}")
            return []
    
    def get_odds(self, league: str, markets: List[str] = ['h2h', 'totals']) -> List[Dict]:
        """
        Get odds for upcoming matches in a league
        
        Args:
            league: League code (EPL, LA_LIGA, etc.)
            markets: List of markets to fetch (h2h, totals, spreads)
        
        Returns:
            List of matches with odds from various bookmakers
        """
        if not self.is_configured():
            return []
        
        sport_key = self.SPORT_KEYS.get(league)
        if not sport_key:
            return []
        
        # Check cache
        cache_key = f"{league}_{','.join(markets)}"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        try:
            response = requests.get(
                f"{self.BASE_URL}/sports/{sport_key}/odds",
                params={
                    'apiKey': self.api_key,
                    'regions': 'uk,eu',  # UK and EU bookmakers
                    'markets': ','.join(markets),
                    'oddsFormat': 'decimal',
                    'dateFormat': 'iso'
                }
            )
            self._update_usage(response)
            
            if response.status_code == 200:
                data = response.json()
                self._cache_response(cache_key, data)
                return data
            elif response.status_code == 401:
                print("Invalid API key")
            elif response.status_code == 429:
                print("Rate limit exceeded")
            return []
        except Exception as e:
            print(f"Error fetching odds: {e}")
            return []
    
    def get_match_odds(self, league: str, home_team: str, away_team: str) -> Optional[Dict]:
        """
        Get odds for a specific match
        
        Args:
            league: League code
            home_team: Home team name
            away_team: Away team name
        
        Returns:
            Dict with odds from various bookmakers, or None if not found
        """
        all_odds = self.get_odds(league)
        
        # Normalize team names for matching
        home_lower = home_team.lower()
        away_lower = away_team.lower()
        
        for match in all_odds:
            match_home = match.get('home_team', '').lower()
            match_away = match.get('away_team', '').lower()
            
            # Fuzzy match (contains check)
            if (home_lower in match_home or match_home in home_lower) and \
               (away_lower in match_away or match_away in away_lower):
                return self._parse_match_odds(match)
        
        return None
    
    def _parse_match_odds(self, match: Dict) -> Dict:
        """Parse match odds into a structured format"""
        result = {
            'home_team': match.get('home_team'),
            'away_team': match.get('away_team'),
            'commence_time': match.get('commence_time'),
            'bookmakers': {},
            'best_odds': {
                'home': {'odds': 0, 'bookmaker': None},
                'draw': {'odds': 0, 'bookmaker': None},
                'away': {'odds': 0, 'bookmaker': None},
                'over_2.5': {'odds': 0, 'bookmaker': None},
                'under_2.5': {'odds': 0, 'bookmaker': None},
            }
        }
        
        for bookmaker in match.get('bookmakers', []):
            bookie_name = bookmaker.get('title')
            result['bookmakers'][bookie_name] = {}
            
            for market in bookmaker.get('markets', []):
                market_key = market.get('key')
                
                if market_key == 'h2h':
                    for outcome in market.get('outcomes', []):
                        name = outcome.get('name')
                        price = outcome.get('price')
                        
                        if name == match.get('home_team'):
                            result['bookmakers'][bookie_name]['home'] = price
                            if price > result['best_odds']['home']['odds']:
                                result['best_odds']['home'] = {'odds': price, 'bookmaker': bookie_name}
                        elif name == match.get('away_team'):
                            result['bookmakers'][bookie_name]['away'] = price
                            if price > result['best_odds']['away']['odds']:
                                result['best_odds']['away'] = {'odds': price, 'bookmaker': bookie_name}
                        elif name == 'Draw':
                            result['bookmakers'][bookie_name]['draw'] = price
                            if price > result['best_odds']['draw']['odds']:
                                result['best_odds']['draw'] = {'odds': price, 'bookmaker': bookie_name}
                
                elif market_key == 'totals':
                    for outcome in market.get('outcomes', []):
                        name = outcome.get('name')
                        point = outcome.get('point', 2.5)
                        price = outcome.get('price')
                        
                        if point == 2.5:
                            if name == 'Over':
                                result['bookmakers'][bookie_name]['over_2.5'] = price
                                if price > result['best_odds']['over_2.5']['odds']:
                                    result['best_odds']['over_2.5'] = {'odds': price, 'bookmaker': bookie_name}
                            elif name == 'Under':
                                result['bookmakers'][bookie_name]['under_2.5'] = price
                                if price > result['best_odds']['under_2.5']['odds']:
                                    result['best_odds']['under_2.5'] = {'odds': price, 'bookmaker': bookie_name}
        
        return result
    
    def calculate_value_bets(self, match_odds: Dict, model_probs: Dict) -> List[Dict]:
        """
        Find value bets by comparing model probabilities with bookmaker odds
        
        Args:
            match_odds: Odds from bookmakers
            model_probs: Model probabilities {home: 0.45, draw: 0.30, away: 0.25, ...}
        
        Returns:
            List of value bets with positive expected value
        """
        value_bets = []
        
        if not match_odds or 'best_odds' not in match_odds:
            return value_bets
        
        market_mapping = {
            'home': 'home_prob',
            'draw': 'draw_prob',
            'away': 'away_prob',
            'over_2.5': 'over_2_5_prob',
            'under_2.5': 'under_2_5_prob',
        }
        
        for market, prob_key in market_mapping.items():
            best = match_odds['best_odds'].get(market, {})
            odds = best.get('odds', 0)
            bookmaker = best.get('bookmaker')
            
            if odds <= 0:
                continue
            
            model_prob = model_probs.get(prob_key, 0)
            if model_prob <= 0:
                continue
            
            # Implied probability from odds
            implied_prob = 1 / odds
            
            # Edge = model probability - implied probability
            edge = model_prob - implied_prob
            
            # Expected value
            ev = (model_prob * (odds - 1)) - (1 - model_prob)
            
            if edge > 0.05:  # Only show if edge > 5%
                value_bets.append({
                    'market': market,
                    'odds': odds,
                    'bookmaker': bookmaker,
                    'model_prob': model_prob * 100,
                    'implied_prob': implied_prob * 100,
                    'edge': edge * 100,
                    'ev': ev * 100,
                    'rating': 'Strong' if edge > 0.15 else 'Good' if edge > 0.10 else 'Moderate'
                })
        
        # Sort by edge
        value_bets.sort(key=lambda x: x['edge'], reverse=True)
        return value_bets
    
    def _update_usage(self, response):
        """Update API usage tracking from response headers"""
        remaining = response.headers.get('x-requests-remaining')
        used = response.headers.get('x-requests-used')
        
        if remaining:
            self.requests_remaining = int(remaining)
        if used:
            self.requests_used = int(used)
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid (5 min cache)"""
        if key not in self.cache:
            return False
        expiry = self.cache_expiry.get(key)
        if not expiry:
            return False
        return datetime.now() < expiry
    
    def _cache_response(self, key: str, data: any):
        """Cache response for 5 minutes"""
        self.cache[key] = data
        self.cache_expiry[key] = datetime.now() + timedelta(minutes=5)
    
    def get_usage_stats(self) -> Dict:
        """Get API usage statistics"""
        return {
            'requests_used': self.requests_used,
            'requests_remaining': self.requests_remaining,
            'limit': 500  # Free tier limit per month
        }
    
    def get_upcoming_fixtures(self, league: str, days_ahead: int = 3) -> List[Dict]:
        """
        Get upcoming fixtures for a league (uses events endpoint - FREE, no quota cost)
        
        Args:
            league: League code (EPL, LA_LIGA, etc.)
            days_ahead: Only return matches within this many days
            
        Returns:
            List of upcoming fixtures with home_team, away_team, commence_time, date
        """
        if not self.is_configured():
            return []
        
        sport_key = self.SPORT_KEYS.get(league)
        if not sport_key:
            return []
        
        # Check cache
        cache_key = f"fixtures_{league}_{days_ahead}"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        try:
            response = requests.get(
                f"{self.BASE_URL}/sports/{sport_key}/events",
                params={
                    'apiKey': self.api_key,
                    'dateFormat': 'iso'
                }
            )
            self._update_usage(response)
            
            if response.status_code == 200:
                data = response.json()
                
                # Filter to matches within our window
                cutoff = datetime.now() + timedelta(days=days_ahead + 1)
                fixtures = []
                
                for match in data:
                    commence_time = match.get('commence_time', '')
                    if commence_time:
                        try:
                            match_dt = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
                            match_local = match_dt.replace(tzinfo=None)
                            
                            if datetime.now() < match_local <= cutoff:
                                fixtures.append({
                                    'id': match.get('id'),
                                    'home_team': match.get('home_team'),
                                    'away_team': match.get('away_team'),
                                    'commence_time': commence_time,
                                    'match_date': match_local.strftime('%Y-%m-%d'),
                                    'league': league
                                })
                        except Exception:
                            pass
                
                self._cache_response(cache_key, fixtures)
                return fixtures
            return []
        except Exception as e:
            print(f"Error fetching fixtures: {e}")
            return []
    
    def get_scores(self, league: str, days_from: int = 3) -> List[Dict]:
        """
        Get recent match scores/results from The Odds API
        
        Args:
            league: League code (EPL, LA_LIGA, etc.)
            days_from: Number of days in the past to fetch (1-3, API limit is 3)
        
        Returns:
            List of completed matches with scores
        """
        if not self.is_configured():
            return []
        
        sport_key = self.SPORT_KEYS.get(league)
        if not sport_key:
            return []
        
        # API only allows 1-3 days
        days_from = min(max(1, days_from), 3)
        
        # Check cache
        cache_key = f"scores_{league}_{days_from}"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        try:
            response = requests.get(
                f"{self.BASE_URL}/sports/{sport_key}/scores",
                params={
                    'apiKey': self.api_key,
                    'daysFrom': days_from,
                    'dateFormat': 'iso'
                }
            )
            self._update_usage(response)
            
            if response.status_code == 200:
                data = response.json()
                # Filter to only completed matches
                completed = [m for m in data if m.get('completed', False)]
                self._cache_response(cache_key, completed)
                return completed
            elif response.status_code == 401:
                print("Invalid API key")
            elif response.status_code == 429:
                print("Rate limit exceeded")
            return []
        except Exception as e:
            print(f"Error fetching scores: {e}")
            return []
    
    def get_recent_results(self, league: str, days: int = 3) -> List[Dict]:
        """
        Get recent match results in a standardized format for verification
        
        Args:
            league: League code
            days: Number of days to look back
        
        Returns:
            List of dicts with: home_team, away_team, home_score, away_score, date
        """
        scores = self.get_scores(league, days)
        results = []
        
        for match in scores:
            if not match.get('completed'):
                continue
            
            scores_data = match.get('scores', [])
            home_score = None
            away_score = None
            
            for score in scores_data:
                if score.get('name') == match.get('home_team'):
                    home_score = int(score.get('score', 0))
                elif score.get('name') == match.get('away_team'):
                    away_score = int(score.get('score', 0))
            
            if home_score is not None and away_score is not None:
                results.append({
                    'home_team': match.get('home_team'),
                    'away_team': match.get('away_team'),
                    'home_score': home_score,
                    'away_score': away_score,
                    'date': match.get('commence_time', '')[:10],  # Extract date part
                    'completed': True
                })
        
        return results


# Singleton instance
_odds_api = None

def get_odds_api(api_key: Optional[str] = None) -> OddsAPI:
    """Get singleton instance of OddsAPI"""
    global _odds_api
    if _odds_api is None:
        _odds_api = OddsAPI(api_key)
    elif api_key and api_key != _odds_api.api_key:
        _odds_api = OddsAPI(api_key)
    return _odds_api
