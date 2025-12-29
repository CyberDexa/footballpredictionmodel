"""
Value Bet Calculator Module
Compares model predictions to implied bookmaker odds to find value bets
"""

from typing import Dict, Optional, Tuple
import math


class ValueBetCalculator:
    """Calculate value bets by comparing predictions to bookmaker odds"""
    
    # Average bookmaker margins and implied probabilities
    # Based on typical market odds for major leagues
    MARKET_BASELINES = {
        'home_win': 0.45,      # Typical home win market
        'draw': 0.27,          # Typical draw probability
        'away_win': 0.28,      # Typical away win
        'over_2_5': 0.52,      # Over 2.5 goals
        'over_1_5': 0.75,      # Over 1.5 goals
        'over_3_5': 0.30,      # Over 3.5 goals
        'btts': 0.50,          # Both teams to score
    }
    
    # Minimum edge required to flag as value bet (in percentage points)
    MIN_EDGE = 5.0  # 5% edge minimum
    
    def __init__(self, min_edge: float = 5.0):
        """
        Initialize the value bet calculator
        
        Args:
            min_edge: Minimum edge percentage to flag as value bet
        """
        self.min_edge = min_edge
    
    def calculate_edge(self, model_prob: float, market_prob: float) -> float:
        """
        Calculate the edge over the market
        
        Args:
            model_prob: Our model's probability (0-1)
            market_prob: Market implied probability (0-1)
            
        Returns:
            Edge as a percentage (can be negative)
        """
        if market_prob <= 0:
            return 0.0
        
        # Edge = (Our Probability - Market Probability) / Market Probability * 100
        # Positive = we think it's more likely than the market
        edge = ((model_prob - market_prob) / market_prob) * 100
        return edge
    
    def prob_to_odds(self, probability: float) -> Tuple[float, str]:
        """
        Convert probability to decimal and fractional odds
        
        Args:
            probability: Probability between 0 and 1
            
        Returns:
            Tuple of (decimal_odds, fractional_odds_string)
        """
        if probability <= 0 or probability >= 1:
            return (0.0, "N/A")
        
        decimal_odds = 1 / probability
        
        # Convert to fractional
        if decimal_odds >= 2:
            # Format as "X/1" or similar
            numerator = decimal_odds - 1
            if abs(numerator - round(numerator)) < 0.1:
                fractional = f"{int(round(numerator))}/1"
            else:
                fractional = f"{numerator:.1f}/1"
        else:
            # Format as "1/X"
            denominator = 1 / (decimal_odds - 1)
            if abs(denominator - round(denominator)) < 0.1:
                fractional = f"1/{int(round(denominator))}"
            else:
                fractional = f"1/{denominator:.1f}"
        
        return (decimal_odds, fractional)
    
    def analyze_value(self, predictions: Dict) -> Dict[str, Dict]:
        """
        Analyze all predictions for value bets
        
        Args:
            predictions: Dictionary of model predictions
            
        Returns:
            Dictionary with value analysis for each market
        """
        value_analysis = {}
        
        # Match result analysis
        match_pred = predictions.get('match_result', {})
        if match_pred and 'probabilities' in match_pred:
            probs = match_pred['probabilities'][0]
            home_prob, draw_prob, away_prob = probs[0], probs[1], probs[2]
            
            # Home win value
            home_edge = self.calculate_edge(home_prob, self.MARKET_BASELINES['home_win'])
            home_decimal, home_frac = self.prob_to_odds(home_prob)
            value_analysis['home_win'] = {
                'model_prob': home_prob * 100,
                'market_prob': self.MARKET_BASELINES['home_win'] * 100,
                'edge': home_edge,
                'is_value': home_edge >= self.min_edge,
                'decimal_odds': home_decimal,
                'fractional_odds': home_frac
            }
            
            # Draw value
            draw_edge = self.calculate_edge(draw_prob, self.MARKET_BASELINES['draw'])
            draw_decimal, draw_frac = self.prob_to_odds(draw_prob)
            value_analysis['draw'] = {
                'model_prob': draw_prob * 100,
                'market_prob': self.MARKET_BASELINES['draw'] * 100,
                'edge': draw_edge,
                'is_value': draw_edge >= self.min_edge,
                'decimal_odds': draw_decimal,
                'fractional_odds': draw_frac
            }
            
            # Away win value
            away_edge = self.calculate_edge(away_prob, self.MARKET_BASELINES['away_win'])
            away_decimal, away_frac = self.prob_to_odds(away_prob)
            value_analysis['away_win'] = {
                'model_prob': away_prob * 100,
                'market_prob': self.MARKET_BASELINES['away_win'] * 100,
                'edge': away_edge,
                'is_value': away_edge >= self.min_edge,
                'decimal_odds': away_decimal,
                'fractional_odds': away_frac
            }
        
        # Over 2.5 goals analysis
        over25 = predictions.get('over_2.5', {})
        if over25 and 'probabilities' in over25:
            over_prob = over25['probabilities'][0][1]
            over_edge = self.calculate_edge(over_prob, self.MARKET_BASELINES['over_2_5'])
            over_decimal, over_frac = self.prob_to_odds(over_prob)
            
            value_analysis['over_2_5'] = {
                'model_prob': over_prob * 100,
                'market_prob': self.MARKET_BASELINES['over_2_5'] * 100,
                'edge': over_edge,
                'is_value': over_edge >= self.min_edge,
                'decimal_odds': over_decimal,
                'fractional_odds': over_frac
            }
            
            # Under 2.5 (inverse)
            under_prob = 1 - over_prob
            under_edge = self.calculate_edge(under_prob, 1 - self.MARKET_BASELINES['over_2_5'])
            under_decimal, under_frac = self.prob_to_odds(under_prob)
            
            value_analysis['under_2_5'] = {
                'model_prob': under_prob * 100,
                'market_prob': (1 - self.MARKET_BASELINES['over_2_5']) * 100,
                'edge': under_edge,
                'is_value': under_edge >= self.min_edge,
                'decimal_odds': under_decimal,
                'fractional_odds': under_frac
            }
        
        # BTTS analysis
        btts = predictions.get('btts', {})
        if btts and 'probabilities' in btts:
            btts_prob = btts['probabilities'][0][1]
            btts_edge = self.calculate_edge(btts_prob, self.MARKET_BASELINES['btts'])
            btts_decimal, btts_frac = self.prob_to_odds(btts_prob)
            
            value_analysis['btts_yes'] = {
                'model_prob': btts_prob * 100,
                'market_prob': self.MARKET_BASELINES['btts'] * 100,
                'edge': btts_edge,
                'is_value': btts_edge >= self.min_edge,
                'decimal_odds': btts_decimal,
                'fractional_odds': btts_frac
            }
            
            # BTTS No
            btts_no_prob = 1 - btts_prob
            btts_no_edge = self.calculate_edge(btts_no_prob, 1 - self.MARKET_BASELINES['btts'])
            btts_no_decimal, btts_no_frac = self.prob_to_odds(btts_no_prob)
            
            value_analysis['btts_no'] = {
                'model_prob': btts_no_prob * 100,
                'market_prob': (1 - self.MARKET_BASELINES['btts']) * 100,
                'edge': btts_no_edge,
                'is_value': btts_no_edge >= self.min_edge,
                'decimal_odds': btts_no_decimal,
                'fractional_odds': btts_no_frac
            }
        
        return value_analysis
    
    def get_best_value_bets(self, value_analysis: Dict, top_n: int = 3) -> list:
        """
        Get the best value bets from the analysis
        
        Args:
            value_analysis: Output from analyze_value()
            top_n: Number of top bets to return
            
        Returns:
            List of (market, details) tuples sorted by edge
        """
        value_bets = [
            (market, details) 
            for market, details in value_analysis.items() 
            if details.get('is_value', False)
        ]
        
        # Sort by edge descending
        value_bets.sort(key=lambda x: x[1]['edge'], reverse=True)
        
        return value_bets[:top_n]
    
    def format_market_name(self, market_key: str) -> str:
        """Format market key to human readable name"""
        names = {
            'home_win': '🏠 Home Win',
            'draw': '🤝 Draw',
            'away_win': '✈️ Away Win',
            'over_2_5': '⚽ Over 2.5 Goals',
            'under_2_5': '⚽ Under 2.5 Goals',
            'over_1_5': '⚽ Over 1.5 Goals',
            'over_3_5': '⚽ Over 3.5 Goals',
            'btts_yes': '🔄 Both Teams to Score',
            'btts_no': '🔄 BTTS No',
        }
        return names.get(market_key, market_key)


# Singleton instance
_calculator = None

def get_value_calculator() -> ValueBetCalculator:
    """Get the singleton value bet calculator"""
    global _calculator
    if _calculator is None:
        _calculator = ValueBetCalculator()
    return _calculator
