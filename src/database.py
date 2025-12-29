"""
Database Module for Football Match Predictor
SQLite-based persistence for predictions, tracking, and accuracy metrics
"""

import sqlite3
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
import pandas as pd


class Database:
    """SQLite database manager for the prediction app"""
    
    def __init__(self, db_path: str = "data/predictions.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_db(self):
        """Initialize database tables"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Predictions table - stores all predictions made
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                league TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                match_date DATE,
                
                -- Match Result Predictions
                home_win_prob REAL,
                draw_prob REAL,
                away_win_prob REAL,
                predicted_result TEXT,
                
                -- Goals Predictions
                over_1_5_prob REAL,
                over_2_5_prob REAL,
                over_3_5_prob REAL,
                
                -- BTTS
                btts_prob REAL,
                
                -- Team Goals
                home_over_0_5_prob REAL,
                home_over_1_5_prob REAL,
                away_over_0_5_prob REAL,
                away_over_1_5_prob REAL,
                
                -- Half Time
                ht_over_0_5_prob REAL,
                ht_over_1_5_prob REAL,
                
                -- Goal Ranges
                goals_0_1_prob REAL,
                goals_2_3_prob REAL,
                goals_4_plus_prob REAL,
                
                -- Confidence scores
                result_confidence REAL,
                overall_confidence REAL,
                
                -- Actual results (filled in later)
                actual_home_goals INTEGER,
                actual_away_goals INTEGER,
                actual_result TEXT,
                match_played BOOLEAN DEFAULT FALSE,
                
                -- Accuracy tracking
                result_correct BOOLEAN,
                over_1_5_correct BOOLEAN,
                over_2_5_correct BOOLEAN,
                over_3_5_correct BOOLEAN,
                btts_correct BOOLEAN
            )
        ''')
        
        # Users table (for future auth)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                subscription_tier TEXT DEFAULT 'free',
                predictions_today INTEGER DEFAULT 0,
                last_prediction_date DATE
            )
        ''')
        
        # Accuracy stats table - pre-aggregated for performance
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS accuracy_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                league TEXT,
                market TEXT,
                period TEXT,  -- 'all', 'last_30', 'last_7'
                total_predictions INTEGER,
                correct_predictions INTEGER,
                accuracy REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Model performance log
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE,
                league TEXT,
                market TEXT,
                predictions_made INTEGER,
                correct INTEGER,
                roi REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_prediction(self, prediction_data: Dict) -> str:
        """Save a new prediction to the database"""
        import uuid
        prediction_id = str(uuid.uuid4())[:8]
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Calculate confidence
        result_probs = [
            prediction_data.get('home_win_prob', 0.33),
            prediction_data.get('draw_prob', 0.33),
            prediction_data.get('away_win_prob', 0.33)
        ]
        result_confidence = max(result_probs)
        
        # Determine predicted result
        if result_probs[0] > result_probs[1] and result_probs[0] > result_probs[2]:
            predicted_result = 'Home Win'
        elif result_probs[2] > result_probs[0] and result_probs[2] > result_probs[1]:
            predicted_result = 'Away Win'
        else:
            predicted_result = 'Draw'
        
        cursor.execute('''
            INSERT INTO predictions (
                prediction_id, league, home_team, away_team, match_date,
                home_win_prob, draw_prob, away_win_prob, predicted_result,
                over_1_5_prob, over_2_5_prob, over_3_5_prob,
                btts_prob,
                home_over_0_5_prob, home_over_1_5_prob,
                away_over_0_5_prob, away_over_1_5_prob,
                ht_over_0_5_prob, ht_over_1_5_prob,
                goals_0_1_prob, goals_2_3_prob, goals_4_plus_prob,
                result_confidence, overall_confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            prediction_id,
            prediction_data.get('league', 'Unknown'),
            prediction_data.get('home_team', ''),
            prediction_data.get('away_team', ''),
            prediction_data.get('match_date'),
            prediction_data.get('home_win_prob'),
            prediction_data.get('draw_prob'),
            prediction_data.get('away_win_prob'),
            predicted_result,
            prediction_data.get('over_1_5_prob'),
            prediction_data.get('over_2_5_prob'),
            prediction_data.get('over_3_5_prob'),
            prediction_data.get('btts_prob'),
            prediction_data.get('home_over_0_5_prob'),
            prediction_data.get('home_over_1_5_prob'),
            prediction_data.get('away_over_0_5_prob'),
            prediction_data.get('away_over_1_5_prob'),
            prediction_data.get('ht_over_0_5_prob'),
            prediction_data.get('ht_over_1_5_prob'),
            prediction_data.get('goals_0_1_prob'),
            prediction_data.get('goals_2_3_prob'),
            prediction_data.get('goals_4_plus_prob'),
            result_confidence,
            prediction_data.get('overall_confidence', result_confidence)
        ))
        
        conn.commit()
        conn.close()
        
        return prediction_id
    
    def update_actual_result(self, prediction_id: str, home_goals: int, away_goals: int):
        """Update prediction with actual match result"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Determine actual result
        if home_goals > away_goals:
            actual_result = 'Home Win'
        elif away_goals > home_goals:
            actual_result = 'Away Win'
        else:
            actual_result = 'Draw'
        
        total_goals = home_goals + away_goals
        
        # Get the prediction to compare
        cursor.execute('SELECT * FROM predictions WHERE prediction_id = ?', (prediction_id,))
        pred = cursor.fetchone()
        
        if pred:
            # Calculate correctness
            result_correct = pred['predicted_result'] == actual_result
            over_1_5_correct = (pred['over_1_5_prob'] > 0.5) == (total_goals > 1.5)
            over_2_5_correct = (pred['over_2_5_prob'] > 0.5) == (total_goals > 2.5)
            over_3_5_correct = (pred['over_3_5_prob'] > 0.5) == (total_goals > 3.5)
            btts_correct = (pred['btts_prob'] > 0.5) == (home_goals > 0 and away_goals > 0)
            
            cursor.execute('''
                UPDATE predictions SET
                    actual_home_goals = ?,
                    actual_away_goals = ?,
                    actual_result = ?,
                    match_played = TRUE,
                    result_correct = ?,
                    over_1_5_correct = ?,
                    over_2_5_correct = ?,
                    over_3_5_correct = ?,
                    btts_correct = ?
                WHERE prediction_id = ?
            ''', (
                home_goals, away_goals, actual_result,
                result_correct, over_1_5_correct, over_2_5_correct,
                over_3_5_correct, btts_correct,
                prediction_id
            ))
        
        conn.commit()
        conn.close()
    
    def get_accuracy_stats(self, league: str = None, days: int = None) -> Dict:
        """Get accuracy statistics for predictions"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        where_clauses = ["match_played = TRUE"]
        params = []
        
        if league:
            where_clauses.append("league = ?")
            params.append(league)
        
        if days:
            date_threshold = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            where_clauses.append("created_at >= ?")
            params.append(date_threshold)
        
        where_sql = " AND ".join(where_clauses)
        
        cursor.execute(f'''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN result_correct THEN 1 ELSE 0 END) as result_correct,
                SUM(CASE WHEN over_1_5_correct THEN 1 ELSE 0 END) as over_1_5_correct,
                SUM(CASE WHEN over_2_5_correct THEN 1 ELSE 0 END) as over_2_5_correct,
                SUM(CASE WHEN over_3_5_correct THEN 1 ELSE 0 END) as over_3_5_correct,
                SUM(CASE WHEN btts_correct THEN 1 ELSE 0 END) as btts_correct
            FROM predictions
            WHERE {where_sql}
        ''', params)
        
        row = cursor.fetchone()
        conn.close()
        
        if row and row['total'] > 0:
            total = row['total']
            return {
                'total_predictions': total,
                'result_accuracy': round(row['result_correct'] / total * 100, 1) if row['result_correct'] else 0,
                'over_1_5_accuracy': round(row['over_1_5_correct'] / total * 100, 1) if row['over_1_5_correct'] else 0,
                'over_2_5_accuracy': round(row['over_2_5_correct'] / total * 100, 1) if row['over_2_5_correct'] else 0,
                'over_3_5_accuracy': round(row['over_3_5_correct'] / total * 100, 1) if row['over_3_5_correct'] else 0,
                'btts_accuracy': round(row['btts_correct'] / total * 100, 1) if row['btts_correct'] else 0
            }
        
        return {
            'total_predictions': 0,
            'result_accuracy': 0,
            'over_1_5_accuracy': 0,
            'over_2_5_accuracy': 0,
            'over_3_5_accuracy': 0,
            'btts_accuracy': 0
        }
    
    def get_recent_predictions(self, limit: int = 50, league: str = None) -> List[Dict]:
        """Get recent predictions"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if league:
            cursor.execute('''
                SELECT * FROM predictions
                WHERE league = ?
                ORDER BY created_at DESC
                LIMIT ?
            ''', (league, limit))
        else:
            cursor.execute('''
                SELECT * FROM predictions
                ORDER BY created_at DESC
                LIMIT ?
            ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_pending_results(self) -> List[Dict]:
        """Get predictions that haven't been verified yet"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM predictions
            WHERE match_played = FALSE
            AND match_date <= date('now')
            ORDER BY match_date ASC
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_high_confidence_accuracy(self, min_confidence: float = 0.6) -> Dict:
        """Get accuracy for high confidence predictions only"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN result_correct THEN 1 ELSE 0 END) as correct,
                AVG(result_confidence) as avg_confidence
            FROM predictions
            WHERE match_played = TRUE
            AND result_confidence >= ?
        ''', (min_confidence,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row and row['total'] > 0:
            return {
                'total': row['total'],
                'correct': row['correct'] or 0,
                'accuracy': round((row['correct'] or 0) / row['total'] * 100, 1),
                'avg_confidence': round(row['avg_confidence'] * 100, 1) if row['avg_confidence'] else 0
            }
        
        return {'total': 0, 'correct': 0, 'accuracy': 0, 'avg_confidence': 0}
    
    def get_predictions_by_confidence_tier(self) -> List[Dict]:
        """Get accuracy broken down by confidence tier"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        tiers = [
            ('Very High (>70%)', 0.70, 1.0),
            ('High (60-70%)', 0.60, 0.70),
            ('Medium (50-60%)', 0.50, 0.60),
            ('Low (<50%)', 0.0, 0.50)
        ]
        
        results = []
        for tier_name, min_conf, max_conf in tiers:
            cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN result_correct THEN 1 ELSE 0 END) as correct
                FROM predictions
                WHERE match_played = TRUE
                AND result_confidence >= ?
                AND result_confidence < ?
            ''', (min_conf, max_conf))
            
            row = cursor.fetchone()
            if row and row['total'] > 0:
                results.append({
                    'tier': tier_name,
                    'total': row['total'],
                    'correct': row['correct'] or 0,
                    'accuracy': round((row['correct'] or 0) / row['total'] * 100, 1)
                })
            else:
                results.append({
                    'tier': tier_name,
                    'total': 0,
                    'correct': 0,
                    'accuracy': 0
                })
        
        conn.close()
        return results
    
    def get_league_performance(self) -> List[Dict]:
        """Get accuracy by league"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                league,
                COUNT(*) as total,
                SUM(CASE WHEN result_correct THEN 1 ELSE 0 END) as correct,
                AVG(result_confidence) as avg_confidence
            FROM predictions
            WHERE match_played = TRUE
            GROUP BY league
            ORDER BY total DESC
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            if row['total'] > 0:
                results.append({
                    'league': row['league'],
                    'total': row['total'],
                    'correct': row['correct'] or 0,
                    'accuracy': round((row['correct'] or 0) / row['total'] * 100, 1),
                    'avg_confidence': round(row['avg_confidence'] * 100, 1) if row['avg_confidence'] else 0
                })
        
        return results
    
    def get_daily_stats(self, days: int = 30) -> pd.DataFrame:
        """Get daily prediction stats for charting"""
        conn = self._get_connection()
        
        query = '''
            SELECT 
                date(created_at) as date,
                COUNT(*) as predictions,
                SUM(CASE WHEN result_correct THEN 1 ELSE 0 END) as correct,
                SUM(CASE WHEN match_played THEN 1 ELSE 0 END) as verified
            FROM predictions
            WHERE created_at >= date('now', ?)
            GROUP BY date(created_at)
            ORDER BY date
        '''
        
        df = pd.read_sql_query(query, conn, params=(f'-{days} days',))
        conn.close()
        
        return df
    
    def get_total_stats(self) -> Dict:
        """Get overall statistics"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_predictions,
                SUM(CASE WHEN match_played THEN 1 ELSE 0 END) as verified,
                SUM(CASE WHEN result_correct THEN 1 ELSE 0 END) as correct
            FROM predictions
        ''')
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'total_predictions': row['total_predictions'] or 0,
                'verified': row['verified'] or 0,
                'correct': row['correct'] or 0,
                'accuracy': round((row['correct'] or 0) / (row['verified'] or 1) * 100, 1)
            }
        
        return {'total_predictions': 0, 'verified': 0, 'correct': 0, 'accuracy': 0}


# Singleton instance
_db_instance = None

def get_database() -> Database:
    """Get the singleton database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance
