"""
Authentication Module for Football Match Predictor
Handles user registration, login, sessions, and subscription management
"""

import sqlite3
import hashlib
import secrets
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import re


class AuthManager:
    """Manages user authentication and sessions"""
    
    # Subscription tiers with limits
    SUBSCRIPTION_TIERS = {
        'free': {
            'name': 'Free',
            'price': 0,
            'predictions_per_day': 3,
            'leagues': ['EPL'],  # Only EPL for free tier
            'features': ['basic_predictions'],
            'description': '3 predictions/day, EPL only'
        },
        'basic': {
            'name': 'Basic',
            'price': 9.99,
            'predictions_per_day': 25,
            'leagues': ['EPL', 'LA_LIGA', 'SERIE_A', 'BUNDESLIGA', 'LIGUE_1'],
            'features': ['basic_predictions', 'value_bets'],
            'description': '25 predictions/day, Top 5 leagues'
        },
        'pro': {
            'name': 'Pro',
            'price': 24.99,
            'predictions_per_day': 100,
            'leagues': 'all',  # All leagues
            'features': ['basic_predictions', 'value_bets', 'track_record', 'api_access'],
            'description': '100 predictions/day, All leagues, Track record'
        },
        'unlimited': {
            'name': 'Unlimited',
            'price': 49.99,
            'predictions_per_day': -1,  # Unlimited
            'leagues': 'all',
            'features': ['basic_predictions', 'value_bets', 'track_record', 'api_access', 'priority_support'],
            'description': 'Unlimited predictions, All features'
        }
    }
    
    def __init__(self, db_path: str = "data/users.db"):
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
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                username TEXT,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                subscription_tier TEXT DEFAULT 'free',
                subscription_expires TIMESTAMP,
                stripe_customer_id TEXT,
                stripe_subscription_id TEXT,
                predictions_today INTEGER DEFAULT 0,
                last_prediction_date DATE,
                is_active BOOLEAN DEFAULT TRUE,
                email_verified BOOLEAN DEFAULT FALSE,
                verification_token TEXT
            )
        ''')
        
        # Sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_token TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Usage log for tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS usage_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                action TEXT NOT NULL,
                details TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Password reset tokens
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS password_resets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                reset_token TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                used BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _hash_password(self, password: str, salt: str = None) -> Tuple[str, str]:
        """Hash a password with salt"""
        if salt is None:
            salt = secrets.token_hex(32)
        
        # Use PBKDF2 with SHA256
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # iterations
        ).hex()
        
        return password_hash, salt
    
    def _validate_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    def _validate_password(self, password: str) -> Tuple[bool, str]:
        """Validate password strength"""
        if len(password) < 8:
            return False, "Password must be at least 8 characters"
        if not re.search(r'[A-Z]', password):
            return False, "Password must contain at least one uppercase letter"
        if not re.search(r'[a-z]', password):
            return False, "Password must contain at least one lowercase letter"
        if not re.search(r'\d', password):
            return False, "Password must contain at least one number"
        return True, "Password is valid"
    
    def register(self, email: str, password: str, username: str = None) -> Tuple[bool, str, Optional[int]]:
        """
        Register a new user
        
        Returns:
            Tuple of (success, message, user_id)
        """
        # Validate email
        if not self._validate_email(email):
            return False, "Invalid email format", None
        
        # Validate password
        is_valid, msg = self._validate_password(password)
        if not is_valid:
            return False, msg, None
        
        # Generate username from email if not provided
        if not username:
            username = email.split('@')[0]
        
        # Hash password
        password_hash, salt = self._hash_password(password)
        
        # Generate verification token
        verification_token = secrets.token_urlsafe(32)
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO users (email, username, password_hash, salt, verification_token)
                VALUES (?, ?, ?, ?, ?)
            ''', (email.lower(), username, password_hash, salt, verification_token))
            
            user_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return True, "Registration successful! Please check your email to verify your account.", user_id
            
        except sqlite3.IntegrityError:
            conn.close()
            return False, "Email already registered", None
    
    def login(self, email: str, password: str) -> Tuple[bool, str, Optional[str]]:
        """
        Authenticate a user and create a session
        
        Returns:
            Tuple of (success, message, session_token)
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, password_hash, salt, is_active
            FROM users
            WHERE email = ?
        ''', (email.lower(),))
        
        user = cursor.fetchone()
        
        if not user:
            conn.close()
            return False, "Invalid email or password", None
        
        if not user['is_active']:
            conn.close()
            return False, "Account is deactivated", None
        
        # Verify password
        password_hash, _ = self._hash_password(password, user['salt'])
        
        if password_hash != user['password_hash']:
            conn.close()
            return False, "Invalid email or password", None
        
        # Create session
        session_token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(days=7)
        
        cursor.execute('''
            INSERT INTO sessions (user_id, session_token, expires_at)
            VALUES (?, ?, ?)
        ''', (user['id'], session_token, expires_at))
        
        # Log the login
        cursor.execute('''
            INSERT INTO usage_log (user_id, action, details)
            VALUES (?, 'login', ?)
        ''', (user['id'], datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
        return True, "Login successful", session_token
    
    def validate_session(self, session_token: str) -> Optional[Dict]:
        """
        Validate a session token and return user info
        
        Returns:
            User info dict or None if invalid
        """
        if not session_token:
            return None
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT u.id, u.email, u.subscription_tier, u.subscription_expires,
                   u.predictions_today, u.last_prediction_date, u.email_verified,
                   s.expires_at
            FROM sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.session_token = ?
            AND s.expires_at > datetime('now')
            AND u.is_active = TRUE
        ''', (session_token,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            # Check if subscription is still valid
            tier = result['subscription_tier']
            expires = result['subscription_expires']
            
            if tier != 'free' and expires:
                if datetime.fromisoformat(expires) < datetime.now():
                    # Subscription expired, downgrade to free
                    self._downgrade_to_free(result['id'])
                    tier = 'free'
            
            return {
                'id': result['id'],
                'email': result['email'],
                'subscription_tier': tier,
                'subscription_expires': expires,
                'predictions_today': result['predictions_today'],
                'last_prediction_date': result['last_prediction_date'],
                'email_verified': result['email_verified'],
                'tier_info': self.SUBSCRIPTION_TIERS.get(tier, self.SUBSCRIPTION_TIERS['free'])
            }
        
        return None
    
    def get_username(self, user_id: int) -> str:
        """Get username by user ID"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT username, email FROM users WHERE id = ?
        ''', (user_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            if row['username']:
                return row['username']
            elif row['email']:
                return row['email'].split('@')[0]
        
        return f"User #{user_id}"
    
    def get_usernames_batch(self, user_ids: List[int]) -> Dict[int, str]:
        """Get usernames for multiple user IDs"""
        if not user_ids:
            return {}
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        placeholders = ','.join('?' * len(user_ids))
        cursor.execute(f'''
            SELECT id, username, email FROM users WHERE id IN ({placeholders})
        ''', user_ids)
        
        rows = cursor.fetchall()
        conn.close()
        
        result = {}
        for row in rows:
            if row['username']:
                result[row['id']] = row['username']
            elif row['email']:
                result[row['id']] = row['email'].split('@')[0]
            else:
                result[row['id']] = f"User #{row['id']}"
        
        return result
    
    def logout(self, session_token: str) -> bool:
        """Invalidate a session"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM sessions
            WHERE session_token = ?
        ''', (session_token,))
        
        conn.commit()
        conn.close()
        return True
    
    def _downgrade_to_free(self, user_id: int):
        """Downgrade user to free tier"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users
            SET subscription_tier = 'free',
                subscription_expires = NULL
            WHERE id = ?
        ''', (user_id,))
        
        conn.commit()
        conn.close()
    
    def can_make_prediction(self, user_id: int) -> Tuple[bool, str, int]:
        """
        Check if user can make a prediction based on their tier limits
        ALL USERS NOW HAVE UNLIMITED ACCESS
        
        Returns:
            Tuple of (can_predict, message, remaining_predictions)
        """
        # All users get unlimited access
        return True, "Unlimited predictions", -1
        
        if remaining <= 0:
            return False, f"Daily limit reached ({limit} predictions). Upgrade for more!", 0
        
        return True, f"{remaining} predictions remaining today", remaining
    
    def record_prediction(self, user_id: int) -> bool:
        """Record that a user made a prediction"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        today = datetime.now().date().isoformat()
        
        cursor.execute('''
            UPDATE users
            SET predictions_today = predictions_today + 1,
                last_prediction_date = ?
            WHERE id = ?
        ''', (today, user_id))
        
        cursor.execute('''
            INSERT INTO usage_log (user_id, action, details)
            VALUES (?, 'prediction', ?)
        ''', (user_id, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        return True
    
    def can_access_league(self, user_id: int, league: str) -> bool:
        """Check if user can access a specific league"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT subscription_tier
            FROM users
            WHERE id = ?
        ''', (user_id,))
        
        user = cursor.fetchone()
        conn.close()
        
        if not user:
            return False
        
        tier_info = self.SUBSCRIPTION_TIERS.get(
            user['subscription_tier'], 
            self.SUBSCRIPTION_TIERS['free']
        )
        
        allowed_leagues = tier_info['leagues']
        
        if allowed_leagues == 'all':
            return True
        
        return league in allowed_leagues
    
    def get_user_stats(self, user_id: int) -> Dict:
        """Get user statistics"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get total predictions
        cursor.execute('''
            SELECT COUNT(*) as total
            FROM usage_log
            WHERE user_id = ?
            AND action = 'prediction'
        ''', (user_id,))
        
        total = cursor.fetchone()['total']
        
        # Get predictions this month
        cursor.execute('''
            SELECT COUNT(*) as monthly
            FROM usage_log
            WHERE user_id = ?
            AND action = 'prediction'
            AND created_at >= date('now', 'start of month')
        ''', (user_id,))
        
        monthly = cursor.fetchone()['monthly']
        
        conn.close()
        
        return {
            'total_predictions': total,
            'monthly_predictions': monthly
        }
    
    def update_subscription(self, user_id: int, tier: str, 
                          stripe_customer_id: str = None,
                          stripe_subscription_id: str = None,
                          months: int = 1) -> bool:
        """Update user subscription"""
        if tier not in self.SUBSCRIPTION_TIERS:
            return False
        
        expires = datetime.now() + timedelta(days=30 * months)
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users
            SET subscription_tier = ?,
                subscription_expires = ?,
                stripe_customer_id = COALESCE(?, stripe_customer_id),
                stripe_subscription_id = COALESCE(?, stripe_subscription_id)
            WHERE id = ?
        ''', (tier, expires, stripe_customer_id, stripe_subscription_id, user_id))
        
        cursor.execute('''
            INSERT INTO usage_log (user_id, action, details)
            VALUES (?, 'subscription_update', ?)
        ''', (user_id, f"Upgraded to {tier}"))
        
        conn.commit()
        conn.close()
        return True
    
    def request_password_reset(self, email: str) -> Tuple[bool, str]:
        """Generate a password reset token"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT id FROM users WHERE email = ?', (email.lower(),))
        user = cursor.fetchone()
        
        if not user:
            conn.close()
            # Don't reveal if email exists
            return True, "If an account exists with this email, you will receive a reset link."
        
        reset_token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(hours=1)
        
        cursor.execute('''
            INSERT INTO password_resets (user_id, reset_token, expires_at)
            VALUES (?, ?, ?)
        ''', (user['id'], reset_token, expires_at))
        
        conn.commit()
        conn.close()
        
        # In production, send email here
        return True, f"Reset token: {reset_token}"  # For testing
    
    def reset_password(self, reset_token: str, new_password: str) -> Tuple[bool, str]:
        """Reset password using a reset token"""
        is_valid, msg = self._validate_password(new_password)
        if not is_valid:
            return False, msg
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_id
            FROM password_resets
            WHERE reset_token = ?
            AND expires_at > datetime('now')
            AND used = FALSE
        ''', (reset_token,))
        
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            return False, "Invalid or expired reset token"
        
        user_id = result['user_id']
        password_hash, salt = self._hash_password(new_password)
        
        cursor.execute('''
            UPDATE users
            SET password_hash = ?, salt = ?
            WHERE id = ?
        ''', (password_hash, salt, user_id))
        
        cursor.execute('''
            UPDATE password_resets
            SET used = TRUE
            WHERE reset_token = ?
        ''', (reset_token,))
        
        # Invalidate all sessions for security
        cursor.execute('DELETE FROM sessions WHERE user_id = ?', (user_id,))
        
        conn.commit()
        conn.close()
        
        return True, "Password reset successful. Please log in with your new password."


# Singleton instance
_auth_instance = None

def get_auth_manager() -> AuthManager:
    """Get the singleton auth manager"""
    global _auth_instance
    if _auth_instance is None:
        _auth_instance = AuthManager()
    return _auth_instance
