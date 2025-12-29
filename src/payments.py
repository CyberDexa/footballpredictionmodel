"""
Stripe Payment Integration for Football Match Predictor
Handles subscription payments, webhooks, and billing management
"""

import os
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

# Stripe will be imported conditionally
STRIPE_AVAILABLE = False
stripe = None

try:
    import stripe as stripe_module
    stripe = stripe_module
    STRIPE_AVAILABLE = True
except ImportError:
    pass


class PaymentManager:
    """Manages Stripe payments and subscriptions"""
    
    # Price IDs for each tier (these would be created in Stripe dashboard)
    PRICE_IDS = {
        'basic_monthly': os.getenv('STRIPE_PRICE_BASIC_MONTHLY', 'price_basic_monthly'),
        'basic_yearly': os.getenv('STRIPE_PRICE_BASIC_YEARLY', 'price_basic_yearly'),
        'pro_monthly': os.getenv('STRIPE_PRICE_PRO_MONTHLY', 'price_pro_monthly'),
        'pro_yearly': os.getenv('STRIPE_PRICE_PRO_YEARLY', 'price_pro_yearly'),
        'unlimited_monthly': os.getenv('STRIPE_PRICE_UNLIMITED_MONTHLY', 'price_unlimited_monthly'),
        'unlimited_yearly': os.getenv('STRIPE_PRICE_UNLIMITED_YEARLY', 'price_unlimited_yearly'),
    }
    
    # Tier pricing (for display)
    PRICING = {
        'basic': {'monthly': 9.99, 'yearly': 99.99},  # ~17% discount
        'pro': {'monthly': 24.99, 'yearly': 249.99},  # ~17% discount
        'unlimited': {'monthly': 49.99, 'yearly': 499.99},  # ~17% discount
    }
    
    def __init__(self):
        self.stripe_available = STRIPE_AVAILABLE
        
        if STRIPE_AVAILABLE:
            stripe.api_key = os.getenv('STRIPE_SECRET_KEY', '')
            self.webhook_secret = os.getenv('STRIPE_WEBHOOK_SECRET', '')
    
    def is_configured(self) -> bool:
        """Check if Stripe is properly configured"""
        if not STRIPE_AVAILABLE:
            return False
        return bool(os.getenv('STRIPE_SECRET_KEY'))
    
    def create_customer(self, email: str, user_id: int) -> Optional[str]:
        """
        Create a Stripe customer for a user
        
        Returns:
            Stripe customer ID or None
        """
        if not self.is_configured():
            return None
        
        try:
            customer = stripe.Customer.create(
                email=email,
                metadata={'user_id': str(user_id)}
            )
            return customer.id
        except Exception as e:
            print(f"Error creating Stripe customer: {e}")
            return None
    
    def create_checkout_session(
        self, 
        customer_id: str,
        tier: str,
        billing_period: str = 'monthly',
        success_url: str = None,
        cancel_url: str = None
    ) -> Optional[Dict]:
        """
        Create a Stripe Checkout session for subscription
        
        Returns:
            Dict with session_id and url, or None
        """
        if not self.is_configured():
            return None
        
        price_key = f"{tier}_{billing_period}"
        price_id = self.PRICE_IDS.get(price_key)
        
        if not price_id:
            return None
        
        try:
            session = stripe.checkout.Session.create(
                customer=customer_id,
                payment_method_types=['card'],
                line_items=[{
                    'price': price_id,
                    'quantity': 1,
                }],
                mode='subscription',
                success_url=success_url or 'http://localhost:8501/?success=true',
                cancel_url=cancel_url or 'http://localhost:8501/?canceled=true',
                metadata={
                    'tier': tier,
                    'billing_period': billing_period
                }
            )
            
            return {
                'session_id': session.id,
                'url': session.url
            }
        except Exception as e:
            print(f"Error creating checkout session: {e}")
            return None
    
    def create_portal_session(self, customer_id: str, return_url: str = None) -> Optional[str]:
        """
        Create a Stripe Customer Portal session for managing subscription
        
        Returns:
            Portal URL or None
        """
        if not self.is_configured():
            return None
        
        try:
            session = stripe.billing_portal.Session.create(
                customer=customer_id,
                return_url=return_url or 'http://localhost:8501/'
            )
            return session.url
        except Exception as e:
            print(f"Error creating portal session: {e}")
            return None
    
    def get_subscription(self, subscription_id: str) -> Optional[Dict]:
        """Get subscription details"""
        if not self.is_configured():
            return None
        
        try:
            subscription = stripe.Subscription.retrieve(subscription_id)
            return {
                'id': subscription.id,
                'status': subscription.status,
                'current_period_end': datetime.fromtimestamp(subscription.current_period_end),
                'cancel_at_period_end': subscription.cancel_at_period_end
            }
        except Exception as e:
            print(f"Error getting subscription: {e}")
            return None
    
    def cancel_subscription(self, subscription_id: str, immediate: bool = False) -> bool:
        """
        Cancel a subscription
        
        Args:
            subscription_id: Stripe subscription ID
            immediate: If True, cancel immediately. If False, cancel at period end.
        """
        if not self.is_configured():
            return False
        
        try:
            if immediate:
                stripe.Subscription.delete(subscription_id)
            else:
                stripe.Subscription.modify(
                    subscription_id,
                    cancel_at_period_end=True
                )
            return True
        except Exception as e:
            print(f"Error canceling subscription: {e}")
            return False
    
    def handle_webhook(self, payload: bytes, sig_header: str) -> Tuple[bool, str, Optional[Dict]]:
        """
        Handle Stripe webhook events
        
        Returns:
            Tuple of (success, event_type, event_data)
        """
        if not self.is_configured():
            return False, "not_configured", None
        
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, self.webhook_secret
            )
        except ValueError:
            return False, "invalid_payload", None
        except stripe.error.SignatureVerificationError:
            return False, "invalid_signature", None
        
        event_type = event['type']
        event_data = event['data']['object']
        
        return True, event_type, event_data
    
    def process_webhook_event(self, event_type: str, event_data: Dict) -> Dict:
        """
        Process a webhook event and return action to take
        
        Returns:
            Dict with 'action' and relevant data
        """
        if event_type == 'checkout.session.completed':
            # Payment successful, activate subscription
            customer_id = event_data.get('customer')
            subscription_id = event_data.get('subscription')
            metadata = event_data.get('metadata', {})
            
            return {
                'action': 'activate_subscription',
                'customer_id': customer_id,
                'subscription_id': subscription_id,
                'tier': metadata.get('tier', 'basic'),
                'billing_period': metadata.get('billing_period', 'monthly')
            }
        
        elif event_type == 'customer.subscription.updated':
            # Subscription updated (could be upgrade/downgrade)
            return {
                'action': 'update_subscription',
                'customer_id': event_data.get('customer'),
                'subscription_id': event_data.get('id'),
                'status': event_data.get('status'),
                'current_period_end': event_data.get('current_period_end')
            }
        
        elif event_type == 'customer.subscription.deleted':
            # Subscription canceled
            return {
                'action': 'cancel_subscription',
                'customer_id': event_data.get('customer'),
                'subscription_id': event_data.get('id')
            }
        
        elif event_type == 'invoice.payment_failed':
            # Payment failed
            return {
                'action': 'payment_failed',
                'customer_id': event_data.get('customer'),
                'subscription_id': event_data.get('subscription')
            }
        
        elif event_type == 'invoice.paid':
            # Payment successful (renewal)
            return {
                'action': 'payment_success',
                'customer_id': event_data.get('customer'),
                'subscription_id': event_data.get('subscription')
            }
        
        return {'action': 'unknown', 'event_type': event_type}
    
    def get_pricing_display(self) -> Dict:
        """Get pricing information for display"""
        return {
            'basic': {
                'name': 'Basic',
                'monthly': self.PRICING['basic']['monthly'],
                'yearly': self.PRICING['basic']['yearly'],
                'yearly_monthly': self.PRICING['basic']['yearly'] / 12,
                'savings': int((1 - (self.PRICING['basic']['yearly'] / 12) / self.PRICING['basic']['monthly']) * 100),
                'features': [
                    '25 predictions per day',
                    'Top 5 leagues (EPL, La Liga, Serie A, Bundesliga, Ligue 1)',
                    'Value bet indicators',
                    'Email support'
                ]
            },
            'pro': {
                'name': 'Pro',
                'monthly': self.PRICING['pro']['monthly'],
                'yearly': self.PRICING['pro']['yearly'],
                'yearly_monthly': self.PRICING['pro']['yearly'] / 12,
                'savings': int((1 - (self.PRICING['pro']['yearly'] / 12) / self.PRICING['pro']['monthly']) * 100),
                'features': [
                    '100 predictions per day',
                    'All 19 leagues',
                    'Value bet indicators',
                    'Full track record access',
                    'API access',
                    'Priority email support'
                ],
                'popular': True
            },
            'unlimited': {
                'name': 'Unlimited',
                'monthly': self.PRICING['unlimited']['monthly'],
                'yearly': self.PRICING['unlimited']['yearly'],
                'yearly_monthly': self.PRICING['unlimited']['yearly'] / 12,
                'savings': int((1 - (self.PRICING['unlimited']['yearly'] / 12) / self.PRICING['unlimited']['monthly']) * 100),
                'features': [
                    'Unlimited predictions',
                    'All 19 leagues',
                    'Value bet indicators',
                    'Full track record access',
                    'API access',
                    'Priority support',
                    'Early access to new features'
                ]
            }
        }


class MockPaymentManager:
    """Mock payment manager for testing without Stripe"""
    
    def __init__(self):
        self.mock_subscriptions = {}
    
    def is_configured(self) -> bool:
        return True
    
    def create_customer(self, email: str, user_id: int) -> str:
        return f"cus_mock_{user_id}"
    
    def create_checkout_session(self, customer_id: str, tier: str, 
                               billing_period: str = 'monthly', **kwargs) -> Dict:
        return {
            'session_id': f"cs_mock_{tier}_{billing_period}",
            'url': f"#mock-checkout?tier={tier}&period={billing_period}"
        }
    
    def activate_mock_subscription(self, user_id: int, tier: str, months: int = 1):
        """Activate a mock subscription (for testing)"""
        self.mock_subscriptions[user_id] = {
            'tier': tier,
            'expires': datetime.now() + timedelta(days=30 * months)
        }
        return True
    
    def get_pricing_display(self) -> Dict:
        return PaymentManager().get_pricing_display()


# Singleton instances
_payment_instance = None

def get_payment_manager(use_mock: bool = False) -> PaymentManager:
    """Get the payment manager (real or mock)"""
    global _payment_instance
    
    if use_mock or not STRIPE_AVAILABLE or not os.getenv('STRIPE_SECRET_KEY'):
        if _payment_instance is None or isinstance(_payment_instance, PaymentManager):
            _payment_instance = MockPaymentManager()
    else:
        if _payment_instance is None or isinstance(_payment_instance, MockPaymentManager):
            _payment_instance = PaymentManager()
    
    return _payment_instance
