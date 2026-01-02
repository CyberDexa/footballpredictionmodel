#!/usr/bin/env python3
"""
Standalone scheduler runner for cron/launchd automation.

Usage:
    # Run once (default - predictions for next 2 days):
    python run_scheduler.py
    
    # Run as daemon (checks every 6 hours):
    python run_scheduler.py --daemon
    
    # Specific leagues only:
    python run_scheduler.py --leagues EPL CHAMPIONSHIP
    
    # Just verify pending predictions:
    python run_scheduler.py --verify
    
    # Custom settings:
    python run_scheduler.py --days 3 --interval 12 --daemon

Cron example (run daily at 9 AM):
    0 9 * * * cd /path/to/footballpredictionmodel && /path/to/venv/bin/python run_scheduler.py >> logs/scheduler.log 2>&1

macOS LaunchAgent example:
    See docs/launchd_example.plist
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.scheduler import main

if __name__ == "__main__":
    main()
