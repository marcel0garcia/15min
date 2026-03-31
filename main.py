#!/usr/bin/env python3
# If you hit "No module named X", run with: python3 main.py
"""
15-Min BTC Prediction Bot — Entry Point

Usage:
  python main.py                   # dashboard, paper mode, signal-only
  python main.py run --trade       # enable auto-trading (still paper)
  python main.py run --trade --live  # REAL MONEY auto-trading
  python main.py markets           # list open markets
  python main.py positions         # current positions
  python main.py balance           # account balance
  python main.py history           # trade log
  python main.py trade yes TICKER AMT  # manual trade
  python main.py train             # train ML model
  python main.py config            # show config
"""
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

from btc15.cli.app import cli

if __name__ == "__main__":
    cli()
