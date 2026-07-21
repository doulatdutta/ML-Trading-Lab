#!/usr/bin/env python
"""CLI script to run live advisory scoring against active MT5 terminal."""

import sys
import os
import time
from dotenv import load_dotenv

from ml_trading_lab.LiveAdvisor.advisor import LiveAdvisor


def main() -> None:
    """Load credentials, connect to MT5, and print live advisory ratings."""
    load_dotenv()
    print("====================================================")
    print("         ML TRADING LAB - LIVE ADVISOR RUNNER       ")
    print("====================================================")
    
    # Load configuration
    login = os.getenv("MT5_LOGIN")
    server = os.getenv("MT5_SERVER")
    
    if not login or not server:
        print("Warning: MT5_LOGIN and MT5_SERVER env variables not found.")
        print("Make sure you copied '.env.example' to '.env' and set credentials.")
        print("Attempting connection to active local MT5 terminal default context...")

    # Initialize advisor
    # In a real environment, we'd pass the path to our promoted model candidate.
    # We will search the models directory or default to standard prior fallback.
    advisor = LiveAdvisor()

    print("\nStarting live polling loop (Press Ctrl+C to stop)...")
    symbol = "XAUUSD"
    timeframe = "M1"
    
    try:
        while True:
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Querying MT5 rates for {symbol}...")
            result = advisor.score(symbol=symbol, timeframe=timeframe)
            
            status = result.get("status")
            if status == "error":
                print(f"Error: {result.get('message')}")
            elif status == "no_setup":
                print(f"No setup: {result.get('message')}")
            elif status == "active_setup":
                print("*** ACTIVE SETUP DETECTED! ***")
                print(f"Direction:   {result.get('direction').upper()}")
                print(f"Timestamp:   {result.get('timestamp')}")
                print(f"Entry Price: {result.get('entry_price')}")
                print(f"Stop Loss:   {result.get('stop_loss')}")
                print(f"Take Profit: {result.get('take_profit')}")
                print(f"Win Prob:    {result.get('win_probability'):.2%}")
                print(f"Action:      {result.get('advisory_action').upper()}")
            
            # Poll every 15 seconds
            time.sleep(15)
            
    except KeyboardInterrupt:
        print("\nPolling stopped by user. Exiting.")
        sys.exit(0)


if __name__ == "__main__":
    main()
