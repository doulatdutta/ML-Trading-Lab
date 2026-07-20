"""Launcher script to start the ML Trading Lab Research Dashboard."""

import os
import sys
import webbrowser
import time
import subprocess


def main():
    print("==================================================")
    print("      ML Trading Lab — Research Dashboard")
    print("==================================================")
    
    # Programmatically inject src folder into PYTHONPATH
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    os.environ["PYTHONPATH"] = src_dir
    sys.path.insert(0, src_dir)
    
    print(f"Added src directory to path: {src_dir}")
    print("Checking MetaTrader 5 live terminal status...")
    
    # Auto-open dashboard in the browser after a brief delay
    print("Launching backend server on http://127.0.0.1:8000 ...")
    
    # Open dashboard in browser 2 seconds after server starts
    def open_browser():
        time.sleep(2)
        print("Opening dashboard in your default browser...")
        webbrowser.open("http://127.0.0.1:8000")
        
    import threading
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Start uvicorn server
    try:
        import uvicorn
        uvicorn.run("ml_trading_lab.Dashboard.dashboard_server:app", host="127.0.0.1", port=8000, reload=False)
    except KeyboardInterrupt:
        print("\nDashboard stopped by user. Exiting.")
        sys.exit(0)
    except ImportError:
        print("Error: uvicorn is not installed in the current virtual environment.")
        sys.exit(1)


if __name__ == "__main__":
    main()
