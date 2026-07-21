"""CLI script — download 1-year of MT5 OHLCV data to Config/raw/ cache."""

import os
import sys

# Ensure src is in path when run from project root
_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _src not in sys.path:
    sys.path.insert(0, _src)


def main():
    print("=" * 56)
    print("   ML Trading Lab — MT5 Historical Data Downloader")
    print("=" * 56)
    print()

    from ml_trading_lab.DataEngine.mt5_downloader import MT5DataDownloader

    downloader = MT5DataDownloader(raw_directory="Config/raw")

    pairs = [
        ("XAUUSD", "M1",  1.0),
        ("XAUUSD", "M3",  1.0),
        ("XAUUSD", "M15", 1.0),
    ]

    for symbol, tf, years in pairs:
        print(f"\n{'─'*56}")
        print(f"Downloading {symbol} {tf} ({years} year)...")

        def cb(pct, msg):
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"\r  [{bar}] {pct:3d}%  {msg}", end="", flush=True)

        try:
            df = downloader.download(symbol, tf, years=years,
                                     overwrite=False, progress_callback=cb)
            print(f"\n  ✅ {len(df):,} bars saved.")
        except Exception as e:
            print(f"\n  ❌ Failed: {e}")

    print("\n" + "=" * 56)
    print("Download complete. Run the dashboard to begin training.")
    print("=" * 56)


if __name__ == "__main__":
    main()
