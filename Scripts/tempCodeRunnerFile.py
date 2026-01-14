import sys
import MetaTrader5 as mt5

from src.data.mt5_fetch import MT5FetchConfig, mt5_initialize, mt5_shutdown, fetch_rates


def main():
    print("RUNNING AS __main__:", __name__ == "__main__")
    print("python:", sys.executable)
    print("cwd:", __import__("os").getcwd())

    try:
        mt5_initialize()

        cfg = MT5FetchConfig(
            symbol="EURUSD",
            timeframe=mt5.TIMEFRAME_M5,
            n_bars=500,
        )

        print("Fetching:", cfg)
        df = fetch_rates(cfg)

        print(df.head())
        print(df.tail())
        print(df.dtypes)
        print("OK: rows =", len(df))

    except Exception as e:
        print("ERROR:", repr(e))
        raise

    finally:
        mt5_shutdown()
        print("MT5 shutdown complete")


import os
import sys
import src

def main():
    print("python:", sys.executable)
    print("cwd:", os.getcwd())
    print("src imported from:", src.__file__)

    # ... dein restlicher code (mt5 init, fetch, prints) ...

if __name__ == "__main__":
    main()
    