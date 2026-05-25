#!/usr/bin/env python3
"""
start.py — Startdatei für den ML-Trading-Bot
Ausführen: python start.py
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def clear():
    print("\033[2J\033[H", end="")


def menu():
    clear()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║             ML-TRADING-BOT  —  STARTMENÜ                    ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  1)  Demo            — Alle Komponenten testen               ║")
    print("║                        (kein MT5, keine echten Daten)        ║")
    print("║                                                              ║")
    print("║  2)  Training (CSV)  — Modell mit CSV-Datei trainieren       ║")
    print("║  3)  Backtest        — Backtest mit CSV-Datei + ML           ║")
    print("║  4)  Optimierung     — Hyperparameter optimieren             ║")
    print("║  8)  Training (Live) — Modell mit Binance-Daten trainieren  ║")
    print("║                                                              ║")
    print("║  5)  MT5 Live        — Echtzeit-Trading via MetaTrader 5     ║")
    print("║                                                              ║")
    print("║  6)  Paper Trading   — Live Binance-Daten, kein echtes Geld ║")
    print("║                        (Bot lernt automatisch dazu)          ║")
    print("║  7)  Echtes Trading  — Live Binance-Orders (API-Key nötig)  ║")
    print("║                                                              ║")
    print("║  0)  Beenden                                                 ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    return input("\n  Auswahl (0–8): ").strip()


def ask_csv() -> str:
    path = input("  Pfad zur CSV-Datei (z.B. data/btc.csv): ").strip()
    if not Path(path).exists():
        print(f"\n  ✗ Datei nicht gefunden: {path}")
        sys.exit(1)
    return path


def ask_symbols() -> str:
    default = "BTCUSDT,ETHUSDT"
    s = input(f"  Symbole (komma-getrennt) [{default}]: ").strip()
    return s if s else default


def run(cmd: list[str]) -> None:
    # Replace sys.executable with the Python that launched start.py
    cmd = [sys.executable if c == sys.executable else c for c in cmd]
    print(f"\n  Starte: {' '.join(cmd)}\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    env["OMP_NUM_THREADS"] = "8"
    subprocess.run(cmd, cwd=ROOT, env=env)


def main():
    while True:
        choice = menu()

        if choice == "0":
            print("\n  Auf Wiedersehen Lars.\n")
            break

        elif choice == "1":
            run([sys.executable, "Scripts/demo_pipeline.py"])
            input("\n  [Enter] zurück zum Menü...")

        elif choice == "2":
            csv = ask_csv()
            run([sys.executable, "src/main.py", "--mode", "train", "--csv", csv])
            input("\n  [Enter] zurück zum Menü...")

        elif choice == "3":
            csv = ask_csv()
            capital = input("  Startkapital in USD [10000]: ").strip() or "10000"
            run([
                sys.executable, "src/main.py",
                "--mode", "backtest",
                "--csv", csv,
                "--use-ml",
                "--capital", capital,
            ])
            input("\n  [Enter] zurück zum Menü...")

        elif choice == "4":
            csv = ask_csv()
            trials = input("  Anzahl Optuna-Trials [50]: ").strip() or "50"
            run([
                sys.executable, "src/main.py",
                "--mode", "optimize",
                "--csv", csv,
                "--trials", trials,
            ])
            input("\n  [Enter] zurück zum Menü...")

        elif choice == "8":
            print("\n  Training mit Live-Daten von Binance (kein API-Key nötig)")
            symbols = ask_symbols()
            run([
                sys.executable, "src/main.py",
                "--mode", "train",
                "--symbols", symbols,
            ])
            input("\n  [Enter] zurück zum Menü...")

        elif choice == "5":
            print("\n  ⚠  MT5 Live-Trading benötigt MetaTrader 5.")
            print("     Config: src/config/default.yaml")
            ok = input("  Wirklich starten? (ja/nein): ").strip().lower()
            if ok == "ja":
                run([sys.executable, "src/main.py", "--mode", "live"])
            input("\n  [Enter] zurück zum Menü...")

        elif choice == "6":
            print("\n  Paper Trading auf Live Binance-Daten")
            print("  Kein API-Key nötig. Der Bot lernt automatisch nach jedem Trade.\n")
            symbols = ask_symbols()
            capital = input("  Startkapital in USD [10000]: ").strip() or "10000"
            retrain = input("  Auto-Retrain alle N Bars (0 = aus) [0]: ").strip() or "0"
            run([
                sys.executable, "src/main.py",
                "--mode", "paper",
                "--symbols", symbols,
                "--capital", capital,
                "--retrain-every", retrain,
            ])
            input("\n  [Enter] zurück zum Menü...")

        elif choice == "7":
            print("\n  ⚠  ECHTES TRADING — ECHTES GELD!")
            print("     Benötigt:")
            print("       - BINANCE_API_KEY  (Umgebungsvariable)")
            print("       - BINANCE_API_SECRET  (Umgebungsvariable)")
            print("       - 'live.real_orders_confirmed: true' in src/config/default.yaml")
            print("       - 'live.use_testnet: false' für Mainnet\n")
            ok = input("  Wirklich starten? (ja/nein): ").strip().lower()
            if ok == "ja":
                symbols = ask_symbols()
                capital = input("  Startkapital-Referenz in USD [10000]: ").strip() or "10000"
                retrain = input("  Auto-Retrain alle N Bars (0 = aus) [0]: ").strip() or "0"
                run([
                    sys.executable, "src/main.py",
                    "--mode", "trade",
                    "--symbols", symbols,
                    "--capital", capital,
                    "--retrain-every", retrain,
                ])
            input("\n  [Enter] zurück zum Menü...")

        else:
            print("  Ungültige Auswahl.")


if __name__ == "__main__":
    main()
