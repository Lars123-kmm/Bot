#!/usr/bin/env python3
"""
start.py — Startdatei für den ML-Trading-Bot
Ausführen: python start.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def clear():
    print("\033[2J\033[H", end="")


def menu():
    clear()
    print("╔══════════════════════════════════════════════════════╗")
    print("║           ML-TRADING-BOT  —  STARTMENÜ              ║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  1)  Demo          — Alle Komponenten testen         ║")
    print("║                      (kein MT5, keine echten Daten)  ║")
    print("║                                                      ║")
    print("║  2)  Training      — Modell mit CSV-Datei trainieren ║")
    print("║  3)  Backtest      — Backtest mit CSV-Datei + ML     ║")
    print("║  4)  Optimierung   — Hyperparameter optimieren       ║")
    print("║  5)  Live-Trading  — Echtzeit-Trading via MT5        ║")
    print("║                                                      ║")
    print("║  0)  Beenden                                         ║")
    print("╚══════════════════════════════════════════════════════╝")
    return input("\n  Auswahl (0–5): ").strip()


def ask_csv() -> str:
    path = input("  Pfad zur CSV-Datei (z.B. data/btc.csv): ").strip()
    if not Path(path).exists():
        print(f"\n  ✗ Datei nicht gefunden: {path}")
        sys.exit(1)
    return path


def run(cmd: list[str]) -> None:
    print(f"\n  Starte: {' '.join(cmd)}\n")
    subprocess.run(cmd, cwd=ROOT)


def main():
    while True:
        choice = menu()

        if choice == "0":
            print("\n  Auf Wiedersehen.\n")
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

        elif choice == "5":
            print("\n  ⚠  Live-Trading benötigt MetaTrader 5.")
            print("     Config: src/config/default.yaml")
            ok = input("  Wirklich starten? (ja/nein): ").strip().lower()
            if ok == "ja":
                run([sys.executable, "src/main.py", "--mode", "live"])
            input("\n  [Enter] zurück zum Menü...")

        else:
            print("  Ungültige Auswahl.")


if __name__ == "__main__":
    main()
