#!/data/data/com.termux/files/usr/bin/bash
# Termux Setup Script — einmalig ausfuehren

echo "=== Termux Bot Setup ==="

# System-Pakete
pkg update -y && pkg upgrade -y
pkg install -y python git libzmq screen

# Python-Pakete (Schritt fuer Schritt wegen ARM-Kompilierung)
pip install --upgrade pip wheel

echo "Installiere numpy/pandas..."
pip install numpy pandas PyYAML joblib requests

echo "Installiere scikit-learn..."
pip install scikit-learn

echo "Installiere LightGBM..."
pip install lightgbm

echo "Installiere weitere ML-Pakete..."
pip install optuna scipy statsmodels hmmlearn river

echo "Installiere ccxt (Binance)..."
pip install ccxt

echo ""
echo "=== Setup abgeschlossen ==="
echo "Bot starten mit: bash termux_start.sh"
