#!/data/data/com.termux/files/usr/bin/bash
# Bot starten in einer screen-Session (laeuft weiter wenn Termux minimiert wird)

cd "$(dirname "$0")/bot"

# CPU-Sleep verhindern
termux-wake-lock 2>/dev/null || true

# Bestehende Session beenden falls vorhanden
screen -S bot -X quit 2>/dev/null || true

# Neue screen-Session starten
screen -dmS bot bash -c "python start.py 2>&1 | tee ../bot.log"

echo "Bot gestartet. Screen-Session: 'bot'"
echo ""
echo "Befehle:"
echo "  screen -r bot        -> Bot-Output anzeigen"
echo "  Ctrl+A dann D        -> Screen verlassen (Bot laeuft weiter)"
echo "  tail -f ../bot.log   -> Log verfolgen"
echo "  screen -S bot -X quit -> Bot stoppen"
