"""Run this once to pre-create all SQLite databases."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.bar_store import BarStore
from src.ml.data_store import TradeDataStore
from src.ml.adaptive_features import AdaptiveFeatureSelector
from src.analysis.efficiency import EfficiencyTracker

BarStore("data/bars.db")
TradeDataStore("data/trade_history.db")
AdaptiveFeatureSelector("data/feature_importance.db")
EfficiencyTracker("data/efficiency_history.db")

for f in ["data/bars.db", "data/trade_history.db",
          "data/feature_importance.db", "data/efficiency_history.db"]:
    size = os.path.getsize(f)
    print(f"  OK  {f}  ({size} bytes)")

print("\nAlle Datenbanken bereit.")
