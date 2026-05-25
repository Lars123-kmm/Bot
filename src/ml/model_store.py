from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.ml.meta_model import MetaModel

logger = logging.getLogger(__name__)

_META_FILE = "model_registry.json"


class ModelStore:
    """
    Speichert und lädt MetaModel-Instanzen mit vollständigen Metadaten.
    Unterstützt Versionierung: jedes Modell bekommt eine inkrementelle Version.

    Verzeichnisstruktur:
        models/
          meta_model_v1.pkl
          meta_model_v2.pkl
          model_registry.json
    """

    def __init__(self, model_dir: str | Path = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._registry_path = self.model_dir / _META_FILE

    # ------------------------------------------------------------------
    # Speichern
    # ------------------------------------------------------------------

    def save(
        self,
        model: MetaModel,
        metadata: Dict[str, Any],
        name: str = "meta_model",
    ) -> Path:
        """
        Speichert das Modell als .pkl und aktualisiert die Registry.
        Gibt den Pfad der gespeicherten Datei zurück.
        """
        registry = self._load_registry()
        version = self._next_version(registry, name)

        filename = f"{name}_v{version}.pkl"
        path = self.model_dir / filename
        model.save(path)

        entry = {
            "name": name,
            "version": version,
            "filename": filename,
            "saved_at": datetime.utcnow().isoformat(),
            "train_metrics": model.train_metrics_,
            "top_features": model.top_features(5),
            **metadata,
        }
        registry.setdefault(name, []).append(entry)
        self._save_registry(registry)

        logger.info("Modell gespeichert: %s (v%d, AUC=%.3f)",
                    path, version, model.train_metrics_.get("auc", 0))
        return path

    # ------------------------------------------------------------------
    # Laden
    # ------------------------------------------------------------------

    def load_latest(
        self,
        cfg: Dict[str, Any],
        name: str = "meta_model",
    ) -> Tuple[MetaModel, Dict[str, Any]]:
        """Lädt das neueste Modell nach Versionsnummer."""
        registry = self._load_registry()
        entries = registry.get(name, [])
        if not entries:
            raise FileNotFoundError(
                f"Kein Modell '{name}' gefunden in {self._registry_path}. "
                "Zuerst --mode train ausführen."
            )
        latest = max(entries, key=lambda e: e["version"])
        path = self.model_dir / latest["filename"]
        model = MetaModel(cfg).load(path)
        return model, latest

    def load_champion(
        self,
        cfg: Dict[str, Any],
        name: str = "meta_model",
    ) -> Tuple[MetaModel, Dict[str, Any]]:
        """
        Lädt das neueste Modell mit champion_status == "champion".
        Fällt auf das neueste Modell zurück, falls noch kein Champion markiert ist
        (z.B. Modelle aus der Zeit vor der Champion-Challenger-Integration).
        """
        registry = self._load_registry()
        entries = registry.get(name, [])
        if not entries:
            raise FileNotFoundError(
                f"Kein Modell '{name}' gefunden in {self._registry_path}. "
                "Zuerst --mode train ausführen."
            )
        champions = [e for e in entries if e.get("champion_status") == "champion"]
        pool = champions if champions else entries
        latest = max(pool, key=lambda e: e["version"])
        path = self.model_dir / latest["filename"]
        model = MetaModel(cfg).load(path)
        return model, latest

    def load_version(
        self,
        cfg: Dict[str, Any],
        version: int,
        name: str = "meta_model",
    ) -> Tuple[MetaModel, Dict[str, Any]]:
        """Lädt eine spezifische Modellversion."""
        registry = self._load_registry()
        entries = registry.get(name, [])
        match = next((e for e in entries if e["version"] == version), None)
        if match is None:
            raise FileNotFoundError(f"Modell '{name}' Version {version} nicht gefunden.")
        path = self.model_dir / match["filename"]
        model = MetaModel(cfg).load(path)
        return model, match

    # ------------------------------------------------------------------
    # Übersicht
    # ------------------------------------------------------------------

    def list_versions(self, name: str = "meta_model") -> List[Dict[str, Any]]:
        """Gibt alle gespeicherten Versionen mit Metadaten zurück."""
        registry = self._load_registry()
        return registry.get(name, [])

    def print_summary(self, name: str = "meta_model") -> None:
        versions = self.list_versions(name)
        if not versions:
            print(f"Keine Modelle für '{name}' gespeichert.")
            return
        print(f"\n{'='*60}")
        print(f"  Modell-Versionen: {name}")
        print(f"{'='*60}")
        for v in versions:
            auc = v.get("train_metrics", {}).get("auc", 0)
            print(f"  v{v['version']:2d}  {v['saved_at'][:10]}  AUC={auc:.3f}  {v.get('description', '')}")
        print()

    # ------------------------------------------------------------------
    # Interne Hilfsmethoden
    # ------------------------------------------------------------------

    def _load_registry(self) -> Dict[str, Any]:
        if self._registry_path.exists():
            with self._registry_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_registry(self, registry: Dict[str, Any]) -> None:
        with self._registry_path.open("w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)

    def _next_version(self, registry: Dict[str, Any], name: str) -> int:
        entries = registry.get(name, [])
        return max((e["version"] for e in entries), default=0) + 1
