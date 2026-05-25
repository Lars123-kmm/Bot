from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from statistics import mean
from typing import Any, List, Optional

from src.common.types import ClosedTrade
from src.risk.guards import ConsecutiveLossGuard, DrawdownGuard

logger = logging.getLogger(__name__)


class DailyReporter:
    """
    Erzeugt am Ende jedes Handelstages einen separaten Bericht als Markdown-Datei.
    Der Reporter ist bewusst von der Trading-Logik entkoppelt –
    er liest nur Trade-Daten aus und schreibt eine eigene Datei.
    """

    def __init__(self, report_dir: Path):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        report_date: Any,
        trades: List[ClosedTrade],
        equity_start: float,
        equity_end: float,
        drawdown_guard: DrawdownGuard,
        loss_guard: ConsecutiveLossGuard,
        indicator_summary: Optional[dict] = None,
        regime_summary: Optional[dict] = None,
        ml_summary: Optional[dict] = None,
        feature_importances: Optional[dict] = None,
    ) -> Path:
        """
        Erstellt den Tagesbericht und speichert ihn als .md Datei.
        Gibt den Pfad zur erzeugten Datei zurück.
        Wirft keine Exceptions – Fehler werden nur geloggt.

        Args:
            regime_summary:      {"trend_pct": 60, "range_pct": 30, "volatile_pct": 10, "dominant": "Trend"}
            ml_summary:          {"avg_proba": 0.63, "signals_filtered": 3, "model_auc": 0.58}
            feature_importances: {"ema_spread": 0.15, "adx_momentum": 0.12, ...} (Top-N)
        """
        try:
            content = self._build_report(
                report_date, trades, equity_start, equity_end,
                drawdown_guard, loss_guard, indicator_summary,
                regime_summary, ml_summary, feature_importances,
            )
            path = self.report_dir / f"{report_date}.md"
            path.write_text(content, encoding="utf-8")
            return path
        except Exception as exc:
            logger.error("DailyReporter.generate() fehlgeschlagen: %s", exc)
            raise

    def _build_report(
        self,
        report_date: Any,
        trades: List[ClosedTrade],
        equity_start: float,
        equity_end: float,
        drawdown_guard: DrawdownGuard,
        loss_guard: ConsecutiveLossGuard,
        indicator_summary: Optional[dict],
        regime_summary: Optional[dict] = None,
        ml_summary: Optional[dict] = None,
        feature_importances: Optional[dict] = None,
    ) -> str:
        day_return = ((equity_end / equity_start) - 1) * 100 if equity_start > 0 else 0.0
        sign = "+" if day_return >= 0 else ""

        _, total_dd = drawdown_guard.update(equity_end)
        dd_limit = drawdown_guard.max_dd * 100
        daily_dd_limit = 2.0  # Standard-Limit
        consec = loss_guard.consecutive_losses
        consec_limit = loss_guard.max_losses

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        n = len(trades)

        win_rate = len(wins) / n * 100 if n > 0 else 0.0
        sum_wins = sum(t.pnl for t in wins)
        sum_losses = abs(sum(t.pnl for t in losses))
        profit_factor = sum_wins / sum_losses if sum_losses > 0 else float("inf")

        best_trade = max((t.pnl for t in trades), default=0.0)
        worst_trade = min((t.pnl for t in trades), default=0.0)

        avg_duration_bars = mean(t.bars_held for t in trades) if trades else 0

        lines = [
            f"# DAILY TRADING REPORT: {report_date}",
            "",
            "## Kontostand",
            f"| | Wert |",
            f"|---|---|",
            f"| Start des Tages | {equity_start:,.2f} USD |",
            f"| Ende des Tages  | {equity_end:,.2f} USD |",
            f"| Tagesrendite    | {sign}{day_return:.2f}% |",
            "",
            f"## Trades heute ({n})",
        ]

        if trades:
            lines += [
                "| Zeit (UTC) | Richtung | Entry | Exit | PnL | Grund |",
                "|---|---|---|---|---|---|",
            ]
            for t in trades:
                direction = "LONG" if t.direction == 1 else "SHORT"
                time_str = t.close_time.strftime("%H:%M") if hasattr(t, "close_time") else "--:--"
                pnl_sign = "+" if t.pnl >= 0 else ""
                lines.append(
                    f"| {time_str} | {direction} | {t.entry_price:.5f} | "
                    f"{t.exit_price:.5f} | {pnl_sign}{t.pnl:.2f} | {t.reason} |"
                )
        else:
            lines.append("_Keine Trades heute._")

        lines += [
            "",
            "## Statistik des Tages",
            f"| Metrik | Wert |",
            f"|---|---|",
            f"| Win Rate | {win_rate:.1f}% ({len(wins)}/{n}) |",
            f"| Profit Factor | {profit_factor:.2f} |",
            f"| Größter Gewinn | +{best_trade:.2f} USD |",
            f"| Größter Verlust | {worst_trade:.2f} USD |",
            f"| Avg. Haltedauer | {avg_duration_bars:.1f} Bars |",
            "",
            "## Risiko-Status",
            f"| Guard | Wert | Limit | Status |",
            f"|---|---|---|---|",
        ]

        dd_status = "✓" if total_dd * 100 < dd_limit else "⚠ LIMIT ERREICHT"
        consec_status = "✓" if consec < consec_limit else "⚠ LIMIT ERREICHT"

        lines += [
            f"| Gesamt-Drawdown seit Peak | {total_dd * 100:.1f}% | {dd_limit:.0f}% | {dd_status} |",
            f"| Aufeinanderfolgende Verluste | {consec} | {consec_limit} | {consec_status} |",
            "",
            "## Einschätzung",
        ]

        # --- KI/ML-Sektionen ---
        if regime_summary:
            lines += [
                "",
                "## Regime-Analyse",
                "| Marktphase | Anteil heute |",
                "|---|---|",
                f"| Trending | {regime_summary.get('trend_pct', 0):.1f}% |",
                f"| Range (kein Trading) | {regime_summary.get('range_pct', 0):.1f}% |",
                f"| Volatil (halbes Sizing) | {regime_summary.get('volatile_pct', 0):.1f}% |",
                f"| **Dominante Phase** | **{regime_summary.get('dominant', '?')}** |",
            ]

        if ml_summary:
            lines += [
                "",
                "## Meta-Modell (KI)",
                "| Metrik | Wert |",
                "|---|---|",
                f"| Durchschnittliche Konfidenz | {ml_summary.get('avg_proba', 0)*100:.1f}% |",
                f"| Gefilterte Signale (zu niedrige Konfidenz) | {ml_summary.get('signals_filtered', 0)} |",
                f"| Modell-AUC (Trainingsset) | {ml_summary.get('model_auc', 0):.3f} |",
            ]

        if feature_importances:
            top5 = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:5]
            lines += [
                "",
                "## Top-5 Einfluss-Faktoren (Feature Importance)",
                "| Feature | Gewicht |",
                "|---|---|",
            ]
            for feat, imp in top5:
                lines.append(f"| {feat} | {imp:.4f} |")

        assessment = _generate_assessment(
            n_trades=n,
            win_rate=win_rate,
            profit_factor=profit_factor,
            day_return=day_return,
            total_dd=total_dd * 100,
            dd_limit=dd_limit,
            consec=consec,
            consec_limit=consec_limit,
            indicator_summary=indicator_summary,
            regime_summary=regime_summary,
            ml_summary=ml_summary,
        )
        lines.append("")
        lines.append("## Einschätzung")
        lines.append(assessment)
        lines.append("")
        lines.append("---")
        lines.append("_Automatisch generiert vom EMA-Crossover-Bot mit KI-Analyse_")

        return "\n".join(lines)


def _generate_assessment(
    n_trades: int,
    win_rate: float,
    profit_factor: float,
    day_return: float,
    total_dd: float,
    dd_limit: float,
    consec: int,
    consec_limit: int,
    indicator_summary: Optional[dict],
    regime_summary: Optional[dict] = None,
    ml_summary: Optional[dict] = None,
) -> str:
    parts = []

    if n_trades == 0:
        parts.append("**Aktivität:** Heute wurden keine Trades ausgeführt – "
                     "der Markt hat die Filterbedingungen (ADX, RSI, Volumen) nicht erfüllt.")
    elif win_rate >= 60 and profit_factor >= 1.5:
        parts.append("**Qualität:** Starker Tag – Win Rate und Profit Factor über Zielwerten.")
    elif win_rate < 40 or profit_factor < 1.0:
        parts.append("**Qualität:** Schwacher Tag – Win Rate oder Profit Factor unter Erwartung. "
                     "Marktbedingungen prüfen (Range-Markt? Erhöhte Volatilität?).")
    else:
        parts.append("**Qualität:** Solider Tag im Erwartungsbereich der Strategie.")

    if day_return > 1.5:
        parts.append("**Rendite:** Überdurchschnittlicher Tagesgewinn.")
    elif day_return < -1.0:
        parts.append("**Rendite:** Verlusttag – innerhalb akzeptabler Grenzen solange Drawdown-Limits gehalten werden.")

    if total_dd >= dd_limit * 0.8:
        parts.append(f"**WARNUNG:** Gesamt-Drawdown bei {total_dd:.1f}% – nahe am Circuit-Breaker-Limit ({dd_limit:.0f}%). "
                     "Positionen und Strategie überprüfen.")
    elif total_dd < dd_limit * 0.3:
        parts.append("**Risiko:** Drawdown im grünen Bereich.")

    if consec >= consec_limit - 1:
        parts.append(f"**WARNUNG:** {consec} aufeinanderfolgende Verluste – ein weiterer Verlust löst die Loss Guard aus.")

    if indicator_summary:
        avg_adx = indicator_summary.get("avg_adx")
        if avg_adx is not None:
            phase = "Trending" if avg_adx > 30 else ("Übergangsphase" if avg_adx > 20 else "Range")
            parts.append(f"**Marktphase:** {phase} (Durchschnittlicher ADX: {avg_adx:.1f}).")

    if regime_summary:
        dominant = regime_summary.get("dominant", "?")
        range_pct = regime_summary.get("range_pct", 0)
        vol_pct = regime_summary.get("volatile_pct", 0)
        if range_pct > 50:
            parts.append(f"**KI-Regime:** Heute überwiegend Range-Phase ({range_pct:.0f}%) – "
                         "viele Signale wurden vom Regime-Gate blockiert. Das ist erwünscht.")
        elif vol_pct > 30:
            parts.append(f"**KI-Regime:** Erhöhte Volatilität ({vol_pct:.0f}% der Zeit) – "
                         "Positionen wurden mit halbem Sizing gefahren.")
        else:
            parts.append(f"**KI-Regime:** Dominante Phase '{dominant}' – optimale Handelsbedingungen.")

    if ml_summary:
        avg_proba = ml_summary.get("avg_proba", 0)
        filtered = ml_summary.get("signals_filtered", 0)
        auc = ml_summary.get("model_auc", 0)
        if avg_proba > 0.65:
            parts.append(f"**KI-Konfidenz:** Hoch ({avg_proba*100:.1f}%) – "
                         "Modell zeigt starke Überzeugung in heutige Setups.")
        elif avg_proba < 0.55:
            parts.append(f"**KI-Konfidenz:** Niedrig ({avg_proba*100:.1f}%) – "
                         f"{filtered} Signale wurden gefiltert. Markt möglicherweise schwer vorhersagbar.")
        if auc > 0:
            parts.append(f"**Modell-Qualität:** AUC={auc:.3f} "
                         f"({'gut' if auc > 0.6 else 'akzeptabel' if auc > 0.55 else 'schwach – Retraining erwägen'}).")

    if not parts:
        parts.append("Strategie läuft planmäßig. Keine Anpassungen nötig.")

    return "\n\n".join(parts)
