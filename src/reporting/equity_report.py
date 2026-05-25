"""Equity curve HTML report — generated after each retrain cycle and daily.

Produces a self-contained HTML file (no server needed) with:
  - Cumulative equity curve
  - Drawdown chart
  - Rolling win-rate (last 20 trades)
  - PnL distribution histogram
  - Per-symbol breakdown table
  - Feature importance bar chart (if model available)
  - Market efficiency history per symbol

Charts rendered with Chart.js (CDN). All data embedded inline as JSON.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _json_default(obj: object) -> str:
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


def generate_report(
    trades: List[Dict[str, Any]],
    initial_capital: float,
    output_path: str = "reports/equity_report.html",
    feature_importances: Optional[Dict[str, float]] = None,
    efficiency_history: Optional[Dict[str, List]] = None,
    symbols: Optional[List[str]] = None,
) -> str:
    """
    Generate a self-contained HTML equity report.

    Args:
        trades:               List of closed trade dicts from PaperBroker.
        initial_capital:      Starting capital in USD.
        output_path:          Where to write the HTML file.
        feature_importances:  Dict {feature: importance} from MetaModel (optional).
        efficiency_history:   Dict {symbol: [{"ts":..,"composite":..}]} (optional).
        symbols:              List of traded symbols.

    Returns:
        Absolute path to the generated report.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # ── Compute series from trades ────────────────────────────────────
    equity_labels, equity_data = _equity_curve(trades, initial_capital)
    dd_data = _drawdown_series(equity_data)
    winrate_labels, winrate_data = _rolling_winrate(trades, window=20)
    pnl_buckets, pnl_counts = _pnl_histogram(trades, n_bins=20)
    per_symbol = _per_symbol_stats(trades)
    summary = _summary_stats(trades, initial_capital, equity_data)

    # ── Serialise optional data ───────────────────────────────────────
    feat_labels = feat_values = "[]"
    if feature_importances:
        top = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:15]
        feat_labels = json.dumps([k for k, _ in top], default=_json_default)
        feat_values = json.dumps([round(v, 2) for _, v in top])

    eff_datasets = "[]"
    if efficiency_history:
        datasets = []
        colors = ["#4fc3f7", "#81c784", "#ffb74d", "#e57373", "#ce93d8"]
        for i, (sym, rows) in enumerate(efficiency_history.items()):
            color = colors[i % len(colors)]
            datasets.append({
                "label": sym,
                "data": [{"x": r.get("ts", ""), "y": round(r.get("composite", 0), 3)}
                         for r in rows],
                "borderColor": color,
                "tension": 0.3,
                "pointRadius": 1,
                "fill": False,
            })
        eff_datasets = json.dumps(datasets, default=_json_default)

    # ── Render HTML ───────────────────────────────────────────────────
    html = _render_html(
        summary=summary,
        equity_labels=json.dumps(equity_labels, default=_json_default),
        equity_data=json.dumps([round(v, 2) for v in equity_data]),
        dd_data=json.dumps([round(v * 100, 2) for v in dd_data]),
        winrate_labels=json.dumps(winrate_labels, default=_json_default),
        winrate_data=json.dumps([round(v * 100, 1) for v in winrate_data]),
        pnl_buckets=json.dumps([round(b, 1) for b in pnl_buckets]),
        pnl_counts=json.dumps(pnl_counts),
        per_symbol=per_symbol,
        feat_labels=feat_labels,
        feat_values=feat_values,
        eff_datasets=eff_datasets,
        initial_capital=initial_capital,
        symbols=symbols or [],
    )

    Path(output_path).write_text(html, encoding="utf-8")
    logger.info("Equity report saved: %s", output_path)
    return str(Path(output_path).resolve())


# ── Data helpers ──────────────────────────────────────────────────────────────

def _equity_curve(
    trades: List[Dict], initial_capital: float
) -> tuple[List[str], List[float]]:
    labels = ["Start"]
    equity = [initial_capital]
    for t in sorted(trades, key=lambda x: x.get("close_time", "")):
        pnl = t.get("pnl") or 0.0
        labels.append(str(t.get("close_time", ""))[:16])
        equity.append(equity[-1] + pnl)
    return labels, equity


def _drawdown_series(equity: List[float]) -> List[float]:
    dd = []
    peak = equity[0] if equity else 1.0
    for v in equity:
        if v > peak:
            peak = v
        dd.append((v - peak) / peak if peak > 0 else 0.0)
    return dd


def _rolling_winrate(
    trades: List[Dict], window: int = 20
) -> tuple[List[str], List[float]]:
    sorted_t = sorted(trades, key=lambda x: x.get("close_time", ""))
    labels, rates = [], []
    wins = []
    for t in sorted_t:
        wins.append(1 if (t.get("pnl") or 0) > 0 else 0)
        if len(wins) >= window:
            rates.append(sum(wins[-window:]) / window)
            labels.append(str(t.get("close_time", ""))[:16])
    return labels, rates


def _pnl_histogram(
    trades: List[Dict], n_bins: int = 20
) -> tuple[List[float], List[int]]:
    pnls = [t.get("pnl") or 0.0 for t in trades]
    if not pnls:
        return [], []
    lo, hi = min(pnls), max(pnls)
    if lo == hi:
        return [lo], [len(pnls)]
    step = (hi - lo) / n_bins
    buckets = [lo + i * step for i in range(n_bins + 1)]
    counts = [0] * n_bins
    for p in pnls:
        idx = min(int((p - lo) / step), n_bins - 1)
        counts[idx] += 1
    return buckets[:-1], counts


def _per_symbol_stats(trades: List[Dict]) -> List[Dict]:
    data: Dict[str, Dict] = {}
    for t in trades:
        sym = t.get("symbol", "?")
        if sym not in data:
            data[sym] = {"n": 0, "wins": 0, "pnl": 0.0}
        pnl = t.get("pnl") or 0.0
        data[sym]["n"] += 1
        data[sym]["pnl"] += pnl
        if pnl > 0:
            data[sym]["wins"] += 1
    result = []
    for sym, d in data.items():
        result.append({
            "symbol": sym,
            "n_trades": d["n"],
            "win_rate": round(d["wins"] / d["n"] * 100, 1) if d["n"] else 0,
            "total_pnl": round(d["pnl"], 2),
            "avg_pnl": round(d["pnl"] / d["n"], 2) if d["n"] else 0,
        })
    return sorted(result, key=lambda x: x["total_pnl"], reverse=True)


def _summary_stats(
    trades: List[Dict], initial_capital: float, equity: List[float]
) -> Dict[str, Any]:
    pnls = [t.get("pnl") or 0.0 for t in trades]
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    total_pnl = sum(pnls)
    final_equity = equity[-1] if equity else initial_capital
    max_dd = min(_drawdown_series(equity)) if equity else 0.0

    win_pnls = [p for p in pnls if p > 0]
    loss_pnls = [abs(p) for p in pnls if p < 0]
    avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0
    avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0
    profit_factor = sum(win_pnls) / sum(loss_pnls) if loss_pnls else float("inf")

    import math
    avg = sum(pnls) / n if n else 0
    var = sum((p - avg) ** 2 for p in pnls) / n if n > 1 else 0
    sharpe = (avg / math.sqrt(var) * math.sqrt(252)) if var > 0 else 0.0

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "n_trades": n,
        "win_rate": round(wins / n * 100, 1) if n else 0,
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round((final_equity / initial_capital - 1) * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "∞",
        "sharpe": round(sharpe, 3),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "initial_capital": initial_capital,
        "final_equity": round(final_equity, 2),
    }


# ── HTML template ─────────────────────────────────────────────────────────────

def _render_html(
    summary: Dict,
    equity_labels: str,
    equity_data: str,
    dd_data: str,
    winrate_labels: str,
    winrate_data: str,
    pnl_buckets: str,
    pnl_counts: str,
    per_symbol: List[Dict],
    feat_labels: str,
    feat_values: str,
    eff_datasets: str,
    initial_capital: float,
    symbols: List[str],
) -> str:
    sym_rows = "\n".join(
        f"""<tr>
          <td>{r['symbol']}</td>
          <td>{r['n_trades']}</td>
          <td class="{'pos' if r['win_rate'] >= 50 else 'neg'}">{r['win_rate']}%</td>
          <td class="{'pos' if r['total_pnl'] >= 0 else 'neg'}">{r['total_pnl']:+.2f}</td>
          <td class="{'pos' if r['avg_pnl'] >= 0 else 'neg'}">{r['avg_pnl']:+.2f}</td>
        </tr>"""
        for r in per_symbol
    )
    pnl_color = "pos" if summary["total_pnl"] >= 0 else "neg"
    ret_color = "pos" if summary["total_return_pct"] >= 0 else "neg"

    return f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bot Report — {summary['generated_at']}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0d1117; color: #c9d1d9; font-family: 'Segoe UI', sans-serif; padding: 24px; }}
  h1 {{ color: #58a6ff; margin-bottom: 4px; }}
  .sub {{ color: #8b949e; font-size: 13px; margin-bottom: 28px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 28px; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }}
  .card .label {{ font-size: 12px; color: #8b949e; text-transform: uppercase; letter-spacing: .5px; }}
  .card .value {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
  .pos {{ color: #3fb950; }} .neg {{ color: #f85149; }}
  .chart-wrap {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                 padding: 20px; margin-bottom: 20px; }}
  .chart-wrap h2 {{ font-size: 15px; color: #58a6ff; margin-bottom: 16px; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th {{ color: #8b949e; font-size: 12px; text-transform: uppercase; letter-spacing: .5px;
        padding: 8px 12px; text-align: left; border-bottom: 1px solid #30363d; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #21262d; font-size: 14px; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  @media(max-width:700px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>Trading Bot — Equity Report</h1>
<p class="sub">Generiert: {summary['generated_at']} &nbsp;·&nbsp; Startkapital: {initial_capital:,.0f} USD</p>

<div class="grid">
  <div class="card">
    <div class="label">Gesamt-PnL</div>
    <div class="value {pnl_color}">{summary['total_pnl']:+,.2f} $</div>
  </div>
  <div class="card">
    <div class="label">Return</div>
    <div class="value {ret_color}">{summary['total_return_pct']:+.2f}%</div>
  </div>
  <div class="card">
    <div class="label">Trades</div>
    <div class="value">{summary['n_trades']}</div>
  </div>
  <div class="card">
    <div class="label">Win-Rate</div>
    <div class="value {'pos' if summary['win_rate'] >= 50 else 'neg'}">{summary['win_rate']}%</div>
  </div>
  <div class="card">
    <div class="label">Max Drawdown</div>
    <div class="value neg">{summary['max_drawdown_pct']:.2f}%</div>
  </div>
  <div class="card">
    <div class="label">Sharpe Ratio</div>
    <div class="value {'pos' if summary['sharpe'] >= 0 else 'neg'}">{summary['sharpe']:.3f}</div>
  </div>
  <div class="card">
    <div class="label">Profit Factor</div>
    <div class="value {'pos' if str(summary['profit_factor']) != '∞' and float(str(summary['profit_factor']).replace('∞','99')) >= 1 else 'neg'}">{summary['profit_factor']}</div>
  </div>
  <div class="card">
    <div class="label">Avg Win / Loss</div>
    <div class="value">{summary['avg_win']:.2f} / {summary['avg_loss']:.2f}</div>
  </div>
</div>

<div class="chart-wrap">
  <h2>📈 Equity-Kurve</h2>
  <canvas id="equityChart" height="80"></canvas>
</div>

<div class="two-col">
  <div class="chart-wrap">
    <h2>📉 Drawdown</h2>
    <canvas id="ddChart" height="120"></canvas>
  </div>
  <div class="chart-wrap">
    <h2>🎯 Rolling Win-Rate (20 Trades)</h2>
    <canvas id="wrChart" height="120"></canvas>
  </div>
</div>

<div class="two-col">
  <div class="chart-wrap">
    <h2>📊 PnL-Verteilung</h2>
    <canvas id="pnlChart" height="120"></canvas>
  </div>
  <div class="chart-wrap">
    <h2>⚡ Feature Importance (Top 15)</h2>
    <canvas id="featChart" height="120"></canvas>
  </div>
</div>

<div class="chart-wrap">
  <h2>🔍 Markteffizienz (composite score) — 0 = Edge, 1 = Random Walk</h2>
  <canvas id="effChart" height="60"></canvas>
</div>

<div class="chart-wrap">
  <h2>🗂 Performance nach Symbol</h2>
  <table>
    <thead><tr><th>Symbol</th><th>Trades</th><th>Win-Rate</th><th>Gesamt-PnL</th><th>Ø PnL</th></tr></thead>
    <tbody>{sym_rows}</tbody>
  </table>
</div>

<script>
const C = Chart.defaults;
C.color = '#c9d1d9';
C.borderColor = '#30363d';

function mkChart(id, type, labels, datasets, opts={{}}) {{
  new Chart(document.getElementById(id), {{
    type, data: {{ labels, datasets }},
    options: {{ responsive: true, plugins: {{ legend: {{ display: datasets.length > 1 }} }},
               scales: {{ x: {{ ticks: {{ maxTicksLimit: 10, maxRotation: 0 }} }},
                          y: {{}} }}, ...opts }}
  }});
}}

mkChart('equityChart', 'line', {equity_labels},
  [{{ label: 'Equity (USD)', data: {equity_data},
     borderColor: '#58a6ff', backgroundColor: 'rgba(88,166,255,.08)',
     fill: true, tension: 0.3, pointRadius: 0 }}]);

mkChart('ddChart', 'line', {equity_labels},
  [{{ label: 'Drawdown (%)', data: {dd_data},
     borderColor: '#f85149', backgroundColor: 'rgba(248,81,73,.15)',
     fill: true, tension: 0.2, pointRadius: 0 }}]);

mkChart('wrChart', 'line', {winrate_labels},
  [{{ label: 'Win-Rate (%)', data: {winrate_data},
     borderColor: '#3fb950', tension: 0.4, pointRadius: 0 }}]);

new Chart(document.getElementById('pnlChart'), {{
  type: 'bar',
  data: {{ labels: {pnl_buckets},
           datasets: [{{ label: 'Anzahl Trades', data: {pnl_counts},
             backgroundColor: ctx => {{
               const v = {pnl_buckets}[ctx.dataIndex];
               return v >= 0 ? 'rgba(63,185,80,.7)' : 'rgba(248,81,73,.7)';
             }} }}] }},
  options: {{ responsive: true, plugins: {{ legend: {{ display: false }} }} }}
}});

new Chart(document.getElementById('featChart'), {{
  type: 'bar',
  data: {{ labels: {feat_labels},
           datasets: [{{ label: 'Importance', data: {feat_values},
             backgroundColor: 'rgba(88,166,255,.7)', borderRadius: 3 }}] }},
  options: {{ indexAxis: 'y', responsive: true, plugins: {{ legend: {{ display: false }} }} }}
}});

new Chart(document.getElementById('effChart'), {{
  type: 'line',
  data: {{ datasets: {eff_datasets} }},
  options: {{
    responsive: true,
    parsing: {{ xAxisKey: 'x', yAxisKey: 'y' }},
    scales: {{ x: {{ type: 'category', ticks: {{ maxTicksLimit: 12 }} }},
               y: {{ min: 0, max: 1,
                     title: {{ display: true, text: '0=Edge · 1=Effizient' }} }} }}
  }}
}});
</script>
</body>
</html>"""
