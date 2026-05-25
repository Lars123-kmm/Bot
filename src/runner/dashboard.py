from __future__ import annotations

"""
Minimal live dashboard — pure Python stdlib, no extra dependencies.

Runs an HTTP server in a daemon thread.  The LiveRunner passes a
`state_provider` callable that returns the current state dict; the dashboard
page polls /api/state every few seconds and renders it.
"""

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)

_HTML = """<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<title>Trading-Bot Dashboard</title>
<style>
  body { font-family: -apple-system, Segoe UI, sans-serif; background:#0d1117;
         color:#c9d1d9; margin:0; padding:24px; }
  h1 { font-size:20px; margin:0 0 4px; }
  .sub { color:#8b949e; font-size:13px; margin-bottom:20px; }
  .cards { display:flex; flex-wrap:wrap; gap:14px; margin-bottom:24px; }
  .card { background:#161b22; border:1px solid #30363d; border-radius:8px;
          padding:14px 18px; min-width:150px; }
  .card .label { color:#8b949e; font-size:12px; text-transform:uppercase; }
  .card .value { font-size:22px; font-weight:600; margin-top:4px; }
  .pos { color:#3fb950; } .neg { color:#f85149; }
  table { border-collapse:collapse; width:100%; margin-bottom:24px; }
  th,td { text-align:left; padding:8px 12px; border-bottom:1px solid #30363d;
          font-size:13px; }
  th { color:#8b949e; text-transform:uppercase; font-size:11px; }
  .tag { padding:2px 8px; border-radius:10px; font-size:11px; font-weight:600; }
  .long { background:#1f6f3f; color:#fff; } .short { background:#8a2b27; color:#fff; }
  h2 { font-size:15px; margin:18px 0 8px; color:#c9d1d9; }
</style>
</head>
<body>
<h1>ML-Trading-Bot Dashboard</h1>
<div class="sub" id="meta">lädt…</div>
<div class="cards" id="cards"></div>
<h2>Offene Positionen</h2>
<table id="positions"><thead><tr>
  <th>Symbol</th><th>Richtung</th><th>Entry</th><th>SL</th><th>TP</th><th>PnL</th>
</tr></thead><tbody></tbody></table>
<h2>Letzte Trades</h2>
<table id="trades"><thead><tr>
  <th>Zeit</th><th>Symbol</th><th>Richtung</th><th>Grund</th><th>PnL</th>
</tr></thead><tbody></tbody></table>
<script>
function cls(v){ return v>=0 ? 'pos' : 'neg'; }
function fmt(v){ return (v>=0?'+':'') + Number(v).toLocaleString('de-DE',
  {minimumFractionDigits:2, maximumFractionDigits:2}); }
async function refresh(){
  try{
    const s = await (await fetch('/api/state')).json();
    document.getElementById('meta').textContent =
      'Modus: ' + s.mode + '  ·  Symbole: ' + (s.symbols_list||[]).join(', ') +
      '  ·  Bar #' + s.total_bars;
    const a = s.account || {};
    document.getElementById('cards').innerHTML = [
      ['Equity', fmt(a.equity||0)+' USD', ''],
      ['Balance', fmt(a.balance||0)+' USD', ''],
      ['Offener PnL', fmt(a.profit||0)+' USD', cls(a.profit||0)],
      ['Trades', (s.n_trades||0), ''],
      ['Win-Rate', ((s.win_rate||0)*100).toFixed(1)+' %', ''],
      ['Drawdown', ((s.drawdown||0)*100).toFixed(1)+' %', cls(-(s.drawdown||0))],
    ].map(c => '<div class="card"><div class="label">'+c[0]+
      '</div><div class="value '+c[2]+'">'+c[1]+'</div></div>').join('');
    const pb = document.querySelector('#positions tbody');
    pb.innerHTML = (s.positions||[]).map(p =>
      '<tr><td>'+p.symbol+'</td><td><span class="tag '+
      (p.direction==1?'long':'short')+'">'+(p.direction==1?'LONG':'SHORT')+
      '</span></td><td>'+p.entry_price+'</td><td>'+p.stop+'</td><td>'+
      p.take_profit+'</td><td class="'+cls(p.profit)+'">'+fmt(p.profit)+
      '</td></tr>').join('') || '<tr><td colspan=6>keine</td></tr>';
    const tb = document.querySelector('#trades tbody');
    tb.innerHTML = (s.recent_trades||[]).slice().reverse().map(t =>
      '<tr><td>'+(t.close_time||'')+'</td><td>'+t.symbol+'</td><td><span class="tag '+
      (t.direction==1?'long':'short')+'">'+(t.direction==1?'LONG':'SHORT')+
      '</span></td><td>'+(t.close_reason||'')+'</td><td class="'+cls(t.pnl)+'">'+
      fmt(t.pnl)+'</td></tr>').join('') || '<tr><td colspan=5>keine</td></tr>';
  }catch(e){ document.getElementById('meta').textContent = 'Verbindung verloren…'; }
}
refresh(); setInterval(refresh, 5000);
</script>
</body>
</html>"""


class DashboardServer:
    """Serves the dashboard HTML + a /api/state JSON endpoint in a daemon thread."""

    def __init__(self, state_provider: Callable[[], Dict[str, Any]], port: int = 8080):
        self._state_provider = state_provider
        self.port = port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        provider = self._state_provider

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence per-request logging
                pass

            def do_GET(self):  # noqa: N802
                if self.path.rstrip("/") == "/api/state":
                    try:
                        body = json.dumps(provider(), default=str).encode("utf-8")
                    except Exception as exc:  # noqa: BLE001
                        body = json.dumps({"error": str(exc)}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    body = _HTML.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

        try:
            self._server = ThreadingHTTPServer(("0.0.0.0", self.port), _Handler)
        except OSError as exc:
            logger.warning("Dashboard could not bind port %d: %s", self.port, exc)
            return

        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True, name="dashboard"
        )
        self._thread.start()
        logger.info("Dashboard: http://localhost:%d", self.port)

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            logger.info("Dashboard stopped.")
