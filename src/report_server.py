"""Tiny confirmation server for TRAI report SMS handoff."""

from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import logging
import os
from urllib.parse import parse_qs, urlparse

from src.trai_report import TRAI_SMS_SHORT_CODE, build_sms_uri

logger = logging.getLogger("report-server")
logger.setLevel(logging.INFO)


class ReportRequestHandler(BaseHTTPRequestHandler):
    server_version = "SpamDetectionReportServer/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_text(200, "ok")
            return

        if parsed.path != "/report":
            self._send_text(404, "not found")
            return

        params = parse_qs(parsed.query)
        destination = _first(params.get("to")) or TRAI_SMS_SHORT_CODE
        body = _first(params.get("body")) or ""
        if not body.strip():
            self._send_text(400, "missing complaint body")
            return

        sms_uri = build_sms_uri(body, destination=destination)
        html = _build_confirmation_html(destination, body, sms_uri)
        encoded = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args) -> None:
        logger.info("%s - %s", self.address_string(), format % args)

    def _send_text(self, status: int, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def run_server() -> None:
    host = os.environ.get("TRAI_REPORT_SERVER_HOST", "127.0.0.1")
    port = int(os.environ.get("TRAI_REPORT_SERVER_PORT", "8787"))
    server = ThreadingHTTPServer((host, port), ReportRequestHandler)
    logger.info("TRAI report confirmation server listening on http://%s:%d", host, port)
    server.serve_forever()


def _build_confirmation_html(destination: str, body: str, sms_uri: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Confirm TRAI Report</title>
  <style>
    body {{
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      color: #18212f;
      background: #f6f7f9;
    }}
    main {{
      max-width: 640px;
      margin: 0 auto;
      padding: 32px 20px;
    }}
    pre {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: #fff;
      border: 1px solid #d8dde6;
      border-radius: 8px;
      padding: 16px;
    }}
    a {{
      display: inline-flex;
      align-items: center;
      min-height: 44px;
      padding: 0 18px;
      border-radius: 8px;
      background: #165dff;
      color: #fff;
      text-decoration: none;
      font-weight: 600;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Confirm TRAI Report</h1>
    <p>Review the complaint draft. Your SMS app will open next so you can send it yourself.</p>
    <p><strong>To:</strong> {escape(destination)}</p>
    <pre>{escape(body)}</pre>
    <a href="{escape(sms_uri)}">Open SMS App</a>
  </main>
</body>
</html>"""


def _first(values: list[str] | None) -> str | None:
    if not values:
        return None
    return values[0]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_server()
