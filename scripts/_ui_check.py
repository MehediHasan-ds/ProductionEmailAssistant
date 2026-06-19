"""Throwaway check: root serves the updated UI with tone chips."""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fastapi.testclient import TestClient

from app.main import create_app

app = create_app()
with TestClient(app) as client:
    html = client.get("/").text
    print("tone-chips container:", "tone-chips" in html)
    print("chip count:", html.count('class="chip"'))
    print("datalist gone:", 'list="tones"' not in html)
