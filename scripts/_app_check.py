"""Throwaway check: system endpoints plus a one attempt generate call."""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fastapi.testclient import TestClient

from app.main import create_app

app = create_app()
with TestClient(app) as client:
    print("/health:", client.get("/health").json())
    resp = client.post(
        "/generate",
        json={
            "intent": "Follow up after a product demo",
            "facts": ["Demo with the team on Monday", "Sending pricing next"],
            "tone": "professional",
            "max_attempts": 1,
        },
    )
    print("generate status:", resp.status_code)
    data = resp.json()
    print("subject:", data.get("subject"))
    print("overall:", data.get("overall"), "attempts:", data.get("attempts"))

    ev = client.post(
        "/evaluate",
        json={
            "subject": "Follow up on Monday demo",
            "body": "Hello,\n\nThank you for the demo on Monday. I will send pricing next.\n\nBest regards,",
            "intent": "Follow up after a product demo",
            "facts": ["Demo with the team on Monday", "Sending pricing next"],
            "tone": "professional",
        },
    )
    print("evaluate status:", ev.status_code)
    print("evaluate overall:", ev.json().get("overall"))
