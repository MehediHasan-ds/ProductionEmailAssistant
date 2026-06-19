"""Throwaway check: generate with a reference returns all three metric groups."""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fastapi.testclient import TestClient

from app.main import create_app

with TestClient(create_app()) as client:
    resp = client.post(
        "/generate",
        json={
            "intent": "Follow up after a meeting",
            "facts": ["Met SG Group on Monday", "Shared the RACOAI deck", "Next step is a pilot"],
            "tone": "professional",
            "provider": "openrouter",
            "max_attempts": 1,
            "reference": "Hello,\n\nThank you for meeting on Monday. The deck is attached. Next we run a pilot.\n\nBest regards,",
        },
    )
    data = resp.json()
    print("status:", resp.status_code)
    print("score groups:", list(data.get("scores", {}).keys()))
    print("overall:", data.get("overall"))
