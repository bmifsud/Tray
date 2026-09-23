from fastapi.testclient import TestClient
from datetime import datetime, timezone
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "Tick Ingestion Engine"}

def test_backfill_ticks():
    response = client.post(
        "/api/v1/ticks/backfill",
        json={
            "symbol": "NAS100",
            "start_date": datetime.now(timezone.utc).isoformat(),
            "end_date": datetime.now(timezone.utc).isoformat()
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "accepted"
    assert data["symbol"] == "NAS100"
