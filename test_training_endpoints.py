from fastapi.testclient import TestClient
from app.main import app
import pandas as pd
import io

client = TestClient(app)

def test_ingest_training_data():
    payload = {
        "data": [
            {"time": "2023-01-01T00:00:00", "open": 1.0, "close": 1.1},
            {"time": "2023-01-01T01:00:00", "open": 1.1, "close": 1.2}
        ]
    }
    response = client.post("/api/v1/training/", json=payload)
    assert response.status_code == 200
    assert response.json() == {"status": "success", "received_records": 2}

def test_get_training_data():
    response = client.get("/api/v1/training/NAS100/H1?dataset_version=v1")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/vnd.apache.parquet"

    # Read the parquet data
    buf = io.BytesIO(response.content)
    df = pd.read_parquet(buf)

    assert len(df) == 10
    assert df["symbol"].iloc[0] == "NAS100"
    assert df["timeframe"].iloc[0] == "H1"
    assert df["version"].iloc[0] == "v1"
