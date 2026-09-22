import pytest
import io
import pandas as pd

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_backfill_ticks(client):
    payload = {
        "data": [
            {"time": "2023-01-01T00:00:00Z", "bid": 10000.0, "ask": 10001.0},
            {"time": "2023-01-01T00:00:01Z", "bid": 10001.0, "ask": 10002.0}
        ]
    }
    response = client.post("/ticks/backfill", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["inserted"] == 2

def test_post_ml_training(client):
    payload = {
        "features": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        "labels": [0, 1]
    }
    response = client.post("/ml/training", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["stored_samples"] == 2

def test_get_ml_training(client):
    # Ensure data is populated first by hitting the POST endpoint
    payload = {
        "features": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        "labels": [0, 1]
    }
    client.post("/ml/training", json=payload)

    response = client.get("/ml/training")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"

    parquet_bytes = response.content
    df = pd.read_parquet(io.BytesIO(parquet_bytes))

    assert not df.empty
    assert "feature1" in df.columns
    assert "feature2" in df.columns
    assert "label" in df.columns
    assert len(df) >= 2

    # Verify the nested JSON array data parsed correctly for at least the newly inserted data
    assert df.iloc[-2]["feature1"] == 1.0
    assert df.iloc[-2]["feature2"] == 2.0
    assert df.iloc[-2]["label"] == 0
    assert df.iloc[-1]["feature1"] == 4.0
    assert df.iloc[-1]["feature2"] == 5.0
    assert df.iloc[-1]["label"] == 1
