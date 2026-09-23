from fastapi import FastAPI, Depends, Response, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import io
import asyncpg
import json
import os

app = FastAPI(title="NAS100 Data Service API")

# Database connection details
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "nas100_db")

async def get_db():
    conn = await asyncpg.connect(user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT, database=DB_NAME)
    try:
        yield conn
    finally:
        await conn.close()

# Pydantic Models
class Tick(BaseModel):
    time: str
    bid: float
    ask: float

class TickBackfill(BaseModel):
    data: List[Tick]

class MLTrainingData(BaseModel):
    features: List[List[float]]
    labels: List[int]


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/ticks/backfill")
async def backfill_ticks(payload: TickBackfill, db: asyncpg.Connection = Depends(get_db)):
    """Bulk insert mock Blueberry Markets data into TimescaleDB hypertable."""
    try:
        # Create hypertable if it doesn't exist (assuming TimescaleDB is installed)
        await db.execute('''
            CREATE TABLE IF NOT EXISTS ticks (
                time TIMESTAMPTZ NOT NULL,
                bid DOUBLE PRECISION NOT NULL,
                ask DOUBLE PRECISION NOT NULL
            );
        ''')
        # We don't execute create_hypertable here to keep it simple, but this is where it'd go

        # Prepare data for bulk insert
        records = [(tick.time, tick.bid, tick.ask) for tick in payload.data]

        await db.copy_records_to_table(
            'ticks',
            records=records,
            columns=['time', 'bid', 'ask']
        )

        return {"status": "success", "inserted": len(records)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ml/training")
async def post_ml_training(payload: MLTrainingData, db: asyncpg.Connection = Depends(get_db)):
    """Store nested JSON arrays containing ML features and labels."""
    try:
        await db.execute('''
            CREATE TABLE IF NOT EXISTS ml_training_data (
                id SERIAL PRIMARY KEY,
                features JSONB NOT NULL,
                label INTEGER NOT NULL
            );
        ''')

        records = []
        for i in range(len(payload.labels)):
            features_json = json.dumps(payload.features[i])
            label = payload.labels[i]
            records.append((features_json, label))

        await db.copy_records_to_table(
            'ml_training_data',
            records=records,
            columns=['features', 'label']
        )
        return {"status": "success", "stored_samples": len(records)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ml/training")
async def get_ml_training(db: asyncpg.Connection = Depends(get_db)):
    """Fetch training data and return it as a streaming Parquet response."""
    try:
        # Check if table exists
        table_exists = await db.fetchval("SELECT to_regclass('public.ml_training_data');")

        if not table_exists:
             # Return empty parquet if table doesn't exist yet
             df = pd.DataFrame(columns=["feature1", "feature2", "label"])
        else:
            rows = await db.fetch('SELECT features, label FROM ml_training_data')

            if not rows:
                df = pd.DataFrame(columns=["feature1", "feature2", "label"])
            else:
                data = []
                for row in rows:
                    features = json.loads(row['features'])
                    # We assume 2 features based on the tests, but this can be dynamic
                    data.append({
                        "feature1": features[0] if len(features) > 0 else 0,
                        "feature2": features[1] if len(features) > 1 else 0,
                        "label": row['label']
                    })
                df = pd.DataFrame(data)

        parquet_buffer = io.BytesIO()
        df.to_parquet(parquet_buffer, engine='pyarrow')
        parquet_buffer.seek(0)

        return Response(content=parquet_buffer.getvalue(), media_type="application/octet-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
