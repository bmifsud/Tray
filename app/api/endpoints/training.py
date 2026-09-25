import io
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

router = APIRouter()

class TrainingDataIngest(BaseModel):
    data: List[Dict[str, Any]]

@router.post("/")
async def ingest_training_data(payload: TrainingDataIngest):
    """
    Ingest training data.
    """
    return {"status": "success", "received_records": len(payload.data)}

@router.get("/{symbol}/{timeframe}")
async def get_training_data(
    symbol: str,
    timeframe: str,
    dataset_version: Optional[str] = Query(None, description="The version of the dataset")
):
    """
    Get training data in Parquet format.
    """
    # Create some dummy data
    df = pd.DataFrame({
        "time": pd.date_range(start="2023-01-01", periods=10, freq="h"),
        "open": [1.0] * 10,
        "high": [1.1] * 10,
        "low": [0.9] * 10,
        "close": [1.05] * 10,
        "volume": [100] * 10,
        "symbol": [symbol] * 10,
        "timeframe": [timeframe] * 10,
        "version": [dataset_version if dataset_version else "latest"] * 10
    })

    table = pa.Table.from_pandas(df)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    buf.seek(0)

    def iterfile():
        yield buf.getvalue()

    return StreamingResponse(
        iterfile(),
        media_type="application/vnd.apache.parquet",
        headers={
            "Content-Disposition": f"attachment; filename=training_data_{symbol}_{timeframe}.parquet"
        }
    )
