from fastapi import APIRouter, BackgroundTasks
from datetime import datetime
from pydantic import BaseModel
from typing import Optional

from app.services.ingestion import ingest_ticks

router = APIRouter()

class BackfillRequest(BaseModel):
    symbol: str = "NAS100"
    start_date: datetime
    end_date: datetime

@router.post("/api/v1/ticks/backfill")
async def backfill_ticks(request: BackfillRequest, background_tasks: BackgroundTasks):
    """
    Manually trigger tick ingestion asynchronously.
    """
    # Trigger background task for high-performance ingestion without blocking the API
    background_tasks.add_task(
        ingest_ticks,
        request.symbol,
        request.start_date,
        request.end_date
    )

    return {
        "status": "accepted",
        "message": f"Backfill job for {request.symbol} started in the background from {request.start_date} to {request.end_date}.",
        "symbol": request.symbol
    }
