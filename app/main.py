from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.database import AsyncSessionLocal

app = FastAPI(title="NAS100 MicroStructure LSTM API")

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

@app.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    health_status = {"api": "ok", "database": "unknown"}

    try:
        # Check database connection
        result = await db.execute(text("SELECT 1"))
        if result.scalar() == 1:
            health_status["database"] = "ok"
    except Exception as e:
        health_status["database"] = f"error: {str(e)}"
        raise HTTPException(status_code=503, detail=health_status)

    return health_status
