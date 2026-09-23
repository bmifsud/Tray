from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging

from app.api.ticks import router as ticks_router
from app.services.scheduler import start_scheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start the background scheduler
    logger.info("Starting up application and initializing scheduler...")
    start_scheduler()
    yield
    # Shutdown
    logger.info("Shutting down application...")

app = FastAPI(title="Tick Ingestion Engine", lifespan=lifespan)

# Include Routers
app.include_router(ticks_router)

@app.get("/")
def health_check():
    return {"status": "ok", "service": "Tick Ingestion Engine"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
