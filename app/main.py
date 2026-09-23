from fastapi import FastAPI
from app.api.endpoints import training

app = FastAPI()

app.include_router(training.router, prefix="/api/v1/training")
