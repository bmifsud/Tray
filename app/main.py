from fastapi import FastAPI
from app.api.v1 import api_router

app = FastAPI(title="NAS100 Ticks API")

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": "Welcome to NAS100 Ticks API"}
