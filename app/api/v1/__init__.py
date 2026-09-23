from fastapi import APIRouter
from app.api.v1.ticks import router as ticks_router

api_router = APIRouter()
api_router.include_router(ticks_router, prefix="/ticks", tags=["ticks"])
