from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.db.session import get_db
from app.models.tick import Tick
from app.schemas.tick import TickResponse

router = APIRouter()

@router.get("/{symbol}", response_model=List[TickResponse])
def get_ticks_by_symbol(symbol: str, limit: int = 100, db: Session = Depends(get_db)):
    ticks = db.query(Tick).filter(Tick.symbol == symbol).order_by(Tick.timestamp.desc()).limit(limit).all()
    if not ticks:
        raise HTTPException(status_code=404, detail="No ticks found for the given symbol")
    return ticks
