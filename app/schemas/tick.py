from pydantic import BaseModel
from datetime import datetime

class TickCreate(BaseModel):
    symbol: str
    time: datetime
    bid: float
    ask: float
    last: float
    volume: int

class TickBase(BaseModel):
    symbol: str
    timestamp: int
    bid: float
    ask: float
    tick_volume: int

class TickResponse(TickBase):
    id: int

    class Config:
        from_attributes = True
