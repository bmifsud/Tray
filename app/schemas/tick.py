from pydantic import BaseModel

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
