from pydantic import BaseModel, ConfigDict
from datetime import datetime

class TickCreate(BaseModel):
    symbol: str
    time: datetime
    bid: float
    ask: float
    last: float
    volume: int

    model_config = ConfigDict(from_attributes=True)
