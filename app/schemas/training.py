from pydantic import BaseModel
from typing import Any, Dict

class TrainingBase(BaseModel):
    dataset_version: str
    symbol: str
    timeframe: str
    features: Dict[str, Any]
    targets: Dict[str, Any]

class TrainingResponse(TrainingBase):
    id: int

    class Config:
        from_attributes = True
