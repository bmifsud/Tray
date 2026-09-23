from sqlalchemy import Column, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from .base import Base

class Training(Base):
    __tablename__ = 'trainings'

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_version = Column(String, nullable=False)
    symbol = Column(String, nullable=False)
    timeframe = Column(String, nullable=False)
    features = Column(JSONB, nullable=False)
    targets = Column(JSONB, nullable=False)
