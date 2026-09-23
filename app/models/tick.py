from sqlalchemy import Column, Integer, String, BigInteger, Float, Index
from .base import Base

class Tick(Base):
    __tablename__ = 'ticks'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False)
    timestamp = Column(BigInteger, nullable=False) # timestamp in milliseconds
    bid = Column(Float, nullable=False)
    ask = Column(Float, nullable=False)
    tick_volume = Column(BigInteger, nullable=False)

    __table_args__ = (
        Index('ix_ticks_symbol_timestamp', 'symbol', 'timestamp'),
    )
