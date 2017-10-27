from pipeline_processor import Base
from sqlalchemy import Column, Boolean
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy_utils import UUIDType


class Measurement(Base):
    __tablename__ = 'measurement'

    id = Column(UUIDType, primary_key=True)
    data = Column(JSONB)
    processed = Column(Boolean)


