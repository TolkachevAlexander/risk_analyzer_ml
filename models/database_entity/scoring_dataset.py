from sqlalchemy import Column, BigInteger, String, Integer
from core.data_source import Base


class ScoringDatasetModel(Base):
    """SQLAlchemy модель для таблицы scoring_dataset"""
    __tablename__ = 'scoring_dataset'
    __table_args__ = {'schema': 'ml_scoring_core'}

    id = Column(BigInteger, primary_key=True)
    inn = Column(String(12), nullable=False)
    field1 = Column(Integer)
    field2 = Column(Integer)
    field3 = Column(Integer)
    field4 = Column(Integer)
    field5 = Column(Integer)
    field6 = Column(Integer)
    field7 = Column(Integer)
    field8 = Column(Integer)
    field9 = Column(Integer)
    field10 = Column(Integer)
    field11 = Column(Integer)
    field12 = Column(Integer)
    field13 = Column(Integer)
    field14 = Column(Integer)
    field15 = Column(Integer)
    field16 = Column(Integer)
    field17 = Column(Integer)
    field18 = Column(Integer)
    field19 = Column(Integer)
    field20 = Column(Integer)