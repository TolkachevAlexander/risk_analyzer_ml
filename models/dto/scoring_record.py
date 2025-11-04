from pydantic import BaseModel
from typing import Optional, List

class ScoringRecord(BaseModel):
    """Модель для записи данных для скоринга (без score)"""
    inn: str
    field1: Optional[int] = None
    field2: Optional[int] = None
    field3: Optional[int] = None
    field4: Optional[int] = None
    field5: Optional[int] = None
    field6: Optional[int] = None
    field7: Optional[int] = None
    field8: Optional[int] = None
    field9: Optional[int] = None
    field10: Optional[int] = None
    field11: Optional[int] = None
    field12: Optional[int] = None
    field13: Optional[int] = None
    field14: Optional[int] = None
    field15: Optional[int] = None
    field16: Optional[int] = None
    field17: Optional[int] = None
    field18: Optional[int] = None
    field19: Optional[int] = None
    field20: Optional[int] = None

    class Config:
        from_attributes = True

class ScoringResult(BaseModel):
    """Результат скоринга (используется в BatchScoringResult)"""
    inn: str
    predicted_score: float
    # Убрали confidence

class BatchScoringResult(BaseModel):
    """Результат пакетного скоринга"""
    results: List[ScoringResult]
    total_processed: int
    successful_predictions: int
    failed_predictions: int