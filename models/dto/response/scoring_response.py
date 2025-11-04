from typing import List, Optional
from pydantic import BaseModel


class ScoringResult(BaseModel):
    """Модель результата скоринга (для пакетного скоринга)"""
    inn: str
    score: float

class ScoringResponse(BaseModel):
    """Модель ответа для пакетного скоринга"""
    status: str
    results: List[ScoringResult]
    total_records: int