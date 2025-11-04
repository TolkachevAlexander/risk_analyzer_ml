from decimal import Decimal

from pydantic import BaseModel, Field
from typing import Optional

"""Pydantic модель для одной записи тренировочных данных"""
class TrainingRecord(BaseModel):
    id: Optional[int] = None
    inn: str
    score: Optional[Decimal] = Field(None, ge=0, le=1)
    field1: Optional[int] = Field(None, ge=1, le=5)
    field2: Optional[int] = Field(None, ge=1, le=5)
    field3: Optional[int] = Field(None, ge=1, le=5)
    field4: Optional[int] = Field(None, ge=1, le=5)
    field5: Optional[int] = Field(None, ge=1, le=5)
    field6: Optional[int] = Field(None, ge=1, le=5)
    field7: Optional[int] = Field(None, ge=1, le=5)
    field8: Optional[int] = Field(None, ge=1, le=5)
    field9: Optional[int] = Field(None, ge=1, le=5)
    field10: Optional[int] = Field(None, ge=1, le=5)
    field11: Optional[int] = Field(None, ge=1, le=5)
    field12: Optional[int] = Field(None, ge=1, le=5)
    field13: Optional[int] = Field(None, ge=1, le=5)
    field14: Optional[int] = Field(None, ge=1, le=5)
    field15: Optional[int] = Field(None, ge=1, le=5)
    field16: Optional[int] = Field(None, ge=1, le=5)
    field17: Optional[int] = Field(None, ge=1, le=5)
    field18: Optional[int] = Field(None, ge=1, le=5)
    field19: Optional[int] = Field(None, ge=1, le=5)
    field20: Optional[int] = Field(None, ge=1, le=5)

    class Config:
        from_attributes = True
