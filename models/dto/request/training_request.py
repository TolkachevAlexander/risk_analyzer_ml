from typing import Optional, Dict, Any

from pydantic import BaseModel


class TrainingRequest(BaseModel):
    """Модель запроса для обучения с выбором конфигурации"""
    preset: Optional[str] = "high_quality"
    custom_config: Optional[Dict[str, Any]] = None