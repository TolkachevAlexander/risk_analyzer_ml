from typing import Optional, List, Dict, Any

from pydantic import BaseModel

class TrainingResponse(BaseModel):
    pipeline_status: bool
    training_metrics: Optional[Dict[str, float]] = None
    file_name: Optional[str] = None
    top_features: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
