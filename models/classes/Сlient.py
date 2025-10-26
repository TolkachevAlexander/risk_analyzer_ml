from dataclasses import dataclass
from typing import Optional
from datetime import datetime

from models.classes.Okved import OKVED
from models.enums.ManagementRiskLevel import ManagementRiskLevel
from models.enums.RiskLevel import RiskLevel


@dataclass
class Client:
    client_id: str
    registration_date: datetime
    last_eio_change_date: Optional[datetime]
    region: str
    okveds: list[OKVED]
    zsk_risk_level: RiskLevel
    has_okved_diversity: bool
    management_risk: ManagementRiskLevel