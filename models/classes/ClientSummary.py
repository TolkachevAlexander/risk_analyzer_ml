from dataclasses import dataclass

from models.classes.Сlient import Client


@dataclass
class ClientSummary:
    client: Client
    total_transactions: int
    risk_score: float