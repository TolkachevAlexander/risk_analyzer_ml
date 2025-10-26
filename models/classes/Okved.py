from dataclasses import dataclass
from typing import Optional


@dataclass
class OKVED:
    code: str
    description: str
    is_risky: bool = False
    main_category: Optional[str] = None