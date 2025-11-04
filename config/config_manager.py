from typing import Dict, Any, Optional, List
from config.training_config import BusinessConfigs, DefaultConfigs, LightGbmPresets


class ConfigManager:
    """Менеджер для управления конфигурациями обучения"""

    def __init__(self):
        self.config_map = {
            "credit_scoring": BusinessConfigs.get_credit_scoring,
            "risk_classification": BusinessConfigs.get_risk_classification,
            "default_regression": BusinessConfigs.get_default_regression,
            "fast_training": DefaultConfigs.get_fast_training,
            "high_quality": DefaultConfigs.get_high_quality,
            "conservative": DefaultConfigs.get_conservative
        }

    def get_available_configs(self) -> List[str]:
        """Получение списка доступных конфигураций"""
        return list(self.config_map.keys())

    def config_exists(self, config_type: str) -> bool:
        """Проверяет, что запрошенный тип конфигурации существует"""
        return config_type in self.config_map

    def get_config(self, config_type: str, custom_config: Optional[Dict[str, Any]] = None) -> LightGbmPresets:
        """Получает конфигурацию"""
        if not self.config_exists(config_type):
            available_configs = self.get_available_configs()
            raise ValueError(
                f"Unknown config type: {config_type}. "
                f"Available configs: {', '.join(available_configs)}"
            )

        # Получаем базовую конфигурацию
        base_config = self.config_map[config_type]()

        # Если есть кастомные параметры, объединяем их
        if custom_config:
            updated_config = {**base_config.dict(), **custom_config}
            return LightGbmPresets(**updated_config)

        return base_config