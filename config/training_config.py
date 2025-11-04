from pydantic import BaseModel
from enum import Enum


class ModelType(str, Enum):
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary"
    MULTICLASS_CLASSIFICATION = "multiclass"


class LightGbmPresets(BaseModel):
    """Конфигурация для обучения модели"""
    model_type: ModelType = ModelType.REGRESSION
    test_size: float = 0.2
    random_state: int = 42
    validation_size: float = 0.1

    # Параметры LightGBM
    learning_rate: float = 0.1
    n_estimators: int = 100
    max_depth: int = -1  # -1 означает нет ограничения
    num_leaves: int = 31
    min_child_samples: int = 20
    subsample: float = 1.0
    colsample_bytree: float = 1.0

    # Ранняя остановка
    early_stopping_rounds: int = 10
    verbose: int = 1

    class Config:
        use_enum_values = True


# Предопределенные конфигурации для разных сценариев
class DefaultConfigs:
    """Предопределенные конфигурации обучения"""

    @staticmethod
    def get_fast_training() -> LightGbmPresets:
        """Быстрое обучение для тестирования"""
        return LightGbmPresets(
            n_estimators=50,
            learning_rate=0.1,
            early_stopping_rounds=5,
            verbose=0
        )

    @staticmethod
    def get_high_quality() -> LightGbmPresets:
        """Высококачественное обучение для продакшена"""
        return LightGbmPresets(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=63,
            min_child_samples=15,
            early_stopping_rounds=50,
            verbose=1
        )

    @staticmethod
    def get_conservative() -> LightGbmPresets:
        """Консервативная конфигурация против переобучения"""
        return LightGbmPresets(
            n_estimators=200,
            learning_rate=0.01,
            max_depth=6,
            num_leaves=31,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=20,
            verbose=1
        )


# Конфигурации для конкретных бизнес-кейсов
class BusinessConfigs:
    """Бизнес-ориентированные конфигурации"""

    @staticmethod
    def get_credit_scoring() -> LightGbmPresets:
        """Для задачи кредитного скоринга"""
        return LightGbmPresets(
            model_type=ModelType.REGRESSION,
            n_estimators=500,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=45,
            min_child_samples=25,
            early_stopping_rounds=25,
            verbose=1
        )

    @staticmethod
    def get_risk_classification() -> LightGbmPresets:
        """Для классификации рисков"""
        return LightGbmPresets(
            model_type=ModelType.BINARY_CLASSIFICATION,
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            num_leaves=32,
            early_stopping_rounds=15,
            verbose=1
        )

    @staticmethod
    def get_default_regression() -> LightGbmPresets:
        """Конфигурация по умолчанию для регрессии"""
        return LightGbmPresets(
            model_type=ModelType.REGRESSION,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=-1,
            num_leaves=31,
            min_child_samples=20,
            early_stopping_rounds=10,
            verbose=1
        )


# Дефолтная конфигурация для быстрого доступа
DEFAULT_CONFIG = DefaultConfigs.get_high_quality()