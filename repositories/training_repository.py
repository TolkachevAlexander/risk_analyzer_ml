import logging
from typing import List, Optional
from decimal import Decimal

from core.data_source import DatabaseSource
from models.database_entity.training_dataset import DatasetModel
from models.dto.training_record import TrainingRecord


class DatasetRepository:
    """Репозиторий для работы с данными dataset"""

    def __init__(self, db_source: DatabaseSource):  # Теперь принимает db_source
        self.db_source = db_source
        self.logger = logging.getLogger(__name__)

    def get_all(self, skip_validation: bool = False) -> List[TrainingRecord]:
        """Получение всех записей из таблицы dataset"""
        try:
            with self.db_source.get_session() as session:
                records = session.query(DatasetModel).all()
                return self._convert_to_training_records(records, skip_validation)
        except Exception as e:
            self.logger.error(f"Error retrieving datasets: {e}")
            raise

    def get_by_inn(self, inn: str) -> Optional[TrainingRecord]:
        """Получение записи по ИНН (предполагается уникальность ИНН)"""
        try:
            with self.db_source.get_session() as session:
                record = session.query(DatasetModel)\
                    .filter(DatasetModel.inn == inn)\
                    .first()

                if record:
                    training_records = self._convert_to_training_records([record])
                    return training_records[0] if training_records else None
                return None

        except Exception as e:
            self.logger.error(f"Error retrieving dataset by INN {inn}: {e}")
            raise

    def _convert_to_training_records(self, records: List[DatasetModel], skip_validation: bool = False) -> List[TrainingRecord]:
        """Конвертация SQLAlchemy моделей в Pydantic модели"""
        training_records = []
        for record in records:
            try:
                record_dict = {}
                for key, value in record.__dict__.items():
                    if key.startswith('_'):
                        continue
                    record_dict[key] = float(value) if isinstance(value, Decimal) else value
                training_record = TrainingRecord(**record_dict)
                training_records.append(training_record)
            except Exception as e:
                self.logger.warning(f"Validation error for record {record.id}: {e}")
                if not skip_validation:
                    raise
        return training_records