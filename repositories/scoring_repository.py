import logging
from typing import List, Optional
from sqlalchemy.exc import SQLAlchemyError

from core.data_source import DatabaseSource
from models.database_entity.scoring_dataset import ScoringDatasetModel
from models.dto.scoring_record import ScoringRecord


class ScoringRepository:
    """Репозиторий для работы с таблицей scoring_dataset"""

    def __init__(self, db_source: DatabaseSource):
        self.db_source = db_source
        self.logger = logging.getLogger(__name__)

    def get_all(self) -> List[ScoringRecord]:
        """Получение всех записей из scoring_dataset"""
        try:
            self.logger.info("Retrieving all records from scoring_dataset")

            with self.db_source.get_session() as session:
                records = session.query(ScoringDatasetModel).all()
                self.logger.info(f"Retrieved {len(records)} records from scoring_dataset")

                return self._convert_to_scoring_records(records)

        except SQLAlchemyError as e:
            self.logger.error(f"Database error while retrieving scoring datasets: {e}")
            raise Exception(f"Database operation failed: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error while retrieving scoring datasets: {e}")
            raise

    def get_by_inn(self, inn: str) -> Optional[ScoringRecord]:
        """Получение записи по ИНН из scoring_dataset"""
        try:
            self.logger.info(f"Searching for scoring record with INN: {inn}")

            with self.db_source.get_session() as session:
                record = session.query(ScoringDatasetModel) \
                    .filter(ScoringDatasetModel.inn == inn) \
                    .first()

                if record:
                    records = self._convert_to_scoring_records([record])
                    return records[0] if records else None
                return None

        except Exception as e:
            self.logger.error(f"Error retrieving scoring record by INN {inn}: {e}")
            raise

    def _convert_to_scoring_records(self, records: List[ScoringDatasetModel]) -> List[ScoringRecord]:
        """Конвертация SQLAlchemy моделей в Pydantic модели"""
        scoring_records = []

        for record in records:
            try:
                # Создаем словарь с данными
                record_dict = {}
                for key, value in record.__dict__.items():
                    if key.startswith('_'):
                        continue
                    record_dict[key] = value

                scoring_record = ScoringRecord(**record_dict)
                scoring_records.append(scoring_record)

            except Exception as e:
                self.logger.warning(f"Error converting scoring record {record.id}: {e}")
                # Продолжаем обработку остальных записей

        return scoring_records