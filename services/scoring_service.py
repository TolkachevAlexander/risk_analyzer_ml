import logging
import os
from typing import Dict, Any
from datetime import datetime

from models.dto.scoring_record import BatchScoringResult, ScoringRecord
from repositories.scoring_repository import ScoringRepository
from services.lightgbm_service import LightGBMService


class ScoringService:
    """Сервис для выполнения скоринга"""

    def __init__(self, scoring_repository: ScoringRepository):
        self.scoring_repository = scoring_repository
        self.lightgbm_service = LightGBMService()
        self.logger = logging.getLogger(__name__)

    def load_latest_model(self) -> Dict[str, Any]:
        """
        Загрузка последней обученной модели
        """
        try:
            # Ищем последнюю модель в training_output/models
            models_dir = "training_output/models"

            if not os.path.exists(models_dir):
                return {
                    'success': False,
                    'message': 'No trained models found. Please run training first.'
                }

            # Получаем все файлы моделей и сортируем по дате изменения
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
            if not model_files:
                return {
                    'success': False,
                    'message': 'No model files found in models directory.'
                }

            # Сортируем по времени создания (новейшие первыми)
            model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
            latest_model_path = os.path.join(models_dir, model_files[0])

            # Загружаем модель
            self.lightgbm_service.load_model(latest_model_path)

            # Проверяем, что модель загружена корректно
            model_info = self.lightgbm_service.get_model_info()

            if model_info['status'] == 'model_loaded':
                return {
                    'success': True,
                    'message': 'Model loaded successfully',
                    'model_info': model_info,
                    'model_path': latest_model_path
                }
            else:
                return {
                    'success': False,
                    'message': 'Failed to load model'
                }

        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return {
                'success': False,
                'message': f'Error loading model: {str(e)}'
            }

    def score_all_records(self) -> BatchScoringResult:
        """
        Скоринг всех записей из scoring_dataset
        """
        try:
            self.logger.info("Starting batch scoring of all records")
            start_time = datetime.now()

            # Получаем все записи для скоринга
            records = self.scoring_repository.get_all()
            self.logger.info(f"Retrieved {len(records)} records for scoring")

            if not records:
                return BatchScoringResult(
                    results=[],
                    total_processed=0,
                    successful_predictions=0,
                    failed_predictions=0
                )

            # Выполняем пакетный скоринг
            batch_result = self.lightgbm_service.batch_score_records(records)

            # Логируем результаты
            scoring_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(
                f"Scoring completed: {batch_result.successful_predictions} successful, "
                f"{batch_result.failed_predictions} failed, "
                f"time: {scoring_time:.2f}s"
            )

            return batch_result

        except Exception as e:
            self.logger.error(f"Error during batch scoring: {e}")
            raise

    def score_single_record(self, scoring_data: ScoringRecord) -> Dict[str, Any]:
        """
        Скоринг переданной записи (не из базы данных)
        """
        try:
            # Проверяем, что модель загружена
            if not self.lightgbm_service.model:
                model_load_result = self.load_latest_model()
                if not model_load_result['success']:
                    return {
                        'status': 'error',
                        'message': f"Model not available: {model_load_result['message']}"
                    }

            # Конвертируем запрос в ScoringRecord
            record = ScoringRecord(**scoring_data.dict())

            # Выполняем скоринг
            scoring_result = self.lightgbm_service.score_record(record)

            return {
                'status': 'success',
                'inn': scoring_result.inn,
                'score': scoring_result.predicted_score,
                'message': 'Scoring completed successfully'
            }

        except Exception as e:
            self.logger.error(f"Error scoring provided record {scoring_data.inn}: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }