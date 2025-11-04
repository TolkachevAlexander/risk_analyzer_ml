import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

from repositories.training_repository import DatasetRepository
from services.lightgbm_service import LightGBMService, LightGbmPresets


class PipelineService:
    """Сервис для управления полным ML пайплайном"""

    def __init__(self, dataset_repository: DatasetRepository):
        self.dataset_repository = dataset_repository
        self.lightgbm_service = LightGBMService()
        self.logger = logging.getLogger(__name__)

    def analyze_data_quality(self) -> Dict[str, Any]:
        """
        Анализ качества данных перед обучением
        """
        try:
            records = self.dataset_repository.get_all()

            if not records:
                return {'status': 'error', 'message': 'No data available'}

            # Конвертируем в DataFrame для анализа
            data = []
            for record in records:
                record_dict = record.dict()
                data.append(record_dict)

            df = pd.DataFrame(data)

            # Анализ столбца score
            score_stats = {
                'total_count': len(df),
                'non_null_count': df['score'].notna().sum(),
                'null_count': df['score'].isna().sum(),
                'unique_count': df['score'].nunique(),
                'dtype': str(df['score'].dtype)
            }

            # Пробуем преобразовать score в числовой тип
            score_numeric = pd.to_numeric(df['score'], errors='coerce')
            score_stats['numeric_count'] = score_numeric.notna().sum()
            score_stats['non_numeric_count'] = score_numeric.isna().sum()

            # Анализ диапазона значений
            if score_stats['numeric_count'] > 0:
                valid_scores = score_numeric[(score_numeric >= 0) & (score_numeric <= 1)]
                score_stats['valid_scores_count'] = len(valid_scores)
                score_stats['invalid_scores_count'] = score_stats['numeric_count'] - len(valid_scores)

                if len(valid_scores) > 0:
                    score_stats['min_score'] = float(valid_scores.min())
                    score_stats['max_score'] = float(valid_scores.max())
                    score_stats['mean_score'] = float(valid_scores.mean())

            return {
                'status': 'success',
                'score_analysis': score_stats,
                'sample_scores': df['score'].head(10).tolist()  # Примеры первых 10 значений
            }

        except Exception as e:
            self.logger.error(f"Error analyzing data quality: {e}")
            return {'status': 'error', 'message': str(e)}

    def run_full_pipeline(
            self,
            config: Optional[LightGbmPresets] = None,
            model_name: str = None
    ) -> Dict[str, Any]:
        """
        Запуск полного пайплайна с улучшенной обработкой ошибок
        """
        try:
            self.logger.info("Starting full ML pipeline")

            # Генерируем имя модели с timestamp, если не предоставлено
            if model_name is None:
                model_name = f"risk_scoring_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # 1. Анализ качества данных
            self.logger.info("Step 1: Analyzing data quality")
            data_quality = self.analyze_data_quality()

            if data_quality['status'] == 'error':
                return {
                    'pipeline_status': 'failed',
                    'error': f"Data quality check failed: {data_quality['message']}",
                    'data_quality': data_quality
                }

            # 2. Получение данных
            self.logger.info("Step 2: Loading data from repository")
            records = self.dataset_repository.get_all()

            if not records:
                return {
                    'pipeline_status': 'failed',
                    'error': "No data available for training",
                    'data_quality': data_quality
                }

            self.logger.info(f"Loaded {len(records)} records for training")

            # 3. Обучение модели
            self.logger.info("Step 3: Training LightGBM model")
            training_results = self.lightgbm_service.train_model(records, config)

            # 4. Сохранение модели с передачей результатов обучения
            self.logger.info("Step 4: Saving trained model")
            save_results = self.lightgbm_service.save_model(
                model_name=model_name,
                training_results=training_results
            )

            # 5. Формирование результатов
            results = {
                'pipeline_status': 'success',
                'data_quality': data_quality,
                'training_records_count': len(records),
                'training_metrics': training_results['metrics'],
                'feature_importance': training_results['feature_importance'],
                'training_info': training_results.get('training_info', {}),
                'model_config': training_results['config'],
                # Информация о модели из save_results
                'model_info': {
                    'model_name': save_results['model_name'],
                    'top_features': save_results.get('top_features', []),
                    'model_path': save_results['model_path']
                }
            }

            self.logger.info("ML pipeline completed successfully")
            self.logger.info(f"Model saved as: {save_results['model_name']}")

            if save_results.get('top_features'):
                self.logger.info(f"Top features: {[f['feature'] for f in save_results['top_features']]}")

            return results

        except Exception as e:
            self.logger.error(f"ML pipeline failed: {e}")
            return {
                'pipeline_status': 'failed',
                'error': str(e),
                'data_quality': self.analyze_data_quality()
            }

    def predict_for_inn(self, inn: str) -> Dict[str, Any]:
        try:
            record = self.dataset_repository.get_by_inn(inn)

            if not record:
                return {
                    'status': 'error',
                    'message': f'No record found for INN: {inn}'
                }

            prediction = self.lightgbm_service.predict_single(record)

            return {
                'status': 'success',
                'inn': inn,
                'prediction': float(prediction),
                'actual_score': record.score,
                'features_used': self.lightgbm_service.feature_columns
            }

        except Exception as e:
            self.logger.error(f"Prediction failed for INN {inn}: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def batch_predict(self, inns: List[str]) -> Dict[str, Any]:
        try:
            results = []

            for inn in inns:
                prediction_result = self.predict_for_inn(inn)
                results.append(prediction_result)

            successful_predictions = [r for r in results if r['status'] == 'success']
            failed_predictions = [r for r in results if r['status'] == 'error']

            return {
                'status': 'completed',
                'total_requests': len(inns),
                'successful_predictions': len(successful_predictions),
                'failed_predictions': len(failed_predictions),
                'results': results
            }

        except Exception as e:
            self.logger.error(f"Batch prediction failed: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def get_model_performance(self) -> Dict[str, Any]:
        model_info = self.lightgbm_service.get_model_info()

        return {
            'model_info': model_info,
            'pipeline_status': 'active' if model_info['status'] == 'model_loaded' else 'inactive'
        }