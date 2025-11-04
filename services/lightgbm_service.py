import json
import logging
import pickle

import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import List, Tuple, Optional, Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import os
from datetime import datetime

from config.training_config import LightGbmPresets, ModelType
from models.dto.scoring_record import ScoringRecord, ScoringResult, BatchScoringResult
from models.dto.training_record import TrainingRecord


class LightGBMService:
    """Сервис для обучения и использования моделей LightGBM"""

    def __init__(self, models_base_dir: str = "training_output"):
        self.logger = logging.getLogger(__name__)
        self.models_base_dir = models_base_dir
        self.model = None
        self.feature_columns = None

        # Создаем только необходимые папки
        self._create_directory_structure()

    def _create_directory_structure(self):
        """Создает только необходимые папки"""
        base_dirs = ["models", "metadata"]
        for dir_name in base_dirs:
            os.makedirs(os.path.join(self.models_base_dir, dir_name), exist_ok=True)

    def prepare_features(self, records: List[TrainingRecord]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Подготовка признаков и целевой переменной из записей
        """
        try:
            self.logger.info("Preparing features from training records")

            # Конвертируем в DataFrame
            data = []
            for record in records:
                record_dict = record.dict()
                data.append(record_dict)

            df = pd.DataFrame(data)

            # Определяем фичи (field1-field20)
            self.feature_columns = [f"field{i}" for i in range(1, 21)]

            # Проверяем наличие всех фичей
            missing_features = [col for col in self.feature_columns if col not in df.columns]
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")

            # Заполняем пропущенные значения в фичах
            X = df[self.feature_columns].fillna(-1)

            # Обрабатываем целевую переменную score
            if 'score' not in df.columns:
                raise ValueError("Target variable 'score' not found in data")

            # Преобразуем score в числовой тип и обрабатываем ошибки
            y = self._convert_score_to_numeric(df['score'])

            self.logger.info(f"Prepared features: {X.shape}, target: {y.shape}")
            self.logger.info(f"Target variable stats - Min: {y.min():.3f}, Max: {y.max():.3f}, Mean: {y.mean():.3f}")

            return X, y

        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            raise

    def _convert_score_to_numeric(self, score_series: pd.Series) -> pd.Series:
        """
        Преобразует столбец score в числовой тип, обрабатывая ошибки
        """
        try:
            # Пробуем преобразовать в float
            y_converted = pd.to_numeric(score_series, errors='coerce')

            # Проверяем, сколько значений не преобразовалось
            nan_count = y_converted.isna().sum()
            if nan_count > 0:
                self.logger.warning(
                    f"Could not convert {nan_count} score values to numeric. They will be filled with -1")

                # Заполняем пропуски значением -1
                y_converted = y_converted.fillna(-1)

            # Проверяем диапазон значений (должны быть от 0 до 1)
            valid_scores = y_converted[(y_converted >= 0) & (y_converted <= 1)]
            if len(valid_scores) == 0:
                self.logger.warning("No valid scores in range [0, 1] found. Check your data.")

            self.logger.info(
                f"Score conversion: {len(valid_scores)} valid scores, {len(y_converted) - len(valid_scores)} invalid scores")

            return y_converted

        except Exception as e:
            self.logger.error(f"Error converting scores to numeric: {e}")
            # В случае ошибки возвращаем серию с -1
            return pd.Series([-1] * len(score_series))

    def train_model(
            self,
            records: List[TrainingRecord],
            config: Optional[LightGbmPresets] = None
    ) -> Dict[str, Any]:
        """
        Обучение модели LightGBM
        """
        try:
            if config is None:
                config = LightGbmPresets()

            self.logger.info(f"Starting model training with {len(records)} records")

            # Подготовка данных
            X, y = self.prepare_features(records)

            # Проверяем, есть ли достаточно валидных данных
            valid_indices = (y >= 0) & (y <= 1)
            valid_count = valid_indices.sum()

            if valid_count < 10:  # Минимум 10 валидных записей
                raise ValueError(f"Not enough valid scores for training. Only {valid_count} valid records found.")

            # Используем только валидные данные
            X_valid = X[valid_indices]
            y_valid = y[valid_indices]

            self.logger.info(f"Using {len(X_valid)} valid records out of {len(X)} total records")

            # Разделение на train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X_valid, y_valid,
                test_size=config.test_size,
                random_state=config.random_state
            )

            # Разделение train на train/validation для ранней остановки
            X_train_final, X_val, y_train_final, y_val = train_test_split(
                X_train, y_train,
                test_size=config.validation_size,
                random_state=config.random_state
            )

            self.logger.info(f"Data split - Train: {X_train_final.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

            # Параметры LightGBM
            lgb_params = {
                'objective': self._get_objective(config.model_type),
                'learning_rate': config.learning_rate,
                'max_depth': config.max_depth,
                'num_leaves': config.num_leaves,
                'min_child_samples': config.min_child_samples,
                'subsample': config.subsample,
                'colsample_bytree': config.colsample_bytree,
                'random_state': config.random_state,
                'verbose': -1,
            }

            # Создаем datasets для LightGBM
            train_data = lgb.Dataset(X_train_final, label=y_train_final)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            # Обучение модели
            self.logger.info("Training LightGBM model...")

            self.model = lgb.train(
                lgb_params,
                train_data,
                num_boost_round=config.n_estimators,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(config.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(period=config.verbose) if config.verbose > 0 else lgb.log_evaluation(0),
                ]
            )

            # Предсказания и оценка
            train_predictions = self.model.predict(X_train_final)
            val_predictions = self.model.predict(X_val)
            test_predictions = self.model.predict(X_test)

            # Метрики
            metrics = self._calculate_metrics(
                y_train_final, train_predictions,
                y_val, val_predictions,
                y_test, test_predictions,
                config.model_type
            )

            # Важность признаков
            feature_importance = self._get_feature_importance(X.columns)

            results = {
                'model': self.model,
                'metrics': metrics,
                'feature_importance': feature_importance,
                'config': config.dict(),
                'training_date': datetime.now().isoformat(),
                'feature_columns': self.feature_columns,
                'training_info': {
                    'total_records': len(records),
                    'valid_records': len(X_valid),
                    'invalid_records': len(records) - len(X_valid)
                }
            }

            self.logger.info("Model training completed successfully")
            return results

        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            raise

    def _get_objective(self, model_type: ModelType) -> str:
        """Получение objective функции для LightGBM"""
        objectives = {
            ModelType.REGRESSION: 'regression',
            ModelType.BINARY_CLASSIFICATION: 'binary',
            ModelType.MULTICLASS_CLASSIFICATION: 'multiclass'
        }
        return objectives[model_type]

    def _calculate_metrics(
            self,
            y_train, train_pred,
            y_val, val_pred,
            y_test, test_pred,
            model_type: ModelType
    ) -> Dict[str, float]:
        """Расчет метрик качества модели"""
        metrics = {}

        if model_type == ModelType.REGRESSION:
            # MSE для регрессии
            metrics['train_mse'] = mean_squared_error(y_train, train_pred)
            metrics['val_mse'] = mean_squared_error(y_val, val_pred)
            metrics['test_mse'] = mean_squared_error(y_test, test_pred)

            # RMSE
            metrics['train_rmse'] = np.sqrt(metrics['train_mse'])
            metrics['val_rmse'] = np.sqrt(metrics['val_mse'])
            metrics['test_rmse'] = np.sqrt(metrics['test_mse'])

        else:
            # Для классификации используем accuracy
            train_pred_class = np.round(train_pred)
            val_pred_class = np.round(val_pred)
            test_pred_class = np.round(test_pred)

            metrics['train_accuracy'] = accuracy_score(y_train, train_pred_class)
            metrics['val_accuracy'] = accuracy_score(y_val, val_pred_class)
            metrics['test_accuracy'] = accuracy_score(y_test, test_pred_class)

        return metrics

    def _get_feature_importance(self, feature_names) -> Dict[str, float]:
        """Получение важности признаков"""
        if self.model is None:
            return {}

        importance = self.model.feature_importance(importance_type='gain')
        feature_importance = dict(zip(feature_names, importance))

        # Сортируем по убыванию важности
        return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    # Остальные методы остаются без изменений...
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not trained. Call train_model first.")

        missing_features = [col for col in self.feature_columns if col not in features.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        return self.model.predict(features[self.feature_columns])

    def predict_single(self, record: TrainingRecord) -> float:
        record_dict = record.dict()
        features = {f"field{i}": getattr(record, f"field{i}") for i in range(1, 21)}

        df = pd.DataFrame([features])
        return self.predict(df)[0]

    def save_model(self, model_name: str = None, training_results: dict = None):
        """
        Сохраняет модель и результаты обучения

        Args:
            model_name: Имя модели (если None, генерируется автоматически)
            training_results: Словарь с результатами обучения

        Returns:
            dict: Информация о сохраненной модели и топ-5 признаков
        """
        if model_name is None:
            model_name = f"credit_scoring_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Основные директории
        models_dir = os.path.join(self.models_base_dir, "models")
        metadata_dir = os.path.join(self.models_base_dir, "metadata")

        # Создаем папки
        for directory in [models_dir, metadata_dir]:
            os.makedirs(directory, exist_ok=True)

        # Получаем важность признаков
        top_features = self._get_top_features(top_n=5)

        # Сохраняем модель
        model_path = os.path.join(models_dir, f"{model_name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_columns': self.feature_columns,
                'timestamp': datetime.now().isoformat(),
                'top_features': top_features
            }, f)

        # Объединяем результаты обучения с топ-признаками
        if training_results is None:
            training_results = {}

        training_results['top_features'] = top_features

        # Сохраняем метаданные
        metadata_path = os.path.join(metadata_dir, f"{model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(training_results, f, indent=4, default=str)

        self.logger.info(f"Model saved to {model_path}")

        # Формируем ответ с метриками и топ-признаками
        result = {
            'model_path': model_path,
            'metadata_path': metadata_path,
            'model_name': model_name,
            'training_metrics': {
                'accuracy': training_results.get('accuracy'),
                'precision': training_results.get('precision'),
                'recall': training_results.get('recall'),
                'f1_score': training_results.get('f1_score'),
                'auc_roc': training_results.get('auc_roc')
            }
        }

        # Добавляем топ-признаки, если они есть
        if top_features:
            result['top_features'] = top_features
            self.logger.info(f"Top 5 features: {[f['feature'] for f in top_features]}")

        return result

    def _get_top_features(self, top_n: int = 5):
        """
        Получает топ-N наиболее значимых признаков
        """
        if self.model is None:
            self.logger.warning("Model is not trained, cannot get feature importance")
            return []

        try:
            # Получаем важность признаков из LightGBM
            feature_importance = self.model.feature_importance(importance_type='gain')

            # Правильное получение имен признаков в LightGBM
            feature_names = self.model.feature_name()  # Без подчеркивания в конце!

            # Создаем пары (признак, важность) и сортируем
            feature_importance_pairs = list(zip(feature_names, feature_importance))
            feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)

            # Берем топ-N
            top_features = []
            total_importance = sum(feature_importance)

            for i, (feature_name, importance) in enumerate(feature_importance_pairs[:top_n]):
                top_features.append({
                    'rank': i + 1,
                    'feature': feature_name,
                    'importance': float(importance),
                    'importance_percentage': float(importance) / total_importance * 100 if total_importance > 0 else 0
                })

            return top_features

        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            # В случае ошибки возвращаем пустой список, чтобы не прерывать процесс
            return []

    def load_model(self, model_path: str):
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Загружаем через pickle, а не joblib
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.feature_columns = model_data['feature_columns']

            self.logger.info(f"Model loaded from {model_path}")
            self.logger.info(f"Feature columns: {self.feature_columns}")

        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        if self.model is None:
            return {'status': 'no_model_loaded'}

        return {
            'status': 'model_loaded',
            'feature_columns': self.feature_columns,
            'num_features': len(self.feature_columns) if self.feature_columns else 0,
            'model_type': type(self.model).__name__
        }

    def score_record(self, record: ScoringRecord) -> ScoringResult:
        """
        Скоринг одной записи
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Call load_model() first.")

        try:
            # Подготавливаем фичи для предсказания
            features = self._prepare_scoring_features(record)

            # Выполняем предсказание
            prediction = self.predict(features)
            predicted_score = float(prediction[0])

            return ScoringResult(
                inn=record.inn,
                predicted_score=predicted_score,
                # Убрали confidence
                features_used=self.feature_columns
            )

        except Exception as e:
            self.logger.error(f"Error scoring record with INN {record.inn}: {e}")
            raise

    def batch_score_records(self, records: List[ScoringRecord]) -> BatchScoringResult:
        """
        Пакетный скоринг нескольких записей
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Call load_model() first.")

        results = []
        successful = 0
        failed = 0

        for record in records:
            try:
                result = self.score_record(record)
                results.append(result)
                successful += 1
            except Exception as e:
                self.logger.error(f"Failed to score record with INN {record.inn}: {e}")
                failed += 1

        return BatchScoringResult(
            results=results,
            total_processed=len(records),
            successful_predictions=successful,
            failed_predictions=failed
        )

    def _prepare_scoring_features(self, record: ScoringRecord) -> pd.DataFrame:
        """
        Подготовка признаков для скоринга
        """
        # Создаем словарь с фичами
        features_dict = {f"field{i}": getattr(record, f"field{i}") for i in range(1, 21)}

        # Создаем DataFrame
        features_df = pd.DataFrame([features_dict])

        # Заполняем пропущенные значения (как при обучении)
        features_df = features_df.fillna(-1)

        return features_df

    def _calculate_confidence(self, features: pd.DataFrame) -> Optional[float]:
        """
        Расчет уверенности предсказания (опционально)
        Можно реализовать на основе дисперсии или других метрик
        """
        try:
            # Простой пример: если все фичи заполнены, уверенность высокая
            missing_features = features.isna().sum().sum()
            total_features = features.shape[1]
            confidence = 1.0 - (missing_features / total_features)
            return round(confidence, 3)
        except:
            return None

    def get_scoring_capabilities(self) -> Dict[str, Any]:
        """
        Информация о возможностях скоринга
        """
        if self.model is None:
            return {
                'status': 'model_not_loaded',
                'message': 'Load a model first to perform scoring'
            }

        return {
            'status': 'ready',
            'model_loaded': True,
            'feature_columns': self.feature_columns,
            'num_features': len(self.feature_columns) if self.feature_columns else 0,
            'model_type': 'LightGBM',
            'can_score': True
        }