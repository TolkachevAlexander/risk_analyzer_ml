from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from config.config_manager import ConfigManager
from core.data_source import DatabaseSource
from models.dto.request.training_request import TrainingRequest
from models.dto.response.training_response import TrainingResponse
from repositories.training_repository import DatasetRepository
from services.pipeline_service import PipelineService

router = APIRouter()

# Инициализация сервисов
db = DatabaseSource()
dataset_repository = DatasetRepository(db)
pipeline = PipelineService(dataset_repository)
config_manager = ConfigManager()


@router.get("/presets_info")
async def get_light_gbm_presets() -> Dict[str, Any]:
    """
    Получение списка доступных пресетов конфигураций
    """
    try:
        available_configs = config_manager.get_available_configs()

        presets_info = {}
        for config_type in available_configs:
            config = config_manager.get_config(config_type)
            presets_info[config_type] = config.dict()

        return {
            "available_presets": available_configs,
            "presets_details": presets_info
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting presets info: {str(e)}")


@router.post("/train", response_model=TrainingResponse)
async def training(request: TrainingRequest = TrainingRequest()):
    """
    Запуск процесса обучения на данных из training_dataset с возможностью выбора конфигурации
    """
    try:
        print(f"🚀 Starting training process with config: {request.preset}")

        # Проверяем, что запрошенный тип конфигурации существует
        if not config_manager.config_exists(request.preset):
            available_configs = list(config_manager.config_map.keys())
            raise HTTPException(
                status_code=400,
                detail=f"Unknown config type: {request.preset}. "
                       f"Available configs: {', '.join(available_configs)}"
            )

        # Получаем конфигурацию
        if request.custom_config:
            training_config = config_manager.get_config(request.preset, request.custom_config)
            print("📋 Using custom configuration")
        else:
            training_config = config_manager.get_config(request.preset)
            print(f"📋 Using {request.preset} configuration")

        # Запускаем полный пайплайн обучения
        pipeline_results = pipeline.run_full_pipeline(config=training_config)

        if pipeline_results['pipeline_status'] == 'success':
            # Получаем метрики и информацию о модели
            metrics = pipeline_results['training_metrics']
            model_info = pipeline_results.get('model_info', {})

            # Формируем структурированные метрики
            training_metrics = {
                "train_rmse": round(metrics.get('train_rmse', 0), 4),
                "validation_rmse": round(metrics.get('val_rmse', 0), 4),
                "test_rmse": round(metrics.get('test_rmse', 0), 4),
                "train_mse": round(metrics.get('train_mse', 0), 4),
                "validation_mse": round(metrics.get('val_mse', 0), 4),
                "test_mse": round(metrics.get('test_mse', 0), 4)
            }

            # Получаем имя модели и топ-фичи
            model_name = model_info.get('model_name')
            top_features = model_info.get('top_features', [])

            print(f"✅ Training completed. Model: {model_name}")
            if top_features:
                print(f"📊 Top features: {[f['feature'] for f in top_features]}")

            return TrainingResponse(
                pipeline_status=True,
                training_metrics=training_metrics,
                file_name=model_name,
                top_features=top_features
            )
        else:
            error_msg = pipeline_results.get('error', 'В ходе обучения была получена неизвестная ошибка')
            print(f"❌ Pipeline failed: {error_msg}")
            return TrainingResponse(
                pipeline_status=False,
                error=error_msg
            )

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))