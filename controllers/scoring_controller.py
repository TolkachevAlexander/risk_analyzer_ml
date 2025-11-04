from fastapi import APIRouter, HTTPException
import logging

from models.dto.response.scoring_response import ScoringResponse, ScoringResult
from models.dto.scoring_record import ScoringRecord
from services.scoring_service import ScoringService
from repositories.scoring_repository import ScoringRepository
from core.data_source import DatabaseSource

router = APIRouter()
logger = logging.getLogger(__name__)

# Инициализация сервисов
db = DatabaseSource()
scoring_repository = ScoringRepository(db)
scoring_service = ScoringService(scoring_repository)


@router.get("/score-dataset", response_model=ScoringResponse)
async def scoring():
    """
    Запуск скоринга на данных из scoring_dataset
    """
    try:
        logger.info("🎯 Starting scoring process...")

        # Загружаем последнюю модель
        model_load_result = scoring_service.load_latest_model()
        if not model_load_result['success']:
            logger.error(f"Model loading failed: {model_load_result['message']}")
            raise HTTPException(
                status_code=400,
                detail=f"Model not available: {model_load_result['message']}"
            )

        logger.info(f"✅ Model loaded successfully")

        # Выполняем скоринг всех записей
        scoring_results = scoring_service.score_all_records()

        # Преобразуем в нужный формат (без confidence)
        results = [
            ScoringResult(inn=result.inn, score=result.predicted_score)
            for result in scoring_results.results
        ]

        logger.info(f"📊 Scoring completed: {len(results)} records processed")

        return ScoringResponse(
            status="success",
            results=results,
            total_records=len(results)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Scoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/score-record", response_model=ScoringResult)
async def score_single_record(request: ScoringRecord):
    """
    Скоринг одной записи, переданной в теле запроса
    """
    try:
        logger.info(f"🎯 Starting single record scoring for INN: {request.inn}")

        # Выполняем скоринг переданной записи
        scoring_result = scoring_service.score_single_record(request)

        if scoring_result['status'] == 'error':
            logger.error(f"Scoring failed: {scoring_result['message']}")
            raise HTTPException(
                status_code=400,
                detail=scoring_result['message']
            )

        logger.info(f"✅ Single record scoring completed for INN: {request.inn}")

        return ScoringResult(
            inn=scoring_result['inn'],
            score=scoring_result['score'],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Single record scoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))