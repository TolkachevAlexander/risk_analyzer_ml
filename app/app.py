from fastapi import FastAPI
from controllers.training_controller import router as training_router
from controllers.scoring_controller import router as scoring_router


def create_app() -> FastAPI:
    """
    Фабрика для создания приложения FastAPI
    """
    app = FastAPI(
        title="ML Scoring API",
        description="API для обучения и использования моделей скоринга",
        version="1.0.0"
    )

    app.include_router(training_router, prefix="/ml-scoring-core/training", tags=["training"])
    app.include_router(scoring_router, prefix="/ml-scoring-core/scoring", tags=["scoring"])

    # Корневой эндпоинт для проверки работы
    @app.get("/")
    async def root():
        return {"message": "ML Scoring API is running"}

    return app