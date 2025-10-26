import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

from config.settings import DB_CONFIG


class DatabaseService:
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self.logger = logging.getLogger(__name__)
        self._connect()

    def _connect(self):
        """Подключение к базе данных"""
        try:
            # Формируем строку подключения
            database_url = (
                f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
                f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
            )

            # Создаем engine с базовыми настройками
            self.engine = create_engine(
                database_url,
                pool_size=5,
                pool_recycle=3600,
                echo=DB_CONFIG.get('DEBUG', False)  # Логируем SQL только в DEBUG режиме
            )

            # Создаем фабрику сессий
            self.SessionLocal = sessionmaker(bind=self.engine)

            self.logger.info("✅ Database connected successfully")

        except Exception as e:
            self.logger.error(f"❌ Database connection failed: {e}")
            raise

    def test_connection(self) -> bool:
        """Тестирование подключения к базе данных"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                return result.scalar() == 1
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False

    @contextmanager
    def get_session(self):
        """Контекстный менеджер для работы с сессией БД"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()

    def close(self):
        """Закрытие соединения"""
        if self.engine:
            self.engine.dispose()
            self.logger.info("🔌 Database connection closed")