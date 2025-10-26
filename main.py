from sqlalchemy import text

from services.DatabaseService import DatabaseService


def main():
    db = DatabaseService()

    # Тестируем подключение
    if db.test_connection():
        print("✅ Database connection: SUCCESS")
    else:
        print("❌ Database connection: FAILED")
        return

    # Используем сессию для работы с БД
    with db.get_session() as session:
        # Выполняем запросы
        result = session.execute(text("SELECT version()"))
        print(f"Database version: {result.scalar()}")

    # Закрываем соединение
    db.close()


if __name__ == "__main__":
    main()