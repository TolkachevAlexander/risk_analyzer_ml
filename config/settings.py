import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / '.env'

def load_environment():
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
        print(f"✅ Environment loaded from: {ENV_PATH}")
    else:
        print(f"⚠️  .env file not found at: {ENV_PATH}")
        print("ℹ️  Using system environment variables")

load_environment()

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'aml_database'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', ''),
    'ssl': os.getenv('DB_SSL', 'false').lower() == 'true'
}

def get_database_url():
    base_url = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    if DB_CONFIG['ssl']:
        base_url += "?sslmode=require"
    return base_url