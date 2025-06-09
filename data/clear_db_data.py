
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env_loader import load_environment

load_environment()



DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("❌ DATABASE_URL tanımlı değil!")

engine = create_engine(DATABASE_URL)

tables = [
    "kline_data",
    "funding_rate",
    "open_interest",
    "long_short_ratio"
]

def clear_tables():
    with engine.begin() as conn:
        for table in tables:
            print(f"🧹 {table} verisi siliniyor...")
            conn.execute(text(f"DELETE FROM {table}"))
        print("✅ Tüm tabloların verileri başarıyla silindi.")

if __name__ == "__main__":
    clear_tables()

