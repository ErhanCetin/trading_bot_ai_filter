from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env_loader import load_environment

load_environment()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/trading_bot")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

