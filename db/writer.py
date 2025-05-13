from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import text
from db.postgresql import engine
from db.models import (
    kline_data_table,
    funding_rate_table,
    open_interest_table,
    long_short_ratio_table
)

def insert_kline(df):
    if df.empty:
        return

    records = df.to_dict(orient="records")
    with engine.begin() as conn:
        for record in records:
            stmt = insert(kline_data_table).values(**record)
            stmt = stmt.on_conflict_do_nothing(
                index_elements=["symbol", "interval", "open_time"]
            )
            conn.execute(stmt)

def insert_funding_rate(df):
    if df.empty:
        return

    records = df.to_dict(orient="records")
    with engine.begin() as conn:
        for row in records:
            stmt = insert(funding_rate_table).values(
                symbol=row["symbol"],
                time=row["fundingTime"],
                funding_rate=row["fundingRate"]
            )
            stmt = stmt.on_conflict_do_nothing(
                index_elements=["symbol", "time"]
            )
            conn.execute(stmt)

def insert_open_interest(df):
    if df.empty:
        return

    records = df.to_dict(orient="records")
    with engine.begin() as conn:
        for row in records:
            stmt = insert(open_interest_table).values(
                symbol=row["symbol"],
                time=row["timestamp"],
                open_interest=row["openInterest"]
            )
            stmt = stmt.on_conflict_do_nothing(
                index_elements=["symbol", "time"]
            )
            conn.execute(stmt)

def insert_long_short_ratio(df):
    if df.empty:
        return

    records = df.to_dict(orient="records")
    with engine.begin() as conn:
        for row in records:
            stmt = insert(long_short_ratio_table).values(
                symbol=row["symbol"],
                time=row["timestamp"],
                long_account=row["longAccount"],
                short_account=row["shortAccount"],
                ratio=row["ratio"]
            )
            stmt = stmt.on_conflict_do_nothing(
                index_elements=["symbol", "time"]
            )
            conn.execute(stmt)
