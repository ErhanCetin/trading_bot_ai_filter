from sqlalchemy import text
from db.postgresql import engine

def insert_kline(df):
    df.to_sql('kline_data', engine, if_exists='append', index=False)

def insert_funding_rate(df):
    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(text("""
                INSERT INTO funding_rate (symbol, time, funding_rate)
                VALUES (:symbol, :time, :rate)
                ON CONFLICT(symbol, time) DO NOTHING
            """), {
                "symbol": row["symbol"],
                "time": row["fundingTime"],
                "rate": row["fundingRate"]
            })

def insert_open_interest(df):
    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(text("""
                INSERT INTO open_interest (symbol, time, open_interest)
                VALUES (:symbol, :time, :oi)
                ON CONFLICT(symbol, time) DO NOTHING
            """), {
                "symbol": row["symbol"],
                "time": row["timestamp"],
                "oi": row["openInterest"]
            })

def insert_long_short_ratio(df):
    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(text("""
                INSERT INTO long_short_ratio (symbol, time, long_account, short_account, ratio)
                VALUES (:symbol, :time, :long_account, :short_account, :ratio)
                ON CONFLICT(symbol, time) DO NOTHING
            """), {
                "symbol": row["symbol"],
                "time": row["timestamp"],
                "long_account": row["longAccount"],
                "short_account": row["shortAccount"],
                "ratio": row["ratio"]
            })
