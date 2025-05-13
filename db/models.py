from sqlalchemy import Table, Column, String, BigInteger, Numeric, Integer, Boolean, MetaData

metadata = MetaData()

kline_data_table = Table(
    "kline_data", metadata,
    Column("id", Integer, primary_key=True),
    Column("symbol", String(20)),
    Column("interval", String(10)),
    Column("open_time", BigInteger),
    Column("open", Numeric),
    Column("high", Numeric),
    Column("low", Numeric),
    Column("close", Numeric),
    Column("volume", Numeric),
    Column("close_time", BigInteger),
    Column("quote_asset_volume", Numeric),
    Column("number_of_trades", Integer),
    Column("taker_buy_base_volume", Numeric),
    Column("taker_buy_quote_volume", Numeric),

    # Optional indicator columns
    Column("rsi", Numeric, nullable=True),
    Column("adx", Numeric, nullable=True),
    Column("super_trend", Numeric, nullable=True),
    Column("cci", Numeric, nullable=True),
    Column("macd", Numeric, nullable=True),
    Column("macd_signal", Numeric, nullable=True),
    Column("stoch_rsi", Numeric, nullable=True),
    Column("volume_sma", Numeric, nullable=True),
    Column("volatility_band_width", Numeric, nullable=True),
    Column("momentum", Numeric, nullable=True),
    Column("trend_direction", Boolean, nullable=True),
)

funding_rate_table = Table(
    "funding_rate", metadata,
    Column("id", Integer, primary_key=True),
    Column("symbol", String(20)),
    Column("time", BigInteger),
    Column("funding_rate", Numeric)
)

open_interest_table = Table(
    "open_interest", metadata,
    Column("id", Integer, primary_key=True),
    Column("symbol", String(20)),
    Column("time", BigInteger),
    Column("open_interest", Numeric)
)

long_short_ratio_table = Table(
    "long_short_ratio", metadata,
    Column("id", Integer, primary_key=True),
    Column("symbol", String(20)),
    Column("time", BigInteger),
    Column("long_account", Numeric),
    Column("short_account", Numeric),
    Column("ratio", Numeric)
)