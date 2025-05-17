
#df = add_indicators(df)
# df = generate_signals(df)


# # ⬇️ SHORT sinyal diagnostiği --> gecici silebilirsin.


# df["short_check_1"] = df["ema_fast"] < df["ema_slow"]
# df["short_check_2"] = df["rsi"] < 50
# df["short_check_3"] = df["macd"] < 0
# df["short_check_4"] = ~df["supertrend"]
# df["short_check_5"] = df["adx"] > 20
# df["short_check_6"] = df["di_neg"] > df["di_pos"]

# print("\n📊 SHORT sinyali koşulları dağılımı:")
# for i in range(1, 7):
#     count = df[f"short_check_{i}"].sum()
#     print(f"✅ short_check_{i}: {count} satırda TRUE")

# df["short_all_passed"] = (
#     df["short_check_1"] &
#     df["short_check_2"] &
#     df["short_check_3"] &
#     df["short_check_4"] &
#     df["short_check_5"] &
#     df["short_check_6"]
# )

# total_short_signals = df["short_all_passed"].sum()
# print(f"\n📉 Tüm SHORT koşullarını sağlayan satır sayısı: {total_short_signals}")
# print(df[df["short_all_passed"]].sample(5)[["open_time", "close", "rsi", "macd", "adx", "supertrend", "di_pos", "di_neg"]])

# df[df["short_all_passed"]][["short_signal", "signal_strength", "signal_passed_filter"]].value_counts()
# cols = ["open_time", "close", "rsi", "macd", "adx", "di_pos", "di_neg", "supertrend", "short_signal", "signal_strength", "signal_passed_filter"]
# print(df[df["short_all_passed"]][cols].sample(10))





# run_backtest_diagnostics()

# shorts = df[df["short_signal"] == True]
# print(f"🔍 Toplam short sinyali: {len(shorts)}")
# if not shorts.empty:
#     print(shorts[["open_time_fmt", "close", "ema_fast", "ema_slow", "rsi", "macd", "supertrend", "adx", "di_pos", "di_neg"]].tail(10))


