import time

def interval_to_milliseconds(interval: str) -> int:
    unit = interval[-1]
    value = int(interval[:-1])
    if unit == "m":
        return value * 60 * 1000
    elif unit == "h":
        return value * 60 * 60 * 1000
    elif unit == "d":
        return value * 24 * 60 * 60 * 1000
    else:
        raise ValueError(f"Desteklenmeyen interval: {interval}")

# Check if the data is fresh based on the last close time and the interval
# tolerance_multiples: kaç katı kadar gecikmeye izin verileceği
# interval: zaman dilimi (örneğin, "1m", "5m", "1h", "1d")
# last_close_time: son kapanış zaman damgası (milisaniye cinsinden)
# interval_to_milliseconds: zaman dilimini milisaniyeye dönüştürmek için kullanılan yardımcı fonksiyon
# is_data_fresh: verinin güncel olup olmadığını kontrol eden fonksiyon
# eski dataya göre işlem yapmamak için kullanılır
def is_data_fresh(last_close_time: int, interval: str, tolerance_multiples=2) -> bool:
    interval_ms = interval_to_milliseconds(interval)
    max_delay = interval_ms * tolerance_multiples
    now = int(time.time() * 1000)
    return (now - last_close_time) <= max_delay
