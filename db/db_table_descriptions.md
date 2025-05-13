# Veritabanı Tablo Açıklamaları

Bu döküman, `trading_bot_ai_filter` projesinde kullanılan veritabanı tablolarının alanlarını ve işlevlerini açıklar.

---

## 📊 1. `kline_data` — Mum Grafiği (Candlestick) Verisi

Bu tablo, belirli bir coin çifti (`symbol`) ve zaman aralığı (`interval`) için geçmiş fiyat hareketlerini saklar.

| Alan Adı                   | Açıklama |
|----------------------------|----------|
| `id`                       | Otomatik artan benzersiz satır ID'si (primary key) |
| `symbol`                  | Coin çifti (örnek: `BTCUSDT`) |
| `interval`               | Zaman aralığı (`1m`, `15m`, `1h`, `1d` gibi) |
| `open_time`             | Mumun başladığı zaman (`timestamp` - milisaniye) |
| `open`                  | Mum açılış fiyatı |
| `high`                  | Mum süresince ulaşılan en yüksek fiyat |
| `low`                   | Mum süresince ulaşılan en düşük fiyat |
| `close`                 | Mum kapanış fiyatı |
| `volume`                | Bu mum süresince gerçekleşen işlem hacmi |
| `close_time`           | Mumun kapanış zamanı |
| `quote_asset_volume`   | Quote cinsinden hacim (örneğin USDT bazlı) |
| `number_of_trades`     | Bu mum süresince yapılan işlem sayısı |
| `taker_buy_base_volume`| Taker alım miktarı (base asset cinsinden) |
| `taker_buy_quote_volume`| Taker alım miktarı (quote asset cinsinden) |

> Teknik analizde kullanılır (RSI, MACD, EMA, vb.).

---

## 💰 2. `funding_rate` — Fonlama Oranı

Binance Futures’ta long-short dengesini korumak için kullanılan fonlama ücretlerinin geçmişini saklar.

| Alan Adı       | Açıklama |
|----------------|----------|
| `id`           | Otomatik ID |
| `symbol`       | Coin çifti (örnek: `BTCUSDT`) |
| `time`         | Fonlama zaman damgası (timestamp) |
| `funding_rate` | Bu zaman aralığındaki funding oranı (örnek: 0.0001) |

> `funding_rate > 0` → Long baskın, `< 0` → Short baskın

---

## 📈 3. `open_interest` — Açık Pozisyon Miktarı

Piyasadaki açık long + short sözleşmelerin toplam sayısını gösterir.

| Alan Adı       | Açıklama |
|----------------|----------|
| `id`           | Otomatik ID |
| `symbol`       | Coin çifti |
| `time`         | Ölçüm zaman damgası |
| `open_interest`| Açık sözleşme miktarı |

> Açık pozisyon büyüyorsa yeni para giriyor olabilir.

---

## ⚖️ 4. `long_short_ratio` — Long vs Short Kullanıcı Oranı

Trader’ların long ve short oranlarını gösterir.

| Alan Adı       | Açıklama |
|----------------|----------|
| `id`           | Otomatik ID |
| `symbol`       | Coin çifti |
| `time`         | Zaman damgası |
| `long_account` | Long pozisyon alan kullanıcı oranı |
| `short_account`| Short pozisyon alan kullanıcı oranı |
| `ratio`        | Long / Short oranı (örnek: 1.3 = long baskın) |

> Sentiment analizinde kullanılır. Aşırı long varsa short fırsatı olabilir.

---
