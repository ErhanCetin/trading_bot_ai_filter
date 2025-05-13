# Binance Futures API Endpoint Açıklamaları

Bu döküman, Binance Futures API tarafından sunulan önemli endpointlerin açıklamalarını içerir. Her endpoint'in ne işe yaradığını ve ne tür veri döndürdüğünü açıklar.

Kaynak: https://binance-docs.github.io/apidocs/futures/en/

---

## 📘 MARKET DATA (Piyasa Verileri)

| Endpoint | Açıklama |
|----------|----------|
| `/fapi/v1/ping` | Sunucunun çalışır olup olmadığını test eder (boş cevap döner) |
| `/fapi/v1/time` | Binance sunucusunun sistem saatini döner |
| `/fapi/v1/exchangeInfo` | Tüm coin çiftlerinin detaylı bilgilerini verir |
| `/fapi/v1/depth` | Order book (bid/ask) verisi (Level 1-100) |
| `/fapi/v1/trades` | En son gerçekleşen işlemleri döner |
| `/fapi/v1/historicalTrades` | Geçmiş işlemleri alır (API key gerekir) |
| `/fapi/v1/aggTrades` | Toplanmış trade verileri |
| `/fapi/v1/klines` | Kline (mum grafiği) verisi |
| `/fapi/v1/continuousKlines` | Sürekli kontratlar için mum verisi |
| `/fapi/v1/indexPriceKlines` | Endeks fiyatı üzerinden mum verisi |
| `/fapi/v1/markPriceKlines` | Mark fiyatı üzerinden mum verisi |
| `/fapi/v1/premiumIndex` | Mark price, funding rate gibi premium veriler |
| `/fapi/v1/fundingRate` | Geçmiş funding rate değerleri |
| `/fapi/v1/ticker/24hr` | 24 saatlik değişim verisi |
| `/fapi/v1/ticker/price` | Sembol için son fiyat |
| `/fapi/v1/ticker/bookTicker` | Bid/ask verisi |
| `/fapi/v1/openInterest` | Açık pozisyon (open interest) değeri |
| `/futures/data/globalLongShortAccountRatio` | Long-short oranları (sentiment) |
| `/futures/data/takerlongshortRatio` | Taker buy/sell oranları |

---

## 🔐 ACCOUNT & TRADE (Hesap ve Emir)

| Endpoint | Açıklama |
|----------|----------|
| `/fapi/v1/account` | Kullanıcının tüm pozisyon ve bakiye bilgisi |
| `/fapi/v2/account` | Detaylı pozisyon bilgisi (tercih edilir) |
| `/fapi/v1/order` | Emir oluşturma (buy/sell) |
| `/fapi/v1/order/test` | Emir test etme (gerçekleşmez) |
| `/fapi/v1/order/{orderId}` | Belirli bir emrin durumunu getirir |
| `/fapi/v1/allOrders` | Tüm emir geçmişini getirir |
| `/fapi/v1/openOrders` | Açık emirleri listeler |
| `/fapi/v1/leverage` | Leverage ayarlama |
| `/fapi/v1/marginType` | Margin modunu değiştir (isolated/cross) |
| `/fapi/v1/positionMargin` | Pozisyon margin ayarı |
| `/fapi/v1/positionRisk` | Pozisyon riski ve unrealized PnL bilgisi |
| `/fapi/v1/userTrades` | Kullanıcının trade geçmişi |

---

## 🔧 ACCOUNT CONFIG

| Endpoint | Açıklama |
|----------|----------|
| `/fapi/v1/leverageBracket` | Leverage kademelerini listeler |
| `/fapi/v1/commissionRate` | Komisyon oranlarını verir |
| `/fapi/v1/income` | Gelir geçmişi (fonlama, bonus, fee vs.) |

---

## 🧪 STREAMS / WEBSOCKETS (Gerçek Zamanlı Veri)

| Endpoint | Açıklama |
|----------|----------|
| `wss://fstream.binance.com/ws` | Tek sembol için websocket (örnek: trades, ticker, depth) |
| `wss://fstream.binance.com/stream?streams=...` | Çoklu stream (multi-symbol websocket) |

---

## ⚙️ SYSTEM & LIMITS

| Endpoint | Açıklama |
|----------|----------|
| `/fapi/v1/rateLimit/order` | Emir limiti bilgisi |
| `/fapi/v1/lotSize/filter` | Lot büyüklüğü bilgisi (symbol bazlı) |

---

