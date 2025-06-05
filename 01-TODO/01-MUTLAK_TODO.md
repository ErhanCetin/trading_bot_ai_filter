NOT : asagidakini hesaplayan kod zaten varmis :
class MarketRegimeDetector:
    """Detect bull/bear/sideways markets for strategy adaptation"""

Marketin durumu belirleyip ona gore configurasyonlari aktif etmek gerekiyor. Asagida ornekler var . Yani sistemi parametrik olarak istedigimiz modda calistirmamiz gerekiyor. 
AMA bunun icin market durumunu saptayan bir kod yazmam gerekiyor. On chain ve temel analizler isin icine giriyor . ama yapabilirsem fiyat datasiyla bulmaya calisacagim. 

BEAR_SHORT_BALANCED ← Recommended (257 trades, +37% ROI)
BEAR_SHORT_STRICT ← Fewer trades, higher quality
BEAR_SHORT_RELAXED ← More trades, lower quality

Bull Market:

BULL_LONG_AGGRESSIVE ← High risk/reward
BULL_LONG_CONSERVATIVE ← Safe bull strategy

Sideways Market:

SIDEWAYS_BOTH_BREAKOUT ← Quick breakout trades

Scalping:

SCALP_SHORT_FAST ← 5m fast trades


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------


