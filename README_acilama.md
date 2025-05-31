TODO : 

Kritik Bulgular ve Öneriler
🔴 Yüksek Öncelikli Sorunlar:

SL/TP Hesaplama: Mevcut ATR multiplier kullanımı çok dar seviyelere neden olabilir
Position Sizing: Risk per trade SL mesafesi gözetilmeden hesaplanıyor
Win Rate Tutarsızlığı: Farklı yerlerde farklı hesaplama mantığı
Trade Outcome Logic: TP/SL kontrolünde öncelik sırası problemi

🟡 Orta Öncelikli İyileştirmeler:

Data Validation: Veri kalitesi kontrolü eksik
Slippage & Commission: Gerçekçi maliyet hesaplama eksik
Risk Management: Consecutive loss protection yok
Performance Metrics: Sharpe, Calmar, Sortino eksik

🟢 Düşük Öncelikli Eklemeler:

Trade Duration Analysis: İşlem süre analizi
Market Regime Awareness: Piyasa rejimi farkındalığı
Monte Carlo Validation: Robust testing
Walk-Forward Analysis: Zamansal validasyon

💡 Uygulama Önerisi
backtest_engine.py dosyasında aşağıdaki metodları değiştir:


1) veri ceken docker ile veriyi isleyip service ureten farkli olmali : 

    1. Ayrı servis olarak çalıştır (önerilen)
    fetch_scheduler.py kendi başına sürekli çalışır (örneğin bir cronjob, Docker container, service olarak)

    runner_scheduler.py sadece veriyi kontrol eder ve sinyal üretir

    İkisi birbirini tetiklemez — sadece aynı veritabanını kullanırlar

    ❗ Bu yaklaşımda runner içinden fetch çağırmana gerek yoktur!


2) __init__.py dosyarini gozden gecir. ilgili klasordeki python larda kullanilan import vey from lari oraya yaz.

3) data eski kalinca monitor alertlari yapma
  Monitoring sistemi kurar
fetch_scheduler.py düzenli çalışıyor mu?

Veritabanında son güncelleme zamanı ne?

Eğer 1 dakikadan eskiyse alarm gönderilir → Slack, Telegram, Prometheus, vs.

Trade lock mekanizması
Eğer veri gecikmişse, işlem açma işlevi (örneğin open_position()) geçici olarak devre dışı bırakılır

4) nedir ? :Execution Lag veya Signal Slippage
   konumuz :
      simdi en son kapanan muma gore sinyal uretecegiz. Hesaplama yaparken devam eden mum hareketleri var ve bizim urettigimiz sinyal rakamsal olarak bir ise yaramayabilri..Biz hesaplama yaparken mum coktan %10 dusmus veya yukselmis olabilir. 
         Profesyonel Sistemler Ne Yapar?
            1. Live fiyat ile sinyal fiyatı arasındaki farkı kontrol ederler
               Eğer fiyat ±1%’den fazla sapmışsa → işlem açılmaz
            2. Trade lock + revalidation sistemi kurarlar
               Her sinyal, pozisyon açılmadan önce yeniden fiyat geçerliliği kontrolüne girer.Koşullar uygunsa devam eder, değilse atlanır
             3. Gerçek zamanlı fiyatla işlem açarlar (WebSocket)
                Sinyal 0.4050'ten geldiyse ama markette anlık 0.3990’a düştüyse:
                Ya sinyali iptal ederler .Ya yeni bir SL/TP ile hesaplamayı baştan yaparlar     
    Burada daha sonra Monitoring / trade lock kismini entegre edecegiz.

5) nedir ? Kendi sermayenle başlayabilir, sonra fon alabilir, hatta white-label servis bile sunabilirsin

 6) funding fee nin hesaplamalara etkisi ... chatgpt te TODO listesine alindi.

Senin dediğin gibi gerçek dünyaya uygun sistem kurmak için aşağıdakiler yapılmalı:

        ✅ Stratejik Kullanım Önerileri
        Strateji	Açıklama
        1. Funding trend analizi	Son 3-4 funding değerine bak → sürekli pozitif mi? Negatif mi?
        2. Anlık funding prediction	WebSocket ile alınır → pozisyon açarken son tahmini bil
        3. Pozisyon açma filtresi	Örneğin: Eğer model LONG sinyali verdiyse ama funding > 0.03 ise → pozisyon açma
        4. Risk düzeltme katsayısı	Funding Rate * Leverage hesaplanır ve risk puanına dahil edilir
        5. TP/SL ayarlama	Eğer pozisyon açıldıysa ve funding zararlıysa → TP’yi daha yakın, SL’yi daha uzak yapabilirsin

        🧩 Teknik Uygulama
        Parça	Gerekli mi?
        ✅ WebSocket üzerinden anlık funding tahmini	✔
        ✅ funding_rate geçmişinden trend çıkarımı	✔
        ✅ Sinyal üretim aşamasına funding entegrasyonu	✔
        ✅ Risk/PnL hesaplamasına funding etkisini dahil etme	✔
        ✅ Gerekirse sinyal filtreleyici AI modeline feature olarak verme	✔✔✔

        📌 Senin örneğinle açıklayayım:
        Funding Rate = -2%
        Model SHORT önerdi
        Pozisyon açıldı, 10x kaldıraç → kazancın üstüne funding'den +2%/pozisyon ödül alırsın
        Ama tam tersi olursa pozisyon kazansa bile zarar edebilir

        Yani funding PnL’yi doğrudan etkiler → kesinlikle hesaba katılmalı.

        🔧 Önerim
        Sırayla şunları yapalım:

        funding_rate geçmişine dayalı trend özelliği çıkar (3 kayıt üst üste negatif/pozitif)

        add_indicators() benzeri add_market_context() gibi fonksiyonla veri zenginleştir

        Modelin “pozisyon açmalı mıyım?” kararına funding’i feature olarak ver

        (opsiyonel) WebSocket ile canlı tahmin de çekip risk değerlendirmeye koy


7) backtest sonuclarinin db ye yazilmasi
        ✅ Yazmamanın nedenleri (şu anki yaklaşımımız):
        Sebep	Açıklama
        🧪 Test kodudur	Sadece “şu an çalışıyor mu?” diye bakıyoruz
        🧹 Geçici analiz	Aynı backtest tekrarlandığında sonuç değişebilir
        📦 Veri büyümesi	Her sinyal, outcome, pozisyon vs. → DB hızla şişer
        🧠 Gereksiz kayıt	Şu an için asıl olan sinyal motorunun mantığını test etmek
        🚀 Ne zaman veritabanına yazmalıyız?
        ✅ Profesyonel, ürünleşmiş sistemlerde:
        Strateji ID’si tanımlanır
        (örneğin: "strategy_v1_macd_ema_rsi")

        Backtest parametreleri kayıt altına alınır
        (symbol, interval, tarih aralığı, sl/tp oranı, sinyal koşulları)

        Pozisyonlar, outcome’lar, sinyaller ayrı tablolara yazılır

        Performans verisi analiz tablosuna gider
        (win rate, avg RRR, PnL vs.)

------------
__init__.py dosyası, Python'da bir klasörü modül (package) haline getirmek için kullanılır.
__init__.py varsa → Python o klasörü bir “paket” olarak görür
__init__.py yoksa → Python klasörü sadece bir dizin olarak görür, import edemez ❌

Ama istersek burada:

Ortak importlar (shared utils)

Paket seviyesi init işlemleri

Versiyon, logger tanımı

vs. yazabiliriz.

-------------
Setup

local de env set etmek icin 
bash:
    >  export PROJECT_GOLDEN_KICK_ENV=local


> zip dosyasini chatgpt den indirdikten sonra vscode ta acilsin diye zip dosyasinin icine girip "code . " komutunu calistirarak tum projeyi vscode editor da actirabilirsin ama onceden vscode editor da asagidakini yap : 

View → Command Palette → yaz:
Shell Command: Install 'code' command in PATH

Bu seçenek sadece macOS’ta görünür. Windows için elle eklemen gerekebilir.


> pip install -r requirements.txt
> pip install black isort flake8



----------------------------


docker run --name my-postgres-container \
  -p 5432:5432 \
  -e POSTGRES_USER=myuser \
  -e POSTGRES_PASSWORD=mypassword \
  -e POSTGRES_DB=mydatabase \
  -v pgdata:/var/lib/postgresql/data \
  -d postgres:13


---------------



- Docker içinde çalıştırıyorsan (örneğin runner.py → Docker):
.env
DATABASE_URL=postgresql://erhan:secret@db:5432/trading_bot
- VSCode veya terminalden çalıştırıyorsan, .env dosyasını böyle yap:
.env
DATABASE_URL=postgresql://erhan:secret@localhost:5432/trading_bot


-------

# env_loader dosyasını bulabilmesi için proje kökünü ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# yada alternatif olarak env_loader.py dosyasını doğrudan içe aktar
# Proje kökünü dinamik olarak sys.path'e ekle
    #BASE_DIR = Path(__file__).resolve().parent.parent
        #sys.path.append(str(BASE_DIR))

        # .env dosyaları da bu klasörde aranacak
        #os.chdir(BASE_DIR)


-----
TODO : bostrap.py dosyasin ne icin kullanilir

-----

db gui icin 

https://www.postgresql.org/ftp/pgadmin/pgadmin4/v9.3/macos/
https://dbeaver.io/download/



------

pip install tabulate
Tabloyu sade ve hizalı yazdırmak için tabulate kullanabilirsin (opsiyonel)
from tabulate import tabulate

print(tabulate(
    df[["open_time", "close", "rsi", "ema_fast", "ema_slow", "macd", "signal_strength", "long_signal", "short_signal"]].tail(5),
    headers="keys",
    tablefmt="psql"
))


-----
setup.py de birsey degistirirsen 

Note.. setup.py deprecated olacak o yuzdn pyproject.toml kullanildi
aayarlari yapdiktan sonra dosya silmen gerekiyor
> find . -type d -name "*.egg-info" -exec rm -rf {} +
> find . -type d -name "__pycache__" -exec rm -rf {} +
> rm -rf build dist
> pip install -e .


-----------------------------


İşlem Verilerinin CSV Olarak Derlenmesi
Her işlem (pozisyon) için bir satır olacak şekilde bir CSV yapısı oluşturulur. Bu CSV, backtest boyunca gerçekleşen tüm işlemlerin detaylarını içerir. Aşağıda CSV’deki sütunlar ve içerikleri listelenmiştir:
time: İşlem zamanı (tarih ve saat damgası). Bu, işlemin açıldığı zamanı temsil eder.
direction: İşlem yönü. "LONG" (uzun pozisyon) veya "SHORT" (kısa pozisyon) olarak belirtilir.
entry_price: İşleme giriş fiyatı. Pozisyonun açıldığı fiyat seviyesidir.
exit_price: İşlemden çıkış fiyatı. Pozisyonun kapandığı fiyat (hedefe ulaşıldıysa TP fiyatı, durdurulduysa SL fiyatı).
atr: İşlemin açıldığı andaki ATR (Average True Range) değeri. ATR, finansal enstrümanın volatilitesini gösteren bir indikatördür
investopedia.com
. Birçok stratejide ATR değeri baz alınarak stop-loss ve take-profit mesafeleri belirlenir; örneğin, stop mesafesi ATR’nin belirli bir katı olarak ayarlanabilir
investopedia.com
. Bu sütun, işlem anındaki ATR değerini kaydeder.
rr_ratio: Risk-getiri oranı (R/R ratio). Bu değer, TP_MULTIPLIER / SL_MULTIPLIER oranı olarak hesaplanır. Örneğin TP_MULTIPLIER = 3 ve SL_MULTIPLIER = 1 ise rr_ratio = 3/1 = 3 olur. Bu, her işlemde hedeflenen kazancın risk edilen miktara oranını gösterir (yani stratejinin risk/ödül oranını).
outcome: İşlemin sonucu. Üç olası değerden biri olabilir:
"TP" (Take Profit) – işlemin kar hedefi ile kapandığını gösterir.
"SL" (Stop Loss) – işlemin zararda durdurulduğunu gösterir.
"OPEN" – işlemin hala açık olduğunu (backtest sonunda henüz kapanmadığını) gösterir.
gain_pct: Kazanç yüzdesi veya R kat sayısı. Bu, işlemin risk bazında getirisi olup rr_ratio veya -1 ya da 0 değerlerini alır.
Eğer outcome = TP ise, işlem kazançla kapandı demektir; getirisi stratejinin risk/ödül oranına eşit olur. Örneğin rr_ratio = 3 ise kazanç +3 * (risk) olarak değerlendirilir, bu durumda gain_pct = +3.0 olarak kaydedilir.
Eğer outcome = SL ise, işlem zararla kapandı ve riskin tamamı kaybedildi demektir; bu durumda gain_pct = -1.0 olarak kaydedilir (–%100 risk kaybı).
Eğer outcome = OPEN ise, işlem henüz kapatılmamıştır; gerçekleşmiş kar/zarar olmadığı için gain_pct = 0.0 olarak bırakılır.
gain_usd: İşlemden elde edilen mutlak kazanç/zarar tutarı (USD cinsinden). Bu değer, kaldıraç etkisini de dikkate alarak hesaplanır. Formül: gain_usd = önceki_balance * risk * gain_pct * LEVERAGE. Burada risk, her işlemde bakiyenin hangi oranının riske atıldığını ifade eder (örneğin risk = 0.01 ise bakiye %1'i riske atılıyor demektir). Eğer işlem SL ile sonuçlanırsa bu tutar negatif olacak (kayıp), TP ile sonuçlanırsa pozitif olacaktır. Kaldıraç (LEVERAGE) değeri, getiriyi aynı oranda büyüteceği için formülde çarpan olarak kullanılır. Örneğin, başlangıç bakiyesi 1000 USD, risk %1 ve kaldıraç 5x ise:
SL durumunda kayıp = 1000 * 0.01 * (-1) * 5 = -50 USD (bakiyenin %5'i kaybedilir).
TP durumunda (rr_ratio=3) kazanç = 1000 * 0.01 * (3) * 5 = +150 USD (bakiyenin %15'i kazanılır).
Bu hesaplama, her işlem sonrası bakiyenin ne kadar değiştiğini dolar cinsinden gösterir.
balance: İşlem sonrası güncel bakiye. Her işlem tamamlandığında (TP veya SL) yeni bakiye, önceki bakiyeye gain_usd tutarının eklenmesiyle bulunur. Eğer işlem hala açık ise bakiye değişmez. Bu sütun, backtest süresince hesabın büyümesini veya azalmasını izlememizi sağlar.

Özet İstatistiklerin Hesaplanması
Her bir işlem kaydının yanında, stratejinin genel performansını özetleyen bazı istatistikler de hesaplanır:
Total Trades (toplam işlem sayısı): Backtest boyunca açılan toplam pozisyon sayısı. (Bu, CSV’deki toplam satır sayısına eşittir. İster TP, ister SL ile kapansın, hatta açık kalmış olsun, tüm girişler toplam işlem sayısını oluşturur.)
TP sayısı (kazançlı işlem adedi): Sonucu TP ile biten işlem sayısı.
SL sayısı (zararlı işlem adedi): Sonucu SL ile biten işlem sayısı.
Win Rate (kazanma yüzdesi): Kazançlı işlemlerin oranı. Genellikle win_rate = TP_sayisi / (TP_sayisi + SL_sayisi) * 100 şeklinde yüzde olarak hesaplanır. Açık pozisyonlar bu orana dahil edilmez çünkü henüz kazanıp kazanmadıkları belli değildir. Örneğin, 40 işlemin 25’i TP ile sonuçlandıysa win rate = 25/40 = %62.5 olur.
Final Balance (nihai bakiye): Backtest sonunda hesabın güncel bakiyesi. (CSV’deki balance sütununun son değeri.) Başlangıç bakiyesi ile karşılaştırarak stratejinin para kazandırıp kazandırmadığını gösterir.
Cumulative Return (kümülatif getiri): Başlangıç bakiyesine göre toplam getiri oranı. Genellikle yüzde olarak ifade edilir ve ((final_balance / INITIAL_BALANCE) - 1) * 100 formülüyle hesaplanır. Örneğin başlangıç 1000 USD iken final bakiye 1200 USD olduysa, kümülatif getiri = ((1200/1000) - 1)*100 = %20 olur.



----------

docs/images/ ciktilara direk bakarsin


--------

batch indikator analizinden sonra cevaplanmasi gereken sorular

	•	Hangi kombinasyonlar tutarlı şekilde kazandırıyor?
 	•	En çok tekrar eden başarılı parametre değerlerini belirle.
	•	Hangi parametre aralıkları daha istikrarlı?
  • En yüksek total_gain_usd ve avg_gain_usd değerlerini incele.
	•	Kaç işlemde bu sonuç alınmış? (total_trades)


  ??? 	4.	Sonraki Geliştirme Aşaması:
	•	AI Filter katmanına geçebilirsin: geçmişe dayalı olarak güçlü sinyalleri öğrenen katman.



  --------

  💡 Neden INDICATOR_LONG ve INDICATOR_SHORT ayırmalıyız?
	1.	Long ve Short stratejiler farklı piyasa koşullarında başarılı olur.
	•	Örneğin: Yükselen trendde RSI, MACD iyi sonuç verirken;
	•	Düşen trendde ADX, CCI veya ATR daha anlamlı sinyaller üretir.
	2.	Aynı parametrelerle hem LONG hem SHORT için yüksek performans almak nadirdir.
Ayrı ayrı incelemek stratejik avantaj sağlar.
	3.	Bu sayede:
	•	Long için özel optimize edilmiş yapı,
	•	Short için ayrı optimize edilmiş yapı kurulur.
	•	İki sistem birlikte çalışarak portföy performansını artırabilir.

🚀 Bonus Fikir

İleride, LONG ve SHORT stratejilerin her biri için ayrı sinyal motorları (signal_engine_long.py, signal_engine_short.py gibi) bile yazabiliriz.

-----

Çok doğru ve kritik bir yaklaşım Erhan. Canlı işlem sistemlerinde milisaniyeler bile pozisyonun kârlı mı zararlı mı olacağını belirleyebilir. Bu doğrultuda sistem tasarımı yaparken şu ilkelere odaklanacağız:

✅ Performansı artırmak için temel öneriler:
	1.	Veri Önişleme:
	•	add_indicators() fonksiyonunu sadece gerekli olanlara göre minimal çalıştır (örneğin INDICATOR_LONG çağrıldığında sadece long’a özgü indikatörleri hesapla).
	•	DataFrame üzerinde tek seferde vectorized işlemler yap.
	2.	Cache / Hafıza Kullanımı:
	•	Hesaplanan indikatörleri önbelleğe al (örneğin bir sinyal iki kez aynı indikatörü istemesin).
	•	Redis gibi bellek içi cache sistemleri düşünebilirsin (özellikle WebSocket ile sürekli veri akışı varsa).
	3.	İşlemci Kullanımı:
	•	Multi-threading veya async yapılarla CPU kullanımını paralel hale getir (özellikle run_batch_with_config gibi yerlerde).
	•	Pandas işlemleri büyük veri için yavaşlarsa PyArrow veya NumPy ile optimizasyon yapılabilir.
	4.	Gereksiz I/O’lardan Kaçınma:
	•	CSV/log kaydı opsiyonel yapılmalı.
	•	Gerekmeyen analizler devre dışı bırakılabilir (örneğin plot_results() yalnızca manuel backtestte çalışmalı).
	5.	Profiling yap:
	•	cProfile ya da line_profiler ile darboğazları analiz et (özellikle backtest loop’larında).
----------

💬 Şimdi birkaç öneri:
	1.	Performans optimizasyonu için gelecekte multiprocessing veya async tabanlı batch sistemi düşünülmeli.
	2.	batch_results.csv’deki en iyi stratejiler otomatik olarak başka bir klasöre kopyalanabilir (örneğin: top_configs/).
	3.	Live trade entegrasyonu için POSITION_DIRECTION dinamik hale getirilebilir (örneğin piyasa trend analizine göre sadece short aç gibi).
	4.	Test sonrası otomatik rapor PDF veya HTML formatında alınabilir.

Devam etmek istediğin yönde hazırım. Bugünlük başka bir aşamaya geçelim mi

------


SHORT ISLEMI ICIN 

Harika, şimdi SHORT işlemler için en verimli indikatör kombinasyonlarını belirlemek adına elimizdeki verileri ve literatürü göz önünde bulundurarak bir değerlendirme yapalım.

⸻

🔍 Mevcut Verilerin ve Literatürün Değerlendirmesi

1. MACD (Moving Average Convergence Divergence)
	•	Avantajları: Trend dönüşlerini ve momentum değişimlerini tespit etmede etkilidir. Özellikle MACD çizgisinin sinyal çizgisini aşağı kesmesi, SHORT pozisyonlar için güçlü bir sinyal olabilir.
	•	Dikkat Edilmesi Gerekenler: Yanıltıcı sinyaller verebilir; bu nedenle RSI veya ADX ile birlikte kullanılması önerilir.

2. RSI (Relative Strength Index)
	•	Avantajları: Aşırı alım bölgelerinde (örneğin RSI > 70) SHORT pozisyonlar için giriş sinyali sağlayabilir.
	•	Dikkat Edilmesi Gerekenler: Tek başına kullanıldığında yanıltıcı olabilir; diğer indikatörlerle birlikte değerlendirilmelidir.

3. ADX (Average Directional Index)
	•	Avantajları: Trendin gücünü ölçer. ADX > 25 olduğunda güçlü bir trendin varlığına işaret eder, bu da SHORT pozisyonlar için uygun bir ortam olabilir.
	•	Dikkat Edilmesi Gerekenler: Yön belirtmez; bu nedenle DI+ ve DI- çizgileriyle birlikte analiz edilmelidir.

4. OBV (On-Balance Volume)
	•	Avantajları: Hacimle fiyat hareketlerini ilişkilendirerek trendin gücünü teyit edebilir.
	•	Dikkat Edilmesi Gerekenler: Tek başına kullanıldığında yanıltıcı olabilir; diğer indikatörlerle birlikte değerlendirilmelidir.

5. SuperTrend
	•	Avantajları: Trendin yönünü ve potansiyel dönüş noktalarını belirlemede yardımcı olabilir.
	•	Dikkat Edilmesi Gerekenler: Yanıltıcı sinyaller verebilir; bu nedenle diğer indikatörlerle birlikte kullanılması önerilir.


🧪 Önerilen İndikatör Kombinasyonları
mbinasyon No
İndikatörler
Açıklama
1
MACD + RSI + ADX
Trend dönüşlerini ve momentum değişimlerini tespit etmek için güçlü bir kombinasyon.
2
SuperTrend + RSI + OBV
Trend yönünü belirlemek ve hacimle teyit etmek için etkili bir kombinasyon.
3
MACD + SuperTrend + ADX
Trendin gücünü ve yönünü belirlemek için uygun bir kombinasyon.
4
RSI + OBV + ADX
Aşırı alım bölgelerinde hacimle teyit edilmiş trendleri tespit etmek için kullanılabilir.
5
MACD + RSI + SuperTrend + ADX
Çoklu indikatörlerle trend dönüşlerini ve gücünü analiz etmek için kapsamlı bir kombinasyon.

✅ Sonuç ve Öneriler
	•	MACD ve SuperTrend, SHORT pozisyonlar için etkili sinyaller sağlayabilir; ancak yanıltıcı sinyalleri filtrelemek için RSI ve ADX ile birlikte kullanılmaları önerilir.
	•	OBV, hacimle fiyat hareketlerini ilişkilendirerek trendin gücünü teyit edebilir; ancak tek başına kullanıldığında yanıltıcı olabilir.
	•	ADX, trendin gücünü ölçmek için kullanılabilir; ancak yön belirtmediği için diğer indikatörlerle birlikte analiz edilmelidir.

Bu değerlendirmeler ışığında, SHORT işlemler için en verimli indikatör kombinasyonlarını belirlemek adına yukarıda önerilen kombinasyonları test etmenizi öneririm. Bu kombinasyonları kullanarak yeni bir config_combinations_short.csv dosyası oluşturabilir ve backtest işlemlerinizi gerçekleştirebilirsiniz.


	1.	Batch Results Summary: Her bir config_id için toplam kazanç, ortalama kazanç, işlem sayısı ve kullanılan indikatör parametrelerini içeriyor.
	2.	All Trades from Batch Test: Her bir işlem için giriş-çıkış fiyatı, outcome (TP, OPEN), kazanç miktarı, sinyal gücü ve filtre geçiş bilgilerini içeriyor.

Bundan sonraki adımda sana aşağıdaki gibi analizler sunabilirim:
	•	En iyi performans gösteren config’ler (gain ve win rate’e göre).
	•	Kazanan/kaybeden işlemlerin ortak özellikleri (örneğin hangi ADX, RSI aralıklarında iyi sonuç alınıyor).
	•	Sinyal gücü ile kazanç ilişkisi.
	•	En çok kullanılan ama düşük performans veren indikatör kombinasyonları.
	•	Farklı gain dağılımlarının histogramı ya da equity curve benzeri grafikler.
