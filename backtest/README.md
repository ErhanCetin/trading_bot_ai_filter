Örnek Kullanım

Tek bir backtest çalıştırma:

python -m backtest.runners.main --mode single --symbol BTCUSDT --interval 5m --output_dir backtest/results/test1

Toplu backtest çalıştırma:

python -m backtest.runners.main --mode batch --config_csv backtest/config/config_combinations.csv --max_workers 4

Özel parametrelerle backtest çalıştırma:

python -m backtest.runners.main --mode single --symbol ETHUSDT --interval 1h --balance 5000 --risk 0.02 --sl_multiplier 2.0 --tp_multiplier 4.0 --leverage 2.0 --direction '{"Long": true, "Short": false}'

----------


Bu dosyaların isimleri ve dizin yapısı, çevre değişkenleri veya komut satırı parametreleri aracılığıyla özelleştirilebilir. Varsayılan olarak backtest/results dizini kullanılır, ancak --output_dir parametresi veya RESULTS_DIR çevre değişkeni kullanılarak değiştirilebilir.
Önemli CSV dosyaları:

trades_[config_id].csv: Her bir backtest işleminin detayları

İşlem zamanı, yön, giriş/çıkış fiyatları, stop-loss/take-profit seviyeleri
İşlem sonucu (TP, SL, OPEN)
Kazanç/kayıp oranları ve bakiye
İşlem sırasındaki indikatör değerleri


batch_results.csv: Toplu backtest sonuçlarının özeti

Konfigürasyon ID'si
Toplam işlem sayısı, win rate, kar/zarar
ROI, max drawdown, Sharpe oranı


batch_trades.csv: Tüm konfigürasyonlardaki tüm işlemlerin birleştirilmiş dosyası
direction_breakdown.csv: İşlem yönüne göre performans kırılımı
strength_breakdown.csv: Sinyal gücüne göre performans kırılımı

CSV dosyaları, daha sonra Excel veya Python ile analiz edilebilir veya sistem tarafından otomatik olarak grafikleştirilebilir.RetryClaude can make mistakes. Please double-check responses.


-----------

daha once yaptigimi anladin .. benim icin onemli noktalari yazayim :
* tum python kodlar en son yazilan signal_engine gore calisack
* kullanmiyacagin herseyi ignore edebilirsin .
* runner_batch_with_config multi thread calismali
* backtest/config/config_combinations.csv dosyasini daha sonra senin olusturacagin indicator combinasyon icin kullanacagiz . Sistem hem csv den hemde env dosyasinda tanimlanan INDICATORS_LONG ve INDICATORS_SHORT indikator listesini alarak calisacak. 
* olusturacagin tum analiz sonuclarini backtest/result klasorune koyacaksin
* * env dosyasinda asagidaki parametreler var . yenilerini onerip kullanabilirsin. var olanlar gereksiz ise kullanmazsin 
* Bizim sistem sadece short signal, sadece long sinyal yada her iki yonlu signal olusturacak sekilde calisacak . env dosyasinda ki 
   -- POSITION_DIRECTION="{\"Long\": false, \"Short\": true}" parametresini kullaniyoruz.
* burdaki parametrelerle hangi indikatorlarin long ve short sinyal icin kullanilacagini belirliyoruz . Bu aslinda statick olmak zorunda degil.Daha sonra konusuruz. 
  -- INDICATORS_LONG="{ <indicator1>:<value1>, <indicator2>:<value2>, ... }"
  -- INDICATORS_SHORT="{ <indicator1>:<value1>, <indicator2>:<value2>, ... }"

Edit
