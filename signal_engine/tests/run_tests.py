"""
Tüm testleri çalıştırmak için test runner.
Komut satırından şu şekilde çalıştırın:
    python run_tests.py
"""
import unittest
import sys
import os
import time

def run_all_tests():
    """Tüm testleri çalıştır ve sonuçları raporla."""
    start_time = time.time()
    
    # Test dizini
    test_dir = 'tests'
    
    # Testleri keşfet
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir)
    
    # Test koşucusu oluştur
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Çıktı başlığı
    print("=" * 70)
    print(f"{'Signal Engine Testleri':^70}")
    print("=" * 70)
    
    # Testleri çalıştır
    result = runner.run(suite)
    
    # Sonuçları raporla
    end_time = time.time()
    run_time = end_time - start_time
    
    print("\n" + "=" * 70)
    print(f"{'Test Sonuçları':^70}")
    print("=" * 70)
    print(f"Çalıştırılan testler: {result.testsRun}")
    print(f"Başarılı: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Başarısız: {len(result.failures)}")
    print(f"Hata: {len(result.errors)}")
    print(f"Çalışma süresi: {run_time:.2f} saniye")
    
    # Ayrıntılı hata raporları
    if result.failures or result.errors:
        print("\n" + "=" * 70)
        print(f"{'Hata Ayrıntıları':^70}")
        print("=" * 70)
        
        if result.failures:
            print("\nBAŞARISIZ TESTLER:")
            for i, (test, traceback) in enumerate(result.failures, 1):
                print(f"\n--- Başarısız Test #{i}: {test} ---")
                print(traceback)
        
        if result.errors:
            print("\nHATA VEREN TESTLER:")
            for i, (test, traceback) in enumerate(result.errors, 1):
                print(f"\n--- Hata #{i}: {test} ---")
                print(traceback)
    
    # Başarı durumuna göre çıkış kodu döndür
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    sys.exit(run_all_tests())