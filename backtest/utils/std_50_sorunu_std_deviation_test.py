def test_std_deviation_indicator():
    """
    std_deviation indikatörünü test eder ve hatanın tam kaynağını bulur
    """
    from signal_engine.indicators import registry
    import pandas as pd
    import numpy as np
    import traceback
    
    # Test verileri oluştur
    test_data = {
        'open': [100 + i for i in range(100)],
        'high': [105 + i for i in range(100)],
        'low': [95 + i for i in range(100)],
        'close': [102 + i for i in range(100)],
        'volume': [1000 for _ in range(100)]
    }
    df = pd.DataFrame(test_data)
    
    # std_deviation indikatörünü al
    std_deviation_class = registry.get_indicator('std_deviation')
    
    if std_deviation_class:
        print("Indikatör sınıfı bulundu, detaylı bilgi:")
        print(f"Default params: {std_deviation_class.default_params}")
        print(f"Requires columns: {std_deviation_class.requires_columns}")
        print(f"Output columns: {std_deviation_class.output_columns}")
        
        try:
            # Test 1: Varsayılan parametrelerle
            indicator = std_deviation_class()
            result_df = indicator.calculate(df)
            print("\nVarsayılan parametrelerle hesaplama başarılı.")
            print(f"Oluşturulan sütunlar: {indicator.output_columns}")
            
            # Test 2: Sorun yaratan parametre ile
            test_params = {'windows': [80]}
            indicator2 = std_deviation_class(test_params)
            result_df2 = indicator2.calculate(df)
            print("\n'windows: [80]' parametresiyle hesaplama başarılı.")
            print(f"Oluşturulan sütunlar: {indicator2.output_columns}")
            
        except Exception as e:
            print(f"\nHata: {e}")
            print("Ayrıntılı hata:")
            traceback.print_exc()
            
    else:
        print("std_deviation indikatörü registry'de bulunamadı.")
    
    print("\nMevcut tüm indikatörler:")
    for name in registry.get_all_indicators().keys():
        print(f"- {name}")

# Test fonksiyonunu çağır
test_std_deviation_indicator()