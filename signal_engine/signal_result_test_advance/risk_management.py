# signal_engine/risk_management.py - YENİ DOSYA OLUŞTUR

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RiskRewardOptimizer:
    """Risk/Reward oranını optimize eden ana sınıf"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize risk/reward optimizer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # 🎯 YENİ HEDEF DEĞERLER
        self.target_rr_ratio = self.config.get("target_rr_ratio", 2.5)  # 1.0 → 2.5
        self.min_rr_ratio = self.config.get("min_rr_ratio", 1.8)        # 1.0 → 1.8
        
        # ATR Çarpanları
        self.stop_loss_atr_multiplier = self.config.get("stop_atr", 1.0)    # 2.0 → 1.0
        self.take_profit_atr_multiplier = self.config.get("tp_atr", 2.5)    # 1.5 → 2.5
        
        # Risk Yönetimi
        self.max_risk_per_trade = self.config.get("max_risk_pct", 2.0)  # %2 max risk
        self.position_sizing_method = self.config.get("position_method", "fixed_risk")
        
        logger.info(f"🎯 RR Optimizer initialized: target={self.target_rr_ratio}, min={self.min_rr_ratio}")
    
    def calculate_optimal_levels(self, entry_price: float, atr: float, 
                               direction: str) -> Dict[str, Any]:
        """
        Optimal stop-loss ve take-profit seviyelerini hesaplar
        
        Args:
            entry_price: Giriş fiyatı
            atr: Average True Range değeri
            direction: 'LONG' veya 'SHORT'
            
        Returns:
            Seviyeleri içeren dictionary
        """
        direction = direction.upper()
        
        if direction == "LONG":
            # 🔧 LONG POZİSYON - İyileştirilmiş
            stop_loss = entry_price - (atr * self.stop_loss_atr_multiplier)
            take_profit = entry_price + (atr * self.take_profit_atr_multiplier)
            
        elif direction == "SHORT":
            # 🔧 SHORT POZİSYON - İyileştirilmiş  
            stop_loss = entry_price + (atr * self.stop_loss_atr_multiplier)
            take_profit = entry_price - (atr * self.take_profit_atr_multiplier)
            
        else:
            raise ValueError(f"Invalid direction: {direction}. Use 'LONG' or 'SHORT'")
        
        # Risk ve ödül hesapla
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0
        
        # Geçerlilik kontrolü
        is_valid = rr_ratio >= self.min_rr_ratio
        
        # Risk yüzdesini hesapla
        risk_pct = (risk / entry_price) * 100
        
        return {
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk": risk,
            "reward": reward,
            "rr_ratio": round(rr_ratio, 2),
            "risk_pct": round(risk_pct, 2),
            "is_valid": is_valid,
            "direction": direction,
            "quality_score": self._calculate_quality_score(rr_ratio, risk_pct)
        }
    
    def _calculate_quality_score(self, rr_ratio: float, risk_pct: float) -> int:
        """
        Sinyal kalite skorunu hesaplar (0-100)
        
        Args:
            rr_ratio: Risk/Reward oranı
            risk_pct: Risk yüzdesi
            
        Returns:
            Kalite skoru (0-100)
        """
        score = 50  # Base score
        
        # RR ratio bonusu
        if rr_ratio >= 3.0:
            score += 25
        elif rr_ratio >= 2.5:
            score += 20
        elif rr_ratio >= 2.0:
            score += 15
        elif rr_ratio >= 1.5:
            score += 10
        else:
            score -= 20
        
        # Risk yüzdesi kontrolü
        if risk_pct <= 1.5:
            score += 15
        elif risk_pct <= 2.0:
            score += 10
        elif risk_pct <= 3.0:
            score += 5
        else:
            score -= 10
        
        return max(0, min(100, score))
    
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tüm DataFrame için risk/reward optimizasyonu yapar
        
        Args:
            df: Trading verileri içeren DataFrame
            
        Returns:
            Optimize edilmiş DataFrame
        """
        result_df = df.copy()
        
        # Gerekli sütunları kontrol et
        required_cols = ["close", "atr"]
        missing_cols = [col for col in required_cols if col not in result_df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return result_df
        
        # Yeni sütunlar ekle
        new_columns = ["optimized_sl", "optimized_tp", "optimized_rr", 
                      "risk_pct", "quality_score", "rr_improved"]
        
        for col in new_columns:
            if col not in result_df.columns:
                result_df[col] = np.nan
        
        improved_count = 0
        total_signals = 0
        
        # Her satır için optimizasyon yap
        for i in range(len(result_df)):
            # Sinyal var mı kontrol et
            has_long = result_df.get("long_signal", pd.Series(False)).iloc[i]
            has_short = result_df.get("short_signal", pd.Series(False)).iloc[i]
            
            if not (has_long or has_short):
                continue
                
            total_signals += 1
            
            # Giriş verilerini al
            entry_price = result_df["close"].iloc[i]
            atr = result_df["atr"].iloc[i]
            
            # Sinyal yönünü belirle
            direction = "LONG" if has_long else "SHORT"
            
            # Optimal seviyeleri hesapla
            levels = self.calculate_optimal_levels(entry_price, atr, direction)
            
            # Mevcut değerlerle karşılaştır
            current_rr = result_df.get("rr_ratio", pd.Series(1.0)).iloc[i]
            improved = levels["rr_ratio"] > current_rr
            
            if improved:
                improved_count += 1
            
            # Sonuçları kaydet
            result_df.loc[result_df.index[i], "optimized_sl"] = levels["stop_loss"]
            result_df.loc[result_df.index[i], "optimized_tp"] = levels["take_profit"]
            result_df.loc[result_df.index[i], "optimized_rr"] = levels["rr_ratio"]
            result_df.loc[result_df.index[i], "risk_pct"] = levels["risk_pct"]
            result_df.loc[result_df.index[i], "quality_score"] = levels["quality_score"]
            result_df.loc[result_df.index[i], "rr_improved"] = improved
            
            # Orijinal değerleri güncelle (opsiyonel)
            if self.config.get("update_original", True):
                result_df.loc[result_df.index[i], "sl"] = levels["stop_loss"]
                result_df.loc[result_df.index[i], "tp"] = levels["take_profit"]
                result_df.loc[result_df.index[i], "rr_ratio"] = levels["rr_ratio"]
        
        # Sonuçları raporla
        if total_signals > 0:
            improvement_pct = (improved_count / total_signals) * 100
            avg_rr = result_df["optimized_rr"].dropna().mean()
            
            logger.info(f"📈 RR Optimization Results:")
            logger.info(f"   Total signals optimized: {total_signals}")
            logger.info(f"   Improved signals: {improved_count} ({improvement_pct:.1f}%)")
            logger.info(f"   Average new RR ratio: {avg_rr:.2f}")
        
        return result_df


class PositionSizer:
    """Pozisyon büyüklüğü hesaplama sınıfı"""
    
    def __init__(self, account_balance: float, risk_per_trade: float = 2.0):
        """
        Initialize position sizer
        
        Args:
            account_balance: Hesap bakiyesi
            risk_per_trade: Trade başına risk yüzdesi
        """
        self.account_balance = account_balance
        self.risk_per_trade = risk_per_trade / 100  # Yüzdeyi orana çevir
        
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                              risk_amount: Optional[float] = None) -> Dict[str, Any]:
        """
        Pozisyon büyüklüğünü hesaplar
        
        Args:
            entry_price: Giriş fiyatı
            stop_loss: Stop loss seviyesi
            risk_amount: Risk miktarı (belirtilmezse hesap bakiyesinin %'si kullanılır)
            
        Returns:
            Pozisyon bilgileri
        """
        if risk_amount is None:
            risk_amount = self.account_balance * self.risk_per_trade
        
        # Risk per share
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share <= 0:
            return {"position_size": 0, "risk_amount": 0, "shares": 0}
        
        # Position size hesapla
        shares = risk_amount / risk_per_share
        position_value = shares * entry_price
        
        # Maksimum pozisyon limiti (%20 hesap)
        max_position = self.account_balance * 0.2
        if position_value > max_position:
            shares = max_position / entry_price
            position_value = max_position
            actual_risk = shares * risk_per_share
        else:
            actual_risk = risk_amount
        
        return {
            "shares": round(shares, 2),
            "position_value": round(position_value, 2),
            "risk_amount": round(actual_risk, 2),
            "risk_pct": round((actual_risk / self.account_balance) * 100, 2),
            "position_pct": round((position_value / self.account_balance) * 100, 2)
        }


# 🧪 TEST VE ÖRNEK KULLANIM

def test_risk_reward_optimizer():
    """Risk/Reward optimizer'ı test eder"""
    
    print("🧪 RISK/REWARD OPTIMIZER TEST")
    print("=" * 50)
    
    # Test verisi oluştur
    test_data = {
        'close': [100, 150, 80, 200, 120],
        'atr': [2.5, 3.0, 1.5, 4.0, 2.0],
        'long_signal': [True, False, True, False, True],
        'short_signal': [False, True, False, True, False],
        'sl': [98, 153, 81.5, 196, 118],  # Mevcut stop loss
        'tp': [103, 147, 78.5, 204, 122],  # Mevcut take profit  
        'rr_ratio': [1.0, 1.0, 1.0, 1.0, 1.0]  # Mevcut RR
    }
    
    df = pd.DataFrame(test_data)
    
    print("📊 MEVCUT DURUM:")
    print(f"Ortalama RR Ratio: {df['rr_ratio'].mean():.2f}")
    
    # Optimizer oluştur ve çalıştır
    optimizer = RiskRewardOptimizer({
        "target_rr_ratio": 2.5,
        "min_rr_ratio": 1.8,
        "stop_atr": 1.0,
        "tp_atr": 2.5
    })
    
    optimized_df = optimizer.optimize_dataframe(df)
    
    print("\n📈 OPTİMİZASYON SONRASI:")
    print(f"Ortalama RR Ratio: {optimized_df['optimized_rr'].mean():.2f}")
    print(f"İyileştirilen sinyal sayısı: {optimized_df['rr_improved'].sum()}")
    
    # Detayları göster
    print("\n📋 DETAY:")
    cols = ['close', 'atr', 'rr_ratio', 'optimized_rr', 'quality_score', 'rr_improved']
    print(optimized_df[cols].round(2))
    
    return optimized_df

def apply_rr_optimization_to_csv(csv_path: str, output_path: str = None):
    """
    CSV dosyasına RR optimizasyonu uygular
    
    Args:
        csv_path: Giriş CSV dosya yolu
        output_path: Çıkış CSV dosya yolu (belirtilmezse aynı dosyaya yazar)
    """
    import pandas as pd
    
    # CSV'yi oku
    df = pd.read_csv(csv_path)
    
    print(f"📂 CSV okundu: {len(df)} satır")
    print(f"📊 Mevcut ortalama RR: {df.get('rr_ratio', pd.Series([1.0])).mean():.2f}")
    
    # Optimizer ile işle
    optimizer = RiskRewardOptimizer()
    optimized_df = optimizer.optimize_dataframe(df)
    
    # Kaydet
    if output_path is None:
        output_path = csv_path.replace('.csv', '_optimized.csv')
    
    optimized_df.to_csv(output_path, index=False)
    
    print(f"✅ Optimize edilmiş veriler kaydedildi: {output_path}")
    print(f"📈 Yeni ortalama RR: {optimized_df['optimized_rr'].mean():.2f}")
    
    return optimized_df

# Ana test
if __name__ == "__main__":
    test_result = test_risk_reward_optimizer()
    
    # Position sizer test
    print("\n" + "="*50)
    print("🧪 POSITION SIZER TEST")
    
    sizer = PositionSizer(account_balance=10000, risk_per_trade=2.0)
    position = sizer.calculate_position_size(entry_price=100, stop_loss=95)
    
    print(f"Position Size: {position}")