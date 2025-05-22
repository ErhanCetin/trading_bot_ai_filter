# signal_engine/position_management.py - YENİ DOSYA OLUŞTUR

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class Position:
    """Tekil pozisyon sınıfı"""
    
    def __init__(self, entry_data: Dict[str, Any]):
        """
        Initialize position
        
        Args:
            entry_data: Pozisyon giriş verileri
        """
        self.id = entry_data.get("id", f"pos_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.symbol = entry_data.get("symbol", "UNKNOWN")
        self.direction = entry_data.get("direction", "LONG")  # LONG or SHORT
        self.entry_price = entry_data["entry_price"]
        self.entry_time = entry_data.get("entry_time", datetime.now())
        self.quantity = entry_data.get("quantity", 0)
        
        # Risk management
        self.stop_loss = entry_data.get("stop_loss")
        self.take_profit = entry_data.get("take_profit")
        self.trailing_stop = entry_data.get("trailing_stop", False)
        self.trailing_distance = entry_data.get("trailing_distance", 0)
        
        # Position tracking
        self.current_price = self.entry_price
        self.highest_price = self.entry_price  # Trailing stop için
        self.lowest_price = self.entry_price   # Trailing stop için
        self.bars_open = 0
        self.max_bars = entry_data.get("max_bars", 100)  # Timeout
        
        # Performance metrics
        self.unrealized_pnl = 0.0
        self.unrealized_pnl_pct = 0.0
        self.max_favorable = 0.0
        self.max_adverse = 0.0
        
        self.status = "OPEN"  # OPEN, CLOSED_SL, CLOSED_TP, CLOSED_TIMEOUT, CLOSED_MANUAL
        
        logger.info(f"📍 New position created: {self.id} {self.direction} @ {self.entry_price}")
    
    def update(self, current_price: float, current_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Pozisyonu günceller ve kapatma kararını verir
        
        Args:
            current_price: Güncel fiyat
            current_time: Güncel zaman
            
        Returns:
            Pozisyon durumu ve aksiyonları
        """
        self.current_price = current_price
        self.bars_open += 1
        
        # Price extremes tracking
        self.highest_price = max(self.highest_price, current_price)
        self.lowest_price = min(self.lowest_price, current_price)
        
        # PnL hesapla
        if self.direction == "LONG":
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
            self.unrealized_pnl_pct = ((current_price / self.entry_price) - 1) * 100
            self.max_favorable = ((self.highest_price / self.entry_price) - 1) * 100
            self.max_adverse = ((self.lowest_price / self.entry_price) - 1) * 100
        else:  # SHORT
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
            self.unrealized_pnl_pct = (1 - (current_price / self.entry_price)) * 100
            self.max_favorable = (1 - (self.lowest_price / self.entry_price)) * 100
            self.max_adverse = (1 - (self.highest_price / self.entry_price)) * 100
        
        # Trailing stop güncelle
        if self.trailing_stop:
            self._update_trailing_stop()
        
        # Kapatma kontrolü
        should_close, reason = self._should_close()
        
        if should_close:
            self.status = f"CLOSED_{reason}"
            logger.info(f"🔴 Position closed: {self.id} - {reason} @ {current_price} PnL: {self.unrealized_pnl_pct:.2f}%")
        
        return {
            "should_close": should_close,
            "close_reason": reason if should_close else None,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "bars_open": self.bars_open,
            "status": self.status
        }
    
    def _update_trailing_stop(self):
        """Trailing stop seviyesini günceller"""
        if not self.trailing_stop or self.trailing_distance <= 0:
            return
        
        if self.direction == "LONG":
            # Long pozisyon: fiyat yükselirse stop'u yukarı çek
            new_stop = self.highest_price - self.trailing_distance
            if self.stop_loss is None or new_stop > self.stop_loss:
                self.stop_loss = new_stop
        else:  # SHORT
            # Short pozisyon: fiyat düşerse stop'u aşağı çek
            new_stop = self.lowest_price + self.trailing_distance
            if self.stop_loss is None or new_stop < self.stop_loss:
                self.stop_loss = new_stop
    
    def _should_close(self) -> Tuple[bool, str]:
        """Pozisyonun kapatılıp kapatılmayacağını kontrol eder"""
        
        # Stop Loss kontrolü
        if self.stop_loss is not None:
            if ((self.direction == "LONG" and self.current_price <= self.stop_loss) or
                (self.direction == "SHORT" and self.current_price >= self.stop_loss)):
                return True, "SL"
        
        # Take Profit kontrolü
        if self.take_profit is not None:
            if ((self.direction == "LONG" and self.current_price >= self.take_profit) or
                (self.direction == "SHORT" and self.current_price <= self.take_profit)):
                return True, "TP"
        
        # Timeout kontrolü
        if self.bars_open >= self.max_bars:
            return True, "TIMEOUT"
        
        return False, ""


class PositionManager:
    """Çoklu pozisyon yönetim sistemi"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize position manager
        
        Args:
            config: Konfigürasyon ayarları
        """
        self.config = config or {}
        
        # Position limits
        self.max_positions = self.config.get("max_positions", 10)
        self.max_positions_per_symbol = self.config.get("max_per_symbol", 3)
        self.max_risk_per_trade = self.config.get("max_risk_pct", 2.0)
        self.max_total_risk = self.config.get("max_total_risk", 10.0)
        
        # Position defaults
        self.default_max_bars = self.config.get("default_timeout", 100)
        self.enable_trailing_stops = self.config.get("trailing_stops", True)
        self.trailing_distance_atr = self.config.get("trailing_atr", 1.5)
        
        # Tracking
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Dict[str, Any]] = []
        self.total_pnl = 0.0
        self.win_count = 0
        self.loss_count = 0
        
        logger.info(f"📊 Position Manager initialized: max_pos={self.max_positions}")
    
    def can_open_position(self, symbol: str, risk_amount: float) -> Tuple[bool, str]:
        """
        Yeni pozisyon açılıp açılamayacağını kontrol eder
        
        Args:
            symbol: Sembol
            risk_amount: Risk miktarı
            
        Returns:
            (açılabilir_mi, sebep)
        """
        # Toplam açık pozisyon kontrolü
        open_positions = len([p for p in self.positions.values() if p.status == "OPEN"])
        if open_positions >= self.max_positions:
            return False, f"Max positions reached: {open_positions}/{self.max_positions}"
        
        # Sembol bazında pozisyon kontrolü
        symbol_positions = len([p for p in self.positions.values() 
                              if p.symbol == symbol and p.status == "OPEN"])
        if symbol_positions >= self.max_positions_per_symbol:
            return False, f"Max positions for {symbol}: {symbol_positions}/{self.max_positions_per_symbol}"
        
        # Toplam risk kontrolü
        current_total_risk = sum(abs(p.unrealized_pnl) for p in self.positions.values() 
                               if p.status == "OPEN")
        if (current_total_risk + risk_amount) > self.max_total_risk:
            return False, f"Total risk limit exceeded"
        
        return True, "OK"
    
    def open_position(self, entry_data: Dict[str, Any]) -> Optional[str]:
        """
        Yeni pozisyon açar
        
        Args:
            entry_data: Pozisyon verileri
            
        Returns:
            Position ID veya None
        """
        symbol = entry_data.get("symbol", "UNKNOWN")
        risk_amount = entry_data.get("risk_amount", 0)
        
        # Açılabilir mi kontrol et
        can_open, reason = self.can_open_position(symbol, risk_amount)
        if not can_open:
            logger.warning(f"❌ Cannot open position: {reason}")
            return None
        
        # Varsayılan değerleri ekle
        entry_data.setdefault("max_bars", self.default_max_bars)
        
        # Trailing stop ayarla
        if self.enable_trailing_stops and "atr" in entry_data:
            entry_data["trailing_stop"] = True
            entry_data["trailing_distance"] = entry_data["atr"] * self.trailing_distance_atr
        
        # Position oluştur
        position = Position(entry_data)
        self.positions[position.id] = position
        
        logger.info(f"✅ Position opened: {position.id}")
        return position.id
    
    def update_positions(self, market_data: Dict[str, float], 
                        current_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Tüm pozisyonları günceller
        
        Args:
            market_data: {symbol: current_price} dictionary
            current_time: Güncel zaman
            
        Returns:
            Güncelleme özeti
        """
        updates = {
            "positions_updated": 0,
            "positions_closed": 0,
            "close_reasons": {},
            "total_pnl": 0.0,
            "open_positions": 0
        }
        
        positions_to_close = []
        
        # Her pozisyonu güncelle
        for pos_id, position in self.positions.items():
            if position.status != "OPEN":
                continue
                
            # Market data var mı kontrol et
            if position.symbol not in market_data:
                continue
            
            current_price = market_data[position.symbol]
            
            # Pozisyonu güncelle
            update_result = position.update(current_price, current_time)
            updates["positions_updated"] += 1
            updates["total_pnl"] += update_result["unrealized_pnl"]
            
            # Kapatılacak pozisyonları işaretle
            if update_result["should_close"]:
                positions_to_close.append((pos_id, update_result["close_reason"]))
                updates["positions_closed"] += 1
                
                reason = update_result["close_reason"]
                updates["close_reasons"][reason] = updates["close_reasons"].get(reason, 0) + 1
        
        # Pozisyonları kapat
        for pos_id, reason in positions_to_close:
            self._close_position(pos_id, reason)
        
        # Açık pozisyon sayısı
        updates["open_positions"] = len([p for p in self.positions.values() if p.status == "OPEN"])
        
        return updates
    
    def _close_position(self, position_id: str, reason: str):
        """Pozisyonu kapatır ve kayıtları tutar"""
        if position_id not in self.positions:
            return
        
        position = self.positions[position_id]
        
        # Closed positions listesine ekle
        close_record = {
            "id": position.id,
            "symbol": position.symbol,
            "direction": position.direction,
            "entry_price": position.entry_price,
            "exit_price": position.current_price,
            "entry_time": position.entry_time,
            "exit_time": datetime.now(),
            "bars_open": position.bars_open,
            "quantity": position.quantity,
            "pnl": position.unrealized_pnl,
            "pnl_pct": position.unrealized_pnl_pct,
            "max_favorable": position.max_favorable,
            "max_adverse": position.max_adverse,
            "close_reason": reason
        }
        
        self.closed_positions.append(close_record)
        
        # İstatistik güncelle
        self.total_pnl += position.unrealized_pnl
        if position.unrealized_pnl > 0:
            self.win_count += 1
        else:
            self.loss_count += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Performans istatistiklerini döndürür"""
        total_trades = self.win_count + self.loss_count
        win_rate = (self.win_count / total_trades * 100) if total_trades > 0 else 0
        
        open_positions = [p for p in self.positions.values() if p.status == "OPEN"]
        
        return {
            "open_positions": len(open_positions),
            "closed_positions": len(self.closed_positions),
            "total_trades": total_trades,
            "win_rate": win_rate,
            "total_pnl": self.total_pnl,
            "unrealized_pnl": sum(p.unrealized_pnl for p in open_positions),
            "avg_bars_open": np.mean([p.bars_open for p in open_positions]) if open_positions else 0
        }
    
    def apply_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DataFrame'e position management mantığını uygular
        
        Args:
            df: Trading data DataFrame
            
        Returns:
            Position management bilgileri eklenmiş DataFrame
        """
        result_df = df.copy()
        
        # Yeni sütunlar
        new_cols = ["position_id", "position_action", "position_pnl", "open_positions_count"]
        for col in new_cols:
            if col not in result_df.columns:
                result_df[col] = ""
        
        current_positions = {}
        position_counter = 1
        
        # Her satırı işle
        for i in range(len(result_df)):
            row = result_df.iloc[i]
            current_price = row["close"]
            
            # Mevcut pozisyonları güncelle
            positions_to_close = []
            for pos_id, pos_data in current_positions.items():
                pos_data["bars_open"] += 1
                
                # PnL hesapla
                if pos_data["direction"] == "LONG":
                    pnl_pct = ((current_price / pos_data["entry_price"]) - 1) * 100
                else:
                    pnl_pct = (1 - (current_price / pos_data["entry_price"])) * 100
                
                # Kapatma kontrolü
                should_close, reason = self._check_close_conditions(
                    current_price, pos_data, row.get("atr", 2.0)
                )
                
                if should_close:
                    positions_to_close.append((pos_id, reason, pnl_pct))
            
            # Pozisyonları kapat
            for pos_id, reason, pnl_pct in positions_to_close:
                result_df.loc[result_df.index[i], "position_action"] += f"CLOSE_{pos_id}({reason}:{pnl_pct:.1f}%) "
                del current_positions[pos_id]
            
            # Yeni pozisyon aç
            if ((row.get("long_signal") or row.get("short_signal")) and 
                row.get("signal_passed_filter", False)):
                
                # Pozisyon limiti kontrolü
                if len(current_positions) < self.max_positions:
                    pos_id = f"P{position_counter}"
                    direction = "LONG" if row.get("long_signal") else "SHORT"
                    
                    atr = row.get("atr", 2.0)
                    if direction == "LONG":
                        stop_loss = current_price - atr
                        take_profit = current_price + (atr * 2.5)
                    else:
                        stop_loss = current_price + atr
                        take_profit = current_price - (atr * 2.5)
                    
                    current_positions[pos_id] = {
                        "entry_price": current_price,
                        "direction": direction,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "bars_open": 0,
                        "entry_index": i
                    }
                    
                    result_df.loc[result_df.index[i], "position_id"] = pos_id
                    result_df.loc[result_df.index[i], "position_action"] += f"OPEN_{pos_id}({direction}) "
                    position_counter += 1
            
            # Açık pozisyon sayısını kaydet
            result_df.loc[result_df.index[i], "open_positions_count"] = len(current_positions)
            
            # Toplam PnL hesapla
            total_pnl = 0
            for pos_data in current_positions.values():
                if pos_data["direction"] == "LONG":
                    pnl = ((current_price / pos_data["entry_price"]) - 1) * 100
                else:
                    pnl = (1 - (current_price / pos_data["entry_price"])) * 100
                total_pnl += pnl
            
            result_df.loc[result_df.index[i], "position_pnl"] = total_pnl
        
        # Özet bilgileri
        total_opened = position_counter - 1
        max_open = result_df["open_positions_count"].max()
        final_open = len(current_positions)
        
        logger.info(f"📊 Position Management Applied:")
        logger.info(f"   Total positions opened: {total_opened}")
        logger.info(f"   Max concurrent positions: {max_open}")
        logger.info(f"   Final open positions: {final_open}")
        
        return result_df
    
    def _check_close_conditions(self, current_price: float, pos_data: Dict[str, Any], 
                               atr: float) -> Tuple[bool, str]:
        """Pozisyon kapatma koşullarını kontrol eder"""
        
        # Stop Loss
        if ((pos_data["direction"] == "LONG" and current_price <= pos_data["stop_loss"]) or
            (pos_data["direction"] == "SHORT" and current_price >= pos_data["stop_loss"])):
            return True, "SL"
        
        # Take Profit
        if ((pos_data["direction"] == "LONG" and current_price >= pos_data["take_profit"]) or
            (pos_data["direction"] == "SHORT" and current_price <= pos_data["take_profit"])):
            return True, "TP"
        
        # Timeout
        if pos_data["bars_open"] >= self.default_max_bars:
            return True, "TIMEOUT"
        
        return False, ""


# 🧪 TEST FONKSİYONLARI

def test_position_management():
    """Position management sistemini test eder"""
    print("🧪 POSITION MANAGEMENT TEST")
    print("=" * 50)
    
    # Test verisi
    test_data = {
        'close': [100, 101, 99, 102, 98, 104, 96, 105, 94, 106],
        'atr': [2.0] * 10,
        'long_signal': [True, False, False, True, False, False, False, True, False, False],
        'short_signal': [False, False, True, False, False, True, False, False, False, False],
        'signal_passed_filter': [True, False, True, True, False, True, False, True, False, False]
    }
    
    df = pd.DataFrame(test_data)
    
    # Position manager ile işle
    pm = PositionManager({
        "max_positions": 5,
        "default_timeout": 5,
        "trailing_stops": True
    })
    
    result_df = pm.apply_to_dataframe(df)
    
    print("📊 SONUÇLAR:")
    cols = ['close', 'long_signal', 'short_signal', 'position_id', 'position_action', 'open_positions_count']
    print(result_df[cols])
    
    return result_df

# Ana test
if __name__ == "__main__":
    test_position_management()