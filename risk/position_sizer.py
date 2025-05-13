def calculate_position_size(account_balance: float, risk_per_trade: float, atr: float, entry_price: float, sl_price: float):
    """
    account_balance: toplam portföy bakiyesi
    risk_per_trade: % olarak risk (ör: 0.01 = %1)
    atr: volatility ölçüsü
    entry_price: işlem giriş fiyatı
    sl_price: stop loss fiyatı

    Geriye: pozisyon büyüklüğü (ör: 0.05 BTC) döner
    """
    risk_amount = account_balance * risk_per_trade
    stop_loss_per_unit = abs(entry_price - sl_price)
    if stop_loss_per_unit == 0:
        return 0
    position_size = risk_amount / stop_loss_per_unit
    return round(position_size, 5)
