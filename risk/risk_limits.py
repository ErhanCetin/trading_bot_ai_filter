class RiskManager:
    def __init__(self, max_daily_loss_pct=0.05, max_position_pct=0.25):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_position_pct = max_position_pct
        self.daily_loss = 0
        self.account_balance = 10000  # örnek bakiye

    def can_trade(self, current_drawdown):
        self.daily_loss += current_drawdown
        if self.daily_loss > self.account_balance * self.max_daily_loss_pct:
            return False
        return True

    def get_max_position_size(self):
        return self.account_balance * self.max_position_pct
