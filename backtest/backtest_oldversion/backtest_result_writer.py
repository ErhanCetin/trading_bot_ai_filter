import csv
import os
from typing import List, Dict, Any

def write_backtest_results(trades: List[Dict[str, Any]], filename: str) -> None:
    """
    Backtest sonuçlarını CSV dosyasına yazar
    
    Args:
        trades: Trade sonuçlarını içeren liste
        filename: Yazılacak dosya yolu
    """
    if not trades:
        print("⚠️ No trades to write.")
        return

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    fieldnames = list(trades[0].keys())

    with open(filename, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for trade in trades:
            writer.writerow(trade)

    print(f"✅ Trades written to {filename}")
    print(f"   Total trades: {len(trades)}")
    
    # Additional summary metrics
    tp_count = sum(1 for t in trades if t.get('outcome') == 'TP')
    sl_count = sum(1 for t in trades if t.get('outcome') == 'SL')
    open_count = sum(1 for t in trades if t.get('outcome') == 'OPEN')
    
    win_rate = (tp_count / (tp_count + sl_count)) * 100 if (tp_count + sl_count) > 0 else 0
    
    print(f"   TP: {tp_count}, SL: {sl_count}, Open: {open_count}")
    print(f"   Win Rate: {win_rate:.2f}%")
    
    # Direction breakdown
    if trades and 'direction' in trades[0]:
        long_count = sum(1 for t in trades if t.get('direction') == 'LONG')
        short_count = sum(1 for t in trades if t.get('direction') == 'SHORT')
        print(f"   Direction - Long: {long_count}, Short: {short_count}")