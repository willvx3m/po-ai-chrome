from data_loader import load_candles_from_json
from strategy import ema_rsi_strategy
from simulator import run_backtest
from performance import performance_summary

df = load_candles_from_json("../lstm/eurusd.json")
signals = ema_rsi_strategy(df)

trades_df = run_backtest(df, signals, expiry_minutes=3, payout=0.8, base_stake=1, martingale=True)

print(trades_df.head())
print(performance_summary(trades_df))
