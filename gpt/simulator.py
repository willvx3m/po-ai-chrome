from staking import martingale_stake
import pandas as pd

def run_backtest(df, signals, expiry_minutes=3, payout=0.8, base_stake=1, martingale=True):
    results = []
    balance = 0
    prev_stake = base_stake
    prev_result = None

    for i in range(len(df) - expiry_minutes):
        signal = signals.iloc[i]
        if signal == 0:
            continue

        # Martingale stake calculation
        if martingale:
            stake = martingale_stake(prev_result, prev_stake, base_stake)
        else:
            stake = base_stake

        entry_price = df["close"].iloc[i]
        expiry_price = df["close"].iloc[i + expiry_minutes]

        win = (signal == 1 and expiry_price > entry_price) or (signal == -1 and expiry_price < entry_price)
        profit = stake * payout if win else -stake
        balance += profit

        results.append({
            "entry_time": df["datetime_point"].iloc[i],
            "signal": signal,
            "entry_price": entry_price,
            "expiry_price": expiry_price,
            "stake": stake,
            "win": win,
            "profit": profit,
            "balance": balance
        })

        prev_result = win
        prev_stake = stake

    return pd.DataFrame(results)
