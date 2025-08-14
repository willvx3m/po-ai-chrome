import pandas as pd

def ema_rsi_strategy(df: pd.DataFrame) -> pd.Series:
    """
    Returns trade signals: 1 = Call/Buy, -1 = Put/Sell, 0 = No trade.
    """
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["ema200"] = df["close"].ewm(span=200).mean()

    # Simple RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / loss))

    signals = []
    for i in range(len(df)):
        if df["ema50"].iloc[i] > df["ema200"].iloc[i] and rsi.iloc[i] > 50:
            signals.append(1)  # Call
        elif df["ema50"].iloc[i] < df["ema200"].iloc[i] and rsi.iloc[i] < 50:
            signals.append(-1)  # Put
        else:
            signals.append(0)  # No trade
    return pd.Series(signals, index=df.index)
