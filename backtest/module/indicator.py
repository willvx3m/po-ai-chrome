"""
This is the indicator module.
Define the following functions:

1. define function calculate_ema_indicator
    parameters:
        - candles: list of candles
        - period: period of the ema (from the last candle)
    return:
        - ema
2. define function calculate_sma_indicator
    parameters:
        - candles: list of candles
        - period: period of the sma (from the last candle)
    return:
        - sma
3. define function calculate_rsi_indicator
    parameters:
        - candles: list of candles
        - period: period of the rsi (from the last candle)
    return:
        - rsi
4. define function calculate_macd_indicator
    parameters:
        - candles: list of candles
        - period: period of the macd (from the last candle)
    return:
        - macd
"""

import numpy as np

def calculate_ema_indicator(input_candles, period, end_offset=0):
    """
    Calculate Exponential Moving Average (EMA).
    
    Args:
        candles: list of candles with 'close' price
        period: period of the ema (from the last candle)
    
    Returns:
        ema: the EMA value for the last candle
    """
    if end_offset > 0:
        candles = input_candles[:end_offset]
    else:
        candles = input_candles

    if len(candles) < period:
        return None
    
    # Extract close prices
    closes = [candle['close'] for candle in candles]
    
    # Calculate multiplier
    multiplier = 2 / (period + 1)
    
    # Initialize EMA with SMA of first 'period' values
    starting_index = len(closes)-period
    ema = closes[starting_index]
    
    # Calculate EMA for remaining values
    for i in range(starting_index + 1, len(closes)):
        ema = (closes[i] * multiplier) + (ema * (1 - multiplier))
    
    return ema

def calculate_sma_indicator(candles, period):
    """
    Calculate Simple Moving Average (SMA).
    
    Args:
        candles: list of candles with 'close' price
        period: period of the sma (from the last candle)
    
    Returns:
        sma: the SMA value for the last candle
    """
    if len(candles) < period:
        return None
    
    # Extract close prices for the last 'period' candles
    closes = [candle['close'] for candle in candles[-period:]]
    
    # Calculate SMA
    sma = sum(closes) / period
    
    return sma

def calculate_rsi_indicator(candles, period):
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        candles: list of candles with 'close' price
        period: period of the rsi (from the last candle)
    
    Returns:
        rsi: the RSI value for the last candle
    """
    if len(candles) < period + 1:
        return None
    
    # Extract close prices
    closes = [candle['close'] for candle in candles]
    
    # Calculate price changes
    changes = []
    for i in range(1, len(closes)):
        changes.append(closes[i] - closes[i-1])
    
    if len(changes) < period:
        return None
    
    # Separate gains and losses
    gains = [change if change > 0 else 0 for change in changes]
    losses = [-change if change < 0 else 0 for change in changes]
    
    # Calculate average gains and losses for the period
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    
    # Calculate RS and RSI
    if avg_loss == 0:
        rsi = 100
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd_indicator(candles, period):
    """
    Calculate MACD (Moving Average Convergence Divergence).
    Uses standard periods: 12 for fast EMA, 26 for slow EMA, 9 for signal line.
    
    Args:
        candles: list of candles with 'close' price
        period: period of the macd (signal line period, typically 9)
    
    Returns:
        macd: dictionary containing MACD line, signal line, and histogram
    """
    if len(candles) < 26:  # Need at least 26 candles for MACD
        return None
    
    # Extract close prices
    closes = [candle['close'] for candle in candles]
    
    # Calculate fast EMA (12-period)
    fast_ema = calculate_ema_indicator(candles, 12)
    
    # Calculate slow EMA (26-period)
    slow_ema = calculate_ema_indicator(candles, 26)
    
    if fast_ema is None or slow_ema is None:
        return None
    
    # Calculate MACD line
    macd_line = fast_ema - slow_ema
    
    # For signal line, we would need to calculate EMA of MACD line
    # Since we only have the current MACD value, we'll return just the MACD line
    # In a full implementation, you'd need to track MACD line values over time
    
    return {
        'macd_line': macd_line,
        'signal_line': None,  # Would need historical MACD values
        'histogram': None     # Would need signal line
    }

# Alternative MACD implementation that returns just the MACD line value
def calculate_macd_line(candles, fast_period=12, slow_period=26):
    """
    Calculate MACD line value.
    
    Args:
        candles: list of candles with 'close' price
        fast_period: period for fast EMA (default 12)
        slow_period: period for slow EMA (default 26)
    
    Returns:
        macd_line: the MACD line value
    """
    if len(candles) < max(fast_period, slow_period):
        return None
    
    # Calculate fast EMA
    fast_ema = calculate_ema_indicator(candles, fast_period)
    
    # Calculate slow EMA
    slow_ema = calculate_ema_indicator(candles, slow_period)
    
    if fast_ema is None or slow_ema is None:
        return None
    
    # Calculate MACD line
    macd_line = fast_ema - slow_ema
    
    return macd_line

def get_trend_based_on_ema(candles, short_period, long_period, trend_persist_period = 0):
    """
    Get the trend based on the EMA.
    """
    short_ema = calculate_ema_indicator(candles, short_period)
    long_ema = calculate_ema_indicator(candles, long_period)

    if not short_ema or not long_ema:
        return None
    
    current_trend = None
    if short_ema > long_ema:
        current_trend = "up"
    else:
        current_trend = "down"

    # check if the trend is persisted
    persist_last_index = len(candles) - 1
    while trend_persist_period > 0 and persist_last_index > 0:
        persist_ema_short = calculate_ema_indicator(candles, short_period, persist_last_index)
        persist_ema_long = calculate_ema_indicator(candles, long_period, persist_last_index)
        persist_trend = None

        if persist_ema_short is None or persist_ema_long is None:
            return None

        if persist_ema_short > persist_ema_long:
            persist_trend = "up"
        else:
            persist_trend = "down"

        if persist_trend != current_trend:
            return None

        if trend_persist_period > 0 and persist_last_index == 0:
            return None
        
        trend_persist_period -= 1
        persist_last_index -= 1
    
    return current_trend