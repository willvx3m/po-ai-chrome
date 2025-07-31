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
    for i in range(len(closes)-period, len(closes)):
        changes.append(closes[i] - closes[i-1])
    
    if len(changes) < period:
        return None
    
    # Separate gains and losses, omit excessive gains and losses
    gains = [change if (change > 0 and change < 0.03) else 0 for change in changes]
    losses = [-change if (change < 0 and change > -0.03) else 0 for change in changes]

    # gains.remove(max(gains))
    # losses.remove(max(losses))

    if len(gains) == 0 or len(losses) == 0:
        return None
    
    # Calculate average gains and losses for the period
    avg_gain = sum(gains) / len(gains)
    avg_loss = sum(losses) / len(losses)
    
    # Calculate RS and RSI
    if avg_loss == 0:
        rsi = 100
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd_indicator(candles, short_period, long_period):
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        candles: list of candles with 'close' price
        short_period: period of the fast EMA
        long_period: period of the slow EMA
    
    Returns:
        macd: dictionary containing MACD line, signal line, and histogram
    """
    if len(candles) < long_period:  # Need at least 26 candles for MACD
        return None
    
    # Calculate fast EMA (12-period)
    fast_ema = calculate_ema_indicator(candles, short_period)
    
    # Calculate slow EMA (26-period)
    slow_ema = calculate_ema_indicator(candles, long_period)
    
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

def calculate_fibonacci_indicator(candles, period):
    """
    Get fibonacci levels from ohlcv data for the given period.
    Percentages: 23.6%, 38.2%, 50%, 61.8%, 78.6%.
    Use the start and end of the period to calculate the fibonacci levels.
    
    Args:
        candles: list of candles with 'low' and 'high' price
        period: period of the fibonacci indicator (from the last candle)
    
    Returns:
        fibonacci_levels: list of fibonacci levels
        trend: "up" or "down"
    """
    if len(candles) < period:
        return None, None
    
    # Get the candles for the specified period (from the end)
    period_candles = candles[-period:]
    
    # Find the highest high and lowest low in the period
    highs = [candle['high'] for candle in period_candles]
    lows = [candle['low'] for candle in period_candles]
    
    swing_high = max(highs)
    swing_low = min(lows)
    
    # Determine the trend based on the start and end prices
    start_price = period_candles[0]['close']
    end_price = period_candles[-1]['close']
    
    if end_price > start_price:
        trend = "up"
        # For uptrend: swing low to swing high
        price_range = swing_high - swing_low
        base_price = swing_low
    else:
        trend = "down"
        # For downtrend: swing high to swing low
        price_range = swing_high - swing_low
        base_price = swing_high
    
    # Calculate Fibonacci retracement levels
    fibonacci_ratios = [0.236, 0.382, 0.500, 0.618, 0.786]
    fibonacci_levels = []
    
    for ratio in fibonacci_ratios:
        if trend == "up":
            # For uptrend, levels are below the swing high
            level = swing_high - (price_range * ratio)
        else:
            # For downtrend, levels are above the swing low
            level = swing_low + (price_range * ratio)
        
        fibonacci_levels.append({ "level": level, "ratio": ratio })
    
    return fibonacci_levels, trend

def calculate_support_resistance_indicator(candles, period, pip_offset=0.0002, merge_threshold=0.0005, min_strength=3):
    """
    Get support and resistance levels from ohlcv data for the given period.
    Range Definition: Set levels with a ±5 pip offset to account for noise (e.g., 1.0860 becomes 1.0855-1.0865).
    
    Confirmation with historical candles to find reliable levels for short-term trading.
    Identify pivot highs (High higher than two candles before and after) and pivot lows (Low lower than two before and after).
    Mark resistance zones around pivot highs and support zones around pivot lows, applying ±5 pips.
    Confirm levels with multiple price tests or reversals.
    
    Args:
        candles: list of candles with 'low' and 'high' price
        period: period of candles to calculate the support and resistance indicator (from the last candle)
    
    Returns:
        support_resistance_levels: list of support and resistance levels
    """
    if len(candles) < period:
        return None
    
    # Get the candles for the specified period (from the end)
    period_candles = candles[-period:]
    
    # Find pivot highs and lows
    pivot_highs = []
    pivot_lows = []
    
    # Need at least 5 candles to identify pivot points (2 before + current + 2 after)
    for i in range(2, len(period_candles) - 2):
        current_high = period_candles[i]['high']
        current_low = period_candles[i]['low']
        
        # Check if current high is higher than 2 candles before and after
        is_pivot_high = True
        for j in range(i-3, i+3):
            if j != i and period_candles[j]['high'] >= current_high:
                is_pivot_high = False
                break
        
        if is_pivot_high:
            pivot_highs.append({
                'type': 'pivot_high',
                'index': i,
                'price': current_high,
                'zone_low': current_high - pip_offset,
                'zone_high': current_high + pip_offset
            })
        
        # Check if current low is lower than 2 candles before and after
        is_pivot_low = True
        for j in range(i-3, i+3):
            if j != i and period_candles[j]['low'] <= current_low:
                is_pivot_low = False
                break
        
        if is_pivot_low:
            pivot_lows.append({
                'type': 'pivot_low',
                'index': i,
                'price': current_low,
                'zone_low': current_low - pip_offset,
                'zone_high': current_low + pip_offset
            })
    
    # Merge nearby levels to avoid duplicates
    def merge_nearby_levels(levels):
        if not levels:
            return []
        
        merged = []
        levels.sort(key=lambda x: x['price'])
        
        current_group = [levels[0]]
        
        for level in levels[1:]:
            if abs(level['price'] - current_group[0]['price']) <= merge_threshold:
                current_group.append(level)
            else:
                # Merge the group
                if len(current_group) >= min_strength:
                    avg_price = sum(l['price'] for l in current_group) / len(current_group)
                    merged.append({
                        'price': avg_price,
                        'zone_low': avg_price - pip_offset,
                        'zone_high': avg_price + pip_offset,
                        'type': 'resistance' if 'pivot_high' in str(current_group[0]) else 'support',
                        'strength': len(current_group)  # Number of confirming levels
                    })
                current_group = [level]
        
        # Handle the last group
        if current_group:
            if len(current_group) >= min_strength:
                avg_price = sum(l['price'] for l in current_group) / len(current_group)
                merged.append({
                    'price': avg_price,
                    'zone_low': avg_price - pip_offset,
                    'zone_high': avg_price + pip_offset,
                    'type': 'resistance' if 'pivot_high' in str(current_group[0]) else 'support',
                    'strength': len(current_group)
                })
        
        return merged
    
    # Merge pivot highs and lows
    resistance_levels = merge_nearby_levels(pivot_highs)
    support_levels = merge_nearby_levels(pivot_lows)
    
    # Combine all levels
    support_resistance_levels = resistance_levels + support_levels
    
    # Sort by price
    support_resistance_levels.sort(key=lambda x: x['price'])
    
    return support_resistance_levels

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