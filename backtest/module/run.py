import json
import os
from datetime import datetime
from trend_lines import calculate_trend_lines
from strategy import run_strategy, STRATEGY_DURATION, MAX_POSITIONS
import itertools
import pandas as pd

# Constants
JSON_PATH = './eurusd-full.json'

# Status constants
CALCULATING_TREND = "CALCULATING_TREND"
CHECKING_TREND_BREAK = "CHECKING_TREND_BREAK"
RUNNING_STRATEGY = "RUNNING_STRATEGY"

def is_trend_line_broken(candle, trend_lines, candle_index):
    """
    Check if the current candle breaks the trend lines.
    
    Args:
        candle: Dictionary with candle data
        trend_lines: Dictionary containing support and resistance line equations
        candle_index: Index of the current candle in the trend calculation
    
    Returns:
        tuple: (support_break, resistance_break)
    """
    if "error" in trend_lines:
        return False, False
    
    if "support" not in trend_lines or trend_lines["support"]["slope"] < 0:
        return False, False
    if "resistance" not in trend_lines or trend_lines["resistance"]["slope"] > 0:
        return False, False
    
    # Calculate support and resistance values at this candle index
    support_slope = trend_lines["support"]["slope"]
    support_intercept = trend_lines["support"]["intercept"]
    resistance_slope = trend_lines["resistance"]["slope"]
    resistance_intercept = trend_lines["resistance"]["intercept"]
    
    support_value = support_slope * candle_index + support_intercept
    resistance_value = resistance_slope * candle_index + resistance_intercept
    
    # Check if candle breaks the trend lines
    support_break = candle["low"] <= support_value * (1)
    resistance_break = candle["high"] >= resistance_value * (1)
    
    return support_break, resistance_break

def is_strategy_finished(strategy_state):
    """
    Check if the strategy has finished.
    
    Args:
        strategy_state: Dictionary containing strategy state
    
    Returns:
        bool: True if strategy is finished, False otherwise
    """
    return strategy_state is not None and strategy_state.get("is_finished", False)

def calculate_trend_lines_for_candles(candles):
    """
    Calculate trend lines for a given set of candles.
    
    Args:
        candles: List of candle dictionaries
    
    Returns:
        dict: Trend lines calculation result
    """
    if len(candles) < 10:  # Minimum required by trend_lines module
        return {"error": f"Need at least 10 candles to calculate trend lines"}
    
    return calculate_trend_lines(candles)

def run_single_backtest(min_candles_for_trend, strategy_duration, max_positions):
    """
    Run a single backtest with specific parameters.
    
    Args:
        min_candles_for_trend: Number of candles to use for trend calculation
        strategy_duration: Duration of the strategy in candles
        max_positions: Maximum number of positions
    
    Returns:
        dict: Backtest results
    """
    # Check if JSON file exists
    if not os.path.exists(JSON_PATH):
        print(f"Error: JSON file not found at {JSON_PATH}")
        return None
    
    # Read JSON file
    try:
        with open(JSON_PATH, 'r') as f:
            candles = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return None
    
    # Override strategy module constants
    import strategy
    original_strategy_duration = strategy.STRATEGY_DURATION
    original_max_positions = strategy.MAX_POSITIONS
    
    strategy.STRATEGY_DURATION = strategy_duration
    strategy.MAX_POSITIONS = max_positions
    
    # Initialize variables
    current_index = min_candles_for_trend - 1
    status = CALCULATING_TREND
    trend_lines = None
    trend_start_index = None
    trend_end_index = None
    strategy_state = None
    results = []
    strategy_start_index = None
    strategy_direction = None
    
    # Main loop
    while current_index < len(candles):
        current_candle = candles[current_index]
        
        if status == CALCULATING_TREND:
            # Calculate trend lines using the last N candles
            temp_start_index = max(0, current_index - min_candles_for_trend)
            temp_end_index = min(current_index, temp_start_index + min_candles_for_trend)
            lookback_candles = candles[temp_start_index:temp_end_index]
            trend_lines = calculate_trend_lines_for_candles(lookback_candles)
            
            if "error" not in trend_lines:
                trend_start_index = temp_start_index
                trend_end_index = temp_end_index
                status = CHECKING_TREND_BREAK
            else:
                current_index += 1
                continue
        
        elif status == CHECKING_TREND_BREAK:
            # Calculate the relative index for trend line checking
            relative_index = current_index - trend_start_index
            
            support_break, resistance_break = is_trend_line_broken(current_candle, trend_lines, relative_index)
            if trend_lines and (support_break or resistance_break):
                status = RUNNING_STRATEGY
                strategy_state = None  # Reset strategy state
                strategy_start_index = current_index
                strategy_direction = "call" if support_break else "put"
            else:
                # if the trend line is not broken and the relative index is greater than min_candles_for_trend // 2, recalculate the trend lines
                if current_index - trend_end_index > min_candles_for_trend // 2:
                    status = CALCULATING_TREND
                    trend_lines = None
                    trend_start_index = None
                    trend_end_index = None
                    continue
                else:
                    current_index += 1
                    continue
        
        elif status == RUNNING_STRATEGY:
            # Run strategy directly
            positions, is_finished, total_profit, total_amount = run_strategy(
                strategy_direction,
                strategy_state.get("positions", []) if strategy_state else [],
                strategy_start_index,
                current_index,
                current_candle
            )
            
            # Update strategy state
            if strategy_state is None:
                strategy_state = {}
            strategy_state["positions"] = positions
            strategy_state["is_finished"] = is_finished
            strategy_state["total_profit"] = total_profit
            strategy_state["total_amount"] = total_amount
            
            if is_strategy_finished(strategy_state):
                results.append({
                    "profit": strategy_state["total_profit"],
                    "amount": strategy_state["total_amount"],
                    "strategy_start_index": strategy_start_index,
                    "strategy_finished_index": current_index,
                    "trend_start_index": trend_start_index,
                    "trend_end_index": trend_end_index,
                    "trend_lines": trend_lines,
                    "positions": strategy_state.get("positions", []),
                })

                status = CALCULATING_TREND
                trend_lines = None
                strategy_start_index = None
                strategy_state = None
            
            current_index += 1
        
    # Restore original constants
    strategy.STRATEGY_DURATION = original_strategy_duration
    strategy.MAX_POSITIONS = original_max_positions
    
    # Calculate summary statistics
    if results:
        total_profit = sum(r["profit"] for r in results)
        total_amount = sum(r["amount"] for r in results)
        num_trades = len(results)
        avg_profit = total_profit / num_trades if num_trades > 0 else 0

        # save results to json
        with open(f"results/{min_candles_for_trend}_{strategy_duration}_{max_positions}.json", "w") as f:
            json.dump(results, f)
        
        return {
            "min_candles_for_trend": min_candles_for_trend,
            "strategy_duration": strategy_duration,
            "max_positions": max_positions,
            "total_trades": num_trades,
            "total_profit": total_profit,
            "total_amount": total_amount,
            "roi": total_profit / total_amount if total_amount > 0 else 0,
            "avg_profit": avg_profit,
            "results": results
        }
    else:
        return {
            "min_candles_for_trend": min_candles_for_trend,
            "strategy_duration": strategy_duration,
            "max_positions": max_positions,
            "total_trades": 0,
            "total_profit": 0,
            "total_amount": 0,
            "roi": 0.0,
            "avg_profit": 0,
            "results": []
        }

if __name__ == "__main__":
    result = run_single_backtest(30, 5, 3) 
    if result:
        print(f" Trades: {result['total_trades']}, "
                f"Profit: {result['total_profit']:.2f}, "
                f"Amount: {result['total_amount']:.2f}, "
                f"ROI: {result['roi']:.2f}")
    else:
        print(f"! Failed")