import json
import os
from datetime import datetime
from indicator import calculate_ema_indicator, get_trend_based_on_ema
from strategy import run_strategy, STRATEGY_DURATION, MAX_POSITIONS
import itertools
import pandas as pd

# Constants
JSON_PATH = './eurusd-full.json'

# Status constants
CHECKING_ENTRY = "CHECKING_ENTRY"
RUNNING_STRATEGY = "RUNNING_STRATEGY"

def can_enter_strategy(candles, short_period, long_period, trend_persist_period = 0):
    """
    Check if the current candle can enter the strategy.
    
    Args:
        candles: List of candle dictionaries
        short_period: Short period of the ema (from the last candle)
        long_period: Long period of the ema (from the last candle)
    
    Returns:
        Boolean: True if the current candle can enter the strategy, False otherwise
    """
    current_trend = get_trend_based_on_ema(candles, short_period, long_period, trend_persist_period)
    if not current_trend:
        return None
    
    short_ema = calculate_ema_indicator(candles, short_period)
    if not short_ema:
        return None

    # if the trend is persisted, check if the current candle pushes the trend
    current_price = (candles[-1]["low"] + candles[-1]["high"]) / 2
    if current_trend == "up" and current_price > short_ema:
        return "call"
    elif current_trend == "down" and current_price < short_ema:
        return "put"
    else:
        return None
    
def run_single_backtest(short_period, long_period, strategy_duration, max_positions, trend_persist_period = 0, debug=False):
    """
    Run a single backtest with specific parameters.
    
    Args:
        short_period: Short period of the ema
        long_period: Long period of the ema
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
    current_index = long_period - 1
    status = CHECKING_ENTRY
    strategy_state = None
    results = []
    strategy_start_index = None
    strategy_direction = None
    
    # Main loop
    while current_index < len(candles):
        # print(f"Index: {current_index}")
        current_candle = candles[current_index]
        
        if status == CHECKING_ENTRY:
            ema_handles = candles[max(0, current_index - long_period - trend_persist_period):current_index + 1]
            can_enter = can_enter_strategy(ema_handles, short_period, long_period, trend_persist_period)
            # print(f"Index: {current_index}, Can enter: {can_enter}")
            if can_enter:
                status = RUNNING_STRATEGY
                strategy_state = None  # Reset strategy state
                strategy_start_index = current_index
                strategy_direction = can_enter
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
            
            if strategy_state is not None and strategy_state.get("is_finished", False):
                results.append({
                    "profit": strategy_state["total_profit"],
                    "amount": strategy_state["total_amount"],
                    "strategy_start_index": strategy_start_index,
                    "strategy_finished_index": current_index,
                    "positions": strategy_state.get("positions", []),
                })

                if debug:
                    print(f"Start: {strategy_start_index}, End: {current_index}, Amount: {strategy_state['total_amount']}, Profit: {strategy_state['total_profit']}, Positions: {len(strategy_state.get('positions', []))}")

                status = CHECKING_ENTRY
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
        with open(f"results/ma_{short_period}_{long_period}_{strategy_duration}_{max_positions}_{trend_persist_period}.json", "w") as f:
            json.dump(results, f)
        
        return {
            "short_period": short_period,
            "long_period": long_period,
            "strategy_duration": strategy_duration,
            "max_positions": max_positions,
            "trend_persist_period": trend_persist_period,
            "total_trades": num_trades,
            "total_profit": total_profit,
            "total_amount": total_amount,
            "roi": total_profit / total_amount if total_amount > 0 else 0,
            "avg_profit": avg_profit,
            "results": results
        }
    else:
        return {
            "short_period": short_period,
            "long_period": long_period,
            "strategy_duration": strategy_duration,
            "max_positions": max_positions,
            "trend_persist_period": trend_persist_period,
            "total_trades": 0,
            "total_profit": 0,
            "total_amount": 0,
            "roi": 0.0,
            "avg_profit": 0,
            "results": []
        }

if __name__ == "__main__":
    result = run_single_backtest(5, 60, 10, 5, 10, True) 
    if result:
        print(f"=> Trades: {result['total_trades']}, "
                f"Profit: {result['total_profit']:.2f}, "
                f"Amount: {result['total_amount']:.2f}, "
                f"ROI: {result['roi']:.2f}")
    else:
        print(f"! Failed")